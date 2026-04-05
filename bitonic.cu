//./bitonic.cu//
//
// GPU parallel evaluator for bitonic sort.
// Implements the recursive algorithm from bitonic.c AS-IS using a
// task/continuation runtime. Tasks return VALUE or SPLIT; continuations
// fire when all children land.
//
// Sequential cutoff: for depths <= SEQ_CUTOFF the recursive functions
// are executed directly on-thread without any task/cont overhead.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

typedef unsigned long long u64;
typedef unsigned int    u32;
typedef unsigned short   u16;

// ════════════════════════════════════════════════════════════════════════════
// Tree encoding  (u64, bit-63 tag)
// ════════════════════════════════════════════════════════════════════════════
#define NODE_TAG (1ULL << 63)
__host__ __device__ inline u64 Leaf(u32 v)  { return (u64)v; }
__host__ __device__ inline u64 MkNode(u32 i) { return NODE_TAG | (u64)i; }
__host__ __device__ inline bool IsNode(u64 t) { return (t & NODE_TAG) != 0; }
__host__ __device__ inline bool IsLeaf(u64 t) { return !IsNode(t); }
__host__ __device__ inline u32 GetVal(u64 t) { return (u32)(t & 0xFFFFFFFFu); }
__host__ __device__ inline u32 GetIdx(u64 t) { return (u32)(t & 0x7FFFFFFFu); }

// ════════════════════════════════════════════════════════════════════════════
// Tunables
// ════════════════════════════════════════════════════════════════════════════
#define HEAP_SIZE     (1u << 30)  // u64 slots  (~8 GB)
#define CONT_BUF_SIZE   (1u << 24)  // 16 M continuations
#define QUEUE_BUF_SIZE  (1u << 22)  // 4 M global-queue entries
#define NUM_BLOCKS    48
#define BLOCK_SIZE    256
#define NUM_THREADS    (NUM_BLOCKS * BLOCK_SIZE)
#define ALLOC_CHUNK    1024     // per-thread bump chunk (u64 slots)
#define CONT_CHUNK    256      // per-thread cont bump chunk
#define SEQ_CUTOFF    3       // sort/flow/swap ≤ this depth → sequential
#define IDLE_POLL_MASK  31      // check global queue every (mask+1) idle spins

// Function IDs  (matching the design in AGENTS.md)
#define FN_SORT       0  // sort(d, s, t)
#define FN_FLOW       1  // flow(d, s, t)
#define FN_SWAP       2  // swap(s, t)  — with depth hint for cutoff
#define FN_SORT_CONT    3  // sort cont: (d, s, sl, sr) → flow
#define FN_FLOW_AFTER_SWAP 4  // after swap: (d, s, t', _) → down
#define FN_FLOW_JOIN    5  // join: (_, _, fl, fr) → Node
#define FN_SWAP_JOIN    6  // join: (_, _, p0, p1) → unzip

#define ROOT_RET 0xFFFFFFFFu
#define R_VALUE 0
#define R_SPLIT 1
#define R_CALL  2

// ════════════════════════════════════════════════════════════════════════════
// Structs
// ════════════════════════════════════════════════════════════════════════════
struct Task { u32 fn, ret; u64 args[4]; };
// Compact cont: 32 bytes instead of 48. Creation-time constants (d,s)
// stored as u16 a0,a1; only 2 result slots instead of 4.
struct Cont { u32 pending, ret; u16 fn, a0, a1, _pad; u64 args[2]; };
struct Result { u32 tag; u64 value; Task t0, t1; };

struct GState {
  u64 *heap;  u32 *heap_bump;
  Cont *conts; u32 *cont_bump;
  Task *gqueue; u32 *gqueue_ready, *gqueue_push, *gqueue_pop;
  u32 *done; u64 *result;
};

#define CHK(call) do { cudaError_t _e=(call); \
  if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} } while(0)

// ════════════════════════════════════════════════════════════════════════════
// Device helpers
// ════════════════════════════════════════════════════════════════════════════
__device__ inline u32 heap_alloc(u32 n, u32 &lp, u32 &le, u32 *g) {
  if (lp+n > le) { lp = atomicAdd(g,(u32)ALLOC_CHUNK); le = lp+ALLOC_CHUNK; }
  u32 r = lp; lp += n; return r;
}
__device__ inline u32 mk_nd(u64 l, u64 r, u64 *H, u32 &lp, u32 &le, u32 *B) {
  u32 i = heap_alloc(2,lp,le,B); H[i]=l; H[i+1]=r; return i;
}
__device__ inline u32 cont_alloc(u32 &cp, u32 &ce, u32 *cb) {
  if (cp >= ce) { cp = atomicAdd(cb,(u32)CONT_CHUNK); ce = cp+CONT_CHUNK; }
  return cp++;
}
__device__ inline u32 alloc_cont(u32 fn,u32 pend,u32 ret,
    u32 a0,u32 a1, Cont*cs, u32*cb, u32 &cp, u32 &ce) {
  u32 ci = cont_alloc(cp,ce,cb); Cont*c=&cs[ci];
  c->pending=pend; c->ret=ret; c->fn=(u16)fn; c->a0=(u16)a0; c->a1=(u16)a1; c->_pad=0;
  c->args[0]=0; c->args[1]=0;
  return ci;
}
__device__ inline u32  enc_ret(u32 ci,u32 s) { return (ci<<1)|s; }
__device__ inline Result mk_value(u64 v) { Result r; r.tag=R_VALUE; r.value=v; return r; }
__device__ inline Result mk_split(Task a,Task b) { Result r; r.tag=R_SPLIT; r.t0=a; r.t1=b; return r; }
__device__ inline Result mk_call (Task a)     { Result r; r.tag=R_CALL;  r.t0=a; return r; }
__device__ inline Task  mk_task(u32 fn,u32 ret,u64 a0,u64 a1,u64 a2,u64 a3) {
  Task t; t.fn=fn; t.ret=ret; t.args[0]=a0; t.args[1]=a1; t.args[2]=a2; t.args[3]=a3; return t;
}

// ════════════════════════════════════════════════════════════════════════════
// Global queue
// ════════════════════════════════════════════════════════════════════════════
__device__ void gqueue_push(GState &gs, Task &task) {
  u32 idx = atomicAdd(gs.gqueue_push,1u);
  gs.gqueue[idx % QUEUE_BUF_SIZE] = task;
  __threadfence();
  atomicExch(&gs.gqueue_ready[idx % QUEUE_BUF_SIZE], 1u);
}
__device__ bool gqueue_pop(GState &gs, Task *out) {
  for (int a = 0; a < 4; a++) {
    u32 pop = *(volatile u32*)gs.gqueue_pop;
    u32 push = *(volatile u32*)gs.gqueue_push;
    if (pop >= push) return false;
    if (atomicCAS(gs.gqueue_pop, pop, pop+1u) == pop) {
      u32 slot = pop % QUEUE_BUF_SIZE;
      while (atomicAdd(&gs.gqueue_ready[slot],0u) != 1u)
        if (*(volatile u32*)gs.done) return false;
      __threadfence();
      *out = gs.gqueue[slot];
      atomicExch(&gs.gqueue_ready[slot], 0u);
      return true;
    }
  }
  return false;
}

// ════════════════════════════════════════════════════════════════════════════
// Task push  (XOR-probe shared-memory mailbox → global queue)
// ════════════════════════════════════════════════════════════════════════════
__device__ void push_task(GState &gs, Task &task, int tid, int bid,
             u32 *s_mbox_flag, Task *s_mbox_task) {
  for (int d = 1; d <= 128; d <<= 1) {
    int tgt = tid ^ d;
    if (atomicCAS(&s_mbox_flag[tgt], 0u, 1u) == 0u) {
      s_mbox_task[tgt] = task;
      __threadfence_block();
      atomicExch(&s_mbox_flag[tgt], 2u); return;
    }
  }
  gqueue_push(gs, task);
}

// ════════════════════════════════════════════════════════════════════════════
// Value resolution
// ════════════════════════════════════════════════════════════════════════════
__device__ bool resolve_value(GState &gs, u32 ret, u64 val, Task *out) {
  if (ret == ROOT_RET) {
    *(volatile u64*)gs.result = val; __threadfence();
    atomicExch(gs.done, 1u); return false;
  }
  u32 ci = ret >> 1, slot = ret & 1;
  Cont *co = &gs.conts[ci];
  *(volatile u64*)&co->args[slot] = val; __threadfence();
  u32 old = atomicSub(&co->pending, 1u);
  if (old == 1u) {
    __threadfence();
    out->fn  = co->fn;  out->ret = co->ret;
    out->args[0] = (u64)co->a0;
    out->args[1] = (u64)co->a1;
    out->args[2] = *(volatile u64*)&co->args[0];
    out->args[3] = *(volatile u64*)&co->args[1];
    return true;
  }
  return false;
}

// ════════════════════════════════════════════════════════════════════════════
// Sequential device functions  (mirror the C code exactly)
//
//   d_warp  ≡  warp(d, s, a, b)
//   d_flow  ≡  flow(d, s, t)
//   d_down  ≡  down(d, s, t)
//   d_sort  ≡  sort(d, s, t)
//
// These are used when depth ≤ SEQ_CUTOFF to avoid task/cont overhead.
// ════════════════════════════════════════════════════════════════════════════
__device__ u64 d_warp(u32 d, u32 s, u64 a, u64 b,
           u64 *H, u32 &lp, u32 &le, u32 *B) {
  if (d == 0) {
    // warp_swap:  c = s ^ (av > bv)
    u32 av = GetVal(a), bv = GetVal(b);
    u32 c = s ^ (av > bv ? 1u : 0u);
    if (c == 0) return MkNode(mk_nd(Leaf(av), Leaf(bv), H,lp,le,B));
    else    return MkNode(mk_nd(Leaf(bv), Leaf(av), H,lp,le,B));
  }
  u64 wa = d_warp(d-1, s, H[GetIdx(a)], H[GetIdx(b)], H,lp,le,B);
  u64 wb = d_warp(d-1, s, H[GetIdx(a)+1], H[GetIdx(b)+1], H,lp,le,B);
  u64 l0=H[GetIdx(wa)], l1=H[GetIdx(wb)];
  u64 r0=H[GetIdx(wa)+1], r1=H[GetIdx(wb)+1];
  u32 li = mk_nd(l0,l1,H,lp,le,B);
  u32 ri = mk_nd(r0,r1,H,lp,le,B);
  return MkNode(mk_nd(MkNode(li),MkNode(ri),H,lp,le,B));
}

__device__ u64 d_flow(u32 d, u32 s, u64 t, u64 *H, u32 &lp, u32 &le, u32 *B);

__device__ u64 d_down(u32 d, u32 s, u64 t, u64 *H, u32 &lp, u32 &le, u32 *B) {
  if (d==0 || IsLeaf(t)) return t;
  u64 fl = d_flow(d-1, s, H[GetIdx(t)],  H,lp,le,B);
  u64 fr = d_flow(d-1, s, H[GetIdx(t)+1], H,lp,le,B);
  return MkNode(mk_nd(fl,fr,H,lp,le,B));
}

__device__ u64 d_flow(u32 d, u32 s, u64 t, u64 *H, u32 &lp, u32 &le, u32 *B) {
  if (d==0 || IsLeaf(t)) return t;
  u64 warped = d_warp(d-1, s, H[GetIdx(t)], H[GetIdx(t)+1], H,lp,le,B);
  return d_down(d, s, warped, H,lp,le,B);
}

__device__ u64 d_sort(u32 d, u32 s, u64 t, u64 *H, u32 &lp, u32 &le, u32 *B) {
  if (d==0 || IsLeaf(t)) return t;
  u64 sl = d_sort(d-1, 0, H[GetIdx(t)],  H,lp,le,B);
  u64 sr = d_sort(d-1, 1, H[GetIdx(t)+1], H,lp,le,B);
  return d_flow(d, s, MkNode(mk_nd(sl,sr,H,lp,le,B)), H,lp,le,B);
}

// ════════════════════════════════════════════════════════════════════════════
// Task functions
// ════════════════════════════════════════════════════════════════════════════
__device__ Result flow_impl(u32 d, u32 s, u64 t, u32 ret,
               GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce);

// ── sort(d, s, t) ─────────────────────────────────────────────────────────
__device__ Result fn_sort(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  u32 d=(u32)a[0], s=(u32)a[1]; u64 t=a[2];
  if (d==0 || IsLeaf(t)) return mk_value(t);
  if (d <= SEQ_CUTOFF)   return mk_value(d_sort(d,s,t,gs.heap,lp,le,gs.heap_bump));
  u64 l = gs.heap[GetIdx(t)], r = gs.heap[GetIdx(t)+1];
  u32 ci = alloc_cont(FN_SORT_CONT,2,ret,d,s,gs.conts,gs.cont_bump,cp,ce);
  return mk_split(mk_task(FN_SORT,enc_ret(ci,0),(u64)(d-1),0ULL,l,0),
          mk_task(FN_SORT,enc_ret(ci,1),(u64)(d-1),1ULL,r,0));
}

// ── sort_cont(d, s, sl, sr) → flow(d, s, Node(sl,sr)) ────────────────────
__device__ Result fn_sort_cont(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  u32 d=(u32)a[0], s=(u32)a[1]; u64 sl=a[2], sr=a[3];
  u32 ni = mk_nd(sl,sr,gs.heap,lp,le,gs.heap_bump);
  return flow_impl(d, s, MkNode(ni), ret, gs, lp, le, cp, ce);
}

// ── flow_impl  (shared by fn_flow and fn_sort_cont) ───────────────────────
__device__ Result flow_impl(u32 d, u32 s, u64 t, u32 ret,
               GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  if (d==0 || IsLeaf(t)) return mk_value(t);
  if (d <= SEQ_CUTOFF)   return mk_value(d_flow(d,s,t,gs.heap,lp,le,gs.heap_bump));
  u32 ci = alloc_cont(FN_FLOW_AFTER_SWAP,1,ret,d,s,gs.conts,gs.cont_bump,cp,ce);
  return mk_call(mk_task(FN_SWAP, enc_ret(ci,0), (u64)s, t, (u64)(d-1), 0));
}

// ── flow(d, s, t) ─────────────────────────────────────────────────────────
__device__ Result fn_flow(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  u32 d=(u32)a[0], s=(u32)a[1]; u64 t=a[2];
  if (d==0 || IsLeaf(t)) return mk_value(t);
  if (d <= SEQ_CUTOFF)   return mk_value(d_flow(d,s,t,gs.heap,lp,le,gs.heap_bump));
  return flow_impl(d, s, t, ret, gs, lp, le, cp, ce);
}

// ── flow_after_swap(d, s, t_swapped, _) → down ───────────────────────────
__device__ Result fn_flow_after_swap(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  u32 d=(u32)a[0], s=(u32)a[1]; u64 t=a[2];
  if (IsLeaf(t)) return mk_value(t);
  u64 l = gs.heap[GetIdx(t)], r = gs.heap[GetIdx(t)+1];
  u32 ci = alloc_cont(FN_FLOW_JOIN,2,ret,0,0,gs.conts,gs.cont_bump,cp,ce);
  return mk_split(mk_task(FN_FLOW,enc_ret(ci,0),(u64)(d-1),(u64)s,l,0),
          mk_task(FN_FLOW,enc_ret(ci,1),(u64)(d-1),(u64)s,r,0));
}

// ── flow_join(_, _, fl, fr) → Node(fl,fr) ─────────────────────────────────
__device__ Result fn_flow_join(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  return mk_value(MkNode(mk_nd(a[2],a[3],gs.heap,lp,le,gs.heap_bump)));
}

// ── swap(s, t)  — args: (s, t, depth_hint, _) ────────────────────────────
//
// Faithfully implements the C code's  warp(d, s, left(t), right(t))
// but in the "compiled" form  swap(s, t)  that operates on the whole node.
//
// depth_hint is carried solely so the sequential cutoff can fire; it does
// not change the computed result.
//
__device__ Result fn_swap(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  u32 s = (u32)a[0];
  u64 t = a[1];
  u32 depth = (u32)a[2]; // depth of children of t (= warp recursion depth)

  if (IsLeaf(t)) return mk_value(t);

  u64 l = gs.heap[GetIdx(t)];
  u64 r = gs.heap[GetIdx(t)+1];

  // Sequential cutoff — switch to d_warp which mirrors the C warp() exactly
  if (depth <= SEQ_CUTOFF) {
    u64 res = d_warp(depth, s, l, r, gs.heap, lp, le, gs.heap_bump);
    return mk_value(res);
  }

  if (IsLeaf(l) && IsLeaf(r)) {
    // Base case: compare-and-swap  (matches warp_swap in C)
    u32 av = GetVal(l), bv = GetVal(r);
    u32 c = s ^ (av > bv ? 1u : 0u);
    u32 ni;
    if (c == 0) ni = mk_nd(Leaf(av), Leaf(bv), gs.heap,lp,le,gs.heap_bump);
    else    ni = mk_nd(Leaf(bv), Leaf(av), gs.heap,lp,le,gs.heap_bump);
    return mk_value(MkNode(ni));
  }

  // Recursive: build wrapper nodes  Node(left(l),left(r)), Node(right(l),right(r))
  u64 ll = gs.heap[GetIdx(l)],  lr = gs.heap[GetIdx(l)+1];
  u64 rl = gs.heap[GetIdx(r)],  rr = gs.heap[GetIdx(r)+1];
  u32 n0 = mk_nd(ll, rl, gs.heap,lp,le,gs.heap_bump);
  u32 n1 = mk_nd(lr, rr, gs.heap,lp,le,gs.heap_bump);

  u32 ci = alloc_cont(FN_SWAP_JOIN,2,ret,s,0,gs.conts,gs.cont_bump,cp,ce);
  return mk_split(
    mk_task(FN_SWAP, enc_ret(ci,0), (u64)s, MkNode(n0), (u64)(depth-1), 0),
    mk_task(FN_SWAP, enc_ret(ci,1), (u64)s, MkNode(n1), (u64)(depth-1), 0));
}

// ── swap_join(s, _, p0, p1) → unzip ──────────────────────────────────────
__device__ Result fn_swap_join(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  u64 p0=a[2], p1=a[3];
  u64 l0=gs.heap[GetIdx(p0)],  l1=gs.heap[GetIdx(p1)];
  u64 r0=gs.heap[GetIdx(p0)+1], r1=gs.heap[GetIdx(p1)+1];
  u32 li = mk_nd(l0,l1,gs.heap,lp,le,gs.heap_bump);
  u32 ri = mk_nd(r0,r1,gs.heap,lp,le,gs.heap_bump);
  return mk_value(MkNode(mk_nd(MkNode(li),MkNode(ri),gs.heap,lp,le,gs.heap_bump)));
}

// ════════════════════════════════════════════════════════════════════════════
// Dispatch
// ════════════════════════════════════════════════════════════════════════════
__device__ Result execute_task(Task &task, GState &gs, u32 &lp, u32 &le, u32 &cp, u32 &ce) {
  switch (task.fn) {
    case FN_SORT:       return fn_sort(task.args,task.ret,gs,lp,le,cp,ce);
    case FN_FLOW:       return fn_flow(task.args,task.ret,gs,lp,le,cp,ce);
    case FN_SWAP:       return fn_swap(task.args,task.ret,gs,lp,le,cp,ce);
    case FN_SORT_CONT:    return fn_sort_cont(task.args,task.ret,gs,lp,le,cp,ce);
    case FN_FLOW_AFTER_SWAP: return fn_flow_after_swap(task.args,task.ret,gs,lp,le,cp,ce);
    case FN_FLOW_JOIN:    return fn_flow_join(task.args,task.ret,gs,lp,le,cp,ce);
    case FN_SWAP_JOIN:    return fn_swap_join(task.args,task.ret,gs,lp,le,cp,ce);
    default:         return mk_value(0);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel
// ════════════════════════════════════════════════════════════════════════════
__global__ void bitonic_kernel(GState gs, Task *init_tasks, u32 num_init_tasks) {
  int tid = threadIdx.x, bid = blockIdx.x, gid = bid*BLOCK_SIZE + tid;

  // Shared-memory mailbox: much faster than global memory for intra-block comm
  __shared__ u32  s_mbox_flag[BLOCK_SIZE];
  __shared__ Task s_mbox_task[BLOCK_SIZE];
  s_mbox_flag[tid] = 0;
  __syncthreads();

  u32 lp = 0, le = 0;
  u32 cp = 0, ce = 0;
  Task my_task; bool have_task = false;

  // Pre-distributed initial tasks: one per block (thread 0 of each block)
  if (tid == 0 && (u32)bid < num_init_tasks) {
    my_task = init_tasks[bid];
    have_task = true;
  }

  u32 idle_spins = 0;
  for (;;) {
    if (*(volatile u32*)gs.done) return;

    if (!have_task) {
      // check shared-memory mailbox (very fast: ~30 cycles vs ~200 for global)
      u32 rdy = *(volatile u32*)&s_mbox_flag[tid];
      if (rdy == 2u) {
        __threadfence_block();
        my_task = s_mbox_task[tid];
        s_mbox_flag[tid] = 0u;
        have_task = true; idle_spins = 0;
      }
      // periodically check global queue
      if (!have_task) {
        idle_spins++;
        if ((idle_spins & IDLE_POLL_MASK) == 0) {
          have_task = gqueue_pop(gs, &my_task);
          if (have_task) idle_spins = 0;
        }
      }
      if (!have_task) { if (idle_spins > 2000000000u) return; continue; }
    }

    Result res = execute_task(my_task, gs, lp, le, cp, ce);
    switch (res.tag) {
      case R_VALUE: have_task = resolve_value(gs, my_task.ret, res.value, &my_task); break;
      case R_SPLIT: my_task = res.t0; push_task(gs, res.t1, tid, bid, s_mbox_flag, s_mbox_task); have_task = true; break;
      case R_CALL:  my_task = res.t0; have_task = true; break;
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Host helpers
// ════════════════════════════════════════════════════════════════════════════
static u64 *h_heap; static u32 h_heap_ptr;

static u64 host_gen(u32 d, u32 x) {
  if (d == 0) return Leaf(x);
  u64 xl = host_gen(d-1, x*2+1), xr = host_gen(d-1, x*2);
  u32 i = h_heap_ptr; h_heap_ptr += 2;
  h_heap[i] = xl; h_heap[i+1] = xr;
  return MkNode(i);
}

static void host_flatten(u64 t, u64 *heap, u32 *arr, int *pos) {
  if (IsLeaf(t)) { arr[*pos] = GetVal(t); (*pos)++; return; }
  host_flatten(heap[GetIdx(t)], heap, arr, pos);
  host_flatten(heap[GetIdx(t)+1], heap, arr, pos);
}

static void host_checksum_go(u64 t, u64 *heap, u32 *result) {
  if (IsLeaf(t)) *result = (u32)(*result * 31u) + GetVal(t);
  else { host_checksum_go(heap[GetIdx(t)],heap,result);
      host_checksum_go(heap[GetIdx(t)+1],heap,result); }
}

static u64 host_checksum(u64 t, u64 *heap) {
  u32 r = 0; host_checksum_go(t, heap, &r); return (u64)r;
}

// ════════════════════════════════════════════════════════════════════════════
// Host-side pre-split: distribute initial sort tasks across all blocks
// ════════════════════════════════════════════════════════════════════════════
static u32 host_enc_ret(u32 ci, u32 s) { return (ci << 1) | s; }

static void host_presplit(u64 tree, u32 depth, u32 s, u32 ret,
              int level, int target_level,
              Cont *conts, int *cont_idx,
              Task *tasks, int *task_idx) {
  if (level == target_level || depth <= SEQ_CUTOFF || IsLeaf(tree)) {
    Task t; t.fn = FN_SORT; t.ret = ret;
    t.args[0] = (u64)depth; t.args[1] = (u64)s; t.args[2] = tree; t.args[3] = 0;
    tasks[*task_idx] = t; (*task_idx)++;
    return;
  }
  u32 ci = (u32)(*cont_idx); (*cont_idx)++;
  Cont *c = &conts[ci];
  c->pending = 2; c->ret = ret;
  c->fn = FN_SORT_CONT; c->a0 = (u16)depth; c->a1 = (u16)s; c->_pad = 0;
  c->args[0] = 0; c->args[1] = 0;

  u64 l = h_heap[GetIdx(tree)], r = h_heap[GetIdx(tree) + 1];
  host_presplit(l, depth-1, 0, host_enc_ret(ci, 0),
         level+1, target_level, conts, cont_idx, tasks, task_idx);
  host_presplit(r, depth-1, 1, host_enc_ret(ci, 1),
         level+1, target_level, conts, cont_idx, tasks, task_idx);
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════
int main(int argc, char **argv) {
  int DEPTH = 20;
  if (argc > 1) DEPTH = atoi(argv[1]);
  if (DEPTH < 1 || DEPTH > 24) { fprintf(stderr,"Depth [1,24]\n"); return 1; }
  u32 N = 1u << DEPTH;

  fprintf(stderr, "Bitonic sort  depth=%d  elems=%u\n", DEPTH, N);
  fprintf(stderr, " config: %d blocks, %d threads/block, SEQ_CUTOFF=%d\n",
      NUM_BLOCKS, BLOCK_SIZE, SEQ_CUTOFF);

  // build tree on host
  size_t h_sz = (DEPTH <= 14) ? (1u<<26) : (1u<<22);
  h_heap = (u64*)malloc(h_sz * sizeof(u64)); h_heap_ptr = 0;
  u64 root = host_gen((u32)DEPTH, 0);
  u32 init_hp = h_heap_ptr;

  // Pre-split sort tree so all blocks start with work
  // Find largest k where 2^k <= NUM_BLOCKS
  int presplit_depth = 0;
  while ((1 << (presplit_depth+1)) <= NUM_BLOCKS) presplit_depth++;
  if (presplit_depth > DEPTH - SEQ_CUTOFF) presplit_depth = DEPTH - SEQ_CUTOFF;
  int max_tasks = 1 << (presplit_depth + 1);
  int max_conts = max_tasks;
  Task *h_init_tasks = (Task*)calloc(max_tasks, sizeof(Task));
  Cont *h_init_conts = (Cont*)calloc(max_conts, sizeof(Cont));
  int n_conts = 0, n_tasks = 0;
  host_presplit(root, (u32)DEPTH, 0, ROOT_RET, 0, presplit_depth,
         h_init_conts, &n_conts, h_init_tasks, &n_tasks);
  fprintf(stderr, " presplit: %d tasks, %d conts (depth %d)\n", n_tasks, n_conts, presplit_depth);

  // set stack for recursive device functions
  CHK(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

  // allocate GPU buffers
  GState gs;
  CHK(cudaMalloc(&gs.heap,     (size_t)HEAP_SIZE*sizeof(u64)));
  CHK(cudaMalloc(&gs.heap_bump,  sizeof(u32)));
  CHK(cudaMalloc(&gs.conts,    (size_t)CONT_BUF_SIZE*sizeof(Cont)));
  CHK(cudaMalloc(&gs.cont_bump,  sizeof(u32)));
  CHK(cudaMalloc(&gs.gqueue,    (size_t)QUEUE_BUF_SIZE*sizeof(Task)));
  CHK(cudaMalloc(&gs.gqueue_ready,(size_t)QUEUE_BUF_SIZE*sizeof(u32)));
  CHK(cudaMalloc(&gs.gqueue_push, sizeof(u32)));
  CHK(cudaMalloc(&gs.gqueue_pop, sizeof(u32)));

  CHK(cudaMalloc(&gs.done,    sizeof(u32)));
  CHK(cudaMalloc(&gs.result,   sizeof(u64)));

  CHK(cudaMemcpy(gs.heap, h_heap, (size_t)init_hp*sizeof(u64), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.heap_bump, &init_hp, sizeof(u32), cudaMemcpyHostToDevice));
  u32 zero = 0;
  // Upload pre-split conts to device cont buffer
  if (n_conts > 0)
    CHK(cudaMemcpy(gs.conts, h_init_conts, (size_t)n_conts*sizeof(Cont), cudaMemcpyHostToDevice));
  u32 init_conts = (u32)n_conts;
  CHK(cudaMemcpy(gs.cont_bump, &init_conts, sizeof(u32), cudaMemcpyHostToDevice));
  // Upload initial tasks
  Task *d_init_tasks;
  CHK(cudaMalloc(&d_init_tasks, (size_t)n_tasks*sizeof(Task)));
  CHK(cudaMemcpy(d_init_tasks, h_init_tasks, (size_t)n_tasks*sizeof(Task), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.gqueue_push, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.gqueue_pop, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.done,    &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemset(gs.gqueue_ready, 0, (size_t)QUEUE_BUF_SIZE*sizeof(u32)));


  // launch
  cudaEvent_t ev0, ev1;
  CHK(cudaEventCreate(&ev0)); CHK(cudaEventCreate(&ev1));
  CHK(cudaEventRecord(ev0));
  bitonic_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(gs, d_init_tasks, (u32)n_tasks);
  CHK(cudaEventRecord(ev1));
  CHK(cudaDeviceSynchronize());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { fprintf(stderr,"Kernel: %s\n",cudaGetErrorString(err)); return 1; }
  float ms; CHK(cudaEventElapsedTime(&ms, ev0, ev1));

  // read results
  u64 result_tree;
  CHK(cudaMemcpy(&result_tree, gs.result, sizeof(u64), cudaMemcpyDeviceToHost));
  u32 final_hp;
  CHK(cudaMemcpy(&final_hp, gs.heap_bump, sizeof(u32), cudaMemcpyDeviceToHost));
  u32 fc; CHK(cudaMemcpy(&fc, gs.cont_bump, sizeof(u32), cudaMemcpyDeviceToHost));

  fprintf(stderr, " kernel %.1f ms | heap %.2f GB | conts %u\n",
      ms, final_hp*8.0/(1024*1024*1024), fc);

  // verify
  u64 *rh = (u64*)malloc((size_t)final_hp * sizeof(u64));
  if (rh) {
    CHK(cudaMemcpy(rh, gs.heap, (size_t)final_hp*sizeof(u64), cudaMemcpyDeviceToHost));
    u32 *sorted = (u32*)malloc(N*sizeof(u32));
    int pos = 0; host_flatten(result_tree, rh, sorted, &pos);

    // checksum — matches the C code's output
    u64 cksum = host_checksum(result_tree, rh);
    printf("%llu\n", cksum);

    bool ok = true;
    for (u32 i = 1; i < N; i++)
      if (sorted[i] < sorted[i-1]) { ok = false; fprintf(stderr," SORT FAIL @%u\n",i); break; }
    fprintf(stderr, " sort %s\n", ok ? "PASS" : "FAIL");

    free(sorted); free(rh);
  }

  // cleanup
  free(h_heap);
  CHK(cudaFree(gs.heap)); CHK(cudaFree(gs.heap_bump));
  CHK(cudaFree(gs.conts)); CHK(cudaFree(gs.cont_bump));
  CHK(cudaFree(gs.gqueue)); CHK(cudaFree(gs.gqueue_ready));
  CHK(cudaFree(gs.gqueue_push)); CHK(cudaFree(gs.gqueue_pop));

  CHK(cudaFree(gs.done)); CHK(cudaFree(gs.result));
  CHK(cudaEventDestroy(ev0)); CHK(cudaEventDestroy(ev1));
  return 0;
}
