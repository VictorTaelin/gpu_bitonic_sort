//./bitonic.cu//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════
typedef unsigned long long u64;
typedef unsigned int    u32;

// ════════════════════════════════════════════════════════════════════════════
// Tree encoding: u64
//  Leaf: bit63=0, bits[31:0] = value
//  Node: bit63=1, bits[31:0] = heap index (left=heap[i], right=heap[i+1])
// ════════════════════════════════════════════════════════════════════════════
#define NODE_TAG (1ULL << 63)

__host__ __device__ inline u64 Leaf(u32 v)   { return (u64)v; }
__host__ __device__ inline u64 MkNode(u32 i)  { return NODE_TAG | (u64)i; }
__host__ __device__ inline bool IsNode(u64 t)  { return (t & NODE_TAG) != 0; }
__host__ __device__ inline bool IsLeaf(u64 t)  { return !IsNode(t); }
__host__ __device__ inline u32 GetVal(u64 t)  { return (u32)(t & 0xFFFFFFFFu); }
__host__ __device__ inline u32 GetIdx(u64 t)  { return (u32)(t & 0x7FFFFFFFu); }

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════
#define HEAP_SIZE    (1u << 26)  // 64M u64 slots (512 MB)
#define CONT_BUF_SIZE  (1u << 23)  // 8M continuations
#define QUEUE_BUF_SIZE (1u << 21)  // 2M overflow queue entries
#define NUM_BLOCKS   256
#define BLOCK_SIZE   256
#define NUM_THREADS   (NUM_BLOCKS * BLOCK_SIZE)
#define ALLOC_CHUNK   256     // u64 slots grabbed per local chunk

// Function IDs
#define FN_SORT       0  // sort(d, s, t)
#define FN_FLOW       1  // flow(d, s, t)
#define FN_SWAP       2  // swap(s, t)
#define FN_SORT_CONT    3  // sort continuation: (d, s, sl, sr) -> flow(d, s, Node(sl,sr))
#define FN_FLOW_AFTER_SWAP 4  // flow-after-swap: (d, s, t', _) -> split two flows
#define FN_FLOW_JOIN    5  // flow join: (_, _, fl, fr) -> Node(fl, fr)
#define FN_SWAP_JOIN    6  // swap join: (_, _, p0, p1) -> unzip

// Special return index meaning "write to root result slot"
#define ROOT_RET 0xFFFFFFFFu

// Result tags
#define R_VALUE 0
#define R_SPLIT 1
#define R_CALL 2

// ════════════════════════════════════════════════════════════════════════════
// Structs
// ════════════════════════════════════════════════════════════════════════════
struct Task {
  u32 fn;    // function id
  u32 ret;   // return index: (cont_index << 2) | arg_slot, or ROOT_RET
  u64 args[4]; // arguments (Trees or integer values cast to u64)
};

struct Cont {
  u32 fn;    // continuation function id
  u32 pending; // atomic: number of pending children (2 -> 1 -> 0)
  u32 ret;   // where this cont's result goes
  u32 _pad;
  u64 args[4]; // arguments; some slots filled by children
};

struct Result {
  u32 tag;
  u64 value;  // for R_VALUE
  Task t0, t1; // for R_SPLIT (both used), R_CALL (only t0)
};

// Global state passed to kernel
struct GState {
  u64 *heap;
  u32 *heap_bump;  // global bump allocator for heap
  Cont *conts;
  u32 *cont_bump;  // next free cont index
  Task *gqueue;    // global overflow task queue
  u32 *gqueue_ready; // per-slot ready flag (0 or 1)
  u32 *gqueue_push; // push counter (atomicAdd by producer)
  u32 *gqueue_pop;  // pop counter (CAS by consumer)
  u32 *mbox_ready;  // per-thread mailbox ready flag (0=empty, 1=reserved, 2=ready)
  Task *mbox;     // per-thread mailbox task slot
  u32 *done;     // set to 1 when root result is ready
  u64 *result;    // root result value
};

// ════════════════════════════════════════════════════════════════════════════
// CUDA error checking
// ════════════════════════════════════════════════════════════════════════════
#define CHK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while(0)

// ════════════════════════════════════════════════════════════════════════════
// Device helpers
// ════════════════════════════════════════════════════════════════════════════

// Allocate n u64 slots from per-thread local bump, falling back to global
__device__ inline u32 heap_alloc(u32 n, u32 &lp, u32 &le, u32 *g_bump) {
  if (lp + n > le) {
    lp = atomicAdd(g_bump, (u32)ALLOC_CHUNK);
    le = lp + ALLOC_CHUNK;
  }
  u32 r = lp;
  lp += n;
  return r;
}

// Allocate a Node(l, r) on the heap, return heap index
__device__ inline u32 make_node(u64 l, u64 r, u64 *heap,
                 u32 &lp, u32 &le, u32 *g_bump) {
  u32 i = heap_alloc(2, lp, le, g_bump);
  heap[i]  = l;
  heap[i+1] = r;
  return i;
}

// Allocate a continuation
__device__ inline u32 alloc_cont(u32 fn, u32 pending, u32 ret,
                 u64 a0, u64 a1, u64 a2, u64 a3,
                 Cont *conts, u32 *cont_bump) {
  u32 ci = atomicAdd(cont_bump, 1u);
  Cont *c = &conts[ci];
  c->fn   = fn;
  c->pending = pending;
  c->ret   = ret;
  c->_pad  = 0;
  c->args[0] = a0;
  c->args[1] = a1;
  c->args[2] = a2;
  c->args[3] = a3;
  return ci;
}

// Encode return index from cont index and arg slot
__device__ inline u32 enc_ret(u32 ci, u32 slot) {
  return (ci << 2) | slot;
}

// Helper constructors for Result
__device__ inline Result mk_value(u64 v) {
  Result r;
  r.tag = R_VALUE;
  r.value = v;
  return r;
}

__device__ inline Result mk_split(Task t0, Task t1) {
  Result r;
  r.tag = R_SPLIT;
  r.value = 0;
  r.t0 = t0;
  r.t1 = t1;
  return r;
}

__device__ inline Result mk_call(Task t0) {
  Result r;
  r.tag = R_CALL;
  r.value = 0;
  r.t0 = t0;
  return r;
}

// Helper constructor for Task
__device__ inline Task mk_task(u32 fn, u32 ret, u64 a0, u64 a1, u64 a2, u64 a3) {
  Task t;
  t.fn = fn;
  t.ret = ret;
  t.args[0] = a0;
  t.args[1] = a1;
  t.args[2] = a2;
  t.args[3] = a3;
  return t;
}

// ════════════════════════════════════════════════════════════════════════════
// Global queue operations
// ════════════════════════════════════════════════════════════════════════════

__device__ void gqueue_push(GState &gs, Task &task) {
  u32 idx = atomicAdd(gs.gqueue_push, 1u);
  gs.gqueue[idx % QUEUE_BUF_SIZE] = task;
  __threadfence();
  atomicExch(&gs.gqueue_ready[idx % QUEUE_BUF_SIZE], 1u);
}

__device__ bool gqueue_pop(GState &gs, Task *out) {
  for (int attempt = 0; attempt < 4; attempt++) {
    u32 pop = *(volatile u32 *)gs.gqueue_pop;
    u32 push = *(volatile u32 *)gs.gqueue_push;
    if (pop >= push) return false;
    if (atomicCAS(gs.gqueue_pop, pop, pop + 1u) == pop) {
      // Successfully claimed slot 'pop'
      u32 slot = pop % QUEUE_BUF_SIZE;
      while (atomicAdd(&gs.gqueue_ready[slot], 0u) != 1u) {
        if (*(volatile u32 *)gs.done) return false;
      }
      __threadfence();
      *out = gs.gqueue[slot];
      atomicExch(&gs.gqueue_ready[slot], 0u);
      return true;
    }
  }
  return false;
}

// ════════════════════════════════════════════════════════════════════════════
// Task pushing: neighbor probing, then global queue overflow
// ════════════════════════════════════════════════════════════════════════════

__device__ void push_task(GState &gs, Task &task, int tid, int bid) {
  int block_base = bid * BLOCK_SIZE;

  // Try neighbor probing within block via XOR distance
  for (int dist = 1; dist <= 128; dist <<= 1) {
    int tgt = block_base + (tid ^ dist);
    u32 old = atomicCAS(&gs.mbox_ready[tgt], 0u, 1u);
    if (old == 0u) {
      // Reserved the slot. Write task, fence, mark ready.
      gs.mbox[tgt] = task;
      __threadfence();
      atomicExch(&gs.mbox_ready[tgt], 2u);
      return;
    }
  }

  // All neighbors busy — push to global overflow queue
  gqueue_push(gs, task);
}

// ════════════════════════════════════════════════════════════════════════════
// Value resolution: write result, decrement continuation, possibly fire
// ════════════════════════════════════════════════════════════════════════════

__device__ bool resolve_value(GState &gs, u32 ret, u64 value, Task *out) {
  if (ret == ROOT_RET) {
    *(volatile u64 *)gs.result = value;
    __threadfence();
    atomicExch(gs.done, 1u);
    return false;
  }

  u32 ci  = ret >> 2;
  u32 slot = ret & 3;

  Cont *co = &gs.conts[ci];

  // Write the child result to the designated arg slot
  *(volatile u64 *)&co->args[slot] = value;
  __threadfence();

  // Decrement pending; if we're the last child, fire
  u32 old = atomicSub(&co->pending, 1u);
  if (old == 1u) {
    // Continuation fully saturated — convert to task
    __threadfence(); // acquire: see the other child's arg write
    out->fn = co->fn;
    out->ret = co->ret;
    out->args[0] = *(volatile u64 *)&co->args[0];
    out->args[1] = *(volatile u64 *)&co->args[1];
    out->args[2] = *(volatile u64 *)&co->args[2];
    out->args[3] = *(volatile u64 *)&co->args[3];
    return true;
  }
  return false;
}

// ════════════════════════════════════════════════════════════════════════════
// Task function implementations
// ════════════════════════════════════════════════════════════════════════════

// Forward declaration: flow logic (shared by FN_FLOW and FN_SORT_CONT)
__device__ Result flow_impl(u32 d, u32 s, u64 t, u32 ret,
               GState &gs, u32 &lp, u32 &le);

// ── FN_SORT: sort(d, s, t) ────────────────────────────────────────────────
__device__ Result fn_sort(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  u32 d = (u32)a[0];
  u32 s = (u32)a[1];
  u64 t = a[2];

  if (d == 0 || IsLeaf(t)) return mk_value(t);

  u64 l = gs.heap[GetIdx(t)];
  u64 r = gs.heap[GetIdx(t) + 1];

  u32 ci = alloc_cont(FN_SORT_CONT, 2, ret,
            (u64)d, (u64)s, 0, 0,
            gs.conts, gs.cont_bump);

  Task t0 = mk_task(FN_SORT, enc_ret(ci, 2), (u64)(d - 1), 0ULL, l, 0);
  Task t1 = mk_task(FN_SORT, enc_ret(ci, 3), (u64)(d - 1), 1ULL, r, 0);

  return mk_split(t0, t1);
}

// ── FN_SORT_CONT: (d, s, sl, sr) → flow(d, s, Node(sl, sr)) ─────────────
__device__ Result fn_sort_cont(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  u32 d = (u32)a[0];
  u32 s = (u32)a[1];
  u64 sl = a[2];
  u64 sr = a[3];

  u32 ni = make_node(sl, sr, gs.heap, lp, le, gs.heap_bump);
  return flow_impl(d, s, MkNode(ni), ret, gs, lp, le);
}

// ── flow_impl: shared flow logic ─────────────────────────────────────────
__device__ Result flow_impl(u32 d, u32 s, u64 t, u32 ret,
               GState &gs, u32 &lp, u32 &le) {
  if (d == 0 || IsLeaf(t)) return mk_value(t);

  // flow first does swap(s, t), then splits into two sub-flows.
  // Since swap can itself SPLIT (parallel), we can't inline it.
  // Create a unary continuation: after swap completes, fire FN_FLOW_AFTER_SWAP.
  u32 ci = alloc_cont(FN_FLOW_AFTER_SWAP, 1, ret,
            (u64)d, (u64)s, 0, 0,
            gs.conts, gs.cont_bump);

  Task swap_task = mk_task(FN_SWAP, enc_ret(ci, 2), (u64)s, t, 0, 0);
  return mk_call(swap_task);
}

// ── FN_FLOW: flow(d, s, t) ───────────────────────────────────────────────
__device__ Result fn_flow(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  return flow_impl((u32)a[0], (u32)a[1], a[2], ret, gs, lp, le);
}

// ── FN_FLOW_AFTER_SWAP: (d, s, t_swapped, _) → split two flows ──────────
__device__ Result fn_flow_after_swap(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  u32 d = (u32)a[0];
  u32 s = (u32)a[1];
  u64 t = a[2]; // swap result

  if (IsLeaf(t)) return mk_value(t);

  u64 l = gs.heap[GetIdx(t)];
  u64 r = gs.heap[GetIdx(t) + 1];

  u32 ci = alloc_cont(FN_FLOW_JOIN, 2, ret,
            0, 0, 0, 0,
            gs.conts, gs.cont_bump);

  Task t0 = mk_task(FN_FLOW, enc_ret(ci, 2), (u64)(d - 1), (u64)s, l, 0);
  Task t1 = mk_task(FN_FLOW, enc_ret(ci, 3), (u64)(d - 1), (u64)s, r, 0);

  return mk_split(t0, t1);
}

// ── FN_FLOW_JOIN: (_, _, fl, fr) → Node(fl, fr) ─────────────────────────
__device__ Result fn_flow_join(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  u64 fl = a[2];
  u64 fr = a[3];
  u32 ni = make_node(fl, fr, gs.heap, lp, le, gs.heap_bump);
  return mk_value(MkNode(ni));
}

// ── FN_SWAP: swap(s, t) ─────────────────────────────────────────────────
//
// swap(s, t):
//  if leaf(t): return t
//  l = left(t), r = right(t)
//  if leaf(l) && leaf(r):      // base case: compare-and-swap
//   if s==0: Node(min, max)
//   else:  Node(max, min)
//  else:              // recursive: zip-swap children
//   p0 = swap(s, Node(left(l), left(r)))
//   p1 = swap(s, Node(right(l), right(r)))
//   return Node(Node(left(p0), left(p1)),  // mins
//         Node(right(p0), right(p1))) // maxes
//
__device__ Result fn_swap(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  u32 s = (u32)a[0];
  u64 t = a[1];

  if (IsLeaf(t)) return mk_value(t);

  u64 l = gs.heap[GetIdx(t)];
  u64 r = gs.heap[GetIdx(t) + 1];

  if (IsLeaf(l) && IsLeaf(r)) {
    // Base case: compare-and-swap two leaf values
    u32 va = GetVal(l), vb = GetVal(r);
    bool need_swap = (s == 0) ? (va > vb) : (va < vb);
    if (need_swap) {
      u32 ni = make_node(Leaf(vb), Leaf(va), gs.heap, lp, le, gs.heap_bump);
      return mk_value(MkNode(ni));
    }
    return mk_value(t);
  }

  // Recursive case: both l and r are Nodes
  u64 ll = gs.heap[GetIdx(l)];
  u64 lr = gs.heap[GetIdx(l) + 1];
  u64 rl = gs.heap[GetIdx(r)];
  u64 rr = gs.heap[GetIdx(r) + 1];

  // Build pairs: Node(left(l), left(r)) and Node(right(l), right(r))
  u32 n0 = make_node(ll, rl, gs.heap, lp, le, gs.heap_bump);
  u32 n1 = make_node(lr, rr, gs.heap, lp, le, gs.heap_bump);

  u32 ci = alloc_cont(FN_SWAP_JOIN, 2, ret,
            (u64)s, 0, 0, 0,
            gs.conts, gs.cont_bump);

  Task t0 = mk_task(FN_SWAP, enc_ret(ci, 2), (u64)s, MkNode(n0), 0, 0);
  Task t1 = mk_task(FN_SWAP, enc_ret(ci, 3), (u64)s, MkNode(n1), 0, 0);

  return mk_split(t0, t1);
}

// ── FN_SWAP_JOIN: (_, _, p0, p1) → unzip ────────────────────────────────
//
// p0 = swap result for "left halves" = Node(min0s, max0s)
// p1 = swap result for "right halves" = Node(min1s, max1s)
// We want: Node(Node(left(p0), left(p1)),  ← all mins
//        Node(right(p0), right(p1))) ← all maxes
//
__device__ Result fn_swap_join(u64 *a, u32 ret, GState &gs, u32 &lp, u32 &le) {
  u64 p0 = a[2];
  u64 p1 = a[3];

  u64 l0 = gs.heap[GetIdx(p0)];   // left(p0)
  u64 l1 = gs.heap[GetIdx(p1)];   // left(p1)
  u64 r0 = gs.heap[GetIdx(p0) + 1]; // right(p0)
  u64 r1 = gs.heap[GetIdx(p1) + 1]; // right(p1)

  u32 li = make_node(l0, l1, gs.heap, lp, le, gs.heap_bump); // Node(left(p0), left(p1))
  u32 ri = make_node(r0, r1, gs.heap, lp, le, gs.heap_bump); // Node(right(p0), right(p1))
  u32 ti = make_node(MkNode(li), MkNode(ri), gs.heap, lp, le, gs.heap_bump);

  return mk_value(MkNode(ti));
}

// ════════════════════════════════════════════════════════════════════════════
// Task dispatch
// ════════════════════════════════════════════════════════════════════════════

__device__ Result execute_task(Task &task, GState &gs, u32 &lp, u32 &le) {
  switch (task.fn) {
    case FN_SORT:      return fn_sort(task.args, task.ret, gs, lp, le);
    case FN_FLOW:      return fn_flow(task.args, task.ret, gs, lp, le);
    case FN_SWAP:      return fn_swap(task.args, task.ret, gs, lp, le);
    case FN_SORT_CONT:    return fn_sort_cont(task.args, task.ret, gs, lp, le);
    case FN_FLOW_AFTER_SWAP: return fn_flow_after_swap(task.args, task.ret, gs, lp, le);
    case FN_FLOW_JOIN:    return fn_flow_join(task.args, task.ret, gs, lp, le);
    case FN_SWAP_JOIN:    return fn_swap_join(task.args, task.ret, gs, lp, le);
    default:         return mk_value(0);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel
// ════════════════════════════════════════════════════════════════════════════

__global__ void bitonic_kernel(GState gs, u64 root_tree, u32 depth) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * BLOCK_SIZE + tid;

  // Per-thread local heap bump allocator
  u32 lp = 0, le = 0;

  // Current task
  Task my_task;
  bool have_task = false;

  // Thread 0 of block 0 seeds the computation with the root sort task
  if (gid == 0) {
    my_task = mk_task(FN_SORT, ROOT_RET, (u64)depth, 0ULL, root_tree, 0ULL);
    have_task = true;
  }

  u32 idle_spins = 0;

  for (;;) {
    // ── Check termination ──────────────────────────────────────────
    if (*(volatile u32 *)gs.done) return;

    // ── Find work if idle ──────────────────────────────────────────
    if (!have_task) {
      // 1. Check per-thread mailbox (fast, local to block)
      u32 rdy = *(volatile u32 *)&gs.mbox_ready[gid];
      if (rdy == 2u) {
        __threadfence();
        my_task = gs.mbox[gid];
        atomicExch(&gs.mbox_ready[gid], 0u);
        have_task = true;
        idle_spins = 0;
      }

      // 2. Periodically check global overflow queue
      if (!have_task) {
        idle_spins++;
        if ((idle_spins & 31) == 0) { // every 32 idle iterations
          have_task = gqueue_pop(gs, &my_task);
          if (have_task) idle_spins = 0;
        }
      }

      if (!have_task) {
        if (idle_spins > 200000000u) return; // safety timeout
        continue;
      }
    }

    // ── Execute task ───────────────────────────────────────────────
    Result res = execute_task(my_task, gs, lp, le);

    switch (res.tag) {
      case R_VALUE:
        have_task = resolve_value(gs, my_task.ret, res.value, &my_task);
        break;

      case R_SPLIT:
        my_task = res.t0;    // keep first child
        push_task(gs, res.t1, tid, bid); // distribute second child
        have_task = true;
        break;

      case R_CALL:
        my_task = res.t0;    // tail-call: execute immediately
        have_task = true;
        break;
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Host: tree building and verification
// ════════════════════════════════════════════════════════════════════════════

static u64 *h_heap;
static u32 h_heap_ptr;

static u64 host_build_tree(int depth) {
  if (depth == 0) {
    return Leaf((u32)(rand() & 0xFFFF));
  }
  u64 l = host_build_tree(depth - 1);
  u64 r = host_build_tree(depth - 1);
  u32 i = h_heap_ptr;
  h_heap_ptr += 2;
  h_heap[i]   = l;
  h_heap[i + 1] = r;
  return MkNode(i);
}

static void host_flatten(u64 t, u64 *heap, u32 *arr, int *pos) {
  if (IsLeaf(t)) {
    arr[*pos] = GetVal(t);
    (*pos)++;
    return;
  }
  host_flatten(heap[GetIdx(t)],   heap, arr, pos);
  host_flatten(heap[GetIdx(t) + 1], heap, arr, pos);
}

// CPU reference implementation for verification on small inputs
static u64 cpu_swap(u32 s, u64 t, u64 *heap) {
  if (IsLeaf(t)) return t;
  u64 l = heap[GetIdx(t)];
  u64 r = heap[GetIdx(t) + 1];
  if (IsLeaf(l) && IsLeaf(r)) {
    u32 a = GetVal(l), b = GetVal(r);
    bool sw = (s == 0) ? (a > b) : (a < b);
    if (sw) {
      u32 ni = h_heap_ptr; h_heap_ptr += 2;
      heap[ni] = Leaf(b); heap[ni+1] = Leaf(a);
      return MkNode(ni);
    }
    return t;
  }
  u64 ll = heap[GetIdx(l)],   lr = heap[GetIdx(l) + 1];
  u64 rl = heap[GetIdx(r)],   rr = heap[GetIdx(r) + 1];
  u32 n0i = h_heap_ptr; h_heap_ptr += 2;
  heap[n0i] = ll; heap[n0i+1] = rl;
  u32 n1i = h_heap_ptr; h_heap_ptr += 2;
  heap[n1i] = lr; heap[n1i+1] = rr;
  u64 p0 = cpu_swap(s, MkNode(n0i), heap);
  u64 p1 = cpu_swap(s, MkNode(n1i), heap);
  u64 l0 = heap[GetIdx(p0)], l1 = heap[GetIdx(p1)];
  u64 r0 = heap[GetIdx(p0)+1], r1 = heap[GetIdx(p1)+1];
  u32 li = h_heap_ptr; h_heap_ptr += 2;
  heap[li] = l0; heap[li+1] = l1;
  u32 ri = h_heap_ptr; h_heap_ptr += 2;
  heap[ri] = r0; heap[ri+1] = r1;
  u32 ti = h_heap_ptr; h_heap_ptr += 2;
  heap[ti] = MkNode(li); heap[ti+1] = MkNode(ri);
  return MkNode(ti);
}

static u64 cpu_flow(u32 d, u32 s, u64 t, u64 *heap);

static u64 cpu_sort(u32 d, u32 s, u64 t, u64 *heap) {
  if (d == 0 || IsLeaf(t)) return t;
  u64 l = heap[GetIdx(t)];
  u64 r = heap[GetIdx(t) + 1];
  u64 sl = cpu_sort(d - 1, 0, l, heap);
  u64 sr = cpu_sort(d - 1, 1, r, heap);
  u32 ni = h_heap_ptr; h_heap_ptr += 2;
  heap[ni] = sl; heap[ni+1] = sr;
  return cpu_flow(d, s, MkNode(ni), heap);
}

static u64 cpu_flow(u32 d, u32 s, u64 t, u64 *heap) {
  if (d == 0 || IsLeaf(t)) return t;
  u64 ts = cpu_swap(s, t, heap);
  u64 l = heap[GetIdx(ts)];
  u64 r = heap[GetIdx(ts) + 1];
  u64 fl = cpu_flow(d - 1, s, l, heap);
  u64 fr = cpu_flow(d - 1, s, r, heap);
  u32 ni = h_heap_ptr; h_heap_ptr += 2;
  heap[ni] = fl; heap[ni+1] = fr;
  return MkNode(ni);
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
  int DEPTH = 16;
  if (argc > 1) DEPTH = atoi(argv[1]);
  if (DEPTH < 1 || DEPTH > 24) {
    fprintf(stderr, "Depth must be in [1, 24]\n");
    return 1;
  }

  u32 N = 1u << DEPTH;
  printf("Bitonic Sort (GPU parallel evaluator)\n");
  printf(" Depth: %d | Elements: %u\n", DEPTH, N);

  // ── Build initial tree on CPU ──────────────────────────────────────
  // Heap needs to be large enough for building + potential CPU verification
  size_t h_heap_size = (size_t)HEAP_SIZE;
  h_heap = (u64 *)malloc(h_heap_size * sizeof(u64));
  if (!h_heap) { fprintf(stderr, "Host heap allocation failed\n"); return 1; }
  h_heap_ptr = 0;

  srand(42);
  u64 root_tree = host_build_tree(DEPTH);
  u32 init_hp = h_heap_ptr;
  printf(" Initial tree heap usage: %u u64 slots (%.2f MB)\n",
      init_hp, init_hp * 8.0 / (1024.0 * 1024.0));

  // Print first few unsorted values
  {
    u32 *arr = (u32 *)malloc(N * sizeof(u32));
    int pos = 0;
    host_flatten(root_tree, h_heap, arr, &pos);
    printf(" Unsorted (first 16):");
    for (int i = 0; i < 16 && i < (int)N; i++) printf(" %u", arr[i]);
    printf(" ...\n");
    free(arr);
  }

  // ── Optional: CPU reference sort for small inputs ──────────────────
  u32 *cpu_sorted = NULL;
  if (DEPTH <= 14) {
    u32 saved_hp = h_heap_ptr;
    u64 cpu_result = cpu_sort((u32)DEPTH, 0, root_tree, h_heap);
    cpu_sorted = (u32 *)malloc(N * sizeof(u32));
    int pos = 0;
    host_flatten(cpu_result, h_heap, cpu_sorted, &pos);
    printf(" CPU sort heap usage: %u u64 slots (%.2f MB)\n",
        h_heap_ptr - saved_hp,
        (h_heap_ptr - saved_hp) * 8.0 / (1024.0 * 1024.0));
    printf(" CPU sorted (first 16):");
    for (int i = 0; i < 16 && i < (int)N; i++) printf(" %u", cpu_sorted[i]);
    printf(" ...\n");
  }

  // ── Allocate GPU memory ────────────────────────────────────────────
  GState gs;

  CHK(cudaMalloc(&gs.heap,     (size_t)HEAP_SIZE * sizeof(u64)));
  CHK(cudaMalloc(&gs.heap_bump,  sizeof(u32)));
  CHK(cudaMalloc(&gs.conts,    (size_t)CONT_BUF_SIZE * sizeof(Cont)));
  CHK(cudaMalloc(&gs.cont_bump,  sizeof(u32)));
  CHK(cudaMalloc(&gs.gqueue,    (size_t)QUEUE_BUF_SIZE * sizeof(Task)));
  CHK(cudaMalloc(&gs.gqueue_ready, (size_t)QUEUE_BUF_SIZE * sizeof(u32)));
  CHK(cudaMalloc(&gs.gqueue_push, sizeof(u32)));
  CHK(cudaMalloc(&gs.gqueue_pop,  sizeof(u32)));
  CHK(cudaMalloc(&gs.mbox_ready,  (size_t)NUM_THREADS * sizeof(u32)));
  CHK(cudaMalloc(&gs.mbox,     (size_t)NUM_THREADS * sizeof(Task)));
  CHK(cudaMalloc(&gs.done,     sizeof(u32)));
  CHK(cudaMalloc(&gs.result,    sizeof(u64)));

  // ── Copy initial tree to GPU heap ──────────────────────────────────
  CHK(cudaMemcpy(gs.heap, h_heap, (size_t)init_hp * sizeof(u64),
          cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.heap_bump, &init_hp, sizeof(u32),
          cudaMemcpyHostToDevice));

  // ── Zero-init counters and flags ───────────────────────────────────
  u32 zero = 0;
  CHK(cudaMemcpy(gs.cont_bump,  &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.gqueue_push, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.gqueue_pop, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(gs.done,    &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemset(gs.gqueue_ready, 0, (size_t)QUEUE_BUF_SIZE * sizeof(u32)));
  CHK(cudaMemset(gs.mbox_ready,  0, (size_t)NUM_THREADS * sizeof(u32)));

  // ── Launch kernel ──────────────────────────────────────────────────
  printf(" Launching kernel: %d blocks x %d threads = %d threads\n",
      NUM_BLOCKS, BLOCK_SIZE, NUM_THREADS);

  cudaEvent_t ev_start, ev_stop;
  CHK(cudaEventCreate(&ev_start));
  CHK(cudaEventCreate(&ev_stop));
  CHK(cudaEventRecord(ev_start));

  bitonic_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(gs, root_tree, (u32)DEPTH);

  CHK(cudaEventRecord(ev_stop));
  CHK(cudaDeviceSynchronize());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  float elapsed_ms;
  CHK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  printf(" Kernel time: %.3f ms\n", elapsed_ms);

  // ── Read back result ───────────────────────────────────────────────
  u64 result_tree;
  CHK(cudaMemcpy(&result_tree, gs.result, sizeof(u64), cudaMemcpyDeviceToHost));

  u32 final_hp;
  CHK(cudaMemcpy(&final_hp, gs.heap_bump, sizeof(u32), cudaMemcpyDeviceToHost));
  printf(" Final heap usage: %u u64 slots (%.2f MB)\n",
      final_hp, final_hp * 8.0 / (1024.0 * 1024.0));

  u32 final_conts;
  CHK(cudaMemcpy(&final_conts, gs.cont_bump, sizeof(u32), cudaMemcpyDeviceToHost));
  printf(" Continuations used: %u\n", final_conts);

  u32 final_qpush;
  CHK(cudaMemcpy(&final_qpush, gs.gqueue_push, sizeof(u32), cudaMemcpyDeviceToHost));
  printf(" Global queue pushes: %u\n", final_qpush);

  // ── Copy GPU heap back for verification ────────────────────────────
  u64 *result_heap = (u64 *)malloc((size_t)final_hp * sizeof(u64));
  if (!result_heap) {
    fprintf(stderr, "Cannot allocate for verification (heap too large)\n");
  } else {
    CHK(cudaMemcpy(result_heap, gs.heap, (size_t)final_hp * sizeof(u64),
            cudaMemcpyDeviceToHost));

    u32 *gpu_sorted = (u32 *)malloc(N * sizeof(u32));
    int pos = 0;
    host_flatten(result_tree, result_heap, gpu_sorted, &pos);

    printf(" GPU sorted (first 16):");
    for (int i = 0; i < 16 && i < (int)N; i++) printf(" %u", gpu_sorted[i]);
    printf(" ...\n");

    // Check ascending order
    bool sorted_ok = true;
    for (u32 i = 1; i < N; i++) {
      if (gpu_sorted[i] < gpu_sorted[i - 1]) {
        sorted_ok = false;
        printf(" !! Out of order at index %u: %u > %u\n",
            i, gpu_sorted[i-1], gpu_sorted[i]);
        break;
      }
    }
    printf(" Sort verification: %s\n", sorted_ok ? "PASS ✓" : "FAIL ✗");

    // Cross-check with CPU sort if available
    if (cpu_sorted) {
      bool match = true;
      for (u32 i = 0; i < N; i++) {
        if (gpu_sorted[i] != cpu_sorted[i]) {
          match = false;
          printf(" !! CPU/GPU mismatch at index %u: cpu=%u gpu=%u\n",
              i, cpu_sorted[i], gpu_sorted[i]);
          break;
        }
      }
      printf(" CPU/GPU match:   %s\n", match ? "PASS ✓" : "FAIL ✗");
    }

    free(gpu_sorted);
    free(result_heap);
  }

  // ── Cleanup ────────────────────────────────────────────────────────
  if (cpu_sorted) free(cpu_sorted);
  free(h_heap);

  CHK(cudaFree(gs.heap));
  CHK(cudaFree(gs.heap_bump));
  CHK(cudaFree(gs.conts));
  CHK(cudaFree(gs.cont_bump));
  CHK(cudaFree(gs.gqueue));
  CHK(cudaFree(gs.gqueue_ready));
  CHK(cudaFree(gs.gqueue_push));
  CHK(cudaFree(gs.gqueue_pop));
  CHK(cudaFree(gs.mbox_ready));
  CHK(cudaFree(gs.mbox));
  CHK(cudaFree(gs.done));
  CHK(cudaFree(gs.result));

  CHK(cudaEventDestroy(ev_start));
  CHK(cudaEventDestroy(ev_stop));

  return 0;
}
