// bitonic.cu — Phase-based parallel evaluator for bitonic sort
//
// Single persistent cooperative kernel: SEED → GROW → WORK loop
// No global queues, no work stealing, no mailboxes. Just a task matrix.
// Per-slot heap allocation (zero contention).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned short     u16;

// ── Tree encoding (u64, bit-63 tag) ──────────────────────────────────────
#define NODE_TAG (1ULL << 63)
__host__ __device__ inline u64  Leaf(u32 v)   { return (u64)v; }
__host__ __device__ inline u64  MkNode(u32 i) { return NODE_TAG | (u64)i; }
__host__ __device__ inline bool IsNode(u64 t) { return (t & NODE_TAG) != 0; }
__host__ __device__ inline bool IsLeaf(u64 t) { return !IsNode(t); }
__host__ __device__ inline u32  GetVal(u64 t) { return (u32)(t & 0xFFFFFFFFu); }
__host__ __device__ inline u32  GetIdx(u64 t) { return (u32)(t & 0x7FFFFFFFu); }

// ── Configuration ────────────────────────────────────────────────────────
#define NB         128        // blocks (1 per SM on RTX 4090)
#define BS         256        // threads per block
#define DEBUG_MATRIX       // uncomment to print task matrix after each phase
#define NSLOTS     (NB * BS)  // 32768
#define HEAP_U64   (2u << 30) // 16 GB
#define CONT_MAX   (1u << 26) // 64M conts
#define CONT_CHUNK 2
#define ROOT_RET   0xFFFFFFFFu

#define FN_SORT   0
#define FN_FLOW   1
#define FN_SWAP   2
#define FN_SORT_C 3
#define FN_FLOW_A 4
#define FN_FLOW_J 5
#define FN_SWAP_J 6
#define FN_GEN    7
#define FN_GEN_J  8
#define FN_CSUM   9
#define FN_CSUM_J 10

#define R_VAL   0
#define R_SPLIT 1
#define R_CALL  2

struct Task   { u32 fn, ret; u64 a[3]; };
struct Cont   { u32 pend, ret; u16 fn, a0, a1, _p; u64 v[2]; };
struct Result { u32 tag; u64 val; Task t0, t1; };

struct G {
  u64  *heap;  u32 *hptrs;
  Cont *conts; u32 *cbump;
  Task *tasks, *flat;
  u32  *fcnt, *bcnt, *done;
  u64  *result;
};

#define CHK(x) do{cudaError_t _e=(x);if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

__host__ __device__ inline u64  pk(u32 a, u32 b) { return (u64)a | ((u64)b << 32); }
__host__ __device__ inline u32  lo(u64 x) { return (u32)x; }
__host__ __device__ inline u32  hi(u64 x) { return (u32)(x >> 32); }
__device__ inline u32  nd(u64 l, u64 r, u64 *H, u32 &p) {
  u32 i = p; p += 2; H[i] = l; H[i+1] = r; return i;
}
__device__ inline u32  er(u32 ci, u32 s) { return (ci << 1) | s; }
__host__ __device__ inline Task mt(u32 fn, u32 ret, u64 a0, u64 a1 = 0, u64 a2 = 0) {
  Task t; t.fn = fn; t.ret = ret; t.a[0] = a0; t.a[1] = a1; t.a[2] = a2; return t;
}
__device__ inline Result rV(u64 v)          { Result r; r.tag = R_VAL;   r.val = v; return r; }
__device__ inline Result rS(Task a, Task b) { Result r; r.tag = R_SPLIT; r.t0 = a; r.t1 = b; return r; }
__device__ inline Result rC(Task a)         { Result r; r.tag = R_CALL;  r.t0 = a; return r; }

// ── Sequential functions ─────────────────────────────────────────────────
__device__ u64 d_gen(u32 d, u32 x, u64 *H, u32 &p) {
  if (d == 0) return Leaf(x);
  u64 l = d_gen(d-1, x*2+1, H, p), r = d_gen(d-1, x*2, H, p);
  return MkNode(nd(l, r, H, p));
}
__device__ u32 d_pow31(u32 n) { u32 r = 1; for (u32 i = 0; i < n; i++) r *= 31u; return r; }
__device__ u64 d_csum(u64 t, u32 d, u64 *H) {
  if (d == 0) return (u64)GetVal(t);
  u32 l = (u32)d_csum(H[GetIdx(t)], d-1, H);
  u32 r = (u32)d_csum(H[GetIdx(t)+1], d-1, H);
  return (u64)(l * d_pow31(1u << (d-1)) + r);
}
__device__ u64 d_warp(u32 d, u32 s, u64 a, u64 b, u64 *H, u32 &p) {
  if (d == 0) {
    u32 av = GetVal(a), bv = GetVal(b), c = s ^ (av > bv ? 1u : 0u);
    return c == 0 ? MkNode(nd(Leaf(av), Leaf(bv), H, p))
                  : MkNode(nd(Leaf(bv), Leaf(av), H, p));
  }
  u64 wa = d_warp(d-1, s, H[GetIdx(a)],   H[GetIdx(b)],   H, p);
  u64 wb = d_warp(d-1, s, H[GetIdx(a)+1], H[GetIdx(b)+1], H, p);
  u32 li = nd(H[GetIdx(wa)], H[GetIdx(wb)], H, p);
  u32 ri = nd(H[GetIdx(wa)+1], H[GetIdx(wb)+1], H, p);
  return MkNode(nd(MkNode(li), MkNode(ri), H, p));
}
__device__ u64 d_flow(u32 d, u32 s, u64 t, u64 *H, u32 &p);
__device__ u64 d_down(u32 d, u32 s, u64 t, u64 *H, u32 &p) {
  if (d == 0 || IsLeaf(t)) return t;
  u64 fl = d_flow(d-1, s, H[GetIdx(t)], H, p);
  u64 fr = d_flow(d-1, s, H[GetIdx(t)+1], H, p);
  return MkNode(nd(fl, fr, H, p));
}
__device__ u64 d_flow(u32 d, u32 s, u64 t, u64 *H, u32 &p) {
  if (d == 0 || IsLeaf(t)) return t;
  u64 w = d_warp(d-1, s, H[GetIdx(t)], H[GetIdx(t)+1], H, p);
  return d_down(d, s, w, H, p);
}
__device__ u64 d_sort(u32 d, u32 s, u64 t, u64 *H, u32 &p) {
  if (d == 0 || IsLeaf(t)) return t;
  u64 sl = d_sort(d-1, 0, H[GetIdx(t)], H, p);
  u64 sr = d_sort(d-1, 1, H[GetIdx(t)+1], H, p);
  return d_flow(d, s, MkNode(nd(sl, sr, H, p)), H, p);
}

// ── Task execution for SEED/GROW ─────────────────────────────────────────
__device__ void init_cont(Cont *c, u32 pend, u32 ret, u16 fn, u16 a0, u16 a1) {
  c->pend = pend; c->ret = ret; c->fn = fn;
  c->a0 = a0; c->a1 = a1; c->_p = 0; c->v[0] = 0; c->v[1] = 0;
}

__device__ Result exec_sg(Task &t, u64 *H, u32 &hp, Cont *C, u32 ci) {
  u32 fn = t.fn, ret = t.ret;
  if (fn == FN_SORT) {
    u32 d = lo(t.a[0]), s = hi(t.a[0]); u64 tr = t.a[1];
    if (d == 0 || IsLeaf(tr)) return rV(tr);
    u64 l = H[GetIdx(tr)], r = H[GetIdx(tr)+1];
    init_cont(&C[ci], 2, ret, FN_SORT_C, (u16)d, (u16)s);
    return rS(mt(FN_SORT, er(ci,0), pk(d-1,0), l),
              mt(FN_SORT, er(ci,1), pk(d-1,1), r));
  }
  if (fn == FN_SWAP) {
    u32 s = lo(t.a[0]), depth = hi(t.a[0]); u64 tr = t.a[1];
    if (IsLeaf(tr)) return rV(tr);
    u64 l = H[GetIdx(tr)], r = H[GetIdx(tr)+1];
    if (IsLeaf(l) && IsLeaf(r)) {
      u32 av = GetVal(l), bv = GetVal(r), c = s ^ (av > bv ? 1u : 0u);
      return rV(MkNode(c == 0 ? nd(Leaf(av),Leaf(bv),H,hp) : nd(Leaf(bv),Leaf(av),H,hp)));
    }
    u32 n0 = nd(H[GetIdx(l)], H[GetIdx(r)], H, hp);
    u32 n1 = nd(H[GetIdx(l)+1], H[GetIdx(r)+1], H, hp);
    init_cont(&C[ci], 2, ret, FN_SWAP_J, 0, 0);
    return rS(mt(FN_SWAP, er(ci,0), pk(s,depth-1), MkNode(n0)),
              mt(FN_SWAP, er(ci,1), pk(s,depth-1), MkNode(n1)));
  }
  if (fn == FN_FLOW) {
    u32 d = lo(t.a[0]), s = hi(t.a[0]); u64 tr = t.a[1];
    if (d == 0 || IsLeaf(tr)) return rV(tr);
    init_cont(&C[ci], 1, ret, FN_FLOW_A, (u16)d, (u16)s);
    return rC(mt(FN_SWAP, er(ci,0), pk(s, d-1), tr));
  }
  if (fn == FN_GEN) {
    u32 d = lo(t.a[0]), x = hi(t.a[0]);
    if (d == 0) return rV(Leaf(x));
    init_cont(&C[ci], 2, ret, FN_GEN_J, 0, 0);
    return rS(mt(FN_GEN, er(ci,0), pk(d-1, x*2+1)),
              mt(FN_GEN, er(ci,1), pk(d-1, x*2)));
  }
  if (fn == FN_CSUM) {
    u32 d = lo(t.a[0]); u64 tr = t.a[1];
    if (d == 0) return rV((u64)GetVal(tr));
    init_cont(&C[ci], 2, ret, FN_CSUM_J, (u16)d, 0);
    return rS(mt(FN_CSUM, er(ci,0), pk(d-1,0), H[GetIdx(tr)]),
              mt(FN_CSUM, er(ci,1), pk(d-1,0), H[GetIdx(tr)+1]));
  }
  return rV(0);
}

// ── Continuation execution ───────────────────────────────────────────────
__device__ Result exec_cont(Cont *co, u64 *H, u32 &hp,
                            Cont *C, u32 &clp, u32 &cle, u32 *cb) {
  u16 fn = co->fn;
  if (fn == FN_SORT_C) {
    u32 d = co->a0, s = co->a1;
    u64 t = MkNode(nd(co->v[0], co->v[1], H, hp));
    if (d == 0 || IsLeaf(t)) return rV(t);
    if (clp >= cle) { clp = atomicAdd(cb, (u32)CONT_CHUNK); cle = clp + CONT_CHUNK; }
    u32 ci = clp++;
    init_cont(&C[ci], 1, co->ret, FN_FLOW_A, (u16)d, (u16)s);
    return rC(mt(FN_SWAP, er(ci,0), pk(s, d-1), t));
  }
  if (fn == FN_FLOW_A) {
    u32 d = co->a0, s = co->a1; u64 t = co->v[0];
    if (IsLeaf(t)) return rV(t);
    u64 l = H[GetIdx(t)], r = H[GetIdx(t)+1];
    if (clp >= cle) { clp = atomicAdd(cb, (u32)CONT_CHUNK); cle = clp + CONT_CHUNK; }
    u32 ci = clp++;
    init_cont(&C[ci], 2, co->ret, FN_FLOW_J, 0, 0);
    return rS(mt(FN_FLOW, er(ci,0), pk(d-1, s), l),
              mt(FN_FLOW, er(ci,1), pk(d-1, s), r));
  }
  if (fn == FN_FLOW_J) return rV(MkNode(nd(co->v[0], co->v[1], H, hp)));
  if (fn == FN_SWAP_J) {
    u64 p0 = co->v[0], p1 = co->v[1];
    u32 li = nd(H[GetIdx(p0)], H[GetIdx(p1)], H, hp);
    u32 ri = nd(H[GetIdx(p0)+1], H[GetIdx(p1)+1], H, hp);
    return rV(MkNode(nd(MkNode(li), MkNode(ri), H, hp)));
  }
  if (fn == FN_GEN_J) return rV(MkNode(nd(co->v[0], co->v[1], H, hp)));
  if (fn == FN_CSUM_J) {
    u32 d = co->a0;
    u32 l = (u32)co->v[0], r = (u32)co->v[1];
    return rV((u64)(l * d_pow31(1u << (d-1)) + r));
  }
  return rV(0);
}

// ── Value resolution ─────────────────────────────────────────────────────
__device__ void resolve(u32 ret, u64 val, u64 *H, u32 &hp,
                        Cont *C, u32 &clp, u32 &cle, u32 *cb,
                        Task *out, u32 *outn, u32 *done, u64 *result) {
  for (;;) {
    if (ret == ROOT_RET) {
      *result = val; __threadfence(); atomicExch(done, 1u); return;
    }
    u32 ci = ret >> 1, slot = ret & 1;
    Cont *co = &C[ci];
    co->v[slot] = val; __threadfence();
    u32 old = atomicSub(&co->pend, 1u);
    if (old != 1u) return;
    __threadfence();
    Result r = exec_cont(co, H, hp, C, clp, cle, cb);
    if (r.tag == R_VAL)   { val = r.val; ret = co->ret; continue; }
    if (r.tag == R_SPLIT) { u32 i = atomicAdd(outn, 2u); out[i] = r.t0; out[i+1] = r.t1; return; }
    if (r.tag == R_CALL)  { u32 i = atomicAdd(outn, 1u); out[i] = r.t0; return; }
    return;
  }
}

// ── Main cooperative kernel ──────────────────────────────────────────────
__global__ void main_kernel(G g) {
  cg::grid_group grid = cg::this_grid();
  int tid = threadIdx.x, bid = blockIdx.x;
  int slot = bid * BS + tid;

  // Shared memory for SEED/GROW double buffer AND WORK output
  __shared__ Task sb[2][BS];
  __shared__ u32  sn, snew, scb, scn;
  __shared__ Task s_out[BS];
  __shared__ u32  s_outn, s_fbase, s_bcnt;

  int round = 0;
  for (;;) {
    grid.sync();
    u32 fc = *(volatile u32*)g.fcnt;
    if (fc == 0 || *(volatile u32*)g.done) return;
#ifdef DEBUG_MATRIX
    if (bid == 0 && tid == 0) printf("R%d fc=%u\n", round, fc);
#endif
    round++;

    // ── SEED phase (block 0 only) ──
    if (fc <= (u32)NB) {
      if (bid == 0) {
        if (tid < fc) sb[0][tid] = g.flat[tid];
        if (tid == 0) { sn = fc; scn = 0; scb = atomicAdd(g.cbump, 512u); }
        __syncthreads();

        u32 hp = g.hptrs[tid];
        int cur = 0;
        for (int iter = 0; iter < 32 && sn > 0 && sn <= (u32)(NB/2); iter++) {
          if (tid == 0) snew = 0;
          __syncthreads();
          u32 n = sn;
          if (tid < n) {
            u32 ci = scb + atomicAdd(&scn, 1u);
            Result r = exec_sg(sb[cur][tid], g.heap, hp, g.conts, ci);
            if (r.tag == R_SPLIT) {
              u32 i = atomicAdd(&snew, 2u); sb[1-cur][i] = r.t0; sb[1-cur][i+1] = r.t1;
            } else if (r.tag == R_CALL) {
              u32 i = atomicAdd(&snew, 1u); sb[1-cur][i] = r.t0;
            }
          }
          __syncthreads();
          cur = 1 - cur;
          if (tid == 0) sn = snew;
          __syncthreads();
#ifdef DEBUG_MATRIX
          if (tid == 0) {
            // During SEED: only block 0 active, show as single-row grid
            for (int b = 0; b < NB; b++) g.bcnt[b] = (b == 0) ? sn : 0;
          }
          __syncthreads();
          if (tid == 0) {
            printf("  SEED iter %d:\n", iter);
            for (int r = 0; r < BS; r++) {
              for (int b = 0; b < NB; b++) printf("%c", r < (int)g.bcnt[b] ? 'X' : '.');
              printf("\n");
            }
          }
          __syncthreads();
#endif
        }
        if (tid < sn) g.flat[tid] = sb[cur][tid];
        g.hptrs[tid] = hp;
        if (tid == 0) { *g.fcnt = sn; __threadfence(); }
        __syncthreads();
      }
      grid.sync();
      fc = *(volatile u32*)g.fcnt;
      if (fc == 0) return;

    }

    // ── GROW phase (all blocks) ──
    {
      u32 K = fc / NB;
      u32 extra = fc % NB;
      u32 my_K = K + (bid < extra ? 1u : 0u);
      u32 base = bid * K + (bid < extra ? (u32)bid : extra);

      if (tid < my_K) sb[0][tid] = g.flat[base + tid];
      if (tid == 0) { sn = my_K; scn = 0; scb = atomicAdd(g.cbump, 256u); }
      __syncthreads();

      u32 hp = g.hptrs[slot];
      int cur = 0;
      for (int iter = 0; iter < 32 && sn > 0 && sn <= (u32)(BS/2); iter++) {
        if (tid == 0) snew = 0;
        __syncthreads();
        u32 n = sn;
        if (tid < n) {
          u32 ci = scb + atomicAdd(&scn, 1u);
          Result r = exec_sg(sb[cur][tid], g.heap, hp, g.conts, ci);
          if (r.tag == R_SPLIT) {
            u32 i = atomicAdd(&snew, 2u); sb[1-cur][i] = r.t0; sb[1-cur][i+1] = r.t1;
          } else if (r.tag == R_CALL) {
            u32 i = atomicAdd(&snew, 1u); sb[1-cur][i] = r.t0;
          }
        }
        __syncthreads();
        cur = 1 - cur;
        if (tid == 0) sn = snew;
        __syncthreads();
#ifdef DEBUG_MATRIX
        if (tid == 0) g.bcnt[bid] = sn;
        grid.sync();
        if (bid == 0 && tid == 0) {
          printf("  GROW iter %d:\n", iter);
          for (int r = 0; r < BS; r++) {
            for (int b = 0; b < NB; b++) printf("%c", r < (int)g.bcnt[b] ? 'X' : '.');
            printf("\n");
          }
        }
        grid.sync();
#endif
      }

      if (tid < sn) g.tasks[bid * BS + tid] = sb[cur][tid];
      g.hptrs[slot] = hp;
      if (tid == 0) g.bcnt[bid] = sn;
      __syncthreads();
    }

    // Reset fcnt before WORK
    if (bid == 0 && tid == 0) { *g.fcnt = 0; }
    grid.sync();

    // ── WORK phase (all blocks) ──
    {
      if (tid == 0) { s_outn = 0; s_bcnt = g.bcnt[bid]; }
      __syncthreads();

      u32 hp = g.hptrs[slot];
      u32 clp = 0, cle = 0;

      if (tid < s_bcnt) {
        Task task = g.tasks[slot];
        u64 value;
        switch (task.fn) {
          case FN_SORT: { u32 d=lo(task.a[0]),s=hi(task.a[0]); value=d_sort(d,s,task.a[1],g.heap,hp); break; }
          case FN_FLOW: { u32 d=lo(task.a[0]),s=hi(task.a[0]); value=d_flow(d,s,task.a[1],g.heap,hp); break; }
          case FN_SWAP: { u32 s_=lo(task.a[0]),depth=hi(task.a[0]); u64 t=task.a[1];
                          value=d_warp(depth,s_,g.heap[GetIdx(t)],g.heap[GetIdx(t)+1],g.heap,hp); break; }
          case FN_GEN:  { u32 d=lo(task.a[0]),x=hi(task.a[0]); value=d_gen(d,x,g.heap,hp); break; }
          case FN_CSUM: { u32 d=lo(task.a[0]); value=d_csum(task.a[1],d,g.heap); break; }
          default: value = 0;
        }
        resolve(task.ret, value, g.heap, hp, g.conts, clp, cle, g.cbump,
                s_out, &s_outn, g.done, g.result);
      }
      __syncthreads();
      g.hptrs[slot] = hp;

      // Bulk-write new tasks to flat buffer
      u32 outn = s_outn;
      if (tid == 0 && outn > 0) s_fbase = atomicAdd(g.fcnt, outn);
      __syncthreads();
      for (u32 i = tid; i < outn; i += BS)
        g.flat[s_fbase + i] = s_out[i];
      __syncthreads();
    }
#ifdef DEBUG_MATRIX
    grid.sync();
    if (bid == 0 && tid == 0) {
      u32 fc2 = *(volatile u32*)g.fcnt;
      // Show flat buffer as if distributed across NB columns
      // Each column gets fc2/NB tasks (approximate)
      printf("  WORK out:\n");
      u32 perb = (fc2 + NB - 1) / NB;  // ceil
      if (perb > (u32)BS) perb = BS;
      for (u32 r = 0; r < (u32)BS; r++) {
        for (int b = 0; b < NB; b++) {
          u32 my = (fc2 / NB) + ((u32)b < (fc2 % NB) ? 1 : 0);
          printf("%c", r < my ? 'X' : '.');
        }
        printf("\n");
      }
    }
    grid.sync();
#endif
    // Loop back → grid.sync() at top reads new fcnt
  }
}


// ── Run one task through the SEED/GROW/WORK pipeline ─────────────────────
static void run_task(G &g, u32 fn, u64 a0, u64 a1, float *ms_out) {
  u32 zero = 0;
  CHK(cudaMemcpy(g.done, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(g.cbump, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  Task t = mt(fn, ROOT_RET, a0, a1);
  CHK(cudaMemcpy(g.flat, &t, sizeof(Task), cudaMemcpyHostToDevice));
  u32 one = 1;
  CHK(cudaMemcpy(g.fcnt, &one, sizeof(u32), cudaMemcpyHostToDevice));
  void *args[] = { &g };
  cudaEvent_t ev0, ev1;
  CHK(cudaEventCreate(&ev0)); CHK(cudaEventCreate(&ev1));
  CHK(cudaEventRecord(ev0));
  CHK(cudaLaunchCooperativeKernel((void*)main_kernel, NB, BS, args));
  CHK(cudaEventRecord(ev1));
  CHK(cudaDeviceSynchronize());
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "Kernel: %s\n", cudaGetErrorString(e)); exit(1); }
  CHK(cudaEventElapsedTime(ms_out, ev0, ev1));
  CHK(cudaEventDestroy(ev0)); CHK(cudaEventDestroy(ev1));
}

int main(int argc, char **argv) {
  int DEPTH = 20;
  if (argc > 1) DEPTH = atoi(argv[1]);
  if (DEPTH < 1 || DEPTH > 24) { fprintf(stderr, "depth [1..24]\n"); return 1; }
  u32 N = 1u << DEPTH;
  fprintf(stderr, "Bitonic sort depth=%d elems=%u  (%d blocks × %d threads)\n", DEPTH, N, NB, BS);

  CHK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));


  // Single GPU allocation
  #define ALIGN256(x) (((x) + 255) & ~(size_t)255)
  size_t off = 0;
  size_t o_heap  = off; off += ALIGN256((size_t)HEAP_U64 * 8);
  size_t o_hptrs = off; off += ALIGN256((size_t)NSLOTS * 4);
  size_t o_conts = off; off += ALIGN256((size_t)CONT_MAX * sizeof(Cont));
  size_t o_cbump = off; off += ALIGN256(4);
  size_t o_tasks = off; off += ALIGN256((size_t)NSLOTS * sizeof(Task));
  size_t o_flat  = off; off += ALIGN256((size_t)NSLOTS * sizeof(Task));
  size_t o_fcnt  = off; off += ALIGN256(4);
  size_t o_bcnt  = off; off += ALIGN256((size_t)NB * 4);
  size_t o_done  = off; off += ALIGN256(4);
  size_t o_res   = off; off += ALIGN256(8);
  char *d_mem;
  CHK(cudaMalloc(&d_mem, off));
  CHK(cudaMemset(d_mem, 0, off));

  G g;
  g.heap  = (u64 *)(d_mem + o_heap);
  g.hptrs = (u32 *)(d_mem + o_hptrs);
  g.conts = (Cont*)(d_mem + o_conts);
  g.cbump = (u32 *)(d_mem + o_cbump);
  g.tasks = (Task*)(d_mem + o_tasks);
  g.flat  = (Task*)(d_mem + o_flat);
  g.fcnt  = (u32 *)(d_mem + o_fcnt);
  g.bcnt  = (u32 *)(d_mem + o_bcnt);
  g.done  = (u32 *)(d_mem + o_done);
  g.result= (u64 *)(d_mem + o_res);

  // Init per-slot heap pointers
  u32 slice = HEAP_U64 / NSLOTS;
  u32 *h_hps = (u32*)malloc(NSLOTS * sizeof(u32));
  for (int i = 0; i < NSLOTS; i++) h_hps[i] = (u32)i * slice;
  CHK(cudaMemcpy(g.hptrs, h_hps, NSLOTS * sizeof(u32), cudaMemcpyHostToDevice));
  free(h_hps);

  float ms;

  // gen(DEPTH, 0)
  run_task(g, FN_GEN, pk(DEPTH, 0), 0, &ms);
  u64 tree;
  CHK(cudaMemcpy(&tree, g.result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  gen  %.1f ms\n", ms);

  // sort(DEPTH, 0, tree)
  run_task(g, FN_SORT, pk(DEPTH, 0), tree, &ms);
  u64 sorted;
  CHK(cudaMemcpy(&sorted, g.result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  sort %.1f ms\n", ms);

  // checksum(sorted, DEPTH)
  run_task(g, FN_CSUM, pk(DEPTH, 0), sorted, &ms);
  u64 cksum;
  CHK(cudaMemcpy(&cksum, g.result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  csum %.1f ms\n", ms);
  printf("%u\n", (u32)cksum);

  CHK(cudaFree(d_mem));
  return 0;
}
