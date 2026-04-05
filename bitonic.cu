// bitonic.cu
// ==========
//
// GPU parallel evaluator for recursive functional programs, demonstrated
// with bitonic sort. Takes pure recursive functions (see bitonic.c) and
// "manually compiles" them into a task-based runtime on the GPU.
//
// The runtime is generic: it knows nothing about sort, flow, warp, gen, or
// checksum. All algorithm-specific code lives in the "Compiled Functions"
// section below, which a future compiler would auto-generate. Everything
// else is reusable infrastructure.
//
// The Original Algorithm
// ----------------------
//
// From bitonic.c (cannot be changed):
//
//   warp(d, s, a, b):
//     if d == 0: compare-and-swap leaves
//     l = warp(d-1, s, left(a),  left(b))    // PARALLEL
//     r = warp(d-1, s, right(a), right(b))   // PARALLEL
//     return Node(Node(left(l),left(r)), Node(right(l),right(r)))
//
//   flow(d, s, t):
//     if d == 0 or is_leaf(t): return t
//     w = warp(d-1, s, left(t), right(t))
//     return down(d, s, w)
//
//   down(d, s, t):
//     if d == 0 or is_leaf(t): return t
//     l = flow(d-1, s, left(t))              // PARALLEL
//     r = flow(d-1, s, right(t))             // PARALLEL
//     return Node(l, r)
//
//   sort(d, s, t):
//     if d == 0 or is_leaf(t): return t
//     l = sort(d-1, 0, left(t))              // PARALLEL
//     r = sort(d-1, 1, right(t))             // PARALLEL
//     return flow(d, s, Node(l, r))
//
//   gen(d, x):
//     if d == 0: return Leaf(x)
//     l = gen(d-1, x*2+1)                    // PARALLEL
//     r = gen(d-1, x*2)                      // PARALLEL
//     return Node(l, r)
//
//   checksum(t, d):
//     if d == 0: return val(t)
//     l = checksum(left(t), d-1)             // PARALLEL
//     r = checksum(right(t), d-1)            // PARALLEL
//     return l * 31^(2^(d-1)) + r
//
// Lines marked PARALLEL have two independent recursive calls that this
// evaluator runs concurrently on the GPU.
//
// How the Compiler Works
// ----------------------
//
// Each function with a PARALLEL annotation compiles into:
//
//   par_<name>_0   "Splitter": checks base cases, returns SPLIT or CALL.
//   par_<name>_1   "Joiner": continuation handler, combines child results.
//
// The sequential execution (WORK phase) uses an iterative evaluator with
// an explicit stack in shared memory, replacing hardware stack recursion
// with fast on-chip access (~5 cycles vs ~200 cycles for L2 misses).
//
// A task function returns one of three results:
//   VALUE(v)        Computation complete.
//   SPLIT(t0, t1)   Two parallel sub-calls needed (creates a continuation).
//   CALL(t0)        One sub-call needed (creates a 1-child continuation).
//
// A Continuation stores which joiner to call, how many children remain,
// where to send the result, and slots for child values. When the last
// child delivers its VALUE, the joiner fires.
//
// How the Runtime Works: SEED / GROW / WORK
// -----------------------------------------
//
// The runtime maintains a task matrix of NUM_BLOCKS × BLOCK_SIZE slots.
// Execution repeats three phases:
//
//   SEED (1 block): Starting from <= NUM_BLOCKS tasks, iteratively split.
//     Each SPLIT doubles the count. After ~log2(NUM_BLOCKS) rounds we
//     have one task per GPU block.
//
//   GROW (all blocks): Each block takes its tasks and splits them in
//     shared memory until it has BLOCK_SIZE tasks. Now the full matrix
//     is populated.
//
//   WORK (all blocks): Every thread runs its task via the iterative
//     evaluator. The resulting VALUE resolves continuations -- writing to
//     parent slots, firing joiners, collecting new tasks for the next round.
//
// Runs as a single cooperative kernel with grid.sync() between phases.
// No host sync, no queues, no work stealing.
//
// The sequential cutoff emerges naturally: 128 blocks x 256 threads means
// 7 + 8 = 15 doublings, so sort(20) bottoms out at sort(5) -- each thread
// sorts a 32-element subtree.
//
// Memory
// ------
//
// Trees live on a flat u64 heap. A Node at index i stores left at heap[i],
// right at heap[i+1]. The heap uses a chunked bump allocator: each thread
// gets an initial 32 KB chunk, and can dynamically acquire more via a
// global atomic counter. No per-thread limit.
//
// Continuations use a separate buffer with a chunked bump allocator.
//
// The WORK phase uses an explicit stack in shared memory for the
// iterative evaluator. Overflow spills to a per-thread global stack.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned short     u16;

// Configuration
// -------------

#define NUM_BLOCKS 128
#define BLOCK_SIZE 256
#define NUM_SLOTS  (NUM_BLOCKS * BLOCK_SIZE)
#define HEAP_CHUNK 4096
#define CONT_CHUNK 2

// Global stack: 128 MB safety net (1024 words = 256 frames per thread).
// Shared memory handles D<=21; this catches deeper recursion.
#define GSTK_SIZE  (1ull << 27)
#define GSTK_WORDS (GSTK_SIZE / NUM_SLOTS / sizeof(u32))

// Shared memory stack: first STK_WORDS of each thread's stack live here.
// Overflow spills to the per-thread global memory stack (GSTK_WORDS deep).
// Stride must be odd for bank-conflict-free shared memory access.
#define STK_WORDS  24
#define STK_STRIDE 25

// #define DEBUG_MATRIX

// Tree Encoding
// -------------
//
// A Tree is a u64. Bit 63 is the tag:
//   Leaf: bit 63 = 0, bits [31:0] = value.
//   Node: bit 63 = 1, bits [31:0] = heap index.

#define NODE_TAG (1ULL << 63)

__host__ __device__ inline u64 make_leaf(u32 val) {
  return (u64)val;
}

__host__ __device__ inline u64 make_node(u32 idx) {
  return NODE_TAG | (u64)idx;
}

__host__ __device__ inline bool is_node(u64 t) {
  return (t & NODE_TAG) != 0;
}

__host__ __device__ inline bool is_leaf(u64 t) {
  return !is_node(t);
}

__host__ __device__ inline u32 get_val(u64 t) {
  return (u32)(t & 0xFFFFFFFFu);
}

__host__ __device__ inline u32 get_idx(u64 t) {
  return (u32)(t & 0xFFFFFFFFu);
}

// Compact tree packing for stack frames.
// Packs u64 tree into u32: bit 31 = node tag, bits [30:0] = value or index.
// Valid for heap indices < 2^31 and leaf values < 2^31.

__device__ inline u32 pack_tree(u64 t) {
  if (is_node(t)) {
    return 0x80000000u | get_idx(t);
  }
  return get_val(t);
}

__device__ inline u64 unpack_tree(u32 p) {
  if (p & 0x80000000u) {
    return make_node(p & 0x7FFFFFFFu);
  }
  return make_leaf(p);
}

// Runtime Structures
// ------------------

#define R_VALUE 0
#define R_SPLIT 1
#define R_CALL  2
#define ROOT_RET 0xFFFFFFFFu

// Task: a deferred function call.
struct Task {
  u32 fn;      // function ID
  u32 ret;     // return address (continuation index << 1 | slot)
  u64 args[3]; // arguments (interpretation depends on fn)
};

// Cont: suspended computation waiting for 1-2 child results.
struct Cont {
  u32 pend;    // children still pending (decremented atomically)
  u32 ret;     // where this cont's result goes
  u16 fn;      // which joiner to call when ready
  u16 a0;      // saved argument (e.g. depth)
  u16 a1;      // saved argument (e.g. side)
  u16 _pad;
  u64 slots[2]; // child results land here
};

// Result: what a task/continuation function returns.
struct Result {
  u32  tag;
  u64  val;
  Task t0;
  Task t1;
};

// All GPU-side state.
struct State {
  u64  *heap;
  u32  *heap_ptrs;
  u32  *heap_ends;
  u32  *heap_bump;
  Cont *conts;
  u32  *cont_bump;
  Task *tasks;
  Task *flat;
  u32  *flat_cnt;
  u32  *block_cnt;
  u32  *done;
  u64  *result;
  u32  *gstk;
};

// Device Helpers
// --------------

__host__ __device__ inline u64 pack(u32 lo, u32 hi) {
  return (u64)lo | ((u64)hi << 32);
}

__host__ __device__ inline u32 lo(u64 p) {
  return (u32)p;
}

__host__ __device__ inline u32 hi(u64 p) {
  return (u32)(p >> 32);
}

__device__ inline u32 alloc_node(u64 l, u64 r, u64 *H, u32 &hp, u32 &he, u32 *hb) {
  if (hp + 2 > he) {
    hp = atomicAdd(hb, (u32)HEAP_CHUNK);
    he = hp + HEAP_CHUNK;
  }
  u32 i = hp;
  hp += 2;
  H[i] = l;
  H[i + 1] = r;
  return i;
}

__device__ inline u32 enc_ret(u32 ci, u32 slot) {
  return (ci << 1) | slot;
}

__host__ __device__ inline Task make_task(u32 fn, u32 ret, u64 a0, u64 a1 = 0, u64 a2 = 0) {
  Task t;
  t.fn = fn;
  t.ret = ret;
  t.args[0] = a0;
  t.args[1] = a1;
  t.args[2] = a2;
  return t;
}

__device__ inline Result make_value(u64 v) {
  Result r;
  r.tag = R_VALUE;
  r.val = v;
  return r;
}

__device__ inline Result make_split(Task a, Task b) {
  Result r;
  r.tag = R_SPLIT;
  r.t0 = a;
  r.t1 = b;
  return r;
}

__device__ inline Result make_call(Task a) {
  Result r;
  r.tag = R_CALL;
  r.t0 = a;
  return r;
}

__device__ inline void init_cont(Cont *c, u32 pend, u32 ret, u16 fn, u16 a0, u16 a1) {
  c->pend = pend;
  c->ret = ret;
  c->fn = fn;
  c->a0 = a0;
  c->a1 = a1;
  c->_pad = 0;
  c->slots[0] = 0;
  c->slots[1] = 0;
}

__device__ inline u32 alloc_cont(Cont *C, u32 *bump, u32 &lp, u32 &le) {
  if (lp >= le) {
    lp = atomicAdd(bump, (u32)CONT_CHUNK);
    le = lp + CONT_CHUNK;
  }
  return lp++;
}

inline void cuda_check(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
    exit(1);
  }
}
#define CHK(x) cuda_check((x), __FILE__, __LINE__)

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

// Compiled Functions
// ==================
//
// Everything in this section is algorithm-specific. The runtime below is
// fully generic -- it calls into this section only through dispatch_split,
// dispatch_cont, and dispatch_seq. A compiler would generate this section
// automatically from the annotated source.
//
// For each function with a PARALLEL annotation:
//   par_<name>_0   Splitter: returns SPLIT/CALL/VALUE (SEED/GROW phases).
//   par_<name>_1   Joiner: continuation handler (WORK resolution).
//
// The sequential implementation uses an iterative evaluator with an
// explicit stack in shared memory (see dispatch_seq below).

// Function IDs

#define FN_GEN    0
#define FN_GEN_J  1
#define FN_SORT   2
#define FN_SORT_C 3
#define FN_FLOW   4
#define FN_FLOW_A 5
#define FN_FLOW_J 6
#define FN_SWAP   7
#define FN_SWAP_J 8
#define FN_CSUM   9
#define FN_CSUM_J 10

// gen(d, x)

__device__ Result par_gen_0(u32 ret, u64 *args, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 ci) {
  u32 d = lo(args[0]);
  u32 x = hi(args[0]);
  if (d == 0) {
    return make_value(make_leaf(x));
  }
  init_cont(&C[ci], 2, ret, FN_GEN_J, 0, 0);
  Task t0 = make_task(FN_GEN, enc_ret(ci, 0), pack(d - 1, x * 2 + 1));
  Task t1 = make_task(FN_GEN, enc_ret(ci, 1), pack(d - 1, x * 2));
  return make_split(t0, t1);
}

__device__ Result par_gen_1(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  return make_value(make_node(alloc_node(co->slots[0], co->slots[1], H, hp, he, hb)));
}

// sort(d, s, t)

__device__ Result par_sort_0(u32 ret, u64 *args, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 ci) {
  u32 d = lo(args[0]);
  u32 s = hi(args[0]);
  u64 t = args[1];
  if (d == 0 || is_leaf(t)) {
    return make_value(t);
  }
  u64 l = H[get_idx(t)];
  u64 r = H[get_idx(t) + 1];
  init_cont(&C[ci], 2, ret, FN_SORT_C, (u16)d, (u16)s);
  Task t0 = make_task(FN_SORT, enc_ret(ci, 0), pack(d - 1, 0), l);
  Task t1 = make_task(FN_SORT, enc_ret(ci, 1), pack(d - 1, 1), r);
  return make_split(t0, t1);
}

__device__ Result par_sort_1(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u32 d = co->a0;
  u32 s = co->a1;
  u64 t = make_node(alloc_node(co->slots[0], co->slots[1], H, hp, he, hb));
  if (d == 0 || is_leaf(t)) {
    return make_value(t);
  }
  u32 ci = alloc_cont(C, cb, lp, le);
  init_cont(&C[ci], 1, co->ret, FN_FLOW_A, (u16)d, (u16)s);
  Task swap = make_task(FN_SWAP, enc_ret(ci, 0), pack(s, d - 1), t);
  return make_call(swap);
}

// flow(d, s, t) / down(d, s, t)

__device__ Result par_flow_0(u32 ret, u64 *args, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 ci) {
  u32 d = lo(args[0]);
  u32 s = hi(args[0]);
  u64 t = args[1];
  if (d == 0 || is_leaf(t)) {
    return make_value(t);
  }
  init_cont(&C[ci], 1, ret, FN_FLOW_A, (u16)d, (u16)s);
  Task swap = make_task(FN_SWAP, enc_ret(ci, 0), pack(s, d - 1), t);
  return make_call(swap);
}

__device__ Result par_flow_after(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u32 d = co->a0;
  u32 s = co->a1;
  u64 t = co->slots[0];
  if (is_leaf(t)) {
    return make_value(t);
  }
  u64 l = H[get_idx(t)];
  u64 r = H[get_idx(t) + 1];
  u32 ci = alloc_cont(C, cb, lp, le);
  init_cont(&C[ci], 2, co->ret, FN_FLOW_J, 0, 0);
  Task t0 = make_task(FN_FLOW, enc_ret(ci, 0), pack(d - 1, s), l);
  Task t1 = make_task(FN_FLOW, enc_ret(ci, 1), pack(d - 1, s), r);
  return make_split(t0, t1);
}

__device__ Result par_flow_join(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  return make_value(make_node(alloc_node(co->slots[0], co->slots[1], H, hp, he, hb)));
}

// warp(d, s, a, b) -- called "swap" in task IDs

__device__ Result par_swap_0(u32 ret, u64 *args, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 ci) {
  u32 s = lo(args[0]);
  u32 d = hi(args[0]);
  u64 t = args[1];
  if (is_leaf(t)) {
    return make_value(t);
  }
  u64 l = H[get_idx(t)];
  u64 r = H[get_idx(t) + 1];
  if (is_leaf(l) && is_leaf(r)) {
    u32 va = get_val(l);
    u32 vb = get_val(r);
    u32 sw = s ^ (va > vb ? 1u : 0u);
    if (sw == 0) {
      return make_value(make_node(alloc_node(make_leaf(va), make_leaf(vb), H, hp, he, hb)));
    } else {
      return make_value(make_node(alloc_node(make_leaf(vb), make_leaf(va), H, hp, he, hb)));
    }
  }
  u32 p0 = alloc_node(H[get_idx(l)], H[get_idx(r)], H, hp, he, hb);
  u32 p1 = alloc_node(H[get_idx(l) + 1], H[get_idx(r) + 1], H, hp, he, hb);
  init_cont(&C[ci], 2, ret, FN_SWAP_J, 0, 0);
  Task t0 = make_task(FN_SWAP, enc_ret(ci, 0), pack(s, d - 1), make_node(p0));
  Task t1 = make_task(FN_SWAP, enc_ret(ci, 1), pack(s, d - 1), make_node(p1));
  return make_split(t0, t1);
}

__device__ Result par_swap_1(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u64 r0 = co->slots[0];
  u64 r1 = co->slots[1];
  u32 li = alloc_node(H[get_idx(r0)], H[get_idx(r1)], H, hp, he, hb);
  u32 ri = alloc_node(H[get_idx(r0) + 1], H[get_idx(r1) + 1], H, hp, he, hb);
  return make_value(make_node(alloc_node(make_node(li), make_node(ri), H, hp, he, hb)));
}

// checksum(t, d)

__device__ u32 pow31(u32 n) {
  u32 r = 1;
  for (u32 i = 0; i < n; i++) {
    r *= 31u;
  }
  return r;
}

__device__ Result par_csum_0(u32 ret, u64 *args, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 ci) {
  u32 d = lo(args[0]);
  u64 t = args[1];
  if (d == 0) {
    return make_value((u64)get_val(t));
  }
  init_cont(&C[ci], 2, ret, FN_CSUM_J, (u16)d, 0);
  Task t0 = make_task(FN_CSUM, enc_ret(ci, 0), pack(d - 1, 0), H[get_idx(t)]);
  Task t1 = make_task(FN_CSUM, enc_ret(ci, 1), pack(d - 1, 0), H[get_idx(t) + 1]);
  return make_split(t0, t1);
}

__device__ Result par_csum_1(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u32 d = co->a0;
  u32 l = (u32)co->slots[0];
  u32 r = (u32)co->slots[1];
  return make_value((u64)(l * pow31(1u << (d - 1)) + r));
}

// Dispatch Tables
//
// The ONLY bridge between compiled functions and the generic runtime.
// The runtime calls these three functions and nothing else.

__device__ Result dispatch_split(Task &task, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 ci) {
  switch (task.fn) {
    case FN_GEN:  return par_gen_0(task.ret, task.args, H, hp, he, hb, C, ci);
    case FN_SORT: return par_sort_0(task.ret, task.args, H, hp, he, hb, C, ci);
    case FN_FLOW: return par_flow_0(task.ret, task.args, H, hp, he, hb, C, ci);
    case FN_SWAP: return par_swap_0(task.ret, task.args, H, hp, he, hb, C, ci);
    case FN_CSUM: return par_csum_0(task.ret, task.args, H, hp, he, hb, C, ci);
    default:      return make_value(0);
  }
}

__device__ Result dispatch_cont(Cont *co, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  switch (co->fn) {
    case FN_GEN_J:  return par_gen_1(co, H, hp, he, hb, C, cb, lp, le);
    case FN_SORT_C: return par_sort_1(co, H, hp, he, hb, C, cb, lp, le);
    case FN_FLOW_A: return par_flow_after(co, H, hp, he, hb, C, cb, lp, le);
    case FN_FLOW_J: return par_flow_join(co, H, hp, he, hb, C, cb, lp, le);
    case FN_SWAP_J: return par_swap_1(co, H, hp, he, hb, C, cb, lp, le);
    case FN_CSUM_J: return par_csum_1(co, H, hp, he, hb, C, cb, lp, le);
    default:        return make_value(0);
  }
}

// Iterative Sequential Evaluator
// ===============================
//
// Replaces recursive seq_* functions with a single iterative function
// using an explicit stack in shared memory. Each stack frame is 4 x u32
// = 16 bytes. Max depth is 6 frames = 96 bytes per thread.
//
// Frame layout (all types use 4 words):
//   word0: tag_word = (type << 28) | (d << 5) | s
//          bit 27 = phase (0: left done, 1: right done)
//   word1: type-specific saved data
//   word2: type-specific saved data
//   word3: left_result (packed tree or csum value)
//
// Stack frame types:
//   CF_GEN:  word1 = x
//   CF_SORT: word1 = right_child (packed tree)
//   CF_FLOW: (no extra data, just d and s)
//   CF_DOWN: word1 = right_child (packed tree)
//   CF_WARP: word1 = a_right_idx, word2 = b_right_idx
//   CF_CSUM: word1 = right_child (packed tree)

#define CF_GEN  0
#define CF_SORT 1
#define CF_FLOW 2
#define CF_DOWN 3
#define CF_WARP 4
#define CF_CSUM 5

#define PHASE_BIT (1u << 27)
#define TW(type, d, s) (((u32)(type) << 28) | ((u32)(d) << 5) | (u32)(s))
#define TW_TYPE(w)     ((w) >> 28)
#define TW_D(w)        (((w) >> 5) & 0x1Fu)
#define TW_S(w)        ((w) & 1u)

// Hybrid stack: shared memory for the first STK_WORDS, global memory beyond.
__device__ inline u32 srd(u32 *sk, u32 *gsk, u32 i) {
  if (i < STK_WORDS) {
    return sk[i];
  }
  return gsk[i - STK_WORDS];
}

__device__ inline void swr(u32 *sk, u32 *gsk, u32 i, u32 v) {
  if (i < STK_WORDS) {
    sk[i] = v;
  } else {
    gsk[i - STK_WORDS] = v;
  }
}

__device__ u64 dispatch_seq(Task &task, u64 *H, u32 &hp, u32 &he, u32 *hb, u32 *sk, u32 *gsk) {
  u32 sp = 0;
  u64 res;
  u32 fn, d, s, x;
  u64 t, a, b;

  // Decode initial task
  switch (task.fn) {
    case FN_GEN: {
      fn = CF_GEN;
      d = lo(task.args[0]);
      x = hi(task.args[0]);
      goto ENTER;
    }
    case FN_SORT: {
      fn = CF_SORT;
      d = lo(task.args[0]);
      s = hi(task.args[0]);
      t = task.args[1];
      goto ENTER;
    }
    case FN_FLOW: {
      fn = CF_FLOW;
      d = lo(task.args[0]);
      s = hi(task.args[0]);
      t = task.args[1];
      goto ENTER;
    }
    case FN_SWAP: {
      fn = CF_WARP;
      s = lo(task.args[0]);
      d = hi(task.args[0]);
      u64 tt = task.args[1];
      a = H[get_idx(tt)];
      b = H[get_idx(tt) + 1];
      goto ENTER;
    }
    case FN_CSUM: {
      fn = CF_CSUM;
      d = lo(task.args[0]);
      t = task.args[1];
      goto ENTER;
    }
    default: {
      return 0;
    }
  }

ENTER:
  switch (fn) {
    case CF_GEN:  goto ENTER_GEN;
    case CF_SORT: goto ENTER_SORT;
    case CF_FLOW: goto ENTER_FLOW;
    case CF_DOWN: goto ENTER_DOWN;
    case CF_WARP: goto ENTER_WARP;
    case CF_CSUM: goto ENTER_CSUM;
    default:      return 0;
  }

ENTER_GEN:
  if (d == 0) {
    res = make_leaf(x);
    goto POP;
  }
  swr(sk, gsk, sp, TW(CF_GEN, d, 0));
  swr(sk, gsk, sp + 1, x);
  sp += 4;
  d--;
  x = x * 2 + 1;
  goto ENTER_GEN;

ENTER_SORT:
  if (d == 0 || is_leaf(t)) {
    res = t;
    goto POP;
  }
  {
    u32 idx = get_idx(t);
    swr(sk, gsk, sp, TW(CF_SORT, d, s));
    swr(sk, gsk, sp + 1, pack_tree(H[idx + 1]));
    sp += 4;
    d--;
    s = 0;
    t = H[idx];
    goto ENTER_SORT;
  }

ENTER_FLOW:
  if (d == 0 || is_leaf(t)) {
    res = t;
    goto POP;
  }
  {
    u32 idx = get_idx(t);
    swr(sk, gsk, sp, TW(CF_FLOW, d, s));
    sp += 4;
    a = H[idx];
    b = H[idx + 1];
    d--;
    goto ENTER_WARP;
  }

ENTER_DOWN:
  if (d == 0 || is_leaf(t)) {
    res = t;
    goto POP;
  }
  {
    u32 idx = get_idx(t);
    swr(sk, gsk, sp, TW(CF_DOWN, d, s));
    swr(sk, gsk, sp + 1, pack_tree(H[idx + 1]));
    sp += 4;
    d--;
    t = H[idx];
    goto ENTER_FLOW;
  }

ENTER_WARP:
  if (d == 0) {
    u32 va = get_val(a);
    u32 vb = get_val(b);
    u32 sw = s ^ (va > vb ? 1u : 0u);
    if (sw == 0) {
      res = make_node(alloc_node(make_leaf(va), make_leaf(vb), H, hp, he, hb));
    } else {
      res = make_node(alloc_node(make_leaf(vb), make_leaf(va), H, hp, he, hb));
    }
    goto POP;
  }
  {
    u32 ai = get_idx(a);
    u32 bi = get_idx(b);
    swr(sk, gsk, sp, TW(CF_WARP, d, s));
    swr(sk, gsk, sp + 1, ai + 1);
    swr(sk, gsk, sp + 2, bi + 1);
    sp += 4;
    d--;
    a = H[ai];
    b = H[bi];
    goto ENTER_WARP;
  }

ENTER_CSUM:
  if (d == 0) {
    res = (u64)get_val(t);
    goto POP;
  }
  {
    u32 idx = get_idx(t);
    swr(sk, gsk, sp, TW(CF_CSUM, d, 0));
    swr(sk, gsk, sp + 1, pack_tree(H[idx + 1]));
    sp += 4;
    d--;
    t = H[idx];
    goto ENTER_CSUM;
  }

POP:
  if (sp == 0) {
    return res;
  }
  sp -= 4;
  {
    u32 w0 = srd(sk, gsk, sp);
    u32 ty = TW_TYPE(w0);

    // FLOW: warp done -> tail-call down (no phases)
    if (ty == CF_FLOW) {
      d = TW_D(w0);
      s = TW_S(w0);
      t = res;
      goto ENTER_DOWN;
    }

    if (w0 & PHASE_BIT) {
      // Phase 1: right call done -> combine and return
      switch (ty) {
        case CF_GEN: {
          u64 left = unpack_tree(srd(sk, gsk, sp + 3));
          res = make_node(alloc_node(left, res, H, hp, he, hb));
          goto POP;
        }
        case CF_SORT: {
          u64 left = unpack_tree(srd(sk, gsk, sp + 3));
          d = TW_D(w0);
          s = TW_S(w0);
          t = make_node(alloc_node(left, res, H, hp, he, hb));
          goto ENTER_FLOW;
        }
        case CF_DOWN: {
          u64 left = unpack_tree(srd(sk, gsk, sp + 3));
          res = make_node(alloc_node(left, res, H, hp, he, hb));
          goto POP;
        }
        case CF_WARP: {
          u64 left = unpack_tree(srd(sk, gsk, sp + 3));
          u64 right = res;
          u32 li = alloc_node(H[get_idx(left)], H[get_idx(right)], H, hp, he, hb);
          u32 ri = alloc_node(H[get_idx(left) + 1], H[get_idx(right) + 1], H, hp, he, hb);
          res = make_node(alloc_node(make_node(li), make_node(ri), H, hp, he, hb));
          goto POP;
        }
        case CF_CSUM: {
          u32 lv = srd(sk, gsk, sp + 3);
          u32 rv = (u32)res;
          d = TW_D(w0);
          res = (u64)(lv * pow31(1u << (d - 1)) + rv);
          goto POP;
        }
      }
    } else {
      // Phase 0: left call done -> save left result, start right call
      swr(sk, gsk, sp + 3, (ty == CF_CSUM) ? (u32)res : pack_tree(res));
      swr(sk, gsk, sp, w0 | PHASE_BIT);
      sp += 4;
      d = TW_D(w0);

      switch (ty) {
        case CF_GEN: {
          x = srd(sk, gsk, sp - 3);
          d--;
          x = x * 2;
          goto ENTER_GEN;
        }
        case CF_SORT: {
          t = unpack_tree(srd(sk, gsk, sp - 3));
          d--;
          s = 1;
          goto ENTER_SORT;
        }
        case CF_DOWN: {
          s = TW_S(w0);
          t = unpack_tree(srd(sk, gsk, sp - 3));
          d--;
          goto ENTER_FLOW;
        }
        case CF_WARP: {
          s = TW_S(w0);
          d--;
          a = H[srd(sk, gsk, sp - 3)];
          b = H[srd(sk, gsk, sp - 2)];
          goto ENTER_WARP;
        }
        case CF_CSUM: {
          t = unpack_tree(srd(sk, gsk, sp - 3));
          d--;
          goto ENTER_CSUM;
        }
      }
    }
  }
  return 0; // unreachable
}

// End of compiled functions -- everything below is generic runtime.

// Value Resolution
// ================
//
// When a VALUE is produced, deliver it to the parent continuation. If
// pending hits zero, fire the joiner. Results may cascade upward.

__device__ void resolve(u32 ret, u64 val, u64 *H, u32 &hp, u32 &he, u32 *hb, Cont *C, u32 &lp, u32 &le, u32 *cb, Task *out, u32 *out_n, u32 *done, u64 *result) {
  for (;;) {
    if (ret == ROOT_RET) {
      *result = val;
      __threadfence();
      atomicExch(done, 1u);
      return;
    }

    u32 ci = ret >> 1;
    u32 slot = ret & 1;
    Cont *co = &C[ci];

    // Write slot via L2 atomic -- ensures visibility without __threadfence().
    atomicExch((unsigned long long *)&co->slots[slot], val);

    u32 old = atomicSub(&co->pend, 1u);
    if (old != 1u) {
      return;
    }

    // Reload the other child's slot from L2 (bypass stale L1).
    u32 other = 1 - slot;
    co->slots[other] = __ldcg(&co->slots[other]);
    Result r = dispatch_cont(co, H, hp, he, hb, C, cb, lp, le);

    if (r.tag == R_VALUE) {
      val = r.val;
      ret = co->ret;
      continue;
    }
    if (r.tag == R_SPLIT) {
      u32 i = atomicAdd(out_n, 2u);
      out[i] = r.t0;
      out[i + 1] = r.t1;
      return;
    }
    if (r.tag == R_CALL) {
      u32 i = atomicAdd(out_n, 1u);
      out[i] = r.t0;
      return;
    }
    return;
  }
}

// Debug Visualization
// ===================

#ifdef DEBUG_MATRIX
__device__ const char *shade(u32 n) {
  if (n == 0) {
    return " ";
  }
  if (n <= (u32)(BLOCK_SIZE / 4)) {
    return "\xe2\x96\x91";
  }
  if (n <= (u32)(BLOCK_SIZE / 2)) {
    return "\xe2\x96\x92";
  }
  if (n <= (u32)(BLOCK_SIZE * 3 / 4)) {
    return "\xe2\x96\x93";
  }
  return "\xe2\x96\x88";
}
#endif

// Evaluator Kernel
// ================
//
// Single persistent cooperative kernel. Loops SEED -> GROW -> WORK until
// the root computation completes. grid.sync() separates phases.

__global__ void evaluator(State S) {
  cg::grid_group grid = cg::this_grid();
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int sid = bid * BLOCK_SIZE + tid;

  __shared__ Task buf[BLOCK_SIZE];
  __shared__ u32  stk[BLOCK_SIZE][STK_STRIDE];
  __shared__ u32  cnt, cnt_new, cb_base, cb_used;
  __shared__ u32  out_n, out_base;

  // Keep heap pointer and chunk end in registers across all rounds.
  u32 hp = S.heap_ptrs[sid];
  u32 he = S.heap_ends[sid];

#ifdef DEBUG_MATRIX
  int tick = 0;
#endif

  for (;;) {
    grid.sync();
    u32 fc = *(volatile u32 *)S.flat_cnt;
    if (fc == 0 || *(volatile u32 *)S.done) {
      break;
    }

    // SEED
    // ----
    // Block 0 iteratively splits < NUM_BLOCKS tasks until we have one
    // per block. Other blocks wait at the grid.sync() below.

    if (fc < (u32)NUM_BLOCKS) {
      if (bid == 0) {
        if (tid < fc) {
          buf[tid] = S.flat[tid];
        }
        if (tid == 0) {
          cnt = fc;
          cb_used = 0;
          cb_base = atomicAdd(S.cont_bump, 512u);
        }
        __syncthreads();

        for (int i = 0; i < 32 && cnt > 0 && cnt <= (u32)(NUM_BLOCKS / 2); i++) {
          Task my_task;
          bool active = (tid < cnt);
          if (active) {
            my_task = buf[tid];
          }
          if (tid == 0) {
            cnt_new = 0;
          }
          __syncthreads();

          if (active) {
            u32 ci = cb_base + atomicAdd(&cb_used, 1u);
            Result r = dispatch_split(my_task, S.heap, hp, he, S.heap_bump, S.conts, ci);
            if (r.tag == R_SPLIT) {
              u32 j = atomicAdd(&cnt_new, 2u);
              buf[j] = r.t0;
              buf[j + 1] = r.t1;
            } else if (r.tag == R_CALL) {
              u32 j = atomicAdd(&cnt_new, 1u);
              buf[j] = r.t0;
            }
          }
          __syncthreads();
          if (tid == 0) {
            cnt = cnt_new;
          }
          __syncthreads();
        }

        if (tid < cnt) {
          S.flat[tid] = buf[tid];
        }
        if (tid == 0) {
          *S.flat_cnt = cnt;
          __threadfence();
        }
        __syncthreads();

#ifdef DEBUG_MATRIX
        if (tid == 0) {
          printf("S%04d:", tick++);
          for (int b = 0; b < NUM_BLOCKS; b++) {
            printf("%s", shade(b == 0 ? cnt : 0));
          }
          printf("\n");
        }
#endif
      }

      grid.sync();
      fc = *(volatile u32 *)S.flat_cnt;
      if (fc == 0) {
        break;
      }
    }

    // GROW
    // ----
    // Each block loads its share of the flat buffer and iteratively splits
    // until it has BLOCK_SIZE tasks. Uses single-buffer with read-to-
    // register to avoid double-buffering.

    {
      u32 per = fc / NUM_BLOCKS;
      u32 extra = fc % NUM_BLOCKS;
      u32 my_n = per + ((u32)bid < extra ? 1u : 0u);
      u32 my_off = bid * per + ((u32)bid < extra ? (u32)bid : extra);

      if (tid < my_n) {
        buf[tid] = S.flat[my_off + tid];
      }
      if (tid == 0) {
        cnt = my_n;
        cb_used = 0;
        cb_base = atomicAdd(S.cont_bump, 256u);
      }
      __syncthreads();

      for (int i = 0; i < 32 && cnt > 0 && cnt <= (u32)(BLOCK_SIZE / 2); i++) {
        Task my_task;
        bool active = (tid < cnt);
        if (active) {
          my_task = buf[tid];
        }
        if (tid == 0) {
          cnt_new = 0;
        }
        __syncthreads();

        if (active) {
          u32 ci = cb_base + atomicAdd(&cb_used, 1u);
          Result r = dispatch_split(my_task, S.heap, hp, he, S.heap_bump, S.conts, ci);
          if (r.tag == R_SPLIT) {
            u32 j = atomicAdd(&cnt_new, 2u);
            buf[j] = r.t0;
            buf[j + 1] = r.t1;
          } else if (r.tag == R_CALL) {
            u32 j = atomicAdd(&cnt_new, 1u);
            buf[j] = r.t0;
          }
        }
        __syncthreads();
        if (tid == 0) {
          cnt = cnt_new;
        }
        __syncthreads();
      }

#ifdef DEBUG_MATRIX
      if (tid == 0) {
        S.block_cnt[bid] = cnt;
      }
#endif
    }

#ifdef DEBUG_MATRIX
    grid.sync();
    if (bid == 0 && tid == 0) {
      printf("G%04d:", tick++);
      for (int b = 0; b < NUM_BLOCKS; b++) {
        printf("%s", shade(S.block_cnt[b]));
      }
      printf("\n");
    }
    grid.sync();
#endif

    if (bid == 0 && tid == 0) {
      *S.flat_cnt = 0;
    }
    grid.sync();

    // WORK
    // ----
    // Each thread runs its task via the iterative evaluator (using the
    // shared memory stack), then resolves continuations. New tasks from
    // fired continuations are collected in buf[] and flushed to flat.

    {
      Task my_task;
      bool active = (tid < cnt);
      if (active) {
        my_task = buf[tid];
      }
      if (tid == 0) {
        out_n = 0;
      }
      __syncthreads();

      u32 clp = 0, cle = 0;

      if (active) {
        u32 *gsk = S.gstk + (u64)sid * GSTK_WORDS;
        u64 val = dispatch_seq(my_task, S.heap, hp, he, S.heap_bump, stk[tid], gsk);
        resolve(my_task.ret, val, S.heap, hp, he, S.heap_bump, S.conts, clp, cle, S.cont_bump, buf, &out_n, S.done, S.result);
      }
      __syncthreads();

      u32 n = out_n;
      if (tid == 0 && n > 0) {
        out_base = atomicAdd(S.flat_cnt, n);
      }
      __syncthreads();
      for (u32 i = tid; i < n; i += BLOCK_SIZE) {
        S.flat[out_base + i] = buf[i];
      }
    }

#ifdef DEBUG_MATRIX
    grid.sync();
    if (bid == 0 && tid == 0) {
      u32 fc2 = *(volatile u32 *)S.flat_cnt;
      u32 per = fc2 / NUM_BLOCKS;
      u32 extra = fc2 % NUM_BLOCKS;
      printf("W%04d:", tick++);
      for (int b = 0; b < NUM_BLOCKS; b++) {
        printf("%s", shade(per + ((u32)b < extra ? 1 : 0)));
      }
      printf("\n");
    }
    grid.sync();
#endif
  }

  // Write heap state back at kernel exit.
  S.heap_ptrs[sid] = hp;
  S.heap_ends[sid] = he;
}

// Heap init kernel: sets up per-thread chunk assignments on GPU.
__global__ void init_heap(u32 *ptrs, u32 *ends, u32 *bump) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid < NUM_SLOTS) {
    ptrs[sid] = (u32)sid * HEAP_CHUNK;
    ends[sid] = (u32)(sid + 1) * HEAP_CHUNK;
  }
  if (sid == 0) {
    *bump = NUM_SLOTS * HEAP_CHUNK;
  }
}

// Host
// ====

// Launch one top-level task through the evaluator. Resets state, places
// the task, and runs the cooperative kernel to completion.

static void run(State &S, u32 fn, u64 a0, u64 a1, float *ms) {
  u32 zero = 0;
  CHK(cudaMemcpy(S.done, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(S.cont_bump, &zero, sizeof(u32), cudaMemcpyHostToDevice));

  Task t = make_task(fn, ROOT_RET, a0, a1);
  CHK(cudaMemcpy(S.flat, &t, sizeof(Task), cudaMemcpyHostToDevice));

  u32 one = 1;
  CHK(cudaMemcpy(S.flat_cnt, &one, sizeof(u32), cudaMemcpyHostToDevice));

  // Maximize L1 cache (minimize shared memory carveout).
  CHK(cudaFuncSetAttribute(
    (void *)evaluator,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1));

  void *args[] = { &S };
  double w0 = now_ms();
  CHK(cudaLaunchCooperativeKernel((void *)evaluator, NUM_BLOCKS, BLOCK_SIZE, args));
  CHK(cudaDeviceSynchronize());
  double w1 = now_ms();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  *ms = (float)(w1 - w0);
}

int main(int argc, char **argv) {
  int depth = 20;
  if (argc > 1) {
    depth = atoi(argv[1]);
  }
  if (depth < 1 || depth > 24) {
    fprintf(stderr, "Usage: %s [depth 1..24]\n", argv[0]);
    return 1;
  }

  fprintf(stderr, "Bitonic sort  depth=%d  elems=%u  (%d blocks × %d threads)\n",
          depth, 1u << depth, NUM_BLOCKS, BLOCK_SIZE);

  // Size heap and continuations based on depth.
  // N<=20: 8 GB heap, 16M conts (512 MB)  → total ~8.6 GB (fast cudaMalloc)
  // N>=21: 16 GB heap, 32M conts (1 GB)   → total ~17 GB
  size_t heap_entries = (depth <= 20) ? (1ull << 30) : (1ull << 31);
  size_t cont_cap    = (depth <= 20) ? (1ull << 24) : (1ull << 25);

  // Allocate all GPU memory in one call.
  #define ALIGN(x) (((x) + 255) & ~(size_t)255)
  size_t off = 0;
  size_t p_heap = off; off += ALIGN(heap_entries * 8);
  size_t p_hptr = off; off += ALIGN((size_t)NUM_SLOTS * 4);
  size_t p_hend = off; off += ALIGN((size_t)NUM_SLOTS * 4);
  size_t p_hbmp = off; off += ALIGN(4);
  size_t p_cont = off; off += ALIGN(cont_cap * sizeof(Cont));
  size_t p_cbmp = off; off += ALIGN(4);
  size_t p_task = off; off += ALIGN((size_t)NUM_SLOTS * sizeof(Task));
  size_t p_flat = off; off += ALIGN((size_t)NUM_SLOTS * sizeof(Task));
  size_t p_fcnt = off; off += ALIGN(4);
  size_t p_bcnt = off; off += ALIGN((size_t)NUM_BLOCKS * 4);
  size_t p_done = off; off += ALIGN(4);
  size_t p_res  = off; off += ALIGN(8);
  size_t p_gstk = off; off += ALIGN((size_t)GSTK_SIZE);

  char *mem;
  CHK(cudaMalloc(&mem, off));

  State S;
  S.heap      = (u64  *)(mem + p_heap);
  S.heap_ptrs = (u32  *)(mem + p_hptr);
  S.heap_ends = (u32  *)(mem + p_hend);
  S.heap_bump = (u32  *)(mem + p_hbmp);
  S.conts     = (Cont *)(mem + p_cont);
  S.cont_bump = (u32  *)(mem + p_cbmp);
  S.tasks     = (Task *)(mem + p_task);
  S.flat      = (Task *)(mem + p_flat);
  S.flat_cnt  = (u32  *)(mem + p_fcnt);
  S.block_cnt = (u32  *)(mem + p_bcnt);
  S.done      = (u32  *)(mem + p_done);
  S.result    = (u64  *)(mem + p_res);
  S.gstk      = (u32  *)(mem + p_gstk);

  // Initialize chunked heap allocator on GPU.
  init_heap<<<(NUM_SLOTS + 255) / 256, 256>>>(S.heap_ptrs, S.heap_ends, S.heap_bump);
  CHK(cudaDeviceSynchronize());

  // Run: gen -> sort -> checksum.
  float ms;
  u64 tree, sorted, cksum;

  run(S, FN_GEN, pack(depth, 0), 0, &ms);
  CHK(cudaMemcpy(&tree, S.result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  gen  %.1f ms\n", ms);

  run(S, FN_SORT, pack(depth, 0), tree, &ms);
  CHK(cudaMemcpy(&sorted, S.result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  sort %.1f ms\n", ms);

  run(S, FN_CSUM, pack(depth, 0), sorted, &ms);
  CHK(cudaMemcpy(&cksum, S.result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  csum %.1f ms\n", ms);

  printf("%u\n", (u32)cksum);

  // Skip cudaFree -- process exit reclaims GPU memory faster.
  return 0;
}
