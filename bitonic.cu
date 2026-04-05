// ==========================================================================
// bitonic.cu — GPU Parallel Evaluator for Recursive Functional Programs
// ==========================================================================
//
// This file implements a general-purpose parallel evaluator for pure
// recursive functions on the GPU. It takes functions written in a simple
// functional style (like the C reference in bitonic.c) and "manually
// compiles" them into a task-based GPU runtime that exploits parallelism.
//
// The runtime is fully generic — it knows nothing about sort, flow, warp,
// gen, or checksum. All algorithm-specific code lives in a single section
// marked "COMPILED FUNCTIONS", which a future compiler would generate
// automatically. Everything else (the scheduler, allocator, resolver) is
// reusable infrastructure.
//
// ==========================================================================
// THE ORIGINAL ALGORITHM (from bitonic.c — cannot be changed)
// ==========================================================================
//
//   gen(d, x):
//     if d == 0: return Leaf(x)
//     l = gen(d-1, x*2+1)   // PARALLEL
//     r = gen(d-1, x*2)     // PARALLEL
//     return Node(l, r)
//
//   sort(d, s, t):
//     if d == 0 or is_leaf(t): return t
//     l = sort(d-1, 0, left(t))   // PARALLEL
//     r = sort(d-1, 1, right(t))  // PARALLEL
//     return flow(d, s, Node(l, r))
//
//   flow(d, s, t):
//     if d == 0 or is_leaf(t): return t
//     w = warp(d-1, s, left(t), right(t))
//     return down(d, s, w)
//
//   down(d, s, t):
//     if d == 0 or is_leaf(t): return t
//     l = flow(d-1, s, left(t))   // PARALLEL
//     r = flow(d-1, s, right(t))  // PARALLEL
//     return Node(l, r)
//
//   warp(d, s, a, b):
//     if d == 0: compare-and-swap leaves a,b by direction s
//     l = warp(d-1, s, left(a),  left(b))    // PARALLEL
//     r = warp(d-1, s, right(a), right(b))   // PARALLEL
//     return Node(Node(left(l),left(r)), Node(right(l),right(r)))
//
//   checksum(t):
//     if is_leaf(t): return val(t)
//     return checksum(left(t)) * 31^(leaf_count(right(t))) + checksum(right(t))
//
// Lines marked PARALLEL are where two independent recursive calls can run
// concurrently. This evaluator exploits that parallelism on the GPU.
//
// ==========================================================================
// HOW THE COMPILER WORKS
// ==========================================================================
//
// Each function with a PARALLEL annotation is compiled into two parts:
//
//   par_<name>_0 (the "splitter"):
//     Checks base cases. If not a base case, allocates a Continuation and
//     returns SPLIT(child_task_0, child_task_1). The continuation stores
//     which "join" function to call when both children complete.
//
//   par_<name>_1 (the "joiner" / continuation handler):
//     Called when both children have produced values. Combines the results
//     (e.g., Node(left, right)) and returns the final VALUE — or chains
//     into another function via CALL.
//
// Functions without PARALLEL annotations (like flow) that still need a
// sub-call use CALL instead of SPLIT, creating a 1-child continuation.
//
// Each function also has a seq_<name> variant: the original recursive
// function compiled directly for single-thread execution. During the WORK
// phase, tasks call seq_* to run the subtree to completion.
//
// ==========================================================================
// HOW THE RUNTIME WORKS: SEED / GROW / WORK
// ==========================================================================
//
// The runtime maintains a "task matrix" of NUM_BLOCKS × BLOCK_SIZE slots.
// Execution proceeds in three repeating phases:
//
//   SEED  (1 block)
//     Starting from ≤ NUM_BLOCKS tasks, iteratively split them. Each SPLIT
//     doubles the count. After ~log2(NUM_BLOCKS) rounds, we have one task
//     per GPU block.
//
//   GROW  (all blocks)
//     Each block takes its task(s) and splits them in shared memory,
//     doubling each round until BLOCK_SIZE tasks fill the column.
//     Now all NUM_BLOCKS × BLOCK_SIZE slots are populated.
//
//   WORK  (all blocks)
//     Every thread executes its task sequentially (calling seq_*). The
//     resulting VALUE feeds into continuation resolution: the value is
//     written to the parent continuation's slot, and if the continuation
//     is now complete, its join function fires. New tasks (from SPLIT/CALL)
//     are collected for the next round.
//
// This runs as a single cooperative kernel with grid.sync() between phases.
// No host synchronization, no global queues, no work stealing.
//
// The sequential cutoff emerges naturally: with 128 blocks × 256 threads,
// SEED uses 7 doublings and GROW uses 8, so sort(20) bottoms out at
// sort(5) — each thread sorts a 32-element subtree sequentially.
//
// ==========================================================================
// MEMORY
// ==========================================================================
//
// Trees live on a flat heap of u64 values. A Node at index i stores its
// left child at heap[i] and right child at heap[i+1].
//
// The heap is split into equal per-thread slices. Each thread bumps a local
// pointer — zero contention, zero atomics.
//
// Continuations use a separate buffer with a chunked bump allocator.
//
// ==========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned short     u16;

// ==========================================================================
// Configuration
// ==========================================================================

#define NUM_BLOCKS 128
#define BLOCK_SIZE 256
#define NUM_SLOTS  (NUM_BLOCKS * BLOCK_SIZE)
#define HEAP_SIZE  (2u << 30)
#define CONT_CAP   (1u << 26)
#define CONT_CHUNK 2

// #define DEBUG_MATRIX

// ==========================================================================
// Tree Encoding
// ==========================================================================
//
// A Tree is a u64 with bit 63 as the tag:
//   Leaf: bit 63 = 0, bits [31:0] = value
//   Node: bit 63 = 1, bits [30:0] = heap index

#define NODE_TAG (1ULL << 63)

__host__ __device__ inline u64  make_leaf(u32 val)  { return (u64)val; }
__host__ __device__ inline u64  make_node(u32 idx)  { return NODE_TAG | (u64)idx; }
__host__ __device__ inline bool is_node(u64 t)      { return (t & NODE_TAG) != 0; }
__host__ __device__ inline bool is_leaf(u64 t)      { return !is_node(t); }
__host__ __device__ inline u32  get_val(u64 t)      { return (u32)(t & 0xFFFFFFFFu); }
__host__ __device__ inline u32  get_idx(u64 t)      { return (u32)(t & 0x7FFFFFFFu); }

// ==========================================================================
// Runtime Data Structures
// ==========================================================================

// Result tags: what a task function returns.
#define R_VALUE 0
#define R_SPLIT 1
#define R_CALL  2

// Special return address for the root computation.
#define ROOT_RET 0xFFFFFFFFu

// A Task: a deferred function call.
//   fn:   function ID
//   ret:  encoded return address (continuation index << 1 | slot)
//   args: up to 3 u64 arguments
struct Task {
  u32 fn;
  u32 ret;
  u64 args[3];
};

// A Cont: a suspended function waiting for 1 or 2 child results.
//   pend:  number of children still pending (decremented atomically)
//   ret:   where this cont's result goes
//   fn:    which join function to call when ready
//   a0,a1: saved small arguments (e.g. depth, side)
//   slots: child results written here as they arrive
struct Cont {
  u32 pend;
  u32 ret;
  u16 fn;
  u16 a0;
  u16 a1;
  u16 _pad;
  u64 slots[2];
};

// A Result: returned by task/continuation functions.
struct Result {
  u32  tag;
  u64  val;
  Task t0;
  Task t1;
};

// All GPU state.
struct State {
  u64  *heap;
  u32  *heap_ptrs;
  Cont *conts;
  u32  *cont_bump;
  Task *tasks;
  Task *flat;
  u32  *flat_cnt;
  u32  *block_cnt;
  u32  *done;
  u64  *result;
};

// ==========================================================================
// Device Helpers
// ==========================================================================

__host__ __device__ inline u64 pack(u32 lo, u32 hi) {
  return (u64)lo | ((u64)hi << 32);
}

__host__ __device__ inline u32 lo(u64 p) { return (u32)p; }
__host__ __device__ inline u32 hi(u64 p) { return (u32)(p >> 32); }

__device__ inline u32 alloc_node(u64 l, u64 r, u64 *H, u32 &hp) {
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

// ==========================================================================
// ==========================================================================
//
//    ██████  ██████  ███    ███ ██████  ██ ██      ███████ ██████
//   ██      ██    ██ ████  ████ ██   ██ ██ ██      ██      ██   ██
//   ██      ██    ██ ██ ████ ██ ██████  ██ ██      █████   ██   ██
//   ██      ██    ██ ██  ██  ██ ██      ██ ██      ██      ██   ██
//    ██████  ██████  ██      ██ ██      ██ ███████ ███████ ██████
//
//   ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
//   ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
//   █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
//   ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
//   ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████
//
// ==========================================================================
// ==========================================================================
//
// This section contains ALL algorithm-specific code. Everything above and
// below is generic runtime infrastructure that works with ANY set of
// recursive functions. A compiler would generate this section automatically.
//
// For each original function with a PARALLEL annotation, we produce:
//   seq_<name>    — direct recursive implementation (for WORK phase)
//   par_<name>_0  — splitter: returns SPLIT/CALL/VALUE (for SEED/GROW)
//   par_<name>_1  — joiner: continuation handler (for WORK resolution)
//
// Function IDs must be consecutive integers starting at 0.
// ==========================================================================

// -- Function IDs --

#define FN_GEN      0
#define FN_GEN_J    1
#define FN_SORT     2
#define FN_SORT_C   3
#define FN_FLOW     4
#define FN_FLOW_A   5
#define FN_FLOW_J   6
#define FN_SWAP     7
#define FN_SWAP_J   8
#define FN_CSUM     9
#define FN_CSUM_J   10

// --------------------------------------------------------------------------
// gen(d, x)
//   if d == 0: Leaf(x)
//   else: Node(gen(d-1, x*2+1), gen(d-1, x*2))   // PARALLEL
// --------------------------------------------------------------------------

__device__ u64 seq_gen(u32 d, u32 x, u64 *H, u32 &hp) {
  if (d == 0) {
    return make_leaf(x);
  }
  u64 l = seq_gen(d - 1, x * 2 + 1, H, hp);
  u64 r = seq_gen(d - 1, x * 2, H, hp);
  return make_node(alloc_node(l, r, H, hp));
}

__device__ Result par_gen_0(u32 ret, u64 *args, u64 *H, u32 &hp, Cont *C, u32 ci) {
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

__device__ Result par_gen_1(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  return make_value(make_node(alloc_node(co->slots[0], co->slots[1], H, hp)));
}

// --------------------------------------------------------------------------
// sort(d, s, t)
//   if d == 0 or is_leaf(t): return t
//   l = sort(d-1, 0, left(t))    // PARALLEL
//   r = sort(d-1, 1, right(t))   // PARALLEL
//   return flow(d, s, Node(l, r))
// --------------------------------------------------------------------------

__device__ u64 seq_flow(u32 d, u32 s, u64 t, u64 *H, u32 &hp);

__device__ u64 seq_sort(u32 d, u32 s, u64 t, u64 *H, u32 &hp) {
  if (d == 0 || is_leaf(t)) {
    return t;
  }
  u64 l = seq_sort(d - 1, 0, H[get_idx(t)], H, hp);
  u64 r = seq_sort(d - 1, 1, H[get_idx(t) + 1], H, hp);
  u64 node = make_node(alloc_node(l, r, H, hp));
  return seq_flow(d, s, node, H, hp);
}

__device__ Result par_sort_0(u32 ret, u64 *args, u64 *H, u32 &hp, Cont *C, u32 ci) {
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

// sort_cont: both halves sorted → Node(l,r) → flow(d, s, node)
__device__ Result par_sort_1(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u32 d = co->a0;
  u32 s = co->a1;
  u64 t = make_node(alloc_node(co->slots[0], co->slots[1], H, hp));
  if (d == 0 || is_leaf(t)) {
    return make_value(t);
  }
  // flow starts with warp, so: CALL swap, then flow_after handles the rest
  u32 ci = alloc_cont(C, cb, lp, le);
  init_cont(&C[ci], 1, co->ret, FN_FLOW_A, (u16)d, (u16)s);
  Task swap = make_task(FN_SWAP, enc_ret(ci, 0), pack(s, d - 1), t);
  return make_call(swap);
}

// --------------------------------------------------------------------------
// flow(d, s, t)
//   if d == 0 or is_leaf(t): return t
//   w = warp(d-1, s, left(t), right(t))
//   return down(d, s, w)
//
// flow has no PARALLEL of its own — it just calls warp then down.
// But down has a PARALLEL, so flow is compiled as:
//   par_flow_0: CALL warp, set up flow_after continuation
//   par_flow_after (flow_a): got warp result → SPLIT into two flows (=down)
//   par_flow_join (flow_j): got both flow results → Node(l, r)
// --------------------------------------------------------------------------

__device__ u64 seq_warp(u32 d, u32 s, u64 a, u64 b, u64 *H, u32 &hp);

__device__ u64 seq_down(u32 d, u32 s, u64 t, u64 *H, u32 &hp) {
  if (d == 0 || is_leaf(t)) {
    return t;
  }
  u64 l = seq_flow(d - 1, s, H[get_idx(t)], H, hp);
  u64 r = seq_flow(d - 1, s, H[get_idx(t) + 1], H, hp);
  return make_node(alloc_node(l, r, H, hp));
}

__device__ u64 seq_flow(u32 d, u32 s, u64 t, u64 *H, u32 &hp) {
  if (d == 0 || is_leaf(t)) {
    return t;
  }
  u64 w = seq_warp(d - 1, s, H[get_idx(t)], H[get_idx(t) + 1], H, hp);
  return seq_down(d, s, w, H, hp);
}

__device__ Result par_flow_0(u32 ret, u64 *args, u64 *H, u32 &hp, Cont *C, u32 ci) {
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

// flow_after: warp done → split into two sub-flows (= down's PARALLEL)
__device__ Result par_flow_after(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
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

// flow_join: both sub-flows done → Node(l, r)
__device__ Result par_flow_join(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  return make_value(make_node(alloc_node(co->slots[0], co->slots[1], H, hp)));
}

// --------------------------------------------------------------------------
// warp(d, s, a, b)  — called "swap" in the task system
//   if d == 0: compare-and-swap leaves
//   l = warp(d-1, s, left(a),  left(b))     // PARALLEL
//   r = warp(d-1, s, right(a), right(b))    // PARALLEL
//   return Node(Node(left(l),left(r)), Node(right(l),right(r)))
// --------------------------------------------------------------------------

__device__ u64 seq_warp(u32 d, u32 s, u64 a, u64 b, u64 *H, u32 &hp) {
  if (d == 0) {
    u32 va = get_val(a);
    u32 vb = get_val(b);
    u32 swap = s ^ (va > vb ? 1u : 0u);
    if (swap == 0) {
      return make_node(alloc_node(make_leaf(va), make_leaf(vb), H, hp));
    } else {
      return make_node(alloc_node(make_leaf(vb), make_leaf(va), H, hp));
    }
  }
  u64 wa = seq_warp(d - 1, s, H[get_idx(a)], H[get_idx(b)], H, hp);
  u64 wb = seq_warp(d - 1, s, H[get_idx(a) + 1], H[get_idx(b) + 1], H, hp);
  u32 li = alloc_node(H[get_idx(wa)], H[get_idx(wb)], H, hp);
  u32 ri = alloc_node(H[get_idx(wa) + 1], H[get_idx(wb) + 1], H, hp);
  return make_node(alloc_node(make_node(li), make_node(ri), H, hp));
}

__device__ Result par_swap_0(u32 ret, u64 *args, u64 *H, u32 &hp, Cont *C, u32 ci) {
  u32 s = lo(args[0]);
  u32 d = hi(args[0]);
  u64 t = args[1];
  if (is_leaf(t)) {
    return make_value(t);
  }
  u64 l = H[get_idx(t)];
  u64 r = H[get_idx(t) + 1];
  // Leaf-leaf base case: direct compare-and-swap
  if (is_leaf(l) && is_leaf(r)) {
    u32 va = get_val(l);
    u32 vb = get_val(r);
    u32 swap = s ^ (va > vb ? 1u : 0u);
    if (swap == 0) {
      return make_value(make_node(alloc_node(make_leaf(va), make_leaf(vb), H, hp)));
    } else {
      return make_value(make_node(alloc_node(make_leaf(vb), make_leaf(va), H, hp)));
    }
  }
  // Recursive case: pair up children and split
  u32 p0 = alloc_node(H[get_idx(l)], H[get_idx(r)], H, hp);
  u32 p1 = alloc_node(H[get_idx(l) + 1], H[get_idx(r) + 1], H, hp);
  init_cont(&C[ci], 2, ret, FN_SWAP_J, 0, 0);
  Task t0 = make_task(FN_SWAP, enc_ret(ci, 0), pack(s, d - 1), make_node(p0));
  Task t1 = make_task(FN_SWAP, enc_ret(ci, 1), pack(s, d - 1), make_node(p1));
  return make_split(t0, t1);
}

// swap_join: both halves swapped → reassemble warp structure
__device__ Result par_swap_1(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u64 r0 = co->slots[0];
  u64 r1 = co->slots[1];
  u32 li = alloc_node(H[get_idx(r0)], H[get_idx(r1)], H, hp);
  u32 ri = alloc_node(H[get_idx(r0) + 1], H[get_idx(r1) + 1], H, hp);
  return make_value(make_node(alloc_node(make_node(li), make_node(ri), H, hp)));
}

// --------------------------------------------------------------------------
// checksum(t, d)
//   if d == 0: return get_val(t)
//   l = checksum(left(t), d-1)     // PARALLEL
//   r = checksum(right(t), d-1)    // PARALLEL
//   return l * 31^(2^(d-1)) + r
// --------------------------------------------------------------------------

__device__ u32 pow31(u32 n) {
  u32 r = 1;
  for (u32 i = 0; i < n; i++) {
    r *= 31u;
  }
  return r;
}

__device__ u64 seq_csum(u64 t, u32 d, u64 *H) {
  if (d == 0) {
    return (u64)get_val(t);
  }
  u32 l = (u32)seq_csum(H[get_idx(t)], d - 1, H);
  u32 r = (u32)seq_csum(H[get_idx(t) + 1], d - 1, H);
  return (u64)(l * pow31(1u << (d - 1)) + r);
}

__device__ Result par_csum_0(u32 ret, u64 *args, u64 *H, u32 &hp, Cont *C, u32 ci) {
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

__device__ Result par_csum_1(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  u32 d = co->a0;
  u32 l = (u32)co->slots[0];
  u32 r = (u32)co->slots[1];
  return make_value((u64)(l * pow31(1u << (d - 1)) + r));
}

// --------------------------------------------------------------------------
// Dispatch tables: the ONLY place the runtime touches function IDs.
// --------------------------------------------------------------------------

// Called during SEED/GROW to split a task.
__device__ Result dispatch_split(Task &task, u64 *H, u32 &hp, Cont *C, u32 ci) {
  switch (task.fn) {
    case FN_GEN:  return par_gen_0(task.ret, task.args, H, hp, C, ci);
    case FN_SORT: return par_sort_0(task.ret, task.args, H, hp, C, ci);
    case FN_FLOW: return par_flow_0(task.ret, task.args, H, hp, C, ci);
    case FN_SWAP: return par_swap_0(task.ret, task.args, H, hp, C, ci);
    case FN_CSUM: return par_csum_0(task.ret, task.args, H, hp, C, ci);
    default:      return make_value(0);
  }
}

// Called during WORK to fire a completed continuation.
__device__ Result dispatch_cont(Cont *co, u64 *H, u32 &hp, Cont *C, u32 *cb, u32 &lp, u32 &le) {
  switch (co->fn) {
    case FN_GEN_J:  return par_gen_1(co, H, hp, C, cb, lp, le);
    case FN_SORT_C: return par_sort_1(co, H, hp, C, cb, lp, le);
    case FN_FLOW_A: return par_flow_after(co, H, hp, C, cb, lp, le);
    case FN_FLOW_J: return par_flow_join(co, H, hp, C, cb, lp, le);
    case FN_SWAP_J: return par_swap_1(co, H, hp, C, cb, lp, le);
    case FN_CSUM_J: return par_csum_1(co, H, hp, C, cb, lp, le);
    default:        return make_value(0);
  }
}

// Called during WORK to execute a task sequentially.
__device__ u64 dispatch_seq(Task &task, u64 *H, u32 &hp) {
  switch (task.fn) {
    case FN_GEN: {
      u32 d = lo(task.args[0]);
      u32 x = hi(task.args[0]);
      return seq_gen(d, x, H, hp);
    }
    case FN_SORT: {
      u32 d = lo(task.args[0]);
      u32 s = hi(task.args[0]);
      return seq_sort(d, s, task.args[1], H, hp);
    }
    case FN_FLOW: {
      u32 d = lo(task.args[0]);
      u32 s = hi(task.args[0]);
      return seq_flow(d, s, task.args[1], H, hp);
    }
    case FN_SWAP: {
      u32 s = lo(task.args[0]);
      u32 d = hi(task.args[0]);
      u64 t = task.args[1];
      return seq_warp(d, s, H[get_idx(t)], H[get_idx(t) + 1], H, hp);
    }
    case FN_CSUM: {
      u32 d = lo(task.args[0]);
      return seq_csum(task.args[1], d, H);
    }
    default:
      return 0;
  }
}

// ==========================================================================
// ==========================================================================
//
//   END OF COMPILED FUNCTIONS — everything below is generic runtime.
//
// ==========================================================================
// ==========================================================================

// ==========================================================================
// Value Resolution
// ==========================================================================
//
// When a VALUE is produced, deliver it to the parent continuation. If the
// continuation becomes complete (pending == 0), fire it. The result may
// cascade upward (another VALUE) or produce new tasks (SPLIT/CALL).

__device__ void resolve(u32 ret, u64 val, u64 *H, u32 &hp, Cont *C, u32 &lp, u32 &le, u32 *cb, Task *out, u32 *out_n, u32 *done, u64 *result) {
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

    co->slots[slot] = val;
    __threadfence();

    u32 old = atomicSub(&co->pend, 1u);
    if (old != 1u) {
      return;
    }

    // We were the last child — fire the continuation.
    __threadfence();
    Result r = dispatch_cont(co, H, hp, C, cb, lp, le);

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

// ==========================================================================
// Debug Visualization
// ==========================================================================

#ifdef DEBUG_MATRIX
__device__ const char *shade(u32 n) {
  if (n == 0)                    { return " "; }
  if (n <= (u32)(BLOCK_SIZE / 4))  { return "░"; }
  if (n <= (u32)(BLOCK_SIZE / 2))  { return "▒"; }
  if (n <= (u32)(BLOCK_SIZE * 3 / 4)) { return "▓"; }
  return "█";
}
#endif

// ==========================================================================
// Evaluator Kernel
// ==========================================================================
//
// Single persistent cooperative kernel. Loops SEED → GROW → WORK until the
// root computation is complete. grid.sync() separates phases.

__global__ void evaluator(State S) {
  cg::grid_group grid = cg::this_grid();
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int sid = bid * BLOCK_SIZE + tid;

  __shared__ Task buf[2][BLOCK_SIZE];
  __shared__ u32  cnt, cnt_new, cb_base, cb_used;
  __shared__ Task out[BLOCK_SIZE];
  __shared__ u32  out_n, out_base, work_n;

#ifdef DEBUG_MATRIX
  int tick = 0;
#endif

  for (;;) {
    grid.sync();
    u32 fc = *(volatile u32 *)S.flat_cnt;
    if (fc == 0 || *(volatile u32 *)S.done) {
      return;
    }

    // ── SEED ─────────────────────────────────────────────────────────
    if (fc <= (u32)NUM_BLOCKS) {
      if (bid == 0) {
        if (tid < fc) {
          buf[0][tid] = S.flat[tid];
        }
        if (tid == 0) {
          cnt = fc;
          cb_used = 0;
          cb_base = atomicAdd(S.cont_bump, 512u);
        }
        __syncthreads();

        u32 hp = S.heap_ptrs[tid];
        int cur = 0;

        for (int i = 0; i < 32 && cnt > 0 && cnt <= (u32)(NUM_BLOCKS / 2); i++) {
          if (tid == 0) {
            cnt_new = 0;
          }
          __syncthreads();

          if (tid < cnt) {
            u32 ci = cb_base + atomicAdd(&cb_used, 1u);
            Result r = dispatch_split(buf[cur][tid], S.heap, hp, S.conts, ci);
            if (r.tag == R_SPLIT) {
              u32 j = atomicAdd(&cnt_new, 2u);
              buf[1 - cur][j] = r.t0;
              buf[1 - cur][j + 1] = r.t1;
            } else if (r.tag == R_CALL) {
              u32 j = atomicAdd(&cnt_new, 1u);
              buf[1 - cur][j] = r.t0;
            }
          }
          __syncthreads();
          cur = 1 - cur;
          if (tid == 0) {
            cnt = cnt_new;
          }
          __syncthreads();
        }

        if (tid < cnt) {
          S.flat[tid] = buf[cur][tid];
        }
        S.heap_ptrs[tid] = hp;
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
        return;
      }
    }

    // ── GROW ─────────────────────────────────────────────────────────
    {
      u32 per = fc / NUM_BLOCKS;
      u32 extra = fc % NUM_BLOCKS;
      u32 my_n = per + ((u32)bid < extra ? 1u : 0u);
      u32 my_off = bid * per + ((u32)bid < extra ? (u32)bid : extra);

      if (tid < my_n) {
        buf[0][tid] = S.flat[my_off + tid];
      }
      if (tid == 0) {
        cnt = my_n;
        cb_used = 0;
        cb_base = atomicAdd(S.cont_bump, 256u);
      }
      __syncthreads();

      u32 hp = S.heap_ptrs[sid];
      int cur = 0;

      for (int i = 0; i < 32 && cnt > 0 && cnt <= (u32)(BLOCK_SIZE / 2); i++) {
        if (tid == 0) {
          cnt_new = 0;
        }
        __syncthreads();

        if (tid < cnt) {
          u32 ci = cb_base + atomicAdd(&cb_used, 1u);
          Result r = dispatch_split(buf[cur][tid], S.heap, hp, S.conts, ci);
          if (r.tag == R_SPLIT) {
            u32 j = atomicAdd(&cnt_new, 2u);
            buf[1 - cur][j] = r.t0;
            buf[1 - cur][j + 1] = r.t1;
          } else if (r.tag == R_CALL) {
            u32 j = atomicAdd(&cnt_new, 1u);
            buf[1 - cur][j] = r.t0;
          }
        }
        __syncthreads();
        cur = 1 - cur;
        if (tid == 0) {
          cnt = cnt_new;
        }
        __syncthreads();
      }

      if (tid < cnt) {
        S.tasks[bid * BLOCK_SIZE + tid] = buf[cur][tid];
      }
      S.heap_ptrs[sid] = hp;
      if (tid == 0) {
        S.block_cnt[bid] = cnt;
      }
      __syncthreads();
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

    // ── WORK ─────────────────────────────────────────────────────────
    {
      if (tid == 0) {
        out_n = 0;
        work_n = S.block_cnt[bid];
      }
      __syncthreads();

      u32 hp = S.heap_ptrs[sid];
      u32 clp = 0, cle = 0;

      if (tid < work_n) {
        Task task = S.tasks[sid];
        u64 val = dispatch_seq(task, S.heap, hp);
        resolve(task.ret, val, S.heap, hp, S.conts, clp, cle, S.cont_bump, out, &out_n, S.done, S.result);
      }
      __syncthreads();

      S.heap_ptrs[sid] = hp;

      u32 n = out_n;
      if (tid == 0 && n > 0) {
        out_base = atomicAdd(S.flat_cnt, n);
      }
      __syncthreads();
      for (u32 i = tid; i < n; i += BLOCK_SIZE) {
        S.flat[out_base + i] = out[i];
      }
      __syncthreads();
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
}

// ==========================================================================
// Host: launch one top-level task through the evaluator
// ==========================================================================

static void run(State &S, u32 fn, u64 a0, u64 a1, float *ms) {
  u32 zero = 0;
  CHK(cudaMemcpy(S.done, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(S.cont_bump, &zero, sizeof(u32), cudaMemcpyHostToDevice));

  Task t = make_task(fn, ROOT_RET, a0, a1);
  CHK(cudaMemcpy(S.flat, &t, sizeof(Task), cudaMemcpyHostToDevice));

  u32 one = 1;
  CHK(cudaMemcpy(S.flat_cnt, &one, sizeof(u32), cudaMemcpyHostToDevice));

  void *args[] = { &S };
  cudaEvent_t t0, t1;
  CHK(cudaEventCreate(&t0));
  CHK(cudaEventCreate(&t1));
  CHK(cudaEventRecord(t0));
  CHK(cudaLaunchCooperativeKernel((void *)evaluator, NUM_BLOCKS, BLOCK_SIZE, args));
  CHK(cudaEventRecord(t1));
  CHK(cudaDeviceSynchronize());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  CHK(cudaEventElapsedTime(ms, t0, t1));
  CHK(cudaEventDestroy(t0));
  CHK(cudaEventDestroy(t1));
}

// ==========================================================================
// Main
// ==========================================================================

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

  CHK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

  // -- Allocate all GPU memory in one call --
  #define ALIGN(x) (((x) + 255) & ~(size_t)255)
  size_t off = 0;
  size_t p_heap = off; off += ALIGN((size_t)HEAP_SIZE * 8);
  size_t p_hptr = off; off += ALIGN((size_t)NUM_SLOTS * 4);
  size_t p_cont = off; off += ALIGN((size_t)CONT_CAP * sizeof(Cont));
  size_t p_cbmp = off; off += ALIGN(4);
  size_t p_task = off; off += ALIGN((size_t)NUM_SLOTS * sizeof(Task));
  size_t p_flat = off; off += ALIGN((size_t)NUM_SLOTS * sizeof(Task));
  size_t p_fcnt = off; off += ALIGN(4);
  size_t p_bcnt = off; off += ALIGN((size_t)NUM_BLOCKS * 4);
  size_t p_done = off; off += ALIGN(4);
  size_t p_res  = off; off += ALIGN(8);

  char *mem;
  CHK(cudaMalloc(&mem, off));
  CHK(cudaMemset(mem, 0, off));

  State S;
  S.heap      = (u64  *)(mem + p_heap);
  S.heap_ptrs = (u32  *)(mem + p_hptr);
  S.conts     = (Cont *)(mem + p_cont);
  S.cont_bump = (u32  *)(mem + p_cbmp);
  S.tasks     = (Task *)(mem + p_task);
  S.flat      = (Task *)(mem + p_flat);
  S.flat_cnt  = (u32  *)(mem + p_fcnt);
  S.block_cnt = (u32  *)(mem + p_bcnt);
  S.done      = (u32  *)(mem + p_done);
  S.result    = (u64  *)(mem + p_res);

  // -- Per-slot heap slices --
  u32 slice = HEAP_SIZE / NUM_SLOTS;
  u32 *hps = (u32 *)malloc(NUM_SLOTS * sizeof(u32));
  for (int i = 0; i < NUM_SLOTS; i++) {
    hps[i] = (u32)i * slice;
  }
  CHK(cudaMemcpy(S.heap_ptrs, hps, NUM_SLOTS * sizeof(u32), cudaMemcpyHostToDevice));
  free(hps);

  // -- Run: gen → sort → checksum --
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

  CHK(cudaFree(mem));
  return 0;
}
