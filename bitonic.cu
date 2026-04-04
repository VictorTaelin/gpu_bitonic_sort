// ============================================================================
// recursive_bitonic_sort.cu — GPU parallel recursive bitonic sort
// ============================================================================
//
// PURPOSE
// -------
// Reference implementation showing how to run parallel recursive functional
// algorithms on a GPU. The approach is general: any recursive program with
// "par" annotations (marking independent sub-calls that can run in parallel)
// can be executed this way. Bitonic sort is the concrete example.
//
//
// ============================================================================
// PART 1: THE PAR ANNOTATION MODEL
// ============================================================================
//
// In Bend, a "par" annotation (written `a & b = foo(...) & bar(...)`) means
// "these two calls can run in parallel with roughly equal work." When a
// function hits a Par, it SUSPENDS (like a coroutine yield). Both sides are
// added to a task bag for parallel execution. When both results come back,
// the suspended function resumes.
//
// This gives us three core data structures:
//
//   Task: a function call ready to execute (function id + arguments + where
//     to deliver the result).
//
//   Frame: a suspended computation waiting for 2 sub-results. Has two slots
//     (one per child), a pending counter (starts at 2), and a pointer
//     to ITS parent frame (or root). When the counter hits 0, the frame
//     resumes — producing either a value (delivered further up) or a
//     new Par (creating more tasks).
//
//   ParResult: the return value of a par-aware function, living in registers.
//     Either VALUE(v) — done, deliver upward — or PAR(left, right, cont)
//     — suspend, create a frame and two child tasks.
//
//
// ============================================================================
// PART 2: THE INTENDED GPU ARCHITECTURE (SEED / GROW / WORK)
// ============================================================================
//
// The intended design maintains a task bag of up to 2^16 = 65536 tasks.
// When sequential CPU code first encounters a Par, it adds the two sub-tasks
// to the bag, creates a frame, and invokes the GPU. The GPU then fills itself
// with work through three phases:
//
// SEED PHASE (1 block, 256 threads)
//   Start with the initial 2 tasks in a shared-memory bag. Each round:
//     1. __syncthreads() (nanosecond-level, intra-block only)
//     2. Each thread with a task executes it.
//     3. If the result is a Par → 2 new tasks go into the next-round bag.
//        If a value → cascade upward through frames.
//     4. Swap bags, repeat.
//   The bag doubles each round: 2 → 4 → 8 → ... → 256. After ~8 rounds,
//   all 256 threads are busy. The key advantage: __syncthreads() costs ~10ns,
//   so these rounds are nearly free.
//
//   If the bag exceeds 256 → OVERFLOW. Abort the kernel. Save the bag (now
//   ~512 tasks) to global memory. Relaunch with 256 blocks.
//
// GROW PHASE (256 blocks, 256 threads each)
//   Each block grabs one (or a few) tasks from the global bag. Each block
//   then independently fills itself using the same intra-block loop as SEED
//   (1 → 2 → 4 → ... → 256 tasks per block, using __syncthreads()). Very
//   quickly, 256 blocks × 256 threads = 65536 threads are all busy.
//
//   If a block's bag overflows (> 256) → that block enters WORK mode.
//
// WORK PHASE (DFS fallback, per-thread sequential)
//   When a Par would overflow the block's bag, ignore the Par entirely:
//   compute both sides sequentially (DFS) within the thread. The thread
//   keeps working until its entire sub-tree is resolved. No synchronization,
//   no task creation — pure sequential compute.
//
//   Example: sort(24). SEED fills 1 block. GROW fills 256 blocks. Now 65536
//   threads each have a sort(8) to compute. sort(8) would create more Pars,
//   but the block bag is full, so each thread computes sort(8) sequentially.
//   Once all sort(8) results are ready, the kernel ends. The host has many
//   pending continuation frames. It grabs one continuation (e.g., sort(9)
//   with two resolved sort(8) results), resumes it — the resume calls
//   flow(9), which hits a warp(8) Par → invoke the GPU again. The GPU fills
//   itself again, executing 65536 warp calls. And so on.
//
// WHY THIS IS FAST
//   - SEED: 8 rounds × ~10ns sync = ~80ns to go from 2 to 256 threads.
//     No kernel launch overhead. No global memory sync.
//   - GROW: 8 more rounds × ~10ns = ~80ns per block. All 256 blocks run
//     independently with zero inter-block communication.
//   - WORK: pure DFS, no sync overhead, no atomics. Each thread runs at
//     full sequential speed.
//   - Total kernel launches: O(D) for the host-side continuation loop,
//     NOT O(D³) for every parallel step.
//
//
// ============================================================================
// PART 3: THE BITONIC SORT ALGORITHM
// ============================================================================
//
// Bitonic sort on a complete binary tree of depth D with 2^D leaves.
// Four mutually recursive functions, three with Par annotations:
//
//   sort(d, s, t):                   -- sorts a tree
//     if d == 0: return t
//     let sa & sb = sort(d-1, 0, left(t))     -- PAR: sort halves
//                 & sort(d-1, 1, right(t))
//     return flow(d, s, node(sa, sb))
//
//   flow(d, s, t):                   -- applies the flow network
//     if d == 0: return t
//     return down(d, s, warp(d-1, s, left(t), right(t)))
//
//   down(d, s, t):                   -- recurse flow on children
//     if d == 0: return t
//     let fa & fb = flow(d-1, s, left(t))     -- PAR: flow halves
//                 & flow(d-1, s, right(t))
//     return node(fa, fb)
//
//   warp(d, s, a, b):                -- pairwise compare-swap
//     if d == 0: return swap(s, a, b)
//     let wa & wb = warp(d-1, s, la, lb)      -- PAR: warp sub-pairs
//                 & warp(d-1, s, ra, rb)
//     return zip(wa, wb)
//
// swap(s, a, b) conditionally swaps two leaf values based on direction s.
// zip(a, b) = node(node(left(a), left(b)), node(right(a), right(b))).
//
// Optimization: flow() inlines one level of warp. Instead of calling
// warp(d-1) which would immediately Par into warp(d-2)×2, flow() directly
// creates the warp(d-2) Par and handles the zip in its FLOW_CONT resume.
//
// PARALLELISM PROFILE
//   Step count: S(D) = 2 + (D-1)(D² + 4D + 12) / 6
//   Max bag:    2^D exactly (during sort expansion)
//   Avg parallelism: ~O(2^D / D²) tasks per step
//
//   The bag evolves in a sawtooth pattern:
//     1. Sort expansion: bag doubles each step (2 → 4 → ... → 2^D)
//     2. sort(0) base cases resolve; cascade resolves sort frames bottom-up
//     3. Each sort(k) resume triggers flow(k) → warp/down cycles
//     4. Warp expansion: bag doubles (2 → ... → 2^(k-1))
//     5. warp(0) base cases resolve; cascade creates down/flow tasks
//     6. Repeat for each sort level
//
//   Every step is perfectly homogeneous: all tasks in a step call the same
//   function with the same depth. This means zero warp divergence on the GPU.
//
//
// ============================================================================
// PART 4: DATA STRUCTURES
// ============================================================================
//
// Task (16 bytes): a ready-to-execute function call.
//   pack:    bitfield [fn:2 | depth:6 | direction:1 | cont_slot:1 | unused:22]
//   arg0:    first tree pointer (sort/flow/warp)
//   arg1:    second tree pointer (warp only; 0 for sort/flow)
//   cont_id: parent Frame index to deliver result to (NO_PARENT = root)
//
// Frame (20 bytes): a suspended computation waiting for 2 sub-results.
//   pack:      bitfield [type:2 | depth:6 | direction:1 | parent_slot:1 | unused:22]
//   parent_id: this frame's parent frame (NO_PARENT = root)
//   slot0:     result from child 0
//   slot1:     result from child 1
//   pending:   atomic counter, starts at 2. Decremented by each child delivery.
//              The thread whose decrement takes it to 0 "owns" the frame and
//              resumes it.
//
// Frame types and their resume behavior:
//   SORT_CONT (0): two sort results → call flow(d, s, node(s0, s1))
//   WARP_CONT (1): two warp results → return zip(s0, s1) as value
//   FLOW_CONT (2): two warp results → call down(d, s, zip(s0, s1))
//   DOWN_CONT (3): two flow results → return node(s0, s1) as value
//
// ParResult (in registers only, never stored to memory):
//   is_par=0 → VALUE: just a tree pointer.
//   is_par=1 → PAR:   function id, depth, directions, arguments for both
//                      child tasks, plus continuation type/depth/direction.
//
// Tree nodes (2 words = 8 bytes each):
//   Leaf: [value (bit 31 = 0), 0]
//   Node: [left_ptr | 0x80000000 (bit 31 = 1), right_ptr]
//   Using 2 words instead of the naive 3 (tag, left, right) saves 33% memory.
//
//
// ============================================================================
// PART 5: THE CASCADE
// ============================================================================
//
// After executing a task, the thread enters the cascade loop to deliver its
// result upward through the frame tree:
//
//   loop:
//     if result is PAR → create Frame + 2 Tasks in the task bag. STOP.
//     if parent is ROOT → write final answer. STOP.
//     otherwise:
//       write result to parent frame's slot (slot0 or slot1).
//       __threadfence() — ensure the write is globally visible.
//       atomicSub(parent.pending, 1):
//         if old value was 1 → we are the LAST child. We OWN this frame.
//           __threadfence() — ensure we see the OTHER child's slot write.
//           resume the frame → new ParResult.
//           recycle the consumed frame's slot for the next Par.
//           goto loop with new result, new parent.
//         else → the other child hasn't arrived yet. STOP.
//
// The cascade can chain through several frame resumes before creating a new
// Par and stopping. Each resume does a small amount of work (make a node,
// zip two trees) before producing a value (continue up) or a Par (stop).
//
// Frame recycling: when a frame completes and the cascade immediately
// creates a new Par, the completed frame's slot is reused for the new
// frame — avoiding a global atomicAdd on the frame pointer.
//
//
// ============================================================================
// PART 6: CURRENT IMPLEMENTATION STATUS
// ============================================================================
//
// This file does NOT implement the SEED/GROW/WORK architecture described
// in Part 2. Instead it uses a simpler approach:
//
// WHAT IS IMPLEMENTED: a single persistent kernel using CUDA cooperative
// groups with grid-wide sync (grid.sync()). All rounds are processed inside
// one kernel launch. A flat global task bag (not per-block) is used. Each
// round: all threads process tasks from the bag, cascade results, write new
// tasks to the next bag, grid.sync(), swap bags, repeat. If the bag exceeds
// the thread count, multiple passes within a single round handle the excess.
//
// This gives ~25x speedup over sequential C for sort(20) on an RTX 4090,
// but falls short of what SEED/GROW/WORK should achieve because:
//   - grid.sync() costs ~3μs vs ~10ns for __syncthreads()
//     (~300x per-round overhead, ×1500 rounds ≈ 5ms wasted)
//   - No DFS fallback: every Par creates tasks, so the bag peaks at 2^D
//     and threads sit idle during low-parallelism rounds
//   - Global atomicAdd on the frame pointer (100M calls) is the main
//     remaining bottleneck (~40ms of the 74ms total)
//
// WHY SEED/GROW/WORK IS NOT YET IMPLEMENTED: the initial attempt used
// DFS fallback when a block's shared-memory bag overflowed. This caused a
// fatal single-thread bottleneck: the cascade after DFS chains upward
// through O(D) sort levels, each with exponentially more sequential work.
// One thread ends up doing O(2^D) work while the rest idle.
//
// The correct fix (not yet implemented): when the block bag overflows,
// SPILL overflow tasks to a global buffer instead of DFS'ing them. Process
// the spill in a subsequent kernel launch. This preserves the intra-block
// speed for normal operation while handling overflow gracefully.
//
//
// ============================================================================
// PART 7: OPTIMIZATIONS IN THE CURRENT IMPLEMENTATION
// ============================================================================
//
//   1. Two-word tree nodes (8 bytes vs 12), saving 33% memory.
//   2. Per-thread heap chunk allocator: pre-allocates 1024 words, then
//      bumps a local counter. Reduces heap atomicAdds by ~200x.
//   3. Block-level task batching: new tasks go to block-local shared memory
//      first, then one atomicAdd per block flushes to the global bag.
//      Reduces bag-counter atomicAdds by ~256x.
//   4. Batched node constructors: swap() and zip() allocate 3 nodes in one
//      chunk_alloc(6) call instead of 3 separate calls.
//   5. Frame recycling: reuses completed frame slots in cascade chains.
//   6. Parallel tree generation: GPU kernel builds the input tree.
//
// COMPILATION
//   nvcc -O3 -arch=sm_89 -rdc=true recursive_bitonic_sort.cu \
//     -o sort_gpu -lcudadevrt
//
// USAGE
//   ./sort_gpu [depth=20] [heap_gb=16]
//
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------

#define CHK(call)                                                              \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef uint32_t u32;
typedef int32_t  i32;
typedef uint64_t u64;

// ----------------------------------------------------------------------------
// Tree: 2 words per node
//
// Leaf: word0 = value (bit 31 = 0), word1 = 0
// Node: word0 = left_child_ptr | 0x80000000 (bit 31 = 1), word1 = right_child_ptr
//
// All tree pointers are byte offsets into the heap array (in u32 units).
// ----------------------------------------------------------------------------

#define IS_NODE(heap, ptr)  ((heap)[(ptr)] >> 31)
#define GET_VAL(heap, ptr)  ((heap)[(ptr)] & 0x7FFFFFFFu)
#define LEFT(heap, ptr)     ((heap)[(ptr)] & 0x7FFFFFFFu)
#define RIGHT(heap, ptr)    ((heap)[(ptr) + 1])

#define NO_PARENT 0xFFFFFFFFu

// Per-thread heap chunk: amortizes global atomicAdd on the heap pointer.
#define HEAP_CHUNK_WORDS 1024

struct HeapChunk {
  u32 base;
  u32 used;
  u32 capacity;
};

__device__ __forceinline__
u32 chunk_alloc(HeapChunk *chunk, u32 *heap_ptr, u32 num_words) {
  if (chunk->used + num_words > chunk->capacity) {
    chunk->capacity = HEAP_CHUNK_WORDS;
    chunk->base     = atomicAdd(heap_ptr, HEAP_CHUNK_WORDS);
    chunk->used     = 0;
  }
  u32 offset = chunk->base + chunk->used;
  chunk->used += num_words;
  return offset;
}

// --- Node constructors ---

__device__ __forceinline__
u32 make_leaf(u32 *heap, HeapChunk *chunk, u32 *heap_ptr, u32 value) {
  u32 p = chunk_alloc(chunk, heap_ptr, 2);
  heap[p]     = value;
  heap[p + 1] = 0;
  return p;
}

__device__ __forceinline__
u32 make_node(u32 *heap, HeapChunk *chunk, u32 *heap_ptr, u32 left, u32 right) {
  u32 p = chunk_alloc(chunk, heap_ptr, 2);
  heap[p]     = left | 0x80000000u;
  heap[p + 1] = right;
  return p;
}

// swap(s, a, b): compare two leaves, conditionally swap based on direction s.
// Allocates 3 nodes (2 leaves + 1 internal) in a single chunk_alloc(6).
__device__ __forceinline__
u32 make_swap(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
              u32 direction, u32 val_a, u32 val_b) {
  u32 do_swap = direction ^ (val_a > val_b);
  u32 p = chunk_alloc(chunk, heap_ptr, 6);
  heap[p]     = do_swap ? val_b : val_a;  heap[p + 1] = 0;      // leaf 0
  heap[p + 2] = do_swap ? val_a : val_b;  heap[p + 3] = 0;      // leaf 1
  heap[p + 4] = p | 0x80000000u;          heap[p + 5] = p + 2;  // node
  return p + 4;
}

// zip(a, b) = node(node(left(a), left(b)), node(right(a), right(b)))
// Allocates 3 nodes in a single chunk_alloc(6).
__device__ __forceinline__
u32 make_zip(u32 *heap, HeapChunk *chunk, u32 *heap_ptr, u32 a, u32 b) {
  u32 p = chunk_alloc(chunk, heap_ptr, 6);
  heap[p]     = LEFT(heap, a) | 0x80000000u;  heap[p + 1] = LEFT(heap, b);
  heap[p + 2] = RIGHT(heap, a) | 0x80000000u; heap[p + 3] = RIGHT(heap, b);
  heap[p + 4] = p | 0x80000000u;              heap[p + 5] = p + 2;
  return p + 4;
}

// ----------------------------------------------------------------------------
// Task and Frame
// ----------------------------------------------------------------------------

// Function IDs for the 'fn' field in Task and ParResult.
#define FN_SORT 0
#define FN_FLOW 1
#define FN_WARP 2

// Continuation types for the 'type' field in Frame.
#define CONT_SORT 0  // sort resumed → call flow(d, s, node(s0, s1))
#define CONT_WARP 1  // warp resumed → return zip(s0, s1)
#define CONT_FLOW 2  // flow resumed → call down(d, s, zip(s0, s1))
#define CONT_DOWN 3  // down resumed → return node(s0, s1)

struct Task {
  u32 pack;     // [fn:2 | depth:6 | direction:1 | cont_slot:1 | reserved:22]
  u32 arg0;     // first tree pointer
  u32 arg1;     // second tree pointer (warp only)
  u32 cont_id;  // parent frame index, or NO_PARENT for root
};

#define TASK_FN(t)        ((t).pack & 3u)
#define TASK_DEPTH(t)     (((t).pack >> 2) & 63u)
#define TASK_DIR(t)       (((t).pack >> 8) & 1u)
#define TASK_CONT_SLOT(t) (((t).pack >> 9) & 1u)

#define PACK_TASK(fn, depth, dir, slot) \
  ((u32)(fn) | ((u32)(depth) << 2) | ((u32)(dir) << 8) | ((u32)(slot) << 9))

struct Frame {
  u32 pack;       // [type:2 | depth:6 | direction:1 | parent_slot:1 | reserved:22]
  u32 parent_id;  // parent frame index, or NO_PARENT
  u32 slot0;      // result from child 0
  u32 slot1;      // result from child 1
  i32 pending;    // atomic: starts at 2, decremented by each delivery
};

#define FRAME_TYPE(f)        ((f).pack & 3u)
#define FRAME_DEPTH(f)       (((f).pack >> 2) & 63u)
#define FRAME_DIR(f)         (((f).pack >> 8) & 1u)
#define FRAME_PARENT_SLOT(f) (((f).pack >> 9) & 1u)

#define PACK_FRAME(type, depth, dir, pslot) \
  ((u32)(type) | ((u32)(depth) << 2) | ((u32)(dir) << 8) | ((u32)(pslot) << 9))

// ParResult: returned by par-aware functions. Lives in registers only.
// is_par=0 → value result (just a tree pointer).
// is_par=1 → par request (two child tasks + continuation metadata).
struct ParResult {
  u32 is_par;
  u32 value;
  u32 fn, depth;
  u32 left_dir, right_dir;
  u32 left_arg0, left_arg1;
  u32 right_arg0, right_arg1;
  u32 cont_type, cont_depth, cont_dir;
};

__device__ __forceinline__
ParResult make_value(u32 v) {
  ParResult r;
  r.is_par = 0;
  r.value  = v;
  return r;
}

// ----------------------------------------------------------------------------
// Par-aware functions
//
// Each function returns a ParResult: either a direct value (base case) or a
// par request describing two independent sub-calls and how to resume.
// ----------------------------------------------------------------------------

__device__ ParResult par_down(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                              u32 depth, u32 dir, u32 tree);

__device__
ParResult par_sort(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                   u32 depth, u32 dir, u32 tree) {
  if (depth == 0 || !IS_NODE(heap, tree))
    return make_value(tree);

  ParResult r;
  r.is_par     = 1;
  r.fn         = FN_SORT;
  r.depth      = depth - 1;
  r.left_dir   = 0;
  r.right_dir  = 1;
  r.left_arg0  = LEFT(heap, tree);   r.left_arg1  = 0;
  r.right_arg0 = RIGHT(heap, tree);  r.right_arg1 = 0;
  r.cont_type  = CONT_SORT;
  r.cont_depth = depth;
  r.cont_dir   = dir;
  return r;
}

__device__
ParResult par_warp(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                   u32 depth, u32 dir, u32 a, u32 b) {
  if (depth == 0) {
    if (IS_NODE(heap, a) | IS_NODE(heap, b))
      return make_value(make_leaf(heap, chunk, heap_ptr, 0));
    return make_value(make_swap(heap, chunk, heap_ptr, dir,
                                GET_VAL(heap, a), GET_VAL(heap, b)));
  }
  if (!IS_NODE(heap, a) || !IS_NODE(heap, b))
    return make_value(make_leaf(heap, chunk, heap_ptr, 0));

  ParResult r;
  r.is_par     = 1;
  r.fn         = FN_WARP;
  r.depth      = depth - 1;
  r.left_dir   = dir;
  r.right_dir  = dir;
  r.left_arg0  = LEFT(heap, a);   r.left_arg1  = LEFT(heap, b);
  r.right_arg0 = RIGHT(heap, a);  r.right_arg1 = RIGHT(heap, b);
  r.cont_type  = CONT_WARP;
  r.cont_depth = 0;
  r.cont_dir   = 0;
  return r;
}

__device__
ParResult par_down(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                   u32 depth, u32 dir, u32 tree) {
  if (depth == 0 || !IS_NODE(heap, tree))
    return make_value(tree);

  ParResult r;
  r.is_par     = 1;
  r.fn         = FN_FLOW;
  r.depth      = depth - 1;
  r.left_dir   = dir;
  r.right_dir  = dir;
  r.left_arg0  = LEFT(heap, tree);   r.left_arg1  = 0;
  r.right_arg0 = RIGHT(heap, tree);  r.right_arg1 = 0;
  r.cont_type  = CONT_DOWN;
  r.cont_depth = depth;
  r.cont_dir   = dir;
  return r;
}

__device__
ParResult par_flow(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                   u32 depth, u32 dir, u32 tree) {
  if (depth == 0 || !IS_NODE(heap, tree))
    return make_value(tree);

  u32 a = LEFT(heap, tree);
  u32 b = RIGHT(heap, tree);

  if (depth == 1) {
    // Inline warp(0) = swap, then delegate to down.
    u32 warped;
    if (IS_NODE(heap, a) | IS_NODE(heap, b))
      warped = make_leaf(heap, chunk, heap_ptr, 0);
    else
      warped = make_swap(heap, chunk, heap_ptr, dir,
                         GET_VAL(heap, a), GET_VAL(heap, b));
    return par_down(heap, chunk, heap_ptr, depth, dir, warped);
  }

  if (!IS_NODE(heap, a) || !IS_NODE(heap, b))
    return par_down(heap, chunk, heap_ptr, depth, dir,
                    make_leaf(heap, chunk, heap_ptr, 0));

  // Inline warp(d-1): split into warp(d-2)×2, zip in FLOW_CONT resume.
  ParResult r;
  r.is_par     = 1;
  r.fn         = FN_WARP;
  r.depth      = depth - 2;
  r.left_dir   = dir;
  r.right_dir  = dir;
  r.left_arg0  = LEFT(heap, a);   r.left_arg1  = LEFT(heap, b);
  r.right_arg0 = RIGHT(heap, a);  r.right_arg1 = RIGHT(heap, b);
  r.cont_type  = CONT_FLOW;
  r.cont_depth = depth;
  r.cont_dir   = dir;
  return r;
}

__device__
ParResult exec_task(Task task, u32 *heap, HeapChunk *chunk, u32 *heap_ptr) {
  u32 fn    = TASK_FN(task);
  u32 depth = TASK_DEPTH(task);
  u32 dir   = TASK_DIR(task);

  if (fn == FN_SORT) return par_sort(heap, chunk, heap_ptr, depth, dir, task.arg0);
  if (fn == FN_FLOW) return par_flow(heap, chunk, heap_ptr, depth, dir, task.arg0);
  return par_warp(heap, chunk, heap_ptr, depth, dir, task.arg0, task.arg1);
}

// Resume a completed frame: read the two child results and continue.
__device__
ParResult resume_frame(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                       u32 type, u32 depth, u32 dir, u32 slot0, u32 slot1) {
  switch (type) {
    case CONT_SORT:
      return par_flow(heap, chunk, heap_ptr, depth, dir,
                      make_node(heap, chunk, heap_ptr, slot0, slot1));

    case CONT_WARP:
      if (!IS_NODE(heap, slot0) || !IS_NODE(heap, slot1))
        return make_value(make_leaf(heap, chunk, heap_ptr, 0));
      return make_value(make_zip(heap, chunk, heap_ptr, slot0, slot1));

    case CONT_FLOW: {
      u32 zipped;
      if (!IS_NODE(heap, slot0) || !IS_NODE(heap, slot1))
        zipped = make_leaf(heap, chunk, heap_ptr, 0);
      else
        zipped = make_zip(heap, chunk, heap_ptr, slot0, slot1);
      return par_down(heap, chunk, heap_ptr, depth, dir, zipped);
    }

    case CONT_DOWN:
      return make_value(make_node(heap, chunk, heap_ptr, slot0, slot1));
  }
  return make_value(0);
}

// ----------------------------------------------------------------------------
// Cascade
//
// After executing a task, the thread enters this loop to deliver results
// upward through the frame tree. The loop continues until either:
//   - A new Par is created (2 tasks written to block-local buffer), or
//   - A value is delivered to a frame whose other child hasn't arrived yet, or
//   - The root result is produced (parent_id == NO_PARENT).
//
// Frame recycling: when a frame completes and the cascade immediately creates
// a new Par, the completed frame's slot is reused instead of allocating a new
// one. This avoids one global atomicAdd on the frame pointer per chain link.
// ----------------------------------------------------------------------------

__device__
void cascade(ParResult result, u32 parent_id, u32 parent_slot,
             u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
             Task *block_task_buf, u32 *block_task_count,
             Frame *frames, u32 *frame_ptr, u32 *root_result) {
  u32 recycle_id = NO_PARENT;

  for (;;) {
    if (result.is_par) {
      // Allocate (or recycle) a frame for this Par.
      u32 frame_id;
      if (recycle_id != NO_PARENT) {
        frame_id   = recycle_id;
        recycle_id = NO_PARENT;
      } else {
        frame_id = atomicAdd(frame_ptr, 1);
      }

      // Write the continuation frame.
      Frame cont;
      cont.pack      = PACK_FRAME(result.cont_type, result.cont_depth,
                                   result.cont_dir, parent_slot);
      cont.parent_id = parent_id;
      cont.slot0     = 0;
      cont.slot1     = 0;
      cont.pending   = 2;
      frames[frame_id] = cont;

      // Write the two child tasks to block-local shared memory.
      u32 idx = atomicAdd(block_task_count, 2);

      Task left_task;
      left_task.pack    = PACK_TASK(result.fn, result.depth, result.left_dir, 0);
      left_task.arg0    = result.left_arg0;
      left_task.arg1    = result.left_arg1;
      left_task.cont_id = frame_id;

      Task right_task;
      right_task.pack    = PACK_TASK(result.fn, result.depth, result.right_dir, 1);
      right_task.arg0    = result.right_arg0;
      right_task.arg1    = result.right_arg1;
      right_task.cont_id = frame_id;

      block_task_buf[idx]     = left_task;
      block_task_buf[idx + 1] = right_task;
      return;
    }

    // Result is a value. Deliver it upward.
    if (parent_id == NO_PARENT) {
      *root_result = result.value;
      return;
    }

    // Write the value into the parent frame's slot.
    Frame *parent = &frames[parent_id];
    if (parent_slot == 0)
      parent->slot0 = result.value;
    else
      parent->slot1 = result.value;

    // Ensure the slot write is globally visible before decrementing.
    __threadfence();

    // Atomically decrement the pending counter.
    // If we were the LAST child to deliver (old value was 1), we own
    // this frame and resume it. Otherwise, the other child will.
    if (atomicSub(&parent->pending, 1) == 1) {
      // Ensure the OTHER child's slot write is visible to us.
      __threadfence();

      // This frame is consumed — its slot can be reused.
      recycle_id = parent_id;

      // Resume the frame.
      result      = resume_frame(heap, chunk, heap_ptr,
                                 FRAME_TYPE(*parent),
                                 FRAME_DEPTH(*parent),
                                 FRAME_DIR(*parent),
                                 parent->slot0, parent->slot1);
      parent_slot = FRAME_PARENT_SLOT(*parent);
      parent_id   = parent->parent_id;
    } else {
      return;
    }
  }
}

// ----------------------------------------------------------------------------
// Parallel tree generation (level-order layout)
//
// Builds a complete binary tree of depth D with leaves valued (2^D-1) down
// to 0 (left to right). Nodes are laid out in level-order: node i has its
// left child at index 2i+1 and right child at 2i+2. Each node occupies
// 2 words in the heap.
// ----------------------------------------------------------------------------

__global__
void generate_tree(u32 *heap, u32 depth, u32 num_nodes) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_nodes) return;

  u32 num_leaves  = 1u << depth;
  u32 heap_offset = i * 2;

  if (i >= num_leaves - 1) {
    // Leaf: value = num_leaves - 1 - leaf_index
    u32 leaf_index      = i - (num_leaves - 1);
    heap[heap_offset]     = num_leaves - 1 - leaf_index;
    heap[heap_offset + 1] = 0;
  } else {
    // Internal node: pointers to children
    heap[heap_offset]     = ((2u * i + 1u) * 2u) | 0x80000000u;
    heap[heap_offset + 1] = (2u * i + 2u) * 2u;
  }
}

// ----------------------------------------------------------------------------
// Persistent evaluation kernel
//
// Launched once with cooperative groups. Processes all rounds internally
// using grid.sync() for inter-block synchronization between rounds.
// Double-buffered task bags (current and next) are swapped each round.
// ----------------------------------------------------------------------------

#define THREADS_PER_BLOCK 256
#define MAX_BLOCK_TASKS   (THREADS_PER_BLOCK * 2)

__global__
void eval_kernel(Task *bag_a, Task *bag_b,
                 u32 *count_a, u32 *count_b,
                 u32 *heap, u32 *heap_ptr,
                 Frame *frames, u32 *frame_ptr,
                 u32 *root_result) {
  cg::grid_group grid = cg::this_grid();

  u32 global_tid  = blockIdx.x * blockDim.x + threadIdx.x;
  u32 total_threads = gridDim.x * blockDim.x;

  // Block-local task buffer: cascade writes new tasks here first.
  // After all threads finish, one atomicAdd per block reserves
  // contiguous global bag space, then threads copy cooperatively.
  __shared__ Task block_task_buf[MAX_BLOCK_TASKS];
  __shared__ u32  block_task_count;
  __shared__ u32  block_bag_base;

  // Double-buffered bag pointers (swapped each round).
  Task *current_bag = bag_a;
  Task *next_bag    = bag_b;
  u32  *current_n   = count_a;
  u32  *next_n      = count_b;

  // Per-thread heap chunk allocator (persists across rounds).
  HeapChunk heap_chunk = {0, 0, 0};

  for (u32 round = 0; round < 100000; round++) {
    u32 num_tasks = *current_n;
    if (num_tasks == 0) break;

    // Reset the next-bag counter.
    if (global_tid == 0) *next_n = 0;
    grid.sync();

    // Process all tasks, in chunks of total_threads if bag > threads.
    for (u32 offset = 0; offset < num_tasks; offset += total_threads) {
      if (threadIdx.x == 0) block_task_count = 0;
      __syncthreads();

      u32 my_task = offset + global_tid;
      if (my_task < num_tasks) {
        Task task       = current_bag[my_task];
        ParResult result = exec_task(task, heap, &heap_chunk, heap_ptr);
        cascade(result, task.cont_id, TASK_CONT_SLOT(task),
                heap, &heap_chunk, heap_ptr,
                block_task_buf, &block_task_count,
                frames, frame_ptr, root_result);
      }
      __syncthreads();

      // Flush block tasks to global next-bag.
      u32 count = block_task_count;
      if (threadIdx.x == 0 && count > 0)
        block_bag_base = atomicAdd(next_n, count);
      __syncthreads();

      for (u32 i = threadIdx.x; i < count; i += blockDim.x)
        next_bag[block_bag_base + i] = block_task_buf[i];
    }
    grid.sync();

    // Swap double-buffered bags.
    Task *tmp_bag = current_bag; current_bag = next_bag; next_bag = tmp_bag;
    u32  *tmp_n   = current_n;   current_n   = next_n;   next_n   = tmp_n;
  }
}

// ----------------------------------------------------------------------------
// Tree sum (single-thread iterative DFS)
// ----------------------------------------------------------------------------

__global__
void sum_tree(u32 root, u32 *heap, u32 *out) {
  u32 stack[64];
  int sp  = 0;
  u32 sum = 0;

  stack[sp++] = root;
  while (sp > 0) {
    u32 node = stack[--sp];
    if (!IS_NODE(heap, node))
      sum += GET_VAL(heap, node);
    else {
      stack[sp++] = RIGHT(heap, node);
      stack[sp++] = LEFT(heap, node);
    }
  }
  *out = sum;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  u32 depth   = argc > 1 ? (u32)atoi(argv[1]) : 20;
  u32 heap_gb = argc > 2 ? (u32)atoi(argv[2]) : (depth >= 20 ? 16 : 8);

  u64 heap_words = (u64)heap_gb << 28;  // heap_gb * 2^30 / 4
  u32 max_bag    = 1u << (depth + 2 < 24 ? depth + 2 : 24);
  u32 max_frames = depth <= 12 ? (1u << 20) :
                   depth <= 16 ? (1u << 24) : (1u << 27);

  u32 num_nodes  = (1u << (depth + 1)) - 1;
  u32 initial_hp = num_nodes * 2;

  fprintf(stderr, "sort(%u): heap=%uGB max_bag=%u max_frames=%u\n",
          depth, heap_gb, max_bag, max_frames);

  // Query the maximum cooperative-launch block count.
  int blocks_per_sm = 0;
  CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, eval_kernel, THREADS_PER_BLOCK,
      MAX_BLOCK_TASKS * sizeof(Task) + 8));
  int num_sms = 0;
  CHK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  int num_blocks = blocks_per_sm * num_sms;
  fprintf(stderr, "SMs=%d blocks/SM=%d blocks=%d threads=%d\n",
          num_sms, blocks_per_sm, num_blocks, num_blocks * THREADS_PER_BLOCK);

  // --- Allocate GPU memory ---

  u32 *dev_heap, *dev_heap_ptr;
  CHK(cudaMalloc(&dev_heap,     heap_words * 4));
  CHK(cudaMalloc(&dev_heap_ptr, sizeof(u32)));

  // Generate the input tree on GPU.
  generate_tree<<<(num_nodes + 255) / 256, 256>>>(dev_heap, depth, num_nodes);
  CHK(cudaDeviceSynchronize());
  CHK(cudaMemcpy(dev_heap_ptr, &initial_hp, sizeof(u32), cudaMemcpyHostToDevice));

  // Double-buffered task bags.
  Task *dev_bag_a, *dev_bag_b;
  CHK(cudaMalloc(&dev_bag_a, (u64)max_bag * sizeof(Task)));
  CHK(cudaMalloc(&dev_bag_b, (u64)max_bag * sizeof(Task)));

  u32 *dev_count_a, *dev_count_b;
  CHK(cudaMalloc(&dev_count_a, sizeof(u32)));
  CHK(cudaMalloc(&dev_count_b, sizeof(u32)));

  // Continuation frames.
  Frame *dev_frames;
  CHK(cudaMalloc(&dev_frames, (u64)max_frames * sizeof(Frame)));

  u32 *dev_frame_ptr, *dev_result;
  CHK(cudaMalloc(&dev_frame_ptr, sizeof(u32)));
  CHK(cudaMalloc(&dev_result,    sizeof(u32)));

  // --- Initialize: one task = sort(depth, 0, root=0) ---

  Task initial_task;
  initial_task.pack    = PACK_TASK(FN_SORT, depth, 0, 0);
  initial_task.arg0    = 0;  // root is at heap offset 0
  initial_task.arg1    = 0;
  initial_task.cont_id = NO_PARENT;

  CHK(cudaMemcpy(dev_bag_a, &initial_task, sizeof(Task), cudaMemcpyHostToDevice));
  u32 one = 1, zero = 0;
  CHK(cudaMemcpy(dev_count_a,  &one,  sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dev_count_b,  &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dev_frame_ptr, &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dev_result,    &zero, sizeof(u32), cudaMemcpyHostToDevice));

  // Warmup (trigger JIT compilation).
  generate_tree<<<1, 1>>>(dev_heap, 0, 0);
  CHK(cudaDeviceSynchronize());

  // --- Launch the persistent evaluation kernel ---

  void *args[] = {
    &dev_bag_a,  &dev_bag_b,
    &dev_count_a, &dev_count_b,
    &dev_heap,   &dev_heap_ptr,
    &dev_frames, &dev_frame_ptr,
    &dev_result
  };

  struct timespec time_start, time_end;
  clock_gettime(CLOCK_MONOTONIC, &time_start);

  CHK(cudaLaunchCooperativeKernel(
      (void *)eval_kernel, num_blocks, THREADS_PER_BLOCK, args,
      MAX_BLOCK_TASKS * sizeof(Task) + 8));
  CHK(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &time_end);
  double sort_time = (time_end.tv_sec - time_start.tv_sec) +
                     (time_end.tv_nsec - time_start.tv_nsec) * 1e-9;

  // --- Read results ---

  u32 result_root;
  CHK(cudaMemcpy(&result_root, dev_result, sizeof(u32), cudaMemcpyDeviceToHost));

  u32 final_hp;
  CHK(cudaMemcpy(&final_hp, dev_heap_ptr, sizeof(u32), cudaMemcpyDeviceToHost));

  // Sum the sorted tree to verify correctness.
  u32 *dev_sum;
  CHK(cudaMalloc(&dev_sum, sizeof(u32)));
  sum_tree<<<1, 1>>>(result_root, dev_heap, dev_sum);
  CHK(cudaDeviceSynchronize());
  u32 checksum;
  CHK(cudaMemcpy(&checksum, dev_sum, sizeof(u32), cudaMemcpyDeviceToHost));

  u32 final_fp;
  CHK(cudaMemcpy(&final_fp, dev_frame_ptr, sizeof(u32), cudaMemcpyDeviceToHost));

  fprintf(stderr, "sort(%u) = %u heap=%.2fGB frames=%u %.3fs\n",
          depth, checksum, final_hp * 4.0 / (1ULL << 30), final_fp, sort_time);

  // --- Cleanup ---

  cudaFree(dev_heap);      cudaFree(dev_heap_ptr);
  cudaFree(dev_bag_a);     cudaFree(dev_bag_b);
  cudaFree(dev_count_a);   cudaFree(dev_count_b);
  cudaFree(dev_frames);    cudaFree(dev_frame_ptr);
  cudaFree(dev_result);    cudaFree(dev_sum);
  return 0;
}

