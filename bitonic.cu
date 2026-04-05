// ==========================================================================
// bitonic.cu — GPU Parallel Evaluator for Recursive Functional Programs
// ==========================================================================
//
// This file implements a general-purpose parallel evaluator for pure
// functional programs on the GPU, demonstrated with a bitonic sort algorithm.
// The key idea: we take recursive functions written in a simple functional
// style (like the C reference in bitonic.c) and "manually compile" them into
// a task-based runtime that exploits GPU parallelism automatically.
//
// THE ALGORITHM (from bitonic.c, cannot be changed):
//
//   sort(d, s, t):
//     if d == 0 or t is a leaf: return t
//     left  = sort(d-1, 0, left_child(t))    // PARALLEL
//     right = sort(d-1, 1, right_child(t))    // PARALLEL
//     return flow(d, s, Node(left, right))
//
//   flow(d, s, t):
//     if d == 0 or t is a leaf: return t
//     warped = warp(d-1, s, left_child(t), right_child(t))
//     return down(d, s, warped)
//
//   down(d, s, t):
//     if d == 0 or t is a leaf: return t
//     left  = flow(d-1, s, left_child(t))     // PARALLEL
//     right = flow(d-1, s, right_child(t))     // PARALLEL
//     return Node(left, right)
//
//   warp(d, s, a, b):
//     if d == 0: compare-and-swap leaves a, b based on s
//     left  = warp(d-1, s, left(a), left(b))   // PARALLEL
//     right = warp(d-1, s, right(a), right(b))  // PARALLEL
//     return Node(Node(left_l, right_l), Node(left_r, right_r))
//
// Lines marked PARALLEL are where two independent recursive calls can run
// concurrently. This evaluator exploits that parallelism on the GPU.
//
// ==========================================================================
// HOW IT WORKS: THE TASK/CONTINUATION RUNTIME
// ==========================================================================
//
// Each recursive function is "compiled" into a task function that, instead of
// recursing, returns one of three results:
//
//   VALUE(v)       — the function completed; return value v
//   SPLIT(t0, t1)  — the function needs two parallel sub-calls; create tasks
//                     t0 and t1, plus a continuation that fires when both done
//   CALL(t0)       — the function needs one sub-call; create task t0 and a
//                     continuation that fires when it returns
//
// A Continuation is a suspended function waiting for its children to finish.
// It stores: which function to resume (fn), how many children are pending
// (pending count), where to send its own result (return address), and slots
// for the children's results. When a child produces a VALUE, it writes into
// the continuation's slot and decrements the pending count. The thread that
// decrements it to zero "fires" the continuation, executing the resume
// function, which may itself return VALUE/SPLIT/CALL.
//
// ==========================================================================
// HOW IT WORKS: THE SEED/GROW/WORK SCHEDULING
// ==========================================================================
//
// The runtime maintains a "task matrix" of NUM_BLOCKS × BLOCK_SIZE slots.
// Each column belongs to one GPU block; each row is one thread's task slot.
// Execution proceeds in three phases that repeat:
//
//   SEED  (1 block, NUM_BLOCKS threads)
//     Starting from a small number of tasks (≤ NUM_BLOCKS), iteratively
//     execute them. Tasks that SPLIT produce 2 children, doubling the count
//     each round. After log2(NUM_BLOCKS) rounds, we have NUM_BLOCKS tasks —
//     one per GPU block. This fills one "row" of the matrix.
//
//   GROW  (NUM_BLOCKS blocks, BLOCK_SIZE threads each)
//     Each block takes its task(s) from the flat buffer and iteratively
//     splits them in shared memory, doubling each round. After log2(BLOCK_SIZE)
//     rounds, each block has BLOCK_SIZE tasks — filling its full column.
//     Now the entire matrix is populated.
//
//   WORK  (NUM_BLOCKS blocks, BLOCK_SIZE threads each)
//     Every thread executes its task *sequentially* (calling the recursive
//     d_sort, d_flow, etc. functions directly). This is where computation
//     happens. When a thread finishes and produces a VALUE, it resolves
//     continuations: writing the value into the parent continuation's slot,
//     and if that continuation is now complete, executing it. The result
//     may be a new task (SPLIT/CALL) which gets added to a flat output buffer
//     for the next round.
//
// After WORK, the flat buffer contains the new tasks. The cycle repeats:
//   - If count ≤ NUM_BLOCKS → SEED first, then GROW, then WORK
//   - If count > NUM_BLOCKS → skip SEED, just GROW then WORK
//   - If count == 0 and done flag is set → computation is complete
//
// This runs as a single cooperative kernel with grid-wide synchronization
// between phases, eliminating all host↔device synchronization overhead.
//
// The effective sequential cutoff emerges naturally from the matrix size:
// with NUM_BLOCKS=128 and BLOCK_SIZE=256, SEED uses 7 doublings and GROW
// uses 8 doublings, so sort(20) bottoms out at sort(5) — each thread
// sequentially sorts a 32-element subtree.
//
// ==========================================================================
// MEMORY MANAGEMENT
// ==========================================================================
//
// Trees are stored in a shared heap (a flat array of u64 values). A Node
// occupies two consecutive heap slots [left_child, right_child] and is
// represented as a tagged u64 pointing to its first slot.
//
// The heap is divided into equal-sized slices, one per thread slot
// (NUM_BLOCKS × BLOCK_SIZE total). Each thread bumps a local pointer within
// its slice — zero contention, zero atomics, provably the fastest possible
// allocation scheme for a fixed thread count.
//
// Continuations use a separate buffer with a chunked bump allocator: each
// thread grabs small chunks (CONT_CHUNK entries) from a global atomic
// counter, then allocates locally within its chunk.
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

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

#define NUM_BLOCKS  128              // GPU blocks (1 per SM on RTX 4090)
#define BLOCK_SIZE  256              // Threads per block
#define TOTAL_SLOTS (NUM_BLOCKS * BLOCK_SIZE)  // 32768 total worker slots

#define HEAP_SIZE   (2u << 30)       // Heap capacity in u64s (16 GB)
#define CONT_CAP    (1u << 26)       // Max continuations (64M)
#define CONT_CHUNK  2                // Cont allocation chunk size (small = less waste)

// #define DEBUG_MATRIX              // Uncomment to print task matrix each step

// --------------------------------------------------------------------------
// Tree Encoding
// --------------------------------------------------------------------------
//
// A Tree is a u64. Bit 63 is the tag:
//   - Leaf: bit 63 = 0, bits [31:0] = value
//   - Node: bit 63 = 1, bits [30:0] = heap index (left at heap[idx], right at heap[idx+1])

#define NODE_TAG (1ULL << 63)

__host__ __device__ inline u64  make_leaf(u32 value) { return (u64)value; }
__host__ __device__ inline u64  make_node(u32 index) { return NODE_TAG | (u64)index; }
__host__ __device__ inline bool is_node(u64 tree)    { return (tree & NODE_TAG) != 0; }
__host__ __device__ inline bool is_leaf(u64 tree)    { return !is_node(tree); }
__host__ __device__ inline u32  leaf_value(u64 tree) { return (u32)(tree & 0xFFFFFFFFu); }
__host__ __device__ inline u32  node_index(u64 tree) { return (u32)(tree & 0x7FFFFFFFu); }

// --------------------------------------------------------------------------
// Function IDs — each recursive function and its continuation
// --------------------------------------------------------------------------

enum FnId {
  FN_SORT       = 0,   // sort(depth, side, tree)
  FN_FLOW       = 1,   // flow(depth, side, tree)
  FN_SWAP       = 2,   // warp/swap(side, tree, depth)
  FN_SORT_CONT  = 3,   // sort continuation: got both sorted halves → call flow
  FN_FLOW_AFTER = 4,   // flow continuation: got swap result → split into two flows
  FN_FLOW_JOIN  = 5,   // flow join: got both flow results → make Node
  FN_SWAP_JOIN  = 6,   // swap join: got both swap results → reassemble
  FN_GEN        = 7,   // gen(depth, x) — tree generation
  FN_GEN_JOIN   = 8,   // gen join: got both subtrees → make Node
  FN_CSUM       = 9,   // checksum(tree, depth)
  FN_CSUM_JOIN  = 10,  // checksum join: combine left and right checksums
};

// --------------------------------------------------------------------------
// Result Tags
// --------------------------------------------------------------------------

enum ResultTag {
  RESULT_VALUE = 0,   // Computation complete, here is the value
  RESULT_SPLIT = 1,   // Need two parallel sub-computations
  RESULT_CALL  = 2,   // Need one sub-computation (with a continuation)
};

// Special return address meaning "this is the root computation"
#define ROOT_RETURN 0xFFFFFFFFu

// --------------------------------------------------------------------------
// Data Structures
// --------------------------------------------------------------------------

// A Task represents a function call to be executed.
//   fn:  which function to call (FnId)
//   ret: where to write the result (encoded continuation index + slot)
//   args: up to 3 arguments (interpretation depends on fn)
struct Task {
  u32 fn;
  u32 ret;
  u64 args[3];
};

// A Continuation is a suspended computation waiting for child results.
//   pending: number of children still running (atomically decremented)
//   ret:     where to write *this* continuation's result
//   fn:      which function to resume when all children are done
//   a0-a1:   small arguments saved from the original call (e.g., depth, side)
//   slots:   two slots for child results (filled as children complete)
struct Continuation {
  u32 pending;
  u32 ret;
  u16 fn;
  u16 a0;
  u16 a1;
  u16 _padding;
  u64 slots[2];
};

// The result of executing a task function.
struct Result {
  u32  tag;         // RESULT_VALUE, RESULT_SPLIT, or RESULT_CALL
  u64  value;       // If VALUE: the result
  Task child0;      // If SPLIT or CALL: first (or only) child task
  Task child1;      // If SPLIT: second child task
};

// All GPU-side state, passed to kernels.
struct GPUState {
  u64          *heap;          // Tree node storage [HEAP_SIZE]
  u32          *heap_pointers; // Per-slot bump pointers [TOTAL_SLOTS]
  Continuation *continuations; // Continuation buffer [CONT_CAP]
  u32          *cont_bump;     // Global continuation allocator counter
  Task         *task_matrix;   // Column-format task storage [TOTAL_SLOTS]
  Task         *flat_buffer;   // Flat task buffer for inter-phase transfer [TOTAL_SLOTS]
  u32          *flat_count;    // Number of tasks in flat buffer
  u32          *block_counts;  // Per-block task count after GROW [NUM_BLOCKS]
  u32          *done_flag;     // Set to 1 when root computation completes
  u64          *root_result;   // The final result of the root computation
};

// --------------------------------------------------------------------------
// CUDA Error Checking
// --------------------------------------------------------------------------

#define CUDA_CHECK(call) do {                                              \
  cudaError_t err = (call);                                                \
  if (err != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
            __FILE__, __LINE__, cudaGetErrorString(err));                   \
    exit(1);                                                               \
  }                                                                        \
} while (0)

// --------------------------------------------------------------------------
// Device Helpers
// --------------------------------------------------------------------------

// Pack two u32 values into a single u64 argument.
// Used to pass (depth, side) or (depth, x) as one argument.
__host__ __device__ inline u64 pack(u32 low, u32 high) {
  return (u64)low | ((u64)high << 32);
}

__host__ __device__ inline u32 unpack_low(u64 packed) {
  return (u32)packed;
}

__host__ __device__ inline u32 unpack_high(u64 packed) {
  return (u32)(packed >> 32);
}

// Allocate a 2-slot tree node on the heap. Returns the heap index.
__device__ inline u32 alloc_node(u64 left, u64 right, u64 *heap, u32 &heap_ptr) {
  u32 index = heap_ptr;
  heap_ptr += 2;
  heap[index]     = left;
  heap[index + 1] = right;
  return index;
}

// Encode a continuation index + slot number into a return address.
// Slot is 0 or 1 (which of the continuation's two slots to fill).
__device__ inline u32 encode_return(u32 cont_index, u32 slot) {
  return (cont_index << 1) | slot;
}

// Construct a Task.
__host__ __device__ inline Task make_task(u32 fn, u32 ret, u64 arg0, u64 arg1 = 0, u64 arg2 = 0) {
  Task task;
  task.fn     = fn;
  task.ret    = ret;
  task.args[0] = arg0;
  task.args[1] = arg1;
  task.args[2] = arg2;
  return task;
}

// Construct Result variants.
__device__ inline Result make_value(u64 value) {
  Result result;
  result.tag   = RESULT_VALUE;
  result.value = value;
  return result;
}

__device__ inline Result make_split(Task child0, Task child1) {
  Result result;
  result.tag    = RESULT_SPLIT;
  result.child0 = child0;
  result.child1 = child1;
  return result;
}

__device__ inline Result make_call(Task child) {
  Result result;
  result.tag    = RESULT_CALL;
  result.child0 = child;
  return result;
}

// --------------------------------------------------------------------------
// Initialize a continuation.
// --------------------------------------------------------------------------

__device__ inline void init_continuation(
    Continuation *cont,
    u32 pending,
    u32 ret,
    u16 fn,
    u16 a0,
    u16 a1) {
  cont->pending  = pending;
  cont->ret      = ret;
  cont->fn       = fn;
  cont->a0       = a0;
  cont->a1       = a1;
  cont->_padding = 0;
  cont->slots[0] = 0;
  cont->slots[1] = 0;
}

// ==========================================================================
// Sequential Device Functions
// ==========================================================================
//
// These are direct translations of the recursive algorithms from bitonic.c.
// They execute entirely within a single thread — no task spawning. Used
// during the WORK phase when the task matrix is fully populated and each
// thread runs its assigned subtask to completion.
//
// Naming convention: seq_* corresponds to the original C function.
// ==========================================================================

// seq_gen: generate a binary tree with leaves labeled by position.
//   gen(d, x) = Leaf(x)               if d == 0
//   gen(d, x) = Node(gen(d-1, 2x+1),  gen(d-1, 2x))
__device__ u64 seq_gen(u32 depth, u32 x, u64 *heap, u32 &heap_ptr) {
  if (depth == 0) return make_leaf(x);
  u64 left  = seq_gen(depth - 1, x * 2 + 1, heap, heap_ptr);
  u64 right = seq_gen(depth - 1, x * 2,     heap, heap_ptr);
  return make_node(alloc_node(left, right, heap, heap_ptr));
}

// seq_pow31: compute 31^n mod 2^32 (used by checksum).
__device__ u32 seq_pow31(u32 n) {
  u32 result = 1;
  for (u32 i = 0; i < n; i++) result *= 31u;
  return result;
}

// seq_checksum: parallel-friendly tree checksum.
//   The original checksum is a left-to-right fold: result = result*31 + val.
//   For a balanced tree, we can split it: if left subtree has n leaves,
//   then combined = left_checksum * 31^n + right_checksum.
__device__ u64 seq_checksum(u64 tree, u32 depth, u64 *heap) {
  if (depth == 0) return (u64)leaf_value(tree);
  u32 left_sum  = (u32)seq_checksum(heap[node_index(tree)],     depth - 1, heap);
  u32 right_sum = (u32)seq_checksum(heap[node_index(tree) + 1], depth - 1, heap);
  u32 leaves_in_right = 1u << (depth - 1);
  return (u64)(left_sum * seq_pow31(leaves_in_right) + right_sum);
}

// seq_warp: the "warp" operation — compare-and-swap across two subtrees.
//   At depth 0, compares two leaf values and swaps based on direction s.
//   At depth > 0, recursively warps corresponding children and reassembles.
__device__ u64 seq_warp(u32 depth, u32 side, u64 a, u64 b, u64 *heap, u32 &heap_ptr) {
  if (depth == 0) {
    u32 val_a = leaf_value(a);
    u32 val_b = leaf_value(b);
    u32 should_swap = side ^ (val_a > val_b ? 1u : 0u);
    if (should_swap == 0)
      return make_node(alloc_node(make_leaf(val_a), make_leaf(val_b), heap, heap_ptr));
    else
      return make_node(alloc_node(make_leaf(val_b), make_leaf(val_a), heap, heap_ptr));
  }
  u64 wa = seq_warp(depth - 1, side,
                    heap[node_index(a)],     heap[node_index(b)],     heap, heap_ptr);
  u64 wb = seq_warp(depth - 1, side,
                    heap[node_index(a) + 1], heap[node_index(b) + 1], heap, heap_ptr);
  u32 left_idx  = alloc_node(heap[node_index(wa)],     heap[node_index(wb)],     heap, heap_ptr);
  u32 right_idx = alloc_node(heap[node_index(wa) + 1], heap[node_index(wb) + 1], heap, heap_ptr);
  return make_node(alloc_node(make_node(left_idx), make_node(right_idx), heap, heap_ptr));
}

// seq_flow and seq_down: the "flow" phase of bitonic sort.
//   flow warps the two halves, then recursively flows each sub-half.
__device__ u64 seq_flow(u32 depth, u32 side, u64 tree, u64 *heap, u32 &heap_ptr);

__device__ u64 seq_down(u32 depth, u32 side, u64 tree, u64 *heap, u32 &heap_ptr) {
  if (depth == 0 || is_leaf(tree)) return tree;
  u64 left  = seq_flow(depth - 1, side, heap[node_index(tree)],     heap, heap_ptr);
  u64 right = seq_flow(depth - 1, side, heap[node_index(tree) + 1], heap, heap_ptr);
  return make_node(alloc_node(left, right, heap, heap_ptr));
}

__device__ u64 seq_flow(u32 depth, u32 side, u64 tree, u64 *heap, u32 &heap_ptr) {
  if (depth == 0 || is_leaf(tree)) return tree;
  u64 warped = seq_warp(depth - 1, side,
                        heap[node_index(tree)], heap[node_index(tree) + 1],
                        heap, heap_ptr);
  return seq_down(depth, side, warped, heap, heap_ptr);
}

// seq_sort: the full bitonic sort — recursively sort halves, then merge.
__device__ u64 seq_sort(u32 depth, u32 side, u64 tree, u64 *heap, u32 &heap_ptr) {
  if (depth == 0 || is_leaf(tree)) return tree;
  u64 sorted_left  = seq_sort(depth - 1, 0, heap[node_index(tree)],     heap, heap_ptr);
  u64 sorted_right = seq_sort(depth - 1, 1, heap[node_index(tree) + 1], heap, heap_ptr);
  u64 merged = make_node(alloc_node(sorted_left, sorted_right, heap, heap_ptr));
  return seq_flow(depth, side, merged, heap, heap_ptr);
}

// ==========================================================================
// Task Execution (SEED/GROW phases)
// ==========================================================================
//
// During SEED and GROW, tasks are not executed to completion. Instead, each
// task function inspects its arguments and returns:
//   - VALUE if it's a base case (leaf, depth=0)
//   - SPLIT if it has two parallel children (allocates a continuation)
//   - CALL  if it has one child with a continuation
//
// The SEED/GROW loop collects the children and iterates, doubling the task
// count each round until the matrix is full.
// ==========================================================================

__device__ Result execute_task_split(
    Task         &task,
    u64          *heap,
    u32          &heap_ptr,
    Continuation *conts,
    u32           cont_index) {

  u32 fn  = task.fn;
  u32 ret = task.ret;

  // -- sort(depth, side, tree) --
  if (fn == FN_SORT) {
    u32 depth = unpack_low(task.args[0]);
    u32 side  = unpack_high(task.args[0]);
    u64 tree  = task.args[1];
    if (depth == 0 || is_leaf(tree)) return make_value(tree);
    u64 left  = heap[node_index(tree)];
    u64 right = heap[node_index(tree) + 1];
    init_continuation(&conts[cont_index], 2, ret, FN_SORT_CONT, (u16)depth, (u16)side);
    return make_split(
      make_task(FN_SORT, encode_return(cont_index, 0), pack(depth - 1, 0), left),
      make_task(FN_SORT, encode_return(cont_index, 1), pack(depth - 1, 1), right));
  }

  // -- swap/warp(side, tree, depth) --
  if (fn == FN_SWAP) {
    u32 side  = unpack_low(task.args[0]);
    u32 depth = unpack_high(task.args[0]);
    u64 tree  = task.args[1];
    if (is_leaf(tree)) return make_value(tree);
    u64 left  = heap[node_index(tree)];
    u64 right = heap[node_index(tree) + 1];
    // Base case: two leaves — compare and swap directly
    if (is_leaf(left) && is_leaf(right)) {
      u32 val_a = leaf_value(left);
      u32 val_b = leaf_value(right);
      u32 should_swap = side ^ (val_a > val_b ? 1u : 0u);
      u32 idx;
      if (should_swap == 0)
        idx = alloc_node(make_leaf(val_a), make_leaf(val_b), heap, heap_ptr);
      else
        idx = alloc_node(make_leaf(val_b), make_leaf(val_a), heap, heap_ptr);
      return make_value(make_node(idx));
    }
    // Recursive case: rearrange children and split
    u32 pair0 = alloc_node(heap[node_index(left)],     heap[node_index(right)],     heap, heap_ptr);
    u32 pair1 = alloc_node(heap[node_index(left) + 1], heap[node_index(right) + 1], heap, heap_ptr);
    init_continuation(&conts[cont_index], 2, ret, FN_SWAP_JOIN, 0, 0);
    return make_split(
      make_task(FN_SWAP, encode_return(cont_index, 0), pack(side, depth - 1), make_node(pair0)),
      make_task(FN_SWAP, encode_return(cont_index, 1), pack(side, depth - 1), make_node(pair1)));
  }

  // -- flow(depth, side, tree) --
  if (fn == FN_FLOW) {
    u32 depth = unpack_low(task.args[0]);
    u32 side  = unpack_high(task.args[0]);
    u64 tree  = task.args[1];
    if (depth == 0 || is_leaf(tree)) return make_value(tree);
    // Flow = warp then split into two sub-flows. But warp must happen first,
    // so we CALL warp and set up a continuation to handle the split afterward.
    init_continuation(&conts[cont_index], 1, ret, FN_FLOW_AFTER, (u16)depth, (u16)side);
    return make_call(
      make_task(FN_SWAP, encode_return(cont_index, 0), pack(side, depth - 1), tree));
  }

  // -- gen(depth, x) --
  if (fn == FN_GEN) {
    u32 depth = unpack_low(task.args[0]);
    u32 x     = unpack_high(task.args[0]);
    if (depth == 0) return make_value(make_leaf(x));
    init_continuation(&conts[cont_index], 2, ret, FN_GEN_JOIN, 0, 0);
    return make_split(
      make_task(FN_GEN, encode_return(cont_index, 0), pack(depth - 1, x * 2 + 1)),
      make_task(FN_GEN, encode_return(cont_index, 1), pack(depth - 1, x * 2)));
  }

  // -- checksum(tree, depth) --
  if (fn == FN_CSUM) {
    u32 depth = unpack_low(task.args[0]);
    u64 tree  = task.args[1];
    if (depth == 0) return make_value((u64)leaf_value(tree));
    init_continuation(&conts[cont_index], 2, ret, FN_CSUM_JOIN, (u16)depth, 0);
    return make_split(
      make_task(FN_CSUM, encode_return(cont_index, 0), pack(depth - 1, 0), heap[node_index(tree)]),
      make_task(FN_CSUM, encode_return(cont_index, 1), pack(depth - 1, 0), heap[node_index(tree) + 1]));
  }

  return make_value(0);
}

// ==========================================================================
// Continuation Execution (WORK phase)
// ==========================================================================
//
// When a continuation's pending count reaches zero (both children done),
// it fires: the continuation's resume function runs with the saved arguments
// and the children's results. This may produce VALUE (propagates up),
// SPLIT (new tasks), or CALL (new task + continuation).
// ==========================================================================

__device__ Result execute_continuation(
    Continuation *cont,
    u64          *heap,
    u32          &heap_ptr,
    Continuation *all_conts,
    u32          &cont_local_ptr,
    u32          &cont_local_end,
    u32          *cont_bump) {

  u16 fn = cont->fn;

  // -- sort_cont: got sorted left and right → build Node, start flow --
  if (fn == FN_SORT_CONT) {
    u32 depth = cont->a0;
    u32 side  = cont->a1;
    u64 tree = make_node(alloc_node(cont->slots[0], cont->slots[1], heap, heap_ptr));
    if (depth == 0 || is_leaf(tree)) return make_value(tree);
    // Allocate a continuation for flow_after_swap
    if (cont_local_ptr >= cont_local_end) {
      cont_local_ptr = atomicAdd(cont_bump, (u32)CONT_CHUNK);
      cont_local_end = cont_local_ptr + CONT_CHUNK;
    }
    u32 ci = cont_local_ptr++;
    init_continuation(&all_conts[ci], 1, cont->ret, FN_FLOW_AFTER, (u16)depth, (u16)side);
    return make_call(
      make_task(FN_SWAP, encode_return(ci, 0), pack(side, depth - 1), tree));
  }

  // -- flow_after: got warp result → split into left and right flow --
  if (fn == FN_FLOW_AFTER) {
    u32 depth = cont->a0;
    u32 side  = cont->a1;
    u64 tree  = cont->slots[0];
    if (is_leaf(tree)) return make_value(tree);
    u64 left  = heap[node_index(tree)];
    u64 right = heap[node_index(tree) + 1];
    if (cont_local_ptr >= cont_local_end) {
      cont_local_ptr = atomicAdd(cont_bump, (u32)CONT_CHUNK);
      cont_local_end = cont_local_ptr + CONT_CHUNK;
    }
    u32 ci = cont_local_ptr++;
    init_continuation(&all_conts[ci], 2, cont->ret, FN_FLOW_JOIN, 0, 0);
    return make_split(
      make_task(FN_FLOW, encode_return(ci, 0), pack(depth - 1, side), left),
      make_task(FN_FLOW, encode_return(ci, 1), pack(depth - 1, side), right));
  }

  // -- flow_join: got both flowed halves → combine into Node --
  if (fn == FN_FLOW_JOIN) {
    return make_value(make_node(alloc_node(cont->slots[0], cont->slots[1], heap, heap_ptr)));
  }

  // -- swap_join: got both swapped halves → reassemble the warp structure --
  if (fn == FN_SWAP_JOIN) {
    u64 result0 = cont->slots[0];
    u64 result1 = cont->slots[1];
    u32 left_idx  = alloc_node(heap[node_index(result0)],     heap[node_index(result1)],     heap, heap_ptr);
    u32 right_idx = alloc_node(heap[node_index(result0) + 1], heap[node_index(result1) + 1], heap, heap_ptr);
    return make_value(make_node(alloc_node(make_node(left_idx), make_node(right_idx), heap, heap_ptr)));
  }

  // -- gen_join: got both generated subtrees → combine into Node --
  if (fn == FN_GEN_JOIN) {
    return make_value(make_node(alloc_node(cont->slots[0], cont->slots[1], heap, heap_ptr)));
  }

  // -- checksum_join: combine left and right checksums --
  if (fn == FN_CSUM_JOIN) {
    u32 depth     = cont->a0;
    u32 left_sum  = (u32)cont->slots[0];
    u32 right_sum = (u32)cont->slots[1];
    u32 right_leaves = 1u << (depth - 1);
    return make_value((u64)(left_sum * seq_pow31(right_leaves) + right_sum));
  }

  return make_value(0);
}

// ==========================================================================
// Value Resolution
// ==========================================================================
//
// When a task or continuation produces a VALUE, the runtime must deliver it:
//   1. If the return address is ROOT_RETURN, the computation is done.
//   2. Otherwise, decode the continuation index and slot from the return
//      address, write the value into that slot, and atomically decrement
//      the pending count.
//   3. If we decremented pending from 1 → 0, we are the last child: fire
//      the continuation. Its result may cascade (another VALUE going up,
//      or new tasks via SPLIT/CALL).
// ==========================================================================

__device__ void resolve_value(
    u32  ret,
    u64  value,
    u64  *heap,
    u32  &heap_ptr,
    Continuation *conts,
    u32  &cont_local_ptr,
    u32  &cont_local_end,
    u32  *cont_bump,
    Task *output_buffer,
    u32  *output_count,
    u32  *done_flag,
    u64  *root_result) {

  for (;;) {
    // Root computation is complete
    if (ret == ROOT_RETURN) {
      *root_result = value;
      __threadfence();
      atomicExch(done_flag, 1u);
      return;
    }

    // Decode return address: continuation index and slot
    u32 cont_index = ret >> 1;
    u32 slot       = ret & 1;
    Continuation *cont = &conts[cont_index];

    // Write our value into the continuation's slot
    cont->slots[slot] = value;
    __threadfence();

    // Atomically decrement pending count
    u32 old_pending = atomicSub(&cont->pending, 1u);

    // If we weren't the last child, we're done
    if (old_pending != 1u) return;

    // We were the last child — fire the continuation
    __threadfence();
    Result result = execute_continuation(
      cont, heap, heap_ptr,
      conts, cont_local_ptr, cont_local_end, cont_bump);

    // Handle the continuation's result
    if (result.tag == RESULT_VALUE) {
      // Cascade: deliver this value to the continuation's parent
      value = result.value;
      ret   = cont->ret;
      continue;  // Loop to resolve the parent
    }
    if (result.tag == RESULT_SPLIT) {
      u32 idx = atomicAdd(output_count, 2u);
      output_buffer[idx]     = result.child0;
      output_buffer[idx + 1] = result.child1;
      return;
    }
    if (result.tag == RESULT_CALL) {
      u32 idx = atomicAdd(output_count, 1u);
      output_buffer[idx] = result.child0;
      return;
    }
    return;
  }
}

// ==========================================================================
// Debug: shade character for task count visualization
// ==========================================================================

#ifdef DEBUG_MATRIX
__device__ const char* shade_char(u32 count) {
  if (count == 0)                       return " ";
  if (count <= (u32)(BLOCK_SIZE / 4))   return "░";
  if (count <= (u32)(BLOCK_SIZE / 2))   return "▒";
  if (count <= (u32)(3 * BLOCK_SIZE / 4)) return "▓";
  return "█";
}
#endif

// ==========================================================================
// Main Cooperative Kernel
// ==========================================================================
//
// This single persistent kernel runs the SEED → GROW → WORK loop until the
// computation completes. Grid-wide synchronization (grid.sync()) separates
// phases, eliminating all host↔device sync overhead.
// ==========================================================================

__global__ void evaluator_kernel(GPUState state) {
  cg::grid_group grid = cg::this_grid();

  int thread_id = threadIdx.x;
  int block_id  = blockIdx.x;
  int slot_id   = block_id * BLOCK_SIZE + thread_id;

  // Shared memory: double buffer for SEED/GROW task expansion,
  // and output buffer for WORK phase results.
  __shared__ Task task_buf[2][BLOCK_SIZE];
  __shared__ u32  buf_count;         // Current number of tasks in active buffer
  __shared__ u32  buf_new_count;     // Count being built in the other buffer
  __shared__ u32  cont_block_base;   // Base index for this block's continuation chunk
  __shared__ u32  cont_block_used;   // How many conts this block has allocated
  __shared__ Task work_output[BLOCK_SIZE];  // New tasks produced during WORK
  __shared__ u32  work_output_count;
  __shared__ u32  work_flat_base;    // Where in flat buffer to write our output
  __shared__ u32  work_task_count;   // How many tasks this block has in WORK

#ifdef DEBUG_MATRIX
  int tick = 0;
#endif

  // ======================================================================
  // Main loop: repeats SEED → GROW → WORK until done
  // ======================================================================
  for (;;) {
    grid.sync();

    // Check termination
    u32 flat_count = *(volatile u32*)state.flat_count;
    if (flat_count == 0 || *(volatile u32*)state.done_flag) return;

    // ==================================================================
    // SEED PHASE — Expand a small number of tasks to NUM_BLOCKS
    // ==================================================================
    // Only runs when flat_count ≤ NUM_BLOCKS (not enough to fill all blocks).
    // Block 0 runs this alone. It loads the tasks into shared memory and
    // iteratively splits them: each SPLIT result doubles the count.
    // After enough iterations, we have NUM_BLOCKS tasks in the flat buffer.
    //
    // Only tasks that SPLIT or CALL contribute to the next iteration.
    // VALUE results (base cases) are dropped — they resolve their
    // continuations but don't add new tasks. For balanced bitonic sort,
    // VALUE never occurs during SEED because depth > 0 on all paths.
    // ==================================================================

    if (flat_count <= (u32)NUM_BLOCKS) {
      if (block_id == 0) {
        // Load tasks from flat buffer into shared memory
        if (thread_id < flat_count) {
          task_buf[0][thread_id] = state.flat_buffer[thread_id];
        }
        if (thread_id == 0) {
          buf_count      = flat_count;
          cont_block_used = 0;
          cont_block_base = atomicAdd(state.cont_bump, 512u);
        }
        __syncthreads();

        u32 heap_ptr = state.heap_pointers[thread_id];
        int current_buf = 0;

        // Iterate: each round, active threads execute their task and collect
        // children into the alternate buffer. Stop when we have enough tasks
        // or when no more splitting occurs.
        // Guard: sn <= NUM_BLOCKS/2 ensures a full SPLIT round can't exceed NUM_BLOCKS.
        for (int iter = 0; iter < 32 && buf_count > 0 && buf_count <= (u32)(NUM_BLOCKS / 2); iter++) {
          if (thread_id == 0) buf_new_count = 0;
          __syncthreads();

          u32 active_count = buf_count;
          if (thread_id < active_count) {
            u32 cont_idx = cont_block_base + atomicAdd(&cont_block_used, 1u);
            Result result = execute_task_split(
              task_buf[current_buf][thread_id],
              state.heap, heap_ptr, state.continuations, cont_idx);

            if (result.tag == RESULT_SPLIT) {
              u32 dest = atomicAdd(&buf_new_count, 2u);
              task_buf[1 - current_buf][dest]     = result.child0;
              task_buf[1 - current_buf][dest + 1] = result.child1;
            } else if (result.tag == RESULT_CALL) {
              u32 dest = atomicAdd(&buf_new_count, 1u);
              task_buf[1 - current_buf][dest] = result.child0;
            }
            // VALUE results: continuation slot is filled but task is not re-queued
          }
          __syncthreads();

          current_buf = 1 - current_buf;
          if (thread_id == 0) buf_count = buf_new_count;
          __syncthreads();
        }

        // Write expanded tasks back to flat buffer
        if (thread_id < buf_count) {
          state.flat_buffer[thread_id] = task_buf[current_buf][thread_id];
        }
        state.heap_pointers[thread_id] = heap_ptr;
        if (thread_id == 0) {
          *state.flat_count = buf_count;
          __threadfence();
        }
        __syncthreads();

#ifdef DEBUG_MATRIX
        if (thread_id == 0) {
          printf("S%04d:", tick++);
          for (int b = 0; b < NUM_BLOCKS; b++)
            printf("%s", shade_char((b == 0) ? buf_count : 0));
          printf("\n");
        }
#endif
      }

      grid.sync();
      flat_count = *(volatile u32*)state.flat_count;
      if (flat_count == 0) return;
    }

    // ==================================================================
    // GROW PHASE — Expand each block's tasks from flat buffer to BLOCK_SIZE
    // ==================================================================
    // The flat buffer has flat_count tasks. Distribute them evenly across
    // NUM_BLOCKS blocks. Each block loads its share into shared memory and
    // iteratively splits, just like SEED, until it has BLOCK_SIZE tasks.
    // Results are written to the task_matrix in column-major order.
    // ==================================================================

    {
      // Compute how many tasks this block gets from the flat buffer
      u32 tasks_per_block = flat_count / NUM_BLOCKS;
      u32 extra_tasks     = flat_count % NUM_BLOCKS;
      u32 my_task_count   = tasks_per_block + ((u32)block_id < extra_tasks ? 1u : 0u);
      u32 my_base_offset  = block_id * tasks_per_block + ((u32)block_id < extra_tasks ? (u32)block_id : extra_tasks);

      // Load our share into shared memory
      if (thread_id < my_task_count) {
        task_buf[0][thread_id] = state.flat_buffer[my_base_offset + thread_id];
      }
      if (thread_id == 0) {
        buf_count       = my_task_count;
        cont_block_used = 0;
        cont_block_base = atomicAdd(state.cont_bump, 256u);
      }
      __syncthreads();

      u32 heap_ptr = state.heap_pointers[slot_id];
      int current_buf = 0;

      // Same doubling loop as SEED, but per-block and targeting BLOCK_SIZE
      for (int iter = 0; iter < 32 && buf_count > 0 && buf_count <= (u32)(BLOCK_SIZE / 2); iter++) {
        if (thread_id == 0) buf_new_count = 0;
        __syncthreads();

        u32 active_count = buf_count;
        if (thread_id < active_count) {
          u32 cont_idx = cont_block_base + atomicAdd(&cont_block_used, 1u);
          Result result = execute_task_split(
            task_buf[current_buf][thread_id],
            state.heap, heap_ptr, state.continuations, cont_idx);

          if (result.tag == RESULT_SPLIT) {
            u32 dest = atomicAdd(&buf_new_count, 2u);
            task_buf[1 - current_buf][dest]     = result.child0;
            task_buf[1 - current_buf][dest + 1] = result.child1;
          } else if (result.tag == RESULT_CALL) {
            u32 dest = atomicAdd(&buf_new_count, 1u);
            task_buf[1 - current_buf][dest] = result.child0;
          }
        }
        __syncthreads();

        current_buf = 1 - current_buf;
        if (thread_id == 0) buf_count = buf_new_count;
        __syncthreads();
      }

      // Write to column-format task matrix and record per-block count
      if (thread_id < buf_count) {
        state.task_matrix[block_id * BLOCK_SIZE + thread_id] = task_buf[current_buf][thread_id];
      }
      state.heap_pointers[slot_id] = heap_ptr;
      if (thread_id == 0) {
        state.block_counts[block_id] = buf_count;
      }
      __syncthreads();
    }

#ifdef DEBUG_MATRIX
    grid.sync();
    if (block_id == 0 && thread_id == 0) {
      printf("G%04d:", tick++);
      for (int b = 0; b < NUM_BLOCKS; b++)
        printf("%s", shade_char(state.block_counts[b]));
      printf("\n");
    }
    grid.sync();
#endif

    // Reset flat_count before WORK (must be visible to all blocks)
    if (block_id == 0 && thread_id == 0) {
      *state.flat_count = 0;
    }
    grid.sync();

    // ==================================================================
    // WORK PHASE — Execute all tasks sequentially, resolve continuations
    // ==================================================================
    // Each thread executes its assigned task by calling the sequential
    // (recursive) version of the function. The result is a VALUE that gets
    // fed into the continuation resolution chain. Fired continuations may
    // produce new tasks (SPLIT/CALL) which are collected in shared memory
    // and then bulk-written to the flat buffer for the next round.
    // ==================================================================

    {
      if (thread_id == 0) {
        work_output_count = 0;
        work_task_count   = state.block_counts[block_id];
      }
      __syncthreads();

      u32 heap_ptr       = state.heap_pointers[slot_id];
      u32 cont_local_ptr = 0;
      u32 cont_local_end = 0;

      // Only execute if this thread has a valid task
      if (thread_id < work_task_count) {
        Task task = state.task_matrix[slot_id];
        u64 value;

        // Dispatch to the appropriate sequential function
        switch (task.fn) {
          case FN_SORT: {
            u32 depth = unpack_low(task.args[0]);
            u32 side  = unpack_high(task.args[0]);
            value = seq_sort(depth, side, task.args[1], state.heap, heap_ptr);
            break;
          }
          case FN_FLOW: {
            u32 depth = unpack_low(task.args[0]);
            u32 side  = unpack_high(task.args[0]);
            value = seq_flow(depth, side, task.args[1], state.heap, heap_ptr);
            break;
          }
          case FN_SWAP: {
            u32 side  = unpack_low(task.args[0]);
            u32 depth = unpack_high(task.args[0]);
            u64 tree  = task.args[1];
            value = seq_warp(depth, side,
                             state.heap[node_index(tree)],
                             state.heap[node_index(tree) + 1],
                             state.heap, heap_ptr);
            break;
          }
          case FN_GEN: {
            u32 depth = unpack_low(task.args[0]);
            u32 x     = unpack_high(task.args[0]);
            value = seq_gen(depth, x, state.heap, heap_ptr);
            break;
          }
          case FN_CSUM: {
            u32 depth = unpack_low(task.args[0]);
            value = seq_checksum(task.args[1], depth, state.heap);
            break;
          }
          default:
            value = 0;
        }

        // Resolve: deliver value to parent continuation, possibly cascading
        resolve_value(
          task.ret, value,
          state.heap, heap_ptr,
          state.continuations, cont_local_ptr, cont_local_end, state.cont_bump,
          work_output, &work_output_count,
          state.done_flag, state.root_result);
      }

      __syncthreads();
      state.heap_pointers[slot_id] = heap_ptr;

      // Bulk-write new tasks from shared memory to the global flat buffer.
      // One atomic per block (not per thread) to claim space.
      u32 output_count = work_output_count;
      if (thread_id == 0 && output_count > 0) {
        work_flat_base = atomicAdd(state.flat_count, output_count);
      }
      __syncthreads();
      for (u32 i = thread_id; i < output_count; i += BLOCK_SIZE) {
        state.flat_buffer[work_flat_base + i] = work_output[i];
      }
      __syncthreads();
    }

#ifdef DEBUG_MATRIX
    grid.sync();
    if (block_id == 0 && thread_id == 0) {
      u32 fc    = *(volatile u32*)state.flat_count;
      u32 per_b = fc / NUM_BLOCKS;
      u32 extra = fc % NUM_BLOCKS;
      printf("W%04d:", tick++);
      for (int b = 0; b < NUM_BLOCKS; b++)
        printf("%s", shade_char(per_b + ((u32)b < extra ? 1 : 0)));
      printf("\n");
    }
    grid.sync();
#endif
  }
}

// ==========================================================================
// Host-side: Run one top-level function through the evaluator
// ==========================================================================
//
// Resets the done flag and continuation allocator, places a single task in
// the flat buffer, and launches the cooperative kernel. The kernel will
// SEED → GROW → WORK until the result is produced.
// ==========================================================================

static void run_task(GPUState &state, u32 fn, u64 arg0, u64 arg1, float *elapsed_ms) {
  u32 zero = 0;
  CUDA_CHECK(cudaMemcpy(state.done_flag,  &zero, sizeof(u32), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(state.cont_bump,  &zero, sizeof(u32), cudaMemcpyHostToDevice));

  Task initial_task = make_task(fn, ROOT_RETURN, arg0, arg1);
  CUDA_CHECK(cudaMemcpy(state.flat_buffer, &initial_task, sizeof(Task), cudaMemcpyHostToDevice));

  u32 one = 1;
  CUDA_CHECK(cudaMemcpy(state.flat_count, &one, sizeof(u32), cudaMemcpyHostToDevice));

  void *kernel_args[] = { &state };

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)evaluator_kernel, NUM_BLOCKS, BLOCK_SIZE, kernel_args));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

// ==========================================================================
// Main
// ==========================================================================

int main(int argc, char **argv) {
  int depth = 20;
  if (argc > 1) depth = atoi(argv[1]);
  if (depth < 1 || depth > 24) {
    fprintf(stderr, "Usage: %s [depth]  (depth 1..24)\n", argv[0]);
    return 1;
  }
  u32 num_elements = 1u << depth;
  fprintf(stderr, "Bitonic sort  depth=%d  elems=%u  (%d blocks × %d threads)\n",
          depth, num_elements, NUM_BLOCKS, BLOCK_SIZE);

  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

  // --------------------------------------------------------------------------
  // Allocate all GPU memory in a single cudaMalloc to minimize overhead.
  // --------------------------------------------------------------------------

  #define ALIGN256(x) (((x) + 255) & ~(size_t)255)

  size_t offset = 0;
  size_t off_heap      = offset; offset += ALIGN256((size_t)HEAP_SIZE * sizeof(u64));
  size_t off_heap_ptrs = offset; offset += ALIGN256((size_t)TOTAL_SLOTS * sizeof(u32));
  size_t off_conts     = offset; offset += ALIGN256((size_t)CONT_CAP * sizeof(Continuation));
  size_t off_cont_bump = offset; offset += ALIGN256(sizeof(u32));
  size_t off_tasks     = offset; offset += ALIGN256((size_t)TOTAL_SLOTS * sizeof(Task));
  size_t off_flat      = offset; offset += ALIGN256((size_t)TOTAL_SLOTS * sizeof(Task));
  size_t off_flat_cnt  = offset; offset += ALIGN256(sizeof(u32));
  size_t off_blk_cnts  = offset; offset += ALIGN256((size_t)NUM_BLOCKS * sizeof(u32));
  size_t off_done      = offset; offset += ALIGN256(sizeof(u32));
  size_t off_result    = offset; offset += ALIGN256(sizeof(u64));

  char *device_memory;
  CUDA_CHECK(cudaMalloc(&device_memory, offset));
  CUDA_CHECK(cudaMemset(device_memory, 0, offset));

  GPUState state;
  state.heap          = (u64 *)         (device_memory + off_heap);
  state.heap_pointers = (u32 *)         (device_memory + off_heap_ptrs);
  state.continuations = (Continuation *)(device_memory + off_conts);
  state.cont_bump     = (u32 *)         (device_memory + off_cont_bump);
  state.task_matrix   = (Task *)        (device_memory + off_tasks);
  state.flat_buffer   = (Task *)        (device_memory + off_flat);
  state.flat_count    = (u32 *)         (device_memory + off_flat_cnt);
  state.block_counts  = (u32 *)         (device_memory + off_blk_cnts);
  state.done_flag     = (u32 *)         (device_memory + off_done);
  state.root_result   = (u64 *)         (device_memory + off_result);

  // Initialize per-slot heap pointers: each slot gets an equal slice of the heap.
  u32 slice_size = HEAP_SIZE / TOTAL_SLOTS;
  u32 *host_heap_ptrs = (u32 *)malloc(TOTAL_SLOTS * sizeof(u32));
  for (int i = 0; i < TOTAL_SLOTS; i++) {
    host_heap_ptrs[i] = (u32)i * slice_size;
  }
  CUDA_CHECK(cudaMemcpy(state.heap_pointers, host_heap_ptrs, TOTAL_SLOTS * sizeof(u32), cudaMemcpyHostToDevice));
  free(host_heap_ptrs);

  // --------------------------------------------------------------------------
  // Run the three phases of the computation — all on GPU, all through the
  // same evaluator pipeline:
  //
  //   1. gen(depth, 0)         → builds the input tree
  //   2. sort(depth, 0, tree)  → sorts it via bitonic sort
  //   3. checksum(tree, depth) → computes a verification checksum
  // --------------------------------------------------------------------------

  float ms;

  // Phase 1: Generate the input tree on the GPU
  run_task(state, FN_GEN, pack(depth, 0), 0, &ms);
  u64 input_tree;
  CUDA_CHECK(cudaMemcpy(&input_tree, state.root_result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  gen  %.1f ms\n", ms);

  // Phase 2: Sort the tree
  run_task(state, FN_SORT, pack(depth, 0), input_tree, &ms);
  u64 sorted_tree;
  CUDA_CHECK(cudaMemcpy(&sorted_tree, state.root_result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  sort %.1f ms\n", ms);

  // Phase 3: Checksum the sorted tree
  run_task(state, FN_CSUM, pack(depth, 0), sorted_tree, &ms);
  u64 checksum;
  CUDA_CHECK(cudaMemcpy(&checksum, state.root_result, sizeof(u64), cudaMemcpyDeviceToHost));
  fprintf(stderr, "  csum %.1f ms\n", ms);

  printf("%u\n", (u32)checksum);

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------

  CUDA_CHECK(cudaFree(device_memory));
  return 0;
}
