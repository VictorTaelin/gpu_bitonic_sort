// bitonic.cu — GPU parallel recursive bitonic sort
//
// Executes the recursive bitonic sort algorithm (sort/flow/warp/down) on GPU
// using a task-bag + continuation-frame model. Each "par" annotation in the
// algorithm (two independent recursive calls) creates a Frame and two Tasks.
// Tasks are processed in rounds; between rounds, a grid-wide sync ensures
// all results are visible.
//
// Key optimizations over the naive approach:
//   - DFS fallback: when there are already enough tasks to fill the GPU,
//     new pars are computed sequentially instead of creating more tasks.
//     Controlled by g_dfs_depth (set so peak bag ≈ 65536 tasks).
//   - Per-thread frame chunk allocator: batches frame index allocation,
//     reducing global atomicAdd contention by ~128x.
//   - Per-thread heap chunk allocator: same idea for tree node allocation.
//   - Block-level task batching: new tasks go to shared memory first,
//     then one atomicAdd per block flushes to the global bag.
//   - Frame recycling: completed frames are reused in cascade chains.
//   - Batched node constructors: swap/zip allocate 3 nodes in one call.
//
// Tree encoding: 2 words per node, tag in bit 0 of word1.
//   Leaf: [value, 1]      (word1 odd)
//   Node: [left,  right]  (word1 even; all pointers are 2-aligned)
// This allows full 32-bit pointers (up to 16 GB heap).
//
// Compilation: nvcc -O3 -arch=sm_89 -rdc=true bitonic.cu -o bitonic_gpu -lcudadevrt
// Usage:       ./bitonic_gpu [depth=20] [heap_gb=16]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

typedef uint32_t u32;
typedef int32_t  i32;
typedef uint64_t u64;

#define CHK(call) do {                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                cudaGetErrorString(err));                                  \
        exit(1);                                                           \
    }                                                                      \
} while (0)

// ---------------------------------------------------------------------------
// Tree encoding
// ---------------------------------------------------------------------------

#define IS_NODE(heap, ptr)  (!((heap)[(ptr) + 1] & 1))
#define GET_VAL(heap, ptr)  ((heap)[(ptr)])
#define LEFT(heap, ptr)     ((heap)[(ptr)])
#define RIGHT(heap, ptr)    ((heap)[(ptr) + 1])

#define NO_PARENT 0xFFFFFFFFu

// DFS threshold: stored in constant memory, set from host before launch.
__constant__ u32 g_dfs_depth;

// ---------------------------------------------------------------------------
// Chunk allocators
// ---------------------------------------------------------------------------

#define HEAP_CHUNK_WORDS 1024
#define FRAME_CHUNK_SIZE 128

struct HeapChunk {
    u32 base, used, capacity;
};

struct FrameChunk {
    u32 base, used, capacity;
};

__device__ __forceinline__
u32 chunk_alloc(HeapChunk *chunk, u32 *heap_ptr, u32 num_words) {
    if (chunk->used + num_words > chunk->capacity) {
        chunk->base     = atomicAdd(heap_ptr, HEAP_CHUNK_WORDS);
        chunk->used     = 0;
        chunk->capacity = HEAP_CHUNK_WORDS;
    }
    u32 offset = chunk->base + chunk->used;
    chunk->used += num_words;
    return offset;
}

__device__ __forceinline__
u32 frame_alloc(FrameChunk *chunk, u32 *frame_ptr) {
    if (chunk->used >= chunk->capacity) {
        chunk->base     = atomicAdd(frame_ptr, FRAME_CHUNK_SIZE);
        chunk->used     = 0;
        chunk->capacity = FRAME_CHUNK_SIZE;
    }
    return chunk->base + chunk->used++;
}

// ---------------------------------------------------------------------------
// Node constructors
// ---------------------------------------------------------------------------

__device__ __forceinline__
u32 make_leaf(u32 *heap, HeapChunk *chunk, u32 *heap_ptr, u32 value) {
    u32 p = chunk_alloc(chunk, heap_ptr, 2);
    heap[p]     = value;
    heap[p + 1] = 1;
    return p;
}

__device__ __forceinline__
u32 make_node(u32 *heap, HeapChunk *chunk, u32 *heap_ptr, u32 left, u32 right) {
    u32 p = chunk_alloc(chunk, heap_ptr, 2);
    heap[p]     = left;
    heap[p + 1] = right;
    return p;
}

// swap: compare two leaves, conditionally swap. Allocates 3 nodes (6 words).
__device__ __forceinline__
u32 make_swap(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
              u32 dir, u32 va, u32 vb) {
    u32 do_swap = dir ^ (va > vb);
    u32 p = chunk_alloc(chunk, heap_ptr, 6);
    heap[p]     = do_swap ? vb : va;  heap[p + 1] = 1;      // leaf 0
    heap[p + 2] = do_swap ? va : vb;  heap[p + 3] = 1;      // leaf 1
    heap[p + 4] = p;                  heap[p + 5] = p + 2;  // node
    return p + 4;
}

// zip(a, b) = node(node(left(a), left(b)), node(right(a), right(b)))
// Allocates 3 nodes (6 words).
__device__ __forceinline__
u32 make_zip(u32 *heap, HeapChunk *chunk, u32 *heap_ptr, u32 a, u32 b) {
    u32 p = chunk_alloc(chunk, heap_ptr, 6);
    heap[p]     = LEFT(heap, a);   heap[p + 1] = LEFT(heap, b);
    heap[p + 2] = RIGHT(heap, a);  heap[p + 3] = RIGHT(heap, b);
    heap[p + 4] = p;               heap[p + 5] = p + 2;
    return p + 4;
}

// ---------------------------------------------------------------------------
// Task and Frame
// ---------------------------------------------------------------------------

#define FN_SORT 0
#define FN_FLOW 1
#define FN_WARP 2

#define CONT_SORT 0
#define CONT_WARP 1
#define CONT_FLOW 2
#define CONT_DOWN 3

struct Task {
    u32 pack;     // [fn:2 | depth:6 | dir:1 | slot:1 | unused:22]
    u32 arg0;
    u32 arg1;
    u32 cont_id;
};

#define TASK_FN(t)        ((t).pack & 3u)
#define TASK_DEPTH(t)     (((t).pack >> 2) & 63u)
#define TASK_DIR(t)       (((t).pack >> 8) & 1u)
#define TASK_SLOT(t)      (((t).pack >> 9) & 1u)

#define PACK_TASK(fn, depth, dir, slot) \
    ((u32)(fn) | ((u32)(depth) << 2) | ((u32)(dir) << 8) | ((u32)(slot) << 9))

struct Frame {
    u32 pack;       // [type:2 | depth:6 | dir:1 | parent_slot:1 | unused:22]
    u32 parent_id;
    u32 slot0;
    u32 slot1;
    i32 pending;    // atomic: 2 → 1 → 0
};

#define FRAME_TYPE(f)   ((f).pack & 3u)
#define FRAME_DEPTH(f)  (((f).pack >> 2) & 63u)
#define FRAME_DIR(f)    (((f).pack >> 8) & 1u)
#define FRAME_PSLOT(f)  (((f).pack >> 9) & 1u)

#define PACK_FRAME(type, depth, dir, pslot) \
    ((u32)(type) | ((u32)(depth) << 2) | ((u32)(dir) << 8) | ((u32)(pslot) << 9))

// ParResult: lives in registers. Either a value or a par request.
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

// ---------------------------------------------------------------------------
// DFS functions — sequential fallback for small subtrees
// ---------------------------------------------------------------------------

__device__ u32 dfs_flow(u32 *heap, HeapChunk *c, u32 *hp, u32 d, u32 s, u32 t);

__device__
u32 dfs_warp(u32 *heap, HeapChunk *c, u32 *hp,
             u32 depth, u32 dir, u32 a, u32 b) {
    if (depth == 0) {
        if (IS_NODE(heap, a) || IS_NODE(heap, b))
            return make_leaf(heap, c, hp, 0);
        return make_swap(heap, c, hp, dir, GET_VAL(heap, a), GET_VAL(heap, b));
    }
    if (!IS_NODE(heap, a) || !IS_NODE(heap, b))
        return make_leaf(heap, c, hp, 0);
    u32 wa = dfs_warp(heap, c, hp, depth - 1, dir, LEFT(heap, a), LEFT(heap, b));
    u32 wb = dfs_warp(heap, c, hp, depth - 1, dir, RIGHT(heap, a), RIGHT(heap, b));
    if (!IS_NODE(heap, wa) || !IS_NODE(heap, wb))
        return make_leaf(heap, c, hp, 0);
    return make_zip(heap, c, hp, wa, wb);
}

__device__
u32 dfs_down(u32 *heap, HeapChunk *c, u32 *hp, u32 d, u32 s, u32 t) {
    if (d == 0 || !IS_NODE(heap, t))
        return t;
    u32 l = dfs_flow(heap, c, hp, d - 1, s, LEFT(heap, t));
    u32 r = dfs_flow(heap, c, hp, d - 1, s, RIGHT(heap, t));
    return make_node(heap, c, hp, l, r);
}

__device__
u32 dfs_flow(u32 *heap, HeapChunk *c, u32 *hp, u32 d, u32 s, u32 t) {
    if (d == 0 || !IS_NODE(heap, t))
        return t;
    u32 warped = dfs_warp(heap, c, hp, d - 1, s, LEFT(heap, t), RIGHT(heap, t));
    return dfs_down(heap, c, hp, d, s, warped);
}

__device__
u32 dfs_sort(u32 *heap, HeapChunk *c, u32 *hp, u32 d, u32 s, u32 t) {
    if (d == 0 || !IS_NODE(heap, t))
        return t;
    u32 l = dfs_sort(heap, c, hp, d - 1, 0, LEFT(heap, t));
    u32 r = dfs_sort(heap, c, hp, d - 1, 1, RIGHT(heap, t));
    return dfs_flow(heap, c, hp, d, s, make_node(heap, c, hp, l, r));
}

// ---------------------------------------------------------------------------
// Par-aware functions
//
// Return a ParResult: either a value (base case or DFS fallback) or a par
// request describing two child tasks and a continuation frame.
// ---------------------------------------------------------------------------

__device__ ParResult par_down(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                              u32 depth, u32 dir, u32 tree);

__device__
ParResult par_sort(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                   u32 depth, u32 dir, u32 tree) {
    if (depth == 0 || !IS_NODE(heap, tree))
        return make_value(tree);
    if (depth <= g_dfs_depth)
        return make_value(dfs_sort(heap, chunk, heap_ptr, depth, dir, tree));

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
    if (depth <= g_dfs_depth)
        return make_value(dfs_warp(heap, chunk, heap_ptr, depth, dir, a, b));

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
    if (depth <= g_dfs_depth)
        return make_value(dfs_down(heap, chunk, heap_ptr, depth, dir, tree));

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
    if (depth <= g_dfs_depth)
        return make_value(dfs_flow(heap, chunk, heap_ptr, depth, dir, tree));

    u32 a = LEFT(heap, tree);
    u32 b = RIGHT(heap, tree);

    // Inline warp(0) for depth == 1, then delegate to down.
    if (depth == 1) {
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

__device__
ParResult resume_frame(u32 *heap, HeapChunk *chunk, u32 *heap_ptr,
                       u32 type, u32 depth, u32 dir, u32 s0, u32 s1) {
    switch (type) {
        case CONT_SORT:
            return par_flow(heap, chunk, heap_ptr, depth, dir,
                            make_node(heap, chunk, heap_ptr, s0, s1));
        case CONT_WARP:
            if (!IS_NODE(heap, s0) || !IS_NODE(heap, s1))
                return make_value(make_leaf(heap, chunk, heap_ptr, 0));
            return make_value(make_zip(heap, chunk, heap_ptr, s0, s1));
        case CONT_FLOW: {
            u32 zipped;
            if (!IS_NODE(heap, s0) || !IS_NODE(heap, s1))
                zipped = make_leaf(heap, chunk, heap_ptr, 0);
            else
                zipped = make_zip(heap, chunk, heap_ptr, s0, s1);
            return par_down(heap, chunk, heap_ptr, depth, dir, zipped);
        }
        case CONT_DOWN:
            return make_value(make_node(heap, chunk, heap_ptr, s0, s1));
    }
    return make_value(0);
}

// ---------------------------------------------------------------------------
// Cascade
// ---------------------------------------------------------------------------

__device__
void cascade(ParResult result, u32 parent_id, u32 parent_slot,
             u32 *heap, HeapChunk *hchunk, u32 *heap_ptr,
             FrameChunk *fchunk,
             Task *block_task_buf, u32 *block_task_count,
             Frame *frames, u32 *frame_ptr, u32 *root_result) {
    u32 recycle_id = NO_PARENT;

    for (;;) {
        if (result.is_par) {
            // Allocate or recycle a frame.
            u32 fid;
            if (recycle_id != NO_PARENT) {
                fid = recycle_id;
                recycle_id = NO_PARENT;
            } else {
                fid = frame_alloc(fchunk, frame_ptr);
            }

            Frame cont;
            cont.pack      = PACK_FRAME(result.cont_type, result.cont_depth,
                                         result.cont_dir, parent_slot);
            cont.parent_id = parent_id;
            cont.slot0     = 0;
            cont.slot1     = 0;
            cont.pending   = 2;
            frames[fid]    = cont;

            u32 idx = atomicAdd(block_task_count, 2);

            Task lt;
            lt.pack    = PACK_TASK(result.fn, result.depth, result.left_dir, 0);
            lt.arg0    = result.left_arg0;
            lt.arg1    = result.left_arg1;
            lt.cont_id = fid;

            Task rt;
            rt.pack    = PACK_TASK(result.fn, result.depth, result.right_dir, 1);
            rt.arg0    = result.right_arg0;
            rt.arg1    = result.right_arg1;
            rt.cont_id = fid;

            block_task_buf[idx]     = lt;
            block_task_buf[idx + 1] = rt;
            return;
        }

        if (parent_id == NO_PARENT) {
            *root_result = result.value;
            return;
        }

        Frame *parent = &frames[parent_id];
        if (parent_slot == 0)
            parent->slot0 = result.value;
        else
            parent->slot1 = result.value;

        __threadfence();

        if (atomicSub(&parent->pending, 1) == 1) {
            __threadfence();
            recycle_id  = parent_id;
            result      = resume_frame(heap, hchunk, heap_ptr,
                                       FRAME_TYPE(*parent),
                                       FRAME_DEPTH(*parent),
                                       FRAME_DIR(*parent),
                                       parent->slot0, parent->slot1);
            parent_slot = FRAME_PSLOT(*parent);
            parent_id   = parent->parent_id;
        } else {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Tree generation kernel (level-order layout)
// ---------------------------------------------------------------------------

__global__
void generate_tree(u32 *heap, u32 depth, u32 num_nodes) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    u32 num_leaves = 1u << depth;
    u32 p = i * 2;

    if (i >= num_leaves - 1) {
        u32 leaf_index = i - (num_leaves - 1);
        heap[p]     = num_leaves - 1 - leaf_index;
        heap[p + 1] = 1;  // leaf tag
    } else {
        heap[p]     = (2u * i + 1u) * 2u;  // left child
        heap[p + 1] = (2u * i + 2u) * 2u;  // right child (even → node)
    }
}

// ---------------------------------------------------------------------------
// Persistent evaluation kernel
// ---------------------------------------------------------------------------

#define THREADS_PER_BLOCK 256
#define MAX_BLOCK_TASKS   (THREADS_PER_BLOCK * 2)

__global__
void eval_kernel(Task *bag_a, Task *bag_b,
                 u32 *count_a, u32 *count_b,
                 u32 *heap, u32 *heap_ptr,
                 Frame *frames, u32 *frame_ptr,
                 u32 *root_result) {
    cg::grid_group grid = cg::this_grid();

    u32 gtid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total = gridDim.x * blockDim.x;

    __shared__ Task block_task_buf[MAX_BLOCK_TASKS];
    __shared__ u32  block_task_count;
    __shared__ u32  block_bag_base;

    Task *cur_bag = bag_a;
    Task *nxt_bag = bag_b;
    u32  *cur_n   = count_a;
    u32  *nxt_n   = count_b;

    HeapChunk  hchunk = {0, 0, 0};
    FrameChunk fchunk = {0, 0, 0};

    for (u32 round = 0; round < 100000; round++) {
        u32 num_tasks = *cur_n;
        if (num_tasks == 0) break;

        if (gtid == 0) *nxt_n = 0;
        grid.sync();

        for (u32 offset = 0; offset < num_tasks; offset += total) {
            if (threadIdx.x == 0) block_task_count = 0;
            __syncthreads();

            u32 my_task = offset + gtid;
            if (my_task < num_tasks) {
                Task task    = cur_bag[my_task];
                ParResult pr = exec_task(task, heap, &hchunk, heap_ptr);
                cascade(pr, task.cont_id, TASK_SLOT(task),
                        heap, &hchunk, heap_ptr, &fchunk,
                        block_task_buf, &block_task_count,
                        frames, frame_ptr, root_result);
            }
            __syncthreads();

            u32 count = block_task_count;
            if (threadIdx.x == 0 && count > 0)
                block_bag_base = atomicAdd(nxt_n, count);
            __syncthreads();

            for (u32 i = threadIdx.x; i < count; i += blockDim.x)
                nxt_bag[block_bag_base + i] = block_task_buf[i];
        }
        grid.sync();

        Task *tmp_bag = cur_bag; cur_bag = nxt_bag; nxt_bag = tmp_bag;
        u32  *tmp_n   = cur_n;   cur_n   = nxt_n;   nxt_n   = tmp_n;
    }
}

// ---------------------------------------------------------------------------
// Tree checksum (single-thread DFS)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    u32 depth   = argc > 1 ? (u32)atoi(argv[1]) : 20;
    u32 heap_gb = argc > 2 ? (u32)atoi(argv[2]) : (depth >= 20 ? 16 : 8);

    // DFS threshold: sort expansion stops creating tasks at this depth,
    // keeping the peak bag near 2^16 = 65536 (one task per useful thread).
    u32 dfs_depth = (depth > 16) ? (depth - 16) : 0;

    u64 heap_words = (u64)heap_gb << 28;
    u32 max_bag    = 1u << (depth - dfs_depth + 2 < 24 ? depth - dfs_depth + 2 : 24);
    u32 max_frames = depth <= 12 ? (1u << 20) :
                     depth <= 16 ? (1u << 24) : (1u << 27);

    u32 num_nodes  = (1u << (depth + 1)) - 1;
    u32 initial_hp = num_nodes * 2;

    fprintf(stderr, "sort(%u): heap=%uGB dfs_depth=%u max_bag=%u max_frames=%u\n",
            depth, heap_gb, dfs_depth, max_bag, max_frames);

    // Copy DFS depth to device constant memory.
    CHK(cudaMemcpyToSymbol(g_dfs_depth, &dfs_depth, sizeof(u32)));

    // Increase stack size for DFS recursion.
    CHK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    // Query cooperative-launch block count.
    int blocks_per_sm = 0;
    CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, eval_kernel, THREADS_PER_BLOCK,
        MAX_BLOCK_TASKS * sizeof(Task) + 8));
    int num_sms = 0;
    CHK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = blocks_per_sm * num_sms;
    fprintf(stderr, "SMs=%d blocks/SM=%d blocks=%d threads=%d\n",
            num_sms, blocks_per_sm, num_blocks, num_blocks * THREADS_PER_BLOCK);

    // --- Allocate ---

    u32 *dev_heap, *dev_heap_ptr;
    CHK(cudaMalloc(&dev_heap,     heap_words * 4));
    CHK(cudaMalloc(&dev_heap_ptr, sizeof(u32)));

    generate_tree<<<(num_nodes + 255) / 256, 256>>>(dev_heap, depth, num_nodes);
    CHK(cudaDeviceSynchronize());
    CHK(cudaMemcpy(dev_heap_ptr, &initial_hp, sizeof(u32), cudaMemcpyHostToDevice));

    Task *dev_bag_a, *dev_bag_b;
    CHK(cudaMalloc(&dev_bag_a, (u64)max_bag * sizeof(Task)));
    CHK(cudaMalloc(&dev_bag_b, (u64)max_bag * sizeof(Task)));

    u32 *dev_count_a, *dev_count_b;
    CHK(cudaMalloc(&dev_count_a, sizeof(u32)));
    CHK(cudaMalloc(&dev_count_b, sizeof(u32)));

    Frame *dev_frames;
    CHK(cudaMalloc(&dev_frames, (u64)max_frames * sizeof(Frame)));

    u32 *dev_frame_ptr, *dev_result;
    CHK(cudaMalloc(&dev_frame_ptr, sizeof(u32)));
    CHK(cudaMalloc(&dev_result,    sizeof(u32)));

    // --- Initialize ---

    Task initial_task;
    initial_task.pack    = PACK_TASK(FN_SORT, depth, 0, 0);
    initial_task.arg0    = 0;
    initial_task.arg1    = 0;
    initial_task.cont_id = NO_PARENT;

    CHK(cudaMemcpy(dev_bag_a, &initial_task, sizeof(Task), cudaMemcpyHostToDevice));
    u32 one = 1, zero = 0;
    CHK(cudaMemcpy(dev_count_a,   &one,  sizeof(u32), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dev_count_b,   &zero, sizeof(u32), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dev_frame_ptr, &zero, sizeof(u32), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dev_result,    &zero, sizeof(u32), cudaMemcpyHostToDevice));

    // Warmup.
    generate_tree<<<1, 1>>>(dev_heap, 0, 0);
    CHK(cudaDeviceSynchronize());

    // --- Launch ---

    void *args[] = {
        &dev_bag_a,    &dev_bag_b,
        &dev_count_a,  &dev_count_b,
        &dev_heap,     &dev_heap_ptr,
        &dev_frames,   &dev_frame_ptr,
        &dev_result
    };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    CHK(cudaLaunchCooperativeKernel(
        (void *)eval_kernel, num_blocks, THREADS_PER_BLOCK, args,
        MAX_BLOCK_TASKS * sizeof(Task) + 8));
    CHK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sort_time = (t1.tv_sec - t0.tv_sec) +
                       (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    // --- Results ---

    u32 result_root, final_hp, final_fp;
    CHK(cudaMemcpy(&result_root, dev_result,    sizeof(u32), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(&final_hp,    dev_heap_ptr,  sizeof(u32), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(&final_fp,    dev_frame_ptr, sizeof(u32), cudaMemcpyDeviceToHost));

    u32 *dev_sum;
    CHK(cudaMalloc(&dev_sum, sizeof(u32)));
    sum_tree<<<1, 1>>>(result_root, dev_heap, dev_sum);
    CHK(cudaDeviceSynchronize());
    u32 checksum;
    CHK(cudaMemcpy(&checksum, dev_sum, sizeof(u32), cudaMemcpyDeviceToHost));

    fprintf(stderr, "sort(%u) = %u  heap=%.2fGB  frames=%u  %.3fs\n",
            depth, checksum, final_hp * 4.0 / (1ULL << 30), final_fp, sort_time);

    // --- Cleanup ---

    cudaFree(dev_heap);      cudaFree(dev_heap_ptr);
    cudaFree(dev_bag_a);     cudaFree(dev_bag_b);
    cudaFree(dev_count_a);   cudaFree(dev_count_b);
    cudaFree(dev_frames);    cudaFree(dev_frame_ptr);
    cudaFree(dev_result);    cudaFree(dev_sum);
    return 0;
}
