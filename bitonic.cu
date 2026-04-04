// bitonic.cu — GPU parallel bitonic sort
//
// Sorts 2^D integers using the bitonic sorting network on GPU.
// Each network step where the comparison distance (j) crosses block
// boundaries is a separate kernel launch on global memory. All steps
// where j fits within a block are batched into a single shared-memory
// kernel, eliminating redundant global memory round-trips.
//
// Memory: O(N) — just the array, no trees, no frames, no task bags.
// D=24 needs 64 MB, D=26 needs 256 MB, D=30 needs 4 GB.
//
// Compilation: nvcc -O3 -arch=sm_89 bitonic.cu -o bitonic_gpu
// Usage:       ./bitonic_gpu [depth=20]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

typedef uint32_t u32;

#define CHK(call) do {                                                 \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "%s:%d: CUDA error: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while (0)

#define BLOCK 256

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Compare-and-swap at distance j in global memory.
// Each thread handles one element. Only the thread with the lower index
// in each pair performs the comparison and swap.
__global__ __launch_bounds__(BLOCK)
void global_step(u32 * __restrict__ arr, u32 n, u32 j, u32 k) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    u32 partner = i ^ j;
    if (partner <= i || partner >= n) {
        return;
    }

    u32 ai = arr[i];
    u32 ap = arr[partner];
    bool ascending = ((i & k) == 0);
    bool needs_swap = ascending ? (ai > ap) : (ai < ap);
    if (needs_swap) {
        arr[i] = ap;
        arr[partner] = ai;
    }
}

// Batched compare-and-swap steps in shared memory.
// Handles all remaining steps where j < BLOCK for a given stage k.
// Each block loads a contiguous tile of BLOCK elements, performs multiple
// butterfly passes with __syncthreads() between them, then writes back.
__global__ __launch_bounds__(BLOCK)
void local_steps(u32 * __restrict__ arr, u32 n, u32 k, u32 start_j) {
    __shared__ u32 tile[BLOCK];

    u32 tid = threadIdx.x;
    u32 gid = blockIdx.x * BLOCK + tid;

    tile[tid] = (gid < n) ? arr[gid] : 0xFFFFFFFFu;
    __syncthreads();

    for (u32 j = start_j; j >= 1; j >>= 1) {
        u32 partner = tid ^ j;
        if (partner > tid) {
            bool ascending = ((gid & k) == 0);
            u32 lo = tile[tid];
            u32 hi = tile[partner];
            bool needs_swap = ascending ? (lo > hi) : (lo < hi);
            if (needs_swap) {
                tile[tid] = hi;
                tile[partner] = lo;
            }
        }
        __syncthreads();
    }

    if (gid < n) {
        arr[gid] = tile[tid];
    }
}

// Fill array with [n-1, n-2, ..., 1, 0].
__global__ __launch_bounds__(BLOCK)
void generate_reversed(u32 * __restrict__ arr, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = n - 1 - i;
    }
}

// Set *ok to 0 if any element is out of place.
__global__ __launch_bounds__(BLOCK)
void verify_sorted(const u32 * __restrict__ arr, u32 n, u32 *ok) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && arr[i] != i) {
        atomicExch(ok, 0u);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static double now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    u32 depth = (argc > 1) ? (u32)atoi(argv[1]) : 20;
    u32 n = 1u << depth;
    u32 blocks = (n + BLOCK - 1) / BLOCK;

    fprintf(stderr, "sort(%u): n = %u (%.0f MB)\n",
            depth, n, (double)n * sizeof(u32) / (1 << 20));

    u32 *dev_arr;
    CHK(cudaMalloc(&dev_arr, (size_t)n * sizeof(u32)));

    generate_reversed<<<blocks, BLOCK>>>(dev_arr, n);
    CHK(cudaDeviceSynchronize());

    double t0 = now();

    for (u32 k = 2; k <= n; k <<= 1) {
        // Global steps: j values that cross block boundaries
        for (u32 j = k >> 1; j >= BLOCK; j >>= 1) {
            global_step<<<blocks, BLOCK>>>(dev_arr, n, j, k);
        }
        // Local steps: all remaining j < BLOCK in one shared-memory pass
        u32 lj = ((k >> 1) < BLOCK) ? (k >> 1) : (BLOCK >> 1);
        local_steps<<<blocks, BLOCK>>>(dev_arr, n, k, lj);
    }

    CHK(cudaDeviceSynchronize());
    double elapsed = now() - t0;

    // Verification
    u32 *dev_ok;
    CHK(cudaMalloc(&dev_ok, sizeof(u32)));
    u32 one = 1;
    CHK(cudaMemcpy(dev_ok, &one, sizeof(u32), cudaMemcpyHostToDevice));
    verify_sorted<<<blocks, BLOCK>>>(dev_arr, n, dev_ok);
    CHK(cudaDeviceSynchronize());

    u32 pass;
    CHK(cudaMemcpy(&pass, dev_ok, sizeof(u32), cudaMemcpyDeviceToHost));

    fprintf(stderr, "sort(%u) %s  n=%u  %.4fs\n",
            depth, pass ? "PASS" : "FAIL", n, elapsed);

    cudaFree(dev_arr);
    cudaFree(dev_ok);
    return 0;
}
