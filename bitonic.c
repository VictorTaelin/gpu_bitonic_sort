// bitonic.c — Sequential bitonic sort baseline
//
// Sorts 2^D integers using the bitonic sorting network.
// The array is initialized to [N-1, N-2, ..., 1, 0] and sorted
// ascending to [0, 1, 2, ..., N-1].
//
// Compilation: gcc -O3 -o bitonic_c bitonic.c
// Usage:       ./bitonic_c [depth=20]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef uint32_t u32;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Iterative bitonic sort on a flat array of n elements (n must be power of 2).
//
// The sorting network has D = log2(n) stages. Stage k uses comparison
// distance j = k/2, k/4, ..., 1. For each (k, j) step, element i is
// compared with partner i ^ j. The sort direction alternates every k
// elements: ascending when (i & k) == 0, descending otherwise.
static void bitonic_sort(u32 *arr, u32 n) {
    for (u32 k = 2; k <= n; k <<= 1) {
        for (u32 j = k >> 1; j > 0; j >>= 1) {
            for (u32 i = 0; i < n; i++) {
                u32 partner = i ^ j;
                if (partner <= i) {
                    continue;
                }
                int ascending = ((i & k) == 0);
                int needs_swap = ascending
                    ? (arr[i] > arr[partner])
                    : (arr[i] < arr[partner]);
                if (needs_swap) {
                    u32 tmp = arr[i];
                    arr[i] = arr[partner];
                    arr[partner] = tmp;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    u32 depth = (argc > 1) ? (u32)atoi(argv[1]) : 20;
    u32 n = 1u << depth;

    fprintf(stderr, "sort(%u): n = %u (%.0f MB)\n",
            depth, n, (double)n * sizeof(u32) / (1 << 20));

    u32 *arr = malloc((size_t)n * sizeof(u32));
    if (!arr) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (u32 i = 0; i < n; i++) {
        arr[i] = n - 1 - i;
    }

    double t0 = now();
    bitonic_sort(arr, n);
    double elapsed = now() - t0;

    int pass = 1;
    for (u32 i = 0; i < n; i++) {
        if (arr[i] != i) {
            pass = 0;
            break;
        }
    }

    fprintf(stderr, "sort(%u) %s  n=%u  %.3fs\n",
            depth, pass ? "PASS" : "FAIL", n, elapsed);

    free(arr);
    return 0;
}
