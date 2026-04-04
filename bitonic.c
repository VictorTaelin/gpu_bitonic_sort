// bitonic.c — Sequential recursive bitonic sort
//
// Four mutually recursive functions on a binary tree:
//   sort(d, s, t)    — sort subtree t of depth d in direction s
//   flow(d, s, t)    — apply bitonic flow network
//   warp(d, s, a, b) — pairwise compare-swap across two subtrees
//   down(d, s, t)    — recurse flow on children
//
// Compilation: gcc -O3 -o bitonic_c bitonic.c
// Usage:       ./bitonic_c [depth=20] [heap_gb=16]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef uint32_t u32;
typedef uint64_t u64;

// ---------------------------------------------------------------------------
// Heap + tree encoding
//
// Tree nodes are 2 words each in a flat heap array.
// All allocations are 2-word aligned, so pointers are always even.
// We use bit 0 of word1 as the node/leaf tag:
//   Leaf: [value, 1]       (word1 is odd)
//   Node: [left,  right]   (word1 is even, since right is a valid pointer)
// This allows pointers to use the full 32-bit range (up to 16 GB heap).
// ---------------------------------------------------------------------------

static u32 *heap;
static u32 heap_ptr;

static inline int  is_node(u32 p) { return !(heap[p + 1] & 1); }
static inline u32  get_val(u32 p) { return heap[p]; }
static inline u32  tree_left(u32 p)  { return heap[p]; }
static inline u32  tree_right(u32 p) { return heap[p + 1]; }

static inline u32 make_leaf(u32 value) {
    u32 p = heap_ptr;
    heap_ptr += 2;
    heap[p]     = value;
    heap[p + 1] = 1;
    return p;
}

static inline u32 make_node(u32 left, u32 right) {
    u32 p = heap_ptr;
    heap_ptr += 2;
    heap[p]     = left;
    heap[p + 1] = right;
    return p;
}

// ---------------------------------------------------------------------------
// Tree generation
// Builds a complete binary tree of depth d with leaves [2^d-1 .. 0].
// ---------------------------------------------------------------------------

static u32 gen(u32 depth, u32 x) {
    if (depth == 0) {
        return make_leaf(x);
    }
    u32 l = gen(depth - 1, x * 2 + 1);
    u32 r = gen(depth - 1, x * 2);
    return make_node(l, r);
}

// ---------------------------------------------------------------------------
// Bitonic sort
// ---------------------------------------------------------------------------

static u32 warp(u32 depth, u32 dir, u32 a, u32 b) {
    if (depth == 0) {
        if (is_node(a) || is_node(b)) {
            return make_leaf(0);
        }
        u32 va = get_val(a);
        u32 vb = get_val(b);
        u32 do_swap = dir ^ (va > vb);
        u32 lo = do_swap ? vb : va;
        u32 hi = do_swap ? va : vb;
        return make_node(make_leaf(lo), make_leaf(hi));
    }
    if (!is_node(a) || !is_node(b)) {
        return make_leaf(0);
    }
    u32 wa = warp(depth - 1, dir, tree_left(a), tree_left(b));
    u32 wb = warp(depth - 1, dir, tree_right(a), tree_right(b));
    if (!is_node(wa) || !is_node(wb)) {
        return make_leaf(0);
    }
    return make_node(make_node(tree_left(wa), tree_left(wb)),
                     make_node(tree_right(wa), tree_right(wb)));
}

static u32 flow(u32 depth, u32 dir, u32 tree);

static u32 down(u32 depth, u32 dir, u32 tree) {
    if (depth == 0 || !is_node(tree)) {
        return tree;
    }
    u32 l = flow(depth - 1, dir, tree_left(tree));
    u32 r = flow(depth - 1, dir, tree_right(tree));
    return make_node(l, r);
}

static u32 flow(u32 depth, u32 dir, u32 tree) {
    if (depth == 0 || !is_node(tree)) {
        return tree;
    }
    u32 warped = warp(depth - 1, dir, tree_left(tree), tree_right(tree));
    return down(depth, dir, warped);
}

static u32 bsort(u32 depth, u32 dir, u32 tree) {
    if (depth == 0 || !is_node(tree)) {
        return tree;
    }
    u32 l = bsort(depth - 1, 0, tree_left(tree));
    u32 r = bsort(depth - 1, 1, tree_right(tree));
    return flow(depth, dir, make_node(l, r));
}

// ---------------------------------------------------------------------------
// Verification: in-order traversal should produce 0, 1, 2, ..., n-1
// ---------------------------------------------------------------------------

static u32 check_idx;
static int check_ok;

static void verify(u32 tree) {
    if (!is_node(tree)) {
        if (get_val(tree) != check_idx) {
            check_ok = 0;
        }
        check_idx++;
    } else {
        verify(tree_left(tree));
        verify(tree_right(tree));
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    u32 depth   = (argc > 1) ? (u32)atoi(argv[1]) : 20;
    u64 heap_gb = (argc > 2) ? (u64)atoi(argv[2]) : (depth >= 20 ? 16 : 8);
    u64 heap_words = (heap_gb << 30) / 4;

    fprintf(stderr, "sort(%u): heap = %llu GB\n",
            depth, (unsigned long long)heap_gb);

    double t0 = now();

    heap = (u32 *)malloc(heap_words * 4);
    if (!heap) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    heap_ptr = 0;

    u32 tree = gen(depth, 0);
    u32 sorted = bsort(depth, 0, tree);

    check_idx = 0;
    check_ok = 1;
    verify(sorted);
    u32 expected = 1u << depth;
    double elapsed = now() - t0;

    fprintf(stderr, "sort(%u) %s  heap = %.2f GB  %.3fs\n",
            depth,
            (check_ok && check_idx == expected) ? "PASS" : "FAIL",
            heap_ptr * 4.0 / (1ULL << 30),
            elapsed);

    free(heap);
    return 0;
}
