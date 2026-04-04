#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/mman.h>

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

typedef uint64_t Tree;

static uint64_t* heap;
static uint64_t heap_pos = 0;
static uint64_t heap_max = 0;

static inline Tree Leaf(uint32_t v) {
  return (uint64_t)v << 1;
}

static inline Tree Node(Tree l, Tree r) {
  uint64_t loc = heap_pos;
  heap[heap_pos++] = l;
  heap[heap_pos++] = r;
  if (heap_pos > heap_max) {
    heap_max = heap_pos;
  }
  return (loc << 1) | 1;
}

static inline uint32_t is_node(Tree t) {
  return t & 1;
}

static inline uint32_t get_val(Tree t) {
  return (uint32_t)(t >> 1);
}

static inline Tree get_l(Tree t) {
  uint64_t loc = t >> 1;
  return heap[loc];
}

static inline Tree get_r(Tree t) {
  uint64_t loc = t >> 1;
  return heap[loc + 1];
}

Tree gen(uint32_t d, uint32_t x) {
  if (d == 0) {
    return Leaf(x);
  } else {
    // PARALLEL:
    Tree xl = gen(d - 1, x * 2 + 1);
    Tree xr = gen(d - 1, x * 2);
    return Node(xl, xr);
  }
}

Tree warp_swap(uint32_t c, uint32_t av, uint32_t bv) {
  if (c == 0) {
    return Node(Leaf(av), Leaf(bv));
  } else {
    return Node(Leaf(bv), Leaf(av));
  }
}

Tree warp(uint32_t d, uint32_t s, Tree a, Tree b) {
  if (d == 0) {
    uint32_t av = get_val(a);
    uint32_t bv = get_val(b);
    return warp_swap(s ^ (av > bv ? 1 : 0), av, bv);
  } else {
    // PARALLEL:
    Tree wa = warp(d - 1, s, get_l(a), get_l(b));
    Tree wb = warp(d - 1, s, get_r(a), get_r(b));
    return Node(Node(get_l(wa), get_l(wb)), Node(get_r(wa), get_r(wb)));
  }
}

Tree flow(uint32_t d, uint32_t s, Tree t);
Tree down(uint32_t d, uint32_t s, Tree t);

Tree flow(uint32_t d, uint32_t s, Tree t) {
  if (d == 0) {
    return t;
  } else if (!is_node(t)) {
    return t;
  } else {
    Tree warped = warp(d - 1, s, get_l(t), get_r(t));
    return down(d, s, warped);
  }
}

Tree down(uint32_t d, uint32_t s, Tree t) {
  if (d == 0) {
    return t;
  } else if (!is_node(t)) {
    return t;
  } else {
    // PARALLEL:
    Tree tl = flow(d - 1, s, get_l(t));
    Tree tr = flow(d - 1, s, get_r(t));
    return Node(tl, tr);
  }
}

Tree sort(uint32_t d, uint32_t s, Tree t) {
  if (d == 0) {
    return t;
  } else if (!is_node(t)) {
    return t;
  } else {
    // PARALLEL:
    Tree sl = sort(d - 1, 0, get_l(t));
    Tree sr = sort(d - 1, 1, get_r(t));
    return flow(d, s, Node(sl, sr));
  }
}

static void checksum_go(Tree t, uint32_t* result) {
  if (!is_node(t)) {
    *result = (uint32_t)((uint32_t)(*result * 31) + (uint32_t)get_val(t));
  } else {
    checksum_go(get_l(t), result);
    checksum_go(get_r(t), result);
  }
}

uint64_t checksum(Tree t) {
  uint32_t result = 0;
  checksum_go(t, &result);
  return (uint64_t)result;
}

int main(void) {
  size_t heap_elems = (uint64_t)1 << 42;
  size_t heap_bytes = heap_elems * sizeof(uint64_t);
  heap = (uint64_t*)mmap(NULL, heap_bytes,
    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (heap == MAP_FAILED) {
    fprintf(stderr, "mmap failed\n");
    return 1;
  }
  printf("%" PRIu64 "\n", checksum(sort(20, 0, gen(20, 0))));
  fprintf(stderr, "heap_used: %.2f GiB\n", (double)(heap_max * sizeof(uint64_t)) / (1024.0 * 1024.0 * 1024.0));
  munmap(heap, heap_bytes);
  return 0;
}
