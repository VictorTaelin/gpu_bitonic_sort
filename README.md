# GPU Bitonic Sort

A GPU parallel evaluator for recursive functional programs, demonstrated with
bitonic sort. This is a **runtime test case** for a future compiler: the
algorithm is fixed, and the GPU implementation must compute it exactly as
written — no algorithmic changes allowed.

## The Algorithm

Bitonic sort expressed as pure recursive functions on binary trees:

```
warp(d, s, a, b):
  if d == 0: compare-and-swap leaves
  l = warp(d-1, s, left(a),  left(b))    // PARALLEL
  r = warp(d-1, s, right(a), right(b))   // PARALLEL
  return Node(Node(left(l),left(r)), Node(right(l),right(r)))

flow(d, s, t):
  if d == 0 or is_leaf(t): return t
  w = warp(d-1, s, left(t), right(t))
  return down(d, s, w)

down(d, s, t):
  if d == 0 or is_leaf(t): return t
  l = flow(d-1, s, left(t))              // PARALLEL
  r = flow(d-1, s, right(t))             // PARALLEL
  return Node(l, r)

sort(d, s, t):
  if d == 0 or is_leaf(t): return t
  l = sort(d-1, 0, left(t))              // PARALLEL
  r = sort(d-1, 1, right(t))             // PARALLEL
  return flow(d, s, Node(l, r))
```

Lines marked `PARALLEL` have two independent recursive calls. The GPU
evaluator runs these concurrently.

## Files

- **bitonic.js** — Reference implementation in JavaScript. Clear and simple.
- **bitonic.c** — Reference implementation in C. Single-threaded, ~2500ms for depth 20.
- **bitonic.cu** — GPU implementation in CUDA. ~38ms for depth 20 on RTX 4090 (**66× faster**).

## How the GPU Evaluator Works

Each function with a `PARALLEL` annotation compiles into three pieces:

- `seq_<name>` — Sequential recursive version (runs when work is small enough).
- `par_<name>_0` — "Splitter": returns SPLIT (two parallel sub-calls) or VALUE.
- `par_<name>_1` — "Joiner": continuation handler, combines child results.

The runtime uses a **SEED / GROW / WORK** architecture:

1. **SEED** (1 block): Starting from a single task, iteratively split until
   there's one task per GPU block.
2. **GROW** (all blocks): Each block splits its task in shared memory until
   it has one task per thread.
3. **WORK** (all blocks): Every thread runs its task sequentially. Results
   resolve continuations, which may produce new tasks for the next round.

This runs as a single cooperative kernel with `grid.sync()` between phases.
No host synchronization, no queues, no work stealing.

The sequential cutoff emerges naturally: 128 blocks × 256 threads = 32768
slots, so `sort(20)` bottoms out at `sort(5)` — each thread sorts a
32-element subtree.

## Building and Running

```bash
# JavaScript (depth 20, ~15s)
node bitonic.js

# C (depth 20, ~2500ms)
gcc -O2 -o bitonic bitonic.c
./bitonic

# CUDA (depth 20, ~38ms on RTX 4090)
nvcc -O3 -use_fast_math -o bitonic_gpu bitonic.cu
./bitonic_gpu 20
```

All three produce checksum `4027056128` for depth 20.

## Performance

On an RTX 4090 (128 SMs, 24GB VRAM), depth 20 (1M elements):

| Phase    | Time    |
|----------|---------|
| gen      | 0.4 ms  |
| sort     | 38 ms   |
| checksum | 0.8 ms  |

66× faster than single-threaded C.
