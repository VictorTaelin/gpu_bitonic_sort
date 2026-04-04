// Our goal is to implement a general-purpose parallel evaluator for a pure
// functional language in CUDA. In order to start our work, we designed the
// bitonic sort algorithm above, which serves as an initial example. We'll start
// by implementing a .cu file that runs this example hardcoded. Once that works,
// we'll turn it into a general recursive function evaluator. That .cu file must
// compute that recursive algorithm AS IS, no algorithm changes are allowed (the
// algorithm is fixed). To do so, we'll implement a global task buffer. A task
// is a struct that includes a function id, N arguments (where each argument is
// a Tree), and a return index, which is a heap index where we write its results
// to. Task functions return Results, which are either a VALUE or a SPLIT. A
// VALUE is an immediate result that must be written to the target location. A
// SPLIT is a request to spread the task into two exact tasks. This SPLIT
// generates two tasks, plus a Continuation. A Continuation is exactly the same
// thing as a Task, except some of its arguments are NULL, meaning they are
// pending. Because of that, it can't be picked for execution yet, and it is
// stored in a separate buffer. When a VALUE is returned by a task function,
// other than writing the result to the target location (which is a
// continuation), we also check if that continuation is saturated. If so, we
// convert it into a proper task. Example:
//
//Tree sort(u32 d, u32 s, Tree t) {
// if (d == 0) {
//  return t;
// } else if (!is_node(t)) {
//  return t;
// } else {
//  // PARALLEL:
//  Tree sl = sort(d - 1, 0, get_l(t));
//  Tree sr = sort(d - 1, 1, get_r(t));
//  return flow(d, s, Node(sl, sr));
// }
//}
//
// Here, there's one PARALLEL annotation. This function would, thus, be
// "manually compiled" to something like:
//
//Result sort_par(u32 d, u32 s, Tree t) {
// if (d == 0) {
//  return Value(t);
// } else if (!is_node(t)) {
//  return Value(t);
// } else {
//  Cont co = new_cont(sort_par_0, [d, s, 0, 0]);
//  Task t0 = new_task(sort_par, co+2, [d-1, 0, get_l(t)]);
//  Task t1 = new_task(sort_par, co+3, [d-1, 1, get_r(t)]);
//  return Split(t0, t1, co);
// }
//}
//
//void sort_par_0(...) {
// ...
//}
// 
// we'll launch a single kernel with fixed grid of 256x256 threads.
//
// the kernel will work kinda like this (just a draft):
// 
// def kernel(...):
//  loop:
//   task = tasks[gid] # ← gid is the global thread id
//   match execute(task):
//    case Value{jx}:
//     resolve(task, x)
//    case Split{ta, tb, co}:
//     tasks[gid] = ta
//     push_task(tb)
//     push_cont(fn)
// 
// - how node allocation works? to avoid contention, we should probably split
// the global heap (a 16 GB buffer) into slices. each thread then allocates on
// their own slice, seamlessly. that is not very flexible, but is as efficient
// as it gets. is that needed though? it depends on whether there is a faster
// solution to global allocation in CUDA. that should be investigated later.
// 
// - how 'push_task()' works? ideally, it should move a task to a free thread.
// in the context of a block, this should check if progressively further apart
// threads are available, by using the shared memory. i.e., the block keeps a
// local copy of a 256-items slice of the global task buffer. when a thread does
// a push_task(), it just attempts to write to 8 consecutively further apart
// local neighbor indices, using xor distance. only if it fails, it pushes to
// the global task buffer, with an atomicInc.
// 
// - 'conts()' work by just appending to the global cont buffer with an atomicInc
// 
// every once in a while (when there are enough tasks pushed to the outside
// buffer), the kernel should call a grid sync operation, allowing its outgoing
// effects to be seen by other blocks...
// 
// Q: does that make sense? anything you'd like to comment on or add?
// A: yes - the basic model makes sense. it is essentially a persistent-kernel
// runtime with continuations, and that is a reasonable way to preserve the
// recursive semantics on the gpu. the main things i'd add are:
//
// - the biggest cuda caveat is grid-wide sync. `grid.sync()` only works with a
//  cooperative launch, and only if all blocks in the grid are resident at the
//  same time. so "256x256 threads" may be fine, but "256 blocks of 256
//  threads" is not always legal for cooperative sync on a given gpu. in
//  practice, i'd usually size the grid to the maximum resident blocks and let
//  it run as a persistent kernel.
//
// - i'd separate "worker slots" from "queues". using `tasks[gid]` as the
//  currently owned task is fine, but for `push_task()` a per-block ring buffer
//  in shared memory is usually simpler and more robust than probing neighbor
//  lanes. local pushes go to the shared queue; overflow goes to a global queue;
//  idle workers first drain local, then global.
// ↑ are you *sure* this will be faster than just invasively pushing to neighbor
// lanes? I'm not so sure of that. also an additional structure will use
// precious shared memory space which is scarce... I think you're confused?
//
// - continuations should probably have an explicit pending-count or ready-mask.
//  when a child returns a value, it writes into its slot, does a fence if
//  needed, then `atomicSub(pending, 1)`. the thread that observes the count
//  going from 1 to 0 is the one that turns the continuation into a runnable
//  task. that avoids races when both children finish together.
// ↑ sure but that should be a single decreasing u8 to avoid wasting space.
//
// - for allocation, a per-thread heap slice is fast but can waste a lot of
//  memory and can fail badly on skewed workloads. a good compromise is a
//  global bump allocator that hands out large chunks to blocks or warps, and
//  then each block does cheap local bump allocation inside its current chunk.
//  since this is a pure evaluator, an arena allocator with no frees during the
//  kernel is probably ideal.
// ↑ let's use a global bump allocator that hands pages to *blocks*. a block
// then reserves a page to it. but I'd still split the page evenly among local
// block threads, since the workload / allocation within a block is balanced.
//
// - if tasks and conts are appended with `atomicInc`, make them bounded ring
//  buffers and define the overflow policy up front. otherwise the hard part
//  becomes not execution but queue exhaustion.
// ↑ what do you mean? not sure that will be a problem in the kernel lifetime
//
// - you will also need an explicit termination condition. "all queues empty" is
//  not enough unless you also know that no thread is still executing a task
//  and no continuation can still become ready. usually this is handled with a
//  global in-flight counter or equivalent quiescence detection.
// ↑ it terminates when the root continuation returns, obviously...
//
// - memory ordering matters. when publishing a task or continuation to another
//  block, write the payload first, then fence, then publish the queue index or
//  ready flag. otherwise another block can observe a partially initialized
//  entry.
// ↑ what do you mean? that is completely moronic. why would you add unecessary
// syncs? you can just spin until the whole data is there. wtf man.
//
// overall: yes, the design is sound, and for this kind of evaluator a
// split/value/continuation runtime is exactly the direction i'd take. the two
// parts i'd nail down first are (1) cooperative-launch/grid-sync constraints
// and (2) the exact atomic protocol for resolving continuations, because those
// are the places where these systems usually break.

// follow up answers:
//
// 1. neighbor probing vs ring buffer:
//  fair enough — you're right that a shared-memory ring buffer eats into a
//  scarce resource, and for a 256-thread block you'd need at least 256 *
//  sizeof(Task) bytes for the queue alone. probing neighbor lanes via xor
//  distance is actually fine *if* you use a simple flag word per slot (e.g.
//  a u32 in shared memory: 0 = idle, 1 = busy). the pushing thread does
//  `atomicExch(&slot_flag[tid ^ dist], 1)` for dist in {1,2,4,...,128}; if
//  the old value was 0, the slot was free and you write the task there.
//  that's 256 bytes of shared mem for the flags, which is negligible. the
//  only real concern is fairness — if the tree is left-heavy, you'll
//  saturate nearby lanes and spill to global more often — but for a balanced
//  workload like bitonic sort that won't matter. so yes, probing is fine
//  here. the global overflow queue is still needed as a fallback, and that
//  one should be a simple atomic-counter-indexed array in global memory.
//
// 2. pending count as u8:
//  agreed, a u8 is plenty. for a binary split it goes 2 → 1 → 0. use
//  `atomicSub` on the u8 (CUDA supports 32-bit atomics at minimum, so
//  you'll want to pack the u8 into a u32 and use `atomicSub(&word, 1)` on
//  the low byte, or just use a u32 for the counter and accept 3 bytes of
//  padding — the continuation struct will likely be padded to 4-byte
//  alignment anyway). the thread that transitions the counter from 1 to 0
//  owns the continuation and converts it to a task. no race possible.
//
// 3. block-level bump allocator:
//  sounds good. concretely: a single global `u32 *g_heap_bump` starts at 0.
//  when a block needs a page, one thread (e.g. lane 0) does
//  `atomicAdd(g_heap_bump, PAGE_SIZE)` and broadcasts the base address via
//  shared memory. each thread in the block then owns a slice of
//  `PAGE_SIZE / blockDim.x` words. for a 64 KB page and 256 threads, that's
//  256 bytes per thread per page — enough for a handful of Node allocations
//  before you need a new page. the thread bumps a local pointer within its
//  slice. when it runs out, the block collectively grabs another page. this
//  is essentially zero-contention allocation for the common case.
//
// 4. queue exhaustion:
//  you're probably right that for a single bitonic sort invocation it won't
//  be a problem — the task count is bounded by O(n log²n) and the
//  continuation count by the same. but when you generalize to arbitrary
//  recursive programs, the queue *can* blow up if the recursion is deep and
//  wide. at that point you'd want a bounded buffer with a back-pressure
//  mechanism (e.g. if the global queue is full, the thread executes the
//  second child itself instead of pushing it — effectively falling back to
//  sequential recursion). but that's a later concern; for now, just size the
//  buffers generously and move on.
//
// 5. termination:
//  right — the root task writes to a designated "root result" slot. you can
//  have all threads poll a `g_done` flag; the thread that writes the root
//  result sets `g_done = 1` (with a `__threadfence()` before it, to ensure
//  the result is visible). every other thread, when it finds itself idle
//  (no local task, no neighbor task, no global task), checks `g_done` and
//  exits the loop if set. cheap and sufficient.
//
// 6. memory ordering:
//  ok, let me be more precise. the concern is this: thread A in block 0
//  writes a task struct to global memory (say 4 words: fn_id, arg0, arg1,
//  arg2) and then increments the global queue tail. thread B in block 1
//  sees the new tail and reads the task. on NVIDIA GPUs, global memory
//  stores from different blocks are *not* guaranteed to be visible in
//  program order without a fence. thread B could read stale data for some
//  of the 4 words. this is not theoretical — it happens in practice on
//  large grids. the fix is trivial: thread A does `__threadfence()` between
//  writing the payload and publishing the index. that's one instruction, not
//  a sync. alternatively, you can pack the entire task into a single 64-bit
//  atomic store if it fits, which gives you atomicity for free.
//  "spinning until the data is there" does NOT work because the reads can
//  return stale L2 cache lines indefinitely — there is no forward progress
//  guarantee on cache coherence timing between blocks. you need the fence
//  on the *writer* side. this is a one-liner and costs ~20 cycles; it is
//  not optional.
//
// with all that settled, here's the concrete plan for the first .cu file:
//
// DATA STRUCTURES
// ===============
//
// Tree: a u64. either a Leaf(value) or a Node(left, right).
//  (omitted)
//  actually, simpler: Node stores a single heap index; left is at heap[idx],
//  right is at heap[idx+1]. so:
//  - if Node:  bits [31:0] = u32 heap index (left = heap[idx], right = heap[idx+1])
//  - if Leaf:  bits [31:0] = u32 value
//  fits in 64 bits. clean.
//
// Task: 32 bytes
//  - u16 fn_id     (function to execute)
//  - u16 padding
//  - u32 ret_ix     (heap index to write result to; this is inside a Cont)
//  - u64 args[3]    (up to 3 Tree arguments; unused args = 0)
//
// Continuation: 40 bytes
//  - u16 fn_id     (continuation function to call when saturated)
//  - u8 pending    (number of pending child results; starts at 2)
//  - u8 padding
//  - u32 ret_ix     (where *this* continuation's result goes; could be
//             another continuation or the root result slot)
//  - u64 args[4]    (arguments; initially some are 0 = pending)
//  the heap indices written to by child tasks point into this args[] array.
//  specifically, for a binary split, child 0 writes to &cont.args[2] and
//  child 1 writes to &cont.args[3] (or whichever slots are designated NULL).
//
// Result: returned by task functions
//  - tag: VALUE or SPLIT
//  - if VALUE: u64 value (a Tree)
//  - if SPLIT: Task t0, Task t1, Continuation co
//
// GLOBAL MEMORY LAYOUT
// ====================
//  - g_heap:    u64[HEAP_SIZE]    — tree node storage + cont arg slots
//  - g_heap_bump: u32          — global bump pointer (in u64 units)
//  - g_tasks:   Task[TASK_BUF_SIZE]  — global overflow task queue
//  - g_task_head: u32          — consumer index (atomicInc)
//  - g_task_tail: u32          — producer index (atomicInc)
//  - g_conts:   Cont[CONT_BUF_SIZE] — continuation storage
//  - g_cont_bump: u32          — next free continuation index
//  - g_done:    u32          — set to 1 when root result is ready
//  - g_result:   u64          — root result (final sorted tree)
//
// SHARED MEMORY (per block)
// =========================
//  - s_slot_flag: u32[256] — 0 = idle, 1 = busy (for neighbor probing)
//  - s_page_base: u32    — current heap page base for this block
//  - s_page_used: u32    — how much of the current page is consumed
//  total: ~1028 bytes. very manageable.
//
// KERNEL PSEUDOCODE
// =================
//
// (omitted)
//
// FUNCTION TABLE
// ==============
// For the bitonic sort example, we have these "compiled" functions:
//
//  FN_SORT = 0  — sort(d, s, t)
//  FN_FLOW = 1  — flow(d, s, t)
//  FN_SWAP = 2  — swap(s, t)
//  FN_SORT_CONT = 3 — sort continuation: receives (d, s, sl, sr) → calls flow
//  FN_FLOW_CONT = 4 — flow continuation: receives (d, s, fl, fr) → makes Node
//  FN_SWAP_CONT = 5 — swap continuation: receives (s, sl, sr) → makes Node
//
// Each FN_*_CONT is a continuation function that fires once both children land.
// It typically does no further splitting — it just combines results and returns
// a VALUE (or chains into another task).
//
// Let's now write the actual code.
// 
// note: include a main() that runs and outputs the same result as bitoni.c,
// hopefully *much* faster by exploiting parallelism.

