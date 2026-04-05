// ═══════════════════════════════════════════════════════════════════════════
// CRITICAL: THIS IS A COMPILER RUNTIME, NOT A ONE-OFF PROGRAM
// ═══════════════════════════════════════════════════════════════════════════
//
// The bitonic sort example is a TEST CASE for a general-purpose parallel
// evaluator. The end product is a COMPILER that auto-generates task
// functions (like fn_sort, fn_flow, fn_swap, etc.) from arbitrary Bend
// programs. Therefore:
//
// - DO NOT optimize by changing the algorithm's function signatures,
//   node structures, or recursive decomposition. Those are fixed by the
//   source language and the compiler will emit them mechanically.
//   e.g. changing swap(s,t) to warp(d,s,a,b) "saves wrapper nodes" but
//   that optimization can never be applied by a general compiler — it
//   requires understanding the specific semantics of warp. Wasted time.
//
// - DO optimize the RUNTIME: task scheduling, continuation management,
//   heap allocation, work distribution, idle handling, memory layout.
//   These improvements apply to ALL compiled programs, not just bitonic.
//
// - The sequential cutoff (SEQ_CUTOFF) is a runtime optimization: it
//   executes the SAME algorithm on a single thread instead of splitting
//   into tasks. This is legal because the compiler can emit both the
//   task version and a sequential version of each function.
//
// - A "depth hint" in task args (like swap's depth_hint) is metadata
//   the compiler can trivially emit. It doesn't change the algorithm.
//
// Summary: optimize the RUNTIME (scheduler, allocator, queues), not
// the ALGORITHM (function bodies, data representation, recursion shape).
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// CURRENT STATUS & PROFILING DATA (2026-04-04)
// ═══════════════════════════════════════════════════════════════════════════
//
// PERFORMANCE SUMMARY
// ===================
//  GPU: RTX 4090 (128 SMs, 24 GB, Ada/sm_89, ~1 TB/s DRAM, ~2.5 GHz)
//  Test: sort(20, 0, gen(20, 0)) — 1M elements, checksum 4027056128
//
//  C reference (gcc -O2, single-threaded):  ~2500 ms
//  GPU current best (48 blocks, cutoff=3):    ~92 ms  (27x)
//  GPU previous  (64 blocks, cutoff=4):     ~167 ms  (15x)
//
//  Target: 125 ms (20x) — ACHIEVED
//
// COMPILATION NOTES
// =================
//  - Default nvcc (no -arch flag) compiles for sm_52 PTX, then the
//    driver JIT-compiles to sm_89 native code at launch time.
//  - The JIT produces BETTER code than -arch=sm_89 static compilation!
//    sm_52 JIT: ~196ms.  sm_89 static: ~270ms.  Reason: the JIT
//    optimizes the recursive device functions (d_sort/d_flow/d_warp/
//    d_down) with less register spilling (68 bytes vs 84 bytes).
//  - Always compile WITHOUT -arch flag for best performance.
//  - Compile flags: nvcc -O3 -use_fast_math -o bitonic_gpu bitonic.cu
//
// HARDWARE PROFILING (ncu on RTX 4090)
// ====================================
//
//  Occupancy
//  ---------
//  Registers per thread: 130 (after JIT)
//  With 256-thread blocks: 130 * 256 = 33,280 regs per block
//  Max per SM: 65,536 → only 1 block per SM
//  Active warps: 8 out of max 48 → 16.67% occupancy
//  NOTE: need ≤128 regs for 2 blocks/SM (33% occupancy)
//        we are just 2 registers over the threshold!
//
//  Throughput utilization
//  ---------------------
//  DRAM throughput:   3.98% of peak (~40 GB/s of 1 TB/s)
//  L2 throughput:    32.67% of peak
//  L1 throughput:    15.34% of peak
//  SM compute:     11.98% of peak
//  → GPU is massively underutilized. Not bandwidth-bound.
//    The bottleneck is MEMORY LATENCY with too few warps to hide it.
//
//  Stall analysis
//  --------------
//  Long scoreboard (memory wait): 21.79× per instruction issued ← DOMINANT
//  Wait (barriers):         2.84×
//  Membar (__threadfence):     0.36×
//  Short scoreboard (ALU):     0.34×
//  → 97% of stalls are waiting for global memory loads.
//    With 8 warps per SM and ~22 cycles of stall per instruction,
//    the SM needs ~22 warps to stay busy. We have 8. The SM is
//    effectively idle ~64% of the time.
//
//  Cache hierarchy
//  ---------------
//  L1 hit: 170.85 GB   L1 miss: 82.75 GB   → 67.4% hit rate
//  L2 hit: 2.83B reqs   L2 miss: 216M reqs  → 92.9% hit rate
//  DRAM read: 1.41 GB   DRAM write: 7.53 GB
//  → Caching works well. Most data served from L1/L2.
//    But the 216M DRAM misses at ~500 cycles each dominate runtime.
//
//  Instruction mix
//  ---------------
//  Total instructions: 21.2 billion
//  ALU: 6.0B (28%)  FMA: 3.5B (17%)  LSU: 4.3B (20%)  Other: 7.4B (35%)
//
// TASK PROFILING (instrumented kernel, depth 20)
// ==============================================
//
//  Task breakdown (15.5M total tasks)
//  ----------------------------------
//   sort:            131,071  ( 0.8%)  — bulk sequential work via d_sort(4)
//   flow:           1,966,082  (12.7%)
//   swap:           7,929,855  (51.1%)  ← dominant task type
//   sort_cont:        65,535  ( 0.4%)
//   flow_after_swap:    983,041  ( 6.3%)
//   flow_join:        983,041  ( 6.3%)
//   swap_join:       3,473,407  (22.4%)
//
//   73.5% of all tasks are swap-related (swap + swap_join)
//   Each swap task does minimal work: read 4 heap entries, create 2
//   wrapper nodes, allocate 1 continuation, then SPLIT. Very high
//   overhead-to-work ratio compared to sort tasks (which call d_sort(4)
//   doing ~1100 operations).
//
//  Result types
//  ------------
//   VALUE: 10,027,008 (64.6%)  — immediate results
//   SPLIT:  4,521,983 (29.1%)  — binary splits → push one child
//   CALL:    983,041  (6.3%)  — tail calls (flow→swap)
//
//  Task sourcing (how threads get their next task)
//  -----------------------------------------------
//   Initial + SPLIT-keep + CALL-chain: 5,505,025 (35.4%)
//   Continuation saturation:      5,505,024 (35.4%)
//   Mailbox receive:          4,200,195 (27.0%)
//   Global queue:             321,788  (2.1%)
//
//   70.8% of tasks stay on the same thread (keep-half-of-split +
//   cont-saturation). Only 29.1% get pushed to other threads.
//   This creates long sequential chains on individual threads.
//
//  Push distance distribution (4.5M pushes)
//  -----------------------------------------
//   XOR dist 1:   1.96M (43.4%)  — nearest neighbor
//   XOR dist 2:    731K (16.2%)
//   XOR dist 4:    457K (10.1%)
//   XOR dist 8-128: 1.05M (23.3%)
//   Global queue:   322K  (7.1%)  — overflow
//   → XOR probing works well. 93% of pushes land within the block.
//
// BLOCK COUNT: THE CRITICAL FINDING
// =================================
//
//  With 768 blocks (current committed config):
//   - 69% of threads execute ZERO tasks
//   - 66.7% of WARPS are fully idle (0 threads active)
//   - 75% of BLOCKS have zero total work
//   - median tasks/thread = 0, max = 5286
//   - Only ~192 blocks (25%) actually participate
//
//  Root cause: with 130 regs/thread, only 1 block/SM fits.
//  128 SMs → 128 resident blocks. The other 640 blocks cycle in and
//  out but miss the work window. Tasks get distributed to resident
//  blocks via mailbox; non-resident blocks never see them.
//
//  With 64 blocks (all guaranteed resident, using 64 of 128 SMs):
//   - 0% idle threads (all 16,384 participate)
//   - min=22, median=794, max=5197 tasks/thread
//   - 167ms (15% faster than 768 blocks)
//
//  The optimal block count is 48-64 with IDLE_POLL_MASK=63-255.
//  Going below 64 (e.g., 32) uses too few SMs and hurts.
//  Going above 128 wastes blocks that never get work.
//
//  Block count sweep (depth 20, SEQ_CUTOFF=4):
//   32 blocks:  ~176ms
//   48 blocks:  ~167ms
//   64 blocks:  ~165ms  ← sweet spot
//   96 blocks:  ~188ms
//   128 blocks:  ~195ms
//   256 blocks:  ~193ms
//   768 blocks:  ~196ms
//
// SEQ_CUTOFF SWEEP (tested with both 768 and 64 blocks)
// =====================================================
//  Cutoff 3:  12.3M conts,  ~1020ms  (too many conts)
//  Cutoff 4:  5.5M conts,  ~167ms  ← optimal
//  Cutoff 5:  2.5M conts,  ~970ms  (too few parallel tasks)
//  Cutoff 6:  1.1M conts, ~1650ms
//  Cutoff 7:  479K conts, ~2600ms
//  Cutoff 8:  209K conts, ~4650ms
//
//  Cutoff 4 creates 2^16 = 65,536 sort tasks (one per subtree of
//  depth 4). Each d_sort(4) processes 16 elements with ~1100 memory
//  ops. Lower cutoffs create exponentially more continuations;
//  higher cutoffs starve the GPU of parallel tasks.
//
// IDLE_POLL_MASK SWEEP (how often idle threads check global queue)
// ===============================================================
//  mask=15  (every 16 spins):   ~815ms — severe contention
//  mask=63  (every 64 spins):   ~530ms
//  mask=127 (every 128 spins):  ~200ms
//  mask=255 (every 256 spins):  ~180ms
//  mask=511 (every 512 spins):  ~196ms ← committed
//  mask=1023:            ~195ms
//  mask=4095:            ~195ms
//
//  Low masks cause atomicCAS contention on global queue head.
//  The optimal mask depends on block count: fewer blocks can
//  tolerate lower masks. With 64 blocks, mask=63-255 is best.
//
// REGISTER PRESSURE EXPERIMENTS
// ============================
//  --maxrregcount=80:  80 regs,  3 blocks/SM, ~680ms  (heavy spilling)
//  --maxrregcount=96:  96 regs,  2 blocks/SM, ~310ms  (moderate spilling)
//  --maxrregcount=112: 112 regs, 2 blocks/SM, ~285ms
//  --maxrregcount=126: 126 regs, 2 blocks/SM, ~310ms
//  --maxrregcount=128: 128 regs, 2 blocks/SM, ~330ms
//  no limit:      130 regs, 1 block/SM, ~196ms  ← best
//
//  IMPORTANT: forcing lower register counts via compiler flags causes
//  spill stores/loads to local memory (backed by same global memory
//  we're already bottlenecked on). The spill traffic OVERWHELMS the
//  occupancy benefit. Must reduce registers through CODE RESTRUCTURING
//  (eliminating live variables) rather than compiler flags.
//
//  Register budget breakdown (estimated):
//   Result struct (returned by task fns):  ~23 regs (2 Tasks + tag + value)
//   GState pointers (12 × 64-bit ptrs):   ~24 regs
//   Task my_task (fn + ret + 4 args):    ~10 regs
//   Kernel loop locals (lp, le, tid, etc):  ~8 regs
//   Task function locals:          ~65 regs
//   Total:                ~130 regs
//
//  Savings needed: 2 registers (130 → 128) for 2 blocks/SM.
//  Potential savings:
//   - Eliminate Result struct (push 2nd task directly): ~15-20 regs
//   - Pack GState into single allocation + offsets:   ~20 regs
//   - Store GState base in shared memory:        ~2 regs
//  Any ONE of these would be sufficient. Combined: ~35 regs saved.
//
// DEPTH SCALING (GPU vs C)
// =======================
//  depth 10:  C=  1ms  GPU=  5.5ms  0.2x (GPU overhead dominates)
//  depth 12:  C=  4ms  GPU= 15.1ms  0.3x
//  depth 14:  C= 21ms  GPU= 34.8ms  0.6x
//  depth 16:  C= 98ms  GPU= 75.3ms  1.3x (crossover)
//  depth 18:  C=489ms  GPU=120.0ms  4.1x
//  depth 20:  C=2477ms  GPU=167.0ms  14.8x
//
//  GPU overhead is ~5ms baseline. Speedup improves with depth because
//  the parallelism (2^depth independent sort tasks) grows while the
//  per-task sequential work (d_sort(4) = fixed) stays constant.
//
// TIME BREAKDOWN (estimated for 167ms kernel)
// ============================================
//  Memory stalls:  ~120ms (72%)  — 216M DRAM misses + L2 latency
//  Task overhead:   ~20ms (12%)  — atomics, cont alloc, mailbox
//  Useful compute:  ~15ms  (9%)  — ALU/FMA instructions
//  Idle spinning:   ~12ms  (7%)  — threads waiting for work
//
// OPTIMIZATION ROADMAP (runtime-only, no algorithm changes)
// =========================================================
//
//  1. DONE: Sequential cutoff (SEQ_CUTOFF=3)
//     Cutoff 4→3: reduced from ~167ms to ~92ms. Creates 2^17=131K
//     sort tasks (vs 2^16=65K at cutoff 4). More parallelism.
//     Cutoff 2 is worse (130ms, too many conts: 28.7M).
//
//  2. DONE: Idle poll frequency tuning (mask=31)
//     With shared-memory mailbox, idle loops are faster (~30 cycles
//     vs ~200 for global memory). mask=31 optimal for 48 blocks.
//
//  3. DONE: Reduce block count to 48
//     With 12K threads, all participate. Sweet spot for work/thread.
//     32→109ms, 40→97ms, 48→92ms, 56→104ms, 64→117ms.
//
//  4. DONE: Shared-memory mailbox
//     Moved XOR-probe mailbox from global to shared memory.
//     ~30 cycle atomicCAS vs ~200 in global. 11 KB smem per block.
//     Side effect: reduced kernel registers 152→102 (GState smaller).
//     Performance impact: marginal (~2ms), but cleaner code.
//
//  5. DONE: Compact Cont struct (48→32 bytes)
//     Pack creation-time args (d, s) as u16 in header instead of
//     u64 args[]. Only 2 result slots instead of 4.
//     Saves 33% memory per cont. Performance impact: marginal.
//
//  6. DONE: Per-thread cont allocation (CONT_CHUNK=256)
//     Like heap allocation: thread-local bump, global atomicAdd
//     only every 256 conts. Saves ~2ms.
//
//  7. DONE: Host-side pre-splitting
//     Host walks sort tree to depth 5, creating 32 initial sort
//     tasks + 31 continuations. All blocks start with work from
//     the first cycle. Saves ~3ms vs single-root-task startup.
//
//  8. DONE: ALLOC_CHUNK=1024 (from 256)
//     Fewer global atomicAdds on heap_bump. ~2ms improvement.
//
// THINGS THAT DON'T WORK (tried and measured)
// ===========================================
//  - u32 tree encoding: 3× slower (implementation issues, not fundamental)
//  - -arch=sm_89: 35% slower than JIT from sm_52 (worse spilling)
//  - Increasing SWAP_SEQ_CUTOFF beyond 3: makes swap sequential on one
//    thread, catastrophically slow (cutoff 4→148ms, 5→210ms, 8→1300ms).
//    The swap MUST be parallelized through the task system.
//  - --maxrregcount=128: spill traffic overwhelms occupancy gain
//  - __launch_bounds__(256,2): 128 regs, no spill, but no perf gain
//    (48 blocks can't fill 2 blocks/SM on 128 SMs)
//  - __noinline__ on execute_task: 126 regs but stack overhead → -5%
//  - Cross-block mailbox probing: random writes to other blocks'
//    mailboxes. No benefit, just adds latency.
//  - Shorter XOR probe distance (d≤4): more global queue overflow,
//    contention kills performance (212ms at d≤4 vs 167ms at d≤128)
//  - More blocks (>48): work doesn't spread due to top-level flow
//    serialization. Even with presplit, 128 blocks → 248ms.
//  - Pre-splitting to more blocks: top-level flow operations are
//    serialized on whichever block resolves the continuation.
//    More blocks = more idle blocks waiting for top-level flow.
//  - Aggressive idle polling (mask=7): contention on global queue

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

