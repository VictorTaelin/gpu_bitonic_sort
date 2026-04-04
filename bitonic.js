// bitonic.js — Emulation of GPU-like parallel bitonic sort
//
// Models three Par-annotated functions:
//   sort:     sa & sb = sort(d-1,0,a) & sort(d-1,1,b)  →  then flow(d)
//   down_go:  fa & fb = flow(d-1,s,a) & flow(d-1,s,b)  →  then node{fa,fb}
//   warp_go:  wa & wb = warp(d-1,s,aa,ba) & warp(d-1,s,ab,bb) → then zip(wa,wb)
//
// Each "step" executes all bag tasks in parallel.  When a task hits Par,
// it suspends: two sub-tasks enter the bag for the next step, and a
// continuation frame records how to resume.  If the bag would exceed
// MAX_PAR, the Par is computed sequentially (DFS fallback).
//
// Usage: node bitonic.js [depth=8] [max_parallel=65536]

var DEPTH = parseInt(process.argv[2] || "12", 10);
var MAX_PAR = parseInt(process.argv[3] || "65536", 10);

// ── Tree ────────────────────────────────────────────────────
var L = v => ({ $: 0, v: v >>> 0 });
var N = (a, b) => ({ $: 1, a, b });

function gen(d, x) {
  return d === 0 ? L(x) : N(gen(d - 1, x * 2 + 1), gen(d - 1, x * 2));
}

function tsum(t) {
  return t.$ === 0 ? t.v : (tsum(t.a) + tsum(t.b)) >>> 0;
}

// ── Warp primitives (always sequential) ─────────────────────
var warpLeafCalls = 0;

function warp_leaf(s, a, b) {
  warpLeafCalls++;
  if (a.$ | b.$) return L(0);
  var c = s ^ (a.v > b.v ? 1 : 0);
  return c ? N(L(b.v), L(a.v)) : N(L(a.v), L(b.v));
}

function warp_zip(a, b) {
  if (a.$ !== 1 || b.$ !== 1) return L(0);
  return N(N(a.a, b.a), N(a.b, b.b));
}

// ── Sequential (DFS) implementations ────────────────────────
function warp_s(d, s, a, b) {
  if (d === 0) return warp_leaf(s, a, b);
  if (a.$ !== 1 || b.$ !== 1) return L(0);
  return warp_zip(warp_s(d - 1, s, a.a, b.a), warp_s(d - 1, s, a.b, b.b));
}

function flow_s(d, s, t) {
  if (d === 0 || t.$ !== 1) return t;
  return down_s(d, s, warp_s(d - 1, s, t.a, t.b));
}

function down_s(d, s, t) {
  if (d === 0 || t.$ !== 1) return t;
  return N(flow_s(d - 1, s, t.a), flow_s(d - 1, s, t.b));
}

function sort_s(d, s, t) {
  if (d === 0 || t.$ !== 1) return t;
  return flow_s(d, s, N(sort_s(d - 1, 0, t.a), sort_s(d - 1, 1, t.b)));
}

// ── Par result: V(value) or Par(left, right, resume) ────────
var V = v => ({ _: 0, v });

// ── Par-aware implementations ───────────────────────────────
function sort_p(d, s, t) {
  if (d === 0 || t.$ !== 1) return V(t);
  return {
    _: 1,
    lf: "sort", la: [d - 1, 0, t.a],
    rf: "sort", ra: [d - 1, 1, t.b],
    k: (sa, sb) => flow_p(d, s, N(sa, sb)),
    tag: "S" + d,
  };
}

function warp_p(d, s, a, b) {
  if (d === 0) return V(warp_leaf(s, a, b));
  if (a.$ !== 1 || b.$ !== 1) return V(L(0));
  return {
    _: 1,
    lf: "warp", la: [d - 1, s, a.a, b.a],
    rf: "warp", ra: [d - 1, s, a.b, b.b],
    k: (wa, wb) => V(warp_zip(wa, wb)),
    tag: "W" + d,
  };
}

function flow_p(d, s, t) {
  if (d === 0 || t.$ !== 1) return V(t);
  var wr = warp_p(d - 1, s, t.a, t.b);
  if (wr._ === 0) return down_p(d, s, wr.v);
  var wk = wr.k;
  return {
    _: 1, lf: wr.lf, la: wr.la, rf: wr.rf, ra: wr.ra,
    k: (wa, wb) => { var w = wk(wa, wb); return down_p(d, s, w.v); },
    tag: "F" + d,
  };
}

function down_p(d, s, t) {
  if (d === 0 || t.$ !== 1) return V(t);
  return {
    _: 1,
    lf: "flow", la: [d - 1, s, t.a],
    rf: "flow", ra: [d - 1, s, t.b],
    k: (fa, fb) => V(N(fa, fb)),
    tag: "D" + d,
  };
}

// ── Dispatch ────────────────────────────────────────────────
function exec_p(fn, a) {
  if (fn === "sort") return sort_p(a[0], a[1], a[2]);
  if (fn === "flow") return flow_p(a[0], a[1], a[2]);
  return warp_p(a[0], a[1], a[2], a[3]);
}

function exec_s(fn, a) {
  if (fn === "sort") return sort_s(a[0], a[1], a[2]);
  if (fn === "flow") return flow_s(a[0], a[1], a[2]);
  return warp_s(a[0], a[1], a[2], a[3]);
}

// ── Execution engine ────────────────────────────────────────
var frames = new Map();
var fid = 0;
var done = null;
var bag = [];
var nxt = [];
var step = 0;
var totalDFS = 0;
var maxBag = 0;
var totalConts = 0;
var totalTasks = 0;
var stepEvents = [];

function mkFrame(k, pid, ps, tag) {
  var id = fid++;
  frames.set(id, { k, pid, ps, s: [null, null], n: 2, tag: tag || null });
  return id;
}

function deliver(cid, slot, val) {
  if (cid < 0) { done = val; return; }
  var f = frames.get(cid);
  f.s[slot] = val;
  if (--f.n === 0) {
    frames.delete(cid);
    totalConts++;
    if (f.tag) stepEvents.push(f.tag);
    handle(f.k(f.s[0], f.s[1]), f.pid, f.ps);
  }
}

function handle(r, pid, ps) {
  if (r._ === 0) {
    deliver(pid, ps, r.v);
  } else if (nxt.length + 2 <= MAX_PAR) {
    var id = mkFrame(r.k, pid, ps, r.tag);
    nxt.push({ fn: r.lf, a: r.la, cid: id, cs: 0 });
    nxt.push({ fn: r.rf, a: r.ra, cid: id, cs: 1 });
  } else {
    totalDFS++;
    handle(r.k(exec_s(r.lf, r.la), exec_s(r.rf, r.ra)), pid, ps);
  }
}

// ── Visualization ───────────────────────────────────────────
var BAR_W = 32;
var maxCon = 0;
var log = [];

function tick() {
  step++;
  nxt = [];
  stepEvents = [];

  // Task type counts
  var bagS = 0, bagW = 0, bagF = 0;
  var groups = {};
  for (var i = 0; i < bag.length; i++) {
    var t = bag[i];
    if (t.fn === "sort") bagS++;
    else if (t.fn === "warp") bagW++;
    else bagF++;
    var key = t.fn[0].toUpperCase() + t.a[0];
    groups[key] = (groups[key] || 0) + 1;
  }
  if (bag.length > maxBag) maxBag = bag.length;
  totalTasks += bag.length;

  // Execute all tasks
  for (var i = 0; i < bag.length; i++) {
    var t = bag[i];
    handle(exec_p(t.fn, t.a), t.cid, t.cs);
  }

  // Count pending frames by type
  var conS = 0, conW = 0, conF = 0, conD = 0;
  var conDetail = {};
  frames.forEach(function(f) {
    if (!f.tag) return;
    conDetail[f.tag] = (conDetail[f.tag] || 0) + 1;
    var c = f.tag[0];
    if (c === "S") conS++;
    else if (c === "W") conW++;
    else if (c === "F") conF++;
    else if (c === "D") conD++;
  });
  var conTotal = frames.size;
  if (conTotal > maxCon) maxCon = conTotal;

  // Cascade events
  var evCounts = {};
  for (var i = 0; i < stepEvents.length; i++) {
    evCounts[stepEvents[i]] = (evCounts[stepEvents[i]] || 0) + 1;
  }

  log.push({
    step: step, size: bag.length,
    bagS: bagS, bagW: bagW, bagF: bagF,
    conS: conS, conW: conW, conF: conF, conD: conD, conTotal: conTotal,
    conDetail: conDetail,
    groups: groups, events: evCounts,
  });

  bag = nxt;
}

function makeBar(w, max, parts) {
  if (max === 0) return ".".repeat(w);
  var bar = "";
  for (var i = 0; i < parts.length; i++) {
    if (parts[i].n === 0) continue;
    var len = Math.max(1, Math.round(parts[i].n / max * w));
    bar += parts[i].c.repeat(len);
  }
  while (bar.length < w) bar += ".";
  return bar.slice(0, w);
}

function render() {
  var W = BAR_W;

  var hdr1 = "  step   bag  task bag" + " ".repeat(W - 8) + "  cont  cont buf" + " ".repeat(W - 8) + "  tasks";
  console.log("");
  console.log(hdr1);

  for (var i = 0; i < log.length; i++) {
    var e = log[i];

    var bagBar = makeBar(W, maxBag, [
      { c: "S", n: e.bagS },
      { c: "W", n: e.bagW },
      { c: "F", n: e.bagF },
    ]);

    var conBar = makeBar(W, maxCon, [
      { c: "S", n: e.conS },
      { c: "F", n: e.conF },
      { c: "D", n: e.conD },
      { c: "W", n: e.conW },
    ]);

    // Compact task breakdown: S7:2, W0:128, etc.
    var bkParts = [];
    var gkeys = Object.keys(e.groups).sort();
    for (var j = 0; j < gkeys.length; j++) {
      bkParts.push(gkeys[j] + ":" + e.groups[gkeys[j]]);
    }
    var bk = bkParts.join(" ");

    // Cascade events
    var evStr = "";
    var evKeys = Object.keys(e.events);
    if (evKeys.length > 0) {
      var parts = [];
      for (var j = 0; j < evKeys.length; j++) {
        parts.push(e.events[evKeys[j]] + "x " + evKeys[j]);
      }
      evStr = "  <- " + parts.join(", ");
    }

    console.log(
      "  " + String(e.step).padStart(4) +
      "  " + String(e.size).padStart(5) +
      "  " + bagBar +
      "  " + String(e.conTotal).padStart(4) +
      "  " + conBar +
      "  " + bk + evStr
    );
  }
}

// ── Main ────────────────────────────────────────────────────
console.log("");
console.log("  Parallel Bitonic Sort \u2014 sort(" + DEPTH + "), max_par=" + MAX_PAR);

var tree = gen(DEPTH, 0);
console.log("  Tree: 2^" + DEPTH + " = " + (1 << DEPTH) + " leaves");

warpLeafCalls = 0;
var t0 = Date.now();
var seqResult = sort_s(DEPTH, 0, tree);
var seqTime = Date.now() - t0;
var expected = tsum(seqResult);
var seqLeafs = warpLeafCalls;

// Parallel emulation
warpLeafCalls = 0;
var t1 = Date.now();

var r0 = exec_p("sort", [DEPTH, 0, tree]);
if (r0._ === 0) {
  done = r0.v;
} else {
  var rid = mkFrame(r0.k, -1, 0, r0.tag);
  bag = [
    { fn: r0.lf, a: r0.la, cid: rid, cs: 0 },
    { fn: r0.rf, a: r0.ra, cid: rid, cs: 1 },
  ];
  while (bag.length > 0 && done === null) tick();
}

var parTime = Date.now() - t1;
var got = done ? tsum(done) : null;
var parLeafs = warpLeafCalls;

render();

console.log("");
console.log("  task bag:  S = sort   W = warp   F = flow   . = idle");
console.log("  cont buf:  S = sort(d) waiting for sub-sorts");
console.log("             F = flow(d) waiting for warp, then will do down");
console.log("             D = down(d) waiting for sub-flows");
console.log("             W = warp(d) waiting for sub-warps");
console.log("");
console.log("  parallel steps : " + step);
console.log("  max bag size   : " + maxBag);
console.log("  max cont buf   : " + maxCon);
console.log("  total tasks    : " + totalTasks);
console.log("  conts resolved : " + totalConts);
console.log("  DFS fallbacks  : " + totalDFS);
console.log("  warp_leaf ops  : " + parLeafs + " (seq: " + seqLeafs + ")");
console.log("  avg parallel   : " + (totalTasks / step).toFixed(1) + " tasks/step");
console.log("  time           : " + parTime + "ms (seq: " + seqTime + "ms)");
console.log("  result         : " + (got === expected ? "PASS" : "FAIL") + " (sum=" + got + ")");
console.log("");
