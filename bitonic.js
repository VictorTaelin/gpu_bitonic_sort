function leaf(v) {
  return { tag: "leaf", v };
}

function node(l, r) {
  return { tag: "node", l, r };
}

function gen(d, x) {
  if (d === 0) return leaf(x);
  return node(gen(d - 1, x * 2 + 1), gen(d - 1, x * 2));
}

function sum(t) {
  if (t.tag === "leaf") return t.v;
  return sum(t.l) + sum(t.r);
}

function warpSwap(c, av, bv) {
  return c === 0
    ? node(leaf(av), leaf(bv))
    : node(leaf(bv), leaf(av));
}

function warp(d, s, a, b) {
  if (d === 0) {
    // Both a and b are leaves
    const av = a.v;
    const bv = b.v;
    return warpSwap(s ^ (av > bv ? 1 : 0), av, bv);
  }
  // Both a and b are nodes; recurse on halves then zip
  const wa = warp(d - 1, s, a.l, b.l);
  const wb = warp(d - 1, s, a.r, b.r);
  // zip: interleave the two results
  return node(node(wa.l, wb.l), node(wa.r, wb.r));
}

function flow(d, s, t) {
  if (d === 0) return t;
  // t is a node; warp its two halves then push fixes downward
  const warped = warp(d - 1, s, t.l, t.r);
  return down(d, s, warped);
}

function down(d, s, t) {
  if (d === 0) return t;
  if (t.tag === "leaf") return t;
  return node(flow(d - 1, s, t.l), flow(d - 1, s, t.r));
}

function depth(t) {
  if (t.tag === "leaf") return 0;
  return 1 + depth(t.l);
}

function sort(t) {
  const d = depth(t);
  return flow(d, 0, t);
}

function main() {
  return sum(sort(gen(20, 0)));
}

console.log(main());
