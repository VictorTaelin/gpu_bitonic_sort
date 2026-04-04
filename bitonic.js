function Leaf(v) {
  return { tag: "Leaf", v };
}

function Node(l, r) {
  return { tag: "Node", l, r };
}

function gen(d, x) {
  if (d === 0) {
    return Leaf(x);
  } else {
    return Node(gen(d - 1, x * 2 + 1), gen(d - 1, x * 2));
  }
}

function warp_swap(s, av, bv) {
  if (s === 0) {
    return Node(Leaf(av), Leaf(bv));
  } else {
    return Node(Leaf(bv), Leaf(av));
  }
}

function warp(d, s, a, b) {
  if (d === 0) {
    return warp_swap(s ^ (a.v > b.v ? 1 : 0), a.v, b.v);
  } else {
    var wa = warp(d - 1, s, a.l, b.l);
    var wb = warp(d - 1, s, a.r, b.r);
    return Node(Node(wa.l, wb.l), Node(wa.r, wb.r));
  }
}

function flow(d, s, t) {
  if (d === 0) {
    return t;
  } else if (t.tag === "Leaf") {
    return t;
  } else {
    return down(d, s, warp(d - 1, s, t.l, t.r));
  }
}

function down(d, s, t) {
  if (d === 0) {
    return t;
  } else if (t.tag === "Leaf") {
    return t;
  } else {
    return Node(flow(d - 1, s, t.l), flow(d - 1, s, t.r));
  }
}

function sort(d, s, t) {
  if (d === 0) {
    return t;
  } else if (t.tag === "Leaf") {
    return t;
  } else {
    return flow(d, s, Node(sort(d - 1, 0, t.l), sort(d - 1, 1, t.r)));
  }
}

function checksum(t) {
  var result = 0;
  function go(t) {
    if (t.tag === "Leaf") {
      result = ((result * 31 + t.v) >>> 0);
    } else {
      go(t.l);
      go(t.r);
    }
  }
  go(t);
  return result;
}

var N      = 20;
var tree   = gen(N, 0);
var sorted = sort(N, 0, tree);
console.log(checksum(sorted));
