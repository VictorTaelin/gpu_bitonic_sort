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
    var xl = gen(d - 1, x * 2 + 1);
    var xr = gen(d - 1, x * 2);
    return Node(xl, xr);
  }
}

function sum(t) {
  if (t.tag === "Leaf") {
    return t.v;
  } else {
    return sum(t.l) + sum(t.r);
  }
}

function warp_swap(c, av, bv) {
  if (c === 0) {
    return Node(Leaf(av), Leaf(bv));
  } else {
    return Node(Leaf(bv), Leaf(av));
  }
}

function warp(d, s, a, b) {
  if (d === 0) {
    var av = a.v;
    var bv = b.v;
    return warp_swap(s ^ (av > bv ? 1 : 0), av, bv);
  } else {
    var wa = warp(d - 1, s, a.l, b.l);
    var wb = warp(d - 1, s, a.r, b.r);
    return Node(Node(wa.l, wb.l), Node(wa.r, wb.r));
  }
}

function flow(d, s, t) {
  if (d === 0) {
    return t;
  } else {
    var warped = warp(d - 1, s, t.l, t.r);
    return down(d, s, warped);
  }
}

function down(d, s, t) {
  if (d === 0) {
    return t;
  } else if (t.tag === "Leaf") {
    return t;
  } else {
    var tl = flow(d - 1, s, t.l);
    var tr = flow(d - 1, s, t.r);
    return Node(tl, tr);
  }
}

function depth(t) {
  if (t.tag === "Leaf") {
    return 0;
  } else {
    return 1 + depth(t.l);
  }
}

function sort(t) {
  var d = depth(t);
  return flow(d, 0, t);
}

function main() {
  return sum(sort(gen(22, 0)));
}

console.log(main());
