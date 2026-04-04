// recursive_bitonic_sort.c — Sequential baseline for recursive bitonic sort
// 2-word tree nodes matching the CUDA version for fair comparison.
// Usage: ./sort_seq [depth=20] [heap_gb=8]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


typedef uint32_t u32;
typedef uint64_t u64;

static u32 *H;
static u32 HP;

// Leaf: [value (bit31=0), 0]
// Node: [left|0x80000000, right]
#define ND(p) (H[(p)]>>31)
#define VL(p) (H[(p)]&0x7FFFFFFFu)
#define LF(p) (H[(p)]&0x7FFFFFFFu)
#define RT(p) (H[(p)+1])

static inline u32 mkL(u32 v){ u32 p=HP; HP+=2; H[p]=v; H[p+1]=0; return p; }
static inline u32 mkN(u32 l,u32 r){ u32 p=HP; HP+=2; H[p]=l|0x80000000u; H[p+1]=r; return p; }

u32 gen(u32 d, u32 x){
    return d==0 ? mkL(x) : mkN(gen(d-1,x*2+1),gen(d-1,x*2));
}

u32 check_idx;
u32 check_ok;
void tree_check(u32 t){
    if(!ND(t)){
        if(VL(t)!=check_idx) check_ok=0;
        check_idx++;
    } else {
        tree_check(LF(t));
        tree_check(RT(t));
    }
}

u32 warp(u32 d, u32 s, u32 a, u32 b){
    if(d==0){
        if(ND(a)|ND(b)) return mkL(0);
        u32 av=VL(a),bv=VL(b),c=s^(av>bv);
        return c ? mkN(mkL(bv),mkL(av)) : mkN(mkL(av),mkL(bv));
    }
    if(!ND(a)||!ND(b)) return mkL(0);
    u32 wa=warp(d-1,s,LF(a),LF(b)), wb=warp(d-1,s,RT(a),RT(b));
    if(!ND(wa)||!ND(wb)) return mkL(0);
    return mkN(mkN(LF(wa),LF(wb)),mkN(RT(wa),RT(wb)));
}

u32 flow(u32 d, u32 s, u32 t);
u32 down(u32 d, u32 s, u32 t){
    if(d==0||!ND(t)) return t;
    return mkN(flow(d-1,s,LF(t)),flow(d-1,s,RT(t)));
}
u32 flow(u32 d, u32 s, u32 t){
    if(d==0||!ND(t)) return t;
    return down(d,s,warp(d-1,s,LF(t),RT(t)));
}

u32 bsort(u32 d, u32 s, u32 t){
    if(d==0||!ND(t)) return t;
    return flow(d,s,mkN(bsort(d-1,0,LF(t)),bsort(d-1,1,RT(t))));
}

int main(int argc, char** argv){
    u32 D=argc>1?(u32)atoi(argv[1]):20;
    u64 gb=argc>2?(u64)atoi(argv[2]):8;
    u64 n=(gb<<30)/4;
    H=(u32*)malloc(n*4);
    if(!H){fprintf(stderr,"malloc %lluGB failed\n",(unsigned long long)gb);return 1;}
    HP=0;

    u32 tree=gen(D,0);
    u32 sorted=bsort(D,0,tree);
    check_idx=0; check_ok=1;
    tree_check(sorted);
    u32 n_leaves=1u<<D;
    fprintf(stderr,"sort(%u) %s  leaves=%u  heap=%.2fGB\n",D,
            (check_ok && check_idx==n_leaves)?"PASS":"FAIL",
            check_idx,HP*4.0/(1ULL<<30));
    free(H);
    return 0;
}
