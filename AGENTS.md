Our goal is to optimize and simplify the bitonic.cu implementation, aiming to
achieve a large speedup against C. The target is to build an elegant template
for parallel recursive function evaluation to be used as the basis of the new
Bend2 CUDA runtime. Currently, we hardcoded a recursive bitonic sort as a demo,
since it is a representative example of a contrived recursive algorithm that
HAS a lot of inherent parallelism and, thus, MUST run fast on Bend2. Currently,
the implementation is still lackluster, but it is already faster than single
threaded C, so, that's a good first step.

You can run it on the `ssh rtx` machine.
