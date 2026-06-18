#ifndef UTILS_H
#define UTILS_H

/* Phase 2 (core math) implements all functions below.
 * Every other component depends on this header. */

/* Allocate a zero-initialized float array of length n.
 * Phase 2: wrap calloc, assert on failure. */
float* utils_alloc(int n);

/* Free a float array allocated by utils_alloc.
 * Phase 2: wrap free. */
void utils_free(float* p);

/* Matrix multiply: out[m x n] = a[m x k] @ b[k x n].
 * Phase 2: naive triple loop; Phase 6 (bench): swap loop order for cache efficiency. */
void matmul(float* out, const float* a, const float* b, int m, int k, int n);

/* In-place softmax over array x of length n.
 * Phase 2: subtract max for numerical stability before exp. */
void softmax(float* x, int n);

/* Transpose: out[cols x rows] = in[rows x cols]^T.
 * Phase 2: simple index swap. */
void transpose(float* out, const float* in, int rows, int cols);

#endif /* UTILS_H */
