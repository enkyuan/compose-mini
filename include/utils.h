#ifndef UTILS_H
#define UTILS_H

/* Float-buffer allocation and row-major math primitives used by the model. */

/* Allocate n zero-initialized floats; n must be positive. */
float* utils_alloc(int n);

/* Free memory returned by utils_alloc; p may be NULL. */
void utils_free(float* p);

/*
 * Multiply row-major matrices: out[m x n] = a[m x k] @ b[k x n].
 * Arrays are contiguous and row-major; output must not alias either input.
 */
void matmul(float* out, const float* a, const float* b, int m, int k, int n);

/* Apply max-shifted softmax to n finite values in place; n must be positive. */
void softmax(float* x, int n);

/* Transpose a row-major [rows x cols] array into separate output storage. */
void transpose(float* out, const float* in, int rows, int cols);

#endif /* UTILS_H */
