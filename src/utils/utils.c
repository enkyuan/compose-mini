#include "utils.h"

/* Phase 2: implement utils_alloc — calloc(n, sizeof(float)), assert non-null. */
float* utils_alloc(int n) { (void)n; return 0; }

/* Phase 2: implement utils_free — free(p). */
void utils_free(float* p) { (void)p; }

/* Phase 2: implement matmul — triple loop: i,j,k over m,n,k dims.
 * Phase 6 (bench): reorder to k,i,j for better cache locality. */
void matmul(float* out, const float* a, const float* b, int m, int k, int n) {
    (void)out; (void)a; (void)b; (void)m; (void)k; (void)n;
}

/* Phase 2: implement softmax — find max, subtract, exp, normalize.
 * Numerical stability: x[i] = exp(x[i]-max) / sum(exp(x[j]-max)). */
void softmax(float* x, int n) { (void)x; (void)n; }

/* Phase 2: implement transpose — out[j*rows+i] = in[i*cols+j]. */
void transpose(float* out, const float* in, int rows, int cols) {
    (void)out; (void)in; (void)rows; (void)cols;
}
