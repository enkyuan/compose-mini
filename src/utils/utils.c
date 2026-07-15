/* Allocation and row-major math primitives shared by model components. */

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"

float* utils_alloc(int n) {
    assert(n > 0);
    float* p = calloc((size_t)n, sizeof *p);
    assert(p);
    return p;
}

void utils_free(float* p) { free(p); }

void matmul(float* out, const float* a, const float* b, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
            out[i * n + j] = sum;
        }
}

/* Subtracting the maximum preserves softmax and prevents expf overflow. */
void softmax(float* x, int n) {
    assert(x && n > 0);
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

void transpose(float* out, const float* in, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) out[j * rows + i] = in[i * cols + j];
}
