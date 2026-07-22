/* Allocation and row-major math primitives shared by model components. */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "utils.h"

size_t utils_size_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) abort();
    return a + b;
}

size_t utils_size_mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a) abort();
    return a * b;
}

float* utils_alloc(size_t n) {
    float* p;
    if (!n || n > SIZE_MAX / sizeof *p) abort();
    p = calloc(n, sizeof *p);
    if (!p) abort();
    return p;
}

void utils_free(float* p) { free(p); }

void matmul(float* out, const float* a, const float* b, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        const size_t a_row = (size_t)i * (size_t)k;
        const size_t out_row = (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
                sum += a[a_row + (size_t)p] *
                       b[(size_t)p * (size_t)n + (size_t)j];
            out[out_row + (size_t)j] = sum;
        }
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
        for (int j = 0; j < cols; j++)
            out[(size_t)j * (size_t)rows + (size_t)i] =
                in[(size_t)i * (size_t)cols + (size_t)j];
}
