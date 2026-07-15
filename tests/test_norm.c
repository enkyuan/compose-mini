#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "norm.h"

static void assert_array_close(const float* actual, const float* expected, int n) {
    for (int i = 0; i < n; i++) assert(fabsf(actual[i] - expected[i]) < 1e-5f);
}

/* This case distinguishes population variance from sample variance. */
static void test_affine(void) {
    const float x[] = {1, 3}, gamma[] = {2, 3}, beta[] = {4, 5};
    const float expected[] = {2.00001f, 7.999985f};
    float out[2];
    layernorm(out, x, gamma, beta, 2);
    assert_array_close(out, expected, 2);
}

/* Zero variance makes the normalized term zero, so output equals beta. */
static void test_constant(void) {
    const float x[] = {7, 7, 7}, gamma[] = {2, -3, 0.5f}, beta[] = {1, -2, 4};
    const float expected[] = {1, -2, 4};
    float out[3];
    layernorm(out, x, gamma, beta, 3);
    assert_array_close(out, expected, 3);
}

int main(void) {
    test_affine();
    test_constant();
    printf("norm tests passed\n");
    return 0;
}
