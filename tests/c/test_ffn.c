#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "ffn.h"

static void assert_array_close(const float* actual, const float* expected, int n) {
    for (int i = 0; i < n; i++) assert(fabsf(actual[i] - expected[i]) < 1e-6f);
}

/* Asymmetric weights expose transposed indexing and the wrong GELU variant. */
static void test_forward(void) {
    const float x[] = {1, 2};
    const float W1[] = {1, 2, -1, 2, -1, 3}, b1[] = {-4, 0.5f, -6};
    const float W2[] = {1, 2, -3, 4, 5, -1}, b2[] = {0.25f, -0.5f};
    const float expected[] = {-0.7391252f, 2.7242696f};
    float out[2] = {99, 99};
    ffn_forward(out, x, W1, b1, W2, b2, 2, 3);
    assert_array_close(out, expected, 2);
}

int main(void) {
    test_forward();
    printf("ffn tests passed\n");
    return 0;
}
