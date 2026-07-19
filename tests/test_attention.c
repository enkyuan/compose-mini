#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "attention.h"

static const float I4[] = {
    1, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 1,
};
static const float Z4[16] = {0};

static void assert_array_close(const float* actual, const float* expected, int n) {
    for (int i = 0; i < n; i++) assert(fabsf(actual[i] - expected[i]) < 1e-5f);
}

/* Zero queries and keys make every attention row the sequence mean. */
static void test_uniform(void) {
    const float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float expected[] = {3, 4, 5, 6, 3, 4, 5, 6};
    float out[8] = {0};
    attention_forward(out, x, Z4, Z4, I4, I4, 2, 4, 2);
    assert_array_close(out, expected, 8);
}

/* Identity projections expose head slicing and sqrt(head_dim) scaling. */
static void test_scaled_heads(void) {
    const float x[] = {1, 0, 0, 1, 0, 1, 1, 0};
    const float expected[] = {
        0.66976154f, 0.33023846f, 0.33023846f, 0.66976154f,
        0.33023846f, 0.66976154f, 0.66976154f, 0.33023846f,
    };
    float out[8] = {0};
    attention_forward(out, x, I4, I4, I4, I4, 2, 4, 2);
    assert_array_close(out, expected, 8);
}

/* Non-identity weights expose projection order and row-wise softmax errors. */
static void test_projection(void) {
    const float x[] = {1, 0, 0, 1};
    const float Wq[] = {1, 0, 0, 2}, Wk[] = {0, 1, 1, 0};
    const float Wv[] = {1, 2, 3, 4}, Wo[] = {1, -1, 2, 1};
    const float expected[] = {9.018569f, 1, 6.173422f, 1};
    float out[4] = {0};
    attention_forward(out, x, Wq, Wk, Wv, Wo, 2, 2, 1);
    assert_array_close(out, expected, 4);
}

int main(void) {
    test_uniform();
    test_scaled_heads();
    test_projection();
    printf("attention tests passed\n");
    return 0;
}
