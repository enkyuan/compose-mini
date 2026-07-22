#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-6f);
}

static void assert_array_close(const float* actual, const float* expected, int n) {
    for (int i = 0; i < n; i++) assert_close(actual[i], expected[i]);
}

static void test_alloc_free(void) {
    float* values = utils_alloc(3);
    const float expected[] = {0.0f, 0.0f, 0.0f};
    assert_array_close(values, expected, 3);
    utils_free(values);
}

static void test_size_math(void) {
    assert(utils_size_add(7, 5) == 12);
    assert(utils_size_mul(7, 5) == 35);
}

static void test_matmul(void) {
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float b[] = {7, 8, 9, 10, 11, 12};
    const float expected[] = {58, 64, 139, 154};
    float out[] = {-1, -1, -1, -1};
    matmul(out, a, b, 2, 3, 2);
    assert_array_close(out, expected, 4);
}

/* Large logits catch implementations that exponentiate before shifting. */
static void test_softmax(void) {
    float values[] = {1000, 1001, 1002};
    const float expected[] = {0.0900306f, 0.2447285f, 0.6652409f};
    softmax(values, 3);
    assert_array_close(values, expected, 3);
    assert_close(values[0] + values[1] + values[2], 1.0f);
}

static void test_transpose(void) {
    const float in[] = {1, 2, 3, 4, 5, 6};
    const float expected[] = {1, 4, 2, 5, 3, 6};
    float out[6];
    transpose(out, in, 2, 3);
    assert_array_close(out, expected, 6);
}

int main(void) {
    test_alloc_free();
    test_size_math();
    test_matmul();
    test_softmax();
    test_transpose();
    printf("utils tests passed\n");
    return 0;
}
