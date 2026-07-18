#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "embed.h"

static void assert_array_close(const float* actual, const float* expected, int n) {
    for (int i = 0; i < n; i++) assert(fabsf(actual[i] - expected[i]) < 1e-6f);
}

static void test_project(void) {
    const float x[] = {1, 2, 3, -1, 0, 2};
    const float weights[] = {1, 2, 0, -1, 3, 0.5f};
    const float expected[] = {10, 1.5f, 5, -1};
    float out[4] = {0};
    embed_project(out, x, weights, 2, 3, 2);
    assert_array_close(out, expected, 4);
}

/* Nonzero input and odd width cover addition and the final unpaired sine. */
static void test_positional(void) {
    float x[] = {1, 1, 1, 1, 1, 1};
    const float expected[] = {1, 2, 1, 1.841471f, 1.540302f, 1.002154f};
    embed_positional(x, 2, 3);
    assert_array_close(x, expected, 6);
}

int main(void) {
    test_project();
    test_positional();
    printf("embed tests passed\n");
    return 0;
}
