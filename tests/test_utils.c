#include <stdio.h>
#include "utils.h"

/* Phase 2 will expand each test with assertions.
 * For now these compile clean and print a placeholder. */

static void test_alloc_free(void) {
    /* Phase 2: alloc 10 floats, assert non-null, assert zero-init, free. */
    printf("test_alloc_free: stub\n");
}

static void test_matmul(void) {
    /* Phase 2: 2x2 @ 2x2 identity, assert result equals input. */
    printf("test_matmul: stub\n");
}

static void test_softmax(void) {
    /* Phase 2: input [1,2,3], assert output sums to 1.0 within 1e-6. */
    printf("test_softmax: stub\n");
}

static void test_transpose(void) {
    /* Phase 2: 2x3 matrix, assert out[i][j] == in[j][i]. */
    printf("test_transpose: stub\n");
}

int main(void) {
    test_alloc_free();
    test_matmul();
    test_softmax();
    test_transpose();
    printf("utils tests: all stubs passed\n");
    return 0;
}
