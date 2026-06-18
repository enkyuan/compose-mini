#include <stdio.h>
#include "norm.h"

static void test_layernorm(void) {
    /* Phase 3: input x=[1,2,3,4], gamma=ones, beta=zeros.
     * Assert out has mean ~0 and variance ~1 within 1e-5. */
    printf("test_layernorm: stub\n");
}

int main(void) {
    test_layernorm();
    printf("norm tests: all stubs passed\n");
    return 0;
}
