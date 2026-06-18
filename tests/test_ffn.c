#include <stdio.h>
#include "ffn.h"

static void test_ffn_forward(void) {
    /* Phase 3: model_dim=4, ff_dim=16, W1/W2=identity-like, b=zeros.
     * Assert output shape matches input, spot-check a value through GELU. */
    printf("test_ffn_forward: stub\n");
}

int main(void) {
    test_ffn_forward();
    printf("ffn tests: all stubs passed\n");
    return 0;
}
