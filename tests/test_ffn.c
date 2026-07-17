#include <stdio.h>
#include "ffn.h"

static void test_ffn_forward(void) {
    /*
     * Planned test: model_dim=4, ff_dim=16, W1/W2=identity-like, b=zeros.
     * Assert output shape matches input, spot-check a value through GELU.
     */
    printf("test_ffn_forward: skipped (implementation stub)\n");
}

int main(void) {
    test_ffn_forward();
    return 0;
}
