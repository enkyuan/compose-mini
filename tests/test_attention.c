#include <stdio.h>
#include "attention.h"

static void test_attention_forward(void) {
    /*
     * Planned test: seq_len=4, model_dim=8, num_heads=2.
     * Set Wq/Wk to zero and Wv/Wo to identity. Attention is uniform, so every
     * output row must equal the sequence mean.
     */
    printf("test_attention_forward: skipped (implementation stub)\n");
}

int main(void) {
    test_attention_forward();
    return 0;
}
