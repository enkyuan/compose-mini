#include <stdio.h>
#include "embed.h"

static void test_embed_project(void) {
    /*
     * Planned test: 3 timesteps, in_dim=5, model_dim=8, W=identity-like.
     * Assert output shape [3 x 8], spot-check a value.
     */
    printf("test_embed_project: skipped (implementation stub)\n");
}

static void test_embed_positional(void) {
    /*
     * Planned test: seq_len=4, model_dim=8.
     * Assert PE[0][0]=sin(0)=0, PE[1][0]=sin(1/10000^0)=sin(1).
     */
    printf("test_embed_positional: skipped (implementation stub)\n");
}

int main(void) {
    test_embed_project();
    test_embed_positional();
    return 0;
}
