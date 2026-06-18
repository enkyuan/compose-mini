#include <stdio.h>
#include "embed.h"

static void test_embed_project(void) {
    /* Phase 3: 3 timesteps, in_dim=5, model_dim=8, W=identity-like.
     * Assert output shape [3 x 8], spot-check a value. */
    printf("test_embed_project: stub\n");
}

static void test_embed_positional(void) {
    /* Phase 3: seq_len=4, model_dim=8.
     * Assert PE[0][0]=sin(0)=0, PE[1][0]=sin(1/10000^0)=sin(1). */
    printf("test_embed_positional: stub\n");
}

int main(void) {
    test_embed_project();
    test_embed_positional();
    printf("embed tests: all stubs passed\n");
    return 0;
}
