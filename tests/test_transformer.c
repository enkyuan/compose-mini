#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "transformer.h"

static void assert_array_close(const float* actual, const float* expected, int n) {
    for (int i = 0; i < n; i++) assert(fabsf(actual[i] - expected[i]) < 1e-5f);
}

/* Two distinct layers expose parameter offsets and pre-norm residual order. */
static void test_forward(void) {
    const TransformerConfig cfg = {2, 1, 2, 2, 2, 2};
    TransformerWeights w = {0};
    transformer_init(&w, cfg);
    assert(w.storage);

    const float g = sqrtf(1.00001f);
    const float embed[] = {1, 0, 0, 1};
    const float Wq[] = {1, 0, 0, 1, 1, -1, -1, 1};
    const float Wk[] = {1, 0, 0, 1, .03125f, -.03125f, -.03125f, .03125f};
    const float Wv[] = {1, 0, 0, 1, 2, 0, 0, 2};
    const float Wo[] = {1, 1, 0, 0, 0, 0, .5f, .5f};
    const float norm1_g[] = {g, g, 2 * g, 2 * g};
    const float norm1_b[] = {0, 0, 1, 1};
    const float W1[] = {1, -1, 0, 0, 0, 0, 1.f / 3, -1.f / 3};
    const float b1[] = {0, 0, 1.f / 3, -1.f / 3};
    const float W2[] = {1, 1, -1, -1, 2, 2, -2, -2};
    const float b2[] = {0, 0, 3, 3};
    const float norm2_g[] = {g, g, 3 * g, 3 * g};
    const float norm2_b[] = {0, 0, 2, 2};

    memcpy(w.embed_W, embed, sizeof embed);
    memcpy(w.Wq, Wq, sizeof Wq); memcpy(w.Wk, Wk, sizeof Wk);
    memcpy(w.Wv, Wv, sizeof Wv); memcpy(w.Wo, Wo, sizeof Wo);
    memcpy(w.norm1_g, norm1_g, sizeof norm1_g);
    memcpy(w.norm1_b, norm1_b, sizeof norm1_b);
    memcpy(w.W1, W1, sizeof W1); memcpy(w.b1, b1, sizeof b1);
    memcpy(w.W2, W2, sizeof W2); memcpy(w.b2, b2, sizeof b2);
    memcpy(w.norm2_g, norm2_g, sizeof norm2_g);
    memcpy(w.norm2_b, norm2_b, sizeof norm2_b);

    const float x[] = {1, -2, -1 - sinf(1), 1 - cosf(1)};
    const float expected[] = {5.6706667f, 3.6706667f, 6.3293333f, 8.3293333f};
    float out[4];
    transformer_forward(out, x, &w, cfg);
    assert_array_close(out, expected, 4);

    transformer_free(&w);
    assert(!w.storage && !w.embed_W && !w.head_W && !w.head_b);
}

/* The scalar head consumes the final sequence row, never an earlier row. */
static void test_predict(void) {
    const TransformerConfig cfg = {3, 1, 1, 1, 2, 1};
    TransformerWeights w;
    transformer_init(&w, cfg);

    /* This configuration places the four head floats at storage[61..64]. */
    const float packed_head[] = {2, -1, .5f, .25f};
    memcpy(w.storage + 61, packed_head, sizeof packed_head);
    assert(w.head_W == w.storage + 61 && w.head_b == w.storage + 64);
    const float hidden[] = {9, 9, 9, 1, 2, 4};
    assert(fabsf(transformer_predict(hidden, &w, cfg) - 2.25f) < 1e-6f);
    transformer_free(&w);
}

int main(void) {
    test_forward();
    test_predict();
    printf("transformer tests passed\n");
    return 0;
}
