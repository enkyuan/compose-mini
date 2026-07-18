/* Position-wise feed-forward network for encoder hidden states. */

#include <assert.h>
#include <math.h>
#include "ffn.h"

void ffn_forward(float* out, const float* x,
                  const float* W1, const float* b1,
                  const float* W2, const float* b2,
                  int model_dim, int ff_dim) {
    assert(out && x && W1 && b1 && W2 && b2 && model_dim > 0 && ff_dim > 0);
    for (int j = 0; j < model_dim; j++) out[j] = b2[j];

    for (int h = 0; h < ff_dim; h++) {
        float sum = b1[h];
        for (int i = 0; i < model_dim; i++) sum += x[i] * W1[i * ff_dim + h];
        const float activation = 0.5f * sum *
            (1.0f + erff(sum * 0.7071067811865475f));
        for (int j = 0; j < model_dim; j++)
            out[j] += activation * W2[h * model_dim + j];
    }
}
