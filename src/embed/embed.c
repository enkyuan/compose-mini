/* Input projection and positional encoding for encoder hidden states. */

#include <assert.h>
#include <math.h>
#include "embed.h"
#include "utils.h"

void embed_project(float* out, const float* x, const float* W,
                    int seq_len, int in_dim, int model_dim) {
    assert(out && x && W && seq_len > 0 && in_dim > 0 && model_dim > 0);
    matmul(out, x, W, seq_len, in_dim, model_dim);
}

void embed_positional(float* x, int seq_len, int model_dim) {
    assert(x && seq_len > 0 && model_dim > 0);
    const float frequency_step = powf(10000.0f, -2.0f / (float)model_dim);

    for (int pos = 0; pos < seq_len; pos++) {
        const size_t row = (size_t)pos * (size_t)model_dim;
        float frequency = 1.0f;
        for (int i = 0; i < model_dim; i += 2) {
            const float angle = (float)pos * frequency;
            x[row + (size_t)i] += sinf(angle);
            if (i + 1 < model_dim) x[row + (size_t)i + 1] += cosf(angle);
            frequency *= frequency_step;
        }
    }
}
