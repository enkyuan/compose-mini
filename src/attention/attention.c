/* Multi-head self-attention over historical encoder states. */

#include <assert.h>
#include <math.h>
#include "attention.h"
#include "utils.h"

void attention_forward(float* out, const float* x,
                       const float* Wq, const float* Wk,
                       const float* Wv, const float* Wo,
                       int seq_len, int model_dim, int num_heads) {
    assert(out && x && Wq && Wk && Wv && Wo);
    assert(seq_len > 0 && model_dim > 0 && num_heads > 0);
    assert(model_dim % num_heads == 0);

    const int head_dim = model_dim / num_heads;
    const size_t matrix_size = utils_size_mul((size_t)seq_len,
                                              (size_t)model_dim);
    const size_t scratch_size = utils_size_add(utils_size_mul(3, matrix_size),
                                               (size_t)seq_len);
    const float scale = 1.0f / sqrtf((float)head_dim);
    float* scratch = utils_alloc(scratch_size);
    float* key = scratch;
    float* value = key + matrix_size;
    float* context = value + matrix_size;
    float* scores = context + matrix_size;

    matmul(out, x, Wq, seq_len, model_dim, model_dim);
    matmul(key, x, Wk, seq_len, model_dim, model_dim);
    matmul(value, x, Wv, seq_len, model_dim, model_dim);

    for (int t = 0; t < seq_len; t++)
        for (int h = 0; h < num_heads; h++) {
            const int head = h * head_dim;
            const size_t query = (size_t)t * (size_t)model_dim + (size_t)head;

            for (int s = 0; s < seq_len; s++) {
                const size_t source =
                    (size_t)s * (size_t)model_dim + (size_t)head;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += out[query + (size_t)d] * key[source + (size_t)d];
                scores[s] = dot * scale;
            }

            softmax(scores, seq_len);
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int s = 0; s < seq_len; s++)
                    sum += scores[s] * value[(size_t)s * (size_t)model_dim +
                                             (size_t)head + (size_t)d];
                context[query + (size_t)d] = sum;
            }
        }

    matmul(out, context, Wo, seq_len, model_dim, model_dim);
    utils_free(scratch);
}
