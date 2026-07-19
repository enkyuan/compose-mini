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
    const int matrix_size = seq_len * model_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float* scratch = utils_alloc(3 * matrix_size + seq_len);
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
            const int query = t * model_dim + head;

            for (int s = 0; s < seq_len; s++) {
                const int source = s * model_dim + head;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += out[query + d] * key[source + d];
                scores[s] = dot * scale;
            }

            softmax(scores, seq_len);
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int s = 0; s < seq_len; s++)
                    sum += scores[s] * value[s * model_dim + head + d];
                context[query + d] = sum;
            }
        }

    matmul(out, context, Wo, seq_len, model_dim, model_dim);
    utils_free(scratch);
}
