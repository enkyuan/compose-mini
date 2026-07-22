/* Pre-norm Transformer encoder lifecycle, forward pass, and scalar head. */

#include <assert.h>
#include <stddef.h>
#include "transformer.h"
#include "attention.h"
#include "embed.h"
#include "ffn.h"
#include "norm.h"
#include "utils.h"

static void assert_config(TransformerConfig cfg) {
    assert(cfg.model_dim > 0 && cfg.num_heads > 0 && cfg.ff_dim > 0);
    assert(cfg.num_layers > 0 && cfg.seq_len > 0 && cfg.in_dim > 0);
    assert(cfg.model_dim % cfg.num_heads == 0);
    (void)cfg;
}

/* Embedding, per-layer attention/FFN parameters, then the scalar head. */
static size_t parameter_count(TransformerConfig cfg) {
    const size_t d = (size_t)cfg.model_dim, f = (size_t)cfg.ff_dim;
    const size_t l = (size_t)cfg.num_layers, in = (size_t)cfg.in_dim;
    const size_t d2 = utils_size_mul(d, d), df = utils_size_mul(d, f);
    size_t per_layer = utils_size_add(utils_size_mul(4, d2),
                                      utils_size_mul(2, df));
    per_layer = utils_size_add(per_layer,
                               utils_size_add(utils_size_mul(5, d), f));
    size_t total = utils_size_add(utils_size_mul(in, d),
                                  utils_size_mul(l, per_layer));
    return utils_size_add(total, utils_size_add(d, 1));
}

static float* take(float* storage, size_t* offset, size_t count) {
    float* view = storage + *offset;
    *offset += count;
    return view;
}

static void assert_encoder_weights(const TransformerWeights* w) {
    assert(w && w->storage && w->embed_W);
    assert(w->Wq && w->Wk && w->Wv && w->Wo);
    assert(w->norm1_g && w->norm1_b && w->W1 && w->b1);
    assert(w->W2 && w->b2 && w->norm2_g && w->norm2_b);
    (void)w;
}

static void normalize_rows(float* out, const float* x,
                           const float* gamma, const float* beta,
                           int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        const size_t offset = (size_t)row * (size_t)cols;
        layernorm(out + offset, x + offset, gamma, beta, cols);
    }
}

static void add_in_place(float* out, const float* residual, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] += residual[i];
}

void transformer_init(TransformerWeights* w, TransformerConfig cfg) {
    assert(w);
    assert_config(cfg);

    const size_t d = (size_t)cfg.model_dim, f = (size_t)cfg.ff_dim;
    const size_t l = (size_t)cfg.num_layers, in = (size_t)cfg.in_dim;
    const size_t d2 = utils_size_mul(d, d), df = utils_size_mul(d, f);
    const size_t ld = utils_size_mul(l, d), lf = utils_size_mul(l, f);
    const size_t ld2 = utils_size_mul(l, d2), ldf = utils_size_mul(l, df);
    const size_t total = parameter_count(cfg);
    TransformerWeights initialized = {0};
    size_t offset = 0;

    initialized.storage = utils_alloc(total);
    initialized.embed_W = take(initialized.storage, &offset,
                               utils_size_mul(in, d));
    initialized.Wq = take(initialized.storage, &offset, ld2);
    initialized.Wk = take(initialized.storage, &offset, ld2);
    initialized.Wv = take(initialized.storage, &offset, ld2);
    initialized.Wo = take(initialized.storage, &offset, ld2);
    initialized.norm1_g = take(initialized.storage, &offset, ld);
    initialized.norm1_b = take(initialized.storage, &offset, ld);
    initialized.W1 = take(initialized.storage, &offset, ldf);
    initialized.b1 = take(initialized.storage, &offset, lf);
    initialized.W2 = take(initialized.storage, &offset, ldf);
    initialized.b2 = take(initialized.storage, &offset, ld);
    initialized.norm2_g = take(initialized.storage, &offset, ld);
    initialized.norm2_b = take(initialized.storage, &offset, ld);
    initialized.head_W = take(initialized.storage, &offset, d);
    initialized.head_b = take(initialized.storage, &offset, 1);
    assert(offset == total);
    *w = initialized;
}

void transformer_free(TransformerWeights* w) {
    if (!w) return;
    utils_free(w->storage);
    *w = (TransformerWeights){0};
}

void transformer_forward(float* out, const float* x,
                         const TransformerWeights* w,
                         TransformerConfig cfg) {
    assert(out && x && out != x);
    assert_config(cfg);
    assert_encoder_weights(w);

    const int d = cfg.model_dim, f = cfg.ff_dim, s = cfg.seq_len;
    const size_t hidden_size = utils_size_mul((size_t)s, (size_t)d);
    const size_t d2 = utils_size_mul((size_t)d, (size_t)d);
    const size_t df = utils_size_mul((size_t)d, (size_t)f);
    /* Both buffers are reused across layers, keeping scratch independent of L. */
    float* scratch = utils_alloc(utils_size_mul(2, hidden_size));
    float* normalized = scratch;
    float* branch = scratch + hidden_size;

    embed_project(out, x, w->embed_W, s, cfg.in_dim, d);
    embed_positional(out, s, d);

    for (int layer = 0; layer < cfg.num_layers; layer++) {
        const size_t d_offset = (size_t)layer * (size_t)d;
        const size_t f_offset = (size_t)layer * (size_t)f;
        const size_t d2_offset = (size_t)layer * d2;
        const size_t df_offset = (size_t)layer * df;

        normalize_rows(normalized, out, w->norm1_g + d_offset,
                       w->norm1_b + d_offset, s, d);
        attention_forward(branch, normalized, w->Wq + d2_offset,
                          w->Wk + d2_offset, w->Wv + d2_offset,
                          w->Wo + d2_offset, s, d, cfg.num_heads);
        add_in_place(out, branch, hidden_size);

        normalize_rows(normalized, out, w->norm2_g + d_offset,
                       w->norm2_b + d_offset, s, d);
        for (int row = 0; row < s; row++) {
            const size_t row_offset = (size_t)row * (size_t)d;
            ffn_forward(branch + row_offset, normalized + row_offset,
                        w->W1 + df_offset, w->b1 + f_offset,
                        w->W2 + df_offset, w->b2 + d_offset, d, f);
        }
        add_in_place(out, branch, hidden_size);
    }

    utils_free(scratch);
}

float transformer_predict(const float* hidden,
                          const TransformerWeights* w,
                          TransformerConfig cfg) {
    assert(hidden && w && w->head_W && w->head_b);
    assert_config(cfg);

    const size_t final_offset =
        (size_t)(cfg.seq_len - 1) * (size_t)cfg.model_dim;
    const float* final = hidden + final_offset;
    float prediction = *w->head_b;
    for (int i = 0; i < cfg.model_dim; i++) prediction += final[i] * w->head_W[i];
    return prediction;
}
