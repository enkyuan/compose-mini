#ifndef TRANSFORMER_H
#define TRANSFORMER_H

/* Phase 3 (components) implements all functions below.
 * Depends on: attention.h, ffn.h, norm.h, utils.h */

/* Model hyperparameters. Set at init time, immutable during inference. */
typedef struct {
    int model_dim;   /* embedding / hidden dimension */
    int num_heads;   /* number of attention heads; must divide model_dim */
    int ff_dim;      /* feed-forward inner dimension; typically 4*model_dim */
    int num_layers;  /* number of stacked encoder blocks */
    int seq_len;     /* input sequence length (number of timesteps) */
    int in_dim;      /* input feature dimension; 5 for OHLCV */
} TransformerConfig;

/* Flat weight arrays for one encoder block.
 * Phase 3: allocate with utils_alloc sized by TransformerConfig.
 * Phase 5 (inference): load from binary weight file instead. */
typedef struct {
    float* embed_W;    /* [in_dim x model_dim] input projection */
    float* Wq;         /* [model_dim x model_dim] query projection */
    float* Wk;         /* [model_dim x model_dim] key projection */
    float* Wv;         /* [model_dim x model_dim] value projection */
    float* Wo;         /* [model_dim x model_dim] output projection */
    float* norm1_g;    /* [model_dim] layernorm1 gamma */
    float* norm1_b;    /* [model_dim] layernorm1 beta */
    float* W1;         /* [model_dim x ff_dim] ffn first layer */
    float* b1;         /* [ff_dim] ffn first bias */
    float* W2;         /* [ff_dim x model_dim] ffn second layer */
    float* b2;         /* [model_dim] ffn second bias */
    float* norm2_g;    /* [model_dim] layernorm2 gamma */
    float* norm2_b;    /* [model_dim] layernorm2 beta */
} TransformerWeights;

/* Allocate all weight arrays sized by cfg.
 * Phase 3: call utils_alloc for each field.
 * Phase 5 (inference): replace with weight file loader. */
void transformer_init(TransformerWeights* w, TransformerConfig cfg);

/* Free all weight arrays.
 * Phase 3: call utils_free for each field. */
void transformer_free(TransformerWeights* w, TransformerConfig cfg);

/* Full forward pass: x[seq_len x in_dim] → out[seq_len x model_dim].
 * Per encoder block:
 *   1. embed_project + embed_positional
 *   2. layernorm1 → attention_forward → residual add
 *   3. layernorm2 → ffn_forward → residual add
 * Phase 3: implement the block loop.
 * Phase 6 (decoder): add causal mask flag to cfg. */
void transformer_forward(float* out, const float* x,
                         const TransformerWeights* w,
                         TransformerConfig cfg);

#endif /* TRANSFORMER_H */
