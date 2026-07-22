#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stddef.h>

/* Encoder weights, hidden-state forward pass, and scalar prediction head. */

/* Model hyperparameters. Set at init time, immutable during inference. */
typedef struct {
    int model_dim;   /* embedding / hidden dimension */
    int num_heads;   /* number of attention heads; must divide model_dim */
    int ff_dim;      /* feed-forward inner dimension; typically 4*model_dim */
    int num_layers;  /* number of stacked encoder blocks */
    int seq_len;     /* input sequence length (number of timesteps) */
    int in_dim;      /* input feature dimension; 5 for OHLCV */
} TransformerConfig;

/*
 * Flat model weights. storage owns one allocation; every other pointer is a
 * view. Each block field contains num_layers consecutive parameter slices.
 */
typedef struct {
    float* storage;    /* sole allocation owner */
    float* embed_W;    /* [in_dim x model_dim] input projection */
    float* Wq;         /* [num_layers x model_dim x model_dim] */
    float* Wk;         /* [num_layers x model_dim x model_dim] */
    float* Wv;         /* [num_layers x model_dim x model_dim] */
    float* Wo;         /* [num_layers x model_dim x model_dim] */
    float* norm1_g;    /* [num_layers x model_dim] */
    float* norm1_b;    /* [num_layers x model_dim] */
    float* W1;         /* [num_layers x model_dim x ff_dim] */
    float* b1;         /* [num_layers x ff_dim] */
    float* W2;         /* [num_layers x ff_dim x model_dim] */
    float* b2;         /* [num_layers x model_dim] */
    float* norm2_g;    /* [num_layers x model_dim] */
    float* norm2_b;    /* [num_layers x model_dim] */
    float* head_W;     /* [model_dim] scalar prediction head */
    float* head_b;     /* [1] */
} TransformerWeights;

/* Validate cfg and return its exact float count; false leaves count unchanged. */
int transformer_parameter_count(TransformerConfig cfg, size_t* count);

/* Return peak internal forward scratch in floats; false leaves count unchanged. */
int transformer_workspace_count(TransformerConfig cfg, size_t* count);

/* Initialize a fresh object and field views; return false on invalid cfg/OOM. */
int transformer_init(TransformerWeights* w, TransformerConfig cfg);

/* Release parameter storage and clear all field views; w may be NULL. */
void transformer_free(TransformerWeights* w);

/*
 * Encode x[seq_len x in_dim] into out[seq_len x model_dim].
 *   1. Project input and add positional encoding once.
 *   2. For each layer: h = h + attention(norm1(h)).
 *   3. For each layer: h = h + FFN(norm2(h)).
 * workspace holds transformer_workspace_count(cfg) floats and may be reused
 * across calls. Input, output, and workspace storage must be separate. This
 * returns hidden states, not a forecast.
 */
void transformer_forward(float* out, const float* x,
                         const TransformerWeights* w,
                         TransformerConfig cfg, float* workspace);

/* Apply the scalar head to the final hidden row; return the model-space target. */
float transformer_predict(const float* hidden,
                          const TransformerWeights* w,
                          TransformerConfig cfg);

#endif /* TRANSFORMER_H */
