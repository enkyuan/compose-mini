#ifndef TRANSFORMER_H
#define TRANSFORMER_H

/*
 * Encoder scaffold and hidden-state forward API.
 * The scalar forecasting head is not defined yet.
 */

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
 * Flat encoder weights. embed_W is shared; every block field must contain
 * num_layers consecutive parameter slices in the final artifact.
 * The current implementation allocates no weights and defines no forecast head.
 */
typedef struct {
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
} TransformerWeights;

/* Allocate all weight arrays sized by cfg; current implementation is a stub. */
void transformer_init(TransformerWeights* w, TransformerConfig cfg);

/* Release all weight arrays; current implementation is a stub. */
void transformer_free(TransformerWeights* w, TransformerConfig cfg);

/*
 * Encode x[seq_len x in_dim] into out[seq_len x model_dim].
 *   1. Project input and add positional encoding once.
 *   2. For each layer: norm1 -> attention -> residual.
 *   3. For each layer: norm2 -> FFN -> residual.
 * This returns hidden states, not a forecast. The prediction head will use the
 * final timestep to predict next-bar log return. The current implementation is
 * a stub.
 */
void transformer_forward(float* out, const float* x,
                         const TransformerWeights* w,
                         TransformerConfig cfg);

#endif /* TRANSFORMER_H */
