#ifndef FFN_H
#define FFN_H

/* Position-wise feed-forward network for encoder blocks. */

/*
 * Apply a two-layer feed-forward network to one hidden-state vector.
 * hidden = GELU(x @ W1 + b1)   [model_dim to ff_dim]
 * out    = hidden @ W2 + b2     [ff_dim to model_dim]
 *
 * ff_dim is typically 4 * model_dim (standard transformer ratio).
 * The current implementation is a stub.
 */
void ffn_forward(float* out, const float* x,
                 const float* W1, const float* b1,
                 const float* W2, const float* b2,
                 int model_dim, int ff_dim);

#endif /* FFN_H */
