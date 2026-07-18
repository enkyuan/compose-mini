#ifndef FFN_H
#define FFN_H

/* Position-wise feed-forward network for encoder blocks. */

/*
 * Apply a two-layer feed-forward network to one hidden-state vector.
 * hidden = GELU(x @ W1 + b1)   [model_dim to ff_dim]
 * out    = hidden @ W2 + b2     [ff_dim to model_dim]
 * GELU(z) = 0.5 * z * (1 + erf(z / sqrt(2))).
 *
 * Weights are row-major; out must not alias inputs; dimensions must be positive.
 * ff_dim is typically 4 * model_dim.
 */
void ffn_forward(float* out, const float* x,
                 const float* W1, const float* b1,
                 const float* W2, const float* b2,
                 int model_dim, int ff_dim);

#endif /* FFN_H */
