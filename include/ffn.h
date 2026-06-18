#ifndef FFN_H
#define FFN_H

/* Phase 3 (components) implements all functions below.
 * Depends on: utils.h */

/* Two-layer feed-forward network (per position).
 * hidden = GELU(x @ W1 + b1)   [model_dim → ff_dim]
 * out    = hidden @ W2 + b2     [ff_dim → model_dim]
 *
 * ff_dim is typically 4 * model_dim (standard transformer ratio).
 * Phase 3: implement with matmul + pointwise GELU.
 * Phase 6 (extensions): swap GELU for ReLU via compile flag. */
void ffn_forward(float* out, const float* x,
                 const float* W1, const float* b1,
                 const float* W2, const float* b2,
                 int model_dim, int ff_dim);

#endif /* FFN_H */
