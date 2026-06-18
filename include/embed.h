#ifndef EMBED_H
#define EMBED_H

/* Phase 3 (components) implements all functions below.
 * Depends on: utils.h */

/* Project OHLCV input [seq_len x in_dim] → [seq_len x model_dim].
 * out[t] = x[t] @ W   where W is [in_dim x model_dim]
 * Phase 3: single matmul per timestep.
 * in_dim = 5 for OHLCV (open, high, low, close, volume). */
void embed_project(float* out, const float* x, const float* W,
                   int seq_len, int in_dim, int model_dim);

/* Add sinusoidal positional encoding to x in-place.
 * PE[pos][2i]   = sin(pos / 10000^(2i/model_dim))
 * PE[pos][2i+1] = cos(pos / 10000^(2i/model_dim))
 * Phase 3: implement the standard Vaswani et al. formula.
 * Phase 6 (extensions): learnable positional embeddings as alternative. */
void embed_positional(float* x, int seq_len, int model_dim);

#endif /* EMBED_H */
