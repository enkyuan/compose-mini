#ifndef EMBED_H
#define EMBED_H

/* Projection from scaled OHLCV features into encoder hidden states. */

/*
 * Project scaled OHLCV input [seq_len x in_dim] to [seq_len x model_dim].
 * out[t] = x[t] @ W   where W is [in_dim x model_dim]
 * in_dim = 5 for OHLCV; timestamps remain request metadata. The current
 * implementation is a stub.
 */
void embed_project(float* out, const float* x, const float* W,
                   int seq_len, int in_dim, int model_dim);

/*
 * Add sinusoidal positional encoding to x in place.
 * PE[pos][2i]   = sin(pos / 10000^(2i/model_dim))
 * PE[pos][2i+1] = cos(pos / 10000^(2i/model_dim))
 * The current implementation is a stub.
 */
void embed_positional(float* x, int seq_len, int model_dim);

#endif /* EMBED_H */
