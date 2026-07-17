#ifndef ATTENTION_H
#define ATTENTION_H

/* Multi-head self-attention over historical context. */

/*
 * Compute self-attention for x[seq_len x model_dim].
 *
 * Q = x @ Wq, K = x @ Wk, V = x @ Wv        [seq_len x model_dim]
 * Split each projection into num_heads feature slices. For each head h:
 *   scores = softmax(Q[h] @ K[h]^T / sqrt(head_dim))
 *   head[h] = scores @ V[h]
 * out = concat(head[0],...,head[n-1]) @ Wo    [seq_len x model_dim]
 *
 * V1 predicts after the final completed bar, so full context attention is safe.
 * A per-position forecasting objective must mask later bars.
 * Wq, Wk, Wv, and Wo are [model_dim x model_dim]; model_dim must be divisible
 * by num_heads. The current implementation is a stub.
 */
void attention_forward(float* out, const float* x,
                       const float* Wq, const float* Wk,
                       const float* Wv, const float* Wo,
                       int seq_len, int model_dim, int num_heads);

#endif /* ATTENTION_H */
