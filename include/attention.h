#ifndef ATTENTION_H
#define ATTENTION_H

/* Phase 3 (components) implements all functions below.
 * Depends on: utils.h — matmul, softmax, transpose, alloc/free. */

/* Multi-head self-attention forward pass.
 *
 * For each head h (head_dim = model_dim / num_heads):
 *   Q[h] = x @ Wq[h]                         [seq_len x head_dim]
 *   K[h] = x @ Wk[h]                         [seq_len x head_dim]
 *   V[h] = x @ Wv[h]                         [seq_len x head_dim]
 *   scores = softmax(Q[h] @ K[h]^T / sqrt(head_dim))
 *   head[h] = scores @ V[h]
 * out = concat(head[0],...,head[n-1]) @ Wo    [seq_len x model_dim]
 *
 * Phase 3: implement single-head first, then generalize to multi-head.
 * Phase 6 (decoder): add causal mask (upper-triangular -inf) before softmax.
 * Wq, Wk, Wv each [model_dim x model_dim]; Wo [model_dim x model_dim]. */
void attention_forward(float* out, const float* x,
                       const float* Wq, const float* Wk,
                       const float* Wv, const float* Wo,
                       int seq_len, int model_dim, int num_heads);

#endif /* ATTENTION_H */
