#include "attention.h"
#include "utils.h"

/* Phase 3: implement attention_forward.
 * Steps:
 *   1. Split x into num_heads slices (head_dim = model_dim/num_heads)
 *   2. For each head: compute Q=xWq, K=xWk, V=xWv
 *   3. scores = Q @ K^T, scale by 1/sqrt(head_dim)
 *   4. softmax(scores) row-wise
 *   5. context = scores @ V
 *   6. concat all heads, project through Wo
 * Use utils_alloc for Q, K, V, scores buffers; utils_free after use. */
void attention_forward(float* out, const float* x,
                       const float* Wq, const float* Wk,
                       const float* Wv, const float* Wo,
                       int seq_len, int model_dim, int num_heads) {
    (void)out; (void)x; (void)Wq; (void)Wk;
    (void)Wv; (void)Wo; (void)seq_len; (void)model_dim; (void)num_heads;
}
