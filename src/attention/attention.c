#include "attention.h"
#include "utils.h"

/* Historical-context self-attention; current implementation is a stub. */
void attention_forward(float* out, const float* x,
                       const float* Wq, const float* Wk,
                       const float* Wv, const float* Wo,
                       int seq_len, int model_dim, int num_heads) {
    (void)out; (void)x; (void)Wq; (void)Wk;
    (void)Wv; (void)Wo; (void)seq_len; (void)model_dim; (void)num_heads;
}
