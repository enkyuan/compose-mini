#include "embed.h"
#include "utils.h"

/* Phase 3: implement embed_project — call matmul(out, x, W, seq_len, in_dim, model_dim). */
void embed_project(float* out, const float* x, const float* W,
                   int seq_len, int in_dim, int model_dim) {
    (void)out; (void)x; (void)W; (void)seq_len; (void)in_dim; (void)model_dim;
}

/* Phase 3: implement embed_positional — nested loop over pos and i,
 * computing sin/cos with pow(10000, 2i/model_dim) denominator. */
void embed_positional(float* x, int seq_len, int model_dim) {
    (void)x; (void)seq_len; (void)model_dim;
}
