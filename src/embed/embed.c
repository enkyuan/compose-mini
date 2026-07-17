#include "embed.h"
#include "utils.h"

/* Input projection; current implementation is a stub. */
void embed_project(float* out, const float* x, const float* W,
                   int seq_len, int in_dim, int model_dim) {
    (void)out; (void)x; (void)W; (void)seq_len; (void)in_dim; (void)model_dim;
}

/* Sinusoidal positional encoding; current implementation is a stub. */
void embed_positional(float* x, int seq_len, int model_dim) {
    (void)x; (void)seq_len; (void)model_dim;
}
