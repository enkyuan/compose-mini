#include "ffn.h"
#include "utils.h"

/* Position-wise feed-forward network; current implementation is a stub. */
void ffn_forward(float* out, const float* x,
                 const float* W1, const float* b1,
                 const float* W2, const float* b2,
                 int model_dim, int ff_dim) {
    (void)out; (void)x; (void)W1; (void)b1;
    (void)W2; (void)b2; (void)model_dim; (void)ff_dim;
}
