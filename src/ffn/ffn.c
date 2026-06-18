#include "ffn.h"
#include "utils.h"

/* Phase 3: implement ffn_forward.
 * Steps:
 *   1. h = matmul(x, W1) + b1        (model_dim → ff_dim)
 *   2. apply GELU pointwise: x*0.5*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
 *   3. out = matmul(h, W2) + b2       (ff_dim → model_dim)
 * Use utils_alloc for intermediate buffer h. */
void ffn_forward(float* out, const float* x,
                 const float* W1, const float* b1,
                 const float* W2, const float* b2,
                 int model_dim, int ff_dim) {
    (void)out; (void)x; (void)W1; (void)b1;
    (void)W2; (void)b2; (void)model_dim; (void)ff_dim;
}
