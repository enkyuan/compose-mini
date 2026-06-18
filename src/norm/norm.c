#include "norm.h"
#include "utils.h"

/* Phase 3: implement layernorm.
 * Steps:
 *   1. mean = sum(x) / d
 *   2. var  = sum((x[i]-mean)^2) / d
 *   3. x_hat[i] = (x[i]-mean) / sqrt(var + 1e-5)
 *   4. out[i] = gamma[i] * x_hat[i] + beta[i]
 * Phase 6 (extensions): fused kernel for better perf. */
void layernorm(float* out, const float* x,
               const float* gamma, const float* beta, int d) {
    (void)out; (void)x; (void)gamma; (void)beta; (void)d;
}
