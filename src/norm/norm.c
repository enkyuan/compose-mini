/* Per-vector layer normalization for encoder hidden states. */

#include <assert.h>
#include <math.h>
#include "norm.h"

void layernorm(float* out, const float* x,
               const float* gamma, const float* beta, int d) {
    assert(out && x && gamma && beta && d > 0);

    float mean = 0.0f, variance = 0.0f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= (float)d;

    /* Center first to avoid cancellation in mean(x^2) - mean(x)^2. */
    for (int i = 0; i < d; i++) {
        const float centered = x[i] - mean;
        variance += centered * centered;
    }

    /* Epsilon prevents division by zero when variance is zero. */
    const float inv_std = 1.0f / sqrtf(variance / (float)d + 1e-5f);
    for (int i = 0; i < d; i++)
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
}
