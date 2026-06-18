#ifndef NORM_H
#define NORM_H

/* Phase 3 (components) implements all functions below.
 * Depends on: utils.h */

/* Layer normalization over vector x of dimension d.
 * out[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
 * Phase 3: compute mean, variance, normalize, scale+shift.
 * eps = 1e-5 for numerical stability. */
void layernorm(float* out, const float* x,
               const float* gamma, const float* beta, int d);

#endif /* NORM_H */
