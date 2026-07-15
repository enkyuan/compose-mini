#ifndef NORM_H
#define NORM_H

/* Layer normalization for individual encoder hidden-state vectors. */

/*
 * Normalize x[d] into out[d] with population variance and epsilon 1e-5,
 * then apply gamma[d] and beta[d]; d must be positive.
 */
void layernorm(float* out, const float* x,
               const float* gamma, const float* beta, int d);

#endif /* NORM_H */
