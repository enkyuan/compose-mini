#include "transformer.h"
#include "attention.h"
#include "ffn.h"
#include "norm.h"
#include "embed.h"
#include "utils.h"

/* Encoder-weight allocation; current implementation is a stub. */
void transformer_init(TransformerWeights* w, TransformerConfig cfg) {
    (void)w; (void)cfg;
}

/* Encoder-weight cleanup; current implementation is a stub. */
void transformer_free(TransformerWeights* w, TransformerConfig cfg) {
    (void)w; (void)cfg;
}

/* Encoder forward pass; current implementation is a stub. */
void transformer_forward(float* out, const float* x,
                         const TransformerWeights* w,
                         TransformerConfig cfg) {
    (void)out; (void)x; (void)w; (void)cfg;
}
