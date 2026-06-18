#include "transformer.h"
#include "attention.h"
#include "ffn.h"
#include "norm.h"
#include "embed.h"
#include "utils.h"

/* Phase 3: implement transformer_init — utils_alloc each weight field
 * sized according to cfg dimensions. Initialize to zeros. */
void transformer_init(TransformerWeights* w, TransformerConfig cfg) {
    (void)w; (void)cfg;
}

/* Phase 3: implement transformer_free — utils_free each weight field. */
void transformer_free(TransformerWeights* w, TransformerConfig cfg) {
    (void)w; (void)cfg;
}

/* Phase 3: implement transformer_forward.
 * Loop num_layers times:
 *   embed → norm1 → attention → residual → norm2 → ffn → residual
 * Allocate intermediate buffers with utils_alloc, free after each block. */
void transformer_forward(float* out, const float* x,
                         const TransformerWeights* w,
                         TransformerConfig cfg) {
    (void)out; (void)x; (void)w; (void)cfg;
}
