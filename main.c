#include <stdio.h>

/* Entry point — Phase 5 (inference loop) will:
 *   - parse CLI args for data path and model config
 *   - call data_load() to read and normalize OHLCV CSV
 *   - call transformer_init() to allocate model weights
 *   - run transformer_forward() over input windows
 *   - print next-step price predictions to stdout
 */
int main(void) {
    printf("compose-mini transformer scaffold\n");
    return 0;
}
