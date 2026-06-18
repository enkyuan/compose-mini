#include <stdio.h>
#include "data.h"

static void test_data_load(void) {
    /* Phase 4: write a temp CSV with 10 rows of OHLCV, seq_len=3.
     * Assert num_windows=8, assert values are normalized to [0,1]. */
    printf("test_data_load: stub\n");
}

static void test_data_free(void) {
    /* Phase 4: load then free, run under valgrind to assert no leaks. */
    printf("test_data_free: stub\n");
}

int main(void) {
    test_data_load();
    test_data_free();
    printf("data tests: all stubs passed\n");
    return 0;
}
