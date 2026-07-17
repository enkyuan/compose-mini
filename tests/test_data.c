#include <stdio.h>
#include "data.h"

static void test_data_load(void) {
    /*
     * Planned test: write 10 timestamped OHLCV rows, seq_len=3.
     * Supply fixed training scaler stats; assert 8 ordered windows and exact
     * transformed values, including values outside [0,1].
     */
    printf("test_data_load: skipped (implementation stub)\n");
}

static void test_data_free(void) {
    /* Load and free under a leak-enabled sanitizer. */
    printf("test_data_free: skipped (implementation stub)\n");
}

int main(void) {
    test_data_load();
    test_data_free();
    return 0;
}
