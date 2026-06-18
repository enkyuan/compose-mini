#include "data.h"
#include "utils.h"

/* Phase 4: implement data_load.
 * Steps:
 *   1. fopen(path, "r"), read header line, count rows
 *   2. allocate raw float buffer [num_rows x 5]
 *   3. parse each row into buffer with strtok + strtod
 *   4. min-max normalize each of the 5 columns independently
 *   5. generate overlapping windows: num_windows = num_rows - seq_len + 1
 *   6. pack into ds.windows [num_windows x seq_len x 5]
 * Phase 6 (extensions): configurable stride, z-score normalization option. */
DataSet data_load(const char* path, int seq_len) {
    (void)path; (void)seq_len;
    DataSet ds = {0};
    return ds;
}

/* Phase 4: implement data_free — utils_free(ds->windows). */
void data_free(DataSet* ds) { (void)ds; }
