#ifndef DATA_H
#define DATA_H

/* Phase 4 (data pipeline) implements all functions below.
 * Depends on: utils.h */

/* Holds normalized sliding windows of OHLCV data.
 * windows: flat array [num_windows x seq_len x 5]
 * Phase 4: populated by data_load.
 * Phase 5 (inference): iterate windows to feed transformer_forward. */
typedef struct {
    float* windows;    /* [num_windows x seq_len x 5] */
    int    num_windows;
    int    seq_len;
    int    in_dim;     /* always 5 for OHLCV */
} DataSet;

/* Load OHLCV CSV from path, min-max normalize each feature column,
 * then produce overlapping sliding windows of length seq_len.
 * CSV format: header row, then rows of open,high,low,close,volume.
 * Phase 4: fopen, parse with strtok/strtod, normalize, window.
 * Phase 6 (extensions): add stride parameter for non-overlapping windows. */
DataSet data_load(const char* path, int seq_len);

/* Free all memory allocated by data_load. */
void data_free(DataSet* ds);

#endif /* DATA_H */
