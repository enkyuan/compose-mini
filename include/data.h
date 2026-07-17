#ifndef DATA_H
#define DATA_H

/* Chronological OHLCV window preparation for inference. */

/*
 * Model-feature windows in oldest-to-newest order.
 * windows is [num_windows x seq_len x 5]. Request metadata and artifact-owned
 * scaling are not represented yet.
 */
typedef struct {
    float* windows;    /* [num_windows x seq_len x 5] */
    int    num_windows;
    int    seq_len;
    int    in_dim;     /* always 5 for OHLCV */
} DataSet;

/*
 * Planned CSV loader; current implementation is a stub.
 * Target CSV rows are timestamp,open,high,low,close,volume. Timestamp is
 * request metadata, not a model feature. The caller supplies completed bars
 * for one instrument and interval; this loader parses and windows them.
 * TODO(contract): accept the artifact's training-fitted scaler and retain each
 * window's as-of timestamp and raw close. Never fit a scaler on inference rows.
 */
DataSet data_load(const char* path, int seq_len);

/* Release DataSet storage; current implementation is a stub. */
void data_free(DataSet* ds);

#endif /* DATA_H */
