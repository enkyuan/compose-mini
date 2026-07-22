#ifndef DATA_H
#define DATA_H

#include <stddef.h>
#include "artifact.h"

/* Validated chronological OHLCV rows and zero-copy inference windows. */

#define DATA_TIMESTAMP_CAP 21

typedef enum {
    DATA_OK = 0,
    DATA_ARGUMENT,
    DATA_IO,
    DATA_FORMAT,
    DATA_ORDER,
    DATA_RANGE,
    DATA_TOO_SHORT,
    DATA_NOMEM
} DataStatus;

typedef struct {
    void* storage;   /* sole owner of every row buffer below */
    float* features; /* scaled [num_rows x 5] */
    float* closes;   /* raw close per row */
    char (*timestamps)[DATA_TIMESTAMP_CAP];
    size_t num_rows;
    size_t num_windows;
    size_t seq_len;
} DataSet;

typedef struct {
    const float* features; /* borrowed [seq_len x 5] */
    const char* as_of;     /* borrowed final-row timestamp */
    float latest_close;    /* raw final-row close */
} DataWindow;

/* Parse, validate, and scale an exact timestamp,OHLCV CSV into a fresh ds. */
DataStatus data_load(DataSet* ds, const char* path,
                     const ModelArtifact* artifact);

/* Borrow window metadata and features; return false when index is invalid. */
int data_window(const DataSet* ds, size_t index, DataWindow* window);

/* Release row storage and clear the dataset; ds may be NULL. */
void data_free(DataSet* ds);

/* Return stable text for logs and CLI errors. */
const char* data_status_string(DataStatus status);

#endif /* DATA_H */
