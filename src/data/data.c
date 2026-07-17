#include "data.h"
#include "utils.h"

/* CSV parsing and window construction; blocked on scaler/timestamp metadata. */
DataSet data_load(const char* path, int seq_len) {
    (void)path; (void)seq_len;
    DataSet ds = {0};
    return ds;
}

/* DataSet cleanup; current implementation is a stub. */
void data_free(DataSet* ds) { (void)ds; }
