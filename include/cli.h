#ifndef CLI_H
#define CLI_H

#include <stdio.h>

/* Stable process results for embedding and command-line tests. */
typedef enum {
    CLI_OK = 0,
    CLI_ERROR = 1,
    CLI_USAGE = 2
} CliStatus;

/* Validate one request and emit oldest-to-newest JSONL; requires LC_NUMERIC=C. */
int cli_run(int argc, char* argv[], FILE* out, FILE* err);

#endif /* CLI_H */
