/* Strict CSV validation and zero-copy chronological window construction. */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "utils.h"

enum { LINE_CAP = 512 };

static const char* const CSV_HEADER =
    "timestamp,open,high,low,close,volume";

static DataStatus read_line(FILE* file, char* line, int* found) {
    size_t length = 0;
    int byte;
    while ((byte = fgetc(file)) != EOF && byte != '\n') {
        if (!byte || length + 1 == LINE_CAP) return DATA_FORMAT;
        line[length++] = (char)byte;
    }
    if (byte == EOF && ferror(file)) {
        *found = 0;
        return DATA_IO;
    }
    if (byte == EOF && !length) {
        *found = 0;
        return DATA_OK;
    }
    if (length && line[length - 1] == '\r') line[--length] = '\0';
    else line[length] = '\0';
    *found = 1;
    return length ? DATA_OK : DATA_FORMAT;
}

static DataStatus read_header(FILE* file) {
    char line[LINE_CAP];
    int found;
    const DataStatus status = read_line(file, line, &found);
    if (status == DATA_OK && found && !strcmp(line, CSV_HEADER)) return DATA_OK;
    return status == DATA_OK ? DATA_FORMAT : status;
}

static DataStatus open_csv(FILE** out, const char* path) {
    *out = fopen(path, "r");
    if (!*out) return DATA_IO;
    const DataStatus status = read_header(*out);
    if (status != DATA_OK) {
        fclose(*out);
        *out = NULL;
    }
    return status;
}

int data_timestamp_valid(const char* value) {
    if (!value) return 0;
    static const int month_days[] =
        {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (strlen(value) != 20 || value[4] != '-' || value[7] != '-' ||
        value[10] != 'T' || value[13] != ':' || value[16] != ':' ||
        value[19] != 'Z')
        return 0;
    for (int i = 0; i < 20; i++)
        if (i != 4 && i != 7 && i != 10 && i != 13 && i != 16 && i != 19 &&
            !isdigit((unsigned char)value[i]))
            return 0;

    const int year = atoi(value), month = atoi(value + 5), day = atoi(value + 8);
    const int hour = atoi(value + 11), minute = atoi(value + 14);
    const int second = atoi(value + 17);
    if (year < 1 || month < 1 || month > 12 || hour > 23 || minute > 59 ||
        second > 59)
        return 0;
    int days = month_days[month];
    if (month == 2 && (year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)))
        days++;
    return day >= 1 && day <= days;
}

static DataStatus split_row(char* line, char** fields) {
    fields[0] = line;
    for (int i = 1; i < 6; i++) {
        char* comma = strchr(fields[i - 1], ',');
        if (!comma) return DATA_FORMAT;
        *comma = '\0';
        fields[i] = comma + 1;
    }
    if (strchr(fields[5], ',')) return DATA_FORMAT;
    for (int i = 0; i < 6; i++) if (!*fields[i]) return DATA_FORMAT;
    return DATA_OK;
}

static DataStatus parse_number(const char* field, float* value) {
    if (isspace((unsigned char)*field)) return DATA_FORMAT;
    errno = 0;
    char* end;
    *value = strtof(field, &end);
    if (end == field || *end) return DATA_FORMAT;
    return errno == ERANGE || !isfinite(*value) ? DATA_RANGE : DATA_OK;
}

typedef struct {
    char timestamp[DATA_TIMESTAMP_CAP];
    float features[ARTIFACT_FEATURE_COUNT];
} ParsedRow;

static DataStatus parse_row(char* line, const char* previous, ParsedRow* row) {
    char* fields[6];
    DataStatus status = split_row(line, fields);
    if (status != DATA_OK || !data_timestamp_valid(fields[0])) return DATA_FORMAT;
    if (*previous && strcmp(previous, fields[0]) >= 0) return DATA_ORDER;
    for (int i = 0; i < ARTIFACT_FEATURE_COUNT; i++) {
        status = parse_number(fields[i + 1], row->features + i);
        if (status != DATA_OK) return status;
    }
    if (row->features[3] <= 0.0f) return DATA_RANGE;
    memcpy(row->timestamp, fields[0], DATA_TIMESTAMP_CAP);
    return DATA_OK;
}

static DataStatus count_rows(FILE* file, size_t* rows) {
    char line[LINE_CAP], previous[DATA_TIMESTAMP_CAP] = {0};
    DataStatus status = DATA_OK;
    int found;
    *rows = 0;
    while ((status = read_line(file, line, &found)) == DATA_OK && found) {
        ParsedRow row;
        status = parse_row(line, previous, &row);
        if (status != DATA_OK) break;
        if (*rows == SIZE_MAX) return DATA_RANGE;
        memcpy(previous, row.timestamp, DATA_TIMESTAMP_CAP);
        (*rows)++;
    }
    return status;
}

static DataStatus parse_rows(DataSet* ds, FILE* file) {
    char line[LINE_CAP], previous[DATA_TIMESTAMP_CAP] = {0};
    size_t row = 0;
    DataStatus status = DATA_OK;
    int found;
    while ((status = read_line(file, line, &found)) == DATA_OK && found) {
        if (row == ds->num_rows) {
            status = DATA_FORMAT;
            break;
        }
        ParsedRow parsed;
        status = parse_row(line, previous, &parsed);
        if (status != DATA_OK) break;
        float* features = ds->features + row * ARTIFACT_FEATURE_COUNT;
        memcpy(features, parsed.features, sizeof parsed.features);
        memcpy(ds->timestamps[row], parsed.timestamp, DATA_TIMESTAMP_CAP);
        memcpy(previous, parsed.timestamp, DATA_TIMESTAMP_CAP);
        ds->closes[row++] = parsed.features[3];
    }
    if (status == DATA_OK && row != ds->num_rows) status = DATA_FORMAT;
    return status;
}

DataStatus data_load(DataSet* ds, const char* path,
                     const ModelArtifact* artifact) {
    if (!ds) return DATA_ARGUMENT;
    *ds = (DataSet){0};
    if (!path || !artifact || artifact->config.seq_len <= 0 ||
        artifact->config.in_dim != ARTIFACT_FEATURE_COUNT)
        return DATA_ARGUMENT;
    if (!utils_c_numeric_locale()) return DATA_LOCALE;

    FILE* file;
    DataStatus status = open_csv(&file, path);
    if (status != DATA_OK) return status;
    size_t rows;
    status = count_rows(file, &rows);
    const size_t seq_len = (size_t)artifact->config.seq_len;
    if (status == DATA_OK && rows < seq_len) status = DATA_TOO_SHORT;
    const size_t row_bytes = 6 * sizeof(float) + DATA_TIMESTAMP_CAP;
    if (status == DATA_OK && rows > SIZE_MAX / row_bytes) status = DATA_RANGE;

    if (status == DATA_OK) {
        ds->storage = malloc(rows * row_bytes);
        if (!ds->storage) status = DATA_NOMEM;
    }
    if (status == DATA_OK) {
        ds->features = ds->storage;
        ds->closes = ds->features + rows * ARTIFACT_FEATURE_COUNT;
        ds->timestamps = (char (*)[DATA_TIMESTAMP_CAP])(ds->closes + rows);
        ds->num_rows = rows;
        ds->num_windows = rows - seq_len + 1;
        ds->seq_len = seq_len;
        if (fseek(file, 0, SEEK_SET)) status = DATA_IO;
        else if ((status = read_header(file)) == DATA_OK)
            status = parse_rows(ds, file);
    }

    if (status == DATA_OK &&
        !artifact_scale_features(ds->features, rows, artifact))
        status = DATA_RANGE;
    if (fclose(file) && status == DATA_OK) status = DATA_IO;
    if (status != DATA_OK) data_free(ds);
    return status;
}

int data_window(const DataSet* ds, size_t index, DataWindow* window) {
    if (!ds || !window || !ds->storage || !ds->seq_len ||
        index >= ds->num_windows)
        return 0;
    const size_t final_row = index + ds->seq_len - 1;
    *window = (DataWindow){
        ds->features + index * ARTIFACT_FEATURE_COUNT,
        ds->timestamps[final_row],
        ds->closes[final_row]
    };
    return 1;
}

void data_free(DataSet* ds) {
    if (!ds) return;
    free(ds->storage);
    *ds = (DataSet){0};
}

const char* data_status_string(DataStatus status) {
    static const char* const messages[] = {
        "ok", "invalid argument", "LC_NUMERIC must be C", "I/O error", "invalid CSV",
        "timestamps out of order", "value out of range", "not enough rows",
        "out of memory"
    };
    return (unsigned)status < sizeof messages / sizeof *messages ?
        messages[status] : "unknown data error";
}
