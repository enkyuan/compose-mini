#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "data.h"

static void write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "w");
    assert(file && fputs(text, file) >= 0 && fclose(file) == 0);
}

static void write_bytes(const char* path, const void* bytes, size_t size) {
    FILE* file = fopen(path, "wb");
    assert(file && fwrite(bytes, 1, size, file) == size && fclose(file) == 0);
}

static ModelArtifact test_artifact(void) {
    ModelArtifact artifact = {.config = {.seq_len = 3, .in_dim = 5}};
    const float mean[] = {100, 200, 90, 95, 1000};
    const float scale[] = {10, 10, 10, 5, 100};
    memcpy(artifact.feature_mean, mean, sizeof mean);
    memcpy(artifact.feature_scale, scale, sizeof scale);
    return artifact;
}

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-6f);
}

static void assert_load_status(const char* path, const ModelArtifact* artifact,
                               DataStatus expected) {
    DataSet ds;
    assert(data_load(&ds, path, artifact) == expected);
    assert(!ds.storage);
}

static void test_valid(const char* path) {
    write_text(path,
        "timestamp,open,high,low,close,volume\n"
        "2028-02-29T10:00:00Z,100,201,90,95,1000\n"
        "2028-02-29T11:00:00Z,110,211,100,100,1100\n"
        "2028-02-29T12:00:00Z,120,221,110,105,1200\n"
        "2028-02-29T13:00:00Z,130,231,120,110,1300");
    const ModelArtifact artifact = test_artifact();
    DataSet ds;
    assert(data_load(&ds, path, &artifact) == DATA_OK);
    assert(ds.num_rows == 4 && ds.num_windows == 2 && ds.seq_len == 3);

    DataWindow first, second;
    assert(data_window(&ds, 0, &first) && data_window(&ds, 1, &second));
    assert(second.features == first.features + ARTIFACT_FEATURE_COUNT);
    for (size_t row = 0; row < 4; row++) {
        assert_close(ds.features[row * 5], (float)row);
        assert_close(ds.features[row * 5 + 1], (float)row + .1f);
    }
    assert(!strcmp(first.as_of, "2028-02-29T12:00:00Z"));
    assert(!strcmp(second.as_of, "2028-02-29T13:00:00Z"));
    assert_close(first.latest_close, 105.0f);
    assert_close(second.latest_close, 110.0f);
    assert(!data_window(&ds, 2, &first));

    data_free(&ds);
    assert(!ds.storage && !ds.num_rows);
}

static void test_invalid(const char* path) {
    const ModelArtifact artifact = test_artifact();
    const struct {
        const char* csv;
        DataStatus status;
    } cases[] = {
        {"bad,header\n", DATA_FORMAT},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,4,5\n", DATA_TOO_SHORT},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,4,5\n"
         "2026-07-21T09:00:00Z,1,2,3,4,5\n", DATA_ORDER},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,4,5\n"
         "2026-07-21T10:00:00Z,1,2,3,4,5\n", DATA_ORDER},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,4,5\n"
         "2026-07-21T09:00:00Z,1,2,3,4,5\n"
         "2026-07-21T11:00:00Z,1,2,3,4,5\n", DATA_ORDER},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,nan,5\n"
         "2026-07-21T11:00:00Z,1,2,3,4,5\n"
         "2026-07-21T12:00:00Z,1,2,3,4,5\n", DATA_RANGE},
        {"timestamp,open,high,low,close,volume\n"
         "2026-02-30T10:00:00Z,1,2,3,4,5\n"
         "2026-03-01T11:00:00Z,1,2,3,4,5\n"
         "2026-03-01T12:00:00Z,1,2,3,4,5\n", DATA_FORMAT},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z, 1,2,3,4,5\n"
         "2026-07-21T11:00:00Z,1,2,3,4,5\n"
         "2026-07-21T12:00:00Z,1,2,3,4,5\n", DATA_FORMAT},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,0,5\n"
         "2026-07-21T11:00:00Z,1,2,3,4,5\n"
         "2026-07-21T12:00:00Z,1,2,3,4,5\n", DATA_RANGE},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,4\n"
         "2026-07-21T11:00:00Z,1,2,3,4,5\n"
         "2026-07-21T12:00:00Z,1,2,3,4,5\n", DATA_FORMAT},
        {"timestamp,open,high,low,close,volume\n"
         "2026-07-21T10:00:00Z,1,2,3,4,5,6\n"
         "2026-07-21T11:00:00Z,1,2,3,4,5\n"
         "2026-07-21T12:00:00Z,1,2,3,4,5\n", DATA_FORMAT}
    };

    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        write_text(path, cases[i].csv);
        assert_load_status(path, &artifact, cases[i].status);
    }

    const char bytes[] =
        "timestamp,open,high,low,close,volume\n"
        "2026-07-21T10:00:00Z,1,2,3,4,5\0junk\n";
    write_bytes(path, bytes, sizeof bytes - 1);
    assert_load_status(path, &artifact, DATA_FORMAT);

    char overlong[1024];
    const char* header = "timestamp,open,high,low,close,volume\n";
    const size_t header_size = strlen(header);
    memcpy(overlong, header, header_size);
    memset(overlong + header_size, '1', sizeof overlong - header_size);
    write_bytes(path, overlong, sizeof overlong);
    assert_load_status(path, &artifact, DATA_FORMAT);

    int marker;
    DataSet ds = {.storage = &marker};
    assert(data_load(&ds, NULL, &artifact) == DATA_ARGUMENT && !ds.storage);
    assert(data_load(&ds, path, NULL) == DATA_ARGUMENT && !ds.storage);
    assert(data_load(NULL, path, &artifact) == DATA_ARGUMENT);
}

int main(void) {
    const char* path = "bin/tests/data.fixture.csv";
    test_valid(path);
    test_invalid(path);
    assert(remove(path) == 0);
    assert(!strcmp(data_status_string(DATA_ORDER), "timestamps out of order"));
    assert(!strcmp(data_status_string((DataStatus)99), "unknown data error"));
    data_free(NULL);
    printf("data tests passed\n");
    return 0;
}
