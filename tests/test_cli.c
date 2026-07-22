#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "artifact_fixture.h"
#include "cli.h"

static const char* const MODEL_PATH = "bin/tests/cli-model.fixture";
static const char* const CSV_PATH = "bin/tests/cli-data.fixture.csv";
static const char* const VALID_CSV =
    "timestamp,open,high,low,close,volume\n"
    "2026-07-21T10:00:00Z,99,101,98,100,1000\n"
    "2026-07-21T11:00:00Z,100,102,99,101,1100\n"
    "2026-07-21T12:00:00Z,101,103,100,102,1200";
static const char* const LATE_OVERFLOW_CSV =
    "timestamp,open,high,low,close,volume\n"
    "2026-07-21T10:00:00Z,1,2,1,1,1000\n"
    "2026-07-21T11:00:00Z,2,3,1,2,1100\n"
    "2026-07-21T12:00:00Z,100,101,99,100,1200";

static void write_csv(const char* csv) {
    FILE* file = fopen(CSV_PATH, "w");
    assert(file);
    assert(fputs(csv, file) >= 0);
    assert(fclose(file) == 0);
}

static void read_stream(FILE* file, char* out, size_t capacity) {
    assert(fflush(file) == 0 && fseek(file, 0, SEEK_SET) == 0);
    const size_t size = fread(out, 1, capacity - 1, file);
    assert(!ferror(file) && fgetc(file) == EOF);
    out[size] = '\0';
}

static int run_cli(ArtifactFixtureKind kind, const char* csv,
                   const char* interval, const char* target_time,
                   char* output, char* error) {
    artifact_fixture_write(MODEL_PATH, kind);
    write_csv(csv);
    FILE* out = tmpfile();
    FILE* err = tmpfile();
    assert(out && err);
    char* argv[] = {
        "transformer", (char*)MODEL_PATH, (char*)CSV_PATH, "A\"B\\C",
        (char*)interval, (char*)target_time
    };
    const int status = cli_run(6, argv, out, err);
    read_stream(out, output, 2048);
    read_stream(err, error, 512);
    assert(fclose(out) == 0 && fclose(err) == 0);
    return status;
}

static void test_success(void) {
    char output[2048], error[512];
    assert(run_cli(ARTIFACT_FIXTURE_ZERO, VALID_CSV, "1h",
                   "2026-07-21T13:00:00Z", output, error) == CLI_OK);
    const char* expected =
        "{\"instrument\":\"A\\\"B\\\\C\",\"interval\":\"1h\","
        "\"as_of\":\"2026-07-21T11:00:00Z\","
        "\"target_time\":\"2026-07-21T12:00:00Z\",\"horizon_bars\":1,"
        "\"predicted_log_return\":0,\"predicted_close\":101,"
        "\"model_version\":\"unit-\\\"v1\\\\\"}\n"
        "{\"instrument\":\"A\\\"B\\\\C\",\"interval\":\"1h\","
        "\"as_of\":\"2026-07-21T12:00:00Z\","
        "\"target_time\":\"2026-07-21T13:00:00Z\",\"horizon_bars\":1,"
        "\"predicted_log_return\":0,\"predicted_close\":102,"
        "\"model_version\":\"unit-\\\"v1\\\\\"}\n";
    assert(!strcmp(output, expected) && !*error);
}

static void test_errors(void) {
    char output[2048], error[512];
    assert(run_cli(ARTIFACT_FIXTURE_ZERO, VALID_CSV, "5m",
                   "2026-07-21T13:00:00Z", output, error) == CLI_USAGE);
    assert(!*output && strstr(error, "interval"));

    assert(run_cli(ARTIFACT_FIXTURE_ZERO, VALID_CSV, "1h",
                   "2026-07-21T12:00:00Z", output, error) == CLI_USAGE);
    assert(!*output && strstr(error, "target time"));

    assert(run_cli(ARTIFACT_FIXTURE_HORIZON_TWO, VALID_CSV, "1h",
                   "2026-07-21T13:00:00Z", output, error) == CLI_ERROR);
    assert(!*output && strstr(error, "unsupported"));

    assert(run_cli(ARTIFACT_FIXTURE_OVERFLOW, VALID_CSV, "1h",
                   "2026-07-21T13:00:00Z", output, error) == CLI_ERROR);
    assert(!*output && strstr(error, "finite float"));

    assert(run_cli(ARTIFACT_FIXTURE_LATE_OVERFLOW, LATE_OVERFLOW_CSV, "1h",
                   "2026-07-21T13:00:00Z", output, error) == CLI_ERROR);
    assert(!*output && strstr(error, "finite float"));

    assert(run_cli(ARTIFACT_FIXTURE_ZERO, "bad\n", "1h",
                   "2026-07-21T13:00:00Z", output, error) == CLI_USAGE);
    assert(!*output && strstr(error, "invalid CSV"));
}

static void test_arguments(void) {
    FILE* out = tmpfile();
    FILE* err = tmpfile();
    assert(out && err);
    char* argv[] = {
        "transformer", "model", "data", "bad instrument", "1h",
        "2026-07-21T13:00:00Z"
    };
    assert(cli_run(1, argv, out, err) == CLI_USAGE);
    assert(cli_run(6, argv, out, err) == CLI_USAGE);
    argv[3] = "AAPL";
    argv[5] = "2026-02-30T13:00:00Z";
    assert(cli_run(6, argv, out, err) == CLI_USAGE);
    assert(fclose(out) == 0 && fclose(err) == 0);
}

int main(void) {
    test_success();
    test_errors();
    test_arguments();
    assert(remove(MODEL_PATH) == 0 && remove(CSV_PATH) == 0);
    printf("cli tests passed\n");
    return 0;
}
