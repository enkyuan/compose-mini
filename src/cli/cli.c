/* Artifact-backed batch inference and JSONL emission. */

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "artifact.h"
#include "cli.h"
#include "data.h"
#include "utils.h"

enum {
    ARG_PROGRAM,
    ARG_MODEL,
    ARG_CSV,
    ARG_INSTRUMENT,
    ARG_INTERVAL,
    ARG_TARGET_TIME,
    ARG_COUNT,
    INSTRUMENT_CAP = 64
};

static int usage(FILE* err) {
    fputs("usage: transformer MODEL CSV INSTRUMENT INTERVAL FINAL_TARGET_TIME\n",
          err);
    return CLI_USAGE;
}

static int instrument_valid(const char* value) {
    size_t length = 0;
    if (!value) return 0;
    while (value[length]) {
        const unsigned char byte = (unsigned char)value[length];
        if (length + 1 == INSTRUMENT_CAP || byte < 33 || byte > 126) return 0;
        length++;
    }
    return length > 0;
}

static int json_string(FILE* out, const char* value) {
    if (fputc('"', out) == EOF) return 0;
    for (; *value; value++) {
        if ((*value == '"' || *value == '\\') && fputc('\\', out) == EOF)
            return 0;
        if (fputc((unsigned char)*value, out) == EOF) return 0;
    }
    return fputc('"', out) != EOF;
}

static int emit_forecast(FILE* out, const char* instrument,
                         const ModelArtifact* artifact,
                         const DataWindow* window, const char* target_time,
                         float log_return, float predicted_close) {
    if (fputs("{\"instrument\":", out) == EOF ||
        !json_string(out, instrument) || fputs(",\"interval\":", out) == EOF ||
        !json_string(out, artifact->interval) || fputs(",\"as_of\":", out) == EOF ||
        !json_string(out, window->as_of) || fputs(",\"target_time\":", out) == EOF ||
        !json_string(out, target_time))
        return 0;
    if (fprintf(out, ",\"horizon_bars\":%" PRIu32
                     ",\"predicted_log_return\":%.9g"
                     ",\"predicted_close\":%.9g,\"model_version\":",
                artifact->horizon_bars, log_return, predicted_close) < 0 ||
        !json_string(out, artifact->model_version))
        return 0;
    return fputs("}\n", out) != EOF;
}

int cli_run(int argc, char* argv[], FILE* out, FILE* err) {
    if (!argv || !out || !err) return CLI_ERROR;
    if (argc != ARG_COUNT || !instrument_valid(argv[ARG_INSTRUMENT]) ||
        !instrument_valid(argv[ARG_INTERVAL]) ||
        !data_timestamp_valid(argv[ARG_TARGET_TIME]))
        return usage(err);
    if (!utils_c_numeric_locale()) {
        fputs("LC_NUMERIC must be C\n", err);
        return CLI_ERROR;
    }

    ModelArtifact artifact = {0};
    DataSet data = {0};
    float* buffers = NULL;
    int result = CLI_ERROR;

    const ArtifactStatus artifact_status =
        artifact_load(&artifact, argv[ARG_MODEL]);
    if (artifact_status != ARTIFACT_OK) {
        fprintf(err, "artifact: %s\n", artifact_status_string(artifact_status));
        goto done;
    }
    if (strcmp(argv[ARG_INTERVAL], artifact.interval)) {
        fputs("interval does not match artifact\n", err);
        result = CLI_USAGE;
        goto done;
    }

    const DataStatus data_status = data_load(&data, argv[ARG_CSV], &artifact);
    if (data_status != DATA_OK) {
        fprintf(err, "data: %s\n", data_status_string(data_status));
        if (data_status != DATA_IO && data_status != DATA_NOMEM &&
            data_status != DATA_LOCALE)
            result = CLI_USAGE;
        goto done;
    }
    if (strcmp(argv[ARG_TARGET_TIME], data.timestamps[data.num_rows - 1]) <= 0) {
        fputs("final target time must follow the final bar\n", err);
        result = CLI_USAGE;
        goto done;
    }

    size_t workspace_count;
    if (!transformer_workspace_count(artifact.config, &workspace_count)) {
        fputs("artifact: invalid workspace dimensions\n", err);
        goto done;
    }
    const size_t hidden_count =
        (size_t)artifact.config.seq_len * (size_t)artifact.config.model_dim;
    if (hidden_count > SIZE_MAX - workspace_count) {
        fputs("artifact: workspace too large\n", err);
        goto done;
    }
    size_t buffer_count = hidden_count + workspace_count;
    if (data.num_windows > (SIZE_MAX - buffer_count) / 2) {
        fputs("data: too many windows\n", err);
        goto done;
    }
    buffer_count += 2 * data.num_windows;
    if (buffer_count > SIZE_MAX / sizeof *buffers) {
        fputs("artifact: workspace too large\n", err);
        goto done;
    }
    buffers = calloc(buffer_count, sizeof *buffers);
    if (!buffers) {
        fputs("out of memory\n", err);
        goto done;
    }
    float* workspace = buffers + hidden_count;
    float* forecasts = workspace + workspace_count;

    for (size_t i = 0; i < data.num_windows; i++) {
        DataWindow window;
        if (!data_window(&data, i, &window)) {
            fputs("internal window error\n", err);
            goto done;
        }
        transformer_forward(buffers, window.features, &artifact.weights,
                            artifact.config, workspace);
        const float log_return = artifact_unscale_target(
            transformer_predict(buffers, &artifact.weights, artifact.config),
            &artifact);
        const float predicted_close = window.latest_close * expf(log_return);
        if (!isfinite(log_return) || !isfinite(predicted_close) ||
            predicted_close <= 0.0f) {
            fputs("forecast is outside the finite float range\n", err);
            goto done;
        }
        forecasts[2 * i] = log_return;
        forecasts[2 * i + 1] = predicted_close;
    }

    for (size_t i = 0; i < data.num_windows; i++) {
        DataWindow window;
        if (!data_window(&data, i, &window)) {
            fputs("internal window error\n", err);
            goto done;
        }
        const char* target_time = i + 1 < data.num_windows ?
            data.timestamps[i + data.seq_len] : argv[ARG_TARGET_TIME];
        if (!emit_forecast(out, argv[ARG_INSTRUMENT], &artifact, &window,
                           target_time, forecasts[2 * i], forecasts[2 * i + 1])) {
            fputs("failed to write forecast output\n", err);
            goto done;
        }
    }
    if (fflush(out)) {
        fputs("failed to flush forecast output\n", err);
        goto done;
    }
    result = CLI_OK;

done:
    free(buffers);
    data_free(&data);
    artifact_free(&artifact);
    return result;
}
