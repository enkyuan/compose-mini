/* Streaming decoder for the checksummed V1 model artifact. */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "artifact.h"

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
               FLT_MAX_EXP == 128,
               "artifact V1 requires IEEE-754 binary32 float");

enum {
    HEADER_SIZE = 32,
    BODY_PREFIX_SIZE = 160,
    FLOAT_CHUNK = 1024
};

static const uint8_t MAGIC[8] = {'C', 'M', 'P', 'M', 'I', 'N', 'I', 0};
static const uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
static const uint64_t FNV_PRIME = UINT64_C(1099511628211);

typedef struct {
    FILE* file;
    uint64_t remaining;
    uint64_t hash;
} BodyReader;

static uint32_t decode_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 |
           (uint32_t)bytes[3] << 24;
}

static uint64_t decode_u64(const uint8_t* bytes) {
    return (uint64_t)decode_u32(bytes) |
           (uint64_t)decode_u32(bytes + 4) << 32;
}

static float decode_f32(const uint8_t* bytes) {
    const uint32_t bits = decode_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static void hash_bytes(uint64_t* hash, const uint8_t* bytes, size_t n) {
    for (size_t i = 0; i < n; i++) *hash = (*hash ^ bytes[i]) * FNV_PRIME;
}

static ArtifactStatus read_file(FILE* file, void* out, size_t n) {
    if (fread(out, 1, n, file) == n) return ARTIFACT_OK;
    return ferror(file) ? ARTIFACT_IO : ARTIFACT_TRUNCATED;
}

static ArtifactStatus read_body(BodyReader* reader, void* out, size_t n) {
    if ((uint64_t)n > reader->remaining) return ARTIFACT_TRUNCATED;
    const ArtifactStatus status = read_file(reader->file, out, n);
    if (status != ARTIFACT_OK) return status;
    hash_bytes(&reader->hash, out, n);
    reader->remaining -= (uint64_t)n;
    return ARTIFACT_OK;
}

static ArtifactStatus validate_file_size(FILE* file, uint64_t body_size) {
    if (body_size > (uint64_t)LONG_MAX - HEADER_SIZE) return ARTIFACT_RANGE;
    if (fseek(file, 0, SEEK_END)) return ARTIFACT_IO;
    const long size = ftell(file);
    if (size < 0) return ARTIFACT_IO;
    const uint64_t expected = HEADER_SIZE + body_size;
    if ((uint64_t)size != expected)
        return (uint64_t)size < expected ? ARTIFACT_TRUNCATED : ARTIFACT_FORMAT;
    return fseek(file, HEADER_SIZE, SEEK_SET) ? ARTIFACT_IO : ARTIFACT_OK;
}

static ArtifactStatus read_weights(BodyReader* reader, float* out, size_t n,
                                   int* all_finite) {
    uint8_t bytes[FLOAT_CHUNK * sizeof(float)];
    while (n) {
        const size_t count = n < FLOAT_CHUNK ? n : FLOAT_CHUNK;
        const ArtifactStatus status = read_body(reader, bytes,
                                                count * sizeof(float));
        if (status != ARTIFACT_OK) return status;
        for (size_t i = 0; i < count; i++) {
            out[i] = decode_f32(bytes + i * sizeof(float));
            *all_finite &= isfinite(out[i]);
        }
        out += count;
        n -= count;
    }
    return ARTIFACT_OK;
}

static ArtifactStatus decode_config(const uint8_t* body,
                                    TransformerConfig* config,
                                    size_t* parameter_count) {
    uint32_t values[6];
    for (size_t i = 0; i < 6; i++) {
        values[i] = decode_u32(body + i * sizeof(uint32_t));
        if (!values[i] || values[i] > INT_MAX) return ARTIFACT_RANGE;
    }

    *config = (TransformerConfig){
        (int)values[0], (int)values[1], (int)values[2],
        (int)values[3], (int)values[4], (int)values[5]
    };
    if (config->in_dim != ARTIFACT_FEATURE_COUNT)
        return ARTIFACT_UNSUPPORTED;
    size_t workspace;
    return transformer_parameter_count(*config, parameter_count) &&
           transformer_workspace_count(*config, &workspace) &&
           workspace <= ARTIFACT_MAX_WORKSPACE_FLOATS ?
        ARTIFACT_OK : ARTIFACT_RANGE;
}

static int decode_token(char* out, const uint8_t* slot, size_t size) {
    size_t length = 0;
    while (length < size && slot[length]) {
        if (slot[length] < 33 || slot[length] > 126) return 0;
        length++;
    }
    if (!length || length == size) return 0;
    for (size_t i = length + 1; i < size; i++)
        if (slot[i]) return 0;
    memcpy(out, slot, length);
    out[length] = '\0';
    return 1;
}

static ArtifactStatus validate_metadata(ModelArtifact* artifact,
                                        const uint8_t* body,
                                        int weights_finite) {
    artifact->horizon_bars = decode_u32(body + 24);
    if (!artifact->horizon_bars || decode_u32(body + 28) || !weights_finite)
        return ARTIFACT_FORMAT;
    if (!decode_token(artifact->model_version, body + 32,
                      ARTIFACT_MODEL_VERSION_CAP) ||
        !decode_token(artifact->interval, body + 96, ARTIFACT_INTERVAL_CAP))
        return ARTIFACT_FORMAT;

    for (size_t i = 0; i < ARTIFACT_FEATURE_COUNT; i++) {
        artifact->feature_mean[i] = decode_f32(body + 112 + i * sizeof(float));
        artifact->feature_scale[i] = decode_f32(body + 132 + i * sizeof(float));
        if (!isfinite(artifact->feature_mean[i]) ||
            !isfinite(artifact->feature_scale[i]) ||
            artifact->feature_scale[i] <= 0.0f)
            return ARTIFACT_FORMAT;
    }
    artifact->target_mean = decode_f32(body + 152);
    artifact->target_scale = decode_f32(body + 156);
    return isfinite(artifact->target_mean) && isfinite(artifact->target_scale) &&
           artifact->target_scale > 0.0f ? ARTIFACT_OK : ARTIFACT_FORMAT;
}

ArtifactStatus artifact_load(ModelArtifact* artifact, const char* path) {
    if (!artifact || !path) return ARTIFACT_ARGUMENT;
    *artifact = (ModelArtifact){0};

    FILE* file = fopen(path, "rb");
    if (!file) return ARTIFACT_IO;

    ModelArtifact loaded = {0};
    uint8_t header[HEADER_SIZE], body[BODY_PREFIX_SIZE];
    ArtifactStatus status = read_file(file, header, sizeof header);
    if (status != ARTIFACT_OK) goto done;
    if (memcmp(header, MAGIC, sizeof MAGIC) || decode_u32(header + 8) != 1 ||
        decode_u32(header + 12) != HEADER_SIZE) {
        status = ARTIFACT_UNSUPPORTED;
        goto done;
    }

    const uint64_t body_size = decode_u64(header + 16);
    const uint64_t expected_hash = decode_u64(header + 24);
    if (body_size < BODY_PREFIX_SIZE) {
        status = ARTIFACT_FORMAT;
        goto done;
    }
    status = validate_file_size(file, body_size);
    if (status != ARTIFACT_OK) goto done;
    BodyReader reader = {file, body_size, FNV_OFFSET};
    status = read_body(&reader, body, sizeof body);
    if (status != ARTIFACT_OK) goto done;

    size_t parameter_count;
    status = decode_config(body, &loaded.config, &parameter_count);
    if (status != ARTIFACT_OK) goto done;
    if (parameter_count > ARTIFACT_MAX_PARAMETERS ||
        parameter_count > (UINT64_MAX - BODY_PREFIX_SIZE) / sizeof(float)) {
        status = ARTIFACT_RANGE;
        goto done;
    }
    if (body_size != (uint64_t)BODY_PREFIX_SIZE +
                     (uint64_t)parameter_count * sizeof(float)) {
        status = ARTIFACT_FORMAT;
        goto done;
    }

    if (!transformer_init(&loaded.weights, loaded.config)) {
        status = ARTIFACT_NOMEM;
        goto done;
    }
    int weights_finite = 1;
    status = read_weights(&reader, loaded.weights.storage,
                          parameter_count, &weights_finite);
    if (status != ARTIFACT_OK) goto done;
    if (reader.remaining) {
        status = ARTIFACT_FORMAT;
        goto done;
    }
    const int extra = fgetc(file);
    if (extra != EOF || ferror(file)) {
        status = ferror(file) ? ARTIFACT_IO : ARTIFACT_FORMAT;
        goto done;
    }
    if (reader.hash != expected_hash) {
        status = ARTIFACT_INTEGRITY;
        goto done;
    }
    status = validate_metadata(&loaded, body, weights_finite);

done:
    if (fclose(file) && status == ARTIFACT_OK) status = ARTIFACT_IO;
    if (status == ARTIFACT_OK) *artifact = loaded;
    else transformer_free(&loaded.weights);
    return status;
}

void artifact_free(ModelArtifact* artifact) {
    if (!artifact) return;
    transformer_free(&artifact->weights);
    *artifact = (ModelArtifact){0};
}

const char* artifact_status_string(ArtifactStatus status) {
    static const char* const messages[] = {
        "ok", "invalid argument", "I/O error", "truncated artifact",
        "invalid artifact", "unsupported artifact", "checksum mismatch",
        "value out of range", "out of memory"
    };
    return (unsigned)status < sizeof messages / sizeof *messages ?
        messages[status] : "unknown artifact error";
}

void artifact_scale_features(float* values, size_t rows,
                             const ModelArtifact* artifact) {
    assert(values && artifact);
    for (size_t row = 0; row < rows; row++) {
        for (size_t feature = 0; feature < ARTIFACT_FEATURE_COUNT; feature++) {
            values[feature] =
                (values[feature] - artifact->feature_mean[feature]) /
                artifact->feature_scale[feature];
        }
        values += ARTIFACT_FEATURE_COUNT;
    }
}

float artifact_unscale_target(float value, const ModelArtifact* artifact) {
    assert(artifact);
    return value * artifact->target_scale + artifact->target_mean;
}
