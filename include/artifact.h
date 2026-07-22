#ifndef ARTIFACT_H
#define ARTIFACT_H

#include <stddef.h>
#include <stdint.h>
#include "transformer.h"

/* Versioned model metadata, training-fitted scalers, and Transformer weights. */

#define ARTIFACT_FEATURE_COUNT 5
#define ARTIFACT_MODEL_VERSION_CAP 64
#define ARTIFACT_INTERVAL_CAP 16
#define ARTIFACT_MAX_PARAMETERS ((size_t)67108864)
#define ARTIFACT_MAX_WORKSPACE_FLOATS ((size_t)67108864)

typedef enum {
    ARTIFACT_OK = 0,
    ARTIFACT_ARGUMENT,
    ARTIFACT_IO,
    ARTIFACT_TRUNCATED,
    ARTIFACT_FORMAT,
    ARTIFACT_UNSUPPORTED,
    ARTIFACT_INTEGRITY,
    ARTIFACT_RANGE,
    ARTIFACT_NOMEM
} ArtifactStatus;

typedef struct {
    TransformerConfig config;
    TransformerWeights weights;
    float feature_mean[ARTIFACT_FEATURE_COUNT];
    float feature_scale[ARTIFACT_FEATURE_COUNT];
    float target_mean;
    float target_scale;
    uint32_t horizon_bars;
    char model_version[ARTIFACT_MODEL_VERSION_CAP];
    char interval[ARTIFACT_INTERVAL_CAP];
} ModelArtifact;

/* Load a V1 little-endian artifact into a fresh object; clear it on failure. */
ArtifactStatus artifact_load(ModelArtifact* artifact, const char* path);

/* Release model storage and clear metadata; artifact may be NULL. */
void artifact_free(ModelArtifact* artifact);

/* Return stable text for logs and CLI errors. */
const char* artifact_status_string(ArtifactStatus status);

/* Apply training-fitted z-score scaling to row-major OHLCV values in place. */
void artifact_scale_features(float* values, size_t rows,
                             const ModelArtifact* artifact);

/* Map the scalar head output from model space back to log-return space. */
float artifact_unscale_target(float value, const ModelArtifact* artifact);

#endif /* ARTIFACT_H */
