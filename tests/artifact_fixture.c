#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "artifact.h"
#include "artifact_fixture.h"

enum {
    HEADER_SIZE = 32,
    BODY_SIZE = 160 + 4 * ARTIFACT_FIXTURE_PARAMETER_COUNT
};

static void put_u32(uint8_t* out, uint32_t value) {
    for (int i = 0; i < 4; i++) out[i] = (uint8_t)(value >> (8 * i));
}

static void put_u64(uint8_t* out, uint64_t value) {
    for (int i = 0; i < 8; i++) out[i] = (uint8_t)(value >> (8 * i));
}

static void put_f32(uint8_t* out, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    put_u32(out, bits);
}

uint64_t artifact_fixture_checksum(const uint8_t* bytes, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++)
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    return hash;
}

static int zero_weights(ArtifactFixtureKind kind) {
    return kind == ARTIFACT_FIXTURE_ZERO ||
           kind == ARTIFACT_FIXTURE_OVERFLOW ||
           kind == ARTIFACT_FIXTURE_LATE_OVERFLOW;
}

static void build_body(uint8_t* body, ArtifactFixtureKind kind) {
    static const char ZERO_VERSION[] = "unit-\"v1\\";
    memset(body, 0, BODY_SIZE);
    const uint32_t config[] = {2, 1, 3, 2, 2, 5};
    for (size_t i = 0; i < 6; i++) put_u32(body + i * 4, config[i]);
    put_u32(body + 24, kind == ARTIFACT_FIXTURE_HORIZON_TWO ? 2 : 1);
    if (zero_weights(kind))
        memcpy(body + 32, ZERO_VERSION, sizeof ZERO_VERSION - 1);
    else
        memcpy(body + 32, "unit-v1", 7);
    memcpy(body + 96, "1h", 2);

    for (size_t i = 0; i < ARTIFACT_FEATURE_COUNT; i++) {
        put_f32(body + 112 + i * 4, (float)i + 1.0f);
        put_f32(body + 132 + i * 4, (float)i + 2.0f);
    }
    put_f32(body + 152, zero_weights(kind) ? 0.0f : .5f);
    put_f32(body + 156, zero_weights(kind) ? 1.0f : 2.0f);
    if (!zero_weights(kind))
        for (size_t i = 0; i < ARTIFACT_FIXTURE_PARAMETER_COUNT; i++)
            put_f32(body + 160 + i * 4, (float)i + .25f);
    if (kind == ARTIFACT_FIXTURE_OVERFLOW)
        put_f32(body + BODY_SIZE - sizeof(float), 100.0f);
    if (kind == ARTIFACT_FIXTURE_LATE_OVERFLOW) {
        put_f32(body + 160 + 6 * sizeof(float), 1.0f);
        put_f32(body + 160 + 92 * sizeof(float), 5.0f);
    }
    if (kind == ARTIFACT_FIXTURE_BAD_SCALE) put_f32(body + 132, 0.0f);
    if (kind == ARTIFACT_FIXTURE_OUT_OF_RANGE) put_u32(body, UINT32_MAX);
    if (kind == ARTIFACT_FIXTURE_WORKSPACE_RANGE)
        put_u32(body + 16, INT32_MAX);
}

void artifact_fixture_write(const char* path, ArtifactFixtureKind kind) {
    static const uint8_t MAGIC[] = {'C', 'M', 'P', 'M', 'I', 'N', 'I', 0};
    uint8_t header[HEADER_SIZE] = {0}, body[BODY_SIZE];
    build_body(body, kind);

    memcpy(header, MAGIC, sizeof MAGIC);
    put_u32(header + 8, 1);
    put_u32(header + 12, HEADER_SIZE);
    put_u64(header + 16, BODY_SIZE);
    uint64_t hash = artifact_fixture_checksum(body, sizeof body);
    if (kind == ARTIFACT_FIXTURE_BAD_CHECKSUM) hash ^= 1;
    put_u64(header + 24, hash);

    FILE* file = fopen(path, "wb");
    assert(file);
    assert(fwrite(header, 1, sizeof header, file) == sizeof header);
    const size_t body_bytes = kind == ARTIFACT_FIXTURE_TRUNCATED ?
        BODY_SIZE - 1 : BODY_SIZE;
    assert(fwrite(body, 1, body_bytes, file) == body_bytes);
    if (kind == ARTIFACT_FIXTURE_TRAILING) assert(fputc(0, file) != EOF);
    assert(fclose(file) == 0);
}
