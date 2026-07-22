#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "artifact.h"

enum { HEADER_SIZE = 32, BODY_SIZE = 540, PARAMETER_COUNT = 95 };

typedef enum {
    FIXTURE_VALID,
    FIXTURE_BAD_CHECKSUM,
    FIXTURE_TRUNCATED,
    FIXTURE_TRAILING,
    FIXTURE_BAD_SCALE,
    FIXTURE_OUT_OF_RANGE,
    FIXTURE_WORKSPACE_RANGE
} FixtureKind;

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

static uint64_t checksum(const uint8_t* bytes, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++)
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    return hash;
}

static void build_body(uint8_t* body) {
    memset(body, 0, BODY_SIZE);
    const uint32_t config[] = {2, 1, 3, 2, 2, 5};
    for (size_t i = 0; i < 6; i++) put_u32(body + i * 4, config[i]);
    put_u32(body + 24, 1);
    memcpy(body + 32, "unit-v1", 7);
    memcpy(body + 96, "1h", 2);

    for (size_t i = 0; i < ARTIFACT_FEATURE_COUNT; i++) {
        put_f32(body + 112 + i * 4, (float)i + 1.0f);
        put_f32(body + 132 + i * 4, (float)i + 2.0f);
    }
    put_f32(body + 152, .5f);
    put_f32(body + 156, 2.0f);
    for (size_t i = 0; i < PARAMETER_COUNT; i++)
        put_f32(body + 160 + i * 4, (float)i + .25f);
}

static void write_fixture(const char* path, FixtureKind kind) {
    const uint8_t magic[] = {'C', 'M', 'P', 'M', 'I', 'N', 'I', 0};
    uint8_t header[HEADER_SIZE] = {0}, body[BODY_SIZE];
    build_body(body);
    if (kind == FIXTURE_BAD_SCALE) put_f32(body + 132, 0.0f);
    if (kind == FIXTURE_OUT_OF_RANGE) put_u32(body, UINT32_MAX);
    if (kind == FIXTURE_WORKSPACE_RANGE) put_u32(body + 16, INT32_MAX);

    memcpy(header, magic, sizeof magic);
    put_u32(header + 8, 1);
    put_u32(header + 12, HEADER_SIZE);
    put_u64(header + 16, BODY_SIZE);
    uint64_t hash = checksum(body, sizeof body);
    if (kind == FIXTURE_BAD_CHECKSUM) hash ^= 1;
    put_u64(header + 24, hash);

    FILE* file = fopen(path, "wb");
    assert(file);
    assert(fwrite(header, 1, sizeof header, file) == sizeof header);
    const size_t body_bytes = kind == FIXTURE_TRUNCATED ? BODY_SIZE - 1 : BODY_SIZE;
    assert(fwrite(body, 1, body_bytes, file) == body_bytes);
    if (kind == FIXTURE_TRAILING) assert(fputc(0, file) != EOF);
    assert(fclose(file) == 0);
}

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-6f);
}

static void test_valid(const char* path) {
    write_fixture(path, FIXTURE_VALID);
    ModelArtifact artifact;
    assert(artifact_load(&artifact, path) == ARTIFACT_OK);
    assert(artifact.config.model_dim == 2 && artifact.config.num_layers == 2);
    assert(artifact.config.ff_dim == 3 && artifact.config.in_dim == 5);
    assert(artifact.horizon_bars == 1);
    assert(!strcmp(artifact.model_version, "unit-v1"));
    assert(!strcmp(artifact.interval, "1h"));
    for (size_t i = 0; i < PARAMETER_COUNT; i++)
        assert_close(artifact.weights.storage[i], (float)i + .25f);
    assert(artifact.weights.embed_W == artifact.weights.storage);
    assert(artifact.weights.Wq == artifact.weights.storage + 10);
    assert(artifact.weights.Wk == artifact.weights.storage + 18);
    assert(artifact.weights.Wv == artifact.weights.storage + 26);
    assert(artifact.weights.Wo == artifact.weights.storage + 34);
    assert(artifact.weights.norm1_g == artifact.weights.storage + 42);
    assert(artifact.weights.norm1_b == artifact.weights.storage + 46);
    assert(artifact.weights.W1 == artifact.weights.storage + 50);
    assert(artifact.weights.b1 == artifact.weights.storage + 62);
    assert(artifact.weights.W2 == artifact.weights.storage + 68);
    assert(artifact.weights.b2 == artifact.weights.storage + 80);
    assert(artifact.weights.norm2_g == artifact.weights.storage + 84);
    assert(artifact.weights.norm2_b == artifact.weights.storage + 88);
    assert(artifact.weights.head_W == artifact.weights.storage + 92);
    assert(artifact.weights.head_b == artifact.weights.storage + 94);

    float values[] = {1, 2, 3, 4, 5, 3, 5, 7, 9, 11};
    assert(artifact_scale_features(values, 2, &artifact));
    for (size_t i = 0; i < ARTIFACT_FEATURE_COUNT; i++) {
        assert_close(values[i], 0.0f);
        assert_close(values[ARTIFACT_FEATURE_COUNT + i], 1.0f);
    }
    assert_close(artifact_unscale_target(1.5f, &artifact), 3.5f);
    artifact_free(&artifact);
    assert(!artifact.weights.storage && !artifact.model_version[0]);
}

static void test_invalid(const char* path) {
    const struct {
        FixtureKind kind;
        ArtifactStatus status;
    } cases[] = {
        {FIXTURE_BAD_CHECKSUM, ARTIFACT_INTEGRITY},
        {FIXTURE_TRUNCATED, ARTIFACT_TRUNCATED},
        {FIXTURE_TRAILING, ARTIFACT_FORMAT},
        {FIXTURE_BAD_SCALE, ARTIFACT_FORMAT},
        {FIXTURE_OUT_OF_RANGE, ARTIFACT_RANGE},
        {FIXTURE_WORKSPACE_RANGE, ARTIFACT_RANGE}
    };

    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        write_fixture(path, cases[i].kind);
        ModelArtifact artifact;
        assert(artifact_load(&artifact, path) == cases[i].status);
        assert(!artifact.weights.storage);
    }
}

int main(void) {
    const char* path = "bin/tests/model-v1.fixture";
    assert(checksum((const uint8_t*)"foobar", 6) ==
           UINT64_C(0x85944171f73967e8));
    test_valid(path);
    test_invalid(path);
    assert(remove(path) == 0);
    assert(!strcmp(artifact_status_string(ARTIFACT_INTEGRITY),
                   "checksum mismatch"));
    printf("artifact tests passed\n");
    return 0;
}
