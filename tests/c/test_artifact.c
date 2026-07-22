#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "artifact.h"
#include "artifact_fixture.h"

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-6f);
}

static void test_valid(const char* path) {
    artifact_fixture_write(path, ARTIFACT_FIXTURE_VALID);
    ModelArtifact artifact;
    assert(artifact_load(&artifact, path) == ARTIFACT_OK);
    assert(artifact.config.model_dim == 2 && artifact.config.num_layers == 2);
    assert(artifact.config.ff_dim == 3 && artifact.config.in_dim == 5);
    assert(artifact.horizon_bars == 1);
    assert(!strcmp(artifact.model_version, "unit-v1"));
    assert(!strcmp(artifact.interval, "1h"));
    for (size_t i = 0; i < ARTIFACT_FIXTURE_PARAMETER_COUNT; i++)
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
        ArtifactFixtureKind kind;
        ArtifactStatus status;
    } cases[] = {
        {ARTIFACT_FIXTURE_BAD_CHECKSUM, ARTIFACT_INTEGRITY},
        {ARTIFACT_FIXTURE_TRUNCATED, ARTIFACT_TRUNCATED},
        {ARTIFACT_FIXTURE_TRAILING, ARTIFACT_FORMAT},
        {ARTIFACT_FIXTURE_BAD_SCALE, ARTIFACT_FORMAT},
        {ARTIFACT_FIXTURE_OUT_OF_RANGE, ARTIFACT_RANGE},
        {ARTIFACT_FIXTURE_WORKSPACE_RANGE, ARTIFACT_RANGE},
        {ARTIFACT_FIXTURE_HORIZON_TWO, ARTIFACT_UNSUPPORTED}
    };

    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        artifact_fixture_write(path, cases[i].kind);
        ModelArtifact artifact;
        assert(artifact_load(&artifact, path) == cases[i].status);
        assert(!artifact.weights.storage);
    }
}

int main(void) {
    const char* path = "bin/tests/model-v1.fixture";
    assert(artifact_fixture_checksum((const uint8_t*)"foobar", 6) ==
           UINT64_C(0x9ef61f95));
    test_valid(path);
    test_invalid(path);
    assert(remove(path) == 0);
    assert(!strcmp(artifact_status_string(ARTIFACT_INTEGRITY),
                   "checksum mismatch"));
    printf("artifact tests passed\n");
    return 0;
}
