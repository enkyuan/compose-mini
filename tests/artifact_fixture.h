#ifndef TEST_ARTIFACT_FIXTURE_H
#define TEST_ARTIFACT_FIXTURE_H

#include <stddef.h>
#include <stdint.h>

enum { ARTIFACT_FIXTURE_PARAMETER_COUNT = 95 };

typedef enum {
    ARTIFACT_FIXTURE_VALID,
    ARTIFACT_FIXTURE_BAD_CHECKSUM,
    ARTIFACT_FIXTURE_TRUNCATED,
    ARTIFACT_FIXTURE_TRAILING,
    ARTIFACT_FIXTURE_BAD_SCALE,
    ARTIFACT_FIXTURE_OUT_OF_RANGE,
    ARTIFACT_FIXTURE_WORKSPACE_RANGE,
    ARTIFACT_FIXTURE_HORIZON_TWO,
    ARTIFACT_FIXTURE_ZERO,
    ARTIFACT_FIXTURE_OVERFLOW,
    ARTIFACT_FIXTURE_LATE_OVERFLOW
} ArtifactFixtureKind;

uint64_t artifact_fixture_checksum(const uint8_t* bytes, size_t n);
void artifact_fixture_write(const char* path, ArtifactFixtureKind kind);

#endif /* TEST_ARTIFACT_FIXTURE_H */
