#!/usr/bin/env python3
"""Verify deterministic, ordered, and atomic V1 artifact export."""

from dataclasses import replace
from pathlib import Path
import struct
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact_v1 import Artifact, Config, WEIGHT_FIELDS, write_artifact
from tools.float32 import f32


def make_artifact() -> Artifact:
    config = Config(model_dim=2, num_heads=1, ff_dim=3, num_layers=2, seq_len=2)
    counts = config.field_counts()
    weights = {
        field: [f32(100 * (field_index + 1) + index) for index in range(counts[field])]
        for field_index, field in enumerate(WEIGHT_FIELDS)
    }
    return Artifact(config, "artifact-test", "1h", (1.0,) * 5, (2.0,) * 5,
                    0.0, 1.0, weights)


def main() -> None:
    artifact = make_artifact()
    with tempfile.TemporaryDirectory(prefix="compose-mini-artifact-") as directory:
        first, second = Path(directory) / "first.bin", Path(directory) / "second.bin"
        write_artifact(first, artifact)
        write_artifact(second, artifact)
        encoded = first.read_bytes()
        assert encoded == second.read_bytes()

        magic, schema, header_size, body_size, checksum = struct.unpack("<8sIIQQ", encoded[:32])
        assert (magic, schema, header_size) == (b"CMPMINI\0", 1, 32)
        assert body_size == len(encoded) - header_size
        assert checksum == zlib.crc32(encoded[header_size:])
        payload = struct.unpack(f"<{artifact.config.parameter_count}f", encoded[192:])
        expected = tuple(value for field in WEIGHT_FIELDS for value in artifact.weights[field])
        assert payload == expected

        destination = Path(directory) / "preserve.bin"
        destination.write_bytes(b"existing artifact")
        for delta in (-1, 1):
            weights = {field: list(values) for field, values in artifact.weights.items()}
            weights["head_W"] = weights["head_W"][:1] if delta < 0 else \
                [*weights["head_W"], 3.0]
            try:
                write_artifact(destination, replace(artifact, weights=weights))
            except ValueError:
                pass
            else:
                raise AssertionError("invalid weight count was accepted")
            assert destination.read_bytes() == b"existing artifact"
    print("artifact exporter tests passed")


if __name__ == "__main__":
    main()
