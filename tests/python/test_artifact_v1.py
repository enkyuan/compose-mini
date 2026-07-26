#!/usr/bin/env python3
"""Verify canonical float32 vectors and deterministic atomic artifact export."""

import base64
from collections.abc import Callable
from dataclasses import replace
import math
from pathlib import Path
import struct
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact_v1 import Artifact, Config, WEIGHT_FIELDS, write_artifact
from tools.float32 import decode_f32le_base64, encode_f32le_base64, f32


class TextSubclass(str):
    pass


def rejects(call: Callable[[], object]) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid float32 payload was accepted")


def verify_float32_payloads() -> None:
    payload = encode_f32le_base64((1.0, -2.5, -0.0))
    assert payload == {
        "encoding": "f32le-base64",
        "count": 3,
        "base64": "AACAPwAAIMAAAACA",
    }
    values = decode_f32le_base64(payload)
    assert decode_f32le_base64(payload, expected_count=3) == values
    assert values[:2] == (1.0, -2.5)
    assert values[2] == 0.0 and math.copysign(1.0, values[2]) == -1.0
    empty = {"encoding": "f32le-base64", "count": 0, "base64": ""}
    assert encode_f32le_base64(()) == empty
    assert decode_f32le_base64(empty) == ()
    assert encode_f32le_base64((1.00000001,)) == \
        encode_f32le_base64((1.0,))

    for value in (True, "1", float("nan"), float("inf"), -float("inf"), 1e40):
        rejects(lambda value=value: encode_f32le_base64((value,)))

    finite = {
        "encoding": "f32le-base64", "count": 1, "base64": "AACAPw==",
    }
    invalid = (
        [],
        {},
        {TextSubclass(key): value for key, value in finite.items()},
        {**finite, "extra": None},
        {**finite, "encoding": "base64"},
        {**finite, "count": True},
        {**finite, "count": -1},
        {**finite, "base64": b"AACAPw=="},
        {**finite, "base64": "AACAPw==="},
        {**finite, "count": 2},
        {
            **finite,
            "base64": base64.b64encode(
                struct.pack("<f", float("nan")),
            ).decode("ascii"),
        },
    )
    for value in invalid:
        rejects(lambda value=value: decode_f32le_base64(value))
    for expected in (2, True, -1):
        rejects(lambda expected=expected: decode_f32le_base64(
            finite, expected_count=expected,
        ))


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
    verify_float32_payloads()
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
