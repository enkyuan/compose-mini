#!/usr/bin/env python3
"""Exercise the real C executable against an independent float32 reference."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact_v1 import Artifact, Config, WEIGHT_FIELDS, write_artifact
from tools.float32 import f32, ulp_distance
from tools.reference import predict_windows

TIMESTAMPS = tuple(f"2026-07-21T{hour:02d}:00:00Z" for hour in range(10, 15))
ROWS = (
    (100.0, 102.0, 99.0, 101.0, 1_000.0),
    (101.0, 104.0, 100.0, 103.0, 1_100.0),
    (103.0, 105.0, 101.0, 102.0, 900.0),
    (102.0, 108.0, 101.0, 107.0, 1_300.0),
    (107.0, 109.0, 105.0, 106.0, 1_200.0),
)


def make_weights(config: Config) -> dict[str, list[float]]:
    weights = {}
    for field_index, field in enumerate(WEIGHT_FIELDS):
        count = config.field_counts()[field]
        if field in {"norm1_g", "norm2_g"}:
            values = [1.0 + 0.01 * ((index + field_index) % 5 - 2)
                      for index in range(count)]
        else:
            values = [0.006 * ((index * 7 + field_index * 3) % 13 - 6)
                      for index in range(count)]
        weights[field] = [f32(value) for value in values]
    return weights


def make_artifact() -> Artifact:
    config = Config(model_dim=4, num_heads=2, ff_dim=6,
                    num_layers=2, seq_len=3)
    return Artifact(
        config=config,
        model_version='parity-"v1\\',
        interval="1h",
        feature_mean=(102.0, 105.0, 100.0, 103.0, 1_050.0),
        feature_scale=(3.0, 4.0, 2.0, 3.0, 150.0),
        target_mean=0.001,
        target_scale=0.02,
        weights=make_weights(config),
    )


def write_csv(path: Path) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    lines.extend(
        f"{timestamp}," + ",".join(format(value, "g") for value in row)
        for timestamp, row in zip(TIMESTAMPS, ROWS, strict=True)
    )
    path.write_text("\n".join(lines), encoding="ascii")


def run(binary: Path, model: Path, csv: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, model, csv, 'TEST-"PAIR\\', "1h", "2026-07-21T15:00:00Z"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )


def main() -> None:
    compile((ROOT / "tools/train.py").read_text(), "tools/train.py", "exec")
    binary = Path(sys.argv[1] if len(sys.argv) == 2 else ROOT / "bin/transformer").resolve()
    artifact = make_artifact()
    expected_values = predict_windows(ROWS, artifact)
    invalid_close = (*ROWS[:-1], (*ROWS[-1][:3], 0.0, ROWS[-1][4]))
    invalid_inputs = (
        (ROWS, replace(artifact, feature_scale=(1.0,))),
        (ROWS, replace(artifact, feature_scale=(3.0, 4.0, 0.0, 3.0, 150.0))),
        (ROWS, replace(artifact, feature_scale=(3.0, 4.0, 2.0 ** -149, 3.0, 150.0))),
        (invalid_close, artifact),
    )
    for rows, invalid_artifact in invalid_inputs:
        try:
            predict_windows(rows, invalid_artifact)
        except ValueError:
            continue
        raise AssertionError("invalid reference input was accepted")
    for bias in (-100_000.0, 100_000.0):
        weights = {field: list(values) for field, values in artifact.weights.items()}
        weights["head_b"] = [bias]
        try:
            predict_windows(ROWS, replace(artifact, weights=weights))
        except ValueError:
            continue
        raise AssertionError("invalid reference forecast was accepted")
    with tempfile.TemporaryDirectory(prefix="compose-mini-") as directory:
        model = Path(directory) / "model.bin"
        csv = Path(directory) / "bars.csv"
        write_artifact(model, artifact)
        write_csv(csv)
        first, second = run(binary, model, csv), run(binary, model, csv)

    assert first.returncode == 0 and not first.stderr, first.stderr
    assert second.returncode == 0 and first.stdout == second.stdout
    records = [json.loads(line) for line in first.stdout.splitlines()]
    assert len(records) == len(expected_values) == 3

    max_ulps = 0
    for index, (record, expected) in enumerate(zip(records, expected_values, strict=True)):
        assert record["instrument"] == 'TEST-"PAIR\\'
        assert record["interval"] == "1h"
        assert record["as_of"] == TIMESTAMPS[index + artifact.config.seq_len - 1]
        target = TIMESTAMPS[index + artifact.config.seq_len] if index < 2 else \
            "2026-07-21T15:00:00Z"
        assert record["target_time"] == target
        assert record["horizon_bars"] == 1
        assert record["model_version"] == artifact.model_version
        for key, value in zip(
            ("predicted_log_return", "predicted_close"), expected, strict=True,
        ):
            distance = ulp_distance(record[key], value)
            max_ulps = max(max_ulps, distance)
            assert distance <= 16, (key, record[key], value, distance)
    print(f"e2e parity passed (maximum binary32 distance {max_ulps} ULP)")


if __name__ == "__main__":
    main()
