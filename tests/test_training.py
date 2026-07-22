#!/usr/bin/env python3
"""Smoke-test PyTorch training, weight export, and C-runtime compatibility."""

from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("training tests require PyTorch") from error

from tools.artifact_v1 import Artifact, Config, write_artifact
from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.float32 import ulp_distance
from tools.reference import predict_windows
from tools.train import ForecastTransformer, export_weights, parse_args, train as train_model


def write_csv(path: Path, rows: list[list[float]]) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    lines.extend(
        f"2026-07-21T10:{index:02d}:00Z," + ",".join(f"{value:.9g}" for value in row)
        for index, row in enumerate(rows)
    )
    path.write_text("\n".join(lines), encoding="ascii")


def run(binary: Path, artifact: Path, csv: Path, interval: str,
        target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, artifact, csv, "TRAINING-TEST", interval, target],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )


def training_options(csv: Path, artifact: Path) -> list[str]:
    return [
        str(csv), str(artifact), "--interval", "1m", "--model-version", "training-smoke",
        "--seq-len", "3", "--model-dim", "4", "--heads", "2",
        "--ff-dim", "6", "--layers", "1", "--epochs", "2",
        "--patience", "2", "--batch-size", "4", "--train-fraction", "0.6",
        "--validation-fraction", "0.2", "--seed", "3",
    ]


def verify_export(binary: Path, directory: Path) -> int:
    torch.manual_seed(11)
    config = Config(model_dim=4, num_heads=2, ff_dim=6, num_layers=2, seq_len=3)
    model = ForecastTransformer(config).eval()
    weights = {field: tuple(values) for field, values in export_weights(model).items()}
    artifact = Artifact(config, "torch-export", "1m", (0.0,) * 5, (1.0,) * 5,
                        0.0, 1.0, weights)
    rows = [[1.0, 2.0, 0.5, 1.5, 10.0], [1.5, 2.5, 1.0, 2.0, 11.0],
            [2.0, 3.0, 1.5, 2.5, 12.0]]
    expected = predict_windows(rows, artifact)[0][0]
    with torch.no_grad():
        actual = float(model(torch.tensor(rows, dtype=torch.float32).unsqueeze(0)).item())
    distance = ulp_distance(actual, expected)
    assert distance <= 256, (actual, expected, distance)

    model_path, csv_path = directory / "export.bin", directory / "export.csv"
    write_artifact(model_path, artifact)
    write_csv(csv_path, rows)
    completed = run(binary, model_path, csv_path, "1m", "2026-07-21T10:03:00Z")
    assert completed.returncode == 0, completed.stderr
    forecast = json.loads(completed.stdout)
    assert ulp_distance(forecast["predicted_log_return"], expected) <= 16
    return distance


def verify_training(binary: Path, directory: Path) -> int:
    rows = []
    for index in range(24):
        close = 100.0 + 0.2 * index + 0.03 * (index % 5)
        rows.append([close - 0.1 * (index % 3), close + 0.8 + 0.02 * (index % 2),
                     close - 0.7, close, 1_000.0 + 7.0 * index + index % 4])
    csv_path, model_path = directory / "training.csv", directory / "trained.bin"
    write_csv(csv_path, rows)
    command = [sys.executable, ROOT / "tools/train.py", *training_options(csv_path, model_path)]
    completed = subprocess.run(command, cwd=ROOT, check=False,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    metrics = (report["best_validation_scaled_mse"], *report["test"].values())
    assert all(math.isfinite(value) for value in metrics)

    repeated_path = directory / "trained-again.bin"
    restored, artifact, repeated_report = train_model(
        parse_args(training_options(csv_path, repeated_path)),
    )
    write_artifact(repeated_path, artifact)
    assert model_path.read_bytes() == repeated_path.read_bytes()
    assert report["best_validation_scaled_mse"] == \
        repeated_report["best_validation_scaled_mse"]
    assert report["test"] == repeated_report["test"]

    forecast = run(binary, model_path, csv_path, "1m", "2026-07-21T10:24:00Z")
    assert forecast.returncode == 0, forecast.stderr
    assert len(forecast.stdout.splitlines()) == len(rows) - 3 + 1
    records = [json.loads(line) for line in forecast.stdout.splitlines()]
    values = read_csv(csv_path)
    raw = torch.frombuffer(values, dtype=torch.float32).view(-1, FEATURE_COUNT).clone()
    mean, scale = torch.tensor(artifact.feature_mean), torch.tensor(artifact.feature_scale)
    scaled = (raw - mean) / scale
    windows = torch.stack([scaled[index:index + 3] for index in range(len(records))])
    with torch.no_grad():
        predicted_return = restored(windows) * artifact.target_scale + artifact.target_mean
        predicted_close = raw[2:, 3] * predicted_return.exp()
    maximum = 0
    for record, expected_return, expected_close in zip(
        records, predicted_return, predicted_close, strict=True,
    ):
        distances = (
            ulp_distance(record["predicted_log_return"], float(expected_return)),
            ulp_distance(record["predicted_close"], float(expected_close)),
        )
        maximum = max(maximum, *distances)
        assert max(distances) <= 256
    return maximum


def verify_restoration(csv: Path, directory: Path) -> None:
    epoch = 0

    def step(model, *_args) -> float:
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(epoch)
        return float(epoch)

    def validate(*_args) -> float:
        return float(epoch)

    metrics = {"return_mse": 0.0, "return_mae": 0.0, "close_mae": 0.0,
               "last_close_baseline_mae": 0.0}
    with patch("tools.train.train_epoch", step), \
         patch("tools.train.mean_loss", validate), \
         patch("tools.train.evaluate", return_value=metrics):
        model, _, report = train_model(parse_args(training_options(
            csv, directory / "restoration.bin",
        )))
    assert report["best_validation_scaled_mse"] == 1.0
    assert all(torch.all(parameter == 1.0) for parameter in model.parameters())


def main() -> None:
    binary = Path(sys.argv[1] if len(sys.argv) == 2 else ROOT / "bin/transformer").resolve()
    with tempfile.TemporaryDirectory(prefix="compose-mini-training-") as directory:
        distance = verify_export(binary, Path(directory))
        trained_distance = verify_training(binary, Path(directory))
        verify_restoration(Path(directory) / "training.csv", Path(directory))
    print("training and export tests passed "
          f"(fixture {distance} ULP, trained maximum {trained_distance} ULP)")


if __name__ == "__main__":
    main()
