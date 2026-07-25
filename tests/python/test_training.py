#!/usr/bin/env python3
"""Smoke-test PyTorch training, weight export, and C-runtime compatibility."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("training tests require PyTorch") from error

from torch.utils.data import DataLoader, Dataset

from tools.artifact_v1 import Artifact, Config, write_artifact
from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.float32 import ulp_distance
from tools.reference import predict_windows
from tools.session_samples import SampleRows
from tools.train import (
    DataSplits, ForecastTransformer, TrainingData, data_loaders, export_weights,
    evaluate, feature_values, fit_epochs, fit_model, fit_training_updates,
    fit_updates, mean_loss, parse_args, prepare_data, prepare_rows,
    train as train_model, train_epoch,
)


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
    assert report["test"]["last_close_baseline_mae"] == \
        report["test"]["zero_return_baseline_mae"]
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
               "zero_return_baseline_mae": 0.0, "direction_accuracy": 0.0}
    with patch("tools.train.train_epoch", step), \
         patch("tools.train.mean_loss", validate), \
         patch("tools.train.evaluate", return_value=metrics):
        model, _, report = train_model(parse_args(training_options(
            csv, directory / "restoration.bin",
        )))
    assert report["best_validation_scaled_mse"] == 1.0
    assert all(torch.all(parameter == 1.0) for parameter in model.parameters())


def verify_fixed_epochs(csv: Path) -> None:
    config = Config(model_dim=4, num_heads=2, ff_dim=6, num_layers=1, seq_len=3)
    data = prepare_data(csv, config, 0.6, 0.2)
    splits = DataSplits(data.train, data.validation, data.test)

    class UnreadLoader:
        def __iter__(self) -> object:
            raise AssertionError("later split loader was read during fixed-epoch fit")

    train_loader = object()
    sentinel_loaders = (train_loader, UnreadLoader(), UnreadLoader())
    with patch("tools.train.data_loaders", return_value=sentinel_loaders), \
         patch("tools.train.train_epoch", return_value=0.0) as train_epoch:
        returned = fit_epochs(
            ForecastTransformer(config), splits, 8, 3, 3e-4, 1e-4, 7,
            torch.device("cpu"),
        )
    assert returned == sentinel_loaders
    assert train_epoch.call_count == 3
    assert all(call.args[1] is train_loader
               for call in train_epoch.call_args_list)

    model = ForecastTransformer(config)
    loaders = fit_epochs(
        model, splits, 8, 2, 3e-4, 1e-4, 7, torch.device("cpu"),
    )
    assert len(loaders) == 3
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def verify_fixed_updates() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    batches = DataLoader(
        tuple((torch.zeros(1), torch.zeros(()), 0.0, 0.0) for _ in range(6)),
        batch_size=1,
    )
    step = 0

    def train_batch(model, *_args) -> tuple[float, int]:
        nonlocal step
        step += 1
        with torch.no_grad():
            model.weight.fill_(step)
        return float(step), 1

    losses = iter((3.0, 1.0, 2.0))
    with patch("tools.train._train_batch", side_effect=train_batch):
        fit = fit_updates(
            model, batches, lambda: next(losses), 3, 2, 3e-4, 1e-4,
            torch.device("cpu"),
        )
    assert fit.best_checkpoint == 2
    assert fit.updates_trained == step == 6
    assert torch.equal(model.weight, torch.full_like(model.weight, 4.0))
    try:
        fit_updates(
            model, batches, lambda: 0.0, 2, 2, 3e-4, 1e-4,
            torch.device("cpu"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched fixed-update loader was accepted")
    with patch("tools.train._train_batch", return_value=(0.0, 1)):
        try:
            fit_updates(
                model, DataLoader(batches.dataset, batch_size=1),
                lambda: math.nan, 1, 6, 3e-4, 1e-4,
                torch.device("cpu"),
            )
        except FloatingPointError:
            pass
        else:
            raise AssertionError("non-finite fixed-update validation was accepted")


def verify_training_updates() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    samples = tuple(
        (torch.zeros(1), torch.tensor(float(index)), 0.0, 0.0)
        for index in range(3)
    )
    loader = DataLoader(samples, batch_size=1, sampler=(2, 0, 2, 1))
    observed = []

    def train_batch(model, batch, *_args) -> tuple[float, int]:
        observed.append(int(batch[1].item()))
        with torch.no_grad():
            model.weight.fill_(len(observed))
        return float(len(observed)), 1

    with patch("tools.train._train_batch", side_effect=train_batch):
        loss = fit_training_updates(
            model, loader, 4, 3e-4, 1e-4, torch.device("cpu"),
        )
    assert observed == [2, 0, 2, 1]
    assert loss == 2.5
    assert torch.equal(model.weight, torch.full_like(model.weight, 4.0))
    for updates in (True, 0, 3):
        try:
            fit_training_updates(
                model, loader, updates, 3e-4, 1e-4, torch.device("cpu"),
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid update budget passed: {updates}")

    class Schedule:
        def __init__(self, indices: tuple[int, ...]) -> None:
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self) -> int:
            return 4

    for indices in ((2, 0, 1), (2, 0, 1, 2, 0)):
        calls = 0

        def count_batch(*_args) -> tuple[float, int]:
            nonlocal calls
            calls += 1
            return 0.0, 1

        with patch("tools.train._train_batch", side_effect=count_batch):
            try:
                fit_training_updates(
                    model,
                    DataLoader(
                        samples, batch_size=1, sampler=Schedule(indices),
                    ),
                    4, 3e-4, 1e-4, torch.device("cpu"),
                )
            except ValueError:
                pass
            else:
                raise AssertionError("dishonest schedule was accepted")
        assert calls == min(len(indices), 4)
    with patch("tools.train._train_batch", return_value=(math.inf, 1)):
        try:
            fit_training_updates(
                model, loader, 4, 3e-4, 1e-4, torch.device("cpu"),
            )
        except FloatingPointError:
            pass
        else:
            raise AssertionError("non-finite training-only loss was accepted")


def verify_sampled_epoch_mean() -> None:
    class Samples(Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
            return (
                torch.ones(1), torch.tensor((1.0, 3.0)[index]),
                torch.ones(()), torch.ones(()),
            )

    class Scalar(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values.squeeze(-1) * self.weight

    model = Scalar()
    loader = DataLoader(Samples(), batch_size=2, sampler=(1, 1, 1, 1))
    assert mean_loss(model, loader, torch.device("cpu")) == 9.0
    loss = train_epoch(
        model, loader, torch.optim.SGD(model.parameters(), lr=0.0),
        torch.device("cpu"),
    )
    assert loss == 9.0


def verify_data_splits(csv: Path) -> None:
    config = Config(model_dim=4, num_heads=2, ff_dim=6, num_layers=1, seq_len=3)
    data = prepare_data(csv, config, 0.6, 0.2)
    splits = DataSplits(data.train, data.validation, data.test)
    try:
        splits.train = splits.test
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("DataSplits must be frozen")
    loaders = data_loaders(splits, 8, 7)
    assert tuple(loader.dataset for loader in loaders) == (
        splits.train, splits.validation, splits.test,
    )

    epoch = 0

    def step(model, *_args) -> float:
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(epoch)
        return float(epoch)

    model = ForecastTransformer(config)
    with patch("tools.train.train_epoch", step), \
         patch("tools.train.mean_loss", side_effect=(1.0, 2.0)):
        fit, _ = fit_model(
            model, splits, 8, 2, 1, 3e-4, 1e-4, 7, torch.device("cpu"),
        )
    assert fit.best_epoch == 1
    assert all(torch.all(parameter == 1.0) for parameter in model.parameters())

    positional = TrainingData(
        data.train, data.validation, data.test, data.feature_mean,
        data.feature_scale, data.target_mean, data.target_scale,
        data.feature_set, data.horizon_bars, data.target_kind,
    )
    changed = replace(positional, horizon_bars=2)
    assert changed.train is positional.train
    assert changed.horizon_bars == 2


def verify_indexed_windows(csv: Path) -> None:
    config = Config(model_dim=4, num_heads=2, ff_dim=6, num_layers=1, seq_len=3)
    rows = read_csv(csv)
    raw = torch.frombuffer(rows, dtype=torch.float32).view(-1, FEATURE_COUNT).clone()
    samples = (
        SampleRows(2, 3, 4, 2),
        SampleRows(6, 7, 8, 6),
        SampleRows(9, 10, 11, 9),
        SampleRows(12, 13, 14, 12),
    )
    data = prepare_rows(
        rows, config, 0.6, 0.2, (2, 1, 1),
        horizon_bars=3, target_kind="executable-return-v1",
        sample_rows=samples,
    )
    expected_rows = torch.cat((raw[:3], raw[4:7]))
    torch.testing.assert_close(data.feature_mean, expected_rows.mean(0))
    torch.testing.assert_close(
        data.feature_scale, expected_rows.std(0, unbiased=False),
    )
    expected_targets = torch.log(torch.stack((
        raw[4, 3] / raw[3, 0],
        raw[8, 3] / raw[7, 0],
    )))
    torch.testing.assert_close(data.target_mean, expected_targets.mean())
    torch.testing.assert_close(
        data.target_scale, expected_targets.std(unbiased=False),
    )
    assert data.train.indexed
    assert data.train.feature_starts == (0, 4, 7, 10)
    assert data.train.sample_rows == samples
    scaled = (raw - data.feature_mean) / data.feature_scale
    for index, start in enumerate((0, 4)):
        values, target, reference, outcome = data.train[index]
        torch.testing.assert_close(values, scaled[start:start + 3])
        torch.testing.assert_close(reference, raw[samples[index].entry, 0])
        torch.testing.assert_close(outcome, raw[samples[index].target, 3])
        torch.testing.assert_close(
            target * data.target_scale + data.target_mean,
            torch.log(outcome / reference),
        )

    stationary_samples = tuple(
        SampleRows(as_of, as_of + 1, as_of + 2, ordinal)
        for as_of, ordinal in ((3, 3), (7, 7), (10, 10), (13, 13))
    )
    stationary = prepare_rows(
        rows, config, 0.6, 0.2, (2, 1, 1),
        feature_set="stationary-v1", horizon_bars=3,
        sample_rows=stationary_samples,
    )
    assert stationary.train.feature_starts == (0, 4, 7, 10)
    transformed = feature_values(raw, "stationary-v1")
    expected = (
        transformed[:3] - stationary.feature_mean
    ) / stationary.feature_scale
    torch.testing.assert_close(stationary.train[0][0], expected)

    sparse = prepare_rows(
        rows[:13 * FEATURE_COUNT], config, 0.6, 0.2, (2, 1, 1),
        horizon_bars=20,
        sample_rows=(
            SampleRows(2, 3, 6, 2),
            SampleRows(5, 6, 7, 5),
            SampleRows(8, 9, 10, 25),
            SampleRows(11, 12, 12, 45),
        ),
    )
    assert tuple(map(len, (sparse.train, sparse.validation, sparse.test))) == \
        (2, 1, 1)
    incomplete = prepare_rows(
        rows, config, 0.6, 0.2, (2, 0, 2), horizon_bars=3,
        sample_rows=samples, allow_empty_later=True,
    )
    assert tuple(map(len, (
        incomplete.train, incomplete.validation, incomplete.test,
    ))) == (2, 0, 2)
    try:
        prepare_rows(
            rows, config, 0.6, 0.2, (2, 0, 2), horizon_bars=3,
            sample_rows=samples,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unbound empty validation split was accepted")

    invalid = list(samples)
    invalid[2] = SampleRows(9, 10, 11, 8)
    late_target = list(samples)
    late_target[1] = replace(late_target[1], target=10)
    for options in (
        {"sample_start": 1, "sample_rows": samples},
        {"sample_rows": invalid},
        {"sample_rows": late_target},
        {"sample_rows": (object(), *samples[1:])},
    ):
        try:
            prepare_rows(
                rows, config, 0.6, 0.2, (2, 1, 1),
                horizon_bars=3, **options,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid indexed sample contract was accepted")


def verify_conditioned_batches() -> None:
    torch.manual_seed(17)
    config = Config(model_dim=4, num_heads=2, ff_dim=6, num_layers=1, seq_len=3)
    model = ForecastTransformer(config)
    batch = 4
    values = torch.randn(batch, config.seq_len, config.in_dim)
    assert model(values).shape == (batch,)

    plain = model(values)
    conditioned = model(values, torch.zeros(batch, config.model_dim))
    assert torch.equal(plain, conditioned)

    context = torch.zeros(batch, config.model_dim)
    context[0, 0] = 1.0
    changed = model(values, context)
    assert not torch.equal(plain[:1], changed[:1])
    assert torch.equal(plain[1:], changed[1:])

    class ConditionedWindows(Dataset):
        def __len__(self) -> int:
            return batch

        def __getitem__(self, index: int) -> tuple[object, ...]:
            return ((values[index], context[index]), torch.zeros(()),
                    torch.ones(()), torch.ones(()))

    loader = DataLoader(ConditionedWindows(), batch_size=2)
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert math.isfinite(mean_loss(model, loader, device))
    assert math.isfinite(train_epoch(model, loader, optimizer, device))
    assert all(math.isfinite(value) for value in evaluate(
        model, loader, torch.zeros(()), torch.ones(()), device,
    ).values())


def main() -> None:
    binary = Path(sys.argv[1] if len(sys.argv) == 2 else ROOT / "bin/transformer").resolve()
    with tempfile.TemporaryDirectory(prefix="compose-mini-training-") as directory:
        distance = verify_export(binary, Path(directory))
        trained_distance = verify_training(binary, Path(directory))
        verify_restoration(Path(directory) / "training.csv", Path(directory))
        verify_data_splits(Path(directory) / "training.csv")
        verify_indexed_windows(Path(directory) / "training.csv")
        verify_fixed_epochs(Path(directory) / "training.csv")
        verify_fixed_updates()
        verify_training_updates()
        verify_sampled_epoch_mean()
        verify_conditioned_batches()
    print("training and export tests passed "
          f"(fixture {distance} ULP, trained maximum {trained_distance} ULP)")


if __name__ == "__main__":
    main()
