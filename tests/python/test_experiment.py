#!/usr/bin/env python3
"""Verify walk-forward selection, aligned targets, baselines, and JSON reports."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("experiment tests require PyTorch") from error

from tools.experiment import (
    Candidate, ConstantReturn, RollingMean, Sweep, expected_runs, holdout_split,
    linear_model, run_experiment, select_candidates, walk_forward_splits,
    write_report,
)
from tools.train import (
    TrainingData, Windows, data_loaders, evaluate, feature_lookback,
    feature_values, prepare_data,
)


def write_csv(path: Path, count: int = 56) -> None:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    lines = ["timestamp,open,high,low,close,volume"]
    for index in range(count):
        close = 100.0 + 0.08 * index + 0.05 * (index % 7) - 0.03 * (index % 3)
        values = (close - 0.1 - 0.01 * (index % 2), close + 0.4,
                  close - 0.5, close, 1_000.0 + 11.0 * index + index % 5)
        timestamp = (start + timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(timestamp + "," + ",".join(f"{value:.9g}" for value in values))
    path.write_text("\n".join(lines), encoding="ascii")


def candidate(name: str, seq_len: int,
              feature_set: str = "ohlcv") -> Candidate:
    return Candidate(name, seq_len, 4, 2, 6, 1, 1e-3, 1e-4,
                     6, min(2, seq_len - 1), 1e-3, feature_set)


def outcomes(dataset: Windows) -> torch.Tensor:
    return torch.stack([torch.stack(dataset[index][1:])
                        for index in range(len(dataset))])


def verify_stationary_features() -> None:
    rows = torch.tensor([
        [100.0, 105.0, 95.0, 102.0, 1_000.0],
        [103.0, 108.0, 101.0, 106.0, 1_300.0],
        [104.0, 107.0, 99.0, 100.0, 900.0],
    ])
    features = feature_values(rows, "stationary-v1")
    scaled = rows.clone()
    scaled[:, :4] *= 7.0
    torch.testing.assert_close(feature_values(scaled, "stationary-v1"), features)
    torch.testing.assert_close(
        features[:, 0] + features[:, 1],
        torch.log(rows[1:, 3] / rows[:-1, 3]),
    )
    torch.testing.assert_close(
        features[:, 2] + features[:, 1].abs() + features[:, 3],
        torch.log(rows[1:, 1] / rows[1:, 2]),
    )
    future = rows.clone()
    future[-1, :4] *= 2.0
    assert torch.equal(feature_values(future, "stationary-v1")[0], features[0])
    assert torch.equal(feature_values(torch.ones(2, 5), "stationary-v1"),
                       torch.zeros(1, 5))
    assert feature_lookback("ohlcv") == 0
    assert feature_lookback("stationary-v1") == 1
    invalid = []
    for column, value in ((0, 0.0), (2, 104.0), (4, -1.0)):
        bad = rows.clone()
        bad[1, column] = value
        invalid.append(bad)
    for bad in invalid:
        try:
            feature_values(bad, "stationary-v1")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid stationary source bar was accepted")


def verify_selection_is_validation_only() -> None:
    candidates = (candidate("first", 3), candidate("second", 5))
    records = [
        {"model": "transformer", "candidate": "first",
         "validation_scaled_mse": 2.0, "test": 0.0},
        {"model": "transformer", "candidate": "second",
         "validation_scaled_mse": 1.0, "test": 100.0},
    ]
    selected = select_candidates(records, ("transformer",), candidates)
    assert selected["transformer"]["candidate"] == "second"
    records[0]["test"], records[1]["test"] = 1_000.0, -1_000.0
    assert select_candidates(records, ("transformer",), candidates) == selected


def verify_ridge() -> None:
    torch.manual_seed(13)
    features, closes = torch.randn(40, 5), torch.ones(40)
    windows = features.unfold(0, 2, 1).transpose(1, 2).reshape(-1, 10)
    weight = torch.linspace(-0.5, 0.5, 10)
    targets = windows @ weight + 0.25
    dataset = Windows(features, targets, closes, 2, 0, 30)
    data = TrainingData(dataset, dataset, dataset, torch.zeros(5),
                        torch.ones(5), torch.tensor(0.0), torch.tensor(1.0))
    torch.testing.assert_close(linear_model(data, 1e-8)(windows[:30]), targets[:30],
                               rtol=1e-5, atol=1e-5)


def verify_caps(directory: Path) -> None:
    path = directory / "oversized.json"
    base = {"name": "oversized", "seq_len": 410, "model_dim": 4,
            "heads": 2, "ff_dim": 6, "layers": 1}
    for model, extra, message in (
        ("linear", {}, "linear feature cap"),
        ("mlp", {"seq_len": 32, "mlp_dim": 100_000}, "MLP parameter cap"),
    ):
        path.write_text(json.dumps({"models": [model], "candidates": [base | extra]}),
                        encoding="utf-8")
        try:
            Sweep.read(path)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{model} size cap was not enforced")


def main() -> None:
    assert walk_forward_splits(20, 2, 0.2) == ((8, 4), (12, 4))
    assert holdout_split(20, 0.2) == (12, 4, 4)
    verify_selection_is_validation_only()
    verify_ridge()
    verify_stationary_features()
    sweep = Sweep(
        (candidate("short", 3), candidate("stationary", 5, "stationary-v1")),
        ("transformer", "linear", "mlp", "rolling_mean", "last_close"),
        (3,), 2, 0.2, 2, 2, 8,
    )
    assert expected_runs(sweep, 1) == 25
    with tempfile.TemporaryDirectory(prefix="compose-mini-experiment-") as directory:
        verify_caps(Path(directory))
        csv = Path(directory) / "series.csv"
        changed = Path(directory) / "changed.csv"
        output = Path(directory) / "report.json"
        repeated_output = Path(directory) / "report-again.json"
        write_csv(csv)
        lines = csv.read_text(encoding="ascii").splitlines()
        for index in range(27, len(lines)):
            fields = lines[index].split(",")
            fields[1:] = [str(float(value) * 1.5) for value in fields[1:]]
            lines[index] = ",".join(fields)
        changed.write_text("\n".join(lines), encoding="ascii")
        base = prepare_data(csv, candidate("leak", 5).config(), 0.7, 0.15,
                            (20, 10, 10))
        perturbed = prepare_data(changed, candidate("leak", 5).config(), 0.7, 0.15,
                                 (20, 10, 10))
        for left, right in zip(
            (base.feature_mean, base.feature_scale, base.target_mean, base.target_scale),
            (perturbed.feature_mean, perturbed.feature_scale,
             perturbed.target_mean, perturbed.target_scale), strict=True,
        ):
            assert torch.equal(left, right)
        assert torch.equal(linear_model(base, 1e-3).weight,
                           linear_model(perturbed, 1e-3).weight)
        stationary = prepare_data(
            csv, candidate("stationary-leak", 5).config(), 0.7, 0.15,
            (20, 10, 10), feature_set="stationary-v1",
        )
        stationary_perturbed = prepare_data(
            changed, candidate("stationary-leak", 5).config(), 0.7, 0.15,
            (20, 10, 10), feature_set="stationary-v1",
        )
        for left, right in zip(
            (stationary.feature_mean, stationary.feature_scale,
             stationary.target_mean, stationary.target_scale),
            (stationary_perturbed.feature_mean, stationary_perturbed.feature_scale,
             stationary_perturbed.target_mean, stationary_perturbed.target_scale),
            strict=True,
        ):
            assert torch.equal(left, right)
        short = prepare_data(csv, candidate("short-aligned", 3).config(),
                             0.7, 0.15, (20, 10, 10), 2)
        for left, right in zip(
            (short.train, short.validation, short.test),
            (base.train, base.validation, base.test), strict=True,
        ):
            torch.testing.assert_close(outcomes(left), outcomes(right),
                                       rtol=0.0, atol=0.0)
        raw_history = prepare_data(csv, candidate("raw-history", 6).config(),
                                   0.7, 0.15, (20, 10, 10))
        for raw_split, stationary_split in zip(
            (raw_history.train, raw_history.validation, raw_history.test),
            (stationary.train, stationary.validation, stationary.test), strict=True,
        ):
            torch.testing.assert_close(outcomes(raw_split), outcomes(stationary_split),
                                       rtol=0.0, atol=0.0)
        loader = data_loaders(base, 8, 0)[1]
        constant = ConstantReturn(base)
        constant_metrics = evaluate(constant, loader, base.target_mean,
                                    base.target_scale, torch.device("cpu"))
        assert constant_metrics["direction_accuracy"] == 0.0
        features = next(iter(loader))[0]
        rolling = RollingMean(base, 2)(features) * base.target_scale + base.target_mean
        closes = features[:, :, 3] * base.feature_scale[3] + base.feature_mean[3]
        expected = torch.log(closes[:, 1:] / closes[:, :-1])[:, -2:].mean(1)
        torch.testing.assert_close(rolling, expected)
        stationary_loader = data_loaders(stationary, 8, 0)[1]
        stationary_features = next(iter(stationary_loader))[0]
        stationary_rolling = RollingMean(stationary, 2)(stationary_features)
        stationary_raw = stationary_features * stationary.feature_scale + \
            stationary.feature_mean
        stationary_expected = stationary_raw[:, -2:, :2].sum(2).mean(1)
        torch.testing.assert_close(
            stationary_rolling * stationary.target_scale + stationary.target_mean,
            stationary_expected,
        )
        try:
            run_experiment(sweep, (("SYNTH", csv),), torch.device("cpu"), 24)
        except ValueError as error:
            assert "requires 25 runs" in str(error)
        else:
            raise AssertionError("run limit was not enforced")
        report = run_experiment(sweep, (("SYNTH", csv),), torch.device("cpu"), 25)
        repeated = run_experiment(sweep, (("SYNTH", csv),), torch.device("cpu"), 25)
        assert repeated == report
        assert report["schema"] == 2
        assert report["protocol"]["aligned_history_bars"] == 6
        assert set(report["protocol"]["feature_sets"]) == {"ohlcv", "stationary-v1"}
        paired = report["validation_summary"]["transformer"]["paired_deltas"]
        assert paired["stationary-minus-short"]["return_mae"]["count"] == 2
        assert set(report["selection"]) == set(sweep.models)
        for record in report["test"]:
            assert record["candidate"] == \
                report["selection"][record["model"]]["candidate"]
            assert all(math.isfinite(value) for value in record["metrics"].values())
            assert 0.0 <= record["metrics"]["direction_accuracy"] <= 1.0
        for fold in range(sweep.folds):
            targets = {
                json.dumps(record["targets"], sort_keys=True)
                for record in report["validation"] if record["fold"] == fold
            }
            assert len(targets) == 1
        validation_end = max(
            record["targets"]["validation"][1] for record in report["validation"]
        )
        test_start = min(record["targets"]["test"][0] for record in report["test"])
        assert validation_end < test_start
        write_report(output, report)
        write_report(repeated_output, repeated)
        assert output.read_bytes() == repeated_output.read_bytes()
        assert json.loads(output.read_text(encoding="utf-8")) == report
        assert "NaN" not in output.read_text(encoding="utf-8")
        integrity_sweep = Sweep(
            (candidate("integrity", 3),), ("last_close",), (3,), 1, 0.2, 1, 1, 8,
        )
        with patch("tools.experiment._sha256", side_effect=("before", "after")):
            try:
                run_experiment(integrity_sweep, (("SYNTH", csv),),
                               torch.device("cpu"), 2)
            except ValueError as error:
                assert "CSV changed" in str(error)
            else:
                raise AssertionError("mid-run CSV replacement was accepted")
    print("experiment tests passed")


if __name__ == "__main__":
    main()
