#!/usr/bin/env python3
"""Select forecasting configurations by walk-forward validation, then test once."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
import argparse
import hashlib
import json
import math
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    raise SystemExit("experiments require PyTorch: python -m pip install torch") from error

from tools.artifact_v1 import Config
from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.train import (
    Fit, ForecastTransformer, TrainingData, Windows, data_loaders, evaluate,
    fit_model, mean_loss, prepare_data,
)

MODELS = ("transformer", "linear", "mlp", "rolling_mean", "last_close")
NEURAL = frozenset(("transformer", "mlp"))
RETURN_METRICS = ("return_mse", "return_mae", "direction_accuracy")
SERIES_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
MAX_FLAT_FEATURES = 2_048
MAX_MLP_PARAMETERS = 8_388_608


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float,
            strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or strict and result == minimum:
        operator = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {operator} {minimum}")
    return result


@dataclass(frozen=True)
class Candidate:
    name: str
    seq_len: int
    model_dim: int
    heads: int
    ff_dim: int
    layers: int
    learning_rate: float
    weight_decay: float
    mlp_dim: int
    rolling_window: int
    ridge: float

    @classmethod
    def parse(cls, value: object) -> Candidate:
        if not isinstance(value, dict):
            raise ValueError("each candidate must be an object")
        allowed = set(cls.__annotations__)
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown candidate field: {sorted(extra)[0]}")
        name = value.get("name")
        if not isinstance(name, str) or not SERIES_NAME.fullmatch(name):
            raise ValueError("candidate name is invalid")
        seq_len = _integer(value.get("seq_len"), f"{name}.seq_len", 2)
        candidate = cls(
            name,
            seq_len,
            _integer(value.get("model_dim"), f"{name}.model_dim"),
            _integer(value.get("heads"), f"{name}.heads"),
            _integer(value.get("ff_dim"), f"{name}.ff_dim"),
            _integer(value.get("layers"), f"{name}.layers"),
            _number(value.get("learning_rate", 3e-4),
                    f"{name}.learning_rate", 0.0, True),
            _number(value.get("weight_decay", 1e-4),
                    f"{name}.weight_decay", 0.0),
            _integer(value.get("mlp_dim", 32), f"{name}.mlp_dim"),
            _integer(value.get("rolling_window", min(8, seq_len - 1)),
                     f"{name}.rolling_window"),
            _number(value.get("ridge", 1e-3), f"{name}.ridge", 0.0, True),
        )
        candidate.config().validate()
        if candidate.rolling_window >= candidate.seq_len:
            raise ValueError(f"{name}.rolling_window must be less than seq_len")
        return candidate

    def config(self) -> Config:
        return Config(self.model_dim, self.heads, self.ff_dim,
                      self.layers, self.seq_len)


@dataclass(frozen=True)
class Sweep:
    candidates: tuple[Candidate, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: int
    fold_fraction: float
    epochs: int
    patience: int
    batch_size: int

    @classmethod
    def read(cls, path: Path) -> Sweep:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read sweep: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("sweep must be an object")
        allowed = {"candidates", "models", "seeds", "folds", "fold_fraction",
                   "epochs", "patience", "batch_size"}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown sweep field: {sorted(extra)[0]}")
        candidates_value = value.get("candidates")
        if not isinstance(candidates_value, list) or not candidates_value:
            raise ValueError("sweep candidates must be a nonempty array")
        candidates = tuple(Candidate.parse(item) for item in candidates_value)
        if len({item.name for item in candidates}) != len(candidates):
            raise ValueError("candidate names must be unique")
        models_value = value.get("models", list(MODELS))
        if not isinstance(models_value, list) or not models_value or \
           any(model not in MODELS for model in models_value) or \
           len(set(models_value)) != len(models_value):
            raise ValueError("models must be unique supported names")
        for candidate in candidates:
            flat = candidate.seq_len * FEATURE_COUNT
            if "linear" in models_value and flat > MAX_FLAT_FEATURES:
                raise ValueError(f"{candidate.name} exceeds the linear feature cap")
            mlp_parameters = flat * candidate.mlp_dim + candidate.mlp_dim + 1
            if "mlp" in models_value and mlp_parameters > MAX_MLP_PARAMETERS:
                raise ValueError(f"{candidate.name} exceeds the MLP parameter cap")
        seeds_value = value.get("seeds", [7])
        if not isinstance(seeds_value, list) or not seeds_value:
            raise ValueError("seeds must be a nonempty array")
        seeds = tuple(_integer(seed, "seed", 0) for seed in seeds_value)
        if len(set(seeds)) != len(seeds) or max(seeds) > 0x7FFF_FFFF_FFFF_FFFF:
            raise ValueError("seeds must be unique signed 64-bit integers")
        folds = _integer(value.get("folds", 3), "folds")
        fraction = _number(value.get("fold_fraction", 0.1),
                           "fold_fraction", 0.0, True)
        if fraction * (folds + 1) >= 1.0:
            raise ValueError("fold_fraction leaves no initial training segment")
        return cls(
            candidates, tuple(models_value), seeds, folds, fraction,
            _integer(value.get("epochs", 100), "epochs"),
            _integer(value.get("patience", 10), "patience"),
            _integer(value.get("batch_size", 64), "batch_size"),
        )


class FlatMLP(nn.Module):
    def __init__(self, seq_len: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * FEATURE_COUNT, hidden),
            nn.GELU(approximate="none"), nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class Affine(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.flatten(1) @ self.weight + self.bias


class ConstantReturn(nn.Module):
    def __init__(self, data: TrainingData) -> None:
        super().__init__()
        self.register_buffer("value", -data.target_mean / data.target_scale)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.value.expand(len(values))


class RollingMean(nn.Module):
    def __init__(self, data: TrainingData, window: int) -> None:
        super().__init__()
        self.window = window
        for name, value in (
            ("close_mean", data.feature_mean[3]),
            ("close_scale", data.feature_scale[3]),
            ("target_mean", data.target_mean),
            ("target_scale", data.target_scale),
        ):
            self.register_buffer(name, value)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        closes = values[:, :, 3] * self.close_scale + self.close_mean
        returns = torch.log(closes[:, 1:] / closes[:, :-1])
        prediction = returns[:, -self.window:].mean(1)
        return (prediction - self.target_mean) / self.target_scale


def walk_forward_splits(samples: int, folds: int,
                        fraction: float) -> tuple[tuple[int, int], ...]:
    block = int(samples * fraction)
    initial = samples - (folds + 1) * block
    if min(block, initial) <= 0:
        raise ValueError("series is too short for the requested walk-forward folds")
    return tuple((initial + fold * block, block) for fold in range(folds))


def holdout_split(samples: int, fraction: float) -> tuple[int, int, int]:
    block = int(samples * fraction)
    if block <= 0 or samples - 2 * block <= 0:
        raise ValueError("series is too short for validation and test holdouts")
    return samples - 2 * block, block, block


def _matrix(dataset: Windows) -> tuple[torch.Tensor, torch.Tensor]:
    windows = dataset.features.unfold(0, dataset.seq_len, 1).transpose(1, 2)
    start, end = dataset.start, dataset.start + dataset.count
    return (windows[start:end].reshape(dataset.count, -1).double(),
            dataset.targets[start:end].double())


def linear_model(data: TrainingData, ridge: float) -> Affine:
    values, targets = _matrix(data.train)
    value_mean, target_mean = values.mean(0), targets.mean()
    centered = values - value_mean
    gram = centered.T @ centered
    gram.diagonal().add_(ridge)
    weight = torch.linalg.solve(gram, centered.T @ (targets - target_mean))
    bias = target_mean - value_mean @ weight
    return Affine(weight.float(), bias.float())


def _timestamps(path: Path, expected: int) -> list[str]:
    with path.open("r", encoding="ascii") as file:
        next(file, None)
        values = [line.partition(",")[0] for line in file]
    if len(values) != expected:
        raise ValueError("CSV changed while the experiment was reading it")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary(timestamps: list[str], max_seq_len: int,
              split: tuple[int, ...]) -> dict[str, list[str]]:
    starts, offset = [], max_seq_len
    for count in split:
        starts.append(offset)
        offset += count
    return {
        name: [timestamps[start], timestamps[start + count - 1]]
        for name, start, count in zip(
            ("train", "validation", "test"), starts, split, strict=False,
        )
    }


def _fit_neural(model_name: str, candidate: Candidate, data: TrainingData,
                sweep: Sweep, seed: int,
                device: torch.device) -> tuple[nn.Module, Fit, tuple[object, ...]]:
    torch.manual_seed(seed)
    model = (ForecastTransformer(candidate.config()) if model_name == "transformer"
             else FlatMLP(candidate.seq_len, candidate.mlp_dim)).to(device)
    fit, loaders = fit_model(
        model, data, sweep.batch_size, sweep.epochs, sweep.patience,
        candidate.learning_rate, candidate.weight_decay, seed, device,
    )
    return model, fit, loaders


def _deterministic(model_name: str, candidate: Candidate,
                   data: TrainingData) -> nn.Module:
    if model_name == "linear":
        return linear_model(data, candidate.ridge)
    if model_name == "rolling_mean":
        return RollingMean(data, candidate.rolling_window)
    return ConstantReturn(data)


def _validation_record(model_name: str, candidate: Candidate, series: str,
                       fold: int, boundary: dict[str, list[str]], seed: int | None,
                       samples: int, loss: float,
                       fit: Fit | None) -> dict[str, object]:
    record: dict[str, object] = {
        "model": model_name, "candidate": candidate.name, "series": series,
        "fold": fold, "seed": seed, "targets": boundary,
        "samples": samples, "validation_scaled_mse": loss,
    }
    if fit:
        record.update(asdict(fit))
    return record


def select_candidates(records: Sequence[Mapping[str, object]],
                      models: Sequence[str], candidates: Sequence[Candidate]
                      ) -> dict[str, dict[str, object]]:
    order = {candidate.name: index for index, candidate in enumerate(candidates)}
    selected: dict[str, dict[str, object]] = {}
    for model in models:
        scores = {
            candidate.name: fmean(
                float(record["validation_scaled_mse"])
                for record in records
                if record["model"] == model and record["candidate"] == candidate.name
            )
            for candidate in candidates
        }
        name, score = min(scores.items(), key=lambda item: (item[1], order[item[0]]))
        selected[model] = {"candidate": name,
                           "mean_validation_scaled_mse": score}
    return selected


def _means(records: Sequence[Mapping[str, object]],
           metrics: Sequence[str]) -> dict[str, float]:
    return {metric: fmean(float(record["metrics"][metric]) for record in records)
            for metric in metrics}


def _statistics(values: Sequence[float | None]) -> dict[str, float | None]:
    finite = [value for value in values if value is not None]
    return ({"mean": fmean(finite), "stddev": pstdev(finite)} if finite else
            {"mean": None, "stddev": None})


def summarize(records: Sequence[Mapping[str, object]],
              models: Sequence[str]) -> dict[str, object]:
    metrics = ("return_mse", "return_mae", "close_mae",
               "last_close_baseline_mae", "direction_accuracy")
    summary: dict[str, object] = {}
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        seeds = sorted({record["seed"] for record in model_records},
                       key=lambda seed: -1 if seed is None else int(seed))
        return_by_seed = {
            "deterministic" if seed is None else str(seed): _means(
                [record for record in model_records if record["seed"] == seed],
                RETURN_METRICS,
            )
            for seed in seeds
        }
        return_values = list(return_by_seed.values())
        return_across_seeds = {
            metric: _statistics([value[metric] for value in return_values])
            for metric in RETURN_METRICS
        }
        by_series = {}
        for series in sorted({str(record["series"]) for record in model_records}):
            series_records = [record for record in model_records
                              if record["series"] == series]
            close_by_seed = {}
            for seed in seeds:
                selected = [record for record in series_records
                            if record["seed"] == seed]
                values = _means(selected, metrics)
                baseline = values["last_close_baseline_mae"]
                values["close_mae_delta"] = values["close_mae"] - baseline
                values["close_mae_relative"] = (
                    values["close_mae"] / baseline - 1.0 if baseline else
                    (0.0 if values["close_mae"] == 0.0 else None)
                )
                close_by_seed[
                    "deterministic" if seed is None else str(seed)
                ] = values
            close_values = list(close_by_seed.values())
            by_series[series] = {
                "by_seed": close_by_seed,
                "across_seeds": {
                    metric: _statistics([value[metric] for value in close_values])
                    for metric in (*metrics, "close_mae_delta", "close_mae_relative")
                },
            }
        summary[model] = {
            "by_series": by_series,
            "return_macro_by_seed": return_by_seed,
            "return_macro_across_seeds": return_across_seeds,
        }
    return summary


def expected_runs(sweep: Sweep, series_count: int) -> int:
    learned = sum(model in NEURAL for model in sweep.models) * len(sweep.seeds)
    deterministic = len(sweep.models) - sum(model in NEURAL for model in sweep.models)
    per_fold = learned + deterministic
    return series_count * per_fold * (sweep.folds * len(sweep.candidates) + 1)


def run_experiment(sweep: Sweep, series: Sequence[tuple[str, Path]],
                   device: torch.device, max_runs: int) -> dict[str, object]:
    if not series or len({name for name, _ in series}) != len(series):
        raise ValueError("series names must be nonempty and unique")
    runs = expected_runs(sweep, len(series))
    if runs > max_runs:
        raise ValueError(f"experiment requires {runs} runs; --max-runs is {max_runs}")
    torch.use_deterministic_algorithms(True)
    max_seq_len = max(candidate.seq_len for candidate in sweep.candidates)
    metadata, folds_by_series = [], {}
    for name, path in series:
        rows = read_csv(path)
        row_count = len(rows) // FEATURE_COUNT
        del rows
        timestamps = _timestamps(path, row_count)
        splits = walk_forward_splits(row_count - max_seq_len,
                                     sweep.folds, sweep.fold_fraction)
        holdout = holdout_split(row_count - max_seq_len, sweep.fold_fraction)
        folds_by_series[name] = (path, timestamps, splits, holdout)
        metadata.append({"name": name, "csv": str(path), "rows": row_count,
                         "sha256": _sha256(path),
                         "first_timestamp": timestamps[0],
                         "last_timestamp": timestamps[-1]})

    validation: list[dict[str, object]] = []
    for candidate in sweep.candidates:
        for name, (path, timestamps, splits, _) in folds_by_series.items():
            sample_start = max_seq_len - candidate.seq_len
            for fold, (train_count, validation_count) in enumerate(splits):
                data = prepare_data(path, candidate.config(), 0.7, 0.15,
                                    (train_count, validation_count, 1), sample_start)
                boundary = _boundary(
                    timestamps, max_seq_len, (train_count, validation_count),
                )
                for model_name in sweep.models:
                    if model_name in NEURAL:
                        for seed in sweep.seeds:
                            model, fit, loaders = _fit_neural(
                                model_name, candidate, data, sweep, seed, device,
                            )
                            validation.append(_validation_record(
                                model_name, candidate, name, fold, boundary, seed,
                                len(data.validation),
                                mean_loss(model, loaders[1], device), fit,
                            ))
                    else:
                        model = _deterministic(model_name, candidate, data).to(device)
                        loader = data_loaders(data, sweep.batch_size, 0)[1]
                        validation.append(_validation_record(
                            model_name, candidate, name, fold, boundary, None,
                            len(data.validation), mean_loss(model, loader, device), None,
                        ))

    selection = select_candidates(validation, sweep.models, sweep.candidates)
    candidates = {candidate.name: candidate for candidate in sweep.candidates}
    test: list[dict[str, object]] = []
    for model_name in sweep.models:
        candidate = candidates[str(selection[model_name]["candidate"])]
        for name, (path, timestamps, _, split) in folds_by_series.items():
            sample_start = max_seq_len - candidate.seq_len
            data = prepare_data(path, candidate.config(), 0.7, 0.15,
                                split, sample_start)
            boundary = _boundary(timestamps, max_seq_len, split)
            seeds = sweep.seeds if model_name in NEURAL else (None,)
            for seed in seeds:
                if seed is None:
                    model = _deterministic(model_name, candidate, data).to(device)
                    fit, loader = None, data_loaders(data, sweep.batch_size, 0)[2]
                else:
                    model, fit, loaders = _fit_neural(
                        model_name, candidate, data, sweep, seed, device,
                    )
                    loader = loaders[2]
                record = {
                    "model": model_name, "candidate": candidate.name,
                    "series": name, "fold": "holdout", "seed": seed,
                    "targets": boundary, "samples": len(data.test),
                    "metrics": evaluate(model, loader, data.target_mean,
                                        data.target_scale, device),
                }
                if fit:
                    record.update(asdict(fit))
                test.append(record)

    return {
        "schema": 1,
        "protocol": {
            "split": "expanding walk-forward by target time",
            "selection": "minimum mean validation scaled-return MSE",
            "aggregation": "macro mean over series, folds, and seeds",
            "test_policy": "evaluate selected candidates on one final holdout",
            "aligned_sequence_length": max_seq_len,
            "folds": sweep.folds, "fold_fraction": sweep.fold_fraction,
            "run_count": runs,
            "diagnostic_caps": {
                "linear_flat_features": MAX_FLAT_FEATURES,
                "mlp_parameters": MAX_MLP_PARAMETERS,
            },
        },
        "runtime": {"device": str(device), "python": sys.version.split()[0],
                    "torch": torch.__version__},
        "series": metadata,
        "sweep": {
            "candidates": [asdict(candidate) for candidate in sweep.candidates],
            "models": list(sweep.models), "seeds": list(sweep.seeds),
            "folds": sweep.folds, "fold_fraction": sweep.fold_fraction,
            "epochs": sweep.epochs, "patience": sweep.patience,
            "batch_size": sweep.batch_size,
        },
        "selection": selection,
        "validation": validation,
        "test": test,
        "summary": summarize(test, sweep.models),
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(report, file, allow_nan=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _series(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not SERIES_NAME.fullmatch(name) or not path:
        raise argparse.ArgumentTypeError("series must be NAME=CSV")
    return name, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("series", nargs="+", type=_series, metavar="NAME=CSV")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-runs", type=int, default=256)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        device = torch.device(args.device)
        if device.type == "meta" or args.max_runs <= 0:
            raise ValueError("device must execute tensors and max-runs must be positive")
        torch.empty(0, device=device)
        sweep = Sweep.read(args.sweep)
        report = run_experiment(sweep, args.series, device, args.max_runs)
        write_report(args.report, report)
    except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"report": str(args.report),
                      "selection": report["selection"]},
                     allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
