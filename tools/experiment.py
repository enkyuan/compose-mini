#!/usr/bin/env python3
"""Select forecasting configurations by walk-forward validation, then test once."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import TextIO
import argparse
import json
import math
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
from tools.data_v1 import (
    CLOSE_RETURN_TARGET, FEATURE_COUNT, TARGET_FORMULAS, TARGET_KINDS, read_bars,
)
from tools.backtest import experiment_fingerprint, read_policy
from tools.files import (
    atomic_text, file_sha256 as _sha256, require_disjoint, series_arg, write_json,
)
from tools.train import (
    FEATURE_NAMES, FEATURE_SETS, Fit, ForecastTransformer, TrainingData, Windows,
    data_loaders, evaluate, feature_lookback, fit_model, mean_loss, prepare_data,
)

MODELS = ("transformer", "linear", "mlp", "rolling_mean", "last_close")
NEURAL = frozenset(("transformer", "mlp"))
RETURN_METRICS = ("return_mse", "return_mae", "direction_accuracy")
EVALUATION_METRICS = (*RETURN_METRICS, "close_mae", "zero_return_baseline_mae")
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
    feature_set: str = "ohlcv"

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
        feature_set = value.get("feature_set", "ohlcv")
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"{name}.feature_set is unsupported")
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
            feature_set,
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
    target_horizon_bars: int = 1
    alignment_horizon_bars: int = 1
    target_kind: str = CLOSE_RETURN_TARGET

    def __post_init__(self) -> None:
        _integer(self.target_horizon_bars, "target_horizon_bars")
        _integer(self.alignment_horizon_bars, "alignment_horizon_bars")
        if self.target_horizon_bars > self.alignment_horizon_bars:
            raise ValueError("target_horizon_bars cannot exceed alignment_horizon_bars")
        if self.target_kind not in TARGET_KINDS:
            raise ValueError("target_kind is unsupported")

    @classmethod
    def read(cls, path: Path) -> Sweep:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read sweep: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("sweep must be an object")
        allowed = {"candidates", "models", "seeds", "folds", "fold_fraction",
                   "epochs", "patience", "batch_size", "target_horizon_bars",
                   "alignment_horizon_bars", "target_kind"}
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
        horizon = _integer(value.get("target_horizon_bars", 1),
                           "target_horizon_bars")
        return cls(
            candidates, tuple(models_value), seeds, folds, fraction,
            _integer(value.get("epochs", 100), "epochs"),
            _integer(value.get("patience", 10), "patience"),
            _integer(value.get("batch_size", 64), "batch_size"),
            horizon, _integer(value.get("alignment_horizon_bars", horizon),
                              "alignment_horizon_bars"),
            value.get("target_kind", CLOSE_RETURN_TARGET),
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
        self.horizon_bars = data.horizon_bars
        self.stationary = data.feature_set != "ohlcv"
        for name, value in (
            ("feature_mean", data.feature_mean),
            ("feature_scale", data.feature_scale),
            ("target_mean", data.target_mean),
            ("target_scale", data.target_scale),
        ):
            self.register_buffer(name, value)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        raw = values * self.feature_scale + self.feature_mean
        returns = (raw[:, :, :2].sum(2) if self.stationary else
                   torch.log(raw[:, 1:, 3] / raw[:, :-1, 3]))
        prediction = returns[:, -self.window:].mean(1) * self.horizon_bars
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


def purged_split(split: tuple[int, ...], gap: int,
                 preserve_last: bool = True) -> tuple[int, ...]:
    """Remove boundary labels whose horizons overlap the following split."""
    purge_count = len(split) - int(preserve_last)
    result = tuple(count - gap if index < purge_count else count
                   for index, count in enumerate(split))
    if min(result) <= 0:
        raise ValueError("series blocks must exceed the horizon embargo")
    return result


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


def _boundary(timestamps: list[str], target_offset: int,
              split: tuple[int, ...], gap: int) -> dict[str, list[str]]:
    starts, offset = [], target_offset
    for count in split:
        starts.append(offset)
        offset += count + gap
    return {
        name: [timestamps[start], timestamps[start + count - 1]]
        for name, start, count in zip(
            ("train", "validation", "test"), starts, split, strict=False,
        )
    }


def _candidate_data(path: Path, candidate: Candidate, split: tuple[int, int, int],
                    max_history: int, sweep: Sweep) -> TrainingData:
    history = candidate.seq_len + feature_lookback(candidate.feature_set)
    return prepare_data(
        path, candidate.config(), 0.7, 0.15, split=split,
        sample_start=max_history - history + sweep.alignment_horizon_bars -
        sweep.target_horizon_bars,
        feature_set=candidate.feature_set,
        horizon_bars=sweep.target_horizon_bars,
        split_gap=sweep.alignment_horizon_bars - 1,
        target_kind=sweep.target_kind,
    )


def _prediction_records(model: str, candidate: Candidate, series: str,
                        seed: int | None, data: TrainingData,
                        timestamps: Sequence[str], csv_sha256: str,
                        predictions: Sequence[float], dataset: Windows,
                        split: str, fold: int | None,
                        ) -> Iterator[dict[str, object]]:
    if split not in ("validation", "test") or \
       (split == "validation") != (fold is not None) or \
       len(predictions) != len(dataset):
        raise ValueError("prediction metadata does not match its split")
    start = feature_lookback(candidate.feature_set) + dataset.start + \
        candidate.seq_len - 1
    for offset, prediction in enumerate(predictions):
        as_of = start + offset
        target = as_of + data.horizon_bars
        yield {
            "schema": 2, "split": split, "fold": fold, "series": series,
            "model": model, "candidate": candidate.name,
            "feature_set": candidate.feature_set, "seed": seed,
            "csv_sha256": csv_sha256,
            "as_of": timestamps[as_of], "target_time": timestamps[target],
            "horizon_bars": data.horizon_bars,
            "target_kind": data.target_kind,
            "predicted_log_return": prediction,
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
                       samples: int, loss: float, metrics: Mapping[str, float],
                       fit: Fit | None) -> dict[str, object]:
    record: dict[str, object] = {
        "model": model_name, "candidate": candidate.name, "series": series,
        "feature_set": candidate.feature_set, "fold": fold, "seed": seed,
        "targets": boundary, "samples": samples,
        "validation_scaled_mse": loss, "metrics": dict(metrics),
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


def _statistics(values: Sequence[float | None]) -> dict[str, object]:
    finite = [value for value in values if value is not None]
    return ({"count": len(finite), "mean": fmean(finite),
             "stddev": pstdev(finite)} if finite else
            {"count": 0, "mean": None, "stddev": None})


def _validation_metric(record: Mapping[str, object], metric: str) -> float:
    return float(record[metric] if metric == "validation_scaled_mse" else
                 record["metrics"][metric])


def summarize_validation(records: Sequence[Mapping[str, object]],
                         models: Sequence[str], candidates: Sequence[Candidate]
                         ) -> dict[str, object]:
    metrics = ("validation_scaled_mse", *EVALUATION_METRICS)
    indexed = {
        (record["model"], record["candidate"], record["series"],
         record["fold"], record["seed"]): record
        for record in records
    }
    summary: dict[str, object] = {}
    for model in models:
        candidate_records = {
            candidate.name: [record for record in records
                             if record["model"] == model and
                             record["candidate"] == candidate.name]
            for candidate in candidates
        }
        pairs = {}
        for candidate in candidates:
            if candidate.feature_set == "ohlcv":
                continue
            for control in candidates:
                if control.feature_set != "ohlcv":
                    continue
                deltas = {metric: [] for metric in metrics}
                for record in candidate_records[candidate.name]:
                    key = (model, control.name, record["series"],
                           record["fold"], record["seed"])
                    if comparison := indexed.get(key):
                        for metric in metrics:
                            deltas[metric].append(
                                _validation_metric(record, metric) -
                                _validation_metric(comparison, metric)
                            )
                pairs[f"{candidate.name}-minus-{control.name}"] = {
                    metric: _statistics(values) for metric, values in deltas.items()
                }
        summary[model] = {
            "candidates": {
                candidate.name: {
                    "feature_set": candidate.feature_set,
                    "metrics": {
                        metric: _statistics([
                            _validation_metric(record, metric)
                            for record in candidate_records[candidate.name]
                        ])
                        for metric in metrics
                    },
                }
                for candidate in candidates
            },
            "paired_deltas": pairs,
        }
    return summary


def summarize(records: Sequence[Mapping[str, object]],
              models: Sequence[str]) -> dict[str, object]:
    metrics = EVALUATION_METRICS
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
                baseline = values["zero_return_baseline_mae"]
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


def _model_runs(sweep: Sweep, models: Sequence[str]) -> int:
    return sum(len(sweep.seeds) if model in NEURAL else 1 for model in models)


def expected_runs(sweep: Sweep, series_count: int,
                  evaluate_test: bool = True,
                  test_models: Sequence[str] | None = None) -> int:
    validation = _model_runs(sweep, sweep.models) * sweep.folds * \
        len(sweep.candidates)
    selected = sweep.models if test_models is None else test_models
    test = _model_runs(sweep, selected) if evaluate_test else 0
    return series_count * (validation + test)


def _validation_contract(sweep: Sweep, series: Sequence[Mapping[str, object]],
                         test_contract: Sequence[Mapping[str, object]],
                         selection: Mapping[str, object],
                         validation: Sequence[Mapping[str, object]],
                         ) -> dict[str, object]:
    return {
        "series": list(series), "test_contract": list(test_contract),
        "sweep": {
            "candidates": [asdict(candidate) for candidate in sweep.candidates],
            "models": list(sweep.models), "seeds": list(sweep.seeds),
            "folds": sweep.folds, "fold_fraction": sweep.fold_fraction,
            "epochs": sweep.epochs, "patience": sweep.patience,
            "batch_size": sweep.batch_size,
            "target_horizon_bars": sweep.target_horizon_bars,
            "alignment_horizon_bars": sweep.alignment_horizon_bars,
            "target_kind": sweep.target_kind,
        },
        "selection": dict(selection), "validation": list(validation),
    }


def _authorize_test(contract: Mapping[str, object],
                    policies: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if not policies:
        raise ValueError("test evaluation requires a frozen policy")
    fingerprint = experiment_fingerprint(contract)
    sweep = contract["sweep"]
    configurations = {item["name"]: item for item in sweep["candidates"]}
    names = sorted(item["name"] for item in contract["series"])
    models = tuple(policy["model"] for policy in policies)
    if len(models) != len(set(models)):
        raise ValueError("test policies must name unique models")
    for policy in policies:
        model = policy["model"]
        try:
            candidate = contract["selection"][model]["candidate"]
            configuration = configurations[candidate]
        except (KeyError, TypeError) as error:
            raise ValueError("test policy model is not in the experiment") from error
        expected_seeds = sorted(sweep["seeds"]) if model in NEURAL else []
        if policy["validation_fingerprint"] != fingerprint or \
           policy["candidate"] != candidate or \
           policy["feature_set"] != configuration["feature_set"] or \
           policy["target_kind"] != sweep["target_kind"] or \
           policy["horizon_bars"] != sweep["target_horizon_bars"] or \
           policy["seeds"] != expected_seeds or policy["series"] != names or \
           policy["test_grid"] != contract["test_contract"]:
            raise ValueError("test policy does not match the validation contract")
    return models


def run_experiment(sweep: Sweep, series: Sequence[tuple[str, Path]],
                   device: torch.device, max_runs: int,
                   prediction_records: list[dict[str, object]] | None = None,
                   validation_prediction_records: list[dict[str, object]] | None = None,
                   evaluate_test: bool = True,
                   test_authorizer: Callable[
                       [Mapping[str, object]], Sequence[str]
                   ] | None = None,
                   ) -> dict[str, object]:
    if not series or len({name for name, _ in series}) != len(series):
        raise ValueError("series names must be nonempty and unique")
    if not evaluate_test and prediction_records is not None:
        raise ValueError("validation-only experiments cannot collect test predictions")
    if evaluate_test and test_authorizer is None:
        raise ValueError("test evaluation requires explicit authorization")
    runs = expected_runs(sweep, len(series), False)
    if runs > max_runs:
        raise ValueError(f"experiment requires at least {runs} runs; "
                         f"--max-runs is {max_runs}")
    torch.use_deterministic_algorithms(True)
    max_history = max(candidate.seq_len + feature_lookback(candidate.feature_set)
                      for candidate in sweep.candidates)
    horizon = sweep.target_horizon_bars
    alignment = sweep.alignment_horizon_bars
    target_offset = max_history + alignment - 1
    gap = alignment - 1
    metadata, test_contract, folds_by_series = [], [], {}
    for name, path in series:
        checksum = _sha256(path)
        timestamps, rows = read_bars(path)
        if _sha256(path) != checksum:
            raise ValueError("CSV changed while the experiment was reading it")
        row_count = len(timestamps)
        del rows
        samples = row_count - target_offset
        splits = tuple(
            purged_split(split, gap, preserve_last=False)
            for split in walk_forward_splits(
                samples, sweep.folds, sweep.fold_fraction,
            )
        )
        holdout = purged_split(holdout_split(samples, sweep.fold_fraction), gap)
        folds_by_series[name] = (path, timestamps, checksum, splits, holdout)
        test_boundary = _boundary(
            timestamps, target_offset, holdout, gap,
        )["test"]
        test_contract.append({
            "series": name, "samples": holdout[2],
            "first_target_time": test_boundary[0],
            "last_target_time": test_boundary[1],
        })
        metadata.append({"name": name, "csv": str(path), "rows": row_count,
                         "sha256": checksum,
                         "first_timestamp": timestamps[0],
                         "last_timestamp": timestamps[-1]})

    validation: list[dict[str, object]] = []
    for candidate in sweep.candidates:
        for name, (path, timestamps, checksum, splits, _) in folds_by_series.items():
            for fold, (train_count, validation_count) in enumerate(splits):
                data = _candidate_data(
                    path, candidate, (train_count, validation_count, 1),
                    max_history, sweep,
                )
                boundary = _boundary(
                    timestamps, target_offset, (train_count, validation_count), gap,
                )
                for model_name in sweep.models:
                    seeds = sweep.seeds if model_name in NEURAL else (None,)
                    for seed in seeds:
                        if seed is None:
                            model = _deterministic(
                                model_name, candidate, data,
                            ).to(device)
                            fit = None
                            loader = data_loaders(data, sweep.batch_size, 0)[1]
                        else:
                            model, fit, loaders = _fit_neural(
                                model_name, candidate, data, sweep, seed, device,
                            )
                            loader = loaders[1]
                        predictions = [] if validation_prediction_records is not None \
                            else None
                        metrics = evaluate(
                            model, loader, data.target_mean, data.target_scale,
                            device, predictions,
                        )
                        validation.append(_validation_record(
                            model_name, candidate, name, fold, boundary, seed,
                            len(data.validation), mean_loss(model, loader, device),
                            metrics, fit,
                        ))
                        if validation_prediction_records is not None and \
                           predictions is not None:
                            validation_prediction_records.extend(
                                _prediction_records(
                                    model_name, candidate, name, seed, data,
                                    timestamps, checksum, predictions,
                                    data.validation, "validation", fold,
                                )
                            )

    selection = select_candidates(validation, sweep.models, sweep.candidates)
    contract = _validation_contract(
        sweep, metadata, test_contract, selection, validation,
    )
    test_models: Sequence[str] = ()
    if evaluate_test and test_authorizer is not None:
        test_models = tuple(test_authorizer(contract))
        if not test_models or len(test_models) != len(set(test_models)) or \
           any(model not in sweep.models for model in test_models):
            raise ValueError("test authorization returned invalid models")
        runs = expected_runs(sweep, len(series), True, test_models)
        if runs > max_runs:
            raise ValueError(f"experiment requires {runs} runs; "
                             f"--max-runs is {max_runs}")
    candidates = {candidate.name: candidate for candidate in sweep.candidates}
    test: list[dict[str, object]] = []
    if evaluate_test:
        for model_name in test_models:
            candidate = candidates[str(selection[model_name]["candidate"])]
            for name, (path, timestamps, checksum, _, split) in \
                    folds_by_series.items():
                data = _candidate_data(path, candidate, split, max_history, sweep)
                boundary = _boundary(timestamps, target_offset, split, gap)
                seeds = sweep.seeds if model_name in NEURAL else (None,)
                for seed in seeds:
                    if seed is None:
                        model = _deterministic(
                            model_name, candidate, data,
                        ).to(device)
                        fit = None
                        loader = data_loaders(data, sweep.batch_size, 0)[2]
                    else:
                        model, fit, loaders = _fit_neural(
                            model_name, candidate, data, sweep, seed, device,
                        )
                        loader = loaders[2]
                    predictions = [] if prediction_records is not None else None
                    record = {
                        "model": model_name, "candidate": candidate.name,
                        "feature_set": candidate.feature_set, "series": name,
                        "fold": "holdout", "seed": seed, "targets": boundary,
                        "samples": len(data.test),
                        "metrics": evaluate(
                            model, loader, data.target_mean, data.target_scale,
                            device, predictions,
                        ),
                    }
                    if fit:
                        record.update(asdict(fit))
                    test.append(record)
                    if prediction_records is not None and predictions is not None:
                        prediction_records.extend(_prediction_records(
                            model_name, candidate, name, seed, data, timestamps,
                            checksum, predictions, data.test, "test", None,
                        ))

    for metadata_record, (_, path) in zip(metadata, series, strict=True):
        if _sha256(path) != metadata_record["sha256"]:
            raise ValueError("CSV changed during the experiment")

    return {
        "schema": 5,
        "protocol": {
            "split": "embargoed expanding walk-forward by target time",
            "selection": "minimum mean validation scaled-return MSE",
            "selection_aggregation": "macro mean over series, folds, and seeds",
            "holdout_aggregation": "macro mean over series and seeds",
            "phase": "validation-and-test" if evaluate_test else "validation",
            "test_policy": (
                "evaluate selected candidates on one final holdout" if evaluate_test
                else "deferred until policy selection"
            ),
            "aligned_history_bars": max_history,
            "target_horizon_bars": horizon,
            "target_kind": sweep.target_kind,
            "target_formula": TARGET_FORMULAS[sweep.target_kind],
            "zero_return_reference": (
                "close[t]" if sweep.target_kind == CLOSE_RETURN_TARGET else
                "open[t + 1]"
            ),
            "alignment_horizon_bars": alignment,
            "embargo_bars": gap,
            "feature_contract": "experiment-only; artifact V1 remains OHLCV",
            "feature_sets": {
                name: list(FEATURE_NAMES[name])
                for name in dict.fromkeys(candidate.feature_set
                                          for candidate in sweep.candidates)
            },
            "folds": sweep.folds, "fold_fraction": sweep.fold_fraction,
            "run_count": runs,
            "diagnostic_caps": {
                "linear_flat_features": MAX_FLAT_FEATURES,
                "mlp_parameters": MAX_MLP_PARAMETERS,
            },
        },
        "runtime": {"device": str(device), "python": sys.version.split()[0],
                    "torch": torch.__version__},
        **contract,
        "validation_summary": summarize_validation(
            validation, sweep.models, sweep.candidates,
        ),
        "test": test,
        "summary": summarize(test, sweep.models),
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    write_json(path, report)


def write_predictions(path: Path,
                      records: Sequence[Mapping[str, object]]) -> None:
    def write(file: TextIO) -> None:
        for record in records:
            json.dump(record, file, allow_nan=False, sort_keys=True)
            file.write("\n")

    atomic_text(path, write)


def _series(value: str) -> tuple[str, Path]:
    return series_arg(value, SERIES_NAME)


def _policy_input(path: Path) -> tuple[Path, str, dict[str, object]]:
    checksum = _sha256(path)
    policy = read_policy(path)
    if _sha256(path) != checksum:
        raise ValueError("test policy changed while it was being read")
    return path, checksum, policy


def _verify_policy_inputs(
    inputs: Sequence[tuple[Path, str, Mapping[str, object]]],
) -> None:
    if any(_sha256(path) != checksum for path, checksum, _ in inputs):
        raise ValueError("test policy changed during the experiment")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("series", nargs="+", type=_series, metavar="NAME=CSV")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon-bars", type=int)
    parser.add_argument("--target-kind", choices=TARGET_KINDS)
    parser.add_argument("--max-runs", type=int, default=256)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--validation-predictions", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--policy", dest="policies", action="append", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        policy_paths = tuple(args.policies or ())
        outputs = [path for path in (
            args.report, args.predictions, args.validation_predictions,
        ) if path]
        require_disjoint(
            [args.sweep, *policy_paths, *(path for _, path in args.series)], outputs,
        )
        device = torch.device(args.device)
        if device.type == "meta" or args.max_runs <= 0:
            raise ValueError("device must execute tensors and max-runs must be positive")
        if args.validation_only and args.predictions:
            raise ValueError("validation-only mode cannot export test predictions")
        if (args.validation_only and policy_paths) or \
           (not args.validation_only and not policy_paths) or \
           len(policy_paths) != len(set(policy_paths)):
            raise ValueError("full experiments require unique frozen policies")
        torch.empty(0, device=device)
        sweep = Sweep.read(args.sweep)
        if args.horizon_bars is not None:
            sweep = replace(sweep, target_horizon_bars=args.horizon_bars)
        if args.target_kind is not None:
            sweep = replace(sweep, target_kind=args.target_kind)
        policy_inputs = tuple(_policy_input(path) for path in policy_paths)
        predictions = [] if args.predictions else None
        validation_predictions = [] if args.validation_predictions else None
        report = run_experiment(
            sweep, args.series, device, args.max_runs, predictions,
            validation_predictions, not args.validation_only,
            (lambda contract: _authorize_test(
                contract, tuple(item[2] for item in policy_inputs),
            )) if policy_inputs else None,
        )
        _verify_policy_inputs(policy_inputs)
        if policy_inputs:
            report["policies"] = [
                {"path": str(path), "sha256": checksum,
                 "model": policy["model"]}
                for path, checksum, policy in policy_inputs
            ]
        if args.predictions and predictions is not None:
            write_predictions(args.predictions, predictions)
            report["prediction_ledger"] = {
                "schema": 2, "path": str(args.predictions),
                "records": len(predictions), "sha256": _sha256(args.predictions),
            }
        if args.validation_predictions and validation_predictions is not None:
            write_predictions(args.validation_predictions, validation_predictions)
            report["validation_prediction_ledger"] = {
                "schema": 2, "path": str(args.validation_predictions),
                "records": len(validation_predictions),
                "sha256": _sha256(args.validation_predictions),
            }
        _verify_policy_inputs(policy_inputs)
        write_report(args.report, report)
    except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    result = {"report": str(args.report), "selection": report["selection"]}
    if args.predictions:
        result["predictions"] = str(args.predictions)
    if args.validation_predictions:
        result["validation_predictions"] = str(args.validation_predictions)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
