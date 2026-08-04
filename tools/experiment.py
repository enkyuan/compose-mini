#!/usr/bin/env python3
"""Select forecasting configurations by walk-forward validation, then test once."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean, median_low, pstdev
from typing import TextIO
import argparse
import hashlib
import json
import math
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import (
        ConcatDataset, DataLoader, Dataset, Sampler,
    )
except ModuleNotFoundError as error:
    raise SystemExit("experiments require PyTorch: python -m pip install torch") from error

from tools.artifact_v1 import Config
from tools.data_v1 import (
    CLOSE_RETURN_TARGET, FEATURE_COUNT, TARGET_FORMULAS, TARGET_KINDS, read_bars,
)
from tools.backtest import experiment_fingerprint, read_policy
from tools.chronology import (
    holdout_split, purged_split, walk_forward_splits,
)
from tools.files import (
    FrozenInput, atomic_text, exclusive_text, file_sha256, freeze_inputs,
    require_disjoint, series_arg, verify_frozen, write_json,
    write_json_exclusive,
)
from tools.panel_contract import (
    PanelExecution, TorchIdentity, executable_binding, freeze_panel_execution,
    source_tree,
)
from tools.session_samples import SampleRows
from tools.train import (
    FEATURE_NAMES, FEATURE_SETS, DataSplits, Fit, ForecastTransformer,
    TrainingData, UpdateFit, Windows, data_loaders, evaluate, feature_lookback,
    fit_epochs, fit_model, fit_updates, mean_loss, prepare_rows,
)
from tools.universe_contract import PackedRows

PANEL_MODELS = ("panel_transformer", "conditioned_panel_transformer")
PANEL_MODEL_SET = frozenset(PANEL_MODELS)
TRANSFORMERS = frozenset(("transformer", *PANEL_MODELS))
NEURAL = TRANSFORMERS | {"mlp"}
SHARED_NEURAL = PANEL_MODEL_SET | {"mlp"}
LOCAL_MODELS = ("transformer", "linear", "mlp", "rolling_mean", "last_close")
MODELS = (*LOCAL_MODELS, *PANEL_MODELS)
RETURN_METRICS = ("return_mse", "return_mae", "direction_accuracy")
EVALUATION_METRICS = (*RETURN_METRICS, "close_mae", "zero_return_baseline_mae")
SERIES_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
MAX_FLAT_FEATURES = 2_048
MAX_MLP_PARAMETERS = 8_388_608
_SAMPLER_CHUNK = 65_536


class _SeriesDataset(Dataset):
    def __init__(self, dataset: Dataset, series_id: int) -> None:
        self.dataset, self.series_id = dataset, series_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[object, ...]:
        values, *rest = self.dataset[index]
        return (values, self.series_id), *rest


class SeriesTransformer(nn.Module):
    def __init__(self, config: Config, series_count: int) -> None:
        super().__init__()
        self.model = ForecastTransformer(config)
        self.series = nn.Embedding(series_count, config.model_dim)
        nn.init.zeros_(self.series.weight)

    def forward(
        self, values: torch.Tensor, series_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(values, self.series(series_id))


def _torch_identity() -> TorchIdentity:
    package = Path(torch.__file__).resolve().parent
    return TorchIdentity(
        executable_binding(
            Path(sys.executable), f"Python {sys.version.split()[0]}",
        ),
        str(torch.__version__), torch.version.git_version,
        torch.version.cuda, torch.__config__.show(), source_tree(package),
    )


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
                      self.layers, self.seq_len,
                      in_dim=len(FEATURE_NAMES[self.feature_set]))


def _model_fingerprint(model: nn.Module, data: TrainingData,
                       candidate: Candidate) -> str:
    digest = hashlib.sha256(json.dumps(
        asdict(candidate), allow_nan=False, separators=(",", ":"),
        sort_keys=True,
    ).encode())
    tensors = (
        ("feature_mean", data.feature_mean),
        ("feature_scale", data.feature_scale),
        ("target_mean", data.target_mean),
        ("target_scale", data.target_scale),
        *sorted(model.state_dict().items()),
    )
    for name, tensor in tensors:
        values = tensor.detach().reshape(-1)
        digest.update(name.encode("ascii") + b"\0")
        for start in range(0, values.numel(), 16_384):
            items = values[start:start + 16_384].to(
                device="cpu", dtype=torch.float32,
            ).tolist()
            digest.update(struct.pack(f"<{len(items)}f", *items))
    return digest.hexdigest()


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
        models_value = value.get("models", list(LOCAL_MODELS))
        if not isinstance(models_value, list) or not models_value or \
           any(model not in MODELS for model in models_value) or \
           len(set(models_value)) != len(models_value):
            raise ValueError("models must be unique supported names")
        for candidate in candidates:
            flat = candidate.seq_len * len(FEATURE_NAMES[candidate.feature_set])
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
    def __init__(self, seq_len: int, hidden: int,
                 in_dim: int = FEATURE_COUNT) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * in_dim, hidden),
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


def _label_available(
    timestamps: Sequence[str], target_offset: int,
    split: Sequence[int], gap: int, horizon: int, *,
    sample_rows: Sequence[SampleRows] | None = None,
) -> None:
    if sample_rows is not None:
        starts, offset = [], 0
        for count in split:
            starts.append(offset)
            offset += count + gap
        segments = tuple(
            (start, count) for start, count in zip(
                starts, split, strict=True,
            ) if count
        )
        if offset - gap > len(sample_rows) or any(
            sample_rows[start + count - 1].target >
            sample_rows[next_start].as_of or
            sample_rows[start + count - 1].as_of_ordinal + horizon >
            sample_rows[next_start].as_of_ordinal
            for (start, count), (next_start, _) in zip(
                segments, segments[1:], strict=False,
            )
        ):
            raise ValueError(
                "training labels are unavailable at the next decision"
            )
        return
    last_train_target = target_offset + split[0] - 1
    first_validation_as_of = target_offset + split[0] + gap - horizon
    if timestamps[last_train_target] > timestamps[first_validation_as_of]:
        raise ValueError(
            "training labels are unavailable at the first validation decision"
        )


def _matrix(dataset: Windows) -> tuple[torch.Tensor, torch.Tensor]:
    start, end = dataset.start, dataset.start + dataset.count
    windows = (
        torch.stack([
            dataset.features[index:index + dataset.seq_len]
            for index in dataset.feature_starts[start:end]
        ])
        if dataset.indexed else
        dataset.features.unfold(0, dataset.seq_len, 1).transpose(1, 2)[
            start:end
        ]
    )
    return (windows.reshape(dataset.count, -1).double(),
            dataset.targets[start:end].double())


def _solve_affine(
    value_mean: torch.Tensor, target_mean: torch.Tensor,
    gram: torch.Tensor, cross: torch.Tensor, ridge: float,
) -> Affine:
    if any(not torch.isfinite(item).all()
           for item in (value_mean, target_mean, gram, cross)):
        raise ValueError("affine moments must be finite")
    gram.diagonal().add_(_number(ridge, "ridge", 0.0, True))
    try:
        weight = torch.linalg.solve(gram, cross)
    except RuntimeError as error:
        raise ValueError("ridge moments do not produce a stable solve") from error
    bias = target_mean - value_mean @ weight
    converted = weight.float(), bias.float()
    if any(not torch.isfinite(item).all()
           for item in (weight, bias, *converted)):
        raise ValueError("ridge parameters must remain finite binary32")
    return Affine(*converted)


def linear_model(data: TrainingData, ridge: float) -> Affine:
    values, targets = _matrix(data.train)
    value_mean, target_mean = values.mean(0), targets.mean()
    centered = values - value_mean
    return _solve_affine(
        value_mean, target_mean, centered.T @ centered,
        centered.T @ (targets - target_mean), ridge,
    )


def stock_macro_linear_model(
    members: Sequence[TrainingData], ridge: float,
) -> Affine:
    """Fit ridge to the equal-stock mean of scaled training losses."""
    if not members or any(
        not isinstance(member, TrainingData) or
        not isinstance(member.train, Windows) or not len(member.train)
        for member in members
    ):
        raise ValueError("stock-macro ridge requires nonempty training series")
    value_means, target_means = [], []
    gram = cross = None
    width = None
    for member in members:
        values, targets = _matrix(member.train)
        if (width is not None and values.shape[1] != width) or \
           not torch.isfinite(values).all() or not torch.isfinite(targets).all():
            raise ValueError("stock-macro ridge matrices are incompatible")
        width = values.shape[1]
        value_mean, target_mean = values.mean(0), targets.mean()
        centered = values - value_mean
        covariance = centered.T @ centered / len(targets)
        relation = centered.T @ (targets - target_mean) / len(targets)
        gram = covariance if gram is None else gram.add_(covariance)
        cross = relation if cross is None else cross.add_(relation)
        value_means.append(value_mean)
        target_means.append(target_mean)
    means, targets = torch.stack(value_means), torch.stack(target_means)
    value_mean, target_mean = means.mean(0), targets.mean()
    centered, count = means - value_mean, len(members)
    gram.div_(count).addmm_(centered.T, centered, alpha=1.0 / count)
    cross.div_(count).addmv_(
        centered.T, targets - target_mean, alpha=1.0 / count,
    )
    return _solve_affine(value_mean, target_mean, gram, cross, ridge)


def _boundary(
    timestamps: Sequence[str], target_offset: int,
    split: tuple[int, ...], gap: int, *,
    sample_rows: Sequence[SampleRows] | None = None,
) -> dict[str, list[str]]:
    starts, offset = [], 0 if sample_rows is not None else target_offset
    for count in split:
        starts.append(offset)
        offset += count + gap
    return {
        name: [] if not count else (
            [
                timestamps[sample_rows[start].target],
                timestamps[sample_rows[start + count - 1].target],
            ] if sample_rows is not None else [
                timestamps[start], timestamps[start + count - 1],
            ]
        )
        for name, start, count in zip(
            ("train", "validation", "test"), starts, split, strict=False,
        )
    }


def _candidate_data(rows: array, candidate: Candidate,
                    split: tuple[int, int, int],
                    max_history: int, sweep: Sweep, *,
                    sample_rows: Sequence[SampleRows] | None = None,
                    prepurged: bool = False,
                    ) -> TrainingData:
    history = candidate.seq_len + feature_lookback(candidate.feature_set)
    if prepurged and sample_rows is None:
        raise ValueError("prepurged data requires indexed samples")
    return prepare_rows(
        rows, candidate.config(), 0.7, 0.15, split=split,
        sample_start=(
            0 if sample_rows is not None else
            max_history - history + sweep.alignment_horizon_bars -
            sweep.target_horizon_bars
        ),
        feature_set=candidate.feature_set,
        horizon_bars=sweep.target_horizon_bars,
        split_gap=0 if prepurged else sweep.alignment_horizon_bars - 1,
        target_kind=sweep.target_kind,
        sample_rows=sample_rows,
        allow_empty_later=prepurged,
    )


def _prepare_packed(rows: array, candidate: Candidate, packed: PackedRows,
                    max_history: int, sweep: Sweep) -> TrainingData:
    """Prepare embargoed development rows without exposing a test block."""
    if not isinstance(packed, PackedRows) or len(packed.counts) != 2 or \
       type(packed.counts[0]) is not int or packed.counts[0] < 1 or \
       type(packed.counts[1]) is not int or packed.counts[1] < 0 or \
       sum(packed.counts) != len(packed.rows):
        raise ValueError("packed rows must cover only train and validation")
    boundary = packed.counts[0]
    if packed.counts[1] and packed.rows[boundary - 1].as_of_ordinal + \
       sweep.alignment_horizon_bars > packed.rows[boundary].as_of_ordinal:
        raise ValueError("packed rows do not preserve the alignment embargo")
    return _candidate_data(
        rows, candidate, (*packed.counts, 0), max_history, sweep,
        sample_rows=packed.rows, prepurged=True,
    )


def _prediction_records(model: str, candidate: Candidate, series: str,
                        seed: int | None, data: TrainingData,
                        timestamps: Sequence[str], csv_sha256: str,
                        predictions: Sequence[float], dataset: Windows,
                        split: str, fold: int | None,
                        ) -> Iterator[dict[str, object]]:
    if split not in ("calibration", "validation", "test") or \
       split == "validation" and (type(fold) is not int or fold < 0) or \
       split != "validation" and fold is not None or \
       len(predictions) != len(dataset):
        raise ValueError("prediction metadata does not match its split")
    start = feature_lookback(candidate.feature_set) + dataset.start + \
        candidate.seq_len - 1
    for offset, prediction in enumerate(predictions):
        if dataset.sample_rows is None:
            as_of = start + offset
            times = {
                "schema": 3, "as_of": timestamps[as_of],
                "target_time": timestamps[as_of + data.horizon_bars],
            }
        else:
            coordinates = dataset.sample_rows[dataset.start + offset]
            times = {
                "schema": 4, "as_of": timestamps[coordinates.as_of],
                "entry_time": timestamps[coordinates.entry],
                "target_time": timestamps[coordinates.target],
            }
        yield times | {
            "split": split, "fold": fold, "series": series,
            "model": model, "candidate": candidate.name,
            "feature_set": candidate.feature_set, "seed": seed,
            "csv_sha256": csv_sha256,
            "horizon_bars": data.horizon_bars,
            "target_kind": data.target_kind,
            "predicted_log_return": prediction,
        }


def _neural_model(model_name: str, candidate: Candidate,
                  series_count: int = 0) -> nn.Module:
    if model_name == "conditioned_panel_transformer":
        return SeriesTransformer(candidate.config(), series_count)
    if model_name in TRANSFORMERS:
        return ForecastTransformer(candidate.config())
    if model_name == "mlp":
        return FlatMLP(candidate.seq_len, candidate.mlp_dim,
                       len(FEATURE_NAMES[candidate.feature_set]))
    raise ValueError("unsupported neural model")


def _fit_neural(model_name: str, candidate: Candidate, data: DataSplits,
                sweep: Sweep, seed: int,
                device: torch.device, series_count: int = 0,
                ) -> tuple[nn.Module, Fit, tuple[object, ...]]:
    torch.manual_seed(seed)
    model = _neural_model(model_name, candidate, series_count).to(device)
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
                       fold: int | None, boundary: dict[str, list[str]],
                       seed: int | None,
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


def _selected_epochs(records: Sequence[Mapping[str, object]], model: str,
                     candidate: str, series: str, seed: int) -> int:
    values = [
        int(record["best_epoch"]) for record in records
        if record["model"] == model and record["candidate"] == candidate and
        record["series"] == series and record["seed"] == seed
    ]
    if not values:
        raise ValueError("neural model has no selected epoch evidence")
    return median_low(values)


def _panel_selected_epochs(
    records: Sequence[Mapping[str, object]], model_name: str,
    candidate: str, series: Sequence[str], seed: int, folds: int,
) -> int:
    selected = [
        record for record in records
        if record["model"] == model_name and
        record["candidate"] == candidate and record["seed"] == seed
    ]
    epochs = []
    for fold in range(folds):
        copies = [record for record in selected if record["fold"] == fold]
        if len(copies) != len(series) or \
           tuple(record["series"] for record in copies) != tuple(series) or \
           len({int(record["best_epoch"]) for record in copies}) != 1:
            raise ValueError("panel model has invalid shared epoch evidence")
        epochs.append(int(copies[0]["best_epoch"]))
    if len(selected) != folds * len(series):
        raise ValueError("panel model has invalid shared epoch evidence")
    return median_low(epochs)


def _conditioned(data: DataSplits, series_id: int) -> DataSplits:
    return DataSplits(*(
        _SeriesDataset(getattr(data, name), series_id)
        for name in ("train", "validation", "test")
    ))


def _panel_members(
    members: Sequence[DataSplits], conditioned: bool = False,
) -> tuple[DataSplits, ...]:
    if not members:
        raise ValueError("panel data requires at least one series")
    return tuple(
        _conditioned(member, series_id) if conditioned else member
        for series_id, member in enumerate(members)
    )


def _panel_data(
    members: Sequence[DataSplits], conditioned: bool = False,
) -> DataSplits:
    splits = _panel_members(members, conditioned)
    return DataSplits(*(
        ConcatDataset([getattr(member, name) for member in splits])
        for name in ("train", "validation", "test")
    ))


class _ChunkedWeightedSampler(Sampler[int]):
    """Draw weighted indices without materializing the full update budget."""

    def __init__(self, weights: torch.Tensor, samples: int, seed: int) -> None:
        self.weights, self.samples = weights, samples
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[int]:
        for start in range(0, self.samples, _SAMPLER_CHUNK):
            yield from torch.multinomial(
                self.weights, min(_SAMPLER_CHUNK, self.samples - start),
                True, generator=self.generator,
            ).tolist()


def _stock_uniform_weights(members: Sequence[DataSplits]) -> torch.Tensor:
    """Give every stock one unit of total sampling weight."""
    lengths = tuple(len(member.train) for member in members)
    if not lengths or min(lengths) < 1:
        raise ValueError("stock-uniform training requires nonempty series")
    return torch.cat(tuple(
        torch.full((length,), 1.0 / length, dtype=torch.double)
        for length in lengths
    ))


def _stock_uniform_loader(
    members: Sequence[DataSplits], batch_size: int, samples: int, seed: int,
    *, drop_last: bool = False,
) -> DataLoader:
    _integer(batch_size, "batch_size")
    _integer(samples, "samples")
    datasets = tuple(member.train for member in members)
    lengths = tuple(map(len, datasets))
    generator = torch.Generator().manual_seed(seed)
    if lengths and len(set(lengths)) == 1 and \
       samples == sum(lengths) and not drop_last:
        return DataLoader(
            ConcatDataset(datasets), batch_size, shuffle=True,
            generator=generator,
        )
    sampler = _ChunkedWeightedSampler(
        _stock_uniform_weights(members), samples, seed,
    )
    return DataLoader(
        ConcatDataset(datasets),
        batch_size, sampler=sampler, drop_last=drop_last,
    )


def _macro_validation_loss(
    model: nn.Module, members: Sequence[DataSplits], batch_size: int,
    device: torch.device,
) -> float:
    """Average complete per-stock validation means without row weighting."""
    datasets = tuple(member.validation for member in members)
    if not datasets or any(not len(dataset) for dataset in datasets):
        raise ValueError("macro validation requires every bound stock")
    return fmean(
        mean_loss(model, DataLoader(dataset, batch_size), device)
        for dataset in datasets
    )


def _validation_members(
    members: Sequence[DataSplits], indices: Sequence[int],
) -> tuple[DataSplits, ...]:
    selected = tuple(indices)
    if not selected or len(set(selected)) != len(selected) or any(
        type(index) is not int or not 0 <= index < len(members)
        for index in selected
    ):
        raise ValueError("shared validation indices are invalid")
    result = tuple(members[index] for index in selected)
    if any(not len(member.validation) for member in result):
        raise ValueError("shared validation requires nonempty series")
    return result


def _fit_shared_updates(
    model_name: str, candidate: Candidate, members: Sequence[TrainingData],
    sweep: Sweep, seed: int, updates_per_checkpoint: int,
    device: torch.device, *, validation_indices: Sequence[int],
) -> tuple[nn.Module, UpdateFit]:
    """Fit one shared model with stock-macro sampling and a fixed budget."""
    if model_name not in SHARED_NEURAL:
        raise ValueError("stock-macro fitting requires a shared model")
    conditioned = model_name == "conditioned_panel_transformer"
    splits = _panel_members(members, conditioned)
    validation = _validation_members(splits, validation_indices)
    torch.manual_seed(seed)
    model = _neural_model(model_name, candidate, len(splits)).to(device)
    loader = _stock_uniform_loader(
        splits, sweep.batch_size,
        sweep.batch_size * sweep.epochs * updates_per_checkpoint, seed,
        drop_last=True,
    )
    fit = fit_updates(
        model, loader,
        lambda: _macro_validation_loss(
            model, validation, sweep.batch_size, device,
        ),
        sweep.epochs, updates_per_checkpoint, candidate.learning_rate,
        candidate.weight_decay, device,
    )
    return model, fit


def _fit_shared_epochs(
    model_name: str, candidate: Candidate, members: Sequence[TrainingData],
    sweep: Sweep, seed: int, device: torch.device, *,
    validation_indices: Sequence[int],
) -> tuple[nn.Module, Fit, tuple[DataLoader, ...]]:
    """Fit the secondary fixed-epoch curve with the same stock-macro objective."""
    if model_name not in SHARED_NEURAL:
        raise ValueError("stock-macro fitting requires a shared model")
    splits = _panel_members(
        members, model_name == "conditioned_panel_transformer",
    )
    validation = _validation_members(splits, validation_indices)
    pooled = DataSplits(*(
        ConcatDataset([getattr(member, name) for member in splits])
        for name in ("train", "validation", "test")
    ))
    torch.manual_seed(seed)
    model = _neural_model(model_name, candidate, len(splits)).to(device)
    loader = _stock_uniform_loader(
        splits, sweep.batch_size, sum(len(member.train) for member in splits),
        seed,
    )
    fit, loaders = fit_model(
        model, pooled, sweep.batch_size, sweep.epochs, sweep.patience,
        candidate.learning_rate, candidate.weight_decay, seed, device,
        train_loader=loader,
        validation_loss=lambda: _macro_validation_loss(
            model, validation, sweep.batch_size, device,
        ),
    )
    return model, fit, loaders


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


def expected_runs(sweep: Sweep, series_count: int) -> int:
    selection = _model_runs(sweep, sweep.models) * sweep.folds * \
        len(sweep.candidates)
    calibration = _model_runs(sweep, sweep.models)
    return series_count * (selection + calibration)


def _calibration_contract(
    sweep: Sweep, series: Sequence[Mapping[str, object]],
    test_contract: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    validation: Sequence[Mapping[str, object]],
    calibration: Sequence[Mapping[str, object]],
    fingerprints: Sequence[Mapping[str, object]],
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
        "calibration": list(calibration),
        "model_fingerprints": list(fingerprints),
    }


def _authorize_test(contract: Mapping[str, object],
                    policies: Sequence[Mapping[str, object]],
                    ) -> dict[str, Mapping[str, object]]:
    if PANEL_MODEL_SET.intersection(contract["sweep"]["models"]):
        raise ValueError("panel models are calibration-only")
    if not policies:
        raise ValueError("test evaluation requires a frozen policy")
    fingerprint = experiment_fingerprint(contract)
    sweep = contract["sweep"]
    configurations = {item["name"]: item for item in sweep["candidates"]}
    names = sorted(item["name"] for item in contract["series"])
    models = tuple(policy["model"] for policy in policies)
    if len(models) != len(set(models)):
        raise ValueError("test policies must name unique models")
    authorized = {}
    for policy in policies:
        model = policy["model"]
        try:
            candidate = contract["selection"][model]["candidate"]
            configuration = configurations[candidate]
        except (KeyError, TypeError) as error:
            raise ValueError("test policy model is not in the experiment") from error
        expected_seeds = sorted(sweep["seeds"]) if model in NEURAL else []
        expected_fingerprints = [
            item for item in contract["model_fingerprints"]
            if item["model"] == model
        ]
        if policy["model_fingerprints"] != expected_fingerprints:
            raise ValueError(
                "test policy does not match reconstructed model states"
            )
        if policy["calibration_fingerprint"] != fingerprint or \
           policy["candidate"] != candidate or \
           policy["feature_set"] != configuration["feature_set"] or \
           policy["target_kind"] != sweep["target_kind"] or \
           policy["horizon_bars"] != sweep["target_horizon_bars"] or \
           policy["seeds"] != expected_seeds or policy["series"] != names or \
           policy["test_grid"] != contract["test_contract"]:
            raise ValueError("test policy does not match the calibration contract")
        authorized[model] = policy
    return authorized


def _verify_test_state(model: nn.Module, data: TrainingData,
                       candidate: Candidate,
                       fingerprint: Mapping[str, object],
                       policy: Mapping[str, object], series: str,
                       seed: int | None) -> None:
    matches = [
        item["sha256"] for item in policy["model_fingerprints"]
        if item["series"] == series and item["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError("frozen policy must contain exactly one test model state")
    expected = matches[0]
    actual = _model_fingerprint(model, data, candidate)
    if actual != fingerprint["sha256"] or actual != expected:
        raise ValueError("test model state does not match its frozen policy")


@dataclass(frozen=True)
class FinalModel:
    candidate: Candidate
    data: TrainingData
    model: nn.Module
    loaders: tuple[object, ...]
    fingerprint: Mapping[str, object]
    epochs: int | None
    boundary: dict[str, list[str]]
    timestamps: Sequence[str]
    csv_sha256: str


def run_experiment(
    sweep: Sweep, series: Sequence[tuple[str, Path | FrozenInput]],
    device: torch.device, max_runs: int,
    prediction_records: list[dict[str, object]] | None = None,
    calibration_prediction_records: list[dict[str, object]] | None = None,
    evaluate_test: bool = True,
    test_authorizer: Callable[
        [Mapping[str, object]], Mapping[str, Mapping[str, object]]
    ] | None = None,
    *, requested_models: frozenset[str],
    panel_execution: PanelExecution | None = None,
) -> dict[str, object]:
    panel = bool(PANEL_MODEL_SET.intersection(sweep.models))
    if panel and (evaluate_test or
                  not isinstance(panel_execution, PanelExecution)):
        raise ValueError("panel models require a bound calibration-only execution")
    if not panel and panel_execution is not None:
        raise ValueError("panel execution requires a panel model")
    if panel_execution is not None and \
       Sweep.read(panel_execution.config_input.snapshot) != sweep:
        raise ValueError("panel sweep does not match the bound config")
    inputs = tuple(item for _, item in series)
    if panel and any(isinstance(item, Path) for item in inputs):
        raise ValueError("panel series must share the bound frozen execution")
    if all(isinstance(item, Path) for item in inputs):
        with freeze_inputs(inputs) as frozen:
            report = _run_experiment(
                sweep, tuple(
                    (name, item)
                    for (name, _), item in zip(series, frozen, strict=True)
                ),
                device, max_runs, prediction_records,
                calibration_prediction_records, evaluate_test,
                test_authorizer, requested_models, panel_execution,
            )
            verify_frozen(frozen)
            return report
    if not all(isinstance(item, FrozenInput) for item in inputs):
        raise TypeError("series inputs must be paths or frozen inputs")
    report = _run_experiment(
        sweep, series, device, max_runs, prediction_records,
        calibration_prediction_records, evaluate_test, test_authorizer,
        requested_models, panel_execution,
    )
    verify_frozen(inputs)
    return report


def _run_experiment(
    sweep: Sweep, series: Sequence[tuple[str, FrozenInput]],
    device: torch.device, max_runs: int,
    prediction_records: list[dict[str, object]] | None,
    calibration_prediction_records: list[dict[str, object]] | None,
    evaluate_test: bool,
    test_authorizer: Callable[
        [Mapping[str, object]], Mapping[str, Mapping[str, object]]
    ] | None,
    requested_models: frozenset[str],
    panel_execution: PanelExecution | None,
) -> dict[str, object]:
    panel_models = tuple(model for model in PANEL_MODELS if model in sweep.models)
    panel = bool(panel_models)
    if panel and (evaluate_test or
                  not isinstance(panel_execution, PanelExecution)):
        raise ValueError("panel models require a bound calibration-only execution")
    if not panel and panel_execution is not None:
        raise ValueError("panel execution requires a panel model")
    if panel_execution is not None and \
       Sweep.read(panel_execution.config_input.snapshot) != sweep:
        raise ValueError("panel sweep does not match the bound config")
    if panel_execution is not None:
        panel_execution.validate()
        if tuple(series) != panel_execution.series:
            raise ValueError("panel series do not match the bound execution")
    if not series or len({name for name, _ in series}) != len(series):
        raise ValueError("series names must be nonempty and unique")
    if not isinstance(requested_models, frozenset) or \
       any(model not in sweep.models for model in requested_models) or \
       evaluate_test != bool(requested_models):
        raise ValueError("requested models do not match the experiment mode")
    if not evaluate_test and prediction_records is not None:
        raise ValueError(
            "calibration-only experiments cannot collect test predictions"
        )
    if evaluate_test and test_authorizer is None:
        raise ValueError("test evaluation requires explicit authorization")
    runs = expected_runs(sweep, len(series))
    if runs > max_runs:
        raise ValueError(f"experiment requires {runs} runs; "
                         f"--max-runs is {max_runs}")
    if panel_execution is not None:
        physical = len(panel_models) * len(sweep.seeds) * (
            len(sweep.candidates) * sweep.folds + 1
        )
        if panel_execution.attempt.expected_equivalent_runs != runs or \
           panel_execution.attempt.expected_panel_fits != physical:
            raise ValueError("panel run accounting does not match the attempt")
    torch.use_deterministic_algorithms(True)
    max_history = max(candidate.seq_len + feature_lookback(candidate.feature_set)
                      for candidate in sweep.candidates)
    horizon = sweep.target_horizon_bars
    alignment = sweep.alignment_horizon_bars
    target_offset = max_history + alignment - 1
    gap = alignment - 1
    metadata, test_contract, folds_by_series = [], [], {}
    for name, frozen in series:
        timestamps, rows = read_bars(frozen.snapshot)
        row_count = len(timestamps)
        samples = row_count - target_offset
        splits = tuple(
            purged_split(split, gap, preserve_last=False)
            for split in walk_forward_splits(
                samples, sweep.folds, sweep.fold_fraction, reserved_blocks=2,
            )
        )
        holdout = purged_split(holdout_split(samples, sweep.fold_fraction), gap)
        folds_by_series[name] = (
            frozen, timestamps, rows, splits, holdout,
        )
        test_boundary = _boundary(
            timestamps, target_offset, holdout, gap,
        )["test"]
        test_contract.append({
            "series": name, "samples": holdout[2],
            "first_target_time": test_boundary[0],
            "last_target_time": test_boundary[1],
        })
        metadata.append({"name": name, "csv": str(frozen.source),
                         "rows": row_count, "sha256": frozen.sha256,
                         "first_timestamp": timestamps[0],
                         "last_timestamp": timestamps[-1]})

    if panel:
        if panel_execution is None:
            raise ValueError("panel execution is missing")
        panel_execution.inputs.validate_timestamps(tuple(
            (name, frozen, timestamps)
            for name, (frozen, timestamps, _, _, _) in folds_by_series.items()
        ))
        grids = [timestamps for _, timestamps, _, _, _
                 in folds_by_series.values()]
        if any(timestamps != grids[0] for timestamps in grids[1:]):
            raise ValueError("panel series must have identical timestamp grids")
        for _, timestamps, _, splits, holdout in folds_by_series.values():
            for split in (*splits, holdout):
                _label_available(
                    timestamps, target_offset, split, gap, horizon,
                )

    validation: list[dict[str, object]] = []
    panel_fold_data: dict[
        tuple[str, str, int], tuple[TrainingData, dict[str, list[str]]]
    ] = {}
    for candidate in sweep.candidates:
        for name, (_, timestamps, rows, splits, _) in folds_by_series.items():
            for fold, (train_count, validation_count) in enumerate(splits):
                data = _candidate_data(
                    rows, candidate, (train_count, validation_count, 1),
                    max_history, sweep,
                )
                boundary = _boundary(
                    timestamps, target_offset, (train_count, validation_count), gap,
                )
                if panel:
                    panel_fold_data[(candidate.name, name, fold)] = data, boundary
                for model_name in sweep.models:
                    if model_name in PANEL_MODEL_SET:
                        continue
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
                        metrics = evaluate(
                            model, loader, data.target_mean, data.target_scale,
                            device,
                        )
                        validation.append(_validation_record(
                            model_name, candidate, name, fold, boundary, seed,
                            len(data.validation), mean_loss(model, loader, device),
                            metrics, fit,
                        ))

    names = tuple(folds_by_series)
    if panel:
        panel_validation = []
        for model_name in panel_models:
            conditioned = model_name == "conditioned_panel_transformer"
            for candidate in sweep.candidates:
                for fold in range(sweep.folds):
                    members = [
                        panel_fold_data[(candidate.name, name, fold)][0]
                        for name in names
                    ]
                    pooled = _panel_data(members, conditioned)
                    for seed in sweep.seeds:
                        model, fit, _ = _fit_neural(
                            model_name, candidate, pooled, sweep, seed, device,
                            len(names) if conditioned else 0,
                        )
                        for series_id, (name, data) in enumerate(zip(
                            names, members, strict=True,
                        )):
                            evaluation = (
                                _conditioned(data, series_id)
                                if conditioned else data
                            )
                            loader = data_loaders(
                                evaluation, sweep.batch_size, seed,
                            )[1]
                            metrics = evaluate(
                                model, loader, data.target_mean,
                                data.target_scale, device,
                            )
                            panel_validation.append(_validation_record(
                                model_name, candidate, name, fold,
                                panel_fold_data[
                                    (candidate.name, name, fold)
                                ][1],
                                seed, len(data.validation),
                                mean_loss(model, loader, device), metrics, fit,
                            ))
        model_order = {
            model: index for index, model in enumerate(panel_models)
        }
        candidate_order = {
            candidate.name: index
            for index, candidate in enumerate(sweep.candidates)
        }
        series_order = {name: index for index, name in enumerate(names)}
        seed_order = {seed: index for index, seed in enumerate(sweep.seeds)}
        panel_validation.sort(key=lambda record: (
            model_order[str(record["model"])],
            candidate_order[str(record["candidate"])],
            series_order[str(record["series"])], int(record["fold"]),
            seed_order[int(record["seed"])],
        ))
        validation.extend(panel_validation)

    selection = select_candidates(validation, sweep.models, sweep.candidates)
    candidates = {candidate.name: candidate for candidate in sweep.candidates}
    calibration: list[dict[str, object]] = []
    fingerprints: list[dict[str, object]] = []
    retained: dict[tuple[str, str, int | None], FinalModel] = {}
    final_data: dict[tuple[str, str], TrainingData] = {}
    for model_name in sweep.models:
        if model_name in PANEL_MODEL_SET:
            continue
        candidate = candidates[str(selection[model_name]["candidate"])]
        for name, (frozen, timestamps, rows, _, split) in \
                folds_by_series.items():
            data_key = (name, candidate.name)
            data = final_data.get(data_key)
            if data is None:
                data = _candidate_data(
                    rows, candidate, split, max_history, sweep,
                )
                final_data[data_key] = data
            boundary = _boundary(timestamps, target_offset, split, gap)
            seeds = sweep.seeds if model_name in NEURAL else (None,)
            for seed in seeds:
                if seed is None:
                    epochs = None
                    model = _deterministic(model_name, candidate, data).to(device)
                    loaders = data_loaders(data, sweep.batch_size, 0)
                else:
                    epochs = _selected_epochs(
                        validation, model_name, candidate.name, name, seed,
                    )
                    torch.manual_seed(seed)
                    model = _neural_model(model_name, candidate).to(device)
                    loaders = fit_epochs(
                        model, data, sweep.batch_size, epochs,
                        candidate.learning_rate, candidate.weight_decay,
                        seed, device,
                    )
                digest = _model_fingerprint(model, data, candidate)
                fingerprint = {
                    "model": model_name, "series": name, "seed": seed,
                    "epochs": epochs, "sha256": digest,
                }
                predictions = [] if calibration_prediction_records is not None \
                    else None
                metrics = evaluate(
                    model, loaders[1], data.target_mean, data.target_scale,
                    device, predictions,
                )
                record = _validation_record(
                    model_name, candidate, name, None, boundary, seed,
                    len(data.validation), mean_loss(model, loaders[1], device),
                    metrics, None,
                )
                record["epochs"] = epochs
                calibration.append(record)
                fingerprints.append(fingerprint)
                if model_name in requested_models:
                    retained[(model_name, name, seed)] = FinalModel(
                        candidate, data, model, loaders, fingerprint, epochs,
                        boundary, timestamps, frozen.sha256,
                    )
                if calibration_prediction_records is not None and \
                   predictions is not None:
                    calibration_prediction_records.extend(_prediction_records(
                        model_name, candidate, name, seed, data, timestamps,
                        frozen.sha256, predictions, data.validation,
                        "calibration", None,
                    ))
    if panel:
        panel_candidates = {
            model_name: candidates[
                str(selection[model_name]["candidate"])
            ]
            for model_name in panel_models
        }
        selected_epochs = {
            (model_name, seed): _panel_selected_epochs(
                validation, model_name, panel_candidates[model_name].name,
                names, seed, sweep.folds,
            )
            for model_name in panel_models
            for seed in sweep.seeds
        }
        panel_calibration = []
        panel_predictions = []
        for model_name in panel_models:
            candidate = panel_candidates[model_name]
            conditioned = model_name == "conditioned_panel_transformer"
            members = []
            for name, (_, _, rows, _, split) in folds_by_series.items():
                data_key = (name, candidate.name)
                data = final_data.get(data_key)
                if data is None:
                    data = _candidate_data(
                        rows, candidate, split, max_history, sweep,
                    )
                    final_data[data_key] = data
                members.append(data)
            pooled = _panel_data(members, conditioned)
            for seed in sweep.seeds:
                epochs = selected_epochs[(model_name, seed)]
                torch.manual_seed(seed)
                model = _neural_model(
                    model_name, candidate,
                    len(names) if conditioned else 0,
                ).to(device)
                fit_epochs(
                    model, pooled, sweep.batch_size, epochs,
                    candidate.learning_rate, candidate.weight_decay,
                    seed, device,
                )
                for series_id, (name, data) in enumerate(zip(
                    names, members, strict=True,
                )):
                    frozen, timestamps, _, _, split = folds_by_series[name]
                    evaluation = (
                        _conditioned(data, series_id)
                        if conditioned else data
                    )
                    loader = data_loaders(
                        evaluation, sweep.batch_size, seed,
                    )[1]
                    digest = _model_fingerprint(model, data, candidate)
                    fingerprints.append({
                        "model": model_name, "series": name,
                        "seed": seed, "epochs": epochs, "sha256": digest,
                    })
                    predictions = (
                        [] if calibration_prediction_records is not None
                        else None
                    )
                    metrics = evaluate(
                        model, loader, data.target_mean, data.target_scale,
                        device, predictions,
                    )
                    record = _validation_record(
                        model_name, candidate, name, None,
                        _boundary(timestamps, target_offset, split, gap), seed,
                        len(data.validation), mean_loss(model, loader, device),
                        metrics, None,
                    )
                    record["epochs"] = epochs
                    panel_calibration.append(record)
                    if predictions is not None:
                        panel_predictions.extend(_prediction_records(
                            model_name, candidate, name, seed, data,
                            timestamps, frozen.sha256, predictions,
                            data.validation, "calibration", None,
                        ))
        panel_calibration.sort(key=lambda record: (
            model_order[str(record["model"])],
            series_order[str(record["series"])],
            seed_order[int(record["seed"])],
        ))
        calibration.extend(panel_calibration)
        if calibration_prediction_records is not None:
            panel_predictions.sort(key=lambda record: (
                model_order[str(record["model"])],
                series_order[str(record["series"])],
                seed_order[int(record["seed"])], str(record["target_time"]),
            ))
            calibration_prediction_records.extend(panel_predictions)
    if "conditioned_panel_transformer" not in panel_models:
        fingerprints.sort(key=lambda item: (
            item["model"], item["series"],
            -1 if item["seed"] is None else item["seed"],
        ))
    else:
        local_fingerprints = [
            item for item in fingerprints
            if item["model"] not in PANEL_MODEL_SET
        ]
        local_fingerprints.sort(key=lambda item: (
            item["model"], item["series"],
            -1 if item["seed"] is None else item["seed"],
        ))
        panel_fingerprints = [
            item for item in fingerprints
            if item["model"] in PANEL_MODEL_SET
        ]
        panel_fingerprints.sort(key=lambda item: (
            model_order[str(item["model"])],
            series_order[str(item["series"])],
            seed_order[int(item["seed"])],
        ))
        fingerprints = [*local_fingerprints, *panel_fingerprints]
    contract = _calibration_contract(
        sweep, metadata, test_contract, selection, validation, calibration,
        fingerprints,
    )
    test_policies: Mapping[str, Mapping[str, object]] = {}
    if evaluate_test and test_authorizer is not None:
        authorization = test_authorizer(contract)
        if not isinstance(authorization, Mapping) or not authorization or \
           any(not isinstance(policy, Mapping)
               for policy in authorization.values()) or \
           set(authorization) != requested_models:
            raise ValueError("test authorization returned invalid models")
        test_policies = authorization
    test: list[dict[str, object]] = []
    if evaluate_test:
        for model_name, policy in test_policies.items():
            for name, _ in series:
                seeds = sweep.seeds if model_name in NEURAL else (None,)
                for seed in seeds:
                    final = retained[(model_name, name, seed)]
                    _verify_test_state(
                        final.model, final.data, final.candidate,
                        final.fingerprint, policy, name, seed,
                    )
                    predictions = [] if prediction_records is not None else None
                    record = {
                        "model": model_name,
                        "candidate": final.candidate.name,
                        "feature_set": final.candidate.feature_set,
                        "series": name, "fold": "holdout", "seed": seed,
                        "targets": final.boundary,
                        "samples": len(final.data.test),
                        "epochs": final.epochs,
                        "metrics": evaluate(
                            final.model, final.loaders[2],
                            final.data.target_mean, final.data.target_scale,
                            device, predictions,
                        ),
                    }
                    test.append(record)
                    if prediction_records is not None and predictions is not None:
                        prediction_records.extend(_prediction_records(
                            model_name, final.candidate, name, seed, final.data,
                            final.timestamps, final.csv_sha256, predictions,
                            final.data.test, "test", None,
                        ))

    report = {
        "schema": 6,
        "protocol": {
            "split": "embargoed expanding walk-forward by target time",
            "selection": "minimum mean validation scaled-return MSE",
            "selection_aggregation": "macro mean over series, folds, and seeds",
            "holdout_aggregation": "macro mean over series and seeds",
            "phase": (
                "selection-calibration-and-test" if evaluate_test
                else "selection-and-calibration"
            ),
            "calibration_policy": (
                "evaluate test only after exact model and policy reproduction"
                if evaluate_test else "deferred until policy selection"
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
    if "conditioned_panel_transformer" in panel_models:
        report["protocol"]["panel_conditioning"] = {
            "model": "conditioned_panel_transformer",
            "kind": "learned-series-embedding",
            "series_order": list(names),
            "initialization": "zeros",
            "application": "additive-before-encoder",
        }
    if panel_execution is not None:
        report.update(panel_execution.provenance())
        panel_execution.validate()
    return report


def write_report(
    path: Path, report: Mapping[str, object], *, exclusive: bool = False,
    directory_fd: int | None = None,
) -> None:
    if exclusive:
        write_json_exclusive(path, report, directory_fd)
    else:
        write_json(path, report)


def write_predictions(path: Path,
                      records: Sequence[Mapping[str, object]], *,
                      exclusive: bool = False,
                      directory_fd: int | None = None) -> None:
    def write(file: TextIO) -> None:
        for record in records:
            json.dump(record, file, allow_nan=False, sort_keys=True)
            file.write("\n")

    if exclusive:
        exclusive_text(path, write, directory_fd)
    else:
        atomic_text(path, write)


def _series(value: str) -> tuple[str, Path]:
    return series_arg(value, SERIES_NAME)


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
    parser.add_argument("--calibration-predictions", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--policy", dest="policies", action="append", type=Path)
    parser.add_argument("--attempt-manifest", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-ledger", type=Path)
    return parser.parse_args(argv)


def _execute(
    args: argparse.Namespace, sweep_input: FrozenInput,
    policy_inputs: Sequence[FrozenInput],
    series_inputs: Sequence[FrozenInput],
    panel_execution: PanelExecution | None,
) -> dict[str, object]:
    frozen = (sweep_input, *policy_inputs, *series_inputs)
    output_directory_fd = (
        panel_execution.run_directory_fd
        if panel_execution is not None else None
    )
    if panel_execution is not None:
        panel_execution.validate_outputs(
            args.report, args.calibration_predictions,
        )

    def verify() -> None:
        verify_frozen(frozen)
        if panel_execution is not None:
            panel_execution.verify()

    sweep = Sweep.read(sweep_input.snapshot)
    if not args.calibration_only and PANEL_MODEL_SET.intersection(sweep.models):
        raise ValueError("panel models are calibration-only")
    if args.horizon_bars is not None:
        sweep = replace(sweep, target_horizon_bars=args.horizon_bars)
    if args.target_kind is not None:
        sweep = replace(sweep, target_kind=args.target_kind)
    policies = tuple(read_policy(item.snapshot) for item in policy_inputs)
    requested_models = frozenset(policy["model"] for policy in policies)
    if len(requested_models) != len(policies):
        raise ValueError("test policies must name unique models")
    predictions = [] if args.predictions else None
    calibration_predictions = [] if args.calibration_predictions else None
    report = run_experiment(
        sweep, tuple(
            (name, item)
            for (name, _), item in zip(
                args.series, series_inputs, strict=True,
            )
        ),
        torch.device(args.device), args.max_runs, predictions,
        calibration_predictions, not args.calibration_only,
        (lambda contract: _authorize_test(contract, policies))
        if policies else None,
        requested_models=requested_models,
        panel_execution=panel_execution,
    )
    report["sweep_input"] = {
        "path": str(sweep_input.source), "sha256": sweep_input.sha256,
    }
    if policies:
        report["policies"] = [
            {
                "path": str(item.source), "sha256": item.sha256,
                "model": policy["model"],
            }
            for item, policy in zip(policy_inputs, policies, strict=True)
        ]
    if args.predictions and predictions is not None:
        verify()
        write_predictions(args.predictions, predictions)
        report["prediction_ledger"] = {
            "schema": 3, "path": str(args.predictions),
            "records": len(predictions),
            "sha256": file_sha256(args.predictions),
        }
    if args.calibration_predictions and \
       calibration_predictions is not None:
        verify()
        if panel_execution is not None:
            panel_execution.prepare_output(
                "calibration_ledger", args.calibration_predictions,
            )
        write_predictions(
            args.calibration_predictions, calibration_predictions,
            exclusive=panel_execution is not None,
            directory_fd=output_directory_fd,
        )
        report["calibration_prediction_ledger"] = {
            "schema": 3, "path": str(args.calibration_predictions),
            "records": len(calibration_predictions),
            "sha256": file_sha256(args.calibration_predictions),
        }
    verify()
    if panel_execution is not None:
        panel_execution.prepare_output("experiment_report", args.report)
    write_report(
        args.report, report, exclusive=panel_execution is not None,
        directory_fd=output_directory_fd,
    )
    return report


def main() -> None:
    args = parse_args()
    try:
        policy_paths = tuple(args.policies or ())
        panel_paths = (
            args.attempt_manifest, args.input_manifest,
            args.baseline_report, args.baseline_ledger,
        )
        outputs = [path for path in (
            args.report, args.predictions, args.calibration_predictions,
        ) if path]
        require_disjoint(
            [
                args.sweep, *policy_paths,
                *(path for path in panel_paths if path),
                *(path for _, path in args.series),
            ],
            outputs,
        )
        device = torch.device(args.device)
        if device.type == "meta" or args.max_runs <= 0:
            raise ValueError(
                "device must execute tensors and max-runs must be positive"
            )
        if args.calibration_only and args.predictions:
            raise ValueError("calibration-only mode cannot export test predictions")
        if (args.calibration_only and policy_paths) or \
           (not args.calibration_only and not policy_paths) or \
           len(policy_paths) != len(set(policy_paths)):
            raise ValueError("full experiments require unique frozen policies")
        torch.empty(0, device=device)
        discovered = Sweep.read(args.sweep)
        panel = bool(PANEL_MODEL_SET.intersection(discovered.models))
        if panel and not args.calibration_only:
            raise ValueError("panel models are calibration-only")
        if panel != all(panel_paths):
            raise ValueError(
                "panel models require attempt, input, and baseline manifests"
            )
        if not panel and any(panel_paths):
            raise ValueError("panel manifests require a panel model")
        if panel:
            with freeze_panel_execution(
                args.attempt_manifest, args.input_manifest, args.sweep,
                args.baseline_report, args.baseline_ledger, args.series,
                ROOT, _torch_identity, sys.argv,
            ) as execution:
                report = _execute(
                    args, execution.config_input, (), tuple(
                        item for _, item in execution.series
                    ), execution,
                )
        else:
            sources = (
                args.sweep, *policy_paths, *(path for _, path in args.series),
            )
            with freeze_inputs(sources) as frozen:
                report = _execute(
                    args, frozen[0], frozen[1:1 + len(policy_paths)],
                    frozen[1 + len(policy_paths):], None,
                )
                verify_frozen(frozen)
    except (FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    result = {"report": str(args.report), "selection": report["selection"]}
    if args.predictions:
        result["predictions"] = str(args.predictions)
    if args.calibration_predictions:
        result["calibration_predictions"] = str(args.calibration_predictions)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
