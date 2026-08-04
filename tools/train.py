#!/usr/bin/env python3
"""Train the exact compose-mini architecture and export a V1 artifact."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import json
import locale
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as error:
    raise SystemExit("training requires PyTorch: python -m pip install torch") from error

from tools.artifact_v1 import (
    Artifact, Config, WEIGHT_CHUNK, WEIGHT_FIELDS, validate_identifiers, write_artifact,
)
from tools.data_v1 import (
    CLOSE_RETURN_TARGET, FEATURE_COUNT, TARGET_KINDS, read_csv,
)
from tools.float32 import f32
from tools.session_samples import SampleRows

# tier-a-v1 rolling context window (bars). 12 30-min bars ~= one session.
TIER_A_WINDOW = 12

FEATURE_NAMES = {
    "ohlcv": ("open", "high", "low", "close", "volume"),
    "stationary-v1": (
        "log_gap", "log_body", "log_upper_wick", "log_lower_wick",
        "log_volume_change",
    ),
    # ponytail: Tier A = stationary-v1 + rolling context, no new data source.
    "tier-a-v1": (
        "log_gap", "log_body", "log_upper_wick", "log_lower_wick",
        "log_volume_change", "realized_vol", "volume_zscore", "range_pct",
    ),
}
FEATURE_SETS = tuple(FEATURE_NAMES)

_LOOKBACK = {"ohlcv": 0, "stationary-v1": 1, "tier-a-v1": TIER_A_WINDOW}


def feature_lookback(feature_set: str) -> int:
    """Return raw bars required before the first model feature row."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unsupported feature set: {feature_set}")
    return _LOOKBACK[feature_set]


def _stationary(raw: torch.Tensor) -> torch.Tensor:
    """Five per-bar log features aligned to each completed bar after the first."""
    open_, high, low, close, volume = raw.unbind(1)
    if not torch.all(raw[:, :4] > 0) or not torch.all(volume >= 0) or \
       not torch.all(low <= torch.minimum(open_, close)) or \
       not torch.all(torch.maximum(open_, close) <= high):
        raise ValueError("stationary features require valid positive OHLC and volume")
    current_open, current_close = open_[1:], close[1:]
    return torch.stack((
        torch.log(current_open / close[:-1]),
        torch.log(current_close / current_open),
        torch.log(high[1:] / torch.maximum(current_open, current_close)),
        torch.log(torch.minimum(current_open, current_close) / low[1:]),
        torch.log1p(volume[1:]) - torch.log1p(volume[:-1]),
    ), 1)


def feature_values(raw: torch.Tensor, feature_set: str) -> torch.Tensor:
    """Return features aligned to each completed source bar past the lookback."""
    if feature_set == "ohlcv":
        return raw
    if feature_set == "stationary-v1":
        return _stationary(raw)
    # tier-a-v1: rolling context from OHLCV only. Every window ends on the
    # completed bar it describes, so no target-period bar leaks into a feature.
    stat = _stationary(raw)                       # rows: len(raw) - 1
    w = TIER_A_WINDOW
    if stat.shape[0] < w:
        raise ValueError("tier-a-v1 requires more bars than the context window")
    log_return = stat[:, 1]                        # log(close/open) per bar
    volume = raw[1:, 4]
    high, low, close = raw[1:, 1], raw[1:, 2], raw[1:, 3]
    # Rolling stats over the trailing w bars, indexed by the window's last bar.
    ret_win = log_return.unfold(0, w, 1)           # rows: len(stat) - w + 1
    vol_win = volume.unfold(0, w, 1)
    realized_vol = ret_win.std(dim=1, unbiased=True)
    vmean, vstd = vol_win.mean(dim=1), vol_win.std(dim=1, unbiased=True)
    volume_zscore = (volume[w - 1:] - vmean) / (vstd + 1e-8)
    range_pct = (high[w - 1:] - low[w - 1:]) / close[w - 1:]
    # Trim the per-bar features to the same tail the rolling stats produce.
    base = _stationary(raw)[w - 1:]
    return torch.cat((
        base,
        torch.stack((realized_vol, volume_zscore, range_pct), 1),
    ), 1)


def positional_encoding(seq_len: int, model_dim: int) -> torch.Tensor:
    values = [0.0] * (seq_len * model_dim)
    step = f32(math.pow(f32(10000.0), f32(f32(-2.0) / f32(model_dim))))
    for position in range(seq_len):
        frequency = 1.0
        for index in range(0, model_dim, 2):
            angle = f32(f32(position) * frequency)
            values[position * model_dim + index] = f32(math.sin(angle))
            if index + 1 < model_dim:
                values[position * model_dim + index + 1] = f32(math.cos(angle))
            frequency = f32(frequency * step)
    return torch.tensor(values, dtype=torch.float32).view(seq_len, model_dim)


class EncoderBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        d, f = config.model_dim, config.ff_dim
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads
        self.Wq = nn.Parameter(torch.empty(d, d))
        self.Wk = nn.Parameter(torch.empty(d, d))
        self.Wv = nn.Parameter(torch.empty(d, d))
        self.Wo = nn.Parameter(torch.empty(d, d))
        self.norm1_g = nn.Parameter(torch.ones(d))
        self.norm1_b = nn.Parameter(torch.zeros(d))
        self.W1 = nn.Parameter(torch.empty(d, f))
        self.b1 = nn.Parameter(torch.zeros(f))
        self.W2 = nn.Parameter(torch.empty(f, d))
        self.b2 = nn.Parameter(torch.zeros(d))
        self.norm2_g = nn.Parameter(torch.ones(d))
        self.norm2_b = nn.Parameter(torch.zeros(d))
        for weight in (self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2):
            nn.init.xavier_uniform_(weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            hidden, (hidden.shape[-1],), self.norm1_g, self.norm1_b, 1e-5,
        )
        batch, seq_len, model_dim = normalized.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        query = (normalized @ self.Wq).view(shape).transpose(1, 2)
        key = (normalized @ self.Wk).view(shape).transpose(1, 2)
        value = (normalized @ self.Wv).view(shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) * (self.head_dim ** -0.5)
        context = scores.softmax(-1) @ value
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, model_dim)
        hidden = hidden + context @ self.Wo
        normalized = F.layer_norm(
            hidden, (model_dim,), self.norm2_g, self.norm2_b, 1e-5,
        )
        return hidden + F.gelu(normalized @ self.W1 + self.b1,
                               approximate="none") @ self.W2 + self.b2


class ForecastTransformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed_W = nn.Parameter(torch.empty(config.in_dim, config.model_dim))
        self.layers = nn.ModuleList(EncoderBlock(config)
                                    for _ in range(config.num_layers))
        self.head_W = nn.Parameter(torch.empty(config.model_dim))
        self.head_b = nn.Parameter(torch.zeros(1))
        self.register_buffer("position", positional_encoding(
            config.seq_len, config.model_dim,
        ), persistent=False)
        nn.init.xavier_uniform_(self.embed_W)
        nn.init.normal_(self.head_W, std=config.model_dim ** -0.5)

    def forward(self, values: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = values @ self.embed_W + self.position
        if context is not None:
            hidden = hidden + context.unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden[:, -1] @ self.head_W + self.head_b


class Windows(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor,
                 references: torch.Tensor, outcomes: torch.Tensor,
                 seq_len: int, start: int, count: int, *,
                 feature_starts: Sequence[int] | None = None,
                 sample_rows: Sequence[SampleRows] | None = None) -> None:
        if (feature_starts is None) != (sample_rows is None):
            raise ValueError("indexed windows require starts and sample rows")
        self.features, self.targets = features, targets
        self.references, self.outcomes = references, outcomes
        self.seq_len, self.start, self.count = seq_len, start, count
        self.feature_starts = (
            tuple(feature_starts) if feature_starts is not None else None
        )
        self.sample_rows = tuple(sample_rows) if sample_rows is not None else None
        if self.indexed and (
            len(self.feature_starts) != len(self.sample_rows) or
            len(self.sample_rows) != len(targets) or
            not 0 <= start <= start + count <= len(self.sample_rows)
        ):
            raise ValueError("indexed window coordinates are invalid")

    @property
    def indexed(self) -> bool:
        return self.sample_rows is not None

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.start + index
        feature = (
            self.feature_starts[sample] if self.feature_starts is not None
            else sample
        )
        return (
            self.features[feature:feature + self.seq_len], self.targets[sample],
            self.references[sample], self.outcomes[sample],
        )


@dataclass(frozen=True)
class DataSplits:
    train: Dataset
    validation: Dataset
    test: Dataset


@dataclass(frozen=True)
class TrainingData(DataSplits):
    """Own one leakage-safe chronological split and its fitted scalers."""

    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    feature_set: str = "ohlcv"
    horizon_bars: int = 1
    target_kind: str = CLOSE_RETURN_TARGET


def tail_training_data(data: TrainingData, seq_len: int) -> TrainingData:
    """Shorten indexed histories without changing samples or fitted scalers."""
    if not isinstance(data, TrainingData) or type(seq_len) is not int:
        raise ValueError("tail views require indexed training data")
    splits = (data.train, data.validation, data.test)
    if any(not isinstance(split, Windows) or not split.indexed
           for split in splits):
        raise ValueError("tail views require indexed training data")
    source = splits[0]
    shared = ("features", "targets", "references", "outcomes")
    if not 1 <= seq_len <= source.seq_len or any(
        split.seq_len != source.seq_len or
        split.feature_starts != source.feature_starts or
        split.sample_rows != source.sample_rows or any(
            getattr(split, name) is not getattr(source, name)
            for name in shared
        )
        for split in splits
    ):
        raise ValueError("tail views require one common indexed preparation")
    if seq_len == source.seq_len:
        return data
    starts = tuple(
        start + source.seq_len - seq_len for start in source.feature_starts
    )

    def tail(split: Windows) -> Windows:
        return Windows(
            split.features, split.targets, split.references, split.outcomes,
            seq_len, split.start, split.count, feature_starts=starts,
            sample_rows=split.sample_rows,
        )

    return replace(data, **dict(zip(
        ("train", "validation", "test"), map(tail, splits), strict=True,
    )))


@dataclass(frozen=True)
class Fit:
    best_validation_scaled_mse: float
    best_epoch: int
    epochs_trained: int


@dataclass(frozen=True)
class UpdateFit:
    best_validation_scaled_mse: float
    best_checkpoint: int
    updates_trained: int


def _model_output(model: nn.Module,
                  values: torch.Tensor | Sequence[torch.Tensor],
                  device: torch.device) -> torch.Tensor:
    inputs = (values,) if isinstance(values, torch.Tensor) else tuple(values)
    return model(*(item.to(device) for item in inputs))


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def mean_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features, targets, *_ in loader:
            loss = F.mse_loss(_model_output(model, features, device),
                              targets.to(device), reduction="sum")
            total += loss.item()
            count += len(targets)
    if not count:
        raise ValueError("loss loader is empty")
    return total / count


def _train_batch(
    model: nn.Module, batch: Sequence[object],
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> tuple[float, int]:
    features, targets, *_ = batch
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(_model_output(model, features, device), targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("training produced a non-finite loss")
    loss.backward()
    optimizer.step()
    count = len(targets)
    return loss.item() * count, count


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        loss, samples = _train_batch(model, batch, optimizer, device)
        total, count = total + loss, count + samples
    if not count:
        raise ValueError("training loader is empty")
    return total / count


def _train_updates(
    model: nn.Module, batches: Iterator[Sequence[object]],
    optimizer: torch.optim.Optimizer, updates: int, device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for _ in range(updates):
        try:
            batch = next(batches)
        except StopIteration as error:
            raise ValueError(
                "training loader ended before its update budget"
            ) from error
        loss, samples = _train_batch(model, batch, optimizer, device)
        total, count = total + loss, count + samples
    if not count or not math.isfinite(total):
        raise FloatingPointError("training produced a non-finite loss")
    return total / count


def _require_exhausted(batches: Iterator[Sequence[object]]) -> None:
    try:
        next(batches)
    except StopIteration:
        return
    raise ValueError("training loader exceeds its update budget")


def evaluate(model: nn.Module, loader: DataLoader, target_mean: torch.Tensor,
             target_scale: torch.Tensor, device: torch.device,
             predictions: list[float] | None = None) -> dict[str, float]:
    totals = {"return_squared": 0.0, "return_absolute": 0.0,
              "close_absolute": 0.0, "baseline_close_absolute": 0.0,
              "direction_correct": 0.0}
    model.eval()
    with torch.no_grad():
        for features, targets, reference, actual in loader:
            predicted_return = _model_output(model, features, device).cpu() * \
                target_scale + target_mean
            actual_return = targets * target_scale + target_mean
            if predictions is not None:
                predictions.extend(predicted_return.tolist())
            difference = predicted_return - actual_return
            predicted_close = reference * predicted_return.exp()
            totals["return_squared"] += difference.square().sum().item()
            totals["return_absolute"] += difference.abs().sum().item()
            totals["close_absolute"] += (predicted_close - actual).abs().sum().item()
            totals["baseline_close_absolute"] += (reference - actual).abs().sum().item()
            totals["direction_correct"] += (
                predicted_return.sign() == actual_return.sign()
            ).sum().item()
    count = len(loader.dataset)
    return {
        "return_mse": totals["return_squared"] / count,
        "return_mae": totals["return_absolute"] / count,
        "close_mae": totals["close_absolute"] / count,
        "zero_return_baseline_mae": totals["baseline_close_absolute"] / count,
        "direction_accuracy": totals["direction_correct"] / count,
    }


def _tensor_values(tensors: tuple[torch.Tensor, ...]) -> Iterator[float]:
    """Yield bounded lists instead of materializing a full tensor in Python."""
    for tensor in tensors:
        values = tensor.detach().cpu().reshape(-1)
        for start in range(0, values.numel(), WEIGHT_CHUNK):
            yield from values[start:start + WEIGHT_CHUNK].tolist()


def export_weights(model: ForecastTransformer) -> dict[str, Iterable[float]]:
    tensors = {"embed_W": (model.embed_W,), "head_W": (model.head_W,),
               "head_b": (model.head_b,)}
    for field in WEIGHT_FIELDS:
        if field not in tensors:
            tensors[field] = tuple(getattr(layer, field) for layer in model.layers)
    return {
        field: _tensor_values(tensors[field])
        for field in WEIGHT_FIELDS
    }


def split_counts(samples: int, train_fraction: float,
                 validation_fraction: float) -> tuple[int, int, int]:
    train = int(samples * train_fraction)
    validation = int(samples * validation_fraction)
    test = samples - train - validation
    if min(train, validation, test) <= 0:
        raise ValueError("chronological train, validation, and test splits must be nonempty")
    return train, validation, test


def prepare_data(path: Path, config: Config, train_fraction: float,
                 validation_fraction: float,
                 split: tuple[int, int, int] | None = None,
                 sample_start: int = 0,
                 feature_set: str = "ohlcv", horizon_bars: int = 1,
                 split_gap: int = 0,
                 target_kind: str = CLOSE_RETURN_TARGET) -> TrainingData:
    return prepare_rows(
        read_csv(path), config, train_fraction, validation_fraction, split,
        sample_start, feature_set, horizon_bars, split_gap, target_kind,
    )


def _indexed_training_rows(
    values: torch.Tensor, starts: Sequence[int], seq_len: int, count: int,
) -> torch.Tensor:
    intervals, left, right = [], starts[0], starts[0] + seq_len
    for start in starts[1:count]:
        end = start + seq_len
        if start > right:
            intervals.append((left, right))
            left, right = start, end
        else:
            right = max(right, end)
    intervals.append((left, right))
    return torch.cat(tuple(values[start:end] for start, end in intervals))


def _prepare_indexed_rows(
    raw: torch.Tensor, config: Config, train_fraction: float,
    validation_fraction: float, split: tuple[int, int, int] | None,
    feature_set: str, horizon_bars: int, split_gap: int, target_kind: str,
    sample_rows: Sequence[SampleRows], allow_empty_later: bool,
) -> TrainingData:
    rows = tuple(sample_rows)
    lookback = feature_lookback(feature_set)
    if not rows or any(not isinstance(item, SampleRows) for item in rows) or \
       any(
           any(type(value) is not int or value < 0 for value in (
               item.as_of, item.entry, item.target, item.as_of_ordinal,
           ))
           for item in rows
       ):
        raise ValueError("indexed sample rows are invalid")
    starts = tuple(
        item.as_of - (config.seq_len + lookback) + 1 for item in rows
    )
    row_count = len(raw)
    if any(
        not 0 <= start <= len(raw) - lookback - config.seq_len or
        item.as_of >= item.entry or item.entry > item.target or
        item.target >= row_count
        for item, start in zip(rows, starts, strict=True)
    ) or any(
        left.as_of_ordinal >= right.as_of_ordinal or left_start >= right_start
        for left, right, left_start, right_start in zip(
            rows[:-1], rows[1:], starts[:-1], starts[1:], strict=True,
        )
    ):
        raise ValueError("indexed sample rows are invalid")

    usable = len(rows) - split_gap * 2
    counts = split or split_counts(usable, train_fraction, validation_fraction)
    if len(counts) != 3 or counts[0] < 1 or min(counts) < 0 or \
       not allow_empty_later and min(counts) < 1 or sum(counts) > usable:
        raise ValueError("chronological split is outside the indexed samples")
    train_count, validation_count, test_count = counts
    validation_start = train_count + split_gap
    test_start = validation_start + validation_count + split_gap
    segments = tuple(
        (start, count) for start, count in (
            (0, train_count),
            (validation_start, validation_count),
            (test_start, test_count),
        ) if count
    )
    if any(
        rows[start + count - 1].target > rows[next_start].as_of or
        rows[start + count - 1].as_of_ordinal + horizon_bars >
        rows[next_start].as_of_ordinal
        for (start, count), (next_start, _) in zip(
            segments, segments[1:], strict=False,
        )
    ):
        raise ValueError("indexed split exposes an unavailable prior label")

    values = feature_values(raw, feature_set)
    references = (
        raw[[item.as_of for item in rows], 3]
        if target_kind == CLOSE_RETURN_TARGET else
        raw[[item.entry for item in rows], 0]
    )
    outcomes = raw[[item.target for item in rows], 3]
    if not torch.all(references > 0):
        raise ValueError("target reference prices must be positive")
    raw_targets = torch.log(outcomes / references)
    if not torch.isfinite(raw_targets).all():
        raise ValueError("target price ratios must produce finite log returns")

    training_rows = _indexed_training_rows(
        values, starts, config.seq_len, train_count,
    )
    training_targets = raw_targets[:train_count]
    feature_mean = training_rows.mean(0)
    feature_scale = training_rows.std(0, unbiased=False)
    target_mean = training_targets.mean()
    target_scale = training_targets.std(unbiased=False)
    if not torch.isfinite(feature_mean).all() or \
       not torch.isfinite(feature_scale).all() or \
       not torch.all(feature_scale > 0) or not torch.isfinite(target_mean) or \
       not torch.isfinite(target_scale) or target_scale <= 0:
        raise ValueError(
            "training rows require positive finite feature and target scales"
        )
    features = values.sub_(feature_mean).div_(feature_scale)
    targets = (raw_targets - target_mean) / target_scale

    def windows(start: int, count: int) -> Windows:
        return Windows(
            features, targets, references, outcomes, config.seq_len,
            start, count, feature_starts=starts, sample_rows=rows,
        )

    return TrainingData(
        windows(0, train_count),
        windows(validation_start, validation_count),
        windows(test_start, test_count),
        feature_mean, feature_scale, target_mean, target_scale, feature_set,
        horizon_bars, target_kind,
    )


def prepare_rows(rows: array, config: Config, train_fraction: float,
                 validation_fraction: float,
                 split: tuple[int, int, int] | None = None,
                 sample_start: int = 0,
                 feature_set: str = "ohlcv", horizon_bars: int = 1,
                 split_gap: int = 0,
                 target_kind: str = CLOSE_RETURN_TARGET, *,
                 sample_rows: Sequence[SampleRows] | None = None,
                 allow_empty_later: bool = False,
                 ) -> TrainingData:
    """Scale purged target-time splits using only their retained training rows."""
    if horizon_bars < 1 or split_gap < 0 or target_kind not in TARGET_KINDS or \
       type(allow_empty_later) is not bool:
        raise ValueError("horizon, split gap, or target kind is invalid")
    row_count = len(rows) // FEATURE_COUNT
    lookback = feature_lookback(feature_set)
    # The clone gives PyTorch ownership while callers retain one compact row buffer.
    raw = torch.frombuffer(rows, dtype=torch.float32).view(
        row_count, FEATURE_COUNT,
    ).clone()
    if not torch.isfinite(raw).all() or not torch.all(raw[:, 3] > 0):
        raise ValueError("CSV values must remain finite binary32 with positive closes")
    if sample_rows is not None:
        if sample_start:
            raise ValueError("indexed samples cannot use a row offset")
        return _prepare_indexed_rows(
            raw, config, train_fraction, validation_fraction, split,
            feature_set, horizon_bars, split_gap, target_kind, sample_rows,
            allow_empty_later,
        )
    if allow_empty_later:
        raise ValueError("empty later splits require indexed samples")
    if row_count < config.seq_len + lookback + horizon_bars:
        raise ValueError("training requires lookback + seq_len + horizon rows")
    # Preserve price anchors before raw OHLCV features are normalized in place.
    opens, closes = raw[lookback:, 0].clone(), raw[lookback:, 3].clone()
    values = feature_values(raw, feature_set)
    outcomes = closes[horizon_bars:]
    references = (closes[:-horizon_bars] if target_kind == CLOSE_RETURN_TARGET else
                  opens[1:1 + len(outcomes)])
    offset = config.seq_len - 1
    references, outcomes = references[offset:], outcomes[offset:]
    if not torch.all(references > 0):
        raise ValueError("target reference prices must be positive")
    raw_targets = torch.log(outcomes / references)
    if not torch.isfinite(raw_targets).all():
        raise ValueError("target price ratios must produce finite log returns")

    available = len(raw_targets) - sample_start
    usable = available - split_gap * 2
    counts = split or split_counts(usable, train_fraction, validation_fraction)
    if sample_start < 0 or len(counts) != 3 or min(counts) <= 0 or \
       sum(counts) > usable:
        raise ValueError("chronological split is outside the available target range")
    train_count, validation_count, test_count = counts
    training_rows = values[
        sample_start:sample_start + config.seq_len + train_count - 1
    ]
    training_targets = raw_targets[sample_start:sample_start + train_count]
    feature_mean = training_rows.mean(0)
    feature_scale = training_rows.std(0, unbiased=False)
    target_mean = training_targets.mean()
    target_scale = training_targets.std(unbiased=False)
    if not torch.isfinite(feature_mean).all() or \
       not torch.isfinite(feature_scale).all() or \
       not torch.all(feature_scale > 0) or not torch.isfinite(target_mean) or \
       not torch.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("training rows require positive finite feature and target scales")
    features = values.sub_(feature_mean).div_(feature_scale)
    targets = (raw_targets - target_mean) / target_scale
    validation_start = sample_start + train_count + split_gap
    test_start = validation_start + validation_count + split_gap
    return TrainingData(
        Windows(features, targets, references, outcomes, config.seq_len,
                sample_start, train_count),
        Windows(features, targets, references, outcomes, config.seq_len,
                validation_start, validation_count),
        Windows(features, targets, references, outcomes, config.seq_len,
                test_start, test_count),
        feature_mean, feature_scale, target_mean, target_scale, feature_set,
        horizon_bars, target_kind,
    )


def data_loaders(data: DataSplits, batch_size: int,
                 seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(data.train, batch_size, shuffle=True, generator=generator),
        DataLoader(data.validation, batch_size, shuffle=False),
        DataLoader(data.test, batch_size, shuffle=False),
    )


def fit_epochs(model: nn.Module, data: DataSplits, batch_size: int,
               epochs: int, learning_rate: float, weight_decay: float,
               seed: int, device: torch.device
               ) -> tuple[DataLoader, ...]:
    """Fit a preselected epoch count without reading later split losses."""
    loaders = data_loaders(data, batch_size, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    for _ in range(epochs):
        if not math.isfinite(train_epoch(model, loaders[0], optimizer, device)):
            raise FloatingPointError("training produced a non-finite loss")
    return loaders


def fit_training_updates(
    model: nn.Module, loader: DataLoader, updates: int,
    learning_rate: float, weight_decay: float, device: torch.device,
) -> float:
    """Fit the caller's exact batch schedule without validation selection."""
    if type(updates) is not int or updates < 1 or len(loader) != updates:
        raise ValueError("training loader does not match its update budget")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    batches = iter(loader)
    loss = _train_updates(model, batches, optimizer, updates, device)
    _require_exhausted(batches)
    return loss


def fit_model(model: nn.Module, data: DataSplits, batch_size: int,
              epochs: int, patience: int, learning_rate: float,
              weight_decay: float, seed: int, device: torch.device,
              log_epochs: bool = False, *,
              train_loader: DataLoader | None = None,
              validation_loss: Callable[[], float] | None = None,
              ) -> tuple[Fit, tuple[DataLoader, ...]]:
    """Fit and restore the checkpoint selected by chronological validation loss."""
    loaders = data_loaders(data, batch_size, seed)
    if train_loader is not None:
        loaders = (train_loader, *loaders[1:])
    validate = validation_loss if validation_loss is not None else (
        lambda: mean_loss(model, loaders[1], device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(model, loaders[0], optimizer, device)
        observed = float(validate())
        if not math.isfinite(training_loss) or not math.isfinite(observed):
            raise FloatingPointError(f"epoch {epoch} produced a non-finite loss")
        if log_epochs:
            print(f"epoch={epoch} train={training_loss:.6g} val={observed:.6g}",
                  file=sys.stderr)
        if observed < best_loss:
            best_loss, best_epoch, stale = observed, epoch, 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale == patience:
                break
    if best_state is None:
        raise FloatingPointError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return Fit(best_loss, best_epoch, epoch), loaders


def fit_updates(
    model: nn.Module, loader: DataLoader, validation_loss: Callable[[], float],
    checkpoints: int, updates_per_checkpoint: int, learning_rate: float,
    weight_decay: float, device: torch.device,
    log_checkpoints: bool = False,
) -> UpdateFit:
    """Run an exact update budget and restore the best validation checkpoint."""
    if type(checkpoints) is not int or type(updates_per_checkpoint) is not int or \
       min(checkpoints, updates_per_checkpoint) < 1 or \
       len(loader) != checkpoints * updates_per_checkpoint:
        raise ValueError("fixed-update loader does not match its budget")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    batches = iter(loader)
    best_loss, best_state, best_checkpoint = math.inf, None, 0
    for checkpoint in range(1, checkpoints + 1):
        training_loss = _train_updates(
            model, batches, optimizer, updates_per_checkpoint, device,
        )
        observed = float(validation_loss())
        if not math.isfinite(observed):
            raise FloatingPointError(
                f"checkpoint {checkpoint} produced a non-finite loss"
            )
        if log_checkpoints:
            print(
                f"checkpoint={checkpoint} train={training_loss:.6g} "
                f"val={observed:.6g}", file=sys.stderr,
            )
        if observed < best_loss:
            best_loss, best_checkpoint = observed, checkpoint
            best_state = _snapshot(model)
    _require_exhausted(batches)
    if best_state is None:
        raise FloatingPointError(
            "training did not produce a finite validation checkpoint"
        )
    model.load_state_dict(best_state)
    return UpdateFit(
        best_loss, best_checkpoint, checkpoints * updates_per_checkpoint,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> tuple[ForecastTransformer, Artifact, dict[str, object]]:
    """Fit, select, and restore one chronological validation checkpoint."""
    if min(args.epochs, args.patience, args.batch_size) <= 0 or \
       not 0 <= args.seed <= 0x7FFF_FFFF_FFFF_FFFF or \
       not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0 or \
       not math.isfinite(args.weight_decay) or args.weight_decay < 0.0 or \
       not 0.0 < args.train_fraction < 1.0 or \
       not 0.0 < args.validation_fraction < 1.0 or \
       args.train_fraction + args.validation_fraction >= 1.0:
        raise SystemExit("invalid training or split arguments")
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
        config = Config(args.model_dim, args.heads, args.ff_dim,
                        args.layers, args.seq_len)
        config.validate()
        validate_identifiers(args.model_version, args.interval)
        device = torch.device(args.device)
        if device.type == "meta":
            raise ValueError("device must execute tensors")
        torch.empty(0, device=device)
        data = prepare_data(args.csv, config, args.train_fraction,
                            args.validation_fraction)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = ForecastTransformer(config).to(device)
    try:
        fit, loaders = fit_model(
            model, data, args.batch_size, args.epochs, args.patience,
            args.learning_rate, args.weight_decay, args.seed, device, True,
        )
    except FloatingPointError as error:
        raise SystemExit(str(error)) from error
    metrics = evaluate(model, loaders[2], data.target_mean,
                       data.target_scale, device)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise SystemExit("test evaluation produced non-finite metrics")
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise SystemExit("selected checkpoint contains non-finite parameters")
    # Preserve the standalone trainer's legacy JSON key; experiments use the
    # target-neutral name because their reference price can be the next open.
    metrics["last_close_baseline_mae"] = metrics["zero_return_baseline_mae"]
    model.cpu()
    artifact = Artifact(
        config=config,
        model_version=args.model_version,
        interval=args.interval,
        feature_mean=tuple(float(value) for value in data.feature_mean),
        feature_scale=tuple(float(value) for value in data.feature_scale),
        target_mean=float(data.target_mean),
        target_scale=float(data.target_scale),
        weights=export_weights(model),
    )
    report = {"artifact": str(args.artifact),
              "best_validation_scaled_mse": fit.best_validation_scaled_mse,
              "best_epoch": fit.best_epoch,
              "epochs_trained": fit.epochs_trained, "test": metrics}
    return model, artifact, report


def main() -> None:
    args = parse_args()
    _, artifact, report = train(args)
    try:
        write_artifact(args.artifact, artifact)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
