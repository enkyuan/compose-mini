#!/usr/bin/env python3
"""Train the exact compose-mini architecture and export a V1 artifact."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
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
from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.float32 import f32

FEATURE_NAMES = {
    "ohlcv": ("open", "high", "low", "close", "volume"),
    "stationary-v1": (
        "log_gap", "log_body", "log_upper_wick", "log_lower_wick",
        "log_volume_change",
    ),
}
FEATURE_SETS = tuple(FEATURE_NAMES)


def feature_lookback(feature_set: str) -> int:
    """Return raw bars required before the first model feature row."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unsupported feature set: {feature_set}")
    return int(feature_set != "ohlcv")


def feature_values(raw: torch.Tensor, feature_set: str) -> torch.Tensor:
    """Return five features aligned to each completed source bar."""
    if feature_set == "ohlcv":
        return raw
    feature_lookback(feature_set)
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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values @ self.embed_W + self.position
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden[:, -1] @ self.head_W + self.head_b


class Windows(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor,
                 closes: torch.Tensor, seq_len: int, start: int, count: int,
                 horizon_bars: int = 1) -> None:
        self.features, self.targets, self.closes = features, targets, closes
        self.seq_len, self.start, self.count = seq_len, start, count
        self.horizon_bars = horizon_bars

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.start + index
        final_row = sample + self.seq_len - 1
        return (
            self.features[sample:sample + self.seq_len], self.targets[sample],
            self.closes[final_row], self.closes[final_row + self.horizon_bars],
        )


@dataclass(frozen=True)
class TrainingData:
    """Own one leakage-safe chronological split and its fitted scalers."""

    train: Windows
    validation: Windows
    test: Windows
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    feature_set: str = "ohlcv"
    horizon_bars: int = 1


@dataclass(frozen=True)
class Fit:
    best_validation_scaled_mse: float
    best_epoch: int
    epochs_trained: int


def mean_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for features, targets, *_ in loader:
            loss = F.mse_loss(model(features.to(device)), targets.to(device), reduction="sum")
            total += loss.item()
    return total / len(loader.dataset)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    total = 0.0
    for features, targets, *_ in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(features), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(features)
    return total / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, target_mean: torch.Tensor,
             target_scale: torch.Tensor, device: torch.device) -> dict[str, float]:
    totals = {"return_squared": 0.0, "return_absolute": 0.0,
              "close_absolute": 0.0, "baseline_close_absolute": 0.0,
              "direction_correct": 0.0}
    model.eval()
    with torch.no_grad():
        for features, targets, latest, actual in loader:
            predicted_return = model(features.to(device)).cpu() * target_scale + target_mean
            actual_return = targets * target_scale + target_mean
            difference = predicted_return - actual_return
            predicted_close = latest * predicted_return.exp()
            totals["return_squared"] += difference.square().sum().item()
            totals["return_absolute"] += difference.abs().sum().item()
            totals["close_absolute"] += (predicted_close - actual).abs().sum().item()
            totals["baseline_close_absolute"] += (latest - actual).abs().sum().item()
            totals["direction_correct"] += (
                predicted_return.sign() == actual_return.sign()
            ).sum().item()
    count = len(loader.dataset)
    return {
        "return_mse": totals["return_squared"] / count,
        "return_mae": totals["return_absolute"] / count,
        "close_mae": totals["close_absolute"] / count,
        "last_close_baseline_mae": totals["baseline_close_absolute"] / count,
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
                 split_gap: int = 0) -> TrainingData:
    """Scale purged target-time splits using only their retained training rows."""
    if horizon_bars < 1 or split_gap < 0:
        raise ValueError("horizon must be positive and split gap nonnegative")
    rows = read_csv(path)
    row_count = len(rows) // FEATURE_COUNT
    lookback = feature_lookback(feature_set)
    if row_count < config.seq_len + lookback + horizon_bars:
        raise ValueError("training requires lookback + seq_len + horizon rows")
    # The clone gives PyTorch ownership before the compact parser buffer is released.
    raw = torch.frombuffer(rows, dtype=torch.float32).view(
        row_count, FEATURE_COUNT,
    ).clone()
    del rows
    if not torch.isfinite(raw).all() or not torch.all(raw[:, 3] > 0):
        raise ValueError("CSV values must remain finite binary32 with positive closes")
    closes = raw[lookback:, 3].clone()
    values = feature_values(raw, feature_set)
    raw_targets = torch.log(
        closes[horizon_bars:] / closes[:-horizon_bars]
    )[config.seq_len - 1:]
    if not torch.isfinite(raw_targets).all():
        raise ValueError("close ratios must produce finite log returns")

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
        Windows(features, targets, closes, config.seq_len,
                sample_start, train_count, horizon_bars),
        Windows(features, targets, closes, config.seq_len,
                validation_start, validation_count, horizon_bars),
        Windows(features, targets, closes, config.seq_len,
                test_start, test_count, horizon_bars),
        feature_mean, feature_scale, target_mean, target_scale, feature_set,
        horizon_bars,
    )


def data_loaders(data: TrainingData, batch_size: int,
                 seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(data.train, batch_size, shuffle=True, generator=generator),
        DataLoader(data.validation, batch_size, shuffle=False),
        DataLoader(data.test, batch_size, shuffle=False),
    )


def fit_model(model: nn.Module, data: TrainingData, batch_size: int,
              epochs: int, patience: int, learning_rate: float,
              weight_decay: float, seed: int, device: torch.device,
              log_epochs: bool = False) -> tuple[Fit, tuple[DataLoader, ...]]:
    """Fit and restore the checkpoint selected by chronological validation loss."""
    loaders = data_loaders(data, batch_size, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(model, loaders[0], optimizer, device)
        validation_loss = mean_loss(model, loaders[1], device)
        if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
            raise FloatingPointError(f"epoch {epoch} produced a non-finite loss")
        if log_epochs:
            print(f"epoch={epoch} train={training_loss:.6g} val={validation_loss:.6g}",
                  file=sys.stderr)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale == patience:
                break
    if best_state is None:
        raise FloatingPointError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return Fit(best_loss, best_epoch, epoch), loaders


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
