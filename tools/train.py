#!/usr/bin/env python3
"""Train the exact compose-mini architecture and export a V1 artifact."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
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
                 closes: torch.Tensor, seq_len: int, start: int, count: int) -> None:
        self.features, self.targets, self.closes = features, targets, closes
        self.seq_len, self.start, self.count = seq_len, start, count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.start + index
        final_row = sample + self.seq_len - 1
        return (
            self.features[sample:sample + self.seq_len], self.targets[sample],
            self.closes[final_row], self.closes[final_row + 1],
        )


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
              "close_absolute": 0.0, "baseline_close_absolute": 0.0}
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
    count = len(loader.dataset)
    return {
        "return_mse": totals["return_squared"] / count,
        "return_mae": totals["return_absolute"] / count,
        "close_mae": totals["close_absolute"] / count,
        "last_close_baseline_mae": totals["baseline_close_absolute"] / count,
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
        rows = read_csv(args.csv)
        device = torch.device(args.device)
        if device.type == "meta":
            raise ValueError("device must execute tensors")
        torch.empty(0, device=device)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    row_count = len(rows) // FEATURE_COUNT
    if row_count <= config.seq_len:
        raise SystemExit("training requires at least seq_len + 1 rows")

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    # The clone gives PyTorch ownership before the compact parser buffer is released.
    raw = torch.frombuffer(rows, dtype=torch.float32).view(row_count, FEATURE_COUNT).clone()
    del rows
    if not torch.isfinite(raw).all() or not torch.all(raw[:, 3] > 0):
        raise SystemExit("CSV values must remain finite binary32 with positive closes")
    closes = raw[:, 3].clone()
    raw_targets = torch.log(closes[1:] / closes[:-1])[config.seq_len - 1:]
    if not torch.isfinite(raw_targets).all():
        raise SystemExit("close ratios must produce finite log returns")
    train_count, validation_count, test_count = split_counts(
        len(raw_targets), args.train_fraction, args.validation_fraction,
    )
    # Fit scalers only on unique rows and targets reachable by training samples.
    training_rows = raw[:config.seq_len + train_count - 1]
    feature_mean = training_rows.mean(0)
    feature_scale = training_rows.std(0, unbiased=False)
    target_mean = raw_targets[:train_count].mean()
    target_scale = raw_targets[:train_count].std(unbiased=False)
    if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_scale).all() or \
       not torch.all(feature_scale > 0) or not torch.isfinite(target_mean) or \
       not torch.isfinite(target_scale) or target_scale <= 0:
        raise SystemExit("training rows require positive finite feature and target scales")
    features = raw.sub_(feature_mean).div_(feature_scale)
    targets = (raw_targets - target_mean) / target_scale

    datasets = (
        Windows(features, targets, closes, config.seq_len, 0, train_count),
        Windows(features, targets, closes, config.seq_len,
                train_count, validation_count),
        Windows(features, targets, closes, config.seq_len,
                train_count + validation_count, test_count),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets[0], args.batch_size, shuffle=True,
                              generator=generator)
    validation_loader = DataLoader(datasets[1], args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], args.batch_size, shuffle=False)
    model = ForecastTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    best_loss, best_state, stale = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        try:
            training_loss = train_epoch(model, train_loader, optimizer, device)
        except FloatingPointError as error:
            raise SystemExit(str(error)) from error
        validation_loss = mean_loss(model, validation_loader, device)
        if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
            raise SystemExit(f"epoch {epoch} produced a non-finite loss")
        print(f"epoch={epoch} train={training_loss:.6g} val={validation_loss:.6g}",
              file=sys.stderr)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale == args.patience:
                break
    if best_state is None:
        raise SystemExit("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, target_mean, target_scale, device)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise SystemExit("test evaluation produced non-finite metrics")
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise SystemExit("selected checkpoint contains non-finite parameters")
    model.cpu()
    artifact = Artifact(
        config=config,
        model_version=args.model_version,
        interval=args.interval,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        target_mean=float(target_mean),
        target_scale=float(target_scale),
        weights=export_weights(model),
    )
    report = {"artifact": str(args.artifact),
              "best_validation_scaled_mse": best_loss, "test": metrics}
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
