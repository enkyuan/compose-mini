"""Scalar float32 reference for small C-runtime parity fixtures."""

from __future__ import annotations

from collections.abc import Sequence
import math

from tools.artifact_v1 import Artifact, WEIGHT_FIELDS
from tools.float32 import f32


def _sum_product(a: Sequence[float], b: Sequence[float], initial: float = 0.0) -> float:
    total = f32(initial)
    for left, right in zip(a, b, strict=True):
        total = f32(total + f32(left * right))
    return total


def _matmul(a: Sequence[float], b: Sequence[float], m: int, k: int, n: int) -> list[float]:
    out = [0.0] * (m * n)
    for row in range(m):
        for column in range(n):
            out[row * n + column] = _sum_product(
                a[row * k:(row + 1) * k], b[column:k * n:n],
            )
    return out


def _layernorm(x: Sequence[float], gamma: Sequence[float],
               beta: Sequence[float]) -> list[float]:
    size = len(x)
    mean = 0.0
    for value in x:
        mean = f32(mean + value)
    mean = f32(mean / f32(size))
    variance = 0.0
    for value in x:
        centered = f32(value - mean)
        variance = f32(variance + f32(centered * centered))
    variance = f32(variance / f32(size))
    inv_std = f32(1.0 / f32(math.sqrt(f32(variance + f32(1e-5)))))
    return [
        f32(f32(f32(scale * f32(value - mean)) * inv_std) + shift)
        for value, scale, shift in zip(x, gamma, beta, strict=True)
    ]


def _normalize_rows(x: Sequence[float], gamma: Sequence[float],
                    beta: Sequence[float], rows: int, columns: int) -> list[float]:
    out: list[float] = []
    for row in range(rows):
        out.extend(_layernorm(x[row * columns:(row + 1) * columns], gamma, beta))
    return out


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in values]
    total = 0.0
    for value in exponentials:
        total = f32(total + value)
    inverse = f32(1.0 / total)
    return [f32(value * inverse) for value in exponentials]


def _attention(x: Sequence[float], wq: Sequence[float], wk: Sequence[float],
               wv: Sequence[float], wo: Sequence[float], seq_len: int,
               model_dim: int, num_heads: int) -> list[float]:
    query = _matmul(x, wq, seq_len, model_dim, model_dim)
    key = _matmul(x, wk, seq_len, model_dim, model_dim)
    value = _matmul(x, wv, seq_len, model_dim, model_dim)
    context = [0.0] * (seq_len * model_dim)
    head_dim = model_dim // num_heads
    scale = f32(1.0 / f32(math.sqrt(f32(head_dim))))
    for target in range(seq_len):
        for head_index in range(num_heads):
            head = head_index * head_dim
            q = target * model_dim + head
            scores = []
            for source in range(seq_len):
                k = source * model_dim + head
                scores.append(f32(_sum_product(
                    query[q:q + head_dim], key[k:k + head_dim],
                ) * scale))
            probabilities = _softmax(scores)
            for dimension in range(head_dim):
                total = 0.0
                for source, probability in enumerate(probabilities):
                    index = source * model_dim + head + dimension
                    total = f32(total + f32(probability * value[index]))
                context[q + dimension] = total
    return _matmul(context, wo, seq_len, model_dim, model_dim)


def _gelu(value: float) -> float:
    erf = f32(math.erf(f32(value * f32(0.7071067811865475))))
    return f32(f32(f32(0.5) * value) * f32(1.0 + erf))


def _ffn(x: Sequence[float], w1: Sequence[float], b1: Sequence[float],
         w2: Sequence[float], b2: Sequence[float], model_dim: int,
         ff_dim: int) -> list[float]:
    out = list(b2)
    for hidden in range(ff_dim):
        total = b1[hidden]
        for index in range(model_dim):
            total = f32(total + f32(x[index] * w1[index * ff_dim + hidden]))
        activation = _gelu(total)
        for index in range(model_dim):
            out[index] = f32(out[index] + f32(
                activation * w2[hidden * model_dim + index],
            ))
    return out


def _positional(x: list[float], seq_len: int, model_dim: int) -> None:
    exponent = f32(f32(-2.0) / f32(model_dim))
    step = f32(math.pow(f32(10000.0), exponent))
    for position in range(seq_len):
        frequency = 1.0
        for index in range(0, model_dim, 2):
            angle = f32(f32(position) * frequency)
            offset = position * model_dim + index
            x[offset] = f32(x[offset] + f32(math.sin(angle)))
            if index + 1 < model_dim:
                x[offset + 1] = f32(x[offset + 1] + f32(math.cos(angle)))
            frequency = f32(frequency * step)


class ReferenceModel:
    def __init__(self, artifact: Artifact) -> None:
        artifact.config.validate()
        counts = artifact.config.field_counts()
        self.artifact = artifact
        if set(artifact.weights) != set(WEIGHT_FIELDS):
            raise ValueError("weights do not contain exactly the V1 fields")
        try:
            self.weights = {
                name: tuple(f32(value) for value in artifact.weights[name])
                for name in WEIGHT_FIELDS
            }
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("weights must fit binary32") from error
        for name, count in counts.items():
            if len(self.weights[name]) != count:
                raise ValueError(f"{name} length does not match the configuration")
            if any(not math.isfinite(value) for value in self.weights[name]):
                raise ValueError(f"{name} must contain only finite values")

    def _layer(self, name: str, layer: int, size: int) -> tuple[float, ...]:
        start = layer * size
        return self.weights[name][start:start + size]

    def forward(self, x: Sequence[float]) -> float:
        config = self.artifact.config
        d, f, seq_len = config.model_dim, config.ff_dim, config.seq_len
        if len(x) != seq_len * config.in_dim:
            raise ValueError("input window has the wrong shape")
        x = tuple(f32(value) for value in x)
        hidden = _matmul(x, self.weights["embed_W"], seq_len, config.in_dim, d)
        _positional(hidden, seq_len, d)
        for layer in range(config.num_layers):
            norm = _normalize_rows(
                hidden, self._layer("norm1_g", layer, d),
                self._layer("norm1_b", layer, d), seq_len, d,
            )
            branch = _attention(
                norm, self._layer("Wq", layer, d * d),
                self._layer("Wk", layer, d * d),
                self._layer("Wv", layer, d * d),
                self._layer("Wo", layer, d * d),
                seq_len, d, config.num_heads,
            )
            hidden = [f32(value + residual) for value, residual in zip(
                hidden, branch, strict=True,
            )]
            norm = _normalize_rows(
                hidden, self._layer("norm2_g", layer, d),
                self._layer("norm2_b", layer, d), seq_len, d,
            )
            branch = []
            for row in range(seq_len):
                branch.extend(_ffn(
                    norm[row * d:(row + 1) * d],
                    self._layer("W1", layer, d * f),
                    self._layer("b1", layer, f),
                    self._layer("W2", layer, f * d),
                    self._layer("b2", layer, d), d, f,
                ))
            hidden = [f32(value + residual) for value, residual in zip(
                hidden, branch, strict=True,
            )]
        final = hidden[(seq_len - 1) * d:seq_len * d]
        return _sum_product(final, self.weights["head_W"], self.weights["head_b"][0])


def predict_windows(rows: Sequence[Sequence[float]],
                    artifact: Artifact) -> list[tuple[float, float]]:
    """Return unscaled log-return and reconstructed close for every window."""
    config = artifact.config
    config.validate()
    if len(rows) < config.seq_len or any(len(row) != config.in_dim for row in rows):
        raise ValueError("rows do not satisfy the model input shape")
    if len(artifact.feature_mean) != config.in_dim or \
       len(artifact.feature_scale) != config.in_dim:
        raise ValueError("feature statistics do not satisfy the model input shape")
    try:
        means = tuple(f32(value) for value in artifact.feature_mean)
        scales = tuple(f32(value) for value in artifact.feature_scale)
        input_rows = tuple(tuple(f32(value) for value in row) for row in rows)
        target_mean, target_scale = f32(artifact.target_mean), f32(artifact.target_scale)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("inputs and statistics must fit binary32") from error
    if not all(math.isfinite(value) for value in (*means, *scales,
                                                   target_mean, target_scale)) or \
       any(scale <= 0.0 for scale in scales) or target_scale <= 0.0 or \
       any(not math.isfinite(value) for row in input_rows for value in row) or \
       any(row[3] <= 0.0 for row in input_rows):
        raise ValueError("inputs require finite values, positive closes, and positive scales")
    try:
        scaled = [
            f32(f32(value - means[index]) / scales[index])
            for row in input_rows for index, value in enumerate(row)
        ]
    except OverflowError as error:
        raise ValueError("scaled inputs must fit binary32") from error
    model = ReferenceModel(artifact)
    predictions = []
    width = config.seq_len * config.in_dim
    for start in range(len(rows) - config.seq_len + 1):
        try:
            model_value = model.forward(
                scaled[start * config.in_dim:start * config.in_dim + width],
            )
            log_return = f32(f32(model_value * target_scale) + target_mean)
            latest_close = input_rows[start + config.seq_len - 1][3]
            predicted_close = f32(latest_close * f32(math.exp(log_return)))
        except (OverflowError, ValueError) as error:
            raise ValueError("forecast is outside the finite binary32 range") from error
        if not math.isfinite(log_return) or not math.isfinite(predicted_close) or \
           predicted_close <= 0.0:
            raise ValueError("forecast is outside the finite binary32 range")
        predictions.append((log_return, predicted_close))
    return predictions
