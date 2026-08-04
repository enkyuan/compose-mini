"""Write validated compose-mini V1 artifacts without framework dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import math
import os
import struct
import tempfile
import zlib

from tools.float32 import f32

FEATURE_COUNT = 5
MAX_PARAMETERS = 67_108_864
MAX_WORKSPACE_FLOATS = 67_108_864
WEIGHT_CHUNK = 16_384
WEIGHT_FIELDS = (
    "embed_W", "Wq", "Wk", "Wv", "Wo", "norm1_g", "norm1_b",
    "W1", "b1", "W2", "b2", "norm2_g", "norm2_b", "head_W", "head_b",
)
@dataclass(frozen=True)
class Config:
    model_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    seq_len: int
    in_dim: int = FEATURE_COUNT

    def validate(self) -> None:
        values = (
            self.model_dim, self.num_heads, self.ff_dim,
            self.num_layers, self.seq_len, self.in_dim,
        )
        if any(type(value) is not int or not 0 < value <= 0x7FFF_FFFF
               for value in values):
            raise ValueError("model dimensions must be positive signed 32-bit integers")
        # in_dim > FEATURE_COUNT is legal for in-memory experiments; the V1
        # export path re-checks the OHLCV width in _prefix (feature stats == 5).
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.parameter_count > MAX_PARAMETERS:
            raise ValueError("model exceeds the V1 parameter cap")
        if self.workspace_count > MAX_WORKSPACE_FLOATS:
            raise ValueError("model exceeds the V1 workspace cap")

    @property
    def parameter_count(self) -> int:
        d, f, layers = self.model_dim, self.ff_dim, self.num_layers
        return self.in_dim * d + layers * (4 * d * d + 2 * d * f + 5 * d + f) + d + 1

    @property
    def workspace_count(self) -> int:
        return 5 * self.seq_len * self.model_dim + self.seq_len

    def field_counts(self) -> dict[str, int]:
        d, f, layers = self.model_dim, self.ff_dim, self.num_layers
        return {
            "embed_W": self.in_dim * d,
            "Wq": layers * d * d,
            "Wk": layers * d * d,
            "Wv": layers * d * d,
            "Wo": layers * d * d,
            "norm1_g": layers * d,
            "norm1_b": layers * d,
            "W1": layers * d * f,
            "b1": layers * f,
            "W2": layers * f * d,
            "b2": layers * d,
            "norm2_g": layers * d,
            "norm2_b": layers * d,
            "head_W": d,
            "head_b": 1,
        }


@dataclass(frozen=True)
class Artifact:
    config: Config
    model_version: str
    interval: str
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    target_mean: float
    target_scale: float
    weights: Mapping[str, Iterable[float]]


class _BodyWriter:
    def __init__(self, file: object) -> None:
        self.file = file
        self.size = 0
        self.checksum = 0

    def write(self, data: bytes) -> None:
        if self.file.write(data) != len(data):
            raise OSError("short artifact write")
        self.checksum = zlib.crc32(data, self.checksum)
        self.size += len(data)


def _token(value: str, size: int, name: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be visible ASCII") from error
    if not encoded or len(encoded) >= size or any(not 33 <= byte <= 126 for byte in encoded):
        raise ValueError(f"{name} must be 1..{size - 1} visible ASCII bytes")
    return encoded.ljust(size, b"\0")


def _identifiers(model_version: str, interval: str) -> tuple[bytes, bytes]:
    return _token(model_version, 64, "model_version"), _token(interval, 16, "interval")


def validate_identifiers(model_version: str, interval: str) -> None:
    """Reject metadata that cannot be represented in an artifact."""
    _identifiers(model_version, interval)


def _f32(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    try:
        return f32(value)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name} must fit IEEE-754 binary32") from error


def _prefix(artifact: Artifact) -> bytes:
    config = artifact.config
    config.validate()
    if config.in_dim != FEATURE_COUNT:
        raise ValueError("artifact V1 requires five OHLCV features")
    model_version, interval = _identifiers(artifact.model_version, artifact.interval)
    means = tuple(_f32(value, "feature mean") for value in artifact.feature_mean)
    scales = tuple(_f32(value, "feature scale") for value in artifact.feature_scale)
    if len(means) != FEATURE_COUNT or len(scales) != FEATURE_COUNT:
        raise ValueError("feature statistics must contain five values")
    target_mean = _f32(artifact.target_mean, "target mean")
    target_scale = _f32(artifact.target_scale, "target scale")
    if any(value <= 0.0 for value in scales) or target_scale <= 0.0:
        raise ValueError("feature and target scales must be positive")
    values = (
        config.model_dim, config.num_heads, config.ff_dim,
        config.num_layers, config.seq_len, config.in_dim, 1, 0,
    )
    prefix = struct.pack(
        "<8I64s16s5f5fff", *values,
        model_version, interval,
        *means, *scales, target_mean, target_scale,
    )
    assert len(prefix) == 160
    return prefix


def _write_weights(writer: _BodyWriter, artifact: Artifact) -> None:
    counts = artifact.config.field_counts()
    if set(artifact.weights) != set(WEIGHT_FIELDS):
        raise ValueError("weights must contain exactly the V1 fields")
    for field in WEIGHT_FIELDS:
        values = iter(artifact.weights[field])
        remaining = counts[field]
        while remaining:
            chunk = []
            for _ in range(min(remaining, WEIGHT_CHUNK)):
                try:
                    chunk.append(_f32(next(values), field))
                except StopIteration as error:
                    raise ValueError(f"{field} has too few values") from error
            writer.write(struct.pack(f"<{len(chunk)}f", *chunk))
            remaining -= len(chunk)
        try:
            next(values)
        except StopIteration:
            continue
        raise ValueError(f"{field} has too many values")


def write_artifact(path: str | os.PathLike[str], artifact: Artifact) -> None:
    """Atomically write one checksummed artifact after validating every value."""
    prefix = _prefix(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w+b") as file:
            if file.write(b"\0" * 32) != 32:
                raise OSError("short artifact header write")
            body = _BodyWriter(file)
            body.write(prefix)
            _write_weights(body, artifact)
            expected = 160 + 4 * artifact.config.parameter_count
            if body.size != expected:
                raise ValueError("artifact body length does not match the configuration")
            file.seek(0)
            header = struct.pack(
                "<8sIIQQ", b"CMPMINI\0", 1, 32, body.size, body.checksum,
            )
            if file.write(header) != len(header):
                raise OSError("short artifact header write")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
