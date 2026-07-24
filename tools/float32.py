"""Provide binary32 rounding and a canonical vector codec for Python tools."""

import base64
import binascii
from collections.abc import Iterable
import math
import struct


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def encode_f32le_base64(values: Iterable[float]) -> dict[str, object]:
    """Encode finite values as one canonical little-endian binary32 payload."""
    rounded = []
    try:
        iterator = iter(values)
    except TypeError as error:
        raise ValueError("float32 values must be iterable") from error
    for value in iterator:
        if type(value) not in (int, float):
            raise ValueError("float32 values must be numeric")
        try:
            number = f32(value)
        except (OverflowError, struct.error, TypeError, ValueError) as error:
            raise ValueError("float32 value is invalid") from error
        if not math.isfinite(number):
            raise ValueError("float32 values must be finite")
        rounded.append(number)
    raw = struct.pack(f"<{len(rounded)}f", *rounded)
    return {
        "encoding": "f32le-base64",
        "count": len(rounded),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def decode_f32le_base64(
    payload: object, *, expected_count: int | None = None,
) -> tuple[float, ...]:
    """Decode one canonical finite little-endian binary32 payload."""
    fields = {"encoding", "count", "base64"}
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError("float32 payload fields are invalid")
    encoding = payload["encoding"]
    count = payload["count"]
    encoded = payload["base64"]
    if type(encoding) is not str or encoding != "f32le-base64":
        raise ValueError("float32 payload encoding is invalid")
    if type(count) is not int or count < 0:
        raise ValueError("float32 payload count is invalid")
    if expected_count is not None and (
        type(expected_count) is not int or expected_count < 0 or
        count != expected_count
    ):
        raise ValueError("float32 payload count is unexpected")
    if type(encoded) is not str:
        raise ValueError("float32 payload base64 is invalid")
    if len(encoded) != 4 * ((4 * count + 2) // 3):
        raise ValueError("float32 payload byte count is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("float32 payload base64 is invalid") from error
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("float32 payload base64 is not canonical")
    if len(raw) != 4 * count:
        raise ValueError("float32 payload byte count is invalid")
    values = tuple(value for value, in struct.iter_unpack("<f", raw))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("float32 payload values must be finite")
    return values


def ulp_distance(left: float, right: float) -> int:
    """Return the representable-float distance between two binary32 values."""
    def ordered(value: float) -> int:
        bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
        return (~bits & 0xFFFF_FFFF) if bits & 0x8000_0000 else bits | 0x8000_0000

    return abs(ordered(left) - ordered(right))
