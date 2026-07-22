"""Provide explicit IEEE-754 binary32 rounding for Python tooling."""

import struct


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def ulp_distance(left: float, right: float) -> int:
    """Return the representable-float distance between two binary32 values."""
    def ordered(value: float) -> int:
        bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
        return (~bits & 0xFFFF_FFFF) if bits & 0x8000_0000 else bits | 0x8000_0000

    return abs(ordered(left) - ordered(right))
