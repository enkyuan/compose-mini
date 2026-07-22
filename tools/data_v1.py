"""Read the strict CSV grammar shared by training and C inference."""

from array import array
from collections.abc import Iterator
from pathlib import Path
import ctypes
import errno
import locale
import math

CSV_HEADER = "timestamp,open,high,low,close,volume"
FEATURE_COUNT = 5
LINE_CAP = 512
# Delegate numeric syntax and range flags to the same libc primitive as C.
_STRTOF = ctypes.CDLL(None, use_errno=True).strtof
_STRTOF.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p))
_STRTOF.restype = ctypes.c_float


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("rb") as file:
        number = 0
        while raw := file.readline(LINE_CAP + 1):
            number += 1
            terminated = raw.endswith(b"\n")
            if len(raw) > LINE_CAP or (len(raw) == LINE_CAP and not terminated) or b"\0" in raw:
                raise ValueError(f"line {number}: invalid or overlong record")
            raw = raw[:-1] if terminated else raw
            raw = raw[:-1] if raw.endswith(b"\r") else raw
            if not raw:
                raise ValueError(f"line {number}: empty record")
            try:
                yield number, raw.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(f"line {number}: records must be ASCII") from error


def _timestamp_valid(value: str) -> bool:
    separators = {4: "-", 7: "-", 10: "T", 13: ":", 16: ":", 19: "Z"}
    if len(value) != 20 or any(value[index] != separator
                               for index, separator in separators.items()) or \
       any(not byte.isascii() or not byte.isdigit()
           for index, byte in enumerate(value) if index not in separators):
        return False
    year, month, day = int(value[:4]), int(value[5:7]), int(value[8:10])
    hour, minute, second = int(value[11:13]), int(value[14:16]), int(value[17:19])
    days = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if year < 1 or not 1 <= month <= 12 or hour > 23 or minute > 59 or second > 59:
        return False
    limit = days[month] + (month == 2 and
                           (year % 400 == 0 or year % 4 == 0 and year % 100 != 0))
    return 1 <= day <= limit


def _number(field: str, number: int) -> float:
    if field[0].isspace():
        raise ValueError(f"line {number}: invalid numeric value")
    encoded = field.encode("ascii")
    end = ctypes.c_char_p()
    ctypes.set_errno(0)
    value = float(_STRTOF(encoded, ctypes.byref(end)))
    if end.value != b"":
        raise ValueError(f"line {number}: invalid numeric value")
    if ctypes.get_errno() == errno.ERANGE or not math.isfinite(value):
        raise ValueError(f"line {number}: numeric value is outside binary32")
    return value


def _records(path: Path) -> Iterator[tuple[str, list[float]]]:
    if locale.setlocale(locale.LC_NUMERIC) != "C":
        raise ValueError("LC_NUMERIC must be C")
    lines = _lines(path)
    try:
        _, header = next(lines)
    except StopIteration as error:
        raise ValueError("CSV is empty") from error
    if header != CSV_HEADER:
        raise ValueError(f"CSV header must be {CSV_HEADER}")

    previous = ""
    for number, line in lines:
        fields = line.split(",")
        if len(fields) != FEATURE_COUNT + 1 or any(not field for field in fields):
            raise ValueError(f"line {number}: expected exactly six fields")
        timestamp = fields[0]
        if not _timestamp_valid(timestamp):
            raise ValueError(f"line {number}: invalid canonical UTC timestamp")
        if previous and timestamp <= previous:
            raise ValueError(f"line {number}: timestamps must increase")
        values = [_number(field, number) for field in fields[1:]]
        if values[3] <= 0.0:
            raise ValueError(f"line {number}: close must be positive")
        yield timestamp, values
        previous = timestamp


def read_csv(path: Path) -> array:
    """Return flat chronological float32 OHLCV from one exact six-field file."""
    rows = array("f")
    for _, values in _records(path):
        rows.extend(values)
    return rows


def read_bars(path: Path) -> tuple[tuple[str, ...], array]:
    """Return timestamps and flat OHLCV without duplicating CSV validation."""
    timestamps, rows = [], array("f")
    for timestamp, values in _records(path):
        timestamps.append(timestamp)
        rows.extend(values)
    return tuple(timestamps), rows
