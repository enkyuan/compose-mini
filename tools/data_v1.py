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
CLOSE_RETURN_TARGET = "close-to-close-v1"
EXECUTABLE_RETURN_TARGET = "executable-return-v1"
TARGET_KINDS = (CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET)
TARGET_FORMULAS = {
    CLOSE_RETURN_TARGET: "log(close[t + horizon] / close[t])",
    EXECUTABLE_RETURN_TARGET: "log(close[t + horizon] / open[t + 1])",
}
LINE_CAP = 512
_TIMESTAMP_SEPARATORS = {
    4: ord("-"),
    7: ord("-"),
    10: ord("T"),
    13: ord(":"),
    16: ord(":"),
    19: ord("Z"),
}
# Delegate numeric syntax and range flags to the same libc primitive as C.
_STRTOF = ctypes.CDLL(None, use_errno=True).strtof
_STRTOF.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p))
_STRTOF.restype = ctypes.c_float


def _timestamp_shape(value: bytes) -> bool:
    return len(value) == 20 and not any(
        value[index] != separator
        for index, separator in _TIMESTAMP_SEPARATORS.items()
    ) and not any(
        not 48 <= byte <= 57
        for index, byte in enumerate(value)
        if index not in _TIMESTAMP_SEPARATORS
    )


def _lines(path: Path, stop: bytes | None = None) -> Iterator[tuple[int, str]]:
    with path.open("rb") as file:
        number = 0
        while raw := file.readline(LINE_CAP + 1):
            number += 1
            # Establish the bound before decoding or validating the payload.
            timestamp = raw[:20]
            if (
                stop is not None
                and _timestamp_shape(timestamp)
                and timestamp > stop
            ):
                return
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
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    if not _timestamp_shape(encoded):
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


def _fields(
    path: Path, stop: str | None = None, *, exact_stop: bool = True,
) -> Iterator[tuple[int, str, list[str]]]:
    if stop is not None and not _timestamp_valid(stop):
        raise ValueError("stop must be a canonical UTC timestamp")
    lines = _lines(path, None if stop is None else stop.encode("ascii"))
    try:
        _, header = next(lines)
    except StopIteration as error:
        raise ValueError("CSV is empty") from error
    if header != CSV_HEADER:
        raise ValueError(f"CSV header must be {CSV_HEADER}")

    previous = ""
    for number, line in lines:
        timestamp, separator, row = line.partition(",")
        if not _timestamp_valid(timestamp):
            raise ValueError(f"line {number}: invalid canonical UTC timestamp")
        fields = row.split(",") if separator else []
        if len(fields) != FEATURE_COUNT or any(not field for field in fields):
            raise ValueError(f"line {number}: expected exactly six fields")
        if previous and timestamp <= previous:
            raise ValueError(f"line {number}: timestamps must increase")
        yield number, timestamp, fields
        previous = timestamp
    if stop is not None and exact_stop and previous != stop:
        raise ValueError("stop timestamp is not present")


def _records(
    path: Path, stop: str | None = None, *, exact_stop: bool = True,
) -> Iterator[tuple[str, list[float]]]:
    if locale.setlocale(locale.LC_NUMERIC) != "C":
        raise ValueError("LC_NUMERIC must be C")
    for number, timestamp, fields in _fields(
        path, stop, exact_stop=exact_stop,
    ):
        values = [_number(field, number) for field in fields]
        if values[3] <= 0.0:
            raise ValueError(f"line {number}: close must be positive")
        yield timestamp, values


def read_csv(path: Path) -> array:
    """Return flat chronological float32 OHLCV from one exact six-field file."""
    rows = array("f")
    for _, values in _records(path):
        rows.extend(values)
    return rows


def _bars(
    path: Path, stop: str | None = None, *, exact_stop: bool = True,
) -> tuple[tuple[str, ...], array]:
    timestamps, rows = [], array("f")
    for timestamp, values in _records(
        path, stop, exact_stop=exact_stop,
    ):
        timestamps.append(timestamp)
        rows.extend(values)
    return tuple(timestamps), rows


def read_bars(path: Path) -> tuple[tuple[str, ...], array]:
    """Return timestamps and flat OHLCV without duplicating CSV validation."""
    return _bars(path)


def read_bars_through(path: Path, stop: str) -> tuple[tuple[str, ...], array]:
    """Return bars through stop without decoding later observations."""
    return _bars(path, stop)


def read_bars_until(path: Path, stop: str) -> tuple[tuple[str, ...], array]:
    """Return bars at or before a cutoff that need not be observed."""
    return _bars(path, stop, exact_stop=False)


def read_timestamps(path: Path) -> tuple[str, ...]:
    """Return validated ordered timestamps without parsing OHLCV values."""
    return tuple(timestamp for _, timestamp, _ in _fields(path))


def read_timestamps_through(path: Path, stop: str) -> tuple[str, ...]:
    """Return timestamps through stop without parsing OHLCV values."""
    return tuple(timestamp for _, timestamp, _ in _fields(path, stop))


def read_timestamps_until(path: Path, stop: str) -> tuple[str, ...]:
    """Return timestamps at or before a cutoff that need not be observed."""
    return tuple(
        timestamp for _, timestamp, _ in _fields(
            path, stop, exact_stop=False,
        )
    )
