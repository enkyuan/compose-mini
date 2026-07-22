"""Shared atomic-output, hashing, and path-boundary helpers."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO
import argparse
import hashlib
import json
import os
import re


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, write: Callable[[TextIO], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    def write(file: TextIO) -> None:
        json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    atomic_text(path, write)


def series_arg(value: str, pattern: re.Pattern[str]) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not pattern.fullmatch(name) or not path:
        raise argparse.ArgumentTypeError("series must be NAME=CSV")
    return name, Path(path)


def require_disjoint(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    """Reject outputs that alias an input or another output."""
    for index, output in enumerate(outputs):
        for other in (*inputs, *outputs[:index]):
            try:
                same = output.samefile(other)
            except FileNotFoundError:
                same = output.resolve() == other.resolve()
            if same:
                raise ValueError("output paths must not alias inputs or each other")
