"""Shared atomic-output, frozen-input, hashing, and path-boundary helpers."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO
import argparse
import hashlib
import json
import os
import re
import secrets
import tempfile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenInput:
    source: Path
    snapshot: Path
    sha256: str


@contextmanager
def freeze_inputs(paths: Sequence[Path]) -> Iterator[tuple[FrozenInput, ...]]:
    """Copy inputs once into a private directory and hash the copied bytes."""
    with tempfile.TemporaryDirectory(prefix="compose-mini-inputs-") as directory:
        frozen = []
        for index, source in enumerate(paths):
            snapshot = Path(directory) / str(index)
            digest = hashlib.sha256()
            with source.open("rb") as input_file, snapshot.open("xb") as output_file:
                while chunk := input_file.read(1 << 20):
                    output_file.write(chunk)
                    digest.update(chunk)
            frozen.append(FrozenInput(source, snapshot, digest.hexdigest()))
        yield tuple(frozen)


def verify_frozen(inputs: Sequence[FrozenInput]) -> None:
    if any(file_sha256(item.source) != item.sha256 for item in inputs):
        raise ValueError("an input changed during the command")


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


def exclusive_text(
    path: Path, write: Callable[[TextIO], None], directory_fd: int | None = None,
) -> None:
    """Publish a new text file atomically without following or replacing it."""
    if directory_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
        )
        temporary = Path(name)
    else:
        name = f".{path.name}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
            getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd,
        )
        temporary = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        if directory_fd is None:
            os.link(temporary, path)
        else:
            os.link(
                name, path.name, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd, follow_symlinks=False,
            )
    finally:
        if directory_fd is None:
            temporary.unlink(missing_ok=True)
        else:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def write_json(path: Path, value: Mapping[str, object]) -> None:
    def write(file: TextIO) -> None:
        json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    atomic_text(path, write)


def write_json_exclusive(
    path: Path, value: Mapping[str, object], directory_fd: int | None = None,
) -> None:
    def write(file: TextIO) -> None:
        json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    exclusive_text(path, write, directory_fd)


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
