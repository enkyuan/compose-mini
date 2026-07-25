"""Shared atomic-output, frozen-input, hashing, and path-boundary helpers."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TextIO
import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
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
    snapshot_identity: tuple[int, int] | None = field(
        default=None, compare=False, repr=False,
    )


@dataclass(frozen=True)
class ExclusiveTemp:
    name: str
    identity: tuple[int, int]
    mode: int


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
            snapshot.chmod(stat.S_IRUSR)
            metadata = snapshot.stat(follow_symlinks=False)
            frozen.append(FrozenInput(
                source, snapshot, digest.hexdigest(),
                (metadata.st_dev, metadata.st_ino),
            ))
        yield tuple(frozen)


def verify_frozen(inputs: Sequence[FrozenInput]) -> None:
    def changed(item: FrozenInput) -> bool:
        if file_sha256(item.source) != item.sha256:
            return True
        if item.snapshot_identity is None:
            return False
        try:
            metadata = item.snapshot.stat(follow_symlinks=False)
            return not stat.S_ISREG(metadata.st_mode) or \
                (metadata.st_dev, metadata.st_ino) != \
                item.snapshot_identity or metadata.st_nlink != 1 or \
                metadata.st_mode & 0o222 != 0 or \
                file_sha256(item.snapshot) != item.sha256
        except OSError:
            return True

    if any(map(changed, inputs)):
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


def _entry_state(
    directory_fd: int,
    name: str,
) -> tuple[tuple[int, int], int, int] | None:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (
        (value.st_dev, value.st_ino),
        stat.S_IFMT(value.st_mode),
        value.st_nlink,
    )


def _owns_entry(
    directory_fd: int,
    binding: ExclusiveTemp,
    links: tuple[int, ...],
) -> bool:
    state = _entry_state(directory_fd, binding.name)
    return state is not None and state[:2] == (
        binding.identity, stat.S_IFREG,
    ) and state[2] in links


def _require_entry(
    directory_fd: int,
    binding: ExclusiveTemp,
    links: tuple[int, ...],
) -> None:
    if not _owns_entry(directory_fd, binding, links):
        raise OSError("exclusive output changed during publication")


def rename_noreplace(
    source_fd: int,
    source: str,
    target_fd: int,
    target: str,
) -> None:
    """Atomically rename without replacing an existing target."""
    names = {
        "darwin": ("renameatx_np", 0x4),
        "linux": ("renameat2", 0x1),
    }
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    if platform not in names:
        raise OSError(errno.ENOTSUP, "exclusive rename is unsupported")
    symbol, flag = names[platform]
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), symbol)
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP, "exclusive rename is unsupported"
        ) from error
    function.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    if function(
        source_fd, os.fsencode(source), target_fd, os.fsencode(target), flag,
    ):
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code), target)


def rename_may_have_committed(error: OSError | None) -> bool:
    return error is None or error.errno in (None, errno.EINTR, errno.EIO)


def exclusive_text(
    path: Path, write: Callable[[TextIO], None], directory_fd: int | None = None,
    before_link: Callable[[], None] | None = None,
    *,
    before_link_with_temp: Callable[[ExclusiveTemp], None] | None = None,
    on_temp_created: Callable[[ExclusiveTemp], None] | None = None,
) -> None:
    """Fsync, revalidate, then rename a writer-bound inode without replacing."""
    if before_link is not None and before_link_with_temp is not None:
        raise ValueError("exclusive output accepts one pre-link callback")
    owned_directory = directory_fd is None
    descriptor: int | None = None
    if directory_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
    try:
        name = f".{path.name}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
            getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        binding = ExclusiveTemp(
            name, (metadata.st_dev, metadata.st_ino),
            stat.S_IMODE(metadata.st_mode),
        )
        if on_temp_created is not None:
            on_temp_created(binding)
        with os.fdopen(
            descriptor, "w", encoding="utf-8", closefd=False,
        ) as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
            _require_entry(directory_fd, binding, (1,))
            if before_link_with_temp is not None:
                before_link_with_temp(binding)
            elif before_link is not None:
                before_link()
            _require_entry(directory_fd, binding, (1,))
            failure: OSError | None = None
            try:
                rename_noreplace(
                    directory_fd, binding.name, directory_fd, path.name,
                )
            except OSError as error:
                failure = error
            source = _entry_state(directory_fd, binding.name)
            target = _entry_state(directory_fd, path.name)
            committed = rename_may_have_committed(failure) and \
                source is None and target == (
                binding.identity, stat.S_IFREG, 1,
            ) and _entry_state(directory_fd, binding.name) is None
            if not committed:
                if failure is not None:
                    raise failure
                raise OSError("exclusive output changed during publication")
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            if owned_directory:
                os.close(directory_fd)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    def write(file: TextIO) -> None:
        json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    atomic_text(path, write)


def write_json_exclusive(
    path: Path, value: Mapping[str, object], directory_fd: int | None = None,
    before_link: Callable[[], None] | None = None,
    *,
    before_link_with_temp: Callable[[ExclusiveTemp], None] | None = None,
    on_temp_created: Callable[[ExclusiveTemp], None] | None = None,
) -> None:
    def write(file: TextIO) -> None:
        json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    exclusive_text(
        path, write, directory_fd, before_link,
        before_link_with_temp=before_link_with_temp,
        on_temp_created=on_temp_created,
    )


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
