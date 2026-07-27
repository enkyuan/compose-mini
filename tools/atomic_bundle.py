"""Publish one verified directory with an exclusive atomic rename."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
import stat

from tools.files import rename_may_have_committed, rename_noreplace

Identity = tuple[int, int]


def absent(path: Path, name: str) -> None:
    if os.path.lexists(path):
        raise ValueError(f"{name} must not already exist")


def path_identity(path: Path, name: str, *, directory: bool = False) -> Identity:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{name} must be a regular path") from error
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(value.st_mode) or (not directory and value.st_nlink != 1):
        raise ValueError(f"{name} must be a regular path")
    return value.st_dev, value.st_ino


def verify_identity(
    path: Path, expected: Identity, name: str, *, directory: bool = False,
) -> None:
    if path_identity(path, name, directory=directory) != expected:
        raise ValueError(f"{name} changed during the fetch")


def without_symlinks(path: Path, name: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{name} must not traverse symlinks")
    return absolute


def entry(directory_fd: int, name: str) -> tuple[Identity, int] | None:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (value.st_dev, value.st_ino), stat.S_IFMT(value.st_mode)


def publish_directory(
    stage: Path, target: Path, verify: Callable[[], None],
) -> Identity:
    """Fsync and exclusively rename one verified directory into place."""
    if stage.parent != target.parent:
        raise ValueError("staged and final bundles must share one parent")
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        identity = path_identity(stage, "staged bundle", directory=True)
        verify()
        stage_fd = os.open(
            stage.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        failure = None
        try:
            rename_noreplace(parent_fd, stage.name, parent_fd, target.name)
        except OSError as error:
            failure = error
        source = entry(parent_fd, stage.name)
        result = entry(parent_fd, target.name)
        committed = (
            rename_may_have_committed(failure) and source is None and
            result == (identity, stat.S_IFDIR) and
            entry(parent_fd, stage.name) is None
        )
        if not committed:
            if failure is not None:
                raise failure
            raise OSError("bundle publication failed")
        os.fsync(parent_fd)
        return identity
    finally:
        os.close(parent_fd)
