"""Derive immutable series and common-calendar coverage for universe scaling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import stat

from tools.fetch_universe import UniverseManifest
from tools.files import file_sha256
from tools.panel_contract import (
    NAME, FileBinding, _directory_identity, _integer, _regular_identity,
    _string, _tree_digest,
)
from tools.session_calendar import SessionCalendar
from tools.session_samples import session_samples
from tools.universe_contract import (
    common_calendar, fixed_update_budget, pack_rows, universe_roles,
)
from tools.universe_scaling_contract import (
    CSV_ROOT, EXPECTED_BUDGETS, PHASES, SELECTION_FILES, SELECTION_ROOT,
    SELECTION_SHA256, PhaseCoverage, ScalingCoverage, SeriesCoverage,
    TreeBinding, timestamp_grid_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ScalingSeries:
    name: str
    csv: FileBinding
    rows: int


def fetch_series(
    value: Mapping[str, object], root: Path = ROOT,
) -> tuple[ScalingSeries, ...]:
    """Read ordered CSV identities only from the hash-bound fetch report."""
    raw = value.get("series")
    if not isinstance(raw, list):
        raise ValueError("fetch report series must be an array")
    series = []
    for index, record in enumerate(raw):
        label = f"fetch series[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        name = _string(record.get("ticker"), f"{label}.ticker")
        raw_csv = record.get("csv")
        if not isinstance(raw_csv, Mapping):
            raise ValueError(f"{label}.csv must be an object")
        csv = FileBinding.parse(
            {key: raw_csv.get(key) for key in ("path", "sha256")},
            f"{label}.csv", relative=False,
        )
        rows = _integer(raw_csv.get("rows"), f"{label}.rows")
        expected = root / CSV_ROOT / f"{name.lower()}-30m.csv"
        if not NAME.fullmatch(name) or Path(csv.path) != expected:
            raise ValueError(f"{label} path is outside the frozen CSV root")
        series.append(ScalingSeries(name, csv, rows))
    result = tuple(series)
    if len(result) != 55 or \
       len({item.name for item in result}) != len(result) or \
       len({item.csv.path for item in result}) != len(result):
        raise ValueError("fetch report must bind 55 ordered unique series")
    return result


def selection_paths(
    root: Path = ROOT / SELECTION_ROOT,
) -> tuple[Path, ...]:
    """List every nonsymlink regular selection member in lexical order."""
    identity = _directory_identity(root)
    absolute = Path(os.path.abspath(root))
    resolved = root.resolve(strict=True)
    if resolved != absolute:
        raise ValueError("selection package root must not be a symlink")
    paths = []
    for path in resolved.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        kind = stat.S_IFMT(metadata.st_mode)
        if kind == stat.S_IFREG:
            _regular_identity(path)
            paths.append(path)
        elif kind == stat.S_IFDIR:
            _directory_identity(path)
        else:
            raise ValueError("selection package contains a nonregular member")
    if _directory_identity(root) != identity:
        raise ValueError("selection package root changed")
    return tuple(sorted(paths))


def selection_binding(root: Path = ROOT / SELECTION_ROOT) -> TreeBinding:
    """Hash every nonsymlink regular file in the frozen selection package."""
    identity = _directory_identity(root)
    resolved = Path(os.path.abspath(root))
    files = [
        FileBinding(
            path.relative_to(resolved).as_posix(), file_sha256(path),
        )
        for path in selection_paths(root)
    ]
    files.sort(key=lambda item: item.path)
    binding = TreeBinding(
        SELECTION_ROOT.as_posix(), len(files), _tree_digest(files),
    )
    if binding != TreeBinding(
        SELECTION_ROOT.as_posix(), SELECTION_FILES, SELECTION_SHA256,
    ) or _directory_identity(root) != identity:
        raise ValueError("selection package does not match the frozen tree")
    return binding


def common_coverage(
    manifest: UniverseManifest, calendar: SessionCalendar,
    timestamps: Mapping[str, Sequence[str]],
) -> ScalingCoverage:
    """Measure train/development rows without materializing reserved targets."""
    names = tuple(item.ticker for item in manifest.series)
    universe_roles(names)
    if tuple(timestamps) != names:
        raise ValueError("coverage timestamps must follow manifest order")
    blocks = common_calendar(5_505, 2, 0.1, 12)
    phase_blocks = (*blocks.folds, blocks.holdout[:2])
    coverage = {phase: [] for phase in PHASES}
    for name in names:
        series_timestamps = timestamps[name]
        samples = session_samples(
            series_timestamps, manifest.interval_minutes, calendar,
            manifest.start, manifest.end, 17, 13, 13,
        )
        if samples.opportunities != 5_505:
            raise ValueError("series opportunity count changed")
        for phase, ranges in zip(PHASES, phase_blocks, strict=True):
            packed = pack_rows(
                samples.rows, ranges, 17, 13, 13,
            )
            train, validation = packed.counts
            if train < 1:
                raise ValueError("every series requires training rows")
            grid = tuple(
                (
                    series_timestamps[row.as_of],
                    series_timestamps[row.entry],
                    series_timestamps[row.target],
                )
                for row in packed.rows[train:]
            )
            coverage[phase].append(SeriesCoverage(
                name, train, validation, timestamp_grid_sha256(grid),
            ))
    phases = tuple(
        PhaseCoverage(phase, tuple(coverage[phase]))
        for phase in PHASES
    )
    budgets = dict(EXPECTED_BUDGETS)
    if any(
        fixed_update_budget(
            sum(train for _, train, _ in item.counts[:11]),
            budgets[item.phase].batch_size,
            budgets[item.phase].checkpoints,
        ) != budgets[item.phase]
        for item in phases
    ):
        raise ValueError("frozen core update budget changed")
    if any(
        not set(names[:11]) <= set(item.evaluable) for item in phases
    ):
        raise ValueError("core validation coverage is incomplete")
    return ScalingCoverage(phases)
