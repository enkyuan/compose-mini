"""Derive immutable series and common-calendar coverage for universe scaling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import os
import stat

from tools.backtest import Bars
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
    SELECTION_SHA256, TreeBinding,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MISSING = (
    ("fold-0", ("ALTR", "ZI", "ENLC")),
    ("fold-1", ("ALTR", "ZI", "INFA", "ENLC")),
    ("calibration", ("ALTR", "ZI", "FYBR", "INFA", "ENLC")),
)


@dataclass(frozen=True, slots=True)
class ScalingSeries:
    name: str
    csv: FileBinding
    rows: int


@dataclass(frozen=True, slots=True)
class PhaseCoverage:
    phase: str
    counts: tuple[tuple[str, int, int], ...]
    evaluable: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScalingCoverage:
    phases: tuple[PhaseCoverage, ...]
    unseen_missing: tuple[str, ...]

    @property
    def promotable(self) -> bool:
        return tuple(item.phase for item in self.phases) == PHASES and \
            not self.unseen_missing

    def require_promotable(self) -> None:
        if tuple(item.phase for item in self.phases) != PHASES:
            raise ValueError("scaling coverage phases are incomplete")
        if not self.promotable:
            raise ValueError(
                "unseen calibration coverage is incomplete: " +
                ", ".join(self.unseen_missing)
            )


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
    bars: Mapping[str, Bars],
) -> ScalingCoverage:
    """Measure train/development rows without materializing reserved targets."""
    names = tuple(item.ticker for item in manifest.series)
    if tuple(bars) != names:
        raise ValueError("coverage bars must follow manifest order")
    blocks = common_calendar(5_505, 2, 0.1, 12)
    phase_blocks = (*blocks.folds, blocks.holdout[:2])
    counts = {phase: [] for phase in PHASES}
    for name in names:
        samples = session_samples(
            bars[name].timestamps, manifest.interval_minutes, calendar,
            manifest.start, manifest.end, 17, 13, 13,
        )
        if samples.opportunities != 5_505:
            raise ValueError("series opportunity count changed")
        for phase, ranges in zip(PHASES, phase_blocks, strict=True):
            train, validation = pack_rows(
                samples.rows, ranges, 17, 13, 13,
            ).counts
            if train < 1:
                raise ValueError("every series requires training rows")
            counts[phase].append((name, train, validation))
    phases = tuple(
        PhaseCoverage(
            phase, tuple(counts[phase]),
            tuple(name for name, _, validation in counts[phase]
                  if validation),
            tuple(name for name, _, validation in counts[phase]
                  if not validation),
        )
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
    if tuple((item.phase, item.missing) for item in phases) != \
            EXPECTED_MISSING or any(
                not set(names[:11]) <= set(item.evaluable) for item in phases
            ):
        raise ValueError("frozen validation coverage changed")
    unseen = set(universe_roles(names).unseen)
    calibration = phases[-1]
    return ScalingCoverage(
        phases, tuple(name for name in calibration.missing if name in unseen),
    )
