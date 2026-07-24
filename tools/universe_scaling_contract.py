"""Define one immutable, development-only universe-scaling attempt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json

from tools.panel_contract import (
    RUN_ID, ExecutableBinding, FileBinding, SourceTree, TorchIdentity,
    _argv, _exact_json, _integer, _object, _relative, _string,
    _sha256, read_canonical_json,
)
from tools.universe_contract import (
    UpdateBudget, fixed_update_budget,
)

TRAINING_COHORTS = (11, 22, 33, 55)
TRANSFER_COHORTS = (11, 22, 33, 44)
UNSEEN_RANKS = tuple(range(45, 56))
SEEDS = (7, 19, 31, 43, 61)
MODES = ("fixed-update", "fixed-epoch")
MODELS = (
    "zero", "global_ridge", "global_mlp", "panel_transformer",
    "conditioned_panel_transformer", "local_transformer",
)
PHASES = ("fold-0", "fold-1", "calibration")
EXPECTED_MISSING = (
    ("fold-0", ("ALTR", "ZI")),
    ("fold-1", ("ALTR", "ZI", "INFA")),
    ("calibration", ("ALTR", "ZI", "FYBR", "INFA")),
)
EXPECTED_FIT_COUNT = 1_215
EXPECTED_PREDICTION_RECORDS = 14_216
COMMANDS = ("validate", "preflight", "calibrate", "analyze")
OUTPUTS = ("fits", "predictions", "summary", "outcome")
FINALIZER_PYTHON_FLAGS = ("-I", "-S", "-B")
POOLED_MODELS = ("global_mlp", "panel_transformer")
SELECTION_ROOT = Path("reports/universe-selection-20260724-06")
FETCH_PATH = Path("reports/liquid-common-55-20260724-03-fetch.json")
CALENDAR_PATH = Path(
    "universes/us-equities-core-2024-07-22_2026-07-21.json",
)
CONFIG_PATH = Path("experiments/executable-h13-universe.example.json")
CSV_ROOT = Path("data/liquid-common-55-20260724-03")
SELECTION_FILES = 77
SELECTION_SHA256 = \
    "bd9366ec5b040555e8b05ae932447b01b97d57e51832c9d5503059fc9119db24"
FETCH_SHA256 = \
    "add7222cd531103696f6835104182d175534dfbd99f99db5c30662069f20e42b"
CALENDAR_SHA256 = \
    "b1e0835a60624a67e21f7941ac00ece6c488937989560bbd4d0333afd869e5f8"
CONFIG_SHA256 = \
    "ea36e6301370fb3ae750aa96e0fc0d34052e58481f76886f5ebc24c033897454"
MANIFEST_BINDINGS = MappingProxyType({
    11: FileBinding(
        "reports/universe-selection-20260724-06/manifests/"
        "liquid-common-11.json",
        "d6a4fb9d5f6d5a96c49d5c9419de2b1e559e4d6fcfc330a31690e471635f3777",
    ),
    22: FileBinding(
        "reports/universe-selection-20260724-06/manifests/"
        "liquid-common-22.json",
        "17cf86450f8c82da84b65b4dad3386d2f1874393cc69d24c5ff20ecbb1948bf7",
    ),
    33: FileBinding(
        "reports/universe-selection-20260724-06/manifests/"
        "liquid-common-33.json",
        "55513c6c0f43081f7d9d092cf4f4bca4f298617bbb52adeb7f8315299e1a35bc",
    ),
    55: FileBinding(
        "reports/universe-coverage-20260724-01/liquid-common-55.json",
        "c4ca707f8e4f98ef72ed584ab82c1adc43fcc581433307c29afbe45bef5d8e4d",
    ),
})
EXPECTED_BUDGETS = (
    ("fold-0", UpdateBudget(34_992, 128, 100, 274, 27_400)),
    ("fold-1", UpdateBudget(41_042, 128, 100, 321, 32_100)),
    ("calibration", UpdateBudget(47_092, 128, 100, 368, 36_800)),
)


@dataclass(frozen=True, slots=True)
class EpochBudget:
    batch_size: int
    epochs: int
    patience: int


FIXED_EPOCH_BUDGET = EpochBudget(128, 100, 10)


SOURCE_PATHS = (
    "tools/__init__.py",
    "tools/analyze_universe.py",
    "tools/arm_universe_scaling.py",
    "tools/artifact_v1.py",
    "tools/backtest.py",
    "tools/chronology.py",
    "tools/data_v1.py",
    "tools/experiment.py",
    "tools/fetch_massive.py",
    "tools/fetch_universe.py",
    "tools/files.py",
    "tools/finalize_universe_scaling.py",
    "tools/float32.py",
    "tools/panel_contract.py",
    "tools/run_universe_scaling.py",
    "tools/session_calendar.py",
    "tools/session_samples.py",
    "tools/train.py",
    "tools/universe_contract.py",
    "tools/universe_scaling.py",
    "tools/universe_scaling_contract.py",
    "tools/universe_scaling_inputs.py",
)
FINALIZER_SOURCE_PATHS = (
    "tools/__init__.py",
    "tools/chronology.py",
    "tools/data_v1.py",
    "tools/files.py",
    "tools/finalize_universe_scaling.py",
    "tools/float32.py",
    "tools/panel_contract.py",
    "tools/session_calendar.py",
    "tools/session_samples.py",
    "tools/universe_contract.py",
    "tools/universe_scaling.py",
    "tools/universe_scaling_contract.py",
)


def timestamp_grid_sha256(rows: Sequence[Sequence[str]]) -> str:
    """Hash ordered ``(as_of, entry_time, target_time)`` timestamp rows."""
    grid = []
    for index, row in enumerate(rows):
        values = tuple(row) if isinstance(row, Sequence) and \
            not isinstance(row, str) else ()
        if len(values) != 3 or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"timestamp grid[{index}] is invalid")
        grid.append(values)
    payload = json.dumps(
        grid, ensure_ascii=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_protocol() -> dict[str, object]:
    """Return a fresh copy of the frozen development protocol."""
    return {
        "alignment_horizon_bars": 13,
        "batch_size": FIXED_EPOCH_BUDGET.batch_size,
        "calendar": {
            "calibration": [[0, 4_393], [4_405, 4_943]],
            "folds": [
                [[0, 3_293], [3_305, 3_843]],
                [[0, 3_843], [3_855, 4_393]],
            ],
            "opportunities": 5_505,
            "reserved_test": [4_955, 5_505],
        },
        "device": "cpu",
        "feature_set": "ohlcv",
        "finalizer_python_flags": list(FINALIZER_PYTHON_FLAGS),
        "fold_fraction": 0.1,
        "folds": 2,
        "epochs": FIXED_EPOCH_BUDGET.epochs,
        "history_bars": 17,
        "models": list(MODELS),
        "modes": list(MODES),
        "patience": FIXED_EPOCH_BUDGET.patience,
        "phases": list(PHASES),
        "prediction_schema": 2,
        "questions": ["cohort-scaling", "unseen-transfer"],
        "seeds": list(SEEDS),
        "target_horizon_bars": 13,
        "target_kind": "executable-return-v1",
        "training_cohorts": list(TRAINING_COHORTS),
        "transfer_cohorts": list(TRANSFER_COHORTS),
        "unseen_ranks": list(UNSEEN_RANKS),
        "unseen_transfer_gradient_exclusion": True,
        "validation_weighting": "stock-macro-training-cohort",
    }


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(map(_freeze, value))
    return value


def _binding(
    value: object, label: str, path: Path, sha256: str,
) -> FileBinding:
    binding = FileBinding.parse(value, label)
    if binding != FileBinding(path.as_posix(), sha256):
        raise ValueError(f"{label} does not match the frozen benchmark")
    return binding


def expected_scaling_commands(
    attempt_path: Path, outputs: Mapping[str, str],
) -> Mapping[str, tuple[str, ...]]:
    if set(outputs) != set(OUTPUTS):
        raise ValueError("scaling outputs are invalid")
    runner = "tools/run_universe_scaling.py"
    attempt = attempt_path.as_posix()
    return {
        **{
            stage: (runner, stage, attempt)
            for stage in COMMANDS
        },
        "finalizer_prefix": (
            "tools/finalize_universe_scaling.py", attempt,
            outputs["outcome"],
        ),
    }


@dataclass(frozen=True, slots=True)
class FitJob:
    kind: str
    mode: str | None
    cohort: int | None
    phase: str
    model: str
    seed: int | None
    members: tuple[str, ...]


def _validate_fit_job(job: FitJob, master: tuple[str, ...]) -> None:
    """Reject structurally incompatible fit-axis combinations."""
    if not isinstance(job, FitJob):
        raise ValueError("fit job axes are invalid")
    cohorts = set((*TRAINING_COHORTS, *TRANSFER_COHORTS))
    cohort = type(job.cohort) is int and job.cohort in cohorts
    seed = type(job.seed) is int and job.seed in SEEDS
    pooled = (
        job.kind == "pooled" and job.mode in MODES and
        cohort and job.phase in PHASES and
        job.model in (
            *POOLED_MODELS,
            *(("conditioned_panel_transformer",)
              if job.cohort in TRAINING_COHORTS else ()),
        ) and seed and
        job.members == master[:job.cohort]
    )
    ridge = (
        job.kind == "ridge" and job.mode is None and
        cohort and job.phase in PHASES and
        job.model == "global_ridge" and job.seed is None and
        job.members == master[:job.cohort]
    )
    local = (
        job.kind == "local" and job.mode is None and
        job.cohort is None and job.phase in PHASES and
        job.model == "local_transformer" and seed and
        type(job.members) is tuple and len(job.members) == 1 and
        job.members[0] in master
    )
    if not (pooled or ridge or local):
        raise ValueError("fit job axes are invalid")


def _master_names(master: Sequence[str]) -> tuple[str, ...]:
    names = tuple(master)
    if len(names) != 55 or len(set(names)) != len(names) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("scaling master must contain 55 unique names")
    return names


def _phase_evaluable(
    master: Sequence[str], evaluable: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(evaluable) != set(PHASES):
        raise ValueError("scaling coverage phases are invalid")
    master_set = set(master)
    coverage = {}
    for phase in PHASES:
        members = tuple(evaluable[phase])
        member_set = set(members)
        if not member_set <= master_set or members != tuple(
            name for name in master if name in member_set
        ):
            raise ValueError("phase coverage must be an ordered master subset")
        coverage[phase] = members
    return coverage


def expected_fit_jobs(
    master: Sequence[str],
    evaluable: Mapping[str, Sequence[str]],
) -> tuple[FitJob, ...]:
    """Return each physical development fit exactly once."""
    names = _master_names(master)
    coverage = _phase_evaluable(names, evaluable)
    if any(
        not set(names[:11]) <= set(coverage[phase]) for phase in PHASES
    ) or not set(names[44:]) <= set(coverage["calibration"]):
        raise ValueError("required scaling coverage is incomplete")

    cohorts = tuple(sorted(set((*TRAINING_COHORTS, *TRANSFER_COHORTS))))
    jobs = []
    for mode in MODES:
        for cohort in cohorts:
            members = names[:cohort]
            for phase in PHASES:
                jobs.extend(
                    FitJob(
                        "pooled", mode, cohort, phase, model, seed, members,
                    )
                    for model in POOLED_MODELS
                    for seed in SEEDS
                )
                if cohort in TRAINING_COHORTS:
                    jobs.extend(
                        FitJob(
                            "pooled", mode, cohort, phase,
                            "conditioned_panel_transformer", seed, members,
                        )
                        for seed in SEEDS
                    )
    jobs.extend(
        FitJob("ridge", None, cohort, phase, "global_ridge", None,
               names[:cohort])
        for cohort in cohorts for phase in PHASES
    )
    jobs.extend(
        FitJob("local", None, None, phase, "local_transformer", seed, (name,))
        for phase in PHASES for name in coverage[phase] for seed in SEEDS
    )
    return tuple(jobs)


def question_uses(
    job: FitJob, master: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    """Return question/cohort views compatible with one fit identity."""
    names = _master_names(master)
    _validate_fit_job(job, names)
    if job.kind == "local":
        rank = names.index(job.members[0]) + 1
        uses = tuple(
            ("cohort-scaling", cohort)
            for cohort in TRAINING_COHORTS if rank <= cohort
        )
        return uses + (
            tuple(
                ("unseen-transfer", cohort)
                for cohort in TRANSFER_COHORTS
            ) if rank in UNSEEN_RANKS else ()
        )
    uses = (
        (("cohort-scaling", job.cohort),)
        if job.cohort in TRAINING_COHORTS else ()
    )
    if job.model != "conditioned_panel_transformer" and \
       job.cohort in TRANSFER_COHORTS:
        uses += (("unseen-transfer", job.cohort),)
    return uses


def required_prediction_series(
    job: FitJob, master: Sequence[str],
    evaluable: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return master-ordered physical prediction destinations for one fit."""
    names = _master_names(master)
    coverage = _phase_evaluable(names, evaluable)
    uses = question_uses(job, names)
    phase_members = coverage[job.phase]
    if job.kind == "local":
        return (
            (job.members[0],) if job.members[0] in phase_members else ()
        )

    destinations = set()
    for question, cohort in uses:
        destinations.update(
            names[:cohort] if question == "cohort-scaling" else
            (names[rank - 1] for rank in UNSEEN_RANKS)
        )
    return tuple(
        name for name in phase_members if name in destinations
    )


@dataclass(frozen=True, slots=True)
class ManifestBinding:
    size: int
    file: FileBinding

    @classmethod
    def parse(cls, value: object, index: int) -> ManifestBinding:
        item = _object(
            value, {"size", "path", "sha256"}, f"manifests[{index}]",
        )
        size = _integer(item["size"], f"manifests[{index}].size")
        try:
            expected = MANIFEST_BINDINGS[size]
        except KeyError as error:
            raise ValueError("manifest cohort size is invalid") from error
        binding = {name: item[name] for name in ("path", "sha256")}
        return cls(
            size, _binding(
                binding, f"manifests[{index}]",
                Path(expected.path), expected.sha256,
            ),
        )


@dataclass(frozen=True, slots=True)
class TreeBinding:
    root: str
    files: int
    sha256: str

    @classmethod
    def parse(cls, value: object) -> TreeBinding:
        item = _object(
            value, {"root", "files", "sha256"}, "selection_tree",
        )
        parsed = cls(
            _relative(item["root"], "selection_tree.root"),
            _integer(item["files"], "selection_tree.files"),
            _sha256(item["sha256"], "selection_tree.sha256"),
        )
        expected = cls(
            SELECTION_ROOT.as_posix(), SELECTION_FILES, SELECTION_SHA256,
        )
        if parsed != expected:
            raise ValueError("selection tree does not match the benchmark")
        return parsed


@dataclass(frozen=True, slots=True)
class SeriesCoverage:
    series: str
    train_rows: int
    validation_rows: int
    timestamp_sha256: str

    @classmethod
    def parse(cls, value: object, label: str) -> SeriesCoverage:
        item = _object(
            value,
            {
                "series", "train_rows", "validation_rows",
                "timestamp_sha256",
            },
            label,
        )
        return cls(
            _string(item["series"], f"{label}.series"),
            _integer(item["train_rows"], f"{label}.train_rows"),
            _integer(
                item["validation_rows"], f"{label}.validation_rows", 0,
            ),
            _sha256(item["timestamp_sha256"],
                    f"{label}.timestamp_sha256"),
        )


@dataclass(frozen=True, slots=True)
class PhaseCoverage:
    phase: str
    series: tuple[SeriesCoverage, ...]

    @property
    def counts(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (item.series, item.train_rows, item.validation_rows)
            for item in self.series
        )

    @property
    def evaluable(self) -> tuple[str, ...]:
        return tuple(
            item.series for item in self.series if item.validation_rows
        )

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            item.series for item in self.series if not item.validation_rows
        )

    @classmethod
    def parse(cls, value: object, index: int) -> PhaseCoverage:
        label = f"coverage[{index}]"
        item = _object(value, {"phase", "series"}, label)
        raw_series = item["series"]
        if not isinstance(raw_series, list) or len(raw_series) != 55:
            raise ValueError(f"{label}.series must contain 55 records")
        return cls(
            _string(item["phase"], f"{label}.phase"),
            tuple(
                SeriesCoverage.parse(
                    record, f"{label}.series[{series_index}]",
                )
                for series_index, record in enumerate(raw_series)
            ),
        )


@dataclass(frozen=True, slots=True)
class ScalingCoverage:
    phases: tuple[PhaseCoverage, ...]

    @property
    def master(self) -> tuple[str, ...]:
        return tuple(item.series for item in self.phases[0].series)

    @property
    def unseen_missing(self) -> tuple[str, ...]:
        unseen = set(self.master[44:])
        return tuple(
            name for name in self.phases[-1].missing if name in unseen
        )

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

    @classmethod
    def parse(cls, value: object) -> ScalingCoverage:
        if not isinstance(value, list) or len(value) != len(PHASES):
            raise ValueError("coverage must contain three phases")
        phases = tuple(
            PhaseCoverage.parse(item, index)
            for index, item in enumerate(value)
        )
        if tuple(item.phase for item in phases) != PHASES:
            raise ValueError("coverage phase order is invalid")
        master = tuple(item.series for item in phases[0].series)
        if len(set(master)) != len(master) or any(
            tuple(item.series for item in phase.series) != master
            for phase in phases[1:]
        ):
            raise ValueError("coverage series order is invalid")
        empty = timestamp_grid_sha256(())
        if any(
            (item.validation_rows == 0) !=
            (item.timestamp_sha256 == empty)
            for phase in phases for item in phase.series
        ):
            raise ValueError("coverage timestamp grid is inconsistent")
        missing = tuple(set(phase.missing) for phase in phases)
        if tuple(map(len, missing)) != tuple(
            len(names) for _, names in EXPECTED_MISSING
        ) or not missing[0] < missing[1] < missing[2]:
            raise ValueError("coverage miss identities are invalid")
        budgets = dict(EXPECTED_BUDGETS)
        if any(
            fixed_update_budget(
                sum(
                    item.train_rows for item in phase.series[:11]
                ),
                budgets[phase.phase].batch_size,
                budgets[phase.phase].checkpoints,
            ) != budgets[phase.phase]
            for phase in phases
        ):
            raise ValueError("coverage core update budget changed")
        coverage = cls(phases)
        coverage.require_promotable()
        jobs = expected_fit_jobs(
            master, {phase.phase: phase.evaluable for phase in phases},
        )
        if len(jobs) != EXPECTED_FIT_COUNT or len(set(jobs)) != len(jobs):
            raise ValueError("coverage fit schedule is invalid")
        return coverage


def _budgets(value: object) -> tuple[tuple[str, UpdateBudget], ...]:
    if not isinstance(value, list):
        raise ValueError("budgets must be an array")
    result = []
    for index, raw in enumerate(value):
        item = _object(
            raw,
            {
                "phase", "control_samples", "batch_size", "checkpoints",
                "updates_per_checkpoint", "total_updates",
            },
            f"budgets[{index}]",
        )
        result.append((
            _string(item["phase"], f"budgets[{index}].phase"),
            UpdateBudget(*(
                _integer(item[name], f"budgets[{index}].{name}")
                for name in (
                    "control_samples", "batch_size", "checkpoints",
                    "updates_per_checkpoint", "total_updates",
                )
            )),
        ))
    result_tuple = tuple(result)
    if result_tuple != EXPECTED_BUDGETS:
        raise ValueError("fixed-update budgets do not match the benchmark")
    return result_tuple


@dataclass(frozen=True, slots=True)
class ScalingAttempt:
    attempt_path: str
    run_id: str
    run_dir: str
    implementation_commit: str
    selection_tree: TreeBinding
    manifests: tuple[ManifestBinding, ...]
    fetch_report: FileBinding
    session_calendar: FileBinding
    config: FileBinding
    protocol: Mapping[str, object]
    budgets: tuple[tuple[str, UpdateBudget], ...]
    coverage: ScalingCoverage
    source_tree: SourceTree
    finalizer_tree: SourceTree
    primary_python: ExecutableBinding
    torch_argv: tuple[str, ...]
    torch_probe: TorchIdentity
    environment: Mapping[str, str]
    commands: Mapping[str, tuple[str, ...]]
    outputs: Mapping[str, str]

    @property
    def training_cohorts(self) -> tuple[int, ...]:
        return self.protocol["training_cohorts"]

    @property
    def transfer_cohorts(self) -> tuple[int, ...]:
        return self.protocol["transfer_cohorts"]

    @property
    def unseen_ranks(self) -> tuple[int, ...]:
        return self.protocol["unseen_ranks"]

    @classmethod
    def read(
        cls, path: Path, logical_path: Path, repository_root: Path,
    ) -> ScalingAttempt:
        value = _object(
            read_canonical_json(path),
            {
                "schema", "status", "attempt_path", "run_id", "run_dir",
                "implementation_commit", "selection_tree", "manifests",
                "fetch_report", "session_calendar", "config", "protocol",
                "budgets", "coverage", "source_tree", "finalizer_tree",
                "primary_python", "torch_argv", "torch_probe", "environment",
                "commands", "outputs",
            },
            "scaling attempt",
        )
        if _integer(value["schema"], "schema") != 1 or \
           value["status"] != "armed":
            raise ValueError("scaling attempt must be schema 1 and armed")
        attempt_path = _relative(value["attempt_path"], "attempt_path")
        if attempt_path != _relative(
            logical_path.as_posix(), "logical attempt path",
        ):
            raise ValueError("attempt path does not match the manifest")
        run_id = _string(value["run_id"], "run_id")
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id is invalid")
        run_dir = _relative(value["run_dir"], "run_dir")
        if (
            attempt_path,
            run_dir,
        ) != (
            f"experiments/{run_id}-attempt.json",
            f"reports/{run_id}",
        ):
            raise ValueError("attempt paths do not match run_id")
        commit = _string(value["implementation_commit"],
                         "implementation_commit")
        if len(commit) != 40 or any(
            byte not in "0123456789abcdef" for byte in commit
        ):
            raise ValueError("implementation_commit is invalid")

        selection = TreeBinding.parse(value["selection_tree"])
        raw_manifests = value["manifests"]
        if not isinstance(raw_manifests, list):
            raise ValueError("manifests must be an array")
        manifests = tuple(
            ManifestBinding.parse(item, index)
            for index, item in enumerate(raw_manifests)
        )
        if tuple(item.size for item in manifests) != TRAINING_COHORTS:
            raise ValueError("manifest order is invalid")

        fetch = _binding(
            value["fetch_report"], "fetch_report", FETCH_PATH, FETCH_SHA256,
        )
        calendar = _binding(
            value["session_calendar"], "session_calendar",
            CALENDAR_PATH, CALENDAR_SHA256,
        )
        config = _binding(
            value["config"], "config", CONFIG_PATH, CONFIG_SHA256,
        )
        if not _exact_json(value["protocol"], expected_protocol()):
            raise ValueError("protocol does not match the frozen benchmark")
        protocol = _freeze(value["protocol"])
        budgets = _budgets(value["budgets"])
        coverage = ScalingCoverage.parse(value["coverage"])
        source = SourceTree.parse(
            value["source_tree"], "source_tree", SOURCE_PATHS,
        )
        finalizer = SourceTree.parse(
            value["finalizer_tree"], "finalizer_tree",
            FINALIZER_SOURCE_PATHS,
        )
        expected_root = str(repository_root.resolve(strict=True))
        if source.root != expected_root or finalizer.root != expected_root:
            raise ValueError("source roots do not match the repository")

        primary = ExecutableBinding.parse(
            value["primary_python"], "primary_python",
        )
        torch = TorchIdentity.parse(value["torch_probe"])
        torch_argv = _argv(value["torch_argv"], "torch_argv")
        if torch_argv != (torch.python.path,):
            raise ValueError("torch_argv must name the bound Torch Python")
        environment = _object(
            value["environment"],
            {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"},
            "environment",
        )
        if environment != {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
        }:
            raise ValueError("scaling environment is invalid")

        raw_outputs = _object(value["outputs"], set(OUTPUTS), "outputs")
        outputs = {
            name: _relative(raw_outputs[name], f"outputs.{name}")
            for name in OUTPUTS
        }
        expected_paths = {
            "fits": f"{run_dir}/fits.jsonl",
            "predictions": f"{run_dir}/predictions.jsonl",
            "summary": f"{run_dir}/summary.json",
            "outcome": f"experiments/{run_id}-outcome.json",
        }
        if any(outputs[name] != expected for name, expected in
               expected_paths.items()) or len({
                   Path(item).resolve(strict=False)
                   for item in outputs.values()
               }) != len(outputs):
            raise ValueError("scaling output paths are invalid")
        inputs = (
            SELECTION_ROOT, FETCH_PATH, CALENDAR_PATH, CONFIG_PATH,
            *(Path(item.file.path) for item in manifests),
        )
        generated = (
            Path(attempt_path), Path(run_dir),
            *(Path(item) for item in outputs.values()),
        )
        if any(
            output == source or output in source.parents or
            source in output.parents
            for output in generated for source in inputs
        ):
            raise ValueError("scaling outputs overlap frozen inputs")

        raw_commands = _object(
            value["commands"], {*COMMANDS, "finalizer_prefix"}, "commands",
        )
        commands = {
            name: _argv(raw_commands[name], f"commands.{name}")
            for name in (*COMMANDS, "finalizer_prefix")
        }
        if commands != expected_scaling_commands(
            Path(attempt_path), outputs,
        ):
            raise ValueError("scaling commands are invalid")
        return cls(
            attempt_path, run_id, run_dir, commit, selection, manifests,
            fetch, calendar, config, protocol, budgets, coverage, source,
            finalizer, primary, torch_argv, torch,
            MappingProxyType(dict(environment)),
            MappingProxyType(commands), MappingProxyType(outputs),
        )
