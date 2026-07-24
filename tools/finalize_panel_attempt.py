#!/usr/bin/env python3
"""Publish one immutable terminal outcome for an armed panel attempt."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.files import (
    FrozenInput, file_sha256, freeze_inputs, verify_frozen,
    write_json_exclusive,
)
from tools.panel_contract import (
    FINALIZER_SOURCE_PATHS, FileBinding, PanelAttempt, PanelInputs, SourceTree,
    _directory_identity, _open_directory, _regular_identity, _regular_inputs,
    _verify_identities, expected_panel_commands, panel_profile,
    read_canonical_json, selected_source_tree, source_tree,
    validate_panel_analysis,
)

TRANSITIONS = {
    "preflight-failure": ("preflight", lambda code: code != 0),
    "setup-failure": ("setup", lambda code: code != 0),
    "experiment-failure": ("experiment", lambda code: code != 0),
    "analysis-integrity-failure": (
        "analysis", lambda code: code not in (0, 3),
    ),
    "gate-failure": ("analysis", lambda code: code == 3),
    "pass": ("analysis", lambda code: code == 0),
}
PROVENANCE_STATUSES = frozenset(("gate-failure", "pass"))
INPUT_FIELDS = {
    "run_id", "attempt", "input_manifest", "config", "baseline_report",
    "baseline_ledger", "experiment_report", "calibration_ledger", "series",
}
ANALYSIS_FIELDS = {
    "schema", "status", "inputs", "protocol", "validation", "calibration",
    "gates",
}


@dataclass(frozen=True)
class OutputState:
    name: str
    path: Path
    identity: tuple[int, int] | None
    parent_identity: tuple[int, int] | None

    @property
    def present(self) -> bool:
        return self.identity is not None


def _timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return parsed


def _transition(stage: str, code: int, status: str) -> None:
    if type(code) is not int or not 0 <= code <= 255 or \
       status not in TRANSITIONS:
        raise ValueError("terminal transition is invalid")
    expected_stage, accepts = TRANSITIONS[status]
    if stage != expected_stage or not accepts(code):
        raise ValueError("terminal stage, exit, and status disagree")


def _clean(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute != path.resolve(strict=False):
        raise ValueError(f"{label} must not contain a symlink or alias")
    return absolute


def _observe(name: str, path: Path) -> OutputState:
    _clean(path, f"{name} output")
    try:
        parent_identity = _directory_identity(path.parent)
    except ValueError:
        if os.path.lexists(path.parent):
            raise
        parent_identity = None
    try:
        identity = _regular_identity(path)
    except ValueError:
        if os.path.lexists(path):
            raise
        identity = None
    return OutputState(name, path, identity, parent_identity)


def _binding_record(binding: FileBinding) -> dict[str, str]:
    return {"path": binding.path, "sha256": binding.sha256}


def _matches(check: Callable[[], None]) -> bool:
    try:
        check()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _binding_matches(binding: FileBinding) -> None:
    path = Path(binding.path)
    _regular_identity(path)
    if file_sha256(path) != binding.sha256:
        raise ValueError("bound file changed")


def _tree_matches(tree: SourceTree) -> None:
    for binding in tree.files:
        _binding_matches(FileBinding(
            str(Path(tree.root) / binding.path), binding.sha256,
        ))


def _torch_tree_matches(attempt: PanelAttempt) -> None:
    _require_equal(
        source_tree(Path(attempt.torch_probe.package_tree.root)),
        attempt.torch_probe.package_tree,
        "Torch package tree",
    )


def _series_match(attempt: PanelAttempt) -> None:
    _binding_matches(attempt.input_manifest)
    inputs = PanelInputs.read(Path(attempt.input_manifest.path))
    for item in inputs.series:
        _binding_matches(item.csv)


def _broader_integrity(attempt: PanelAttempt) -> dict[str, bool]:
    torch = attempt.torch_probe
    return {
        "input_manifest": _matches(
            lambda: _binding_matches(attempt.input_manifest)
        ),
        "config": _matches(lambda: _binding_matches(attempt.config)),
        "baseline_report": _matches(
            lambda: _binding_matches(attempt.baseline_report)
        ),
        "baseline_ledger": _matches(
            lambda: _binding_matches(attempt.baseline_ledger)
        ),
        "series": _matches(lambda: _series_match(attempt)),
        "source_tree": _matches(lambda: _tree_matches(attempt.source_tree)),
        "uv": _matches(lambda: attempt.uv.validate_live("uv")),
        "torch_python": _matches(
            lambda: _binding_matches(torch.python)
        ),
        "torch_package_tree": _matches(
            lambda: _torch_tree_matches(attempt)
        ),
    }


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed")


def _trusted_paths(attempt: PanelAttempt) -> tuple[Path, ...]:
    if Path(attempt.finalizer_tree.root) != ROOT.resolve(strict=True):
        raise ValueError("trusted finalizer root is invalid")
    if attempt.finalizer_tree != selected_source_tree(
        ROOT, FINALIZER_SOURCE_PATHS,
    ):
        raise ValueError("trusted finalizer closure is invalid")
    return tuple(
        Path(attempt.finalizer_tree.root) / item.path
        for item in attempt.finalizer_tree.files
    )


def _validate_trusted(
    attempt: PanelAttempt, by_path: Mapping[Path, FrozenInput],
) -> None:
    for binding in attempt.finalizer_tree.files:
        path = Path(attempt.finalizer_tree.root) / binding.path
        if by_path[path].sha256 != binding.sha256:
            raise ValueError("trusted finalizer closure changed")
    executable = Path(attempt.primary_python.path)
    if executable.resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ValueError("finalizer is not running under bound primary Python")
    attempt.primary_python.validate_live("primary Python")
    if by_path[executable].sha256 != attempt.primary_python.sha256:
        raise ValueError("bound primary Python changed")


def _success_paths(attempt: PanelAttempt, inputs: PanelInputs
                   ) -> tuple[Path, ...]:
    return tuple(dict.fromkeys((
        Path(attempt.input_manifest.path), Path(attempt.config.path),
        Path(attempt.baseline_report.path), Path(attempt.baseline_ledger.path),
        *(Path(item.csv.path) for item in inputs.series),
        *(
            Path(attempt.source_tree.root) / item.path
            for item in attempt.source_tree.files
        ),
        Path(attempt.uv.path), Path(attempt.torch_probe.python.path),
    )))


def _validate_success_inputs(
    attempt: PanelAttempt, inputs: PanelInputs,
    by_path: Mapping[Path, FrozenInput],
) -> None:
    for binding in (
        attempt.input_manifest, attempt.config, attempt.baseline_report,
        attempt.baseline_ledger, *(item.csv for item in inputs.series),
    ):
        binding.validate(by_path[Path(binding.path)], "successful input")
    for binding in attempt.source_tree.files:
        path = Path(attempt.source_tree.root) / binding.path
        if by_path[path].sha256 != binding.sha256:
            raise ValueError("successful source closure changed")
    for binding in (attempt.uv, attempt.torch_probe.python):
        if by_path[Path(binding.path)].sha256 != binding.sha256:
            raise ValueError("successful runtime binary changed")
    attempt.uv.validate_live("uv")
    if source_tree(Path(attempt.torch_probe.package_tree.root)) != \
            attempt.torch_probe.package_tree:
        raise ValueError("successful Torch package tree changed")


def _output_records(
    states: Sequence[OutputState],
    by_path: Mapping[Path, FrozenInput],
) -> dict[str, object]:
    return {
        state.name: {
            "path": str(state.path),
            "state": "present" if state.present else "absent",
            "sha256": (
                by_path[state.path].sha256 if state.present else None
            ),
        }
        for state in states
    }


def _verify_states(
    states: Sequence[OutputState],
    frozen: Sequence[FrozenInput],
) -> None:
    verify_frozen(frozen)
    for state in states:
        _clean(state.path, f"{state.name} output")
        if state.parent_identity is None:
            if os.path.lexists(state.path.parent):
                raise ValueError("an absent output directory appeared")
        elif _directory_identity(state.path.parent) != state.parent_identity:
            raise ValueError("declared output directory changed")
        if state.present:
            if _regular_identity(state.path) != state.identity:
                raise ValueError("declared output identity changed")
        elif os.path.lexists(state.path) or \
                os.path.lexists(state.path.resolve(strict=False)):
            raise ValueError("an absent declared output appeared")


def _validate_provenance(
    attempt_path: Path, attempt_input: FrozenInput, attempt: PanelAttempt,
    inputs: PanelInputs, states: Mapping[str, OutputState],
    frozen: Mapping[Path, FrozenInput], status: str,
) -> None:
    report_state = states["experiment_report"]
    ledger_state = states["calibration_ledger"]
    analysis_state = states["analysis_report"]
    if not all((report_state.present, ledger_state.present,
                analysis_state.present)):
        raise ValueError("successful outcomes require all benchmark outputs")
    attempt_record = {
        "path": str(attempt_path), "sha256": attempt_input.sha256,
    }
    report = read_canonical_json(frozen[report_state.path].snapshot)
    if report.get("attempt_manifest") != {
        **attempt_record, "run_id": attempt.run_id,
    } or report.get("input_manifest") != \
            _binding_record(attempt.input_manifest):
        raise ValueError("experiment report provenance is invalid")
    expected_inputs: dict[str, object] = {
        "run_id": attempt.run_id,
        "attempt": attempt_record,
        "input_manifest": _binding_record(attempt.input_manifest),
        "config": _binding_record(attempt.config),
        "baseline_report": _binding_record(attempt.baseline_report),
        "baseline_ledger": _binding_record(attempt.baseline_ledger),
        "experiment_report": {
            "path": str(report_state.path),
            "sha256": frozen[report_state.path].sha256,
        },
        "calibration_ledger": {
            "path": str(ledger_state.path),
            "sha256": frozen[ledger_state.path].sha256,
        },
        "series": [
            {"name": item.name, **_binding_record(item.csv)}
            for item in inputs.series
        ],
    }
    analysis = read_canonical_json(frozen[analysis_state.path].snapshot)
    config = read_canonical_json(frozen[Path(attempt.config.path)].snapshot)
    profile = panel_profile(config)
    if attempt.expected_equivalent_runs != profile.expected_runs or \
       attempt.expected_panel_fits != profile.expected_panel_fits or \
       dict(attempt.commands) != expected_panel_commands(
           attempt_path, attempt.input_manifest.path, attempt.config.path,
           attempt.baseline_report.path, attempt.baseline_ledger.path,
           attempt.outputs, inputs, profile,
       ):
        raise ValueError("attempt profile is invalid")
    validate_panel_analysis(analysis, profile)
    gates = analysis["gates"]
    analysis_inputs = analysis.get("inputs")
    if set(analysis) != ANALYSIS_FIELDS or \
       gates["all_pass"] != (status == "pass") or \
       not isinstance(analysis_inputs, Mapping) or \
       set(analysis_inputs) != INPUT_FIELDS or \
       analysis_inputs != expected_inputs or \
       analysis.get("status") != status:
        raise ValueError("analysis provenance is invalid")


def _finalizer_argv(
    attempt: PanelAttempt, started: str, ended: str,
    stage: str, code: int, status: str,
) -> None:
    expected = (
        *attempt.commands["finalizer_prefix"],
        "--started", started, "--ended", ended, "--stage", stage,
        "--exit", str(code), "--status", status,
    )
    if tuple(sys.argv) != expected:
        raise ValueError("finalizer arguments do not match the armed attempt")


def finalize(
    attempt_path: Path, outcome: Path, started: str, ended: str,
    stage: str, code: int, status: str,
) -> dict[str, object]:
    """Validate and publish one terminal outcome without replacing a file."""
    _transition(stage, code, status)
    if _timestamp(ended, "ended") < _timestamp(started, "started"):
        raise ValueError("ended must not precede started")
    _clean(attempt_path, "attempt")
    discovery_identity = _regular_inputs((attempt_path,))
    with freeze_inputs((attempt_path,)) as discovery:
        attempt = PanelAttempt.read(discovery[0].snapshot)
        if str(outcome) != attempt.outputs["outcome"]:
            raise ValueError("outcome path does not match the armed attempt")
        _finalizer_argv(attempt, started, ended, stage, code, status)
        declared = tuple(
            (name, Path(path)) for name, path in attempt.outputs.items()
        )
        paths = (attempt_path, *(path for _, path in declared))
        resolved = tuple(_clean(path, "declared path") for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("attempt and output paths must be disjoint")
        states = tuple(_observe(name, path) for name, path in declared)
        state_by_name = {state.name: state for state in states}
        if state_by_name["outcome"].present:
            raise ValueError("outcome must be fresh and absent")
        inputs = None
        if status in PROVENANCE_STATUSES:
            with freeze_inputs((Path(attempt.input_manifest.path),)) as frozen:
                inputs = PanelInputs.read(frozen[0].snapshot)
                verify_frozen(frozen)
        trusted = _trusted_paths(attempt)
        present = tuple(state.path for state in states if state.present)
        success = _success_paths(attempt, inputs) if inputs is not None else ()
        sources = tuple(dict.fromkeys((
            attempt_path, *trusted, Path(attempt.primary_python.path),
            *present, *success,
        )))
        identities = _regular_inputs(sources)
        _verify_identities(discovery_identity)
        parent_fd, parent_identity = _open_directory(outcome.parent)
        try:
            with freeze_inputs(sources) as frozen:
                verify_frozen(discovery)
                by_path = dict(zip(sources, frozen, strict=True))
                frozen_attempt = PanelAttempt.read(
                    by_path[attempt_path].snapshot,
                )
                if frozen_attempt != attempt:
                    raise ValueError("attempt changed during finalization")
                _validate_trusted(attempt, by_path)
                if inputs is not None:
                    _validate_success_inputs(attempt, inputs, by_path)
                    _validate_provenance(
                        attempt_path, by_path[attempt_path], attempt, inputs,
                        state_by_name, by_path, status,
                    )
                broader = _broader_integrity(attempt)
                if status in PROVENANCE_STATUSES and \
                   not all(broader.values()):
                    raise ValueError("successful broader integrity changed")
                result = {
                    "schema": 1,
                    "attempt": {
                        "path": str(attempt_path),
                        "sha256": by_path[attempt_path].sha256,
                        "run_id": attempt.run_id,
                    },
                    "started": started, "ended": ended,
                    "stage": stage, "exit": code, "status": status,
                    "outputs": _output_records(states, by_path),
                    "integrity": {
                        "trusted_finalizer_tree":
                            attempt.finalizer_tree.sha256,
                        "primary_python": _binding_record(
                            attempt.primary_python,
                        ),
                        "broader": broader,
                    },
                }
                def verify() -> None:
                    _verify_identities(identities)
                    if _directory_identity(outcome.parent) != parent_identity:
                        raise ValueError("outcome directory changed")
                    _verify_states(states, frozen)
                    if status in PROVENANCE_STATUSES:
                        _torch_tree_matches(attempt)

                write_json_exclusive(
                    outcome, result, parent_fd, verify,
                )
                os.fsync(parent_fd)
                return result
        finally:
            os.close(parent_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("outcome", type=Path)
    parser.add_argument("--started", required=True)
    parser.add_argument("--ended", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exit", dest="exit_code", required=True, type=int)
    parser.add_argument("--status", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        result = finalize(
            args.attempt, args.outcome, args.started, args.ended,
            args.stage, args.exit_code, args.status,
        )
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "outcome": str(args.outcome), "status": result["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
