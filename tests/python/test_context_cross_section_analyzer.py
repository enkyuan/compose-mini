#!/usr/bin/env python3
"""Verify authenticated context stock-selection analysis glue."""

from array import array
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_context_diagnostic_finalizer import MASTER, phase_for
import tools.analyze_context_cross_section as analyzer
from tools.analyze_context_cross_section import (
    _common_groups, _phase_truth, _publish, _report,
)
from tools.context_diagnostic_contract import context_phase_sha256
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.files import file_sha256, freeze_inputs, write_json
from tools.panel_contract import FileBinding, _directory_identity
from tools.run_context_diagnostic import (
    context_access_value, phase_artifacts,
)
from tools.session_samples import SampleRows
from tools.universe_contract import PackedRows

TIMESTAMPS = (
    "2026-01-02T14:30:00Z", "2026-01-02T15:00:00Z",
    "2026-01-02T15:30:00Z", "2026-01-02T16:00:00Z",
    "2026-01-02T16:30:00Z", "2026-01-02T17:00:00Z",
)
ROWS = (
    SampleRows(0, 1, 2, 0),
    SampleRows(3, 4, 5, 1),
)


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (OSError, ValueError):
        return
    raise AssertionError("expected ValueError")


def test_common_groups_use_only_complete_timestamp_cells() -> None:
    phase = phase_for()
    names = tuple(series for series, _, _ in phase.evaluation_rows)
    timestamps = dict.fromkeys(names, TIMESTAMPS)
    packed = {
        name: PackedRows(
            ROWS[1:] if name == names[-1] else ROWS,
            (0, 1 if name == names[-1] else 2),
        )
        for name in names
    }
    expected = ((
        TIMESTAMPS[3], TIMESTAMPS[4], TIMESTAMPS[5],
    ),)
    assert _common_groups(phase, timestamps, packed) == expected
    packed[names[-1]] = PackedRows((), (0, 0))
    raises(_common_groups, phase, timestamps, packed)


def test_report_keeps_analysis_non_executable() -> None:
    contracts = (phase_for(), phase_for("calibration"))
    names = sorted(
        series for series, _, _ in contracts[0].evaluation_rows
    )
    diagnostic = {
        "date_count": 20,
        "effective_breadth": {
            "excluded": [],
            "included": names,
            "reason": None,
            "value": 1.0,
        },
        "eligible_spearman_groups": 20,
        "excluded_spearman_groups": 0,
        "group_count": 20,
        "intervals": {
            "5": (-0.2, -0.1),
            "10": (-0.2, -0.1),
            "20": (-0.2, -0.1),
        },
        "mean_spearman": 1.0,
        "meets_statistical_gate": True,
        "paired_mean": -0.1,
        "r2": 0.5,
        "raw_breadth": 11,
    }
    results = tuple({
        "diagnostic": diagnostic,
        "evidence_role": "development-post-hoc-not-forward-clean",
        "group_grid_sha256": str(index) * 64,
        "history": 17,
        "model": "panel_transformer",
        "phase": phase,
        "phase_sha256": context_phase_sha256(contract),
        "schema": 1,
    } for index, (phase, contract) in enumerate(zip(
        ("fold-1", "calibration"), contracts, strict=True,
    )))
    inputs = {
        "analysis_sources": [{
            "path": path,
            "sha256": "a" * 64,
        } for path in (
            "tools/analyze_context_cross_section.py",
            "tools/context_cross_section.py",
            "tools/universe_cross_section.py",
        )],
        "attempt": {
            "path": "experiments/attempt.json",
            "sha256": "b" * 64,
        },
        "outcome": {
            "path": "reports/run/outcome.json",
            "sha256": "c" * 64,
        },
    }
    report = _report(inputs, results, contracts)
    assert set(report) == {
        "evidence_role", "inputs", "locks", "phases", "schema",
    }
    assert report["locks"] == {
        "backtest_run": False,
        "forward_clean": False,
        "trading_authorized": False,
        "universe_expansion_authorized": False,
    }
    assert tuple(item["phase"] for item in report["phases"]) == (
        "fold-1", "calibration",
    )
    raises(_report, report["inputs"], tuple(reversed(results)), contracts)
    poisoned = deepcopy(results)
    poisoned[0]["diagnostic"]["raw_bars"] = [1.0]
    raises(_report, inputs, poisoned, contracts)
    poisoned = deepcopy(results)
    poisoned[0]["diagnostic"]["effective_breadth"]["included"][0] = \
        "100.0,101.0,102.0 raw rows"
    raises(_report, inputs, poisoned, contracts)
    contradictory = deepcopy(results)
    contradictory[0]["diagnostic"]["r2"] = -1.0
    raises(_report, inputs, contradictory, contracts)
    impossible = deepcopy(results)
    impossible[0]["diagnostic"]["date_count"] = 2
    raises(_report, inputs, impossible, contracts)
    impossible = deepcopy(results)
    impossible[0]["diagnostic"]["eligible_spearman_groups"] = 0
    impossible[0]["diagnostic"]["excluded_spearman_groups"] = 20
    raises(_report, inputs, impossible, contracts)
    malformed = deepcopy(results)
    malformed[0]["schema"] = True
    raises(_report, inputs, malformed, contracts)
    bad_inputs = deepcopy(inputs)
    bad_inputs["attempt"]["raw_csv"] = []
    raises(_report, bad_inputs, results, contracts)


def test_imported_entrypoint_cannot_publish() -> None:
    code = (
        "import sys;"
        f"sys.path.append({str(ROOT)!r});"
        "from pathlib import Path;"
        "from tools.analyze_context_cross_section import "
        "analyze_context_cross_section;"
        "analyze_context_cross_section(Path('missing.json'))"
    )
    result = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", code),
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "context analysis requires" in result.stderr


def test_group_grid_precedes_truth_reads() -> None:
    phase = phase_for()
    names = tuple(series for series, _, _ in phase.evaluation_rows)
    timestamps = dict.fromkeys(names, TIMESTAMPS[:3])
    packed = dict.fromkeys(
        names, PackedRows((SampleRows(0, 1, 2, 0),), (0, 1)),
    )
    events = []

    class Lease:
        snapshots = SimpleNamespace(csv=tuple(
            (name, SimpleNamespace(snapshot=Path(name)))
            for name in names
        ))

        def __call__(self) -> None:
            events.append("lease")

    def phase_rows(*args: object) -> tuple[object, object]:
        events.append("phase_rows")
        return timestamps, packed

    def groups(*args: object) -> object:
        events.append("groups")
        return _common_groups(*args)

    def bars(*args: object) -> array:
        events.append("bars")
        return array("f", [1.0] * 15)

    with patch(
        "tools.analyze_context_cross_section._common_groups", new=groups,
    ), patch(
        "tools.analyze_context_cross_section.context_bar_prefix", new=bars,
    ):
        truth, common = _phase_truth(
            SimpleNamespace(), phase, Lease(),
            SimpleNamespace(target_kind=EXECUTABLE_RETURN_TARGET),
            SimpleNamespace(), SimpleNamespace(), phase_rows,
        )
    assert common == ((TIMESTAMPS[0], TIMESTAMPS[1], TIMESTAMPS[2]),)
    assert tuple(truth) == names
    assert events[:3] == ["phase_rows", "groups", "lease"]
    assert events[3:-1] == ["bars"] * len(names)
    assert events[-1] == "lease"


def test_publication_is_exclusive_and_revalidated() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-cross-section-", dir=ROOT,
    ) as directory:
        output = Path(directory) / "cross-section.json"
        descriptor = os.open(directory, os.O_RDONLY)
        calls = 0

        def verify() -> None:
            nonlocal calls
            calls += 1

        try:
            _publish(output, {"schema": 1}, descriptor, verify)
            assert calls == 2
            raises(
                _publish, output, {"schema": 2}, descriptor, verify,
            )
        finally:
            os.close(descriptor)
        assert output.read_text(encoding="utf-8") == '{\n  "schema": 1\n}\n'


def test_completed_run_binds_every_terminal_artifact() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-completed-", dir=ROOT,
    ) as directory:
        root = Path(directory).resolve()
        attempt = root / "experiments" / "run-attempt.json"
        run = root / "reports" / "run"
        (root / "tools").mkdir(parents=True)
        attempt.parent.mkdir()
        run.mkdir(parents=True)
        write_json(attempt, {"fixture": "attempt"})
        phases = (phase_for(), phase_for("calibration"))
        attempt_binding = FileBinding(
            "experiments/run-attempt.json", file_sha256(attempt),
        )
        phase_inputs, artifact_paths = [], []
        for phase in phases:
            artifacts = phase_artifacts(root, attempt, phase)
            for path in (
                artifacts.fits, artifacts.predictions,
                artifacts.receipt, artifacts.evaluation,
            ):
                write_json(path, {"fixture": path.name})
            receipt = FileBinding(
                artifacts.receipt.relative_to(root).as_posix(),
                file_sha256(artifacts.receipt),
            )
            write_json(
                artifacts.access,
                context_access_value(attempt_binding, receipt, phase),
            )
            bindings = {
                name: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                }
                for name, path in zip(
                    (
                        "fits", "predictions", "receipt",
                        "access", "evaluation",
                    ),
                    artifacts, strict=True,
                )
            }
            phase_inputs.append({
                "access": bindings["access"],
                "evaluation": bindings["evaluation"],
                "fits": bindings["fits"],
                "phase": phase.phase,
                "predictions": bindings["predictions"],
                "receipt": bindings["receipt"],
            })
            artifact_paths.extend(artifacts)
        decision = {
            "qualifies": {"34": False, "68": False},
            "selected_history": 17,
        }
        source_tree = "d" * 64
        config = FileBinding("config.json", "e" * 64)
        failure = FileBinding("failure.json", "f" * 64)
        outcome = run / "outcome.json"
        write_json(outcome, {
            "decision": decision,
            "evidence_role":
                "development-diagnostic-not-forward-clean",
            "inputs": {
                "attempt": {
                    "path": attempt_binding.path,
                    "sha256": attempt_binding.sha256,
                },
                "phases": phase_inputs,
            },
            "integrity": {
                "config_sha256": config.sha256,
                "source_failure_sha256": failure.sha256,
                "source_tree_sha256": source_tree,
            },
            "schema": 1,
        })
        sources = tuple(
            root / path for path in analyzer.ANALYSIS_SOURCE_PATHS
        )
        for path in sources:
            path.write_text(f"# {path.name}\n", encoding="utf-8")
        paths = (attempt, outcome, *artifact_paths, *sources)
        fake_attempt = SimpleNamespace(
            phases=phases, master=MASTER,
            source_tree=SimpleNamespace(sha256=source_tree),
            config=config, run_dir="reports/run",
            source_binding=lambda name: failure,
        )
        run_identity = _directory_identity(run)
        receipt_validations, ledger_validations = [], []

        def ledgers(
            master: object, phase: object,
            fits: Path, predictions: Path,
            source_failure: str, config_sha256: str,
        ) -> tuple[str, ...]:
            ledger_validations.append((
                master, phase, file_sha256(fits),
                file_sha256(predictions), source_failure, config_sha256,
            ))
            return ("evidence",)

        def completed() -> object:
            with freeze_inputs(paths) as snapshots:
                frozen = {item.source: item for item in snapshots}
                with patch.object(
                    analyzer.ContextAttempt, "read",
                    return_value=fake_attempt,
                ), patch.object(
                    analyzer.ContextReceipt, "parse",
                    return_value=SimpleNamespace(
                        validate=lambda *args: receipt_validations.append(args),
                    ),
                ), patch.object(
                    analyzer, "validate_context_ledgers",
                    side_effect=ledgers,
                ), patch.object(
                    analyzer, "_select_context_history",
                    return_value=decision,
                ), patch.object(analyzer, "ROOT", root):
                    return analyzer._completed_run(
                        attempt, Path("experiments/run-attempt.json"),
                        frozen, run_identity,
                    )

        parsed, evidence, inputs = completed()
        assert parsed is fake_attempt
        assert tuple(evidence) == ("fold-1", "calibration")
        assert len(receipt_validations) == len(ledger_validations) == 2
        for phase, phase_input, receipt_args, ledger_args in zip(
            phases, phase_inputs, receipt_validations,
            ledger_validations, strict=True,
        ):
            fit = FileBinding(**phase_input["fits"])
            prediction = FileBinding(**phase_input["predictions"])
            assert receipt_args == (
                phase, attempt_binding, fit, prediction,
                source_tree, run_identity,
            )
            assert ledger_args == (
                MASTER, phase, fit.sha256, prediction.sha256,
                failure.sha256, config.sha256,
            )
        assert inputs["outcome"]["sha256"] == file_sha256(outcome)
        for path in (attempt, *artifact_paths, outcome):
            original = path.read_bytes()
            path.write_bytes(b'{"tampered":true}\n')
            raises(completed)
            path.write_bytes(original)


def main() -> None:
    test_common_groups_use_only_complete_timestamp_cells()
    test_report_keeps_analysis_non_executable()
    test_imported_entrypoint_cannot_publish()
    test_group_grid_precedes_truth_reads()
    test_publication_is_exclusive_and_revalidated()
    test_completed_run_binds_every_terminal_artifact()
    print("context cross-section analyzer tests passed")


if __name__ == "__main__":
    main()
