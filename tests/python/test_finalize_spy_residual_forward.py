#!/usr/bin/env python3
"""Verify one-shot SPY-residual truth evaluation and terminal publication."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from datetime import date
from math import isclose
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch
import hashlib
import json
import os
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.finalize_spy_residual_forward import (
    evaluate_forward_candidate, finalize_forward_run,
)
from tools.panel_contract import (
    _directory_identity, read_canonical_json, selected_source_tree,
)
from tools.relative_context_contract import SEEDS, ResidualTruthRow
from tools.session_calendar import SessionCalendar, expected_bins
from tools.spy_residual_forward_contract import (
    FORWARD_CALENDAR, FORWARD_CONFIG, FORWARD_RUN_ID, FORWARD_SOURCE_PATHS,
    FORWARD_SOURCES, FORWARD_UNIVERSE, STATE_FINGERPRINTS,
    expected_forward_protocol,
)
from tools.spy_residual_forward_inputs import (
    CandidateLedger, ForwardGrid, ForwardRunBinding, ForwardSeriesPrediction,
    SeedPrediction, TruthReader, _grid_sha256, _json_line, _prediction_record,
    _private_identity, _publish, _publish_receipt, derive_forward_grid,
)
from tools.spy_residual_gate import SPY_DIRECTION_SCALE
from tools import finalize_spy_residual_forward as finalizer
from tools import run_spy_residual_forward as runner

SEED_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5)


def rejects(function: Callable[..., object], *args: object) -> BaseException:
    try:
        function(*args)
    except (OSError, TypeError, ValueError) as error:
        return error
    raise AssertionError("invalid forward finalization succeeded")


def calendar(start: date, end: date) -> SessionCalendar:
    return SessionCalendar(
        start, end, 570, 960, ("XNAS", "XNYS"), (), (),
    )


def json_value(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def run_manager(manager: AbstractContextManager[object]) -> None:
    with manager:
        pass


def fixture(
    parent: Path,
) -> tuple[
    CandidateLedger, tuple[Mapping[str, object], ...],
    Mapping[str, tuple[ResidualTruthRow, ...]], TruthReader, list[int],
    ForwardGrid,
]:
    source = calendar(date(2026, 5, 1), date(2026, 5, 18))
    future = calendar(date(2026, 5, 19), date(2026, 8, 5))
    boundary = tuple(expected_bins(
        source, source.start, source.end, 30,
    ))[-1].timestamp
    grid = derive_forward_grid(source, future, boundary)
    bins_per_session = len(grid.triples) // len(grid.target_sessions)
    predictions = tuple(
        ForwardSeriesPrediction(
            series,
            tuple(
                SeedPrediction(
                    seed, fingerprint, prediction * (1.0 + member / 10),
                )
                for seed, fingerprint, prediction in zip(
                    SEEDS, STATE_FINGERPRINTS, SEED_VALUES, strict=True,
                )
            ),
        )
        for member, series in enumerate(FORWARD_UNIVERSE)
    )
    records = tuple(
        _prediction_record(
            prediction, triple,
            "negative"
            if index % bins_per_session == 0 else "nonnegative",
        )
        for index, triple in enumerate(grid.triples)
        for prediction in predictions
    )
    tree = selected_source_tree(ROOT, FORWARD_SOURCE_PATHS)
    bindings = {
        name: {"path": path, "sha256": sha256}
        for name, path, sha256 in (*FORWARD_SOURCES, FORWARD_CALENDAR)
    }

    def evidence(
        bound_grid: object, rows: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        digest = hashlib.sha256()
        for row in rows:
            digest.update(_json_line(row).encode())
        return {
            "future": {
                "feature_inputs_sha256": "a" * 64,
                "gate_inputs_sha256": "b" * 64,
                "timestamp_grid_sha256": _grid_sha256(grid),
            },
            "historical": {
                "attempt": bindings["residual_attempt"],
                "calibration_fits": bindings["calibration_fits"],
                "evaluation_grid_sha256": "c" * 64,
                "residual_phase_sha256": "d" * 64,
                "runtime_source_tree_sha256": "e" * 64,
                "scaler_inputs_sha256": "f" * 64,
                "source_phase_sha256": "1" * 64,
                "training_grid_sha256": "2" * 64,
            },
            "implementation_tree": json_value(asdict(tree)),
            "prediction_rows_sha256": digest.hexdigest(),
            "protocol": asdict(FORWARD_CONFIG),
            "run": {
                "directory_identity": list(_directory_identity(parent)),
                "id": FORWARD_RUN_ID,
            },
            "sources": {
                "forward_calendar": bindings["forward_calendar"],
                "future_bundle_report": {
                    "path": str(parent / "fetch.json"),
                    "sha256": "3" * 64,
                },
            },
            "states": [
                {"seed": seed, "state_fingerprint": fingerprint}
                for seed, fingerprint in zip(
                    SEEDS, STATE_FINGERPRINTS, strict=True,
                )
            ],
        }

    candidate = _publish(
        parent / "candidate.jsonl", grid, records,
        ForwardRunBinding(lambda _batch: predictions, evidence),
        lambda: None, _directory_identity(parent),
    )
    truth = MappingProxyType({
        series: tuple(
            ResidualTruthRow(
                *triple,
                (1.0 + member / 10) * SPY_DIRECTION_SCALE *
                (0.5 if index % bins_per_session == 0 else 1.0),
            )
            for index, triple in enumerate(grid.triples)
        )
        for member, series in enumerate(FORWARD_UNIVERSE)
    })
    calls: list[int] = []

    def read_truth(claim: CandidateLedger) -> Mapping[
        str, tuple[ResidualTruthRow, ...],
    ]:
        if claim is not candidate or calls:
            raise ValueError("fixture truth was not called exactly once")
        calls.append(1)
        _publish_receipt(
            parent / "truth-access.json", candidate, grid, lambda: None,
            _directory_identity(parent),
        )
        return truth

    return candidate, records, truth, read_truth, calls, grid


@contextmanager
def bound_paths(parent: Path) -> Iterator[None]:
    with patch.multiple(
        finalizer,
        FORWARD_CANDIDATE=parent / "candidate.jsonl",
        FORWARD_TRUTH_RECEIPT=parent / "truth-access.json",
    ), patch.multiple(
        runner,
        FORWARD_CANDIDATE=parent / "candidate.jsonl",
        FORWARD_RUN_DIR=parent,
        FORWARD_TRUTH_RECEIPT=parent / "truth-access.json",
    ):
        yield


def test_reports_preliminary_diagnostic_without_a_decision() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-finalizer-", dir=ROOT,
    ) as directory:
        _, records, truth, _, _, _ = fixture(Path(directory))
        value = evaluate_forward_candidate(records, truth)

    diagnostic = value["diagnostic"]
    assert isclose(
        diagnostic["pooled_raw_residual_r2_vs_zero"],
        48 / 49, rel_tol=0.0, abs_tol=1e-15,
    )
    zero = diagnostic["paired_squared_error"]["zero"]
    assert zero["wins"] == 11
    assert set(zero) == {
        "candidate", "date_count", "intervals", "losses",
        "mean_gain", "per_stock_mean_gain", "reference", "ties", "wins",
    }
    assert tuple(zero["intervals"]) == ("5",)
    assert value["descriptive"]["nondecision_block_intervals"]["zero"] == \
        zero["intervals"]
    assert value["descriptive"]["direction_accuracy"][
        "unchanged-five-seed-mean"
    ] > value["descriptive"]["direction_accuracy"][
        "spy-direction-gated-five-seed-mean"
    ]
    assert value["interpretation"] == {
        "candidate_status":
            "unchanged-pending-confirmatory-forward-evidence",
        "output_role": "residual-only-not-executable-return",
        "policy": "none-preliminary-diagnostic",
        "uncertainty_role":
            "conditional-descriptive-not-confidence-interval",
    }
    assert not {"decision", "gates", "primary"} & set(value)
    assert value["locks"] == expected_forward_protocol()["locks"]


def test_finalization_keeps_evidence_live_through_publication() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-terminal-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        parent = root / "run"
        with bound_paths(parent), patch.multiple(
            runner, _ACTIVE_BUNDLE=root, _CONTROLLER_BUNDLE=root,
        ):
            claim = runner._claim_run()
            try:
                candidate, _, _, read_truth, calls, _ = fixture(parent)
                with finalize_forward_run(
                    candidate, read_truth,
                ) as (value, verify):
                    runner._validate_outcome(claim, value)
                    for mutate in (
                        lambda item: item.update(
                            {"diagnostic": {"x": True}},
                        ),
                        lambda item: item["diagnostic"][
                            "paired_squared_error"
                        ]["zero"].update({"date_count": 5}),
                        lambda item: item["diagnostic"][
                            "paired_squared_error"
                        ]["zero"]["intervals"].update({"20": [0.0, 1.0]}),
                        lambda item: item.update(
                            {"descriptive": {"x": True}},
                        ),
                        lambda item: item["descriptive"][
                            "direction_accuracy"
                        ].update({
                            "spy-direction-gated-five-seed-mean": 1.1,
                        }),
                        lambda item: item["descriptive"][
                            "market_regime_cells"
                        ]["negative"].update({"observation_count": 1}),
                        lambda item: item["descriptive"][
                            "nondecision_block_intervals"
                        ]["zero"].update({"5": [1.0, 0.0]}),
                        lambda item: item["descriptive"][
                            "seed_dispersion"
                        ].update({"mean_raw_population_std": -1.0}),
                    ):
                        changed = json_value(value)
                        mutate(changed)
                        rejects(runner._validate_outcome, claim, changed)
                    changed = json_value(value)
                    name = expected_forward_protocol()["candidate"]["name"]
                    accuracy = changed["descriptive"][
                        "direction_accuracy"
                    ]
                    accuracy[name] = 0.0 if accuracy[name] != 0.0 else 1.0
                    runner._validate_outcome(claim, changed)
                    rejects(
                        runner.publish_forward_outcome,
                        claim, changed, verify,
                    )
                    assert not (parent / "outcome.json").exists()
                    marker = runner.publish_forward_outcome(
                        claim, value, verify,
                    )
                    verify(value)
            finally:
                runner._CLAIMS.pop(id(claim), None)

        outcome = parent / "outcome.json"
        metadata = outcome.stat(follow_symlinks=False)
        assert calls == [1]
        assert marker.path == outcome
        assert marker.identity == (metadata.st_dev, metadata.st_ino)
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert read_canonical_json(outcome) == value
        assert set(path.name for path in parent.iterdir()) == {
            "candidate.jsonl", "outcome.json", "truth-access.json",
        }
        os.link(outcome, parent / "outcome-link.json")
        rejects(runner._verify_terminal_outcome, marker)
        rejects(read_truth, candidate)


def test_imported_callers_cannot_claim_or_publish() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-claim-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        parent = root / "run"
        with bound_paths(parent), patch.multiple(
            runner, _ACTIVE_BUNDLE=None, _CONTROLLER_BUNDLE=None,
        ):
            rejects(runner._claim_run)
            assert not parent.exists()
            parent.mkdir()
            forged = runner.ForwardRunClaim(
                parent, _directory_identity(parent),
            )
            rejects(
                runner.publish_forward_outcome, forged,
                runner._failure_value("claim"), lambda: None,
            )
            assert not (parent / "outcome.json").exists()


def test_candidate_and_receipt_tampering_after_truth_is_rejected() -> None:
    for change in ("candidate-link", "receipt-mode"):
        with tempfile.TemporaryDirectory(
            prefix=f"spy-forward-{change}-", dir=ROOT,
        ) as directory:
            parent = Path(directory)
            candidate, _, _, read_truth, calls, _ = fixture(parent)

            def tampered(
                claim: CandidateLedger,
            ) -> Mapping[str, tuple[ResidualTruthRow, ...]]:
                truth = read_truth(claim)
                if change == "candidate-link":
                    os.link(candidate.path, parent / "candidate-link.jsonl")
                else:
                    (parent / "truth-access.json").chmod(0o644)
                return truth

            with bound_paths(parent):
                rejects(
                    run_manager,
                    finalize_forward_run(candidate, tampered),
                )
            assert calls == [1]


def test_candidate_provenance_is_exactly_bound() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-provenance-", dir=ROOT,
    ) as directory:
        candidate, _, _, _, _, _ = fixture(Path(directory))
        header = json.loads(candidate.path.read_text(
            encoding="utf-8",
        ).splitlines()[0])
        closure = header["closure"]
        finalizer._provenance(closure)
        for change in ("attempt", "calendar", "bundle"):
            changed = json_value(closure)
            if change == "attempt":
                changed["historical"]["attempt"]["sha256"] = "0" * 64
            elif change == "calendar":
                changed["sources"]["forward_calendar"]["path"] = "other.json"
            else:
                changed["sources"]["future_bundle_report"]["path"] = \
                    "fetch.json"
            rejects(finalizer._provenance, changed)


def test_invalid_candidate_and_preexisting_outputs_preserve_truth() -> None:
    for change in ("mode", "outcome", "receipt"):
        with tempfile.TemporaryDirectory(
            prefix=f"spy-forward-{change}-", dir=ROOT,
        ) as directory:
            parent = Path(directory)
            candidate, _, _, read_truth, calls, _ = fixture(parent)
            if change == "mode":
                candidate.path.chmod(0o644)
            elif change == "outcome":
                (parent / "outcome.json").write_text("{}\n", encoding="ascii")
            else:
                (parent / "truth-access.json").write_text(
                    "{}\n", encoding="ascii",
                )
            with bound_paths(parent):
                error = rejects(
                    run_manager, finalize_forward_run(
                        candidate, read_truth,
                    ),
                )
            assert isinstance(error, (OSError, TypeError, ValueError))
            assert not calls
            assert change == "receipt" or \
                not (parent / "truth-access.json").exists()


def test_truth_grid_must_exactly_match_the_candidate() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-grid-", dir=ROOT,
    ) as directory:
        parent = Path(directory)
        candidate, _, truth, _, calls, grid = fixture(parent)
        first = truth[FORWARD_UNIVERSE[0]]
        changed_target = first[0].target[:-3] + "01Z"
        changed = {
            **truth,
            FORWARD_UNIVERSE[0]: (
                ResidualTruthRow(
                    first[0].as_of, first[0].entry,
                    changed_target, first[0].value,
                ),
                *first[1:],
            ),
        }

        def read_truth(claim: CandidateLedger) -> Mapping[
            str, tuple[ResidualTruthRow, ...],
        ]:
            calls.append(1)
            _publish_receipt(
                parent / "truth-access.json", claim, grid,
                lambda: None, _directory_identity(parent),
            )
            return changed

        with bound_paths(parent):
            rejects(
                run_manager, finalize_forward_run(candidate, read_truth),
            )
        assert calls == [1]


def main() -> None:
    test_reports_preliminary_diagnostic_without_a_decision()
    test_finalization_keeps_evidence_live_through_publication()
    test_imported_callers_cannot_claim_or_publish()
    test_candidate_and_receipt_tampering_after_truth_is_rejected()
    test_candidate_provenance_is_exactly_bound()
    test_invalid_candidate_and_preexisting_outputs_preserve_truth()
    test_truth_grid_must_exactly_match_the_candidate()
    print("SPY residual forward finalizer tests passed")


if __name__ == "__main__":
    main()
