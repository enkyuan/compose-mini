#!/usr/bin/env python3
"""Exercise the complete calibration replay without publishing a report."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.analyze_spy_residual_shrinkage as analyzer
import tools.fetch_massive as massive
import tools.fetch_universe as universe_fetch
import tools.files as files
import tools.residual_calibration_evidence as evidence_module


def main() -> None:
    attempt = ROOT / analyzer.SOURCE_ATTEMPT.path
    output = ROOT / "reports/h13-spy-residual-20260725-01-sensitivity"
    groups = []
    validate_commit = evidence_module._validate_commit

    @contextmanager
    def record_frozen(paths):
        with files.freeze_inputs(paths) as frozen:
            groups.append(frozen)
            yield frozen

    def validate_historical(commit, tree):
        if commit != "d" * 40:
            validate_commit(commit, tree)

    if output.exists():
        raise AssertionError("sensitivity output must be absent before replay")
    with ExitStack() as stack:
        for owner, name in (
            (massive, "api_key"), (massive, "request_json"),
            (massive, "fetch_bars"), (massive, "urlopen"),
            (universe_fetch, "api_key"), (universe_fetch, "request_json"),
            (universe_fetch, "fetch_bars"),
            (universe_fetch, "fetch_universe"),
            (urllib.request, "urlopen"),
        ):
            stack.enter_context(patch.object(
                owner, name, side_effect=AssertionError(
                    "network access is forbidden during evidence replay",
                ),
            ))
        for name in (
            "_require_isolated_execution", "_require_exact_launch",
            "_require_package_alias", "_validate_commit",
        ):
            stack.enter_context(patch.object(analyzer, name))
        stack.enter_context(patch.object(
            analyzer, "freeze_inputs", record_frozen,
        ))
        stack.enter_context(patch.object(
            evidence_module, "freeze_inputs", record_frozen,
        ))
        stack.enter_context(patch.object(
            evidence_module, "_validate_commit", validate_historical,
        ))
        with analyzer._authenticate_completed_calibration(
            attempt, "d" * 40,
        ) as session:
            session.evidence.verify()
            assert session.evidence.source.phase == "calibration"
            assert len(session.evidence.series) == 11
            assert {
                series.name: len(series.truth)
                for series in session.evidence.series
            } == {
                name: count
                for name, count, _ in session.evidence.source.evaluation_rows
            }
            assert not callable(session.evidence)
            assert len(groups) == 4 and all(groups)
            frozen = tuple(item for group in groups for item in group)
            expected = {
                path: item.sha256
                for item in frozen
                for path in (item.source, item.snapshot)
            }
            original_hash, changed = files.file_sha256, None

            def cached_hash(path):
                return original_hash(path) if path == changed else \
                    expected[path] if path in expected else original_hash(path)

            with patch.object(
                files, "file_sha256", side_effect=cached_hash,
            ):
                for item in frozen:
                    path = item.snapshot
                    path.chmod(0o600)
                    with path.open("r+b") as file:
                        byte = file.read(1)
                        assert byte
                        file.seek(0)
                        file.write(bytes((byte[0] ^ 0xff,)))
                    path.chmod(0o400)
                    changed = path
                    try:
                        session.evidence.verify()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(
                            f"tampered snapshot was accepted: {item.source}",
                        )
                    finally:
                        path.chmod(0o600)
                        with path.open("r+b") as file:
                            file.write(byte)
                        path.chmod(0o400)
                        changed = None
                session.evidence.verify()
    assert not output.exists()
    print("SPY residual completed-evidence tests passed")


if __name__ == "__main__":
    main()
