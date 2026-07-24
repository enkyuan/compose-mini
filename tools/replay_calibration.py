#!/usr/bin/env python3
"""Replay one frozen policy on its calibration predictions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_universe import (
    POLICY_MODELS, DirectoryMembership, _absent, build_replay_report,
    directory_membership, read_json, read_ledger, regular_file_identities,
    resolve_fresh_output, validate_experiment, validate_fetch,
    validate_one_policy, validate_prediction_grid, verify_membership,
)
from tools.backtest import load_frozen_bars
from tools.fetch_universe import UniverseManifest
from tools.files import (
    FrozenInput, freeze_inputs, require_disjoint, verify_frozen, write_json,
)


def _sources(
    manifest: Path, run_dir: Path, model: str, output: Path,
    session_calendar: Path | None,
) -> tuple[
    list[Path], tuple[Path, ...], Path, DirectoryMembership,
]:
    resolved_output = resolve_fresh_output(output)
    if model not in POLICY_MODELS:
        raise ValueError("model must be transformer, mlp, or linear")
    membership = directory_membership(run_dir)
    fixed = (
        manifest, *((session_calendar,) if session_calendar is not None else ()),
        run_dir / "fetch-report.json",
        run_dir / "experiment.json", run_dir / "calibration.jsonl",
        run_dir / f"policy-{model}.json",
    )
    csv_paths = tuple(sorted(path for path, _identity in membership.csv_files))
    sources = [*fixed, *csv_paths]
    regular_file_identities(sources)
    require_disjoint(sources, [resolved_output])
    return sources, csv_paths, resolved_output, membership


def replay(
    manifest_input: FrozenInput, fetch_input: FrozenInput,
    experiment_input: FrozenInput, ledger_input: FrozenInput,
    policy_input: FrozenInput, csv_inputs: Sequence[FrozenInput],
    run_dir: Path, model: str,
    session_calendar_input: FrozenInput | None = None,
) -> dict[str, object]:
    manifest = UniverseManifest.read(manifest_input.snapshot)
    names = tuple(item.ticker for item in manifest.series)
    if len(names) != 11 or len(set(names)) != 11 or \
       len({item.stratum for item in manifest.series}) != 11:
        raise ValueError("replay requires 11 unique tickers and strata")
    by_path = {item.source: item for item in csv_inputs}
    csv_dir = (run_dir / "csv").resolve()
    expected = tuple(
        csv_dir / f"{name.lower()}-{manifest.interval_minutes}m.csv"
        for name in names
    )
    if set(by_path) != set(expected) or len(by_path) != len(expected):
        raise ValueError("CSV sets and counts do not match the manifest")
    bars = {
        name: load_frozen_bars(by_path[path])
        for name, path in zip(names, expected, strict=True)
    }
    fetch = read_json(fetch_input.snapshot, canonical=True)
    validate_fetch(
        fetch, manifest, manifest_input, bars, session_calendar_input,
    )
    forecasts = read_ledger(ledger_input)
    experiment = read_json(experiment_input.snapshot, canonical=True)
    validate_experiment(
        experiment, None, None, names, bars, ledger_input, forecasts,
    )
    validate_prediction_grid(forecasts, experiment, names, bars)
    policy = validate_one_policy(
        read_json(policy_input.snapshot, canonical=True), model, names,
        experiment, experiment_input, ledger_input, forecasts,
    )
    return build_replay_report(
        policy, policy_input, experiment_input, ledger_input, forecasts, bars,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--session-calendar", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        sources, csv_paths, output, membership = _sources(
            args.manifest, args.run_dir, args.model, args.output,
            args.session_calendar,
        )
        with freeze_inputs(sources) as frozen:
            by_source = dict(zip(sources, frozen, strict=True))
            report = replay(
                by_source[args.manifest],
                by_source[args.run_dir / "fetch-report.json"],
                by_source[args.run_dir / "experiment.json"],
                by_source[args.run_dir / "calibration.jsonl"],
                by_source[args.run_dir / f"policy-{args.model}.json"],
                tuple(by_source[path] for path in csv_paths),
                args.run_dir, args.model,
                (
                    by_source[args.session_calendar]
                    if args.session_calendar is not None else None
                ),
            )
            verify_frozen(frozen)
            verify_membership(args.run_dir, membership)
            _absent(output)
            write_json(output, report)
    except (
        IndexError, KeyError, OSError, OverflowError, TypeError,
        UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "output": str(output), "model": args.model,
        "results": len(report["results"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
