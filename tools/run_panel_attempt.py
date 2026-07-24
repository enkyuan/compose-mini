#!/usr/bin/env python3
"""Run one frozen panel attempt and finalize every catchable exit path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import os
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.panel_contract import mkdir_nofollow

PRIMARY = Path(
    "/Users/Enkang.Yuan1/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/python/bin/python3"
)
UV = Path("/Users/Enkang.Yuan1/.local/bin/uv")
ATTEMPT = Path("experiments/executable-h13-panel-attempt.json")
INPUTS = Path("experiments/executable-h13-panel-inputs.json")
CONFIG = Path("experiments/executable-h13-panel.example.json")
BASELINE_REPORT = Path("reports/executable-h13-calibration.json")
BASELINE_LEDGER = Path("reports/executable-h13-calibration.jsonl")
RUN_DIR = Path("reports/h13-panel-20260723-01")
OUTCOME = Path("experiments/executable-h13-panel-outcome.json")
SERIES = (
    "AAPL=data/aapl-30m.csv",
    "MSFT=data/msft-30m.csv",
    "SPY=data/spy-30m.csv",
)
SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
UNKNOWN_TIME = "1970-01-01T00:00:00Z"


class Interrupted(Exception):
    pass


def _interrupt(signum: int, _frame: object) -> None:
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    raise Interrupted(signum)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def _run(command: Sequence[object], environment: Mapping[str, str]) -> int:
    process = subprocess.Popen(
        [str(item) for item in command], cwd=ROOT, env=environment,
        start_new_session=True,
    )
    try:
        code = process.wait()
    except BaseException:
        _terminate(process)
        raise
    return 128 - code if code < 0 else min(code, 255)


def _analyzer(mode: str, *outputs: Path) -> tuple[object, ...]:
    return (
        PRIMARY, "tools/analyze_panel.py", mode, ATTEMPT, INPUTS, CONFIG,
        BASELINE_REPORT, BASELINE_LEDGER, *outputs, *SERIES,
    )


def _experiment() -> tuple[object, ...]:
    return (
        UV, "run", "--offline", "--with", "torch", "python",
        "tools/experiment.py", CONFIG, RUN_DIR / "experiment.json", *SERIES,
        "--attempt-manifest", ATTEMPT,
        "--input-manifest", INPUTS,
        "--baseline-report", BASELINE_REPORT,
        "--baseline-ledger", BASELINE_LEDGER,
        "--device", "cpu", "--calibration-only",
        "--calibration-predictions", RUN_DIR / "calibration.jsonl",
        "--max-runs", "162",
    )


def _finalizer(
    started: str, ended: str, stage: str, code: int, status: str,
) -> tuple[object, ...]:
    return (
        PRIMARY, "tools/finalize_panel_attempt.py", ATTEMPT, OUTCOME,
        "--started", started, "--ended", ended,
        "--stage", stage, "--exit", code, "--status", status,
    )


def execute() -> int:
    if Path(sys.executable).resolve(strict=True) != PRIMARY.resolve(strict=True):
        print("driver requires the bound primary Python", file=sys.stderr)
        return 2
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"{RUN_DIR}/.pycache",
    }
    stage, code, status = "preflight", 1, "preflight-failure"
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    started = UNKNOWN_TIME
    handlers: dict[int, object] = {}
    interruptions: list[int] = []

    def interrupt(signum: int, frame: object) -> None:
        interruptions.append(signum)
        _interrupt(signum, frame)

    finalizer_exit = 2
    try:
        try:
            for item in SIGNALS:
                handlers[item] = signal.getsignal(item)
            started = _utc_now()
            for item in SIGNALS:
                signal.signal(item, interrupt)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            code = _run(_analyzer("preflight"), environment)
            if code == 0:
                stage, status = "setup", "setup-failure"
                try:
                    mkdir_nofollow(ROOT / RUN_DIR)
                    code = 0
                except (OSError, ValueError) as error:
                    print(f"setup error: {error}", file=sys.stderr)
                    code = 1
            if stage == "setup" and code == 0:
                stage, status = "experiment", "experiment-failure"
                code = _run(_experiment(), environment)
            if stage == "experiment" and code == 0:
                stage, status = "analysis", "analysis-integrity-failure"
                code = _run(_analyzer(
                    "analyze", RUN_DIR / "experiment.json",
                    RUN_DIR / "calibration.jsonl", RUN_DIR / "analysis.json",
                ), environment)
                status = {
                    0: "pass", 3: "gate-failure",
                }.get(code, "analysis-integrity-failure")
            signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        except Interrupted as error:
            code = 128 + int(error.args[0])
            if stage == "analysis":
                status = "analysis-integrity-failure"
            signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        except Exception as error:
            code = 2 if stage == "analysis" else 1
            if stage == "analysis":
                status = "analysis-integrity-failure"
            print(f"{stage} error: {error}", file=sys.stderr)
            signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    except Interrupted:
        pass
    finally:
        while True:
            try:
                signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                break
            except Interrupted:
                continue
        for item in handlers:
            signal.signal(item, signal.SIG_IGN)
        if interruptions:
            code = 128 + interruptions[-1]
            if stage == "analysis":
                status = "analysis-integrity-failure"
        try:
            try:
                ended = _utc_now()
            except Exception as error:
                print(f"timestamp error: {error}", file=sys.stderr)
                ended = started
            finalizer_exit = _run(
                _finalizer(started, ended, stage, code, status), environment,
            )
        except Exception as error:
            print(f"finalizer error: {error}", file=sys.stderr)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            for item, handler in handlers.items():
                signal.signal(item, handler)
    return finalizer_exit or code


def main() -> None:
    raise SystemExit(execute())


if __name__ == "__main__":
    main()
