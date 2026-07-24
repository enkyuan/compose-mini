#!/usr/bin/env python3
"""Verify that one panel attempt reaches exactly one terminal finalizer."""

from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import run_panel_attempt as driver


def verify_frozen_bindings() -> None:
    assert driver.ATTEMPT == Path(
        "experiments/executable-h13-conditioned-panel-attempt.json"
    )
    assert driver.INPUTS == Path(
        "experiments/executable-h13-panel-inputs.json"
    )
    assert driver.CONFIG == Path(
        "experiments/executable-h13-conditioned-panel.example.json"
    )
    assert driver.BASELINE_REPORT == Path(
        "reports/executable-h13-calibration.json"
    )
    assert driver.BASELINE_LEDGER == Path(
        "reports/executable-h13-calibration.jsonl"
    )
    assert driver.RUN_DIR == Path(
        "reports/h13-conditioned-panel-20260724-01"
    )
    assert driver.OUTCOME == Path(
        "experiments/executable-h13-conditioned-panel-outcome.json"
    )
    assert driver.SERIES == (
        "AAPL=data/aapl-30m.csv",
        "MSFT=data/msft-30m.csv",
        "SPY=data/spy-30m.csv",
    )

    command = tuple(map(str, driver._experiment()))
    assert "--calibration-only" in command
    assert command[command.index("--max-runs") + 1] == "207"
    assert not {
        "--predictions", "--policy", "--authorization", "--test",
        "tools/backtest.py",
    } & set(command)


def stage(command: tuple[object, ...]) -> str:
    values = tuple(map(str, command))
    if any(item.endswith("finalize_panel_attempt.py") for item in values):
        return "finalizer"
    if any(item.endswith("experiment.py") for item in values):
        return "experiment"
    return values[2]


def verify_transition(
    exits: dict[str, int], expected: tuple[str, int, str],
    *, setup_failure: bool = False,
) -> None:
    calls: list[tuple[object, ...]] = []

    def run(command: tuple[object, ...], _environment: object) -> int:
        calls.append(command)
        return exits[stage(command)]

    mkdir = OSError("synthetic setup failure") if setup_failure else None
    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=(
             "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
         )), patch.object(
             driver.signal, "signal", return_value=signal.SIG_DFL,
         ), patch.object(driver, "mkdir_nofollow", side_effect=mkdir):
        result = driver.execute()

    names = tuple(map(stage, calls))
    assert names.count("finalizer") == 1
    assert names[-1] == "finalizer"
    finalizer = tuple(map(str, calls[-1]))
    terminal = (
        finalizer[finalizer.index("--stage") + 1],
        int(finalizer[finalizer.index("--exit") + 1]),
        finalizer[finalizer.index("--status") + 1],
    )
    assert terminal == expected
    assert result == (exits["finalizer"] or expected[1])


def verify_signal() -> None:
    calls: list[tuple[object, ...]] = []

    def run(command: tuple[object, ...], _environment: object) -> int:
        calls.append(command)
        if stage(command) == "experiment":
            raise driver.Interrupted(signal.SIGTERM)
        return 0

    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=(
             "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
         )), patch.object(
             driver.signal, "signal", return_value=signal.SIG_DFL,
         ), patch.object(driver, "mkdir_nofollow"):
        assert driver.execute() == 143

    finalizer = tuple(map(str, calls[-1]))
    assert finalizer[finalizer.index("--stage") + 1] == "experiment"
    assert finalizer[finalizer.index("--exit") + 1] == "143"
    assert finalizer[finalizer.index("--status") + 1] == \
        "experiment-failure"
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    try:
        try:
            driver._interrupt(signal.SIGTERM, None)
        except driver.Interrupted as error:
            assert error.args == (signal.SIGTERM,)
        else:
            raise AssertionError("termination signal was ignored")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def verify_exception(stage_name: str) -> None:
    calls: list[tuple[object, ...]] = []

    def run(command: tuple[object, ...], _environment: object) -> int:
        calls.append(command)
        if stage(command) == stage_name:
            raise RuntimeError("synthetic failure")
        return 0

    mkdir = RuntimeError("synthetic failure") \
        if stage_name == "setup" else None
    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=(
             "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
         )), patch.object(
             driver.signal, "signal", return_value=signal.SIG_DFL,
         ), patch.object(driver, "mkdir_nofollow", side_effect=mkdir):
        result = driver.execute()

    if stage_name == "finalizer":
        assert result == 2
        return
    finalizer = tuple(map(str, calls[-1]))
    expected = {
        "preflight": ("preflight", "1", "preflight-failure"),
        "setup": ("setup", "1", "setup-failure"),
        "experiment": ("experiment", "1", "experiment-failure"),
        "analyze": ("analysis", "2", "analysis-integrity-failure"),
    }[stage_name]
    assert (
        finalizer[finalizer.index("--stage") + 1],
        finalizer[finalizer.index("--exit") + 1],
        finalizer[finalizer.index("--status") + 1],
    ) == expected
    assert result == int(expected[1])


def verify_signal_windows() -> None:
    calls: list[tuple[object, ...]] = []
    install_interrupted = False

    def run(command: tuple[object, ...], _environment: object) -> int:
        calls.append(command)
        return 0

    def install(signum: int, handler: object) -> object:
        nonlocal install_interrupted
        if callable(handler) and signum == signal.SIGINT and \
           not install_interrupted:
            install_interrupted = True
            raise driver.Interrupted(signum)
        return signal.SIG_DFL

    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=(
             "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
         )), patch.object(driver.signal, "signal", side_effect=install):
        assert driver.execute() == 130
    assert tuple(map(stage, calls)).count("finalizer") == 1

    calls.clear()

    def timestamp() -> str:
        if calls:
            driver._interrupt(signal.SIGTERM, None)
        return "2026-07-24T00:00:00Z"

    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=timestamp), \
         patch.object(
             driver.signal, "signal", return_value=signal.SIG_DFL,
         ), patch.object(driver, "mkdir_nofollow"):
        assert driver.execute() == 0
    assert tuple(map(stage, calls)).count("finalizer") == 1

    calls.clear()
    installed: list[object] = []
    raised = False
    runtime_failed = False
    pthread_sigmask = signal.pthread_sigmask

    def run_with_race(
        command: tuple[object, ...], _environment: object,
    ) -> int:
        nonlocal runtime_failed
        calls.append(command)
        if stage(command) == "preflight":
            runtime_failed = True
            raise RuntimeError("synthetic failure")
        return 0

    def capture(_signum: int, handler: object) -> object:
        if callable(handler):
            installed[:] = [handler]
        return signal.SIG_DFL

    def interrupt_exception(how: int, signals: object) -> set[signal.Signals]:
        nonlocal raised
        if runtime_failed and installed and not raised and \
           how == signal.SIG_BLOCK:
            raised = True
            installed[0](signal.SIGTERM, None)
        return pthread_sigmask(how, signals)

    with patch.object(driver, "_run", run_with_race), \
         patch.object(driver, "_utc_now", side_effect=(
             "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
         )), patch.object(driver.signal, "signal", side_effect=capture), \
         patch.object(
             driver.signal, "pthread_sigmask",
             side_effect=interrupt_exception,
         ):
        assert driver.execute() == 143
    assert tuple(map(stage, calls)).count("finalizer") == 1

    calls.clear()
    clock_calls = 0

    def failing_clock() -> str:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 1:
            raise RuntimeError("synthetic clock failure")
        return "2026-07-24T00:01:00Z"

    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=failing_clock), \
         patch.object(
             driver.signal, "signal", return_value=signal.SIG_DFL,
         ):
        assert driver.execute() == 1
    assert tuple(map(stage, calls)) == ("finalizer",)
    finalizer = tuple(map(str, calls[0]))
    assert finalizer[finalizer.index("--started") + 1] == \
        driver.UNKNOWN_TIME

    calls.clear()
    mask_failed = False

    def fail_after_analysis(
        how: int, signals: object,
    ) -> set[signal.Signals]:
        nonlocal mask_failed
        if calls and stage(calls[-1]) == "analyze" and \
           how == signal.SIG_BLOCK and not mask_failed:
            mask_failed = True
            raise RuntimeError("synthetic mask failure")
        return pthread_sigmask(how, signals)

    with patch.object(driver, "_run", run), \
         patch.object(driver, "_utc_now", side_effect=(
             "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
         )), patch.object(
             driver.signal, "signal", return_value=signal.SIG_DFL,
         ), patch.object(
             driver.signal, "pthread_sigmask",
             side_effect=fail_after_analysis,
         ), patch.object(driver, "mkdir_nofollow"):
        assert driver.execute() == 2
    finalizer = tuple(map(str, calls[-1]))
    assert finalizer[finalizer.index("--stage") + 1] == "analysis"
    assert finalizer[finalizer.index("--exit") + 1] == "2"
    assert finalizer[finalizer.index("--status") + 1] == \
        "analysis-integrity-failure"


def verify_descendant_cleanup() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-process-", dir=ROOT,
    ) as directory:
        child_pid = Path(directory) / "child.pid"
        child = (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(60)']);"
            f"open({str(child_pid)!r},'w').write(str(p.pid));"
            "time.sleep(60)"
        )
        probe = (
            "import os,signal,sys;"
            "from tools import run_panel_attempt as d;"
            "signal.signal(signal.SIGTERM,d._interrupt);"
            f"d._run(({sys.executable!r},'-c',{child!r}),os.environ)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", probe], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(500):
                if child_pid.exists():
                    break
                time.sleep(0.01)
            assert child_pid.exists()
            descendant = int(child_pid.read_text(encoding="ascii"))
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=5) != 0
            for _ in range(200):
                try:
                    os.kill(descendant, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("interrupted descendant remained alive")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def verify_primary_runtime() -> None:
    with patch.object(driver.sys, "executable", "/usr/bin/python3"), \
         patch.object(driver, "_run") as run:
        assert driver.execute() == 2
        run.assert_not_called()
    result = subprocess.run(
        [
            sys.executable, "-I", "-c",
            "import runpy,sys;"
            "sys.executable='/usr/bin/python3';"
            "runpy.run_path('tools/run_panel_attempt.py',run_name='__main__')",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "driver requires the bound primary Python" in result.stderr


def verify_symlinked_setup_parent() -> None:
    calls: list[tuple[object, ...]] = []

    def run(command: tuple[object, ...], _environment: object) -> int:
        calls.append(command)
        return 0

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-parent-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        repository = root / "repository"
        outside = root / "outside"
        repository.mkdir()
        outside.mkdir()
        (repository / "reports").symlink_to(
            outside, target_is_directory=True,
        )
        with patch.object(driver, "ROOT", repository), \
             patch.object(driver, "_run", run), \
             patch.object(driver, "_utc_now", side_effect=(
                 "2026-07-24T00:00:00Z", "2026-07-24T00:01:00Z",
             )), patch.object(
                 driver.signal, "signal", return_value=signal.SIG_DFL,
             ):
            assert driver.execute() == 1
        assert tuple(map(stage, calls)) == ("preflight", "finalizer")
        assert not (outside / driver.RUN_DIR.name).exists()


def main() -> None:
    verify_frozen_bindings()
    cases = (
        ({"preflight": 7, "finalizer": 0},
         ("preflight", 7, "preflight-failure"), False),
        ({"preflight": 0, "finalizer": 0},
         ("setup", 1, "setup-failure"), True),
        ({"preflight": 0, "experiment": 9, "finalizer": 0},
         ("experiment", 9, "experiment-failure"), False),
        ({"preflight": 0, "experiment": 0, "analyze": 0, "finalizer": 0},
         ("analysis", 0, "pass"), False),
        ({"preflight": 0, "experiment": 0, "analyze": 3, "finalizer": 0},
         ("analysis", 3, "gate-failure"), False),
        ({"preflight": 0, "experiment": 0, "analyze": 2, "finalizer": 0},
         ("analysis", 2, "analysis-integrity-failure"), False),
    )
    for exits, expected, setup_failure in cases:
        verify_transition(
            exits, expected, setup_failure=setup_failure,
        )
    verify_signal()
    for stage_name in (
        "preflight", "setup", "experiment", "analyze", "finalizer",
    ):
        verify_exception(stage_name)
    verify_signal_windows()
    verify_descendant_cleanup()
    verify_primary_runtime()
    verify_symlinked_setup_parent()
    print("panel driver tests passed")


if __name__ == "__main__":
    main()
