#!/usr/bin/env python3
"""Execute one authenticated SPY-residual forward attempt."""

from __future__ import annotations

import sys

_BOOTSTRAP_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None
_PACKAGE_NAME = "tools.run_spy_residual_forward"


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or \
       not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "forward runner requires isolated bytecode-free Python",
        )
    if bootstrapped and (
        _BOOTSTRAP_CACHE_PREFIX is None or
        sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
    ):
        raise ValueError("forward runner requires authenticated bootstrap")


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("forward runner launch inspection is already loaded")

    from ctypes import POINTER, byref, c_int, c_wchar_p, pythonapi
    import os

    argc = c_int()
    argv = POINTER(c_wchar_p)()
    get_argv = pythonapi.Py_GetArgcArgv
    get_argv.argtypes = (POINTER(c_int), POINTER(POINTER(c_wchar_p)))
    get_argv.restype = None
    get_argv(byref(argc), byref(argv))
    observed = tuple(argv[index] for index in range(argc.value))
    canonical = lambda values: (os.path.realpath(values[0]), *values[1:])
    expected = (
        os.path.realpath(sys.executable), *_BOOTSTRAP_FLAGS, *sys.argv,
    )
    if not observed or canonical(observed) != expected or \
       canonical(tuple(sys.orig_argv)) != expected or \
       os.path.realpath(sys.argv[0]) != os.path.realpath(__file__):
        raise ValueError("forward runner requires the exact bound launch")


def _register_package_alias() -> None:
    module = sys.modules[__name__]
    if _PACKAGE_NAME in sys.modules:
        raise ValueError("forward runner package alias is unsafe")
    sys.modules[_PACKAGE_NAME] = module


def _require_package_alias() -> None:
    if sys.modules.get(_PACKAGE_NAME) is not sys.modules[__name__]:
        raise ValueError("forward runner package alias changed")


def _bootstrap_main() -> None:
    """Authenticate the import namespace before exposing repository code."""
    global _BOOTSTRAP_CACHE_PREFIX

    from importlib.machinery import PathFinder
    import os
    import stat
    import tempfile

    while True:
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"compose-mini-forward-runner-{os.urandom(32).hex()}",
        )
        if not os.path.lexists(prefix):
            break
    sys.pycache_prefix = prefix
    sys.dont_write_bytecode = True

    tools = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tools)
    initializer = os.path.join(tools, "__init__.py")
    if not stat.S_ISDIR(os.lstat(tools).st_mode) or \
       not stat.S_ISREG(os.lstat(initializer).st_mode):
        raise ValueError("tools namespace is not a real package")
    for entry in os.scandir(tools):
        mode = entry.stat(follow_symlinks=False).st_mode
        valid = (
            stat.S_ISDIR(mode) if entry.name == "__pycache__" else
            entry.name.endswith(".py") and stat.S_ISREG(mode)
        )
        if not valid:
            raise ValueError("tools namespace contains an unsafe entry")
    if any(
        name == "tools" or name.startswith("tools.") for name in sys.modules
    ):
        raise ValueError("tools namespace is already loaded")
    spec = PathFinder.find_spec("tools", (*sys.path, root))
    locations = tuple(
        os.path.realpath(path)
        for path in (spec.submodule_search_locations or ())
    ) if spec is not None else ()
    if spec is None or os.path.realpath(spec.origin or "") != \
            os.path.realpath(initializer) or \
            locations != (os.path.realpath(tools),):
        raise ValueError("tools namespace resolver is unsafe")
    sys.path.append(root)
    import tools as package
    if os.path.realpath(package.__file__ or "") != \
            os.path.realpath(initializer) or tuple(
                map(os.path.realpath, package.__path__)
            ) != locations:
        raise ValueError("tools namespace import is unsafe")
    _register_package_alias()
    _BOOTSTRAP_CACHE_PREFIX = prefix


if __name__ == "__main__":
    _require_isolated_execution()
    _require_exact_launch(pristine=True)
    _bootstrap_main()

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
import argparse
import math
import os
import signal

from tools.arm_spy_residual import _directory_members
from tools.arm_spy_residual_forward import (
    FORWARD_CANDIDATE, FORWARD_RUN_DIR, FORWARD_TRUTH_RECEIPT,
    ForwardLease, ForwardRunContext, arm_forward_inputs,
)
from tools.files import ExclusiveTemp
from tools.panel_contract import (
    _absent, _directory_identity, _exact_json, _open_directory, _sha256,
    mkdir_nofollow, selected_source_tree,
)
from tools.relative_context_contract import HORIZON_BARS
from tools.run_context_diagnostic import (
    Interrupted, _TerminalOutcome, _verify_terminal_outcome, _write_json,
)
from tools.spy_residual_forward_contract import (
    FORWARD_CONFIG, FORWARD_RUN_ID, FORWARD_SOURCE_PATHS, FORWARD_UNIVERSE,
    expected_forward_protocol,
)
from tools.spy_residual_forward_inputs import (
    TARGET_SESSIONS, CandidateLedger, SpyResidualForwardInputs, TruthReader,
)

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
Verify = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class ForwardRunClaim:
    """Bind one process to a newly created canonical run directory."""

    path: Path
    identity: tuple[int, int]


_ACTIVE_BUNDLE: Path | None = None
_CONTROLLER_BUNDLE: Path | None = None
_TERMINAL_OUTCOME: _TerminalOutcome | None = None
_SIGNAL_NUMBER: int | None = None
_CLAIMS: dict[int, ForwardRunClaim] = {}


def _verify_claim(claim: ForwardRunClaim) -> None:
    if type(claim) is not ForwardRunClaim or \
       _CLAIMS.get(id(claim)) is not claim or \
       _CONTROLLER_BUNDLE is None or \
       _ACTIVE_BUNDLE != _CONTROLLER_BUNDLE or \
       claim.path != FORWARD_RUN_DIR or \
       claim.identity != _directory_identity(claim.path):
        raise ValueError("forward run claim changed")


def _claim_run() -> ForwardRunClaim:
    if _CONTROLLER_BUNDLE is None or _ACTIVE_BUNDLE != _CONTROLLER_BUNDLE:
        raise ValueError("forward run claim is not authorized")
    _absent(FORWARD_RUN_DIR, "forward run directory")
    parent, parent_identity = _open_directory(FORWARD_RUN_DIR.parent)
    try:
        identity = mkdir_nofollow(FORWARD_RUN_DIR)
        os.fsync(parent)
        if _directory_identity(FORWARD_RUN_DIR.parent) != parent_identity or \
           _directory_members(FORWARD_RUN_DIR, ()) != identity:
            raise ValueError("forward run claim changed")
    finally:
        os.close(parent)
    claim = ForwardRunClaim(FORWARD_RUN_DIR, identity)
    _CLAIMS[id(claim)] = claim
    return claim


def publish_forward_candidate(
    lease: ForwardLease, context: ForwardRunContext,
) -> tuple[CandidateLedger, TruthReader]:
    """Stream all cutoffs and stop before the deferred truth reader."""
    session, read_truth = lease._prepare(context)
    result: SpyResidualForwardInputs | CandidateLedger = session.current()
    while isinstance(result, SpyResidualForwardInputs):
        result = session.submit(result)
    if type(result) is not CandidateLedger:
        raise ValueError("forward candidate publication changed")
    return result, read_truth


def publish_forward_outcome(
    claim: ForwardRunClaim, value: Mapping[str, object], verify: Verify,
) -> _TerminalOutcome:
    """Publish one private inode-bound terminal report."""
    global _TERMINAL_OUTCOME

    _verify_claim(claim)
    locks = expected_forward_protocol()["locks"]
    if not isinstance(value, Mapping) or \
       not _exact_json(value.get("locks"), locks) or \
       not callable(verify):
        raise ValueError("forward outcome inputs are invalid")
    _validate_outcome(claim, value)
    outcome = claim.path / "outcome.json"
    _absent(outcome, "forward outcome")
    marker: _TerminalOutcome | None = None
    active = _ACTIVE_BUNDLE is not None

    def committed(
        binding: ExclusiveTemp, digest: str, size: int,
    ) -> None:
        nonlocal marker
        global _TERMINAL_OUTCOME

        marker = _TerminalOutcome(
            outcome, binding.identity, claim.identity, 0o600, size, digest,
        )
        if active:
            _TERMINAL_OUTCOME = marker

    _write_json(
        outcome, value, lambda: verify(value), claim.identity,
        accept_committed_error=True, on_committed=committed,
    )
    if marker is None:
        raise OSError("terminal forward outcome was not authenticated")
    _verify_terminal_outcome(marker)
    return marker


def _validate_diagnostic(
    value: object, protocol: Mapping[str, object],
) -> None:
    """Require the exact finite nondecision metric schema."""
    fields = {
        "paired_squared_error", "pooled_raw_residual_r2_vs_zero",
        "pooled_raw_residual_r2_without_stock",
    }
    if not isinstance(value, Mapping) or set(value) != fields or \
       not isinstance(value["paired_squared_error"], Mapping) or \
       set(value["paired_squared_error"]) != set(
           protocol["metrics"]["references"],
       ) or not isinstance(
           value["pooled_raw_residual_r2_without_stock"], Mapping,
       ) or set(
           value["pooled_raw_residual_r2_without_stock"],
       ) != set(FORWARD_UNIVERSE) or not all(
           type(item) is float and math.isfinite(item)
           for item in (
               value["pooled_raw_residual_r2_vs_zero"],
               *value["pooled_raw_residual_r2_without_stock"].values(),
           )
       ):
        raise ValueError("forward diagnostic changed")

    candidate = protocol["candidate"]["name"]
    widths = tuple(map(
        str, protocol["metrics"]["bootstrap"]["block_sessions"],
    ))
    comparison_fields = {
        "candidate", "date_count", "intervals", "losses", "mean_gain",
        "per_stock_mean_gain", "reference", "ties", "wins",
    }
    for reference, comparison in value["paired_squared_error"].items():
        if not isinstance(comparison, Mapping) or \
           set(comparison) != comparison_fields or \
           comparison["candidate"] != candidate or \
           comparison["reference"] != reference or \
           comparison["date_count"] != TARGET_SESSIONS or \
           not isinstance(comparison["intervals"], Mapping) or \
           tuple(comparison["intervals"]) != widths or \
           not isinstance(comparison["per_stock_mean_gain"], Mapping) or \
           set(comparison["per_stock_mean_gain"]) != set(FORWARD_UNIVERSE):
            raise ValueError("forward paired diagnostic changed")
        gains = tuple(comparison["per_stock_mean_gain"].values())
        counts = tuple(comparison[name] for name in (
            "wins", "ties", "losses",
        ))
        bounds = tuple(comparison["intervals"].values())
        if any(
            type(item) is not float or not math.isfinite(item)
            for item in (*gains, comparison["mean_gain"])
        ) or comparison["mean_gain"] != fmean(gains) or any(
            type(item) is not int or item < 0 for item in counts
        ) or counts != (
            sum(item > 0.0 for item in gains),
            sum(item == 0.0 for item in gains),
            sum(item < 0.0 for item in gains),
        ) or any(
            not isinstance(interval, list) or len(interval) != 2 or
            any(
                type(item) is not float or not math.isfinite(item)
                for item in interval
            ) or interval[0] > interval[1]
            for interval in bounds
        ):
            raise ValueError("forward paired diagnostic changed")


def _finite(value: object, lower: float = -math.inf,
            upper: float = math.inf) -> bool:
    return type(value) is float and math.isfinite(value) and \
        lower <= value <= upper


def _validate_descriptive(
    value: object, diagnostic: Mapping[str, object],
    protocol: Mapping[str, object], expected_count: int,
) -> None:
    """Require the exact bounded descriptive metric schema."""
    fields = {
        "direction_accuracy", "market_regime_cells", "mean_absolute_error",
        "nondecision_block_intervals", "seed_dispersion",
    }
    candidate = protocol["candidate"]["name"]
    references = tuple(protocol["metrics"]["references"])
    family = {candidate, *references}
    if not isinstance(value, Mapping) or set(value) != fields or any(
        not isinstance(value.get(name), Mapping)
        for name in fields
    ) or set(value["direction_accuracy"]) != family or \
       set(value["mean_absolute_error"]) != family or any(
           not _finite(item, 0.0, 1.0)
           for item in value["direction_accuracy"].values()
       ) or any(
           not _finite(item, 0.0)
           for item in value["mean_absolute_error"].values()
       ):
        raise ValueError("forward descriptive metrics changed")

    cells = value["market_regime_cells"]
    cell_fields = {
        "candidate_direction_accuracy", "candidate_mean_absolute_error",
        "mean_squared_error_gain_vs_unchanged_five_seed_mean",
        "mean_squared_error_gain_vs_zero", "observation_count",
    }
    if set(cells) != {"negative", "nonnegative"}:
        raise ValueError("forward regime cells changed")
    counts = []
    for cell in cells.values():
        if not isinstance(cell, Mapping) or set(cell) != cell_fields:
            raise ValueError("forward regime cell changed")
        count = cell["observation_count"]
        values = tuple(cell[name] for name in cell_fields - {
            "observation_count",
        })
        if type(count) is not int or count < 0 or \
           (count == 0) != all(item is None for item in values) or \
           count and (
               not _finite(cell["candidate_direction_accuracy"], 0.0, 1.0) or
               not _finite(cell["candidate_mean_absolute_error"], 0.0) or
               any(not _finite(cell[name]) for name in (
                   "mean_squared_error_gain_vs_unchanged_five_seed_mean",
                   "mean_squared_error_gain_vs_zero",
               ))
           ):
            raise ValueError("forward regime cell changed")
        counts.append(count)
    if sum(counts) != expected_count or any(
        count % len(FORWARD_UNIVERSE) for count in counts
    ):
        raise ValueError("forward regime counts changed")

    widths = tuple(map(
        str, protocol["metrics"]["bootstrap"]["block_sessions"],
    ))
    intervals = value["nondecision_block_intervals"]
    paired = diagnostic["paired_squared_error"]
    if set(intervals) != set(references) or any(
        not isinstance(intervals[reference], Mapping) or
        tuple(intervals[reference]) != widths or
        not _exact_json(
            intervals[reference], paired[reference]["intervals"],
        )
        for reference in references
    ):
        raise ValueError("forward descriptive intervals changed")
    dispersion = value["seed_dispersion"]
    if set(dispersion) != {
        "mean_gated_population_std", "mean_raw_population_std",
    } or any(not _finite(item, 0.0) for item in dispersion.values()):
        raise ValueError("forward seed dispersion changed")


def _validate_outcome(
    claim: ForwardRunClaim, value: Mapping[str, object],
) -> None:
    kind = value.get("type")
    if kind == "spy-residual-forward-integrity-failure":
        expected = _failure_value(value.get("stage"))
        if not _exact_json(value, expected):
            raise ValueError("forward failure outcome changed")
        return
    fields = {
        "descriptive", "diagnostic", "evidence_role", "inputs", "integrity",
        "interpretation", "locks", "run", "sample", "schema", "type",
    }
    batches = TARGET_SESSIONS * HORIZON_BARS
    expected_count = batches * len(FORWARD_UNIVERSE)
    interpretation = value.get("interpretation")
    run = value.get("run")
    sample = value.get("sample")
    inputs = value.get("inputs")
    integrity = value.get("integrity")
    protocol = expected_forward_protocol()
    if set(value) != fields or value.get("schema") != 1 or \
       kind != "spy-residual-forward-outcome" or \
       value.get("evidence_role") != \
            "predeclared-expedited-forward-diagnostic-terminal" or \
       not isinstance(value.get("descriptive"), Mapping) or \
       not value["descriptive"] or \
       not isinstance(value.get("diagnostic"), Mapping) or \
       not value["diagnostic"] or \
       not isinstance(integrity, Mapping) or set(integrity) != {
           "config_sha256", "grid_sha256",
           "implementation_tree_sha256", "prediction_rows_sha256",
           "transformer_states",
       } or integrity["config_sha256"] != FORWARD_CONFIG.sha256 or \
       not _exact_json(
           integrity["transformer_states"],
           protocol["transformer_states"],
       ) or \
       not _exact_json(interpretation, {
           "candidate_status":
               "unchanged-pending-confirmatory-forward-evidence",
           "output_role": "residual-only-not-executable-return",
           "policy": "none-preliminary-diagnostic",
           "uncertainty_role":
               "conditional-descriptive-not-confidence-interval",
       }) or \
       not _exact_json(run, {
           "batches": batches,
           "id": FORWARD_RUN_ID,
           "observation_count": expected_count,
           "stock_count": len(FORWARD_UNIVERSE),
           "target_session_count": TARGET_SESSIONS,
       }) or not _exact_json(sample, {
           "batches": batches,
           "observation_count": expected_count,
           "stock_count": len(FORWARD_UNIVERSE),
           "target_session_count": TARGET_SESSIONS,
       }) or not isinstance(inputs, Mapping) or set(inputs) != {
           "candidate", "truth_access",
        }:
        raise ValueError("forward success outcome changed")
    _validate_diagnostic(value["diagnostic"], protocol)
    _validate_descriptive(
        value["descriptive"], value["diagnostic"], protocol, expected_count,
    )
    candidate, receipt = inputs["candidate"], inputs["truth_access"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "directory_identity", "identity", "path", "records", "sha256",
    } or candidate["directory_identity"] != list(claim.identity) or \
       candidate["path"] != str(FORWARD_CANDIDATE) or \
       candidate["records"] != expected_count or \
       not isinstance(receipt, Mapping) or set(receipt) != {
           "identity", "path", "sha256",
       } or receipt["path"] != str(FORWARD_TRUTH_RECEIPT):
        raise ValueError("forward outcome bindings changed")
    for binding in (candidate, receipt):
        identity = binding["identity"]
        if not isinstance(identity, list) or len(identity) != 2 or any(
            type(item) is not int or item < 0 for item in identity
        ):
            raise ValueError("forward outcome identity changed")
        _sha256(binding["sha256"], "forward outcome binding")
    for name in (
        "grid_sha256", "implementation_tree_sha256",
        "prediction_rows_sha256",
    ):
        _sha256(integrity[name], f"forward outcome {name}")


def _validate_controller(bundle: Path) -> None:
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    _require_package_alias()
    if _ACTIVE_BUNDLE != bundle or not isinstance(bundle, Path) or \
       bundle != Path(os.path.abspath(bundle)) or \
       bundle != bundle.resolve(strict=True) or \
       not bundle.is_dir() or tuple(sys.argv) != (
           str(Path(__file__).resolve()), str(bundle),
       ):
        raise ValueError("forward bundle or command changed")


def _failure_value(stage: object) -> dict[str, object]:
    allowed = {"authenticate", "claim", "predict", "finalize"}
    if not isinstance(stage, str) or stage not in allowed:
        raise ValueError("forward failure stage changed")
    return {
        "locks": expected_forward_protocol()["locks"],
        "schema": 1,
        "stage": stage,
        "status": "integrity-failure",
        "type": "spy-residual-forward-integrity-failure",
    }


def execute_forward_attempt(
    massive_bundle: Path,
) -> Mapping[str, object]:
    """Run the fixed candidate, open truth once, and close terminal evidence."""
    global _CONTROLLER_BUNDLE

    _validate_controller(massive_bundle)
    if _CONTROLLER_BUNDLE is not None:
        raise ValueError("forward controller is already active")
    _CONTROLLER_BUNDLE = massive_bundle
    claim: ForwardRunClaim | None = None
    terminal: _TerminalOutcome | None = None
    stage = "authenticate"
    try:
        tree = selected_source_tree(ROOT, FORWARD_SOURCE_PATHS)
        with arm_forward_inputs(massive_bundle) as lease:
            _require_package_alias()
            from tools.finalize_spy_residual_forward import \
                finalize_forward_run

            if selected_source_tree(ROOT, FORWARD_SOURCE_PATHS) != tree:
                raise ValueError("forward implementation tree changed")
            stage = "claim"
            claim = _claim_run()
            context = ForwardRunContext(
                tree, FORWARD_RUN_ID, claim.identity,
            )

            def verify() -> None:
                lease()
                _verify_claim(claim)
                if selected_source_tree(
                    ROOT, FORWARD_SOURCE_PATHS,
                ) != tree:
                    raise ValueError("forward implementation tree changed")

            stage = "predict"
            candidate, read_truth = publish_forward_candidate(
                lease, context,
            )
            stage = "finalize"
            with finalize_forward_run(
                candidate, read_truth,
            ) as (value, verify_evidence):

                def verify_terminal(
                    observed: Mapping[str, object],
                ) -> None:
                    verify()
                    verify_evidence(observed)

                terminal = publish_forward_outcome(
                    claim, value, verify_terminal,
                )
            verify()
            if _directory_members(claim.path, (
                "candidate.jsonl", "outcome.json", "truth-access.json",
            )) != claim.identity:
                raise ValueError("forward terminal topology changed")
            return value
    except BaseException:
        if claim is not None and terminal is None:
            try:
                publish_forward_outcome(
                    claim, _failure_value(stage),
                    lambda _value: _verify_claim(claim),
                )
            except (OSError, ValueError):
                pass
        raise
    finally:
        if claim is not None:
            _CLAIMS.pop(id(claim), None)
        _CONTROLLER_BUNDLE = None


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("massive_bundle", type=Path)
    return parser.parse_args(argv)


def _run(path: Path) -> int:
    """Execute once, mapping the first signal to its shell exit status."""
    global _ACTIVE_BUNDLE, _SIGNAL_NUMBER, _TERMINAL_OUTCOME

    _ACTIVE_BUNDLE = Path(os.path.abspath(path))
    _TERMINAL_OUTCOME = None
    _SIGNAL_NUMBER = None
    previous = []
    failure: BaseException | None = None
    completed = False

    def interrupt(number: int, _frame: object) -> None:
        global _SIGNAL_NUMBER

        if _TERMINAL_OUTCOME is None and _SIGNAL_NUMBER is None:
            _SIGNAL_NUMBER = number
            raise Interrupted(number)

    def remember(error: BaseException) -> None:
        nonlocal failure
        if failure is None:
            failure = error

    try:
        for number in SIGNALS:
            previous.append((number, signal.getsignal(number)))
            signal.signal(number, interrupt)
        execute_forward_attempt(path)
        completed = True
    except BaseException as error:
        remember(error)
    if completed and _TERMINAL_OUTCOME is None:
        remember(ValueError(
            "forward run returned without a terminal outcome",
        ))
    elif _TERMINAL_OUTCOME is not None:
        try:
            _verify_terminal_outcome(_TERMINAL_OUTCOME)
        except BaseException as error:
            remember(error)
    for number, handler in previous:
        try:
            signal.signal(number, handler)
        except Interrupted as error:
            try:
                signal.signal(number, handler)
            except BaseException as retry_error:
                remember(retry_error)
            else:
                if not completed:
                    remember(error)
        except BaseException as error:
            remember(error)
    _ACTIVE_BUNDLE = _TERMINAL_OUTCOME = None
    _SIGNAL_NUMBER = None
    if isinstance(failure, Interrupted):
        return 128 + failure.number
    if failure is not None:
        raise failure
    return 0


def main() -> None:
    try:
        code = _run(parse_args().massive_bundle)
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"forward runner error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
