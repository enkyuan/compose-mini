"""Freeze one development-only SPY-residual calibration protocol."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tools.context_diagnostic_contract import (
    ContextFit, ContextPhase, _json_sha256, context_phase_sha256,
    expected_context_fits,
)
from tools.panel_contract import (
    FileBinding, _exact_json, _integer, _object, _sha256, _string,
)
from tools.universe_contract import universe_roles
from tools.universe_cross_section import CROSS_SECTION_SEED
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES,
)

SPY_RESIDUAL_TARGET = "spy-residual-executable-return-v1"
EVIDENCE_ROLE = "development-calibration-not-forward-clean"
HISTORY_BARS = 17
HORIZON_BARS = 13
INTERVAL_MINUTES = 30
SPY_START = "2024-11-01"
SPY_END = "2026-07-21"
MODELS = ("global_ridge", "global_mlp", "panel_transformer")
SEEDS = (7, 19, 31, 43, 61)
PHASE_BUDGETS = (("fold-1", 302), ("calibration", 349))
PAIRED_COMPARISONS = (
    ("global_ridge", "zero"),
    ("global_mlp", "zero"),
    ("global_mlp", "global_ridge"),
    ("panel_transformer", "zero"),
    ("panel_transformer", "global_ridge"),
    ("panel_transformer", "global_mlp"),
)
RESIDUAL_CONFIG = FileBinding(
    "experiments/executable-h13-spy-residual.example.json",
    "cd5103fa93835222ae789a228ff776765c23bd7d0de6a2200c1c610ec557af19",
)
RESIDUAL_SOURCE = MappingProxyType({
    "context_attempt": FileBinding(
        "experiments/h13-context-diagnostic-20260725-03-attempt.json",
        "700d4e27ccd714e6156522be22515c9b3b04aa97dbdd6f09fd199e13463c1394",
    ),
    "context_outcome": FileBinding(
        "reports/h13-context-diagnostic-20260725-03/outcome.json",
        "bc33d4c86afeab4d7273215a81f2f701c68ff1a251fcb9935508098677063040",
    ),
})
RESIDUAL_BENCHMARK = MappingProxyType({
    "fetch_report": FileBinding(
        "data/spy-residual-20260725/fetch.json",
        "024e710102f866a3ffcd89ae22688d333f2736ed99b086f03680f380f3fbbaf6",
    ),
    "spy_csv": FileBinding(
        "data/spy-residual-20260725/spy.csv",
        "ce8de54c6fddac96d2866687e97cea2367579051c9da5b360ad4ccda53c1ed2b",
    ),
})
RESIDUAL_CALENDAR = FileBinding(
    "universes/us-equities-core-2024-07-22_2026-07-21.json",
    "b1e0835a60624a67e21f7941ac00ece6c488937989560bbd4d0333afd869e5f8",
)
_SOURCE_PHASE_OUTPUTS = (
    ("fold-1", (
        ("access", "d82ae9819c08c43a805cc7d0c49e8997ce06404a64b2710b6c3089430b29c970"),
        ("evaluation",
         "2cfe45ff04145a9b0587bf6a12e04eb2d7083df461d9736777e7dbfac8fe7053"),
        ("fits", "d79fce757a67c7fbdf4138ea8c39d50d8907322dc0c50208fd3318a4023b8329"),
        ("predictions",
         "a01334b98f2ce90219d5a051f9975d54091943702b9a1f0902039e4efeefab15"),
        ("receipt", "fa81f7bd2a86e7d4b911fca14c405072f3c65701cce13a6118a19dcd7e7c8167"),
    )),
    ("calibration", (
        ("access", "870384b811febdeafdb9dab267c3fd41a41774cebf371d8bf1659068cacf7045"),
        ("evaluation",
         "3fc6f400de16c723813eaab573b445838bab8ff24100c191fc9ad02a37686762"),
        ("fits", "389ab192e532b27051172d3711c2efe748640ca73ee725bb13d3af9f4b9fd3a2"),
        ("predictions",
         "1a011efedb4a63785c6923d85c4f838ff6c76f4da2bb181c6d3dca9e9532581f"),
        ("receipt", "dfd7d6a8cbb04e9394ae28d32c1bb5f803639815c66c9333e281f0167524e183"),
    )),
)


def _binding_value(binding: FileBinding) -> dict[str, str]:
    return {"path": binding.path, "sha256": binding.sha256}


def _source_output_path(run: str, phase: str, name: str) -> str:
    suffix = "truth-access" if name == "access" else name
    extension = "jsonl" if name in ("fits", "predictions") else "json"
    return f"{run}/{phase}-{suffix}.{extension}"


def expected_source_context_outcome() -> dict[str, object]:
    """Return the exact successful context outcome used as source evidence."""
    run = "reports/h13-context-diagnostic-20260725-03"
    return {
        "decision": {
            "qualifies": {"34": False, "68": False},
            "selected_history": HISTORY_BARS,
        },
        "evidence_role": "development-diagnostic-not-forward-clean",
        "inputs": {
            "attempt": _binding_value(RESIDUAL_SOURCE["context_attempt"]),
            "phases": [
                {
                    **{
                        name: {
                            "path": _source_output_path(
                                run, phase, name,
                            ),
                            "sha256": sha256,
                        }
                        for name, sha256 in outputs
                    },
                    "phase": phase,
                }
                for phase, outputs in _SOURCE_PHASE_OUTPUTS
            ],
        },
        "integrity": {
            "config_sha256":
                "4224b532e0b3b3b16d9638be7b53b59003b9b04841ce45d1ca8e998afbd89a04",
            "source_failure_sha256":
                "8ff90ca089ac4e3e0836b8e76d436ff0748a547eaee21fe623e2846ab9f86db6",
            "source_tree_sha256":
                "5ae25974c2d4e109fafe249ea17a5e14885ecd838d71fdbe7b47cd3550793fcf",
        },
        "schema": 1,
    }


def validate_source_context_outcome(
    value: object,
) -> Mapping[str, object]:
    """Reject source evidence other than the successful history-17 outcome."""
    expected = expected_source_context_outcome()
    if not _exact_json(value, expected):
        raise ValueError("source context outcome changed")
    return expected


def _repository_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError("repository root must be an absolute path")
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("repository root is invalid") from error
    if value != resolved:
        raise ValueError("repository root must not use an alias")
    return value


def expected_spy_fetch_report(
    repository_root: Path,
) -> dict[str, object]:
    """Return the exact authenticated SPY fetch report for this checkout."""
    root = _repository_root(repository_root)
    return {
        "adjusted": True,
        "aggregate": {
            "request": {
                "path":
                    "/v2/aggs/ticker/SPY/range/30/minute/2024-11-01/2026-07-21",
                "query": {
                    "adjusted": "true",
                    "limit": "50000",
                    "sort": "asc",
                },
            },
        },
        "calendar": {
            "applicability": {
                "benchmark": "SPY",
                "calendar_venue": "XNYS",
                "exchange_source":
                    "https://massive.com/docs/rest/stocks/market-operations/exchanges",
                "market_group": "NYSE Group",
                "operating_mic": "XNYS",
                "primary_exchange": "ARCX",
                "session": "core",
                "session_source":
                    "https://www.nyse.com/trade/hours-calendars",
            },
            "path": str(root / RESIDUAL_CALENDAR.path),
            "sha256": RESIDUAL_CALENDAR.sha256,
        },
        "csv": {
            "path": str(root / RESIDUAL_BENCHMARK["spy_csv"].path),
            "rows": 5_534,
            "session_audit": {
                "affected_sessions": 0,
                "expected_bins": 5_534,
                "expected_sessions": 428,
                "missing_bins": 0,
                "missing_sessions": [],
                "ranges": [],
                "scope": "all-expected-session-bins",
            },
            "sessions": 428,
            "sha256": RESIDUAL_BENCHMARK["spy_csv"].sha256,
            "source_rows": 13_653,
        },
        "end": SPY_END,
        "gap_policy": "require-complete-core-session-grid",
        "interval_minutes": INTERVAL_MINUTES,
        "purpose":
            "Authenticate SPY bars for development-only residual calibration.",
        "reference": {
            "identity": {
                "active": True,
                "currency_name": "usd",
                "locale": "us",
                "market": "stocks",
                "primary_exchange": "ARCX",
                "ticker": "SPY",
                "type": "ETF",
            },
            "request": {
                "path": "/v3/reference/tickers/SPY",
                "query": {"date": "2024-10-31"},
            },
        },
        "reference_date": "2024-10-31",
        "return_basis": "split-adjusted-price-return-not-dividend-adjusted",
        "schema": 1,
        "session": "regular",
        "start": SPY_START,
        "ticker": "SPY",
    }


def validate_spy_fetch_report(
    value: object, repository_root: Path,
) -> Mapping[str, object]:
    """Reject SPY metadata outside the authenticated complete-grid bundle."""
    expected = expected_spy_fetch_report(repository_root)
    if not _exact_json(value, expected):
        raise ValueError("SPY fetch report changed")
    return expected


def expected_residual_protocol() -> dict[str, object]:
    """Return a fresh copy of the frozen residual-calibration choices."""
    return {
        "alignment_horizon_bars": HORIZON_BARS,
        "baselines": ["zero"],
        "batch_size": 128,
        "bootstrap": {
            "applies_to": "stock-macro-paired-absolute-error",
            "block_days": list(BOOTSTRAP_BLOCK_DAYS),
            "interval": "equal-tailed-95-percentile",
            "method": "shared-circular-date-block",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": CROSS_SECTION_SEED,
            "weighting": "stock-macro",
        },
        "data_interval": {
            "end": SPY_END,
            "minutes": INTERVAL_MINUTES,
            "start": SPY_START,
        },
        "evidence_role": EVIDENCE_ROLE,
        "feature_set": "ohlcv",
        "history_bars": HISTORY_BARS,
        "locks": {
            "absolute_forecast_authorized": False,
            "backtest_run": False,
            "forward_clean": False,
            "trading_authorized": False,
            "universe_expansion_authorized": False,
        },
        "metrics": {
            "primary": [
                "pooled-raw-residual-r2-vs-zero",
                "stock-macro-paired-absolute-error",
            ],
            "secondary": [
                "pooled-timestamp-centered-cross-sectional-r2",
                "mean-valid-timestamp-spearman-rank-ic",
            ],
        },
        "model_inputs": {
            "global_mlp": "stock-only",
            "global_ridge": "stock-only",
            "panel_transformer": "stock-plus-final-completed-spy-row",
        },
        "models": list(MODELS),
        "neural_checkpoint_policy": "reuse-context-prior-selections",
        "output_role": "residual-only-not-executable-return",
        "paired_absolute_error_comparisons": [
            list(pair) for pair in PAIRED_COMPARISONS
        ],
        "paired_absolute_error_convention":
            "reference-mae-minus-candidate-mae-positive",
        "phases": [
            {"name": name, "updates_per_checkpoint": updates}
            for name, updates in PHASE_BUDGETS
        ],
        "residualization": "stock-minus-spy-fixed-beta-1",
        "sampling_policy": "stock-balanced",
        "scalers": {
            "residual_target": "per-stock-training-prefix",
            "spy_features": "per-stock-aligned-training-prefix",
            "stock_features": "per-stock-training-prefix",
        },
        "seed_aggregation": {
            "primary": "arithmetic-mean-predictions-before-metrics",
            "report": {
                "per_observation":
                    "population-standard-deviation-across-seeds",
                "summary": "stock-macro-mean-over-common-grid",
            },
        },
        "seeds": list(SEEDS),
        "target_horizon_bars": HORIZON_BARS,
        "target_kind": SPY_RESIDUAL_TARGET,
        "timestamp_policy": "exact-triples-and-common-cross-section",
    }


def validate_residual_protocol(
    value: object,
) -> Mapping[str, object]:
    """Reject any choice outside the predeclared residual calibration."""
    expected = expected_residual_protocol()
    if not _exact_json(value, expected):
        raise ValueError("config does not match the residual calibration")
    return expected


@dataclass(frozen=True, slots=True)
class ResidualScalerInput:
    series: str
    stock_training_prefix_sha256: str
    spy_training_prefix_sha256: str
    training_rows: int
    training_grid_sha256: str


def residual_scaler_inputs_sha256(
    master: Sequence[str],
    values: Sequence[ResidualScalerInput],
) -> str:
    """Bind each stock's aligned training-only scaler inputs."""
    names = tuple(master)
    universe_roles(names)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("residual scaler inputs are invalid")
    inputs = tuple(values)
    if len(inputs) != len(names):
        raise ValueError("residual scaler inputs must contain 55 stocks")
    records = []
    for index, (value, series) in enumerate(zip(
        inputs, names, strict=True,
    )):
        if not isinstance(value, ResidualScalerInput) or \
           _string(value.series, f"scaler input {index} series") != series:
            raise ValueError("residual scaler input order changed")
        records.append({
            "series": series,
            "spy_training_prefix_sha256": _sha256(
                value.spy_training_prefix_sha256,
                f"{series} SPY training prefix",
            ),
            "stock_training_prefix_sha256": _sha256(
                value.stock_training_prefix_sha256,
                f"{series} stock training prefix",
            ),
            "training_grid_sha256": _sha256(
                value.training_grid_sha256,
                f"{series} training grid",
            ),
            "training_rows": _integer(
                value.training_rows, f"{series} training rows",
            ),
        })
    return _json_sha256({
        "history_bars": HISTORY_BARS,
        "inputs": records,
        "role": "residual-scaler-inputs",
        "schema": 1,
    })


@dataclass(frozen=True, slots=True)
class ResidualPhaseInput:
    phase: str
    source_phase_sha256: str
    aligned_training_grid_sha256: str
    aligned_evaluation_grid_sha256: str
    scaler_inputs_sha256: str

    @classmethod
    def parse(
        cls, value: object, source: ContextPhase,
    ) -> "ResidualPhaseInput":
        if not isinstance(source, ContextPhase):
            raise ValueError("residual source phase is invalid")
        expected_updates = dict(PHASE_BUDGETS).get(source.phase)
        if source.updates_per_checkpoint != expected_updates:
            raise ValueError("residual source phase budget changed")
        item = _object(value, {
            "aligned_evaluation_grid_sha256",
            "aligned_training_grid_sha256",
            "phase",
            "scaler_inputs_sha256",
            "source_phase_sha256",
        }, "residual phase")
        parsed = cls(
            _string(item["phase"], "residual phase name"),
            _sha256(item["source_phase_sha256"], "source phase"),
            _sha256(item["aligned_training_grid_sha256"], "training grid"),
            _sha256(
                item["aligned_evaluation_grid_sha256"], "evaluation grid",
            ),
            _sha256(item["scaler_inputs_sha256"], "scaler inputs"),
        )
        if (
            parsed.phase != source.phase
            or parsed.source_phase_sha256 != context_phase_sha256(source)
            or parsed.aligned_training_grid_sha256
            != source.training_grid_sha256
            or parsed.aligned_evaluation_grid_sha256
            != source.evaluation_grid_sha256
        ):
            raise ValueError("residual phase differs from its source")
        return parsed


def parse_residual_phases(
    value: object,
    source: Sequence[ContextPhase],
) -> tuple[ResidualPhaseInput, ...]:
    """Bind the two residual phases to their authenticated source phases."""
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError("residual phases changed")
    phases = tuple(source)
    if not isinstance(value, list) or len(value) != len(PHASE_BUDGETS) or \
       len(phases) != len(PHASE_BUDGETS) or any(
           not isinstance(phase, ContextPhase) for phase in phases
       ) or tuple(
           phase.phase for phase in phases
       ) != tuple(name for name, _ in PHASE_BUDGETS):
        raise ValueError("residual phases changed")
    parsed = tuple(
        ResidualPhaseInput.parse(item, phase)
        for item, phase in zip(value, phases, strict=True)
    )
    if len({phase.scaler_inputs_sha256 for phase in parsed}) != len(parsed):
        raise ValueError("residual phase scaler inputs were reused")
    return parsed


def expected_residual_fits(
    master: Sequence[str], phase: ContextPhase,
) -> tuple[ContextFit, ...]:
    """Reuse only the selected history-17 context fit schedule."""
    if not isinstance(phase, ContextPhase) or \
       phase.updates_per_checkpoint != dict(PHASE_BUDGETS).get(phase.phase):
        raise ValueError("residual source phase budget changed")
    fits = tuple(
        fit for fit in expected_context_fits(master, phase)
        if fit.history == HISTORY_BARS
    )
    expected = (
        (phase.phase, "global_ridge", HISTORY_BARS, None),
        *(
            (phase.phase, model, HISTORY_BARS, seed)
            for model in MODELS[1:] for seed in SEEDS
        ),
    )
    if tuple(
        (fit.phase, fit.model, fit.history, fit.seed) for fit in fits
    ) != expected:
        raise ValueError("residual fit closure changed")
    return fits


def validate_spy_session_audit(
    value: object,
) -> Mapping[str, object]:
    """Validate the declared clean SPY grid; the armer binds source bytes."""
    expected = {
        "scope": "all-expected-session-bins",
        "expected_sessions": 428,
        "affected_sessions": 0,
        "missing_sessions": [],
        "expected_bins": 5_534,
        "missing_bins": 0,
        "ranges": [],
    }
    if not _exact_json(value, expected):
        raise ValueError("SPY session grid is incomplete")
    return expected
