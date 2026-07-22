#!/usr/bin/env python3
"""Backtest forecast ledgers with a frozen long-or-cash execution policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import fmean
import argparse
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import (
    CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET, FEATURE_COUNT, TARGET_FORMULAS,
    TARGET_KINDS, read_bars,
)
from tools.files import (
    file_sha256 as _sha256, require_disjoint, series_arg, write_json,
)

NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
LEDGER_V1_FIELDS = frozenset((
    "schema", "split", "series", "model", "candidate", "feature_set",
    "seed", "csv_sha256", "as_of", "target_time", "horizon_bars",
    "predicted_log_return",
))
LEDGER_V2_FIELDS = LEDGER_V1_FIELDS | {"fold", "target_kind"}
POLICY_FIELDS = frozenset((
    "schema", "action", "model", "candidate", "feature_set", "target_kind",
    "horizon_bars", "seeds", "series", "initial_cash", "costs", "safety_bps",
    "minimum_predicted_log_return", "selection_objective",
    "validation_report", "validation_prediction_ledger", "threshold_trials",
    "test_grid", "validation_fingerprint",
))
POLICY_COST_FIELDS = ("spread_bps", "slippage_bps", "fee_bps")
TRIAL_FIELDS = frozenset((
    "action", "safety_bps", "objective", "mean_final_equity",
    "mean_gross_turnover", "signal_coverage", "execution_coverage",
    "trade_count",
))
LINE_CAP = 4_096


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) \
        is not None


def experiment_fingerprint(report: Mapping[str, object]) -> str:
    """Hash the validation contract shared by validation and test phases."""
    try:
        payload = {field: report[field] for field in (
            "series", "sweep", "selection", "validation", "test_contract",
        )}
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True,
        ).encode()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("experiment report cannot form a policy fingerprint") \
            from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Forecast:
    series: str
    model: str
    candidate: str
    feature_set: str
    seed: int | None
    csv_sha256: str
    as_of: str
    target_time: str
    horizon_bars: int
    predicted_log_return: float
    split: str = "test"
    fold: int | None = None
    target_kind: str = CLOSE_RETURN_TARGET

    @classmethod
    def parse(cls, value: object) -> Forecast:
        if not isinstance(value, dict):
            raise ValueError("forecast record has an invalid schema")
        schema = value.get("schema")
        fields = LEDGER_V1_FIELDS if schema == 1 else \
            LEDGER_V2_FIELDS if schema == 2 else None
        if type(schema) is not int or fields is None or set(value) != fields:
            raise ValueError("forecast record has an invalid schema")
        split = value["split"]
        fold = None if schema == 1 else value["fold"]
        target_kind = CLOSE_RETURN_TARGET if schema == 1 else value["target_kind"]
        if split not in ("validation", "test") or schema == 1 and split != "test" or \
           split == "validation" and (type(fold) is not int or fold < 0) or \
           split == "test" and fold is not None or target_kind not in TARGET_KINDS:
            raise ValueError("forecast split, fold, or target kind is invalid")
        seed, horizon = value["seed"], value["horizon_bars"]
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be null or a nonnegative integer")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon_bars must be a positive integer")
        as_of, target = value["as_of"], value["target_time"]
        if not isinstance(as_of, str) or not isinstance(target, str):
            raise ValueError("forecast timestamps must be strings")
        csv_sha256 = value["csv_sha256"]
        if not _digest(csv_sha256):
            raise ValueError("csv_sha256 must be a lowercase SHA-256 digest")
        return cls(
            _name(value["series"], "series"), _name(value["model"], "model"),
            _name(value["candidate"], "candidate"),
            _name(value["feature_set"], "feature_set"), seed, csv_sha256,
            as_of, target, horizon,
            _finite(value["predicted_log_return"], "predicted_log_return"),
            split, fold, target_kind,
        )


@dataclass(frozen=True)
class Bars:
    path: str
    sha256: str
    timestamps: tuple[str, ...]
    opens: tuple[float, ...]
    closes: tuple[float, ...]


@dataclass(frozen=True)
class Costs:
    spread_bps: float
    slippage_bps: float
    fee_bps: float

    def __post_init__(self) -> None:
        for field in ("spread_bps", "slippage_bps", "fee_bps"):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value < 10_000.0:
                raise ValueError(f"{field} must be in [0, 10000)")
        if self.impact >= 1.0:
            raise ValueError("spread and slippage make the exit price nonpositive")

    @property
    def impact(self) -> float:
        # Each execution pays half the quoted spread plus all slippage.
        return self.spread_bps / 20_000.0 + self.slippage_bps / 10_000.0

    @property
    def fee(self) -> float:
        return self.fee_bps / 10_000.0

    @property
    def break_even_log_return(self) -> float:
        """Return the gross log return that exactly pays round-trip friction."""
        return math.log(
            (1.0 + self.impact) * (1.0 + self.fee) /
            ((1.0 - self.impact) * (1.0 - self.fee))
        )


def select_trial(trials: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Choose maximum growth, then minimum turnover, then safest threshold."""
    return max(
        trials,
        key=lambda item: (
            item["objective"], -item["mean_gross_turnover"],
            math.inf if item["safety_bps"] is None else item["safety_bps"],
        ),
    )


def _validate_trial(value: object, initial_cash: float) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != TRIAL_FIELDS or \
       value["action"] not in ("cash", "long_above") or \
       type(value["trade_count"]) is not int or value["trade_count"] < 0:
        raise ValueError("policy threshold trial is invalid")
    objective = _finite(value["objective"], "trial objective")
    final = _finite(value["mean_final_equity"], "trial mean_final_equity")
    turnover = _finite(value["mean_gross_turnover"],
                       "trial mean_gross_turnover")
    signal = _finite(value["signal_coverage"], "trial signal_coverage")
    execution = _finite(value["execution_coverage"],
                        "trial execution_coverage")
    safety = value["safety_bps"]
    if final <= 0.0 or turnover < 0.0 or not 0.0 <= execution <= signal <= 1.0:
        raise ValueError("policy threshold trial metrics are invalid")
    if value["action"] == "cash":
        if safety is not None or objective != 0.0 or turnover != 0.0 or \
           signal != 0.0 or execution != 0.0 or value["trade_count"] != 0 or \
           not math.isclose(final, initial_cash, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("policy cash trial is invalid")
    elif not 0.0 <= _finite(safety, "trial safety_bps") < 10_000.0:
        raise ValueError("policy trial threshold is invalid")
    return value


def validate_policy(value: object) -> dict[str, object]:
    """Validate and return one frozen executable-return policy."""
    if not isinstance(value, dict) or set(value) != POLICY_FIELDS or \
       type(value.get("schema")) is not int or value["schema"] != 1:
        raise ValueError("policy has an invalid schema")
    action = value["action"]
    if action not in ("cash", "long_above") or \
       value["target_kind"] != EXECUTABLE_RETURN_TARGET:
        raise ValueError("policy action or target kind is invalid")
    for field in ("model", "candidate", "feature_set"):
        _name(value[field], field)
    horizon = value["horizon_bars"]
    seeds, names = value["seeds"], value["series"]
    if type(horizon) is not int or horizon < 1 or not isinstance(seeds, list) or \
       any(type(seed) is not int or seed < 0 for seed in seeds) or \
       seeds != sorted(set(seeds)) or not isinstance(names, list) or not names or \
       any(not isinstance(name, str) or not NAME.fullmatch(name) for name in names) or \
       len(names) != len(set(names)):
        raise ValueError("policy horizon, seeds, or series are invalid")
    costs_value = value["costs"]
    if not isinstance(costs_value, dict) or \
       set(costs_value) != set(POLICY_COST_FIELDS):
        raise ValueError("policy costs are invalid")
    costs = Costs(*(costs_value[field] for field in POLICY_COST_FIELDS))
    initial_cash = _finite(value["initial_cash"], "initial_cash")
    safety = value["safety_bps"]
    minimum = value["minimum_predicted_log_return"]
    if initial_cash <= 0.0 or action == "cash" and \
       (safety is not None or minimum is not None):
        raise ValueError("cash policy parameters are invalid")
    if action == "long_above":
        safety = _finite(safety, "safety_bps")
        minimum = _finite(minimum, "minimum_predicted_log_return")
        expected = costs.break_even_log_return + safety / 10_000.0
        if not 0.0 <= safety < 10_000.0 or \
           not math.isclose(minimum, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("policy threshold is inconsistent with its costs")
    trials_value = value["threshold_trials"]
    if value["selection_objective"] != "macro_mean_terminal_log_growth" or \
       not isinstance(trials_value, list) or not trials_value:
        raise ValueError("policy selection evidence is invalid")
    trials = tuple(_validate_trial(item, initial_cash) for item in trials_value)
    cash = tuple(item for item in trials if item["action"] == "cash")
    safeties = tuple(item["safety_bps"] for item in trials
                     if item["action"] == "long_above")
    winner = select_trial(trials)
    if len(cash) != 1 or not safeties or len(safeties) != len(set(safeties)) or \
       action != winner["action"] or safety != winner["safety_bps"]:
        raise ValueError("policy selection is inconsistent with its trials")
    if not _digest(value["validation_fingerprint"]):
        raise ValueError("policy validation fingerprint is invalid")
    report = value["validation_report"]
    ledger = value["validation_prediction_ledger"]
    if not isinstance(report, dict) or set(report) != {"path", "sha256"} or \
       not isinstance(report["path"], str) or not report["path"] or \
       not _digest(report["sha256"]):
        raise ValueError("policy validation_report provenance is invalid")
    if not isinstance(ledger, dict) or set(ledger) != {
        "path", "sha256", "source_records", "selected_records",
    } or not isinstance(ledger["path"], str) or not ledger["path"] or \
       not _digest(ledger["sha256"]) or \
       type(ledger["source_records"]) is not int or \
       type(ledger["selected_records"]) is not int or \
       not 0 < ledger["selected_records"] <= ledger["source_records"]:
        raise ValueError("policy validation_prediction_ledger provenance is invalid")
    grid = value["test_grid"]
    if not isinstance(grid, list) or len(grid) != len(names) or any(
        not isinstance(item, dict) or set(item) != {
            "series", "samples", "first_target_time", "last_target_time",
        } or item["series"] not in names or type(item["samples"]) is not int or
        item["samples"] < 1 or not isinstance(item["first_target_time"], str) or
        not isinstance(item["last_target_time"], str) for item in grid
    ) or len({item["series"] for item in grid}) != len(grid):
        raise ValueError("policy test grid is invalid")
    return value


def read_policy(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read policy: {error}") from error
    return validate_policy(value)


def _validate_test_grid(forecasts: Sequence[Forecast],
                        policy: Mapping[str, object]) -> None:
    seeds = tuple(policy["seeds"]) or (None,)
    actual: dict[tuple[object, ...], list[Forecast]] = defaultdict(list)
    for forecast in forecasts:
        actual[(forecast.series, forecast.seed)].append(forecast)
    grid = {item["series"]: item for item in policy["test_grid"]}
    expected = {(series, seed) for series in grid for seed in seeds}
    if set(actual) != expected:
        raise ValueError("test ledger does not cover the policy seed grid")
    for (series, _), rows in actual.items():
        rows.sort(key=lambda item: item.target_time)
        contract = grid[series]
        if len(rows) != contract["samples"] or \
           rows[0].target_time != contract["first_target_time"] or \
           rows[-1].target_time != contract["last_target_time"]:
            raise ValueError("test ledger does not match the policy boundaries")


def validate_test_experiment(value: object, ledger_hash: str,
                             forecasts: Sequence[Forecast], policy_hash: str,
                             policy: Mapping[str, object]) -> Mapping[str, object]:
    """Require one strict report that authorizes every model in its test ledger."""
    if not isinstance(value, dict) or type(value.get("schema")) is not int or \
       value["schema"] != 5 or not isinstance(value.get("protocol"), dict) or \
       value["protocol"].get("phase") != "validation-and-test" or \
       experiment_fingerprint(value) != policy["validation_fingerprint"]:
        raise ValueError("test experiment does not match the policy")
    entries = value.get("policies")
    if not isinstance(entries, list) or not entries:
        raise ValueError("test experiment policy authorizations are invalid")
    authorizations: dict[str, str] = {}
    hashes: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "path", "sha256", "model",
        } or not isinstance(item["path"], str) or not item["path"] or \
           not _digest(item["sha256"]) or not isinstance(item["model"], str) or \
           not NAME.fullmatch(item["model"]) or \
           item["model"] in authorizations or item["sha256"] in hashes:
            raise ValueError("test experiment policy authorizations are invalid")
        authorizations[item["model"]] = item["sha256"]
        hashes.add(item["sha256"])
    metadata = value.get("prediction_ledger")
    tests = value.get("test")
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema", "path", "records", "sha256",
    } or type(metadata["schema"]) is not int or metadata["schema"] != 2 or \
       not isinstance(metadata["path"], str) or not metadata["path"] or \
       type(metadata["records"]) is not int or metadata["records"] < 1 or \
       not _digest(metadata["sha256"]) or metadata["sha256"] != ledger_hash or \
       metadata["records"] != len(forecasts) or not isinstance(tests, list) or \
       not tests or any(not isinstance(item, dict) or
                        not isinstance(item.get("model"), str)
                        for item in tests):
        raise ValueError("test experiment ledger metadata is invalid")
    authorized = set(authorizations)
    if authorizations.get(policy["model"]) != policy_hash or \
       {item.model for item in forecasts} != authorized or \
       {item["model"] for item in tests} != authorized:
        raise ValueError("test experiment models are not exactly authorized")
    return value


@dataclass(frozen=True)
class Trade:
    entry: int
    exit: int
    shares: float
    entry_notional: float
    exit_notional: float
    cash_before: float
    cash_after: float
    signal: float | None


def read_forecasts(path: Path) -> tuple[Forecast, ...]:
    forecasts = []
    with path.open("r", encoding="utf-8") as file:
        number = 0
        while line := file.readline(LINE_CAP + 1):
            number += 1
            if len(line) > LINE_CAP or not line.strip():
                raise ValueError(f"forecast line {number} is invalid")
            try:
                forecasts.append(Forecast.parse(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"forecast line {number}: {error}") from error
    if not forecasts:
        raise ValueError("forecast ledger is empty")
    return tuple(forecasts)


def load_bars(path: Path) -> Bars:
    checksum = _sha256(path)
    timestamps, values = read_bars(path)
    if _sha256(path) != checksum:
        raise ValueError("CSV changed while the backtest was reading it")
    opens = tuple(values[0::FEATURE_COUNT])
    closes = tuple(values[3::FEATURE_COUNT])
    if len(timestamps) < 2 or any(
        value <= 0.0 for column in (opens, closes) for value in column
    ):
        raise ValueError("backtest bars require positive opens and closes")
    return Bars(str(path), checksum, timestamps, opens, closes)


def _group_key(forecast: Forecast) -> tuple[object, ...]:
    return (forecast.series, forecast.model, forecast.candidate,
            forecast.feature_set, forecast.split, forecast.fold, forecast.seed,
            forecast.horizon_bars, forecast.target_kind)


def _sort_key(key: tuple[object, ...]) -> tuple[object, ...]:
    return (*key[:5], -1 if key[5] is None else key[5],
            -1 if key[6] is None else key[6], *key[7:])


def _ensemble_key(forecast: Forecast) -> tuple[object, ...]:
    return (forecast.series, forecast.model, forecast.candidate,
            forecast.feature_set, forecast.split, forecast.fold,
            forecast.horizon_bars, forecast.target_kind)


def _aggregate_seeds(forecasts: Sequence[Forecast],
                     expected_seeds: Sequence[int] | None = None,
                     ) -> tuple[tuple[Forecast, ...],
                                dict[tuple[object, ...], tuple[int | None, ...]]]:
    """Average complete, grid-aligned seed streams into one signal path."""
    streams: dict[tuple[object, ...], dict[int | None, list[Forecast]]] = \
        defaultdict(lambda: defaultdict(list))
    for forecast in forecasts:
        streams[_ensemble_key(forecast)][forecast.seed].append(forecast)
    expected, seeds_by_group, averaged = {}, {}, []
    for key in sorted(streams, key=lambda item: (
        *item[:5], -1 if item[5] is None else item[5], *item[6:],
    )):
        by_seed = streams[key]
        seeds = tuple(sorted(by_seed, key=lambda seed: -1 if seed is None else seed))
        signature = (*key[1:5], *key[6:])
        required = tuple(expected_seeds) if expected_seeds else (None,)
        if expected_seeds is not None and seeds != required or \
           None in seeds and len(seeds) != 1 or \
           signature in expected and expected[signature] != seeds:
            raise ValueError("ensemble groups must share one complete seed set")
        expected[signature] = seeds
        rows = [sorted(by_seed[seed], key=lambda item: item.as_of) for seed in seeds]
        grid = tuple((item.csv_sha256, item.as_of, item.target_time)
                     for item in rows[0])
        if any(tuple((item.csv_sha256, item.as_of, item.target_time)
                     for item in stream) != grid for stream in rows[1:]):
            raise ValueError("ensemble seed streams must share one forecast grid")
        averaged.extend(
            replace(items[0], seed=None,
                    predicted_log_return=fmean(item.predicted_log_return
                                               for item in items))
            for items in zip(*rows, strict=True)
        )
        seeds_by_group[key] = seeds
    return tuple(averaged), seeds_by_group


def _align(forecasts: Sequence[Forecast], bars: Bars,
           indexes: Mapping[str, int]
           ) -> tuple[tuple[Forecast, int, int], ...]:
    aligned, previous = [], None
    for forecast in forecasts:
        try:
            as_of, target = indexes[forecast.as_of], indexes[forecast.target_time]
        except KeyError as error:
            raise ValueError(f"forecast timestamp is missing from {forecast.series}") \
                from error
        if target != as_of + forecast.horizon_bars:
            raise ValueError("target_time does not match horizon_bars")
        if previous is not None and as_of != previous + 1:
            raise ValueError("each forecast group must cover one contiguous holdout")
        aligned.append((forecast, as_of, target))
        previous = as_of
    return tuple(aligned)


def _execute(cash: float, entry: int, exit_: int,
             bars: Bars, costs: Costs, signal: float | None = None) -> Trade:
    entry_price = bars.opens[entry] * (1.0 + costs.impact)
    shares = cash / (entry_price * (1.0 + costs.fee))
    exit_notional = shares * bars.closes[exit_] * (1.0 - costs.impact)
    return Trade(entry, exit_, shares, shares * entry_price, exit_notional,
                 cash, exit_notional * (1.0 - costs.fee), signal)


def _schedule(aligned: Sequence[tuple[Forecast, int, int]], bars: Bars,
              costs: Costs, initial_cash: float, always_up: bool,
              minimum_return: float | None = 0.0
              ) -> tuple[tuple[Trade, ...], int]:
    trades, cash, unavailable_before, decisions = [], initial_cash, -1, 0
    for forecast, as_of, target in aligned:
        # A signal at the prior exit close may enter at the following open.
        if as_of < unavailable_before:
            continue
        decisions += 1
        if not always_up and (minimum_return is None or
                              forecast.predicted_log_return <= minimum_return):
            continue
        signal = None if always_up else forecast.predicted_log_return
        trade = _execute(cash, as_of + 1, target, bars, costs, signal)
        trades.append(trade)
        cash, unavailable_before = trade.cash_after, target
    return tuple(trades), decisions


def _curve(bars: Bars, start: int, end: int, trades: Sequence[Trade],
           costs: Costs, initial_cash: float) -> tuple[tuple[str, float], ...]:
    entries = {trade.entry: trade for trade in trades}
    cash, active, points = initial_cash, None, []
    for index in range(start, end + 1):
        if trade := entries.get(index):
            if active is not None:
                raise ValueError("trades overlap")
            active, cash = trade, 0.0
        if active is not None and index == active.exit:
            cash, active = active.cash_after, None
            equity = cash
        elif active is not None:
            equity = active.shares * bars.closes[index] * \
                (1.0 - costs.impact) * (1.0 - costs.fee)
        else:
            equity = cash
        points.append((bars.timestamps[index], equity))
    if active is not None:
        raise ValueError("position remains open after the evaluation range")
    return tuple(points)


def _period_keys(timestamp: str) -> tuple[str, str, str]:
    day = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    year, week, _ = day.isocalendar()
    return timestamp[:10], f"{year}-W{week:02d}", timestamp[:7]


def _periods(curve: Sequence[tuple[str, float]], initial_cash: float
             ) -> dict[str, list[dict[str, object]]]:
    endpoints: dict[str, list[tuple[str, float]]] = {
        "daily": [], "weekly": [], "monthly": [],
    }
    for timestamp, equity in curve:
        keys = _period_keys(timestamp)
        for name, key in zip(endpoints, keys, strict=True):
            if endpoints[name] and endpoints[name][-1][0] == key:
                endpoints[name][-1] = (key, equity)
            else:
                endpoints[name].append((key, equity))
    result = {}
    for name, values in endpoints.items():
        previous, records = initial_cash, []
        for period, equity in values:
            records.append({
                "period": period, "ending_equity": equity,
                "period_return": equity / previous - 1.0,
                "cumulative_return": equity / initial_cash - 1.0,
            })
            previous = equity
        result[name] = records
    return result


def _summary(bars: Bars, start: int, end: int, trades: Sequence[Trade],
             costs: Costs, initial_cash: float) -> dict[str, object]:
    curve = _curve(bars, start, end, trades, costs, initial_cash)
    peak, drawdown = initial_cash, 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        drawdown = max(drawdown, 1.0 - equity / peak)
    final = curve[-1][1]
    return {
        "initial_equity": initial_cash, "final_equity": final,
        "net_return": final / initial_cash - 1.0,
        "bar_close_max_drawdown": drawdown, "trade_count": len(trades),
        "winning_trade_rate": (
            sum(trade.cash_after > trade.cash_before for trade in trades) /
            len(trades) if trades else None
        ),
        "gross_turnover": sum(
            trade.entry_notional + trade.exit_notional for trade in trades
        ) / initial_cash,
        "periods": _periods(curve, initial_cash),
        "trades": [
            {
                "as_of": bars.timestamps[trade.entry - 1],
                "predicted_log_return": trade.signal,
                "entry_time": bars.timestamps[trade.entry],
                "exit_time": bars.timestamps[trade.exit],
                "entry_open": bars.opens[trade.entry],
                "exit_close": bars.closes[trade.exit],
                "entry_execution_price": trade.entry_notional / trade.shares,
                "exit_execution_price": trade.exit_notional / trade.shares,
                "shares": trade.shares,
                "entry_notional": trade.entry_notional,
                "exit_notional": trade.exit_notional,
                "cash_before": trade.cash_before,
                "cash_after": trade.cash_after,
                "net_return": trade.cash_after / trade.cash_before - 1.0,
            }
            for trade in trades
        ],
    }


def run_backtests(forecasts: Sequence[Forecast], series: Mapping[str, Bars],
                  initial_cash: float, costs: Costs, safety_bps: float = 0.0,
                  ensemble_seeds: bool = False,
                  expected_seeds: Sequence[int] | None = None,
                  cash_only: bool = False) -> dict[str, object]:
    initial_cash = _finite(initial_cash, "initial_cash")
    safety_bps = _finite(safety_bps, "safety_bps")
    if initial_cash <= 0.0 or not 0.0 <= safety_bps < 10_000.0:
        raise ValueError("initial_cash or safety_bps is invalid")
    splits, target_kinds = ({forecast.split for forecast in forecasts},
                            {forecast.target_kind for forecast in forecasts})
    if len(splits) != 1 or not splits <= {"validation", "test"} or \
       len(target_kinds) != 1 or not target_kinds <= set(TARGET_KINDS) or \
       expected_seeds is not None and not ensemble_seeds:
        raise ValueError("forecast ledger must use one split and target kind")
    seed_groups = {}
    if ensemble_seeds:
        forecasts, seed_groups = _aggregate_seeds(forecasts, expected_seeds)
    groups: dict[tuple[object, ...], list[Forecast]] = defaultdict(list)
    for forecast in forecasts:
        groups[_group_key(forecast)].append(forecast)
    if not groups:
        raise ValueError("forecasts must be nonempty")
    names = {str(key[0]) for key in groups}
    if names != set(series):
        raise ValueError("forecast and CSV series names must match exactly")
    indexes = {
        name: {timestamp: index for index, timestamp in enumerate(bars.timestamps)}
        for name, bars in series.items()
    }
    aligned_groups, grids = {}, {}
    for key in sorted(groups, key=_sort_key):
        name, _, _, _, split, fold, _, horizon, target_kind = key
        bars = series[str(name)]
        if any(forecast.csv_sha256 != bars.sha256 for forecast in groups[key]):
            raise ValueError("forecast CSV hash does not match the supplied series")
        aligned = _align(groups[key], bars, indexes[str(name)])
        grid = tuple((item.as_of, item.target_time) for item, _, _ in aligned)
        grid_key = (name, split, fold, horizon, target_kind)
        if grid_key in grids and grids[grid_key] != grid:
            raise ValueError("model groups must share one evaluation grid")
        grids[grid_key], aligned_groups[key] = grid, aligned
    threshold = None if cash_only else \
        costs.break_even_log_return + safety_bps / 10_000.0
    results = []
    for key in sorted(groups, key=_sort_key):
        name, model, candidate, feature_set, split, fold, seed, horizon, \
            target_kind = key
        bars, aligned = series[str(name)], aligned_groups[key]
        start, end = aligned[0][1] + 1, aligned[-1][2]
        signals = sum(
            threshold is not None and item[0].predicted_log_return > threshold
            for item in aligned
        )
        model_trades, decisions = _schedule(
            aligned, bars, costs, initial_cash, False, threshold,
        )
        always_up, _ = _schedule(aligned, bars, costs, initial_cash, True)
        buy_hold = (_execute(initial_cash, start, end, bars, costs),)
        forecast_summary = _summary(
            bars, start, end, model_trades, costs, initial_cash,
        )
        forecast_summary.update({
            "decision_count": decisions,
            "signal_coverage": signals / len(aligned),
            "execution_coverage": len(model_trades) / len(aligned),
            "eligible_entry_hit_rate": (
                len(model_trades) / decisions if decisions else 0.0
            ),
            "mean_net_return_per_trade": (
                fmean(trade.cash_after / trade.cash_before - 1.0
                      for trade in model_trades) if model_trades else None
            ),
        })
        result = {
            "series": name, "model": model, "candidate": candidate,
            "feature_set": feature_set, "split": split, "fold": fold,
            "horizon_bars": horizon, "target_kind": target_kind,
            "evaluation": {
                "forecasts": len(aligned),
                "first_as_of": aligned[0][0].as_of,
                "first_entry_time": bars.timestamps[start],
                "last_target_time": aligned[-1][0].target_time,
            },
            "strategies": {
                "forecast_long_cash": forecast_summary,
                "always_up": _summary(
                    bars, start, end, always_up, costs, initial_cash,
                ),
                "buy_and_hold": _summary(
                    bars, start, end, buy_hold, costs, initial_cash,
                ),
                "cash": _summary(bars, start, end, (), costs, initial_cash),
            },
        }
        if ensemble_seeds:
            seeds = seed_groups[_ensemble_key(aligned[0][0])]
            result.update({
                "seed_aggregation": (
                    "deterministic" if seeds == (None,) else "arithmetic_mean"
                ),
                "seeds": [value for value in seeds if value is not None],
            })
        else:
            result["seed"] = seed
        results.append(result)
    split, target_kind = next(iter(splits)), next(iter(target_kinds))
    return {
        "schema": 2,
        "protocol": {
            "performance_kind": f"hypothetical {split} backtest",
            "split": split,
            "position": "long or cash; fractional shares; no leverage",
            "sizing": "100% of available equity per entry",
            "signal": (
                "cash" if cash_only else
                "long when predicted_log_return exceeds the threshold"
            ),
            "forecast_target": TARGET_FORMULAS[target_kind],
            "target_kind": target_kind,
            "traded_return": "close[t + horizon] / open[t + 1]",
            "entry": "next bar open after the completed as_of bar",
            "exit": "close horizon_bars after the as_of bar",
            "overlap": "ignore forecasts made before an open position exits",
            "mark_to_market": "cost-adjusted liquidation value at each bar close",
            "period_timezone": "UTC",
            "period_scope": "first and last periods may be partial",
            "allocation": (
                "each series and validation fold independently starts with "
                "initial_cash"
            ),
            "seed_aggregation": (
                "arithmetic mean per timestamp" if ensemble_seeds else "none"
            ),
            "cost_break_even_log_return": costs.break_even_log_return,
            "threshold_semantics": (
                "not applicable" if cash_only else
                "exact for the executable entry and exit prices" if
                target_kind == EXECUTABLE_RETURN_TARGET else
                "heuristic because the target excludes the next-open gap"
            ),
            "safety_bps": safety_bps,
            "minimum_predicted_log_return": threshold,
            "cash_yield": 0.0,
            "dividends": "not credited separately from supplied OHLCV prices",
            "gross_turnover": "entry plus exit notional divided by initial_cash",
            "initial_cash": initial_cash,
            "costs": {
                "full_spread_bps": costs.spread_bps,
                "slippage_bps_per_side": costs.slippage_bps,
                "fee_bps_per_side": costs.fee_bps,
            },
        },
        "series": [
            {"name": name, "csv": bars.path, "sha256": bars.sha256,
             "first_timestamp": bars.timestamps[0],
             "last_timestamp": bars.timestamps[-1]}
            for name, bars in sorted(series.items())
        ],
        "results": results,
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    write_json(path, report)


def _series(value: str) -> tuple[str, Path]:
    return series_arg(value, NAME)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("series", nargs="+", type=_series, metavar="NAME=CSV")
    parser.add_argument("--initial-cash", type=float)
    parser.add_argument("--spread-bps", type=float)
    parser.add_argument("--slippage-bps", type=float)
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--safety-bps", type=float)
    parser.add_argument("--ensemble-seeds", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--experiment-report", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        inputs = [args.predictions, *(path for _, path in args.series)]
        if args.policy:
            inputs.append(args.policy)
        if args.experiment_report:
            inputs.append(args.experiment_report)
        require_disjoint(
            inputs, [args.report],
        )
        paths = dict(args.series)
        if len(paths) != len(args.series):
            raise ValueError("series names must be unique")
        bars = {name: load_bars(path) for name, path in paths.items()}
        ledger_hash = _sha256(args.predictions)
        source = read_forecasts(args.predictions)
        forecasts = source
        policy, policy_hash, experiment_hash = None, None, None
        if args.policy:
            if any(value is not None for value in (
                args.initial_cash, args.spread_bps, args.slippage_bps,
                args.fee_bps, args.safety_bps, args.model,
            )) or args.ensemble_seeds:
                raise ValueError("policy mode does not accept diagnostic overrides")
            policy_hash = _sha256(args.policy)
            policy = read_policy(args.policy)
            if _sha256(args.policy) != policy_hash:
                raise ValueError("policy changed while the backtest was reading it")
            if {item.split for item in source} != {"test"} or \
               set(paths) != set(policy["series"]) or not args.experiment_report:
                raise ValueError("policy mode requires its exact test series")
            experiment_hash = _sha256(args.experiment_report)
            try:
                experiment = json.loads(
                    args.experiment_report.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read experiment report: {error}") from error
            experiment = validate_test_experiment(
                experiment, ledger_hash, source, policy_hash, policy,
            )
            forecasts = tuple(item for item in source if
                              item.model == policy["model"] and
                              item.candidate == policy["candidate"])
            if not forecasts or any(
                item.feature_set != policy["feature_set"] or
                item.target_kind != policy["target_kind"] or
                item.horizon_bars != policy["horizon_bars"]
                for item in forecasts
            ):
                raise ValueError("test forecasts do not match the policy")
            _validate_test_grid(forecasts, policy)
            costs = Costs(*(policy["costs"][field]
                            for field in POLICY_COST_FIELDS))
            initial_cash = float(policy["initial_cash"])
            safety_bps = float(policy["safety_bps"] or 0.0)
            ensemble_seeds, expected_seeds = True, policy["seeds"]
            cash_only = policy["action"] == "cash"
        else:
            if args.experiment_report:
                raise ValueError("experiment reports are only accepted with policies")
            if {item.split for item in source} != {"validation"}:
                raise ValueError("test backtests require a frozen policy")
            if any(value is None for value in (
                args.spread_bps, args.slippage_bps, args.fee_bps,
            )):
                raise ValueError("diagnostic costs are required")
            costs = Costs(args.spread_bps, args.slippage_bps, args.fee_bps)
            initial_cash = args.initial_cash if args.initial_cash is not None else 100.0
            safety_bps = args.safety_bps if args.safety_bps is not None else 0.0
            ensemble_seeds, expected_seeds, cash_only = \
                args.ensemble_seeds, None, False
        if args.model:
            model = _name(args.model, "model")
            forecasts = tuple(item for item in forecasts if item.model == model)
            if not forecasts:
                raise ValueError("forecast ledger has no records for model")
        if _sha256(args.predictions) != ledger_hash:
            raise ValueError("prediction ledger changed while it was being read")
        report = run_backtests(
            forecasts, bars, initial_cash, costs, safety_bps,
            ensemble_seeds, expected_seeds, cash_only,
        )
        if any(_sha256(Path(item.path)) != item.sha256 for item in bars.values()):
            raise ValueError("CSV changed during the backtest")
        report["prediction_ledger"] = {
            "path": str(args.predictions), "sha256": ledger_hash,
            "source_records": len(source), "selected_records": len(forecasts),
        }
        if policy is not None and policy_hash is not None:
            if _sha256(args.policy) != policy_hash or \
               _sha256(args.experiment_report) != experiment_hash:
                raise ValueError("policy inputs changed during the backtest")
            report["policy"] = {"path": str(args.policy), "sha256": policy_hash}
            report["experiment_report"] = {
                "path": str(args.experiment_report), "sha256": experiment_hash,
            }
        write_report(args.report, report)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"report": str(args.report),
                      "results": len(report["results"])}, sort_keys=True))


if __name__ == "__main__":
    main()
