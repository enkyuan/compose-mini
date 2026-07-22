#!/usr/bin/env python3
"""Backtest holdout forecasts with a frozen long-or-cash execution policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import FEATURE_COUNT, read_bars
from tools.files import (
    file_sha256 as _sha256, require_disjoint, series_arg, write_json,
)

NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
LEDGER_FIELDS = frozenset((
    "schema", "split", "series", "model", "candidate", "feature_set",
    "seed", "csv_sha256", "as_of", "target_time", "horizon_bars",
    "predicted_log_return",
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

    @classmethod
    def parse(cls, value: object) -> Forecast:
        if not isinstance(value, dict) or set(value) != LEDGER_FIELDS:
            raise ValueError("forecast record has an invalid schema")
        if type(value["schema"]) is not int or value["schema"] != 1 or \
           value["split"] != "test":
            raise ValueError("forecast record must be schema 1 test data")
        seed, horizon = value["seed"], value["horizon_bars"]
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be null or a nonnegative integer")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon_bars must be a positive integer")
        as_of, target = value["as_of"], value["target_time"]
        if not isinstance(as_of, str) or not isinstance(target, str):
            raise ValueError("forecast timestamps must be strings")
        csv_sha256 = value["csv_sha256"]
        if not isinstance(csv_sha256, str) or \
           not re.fullmatch(r"[0-9a-f]{64}", csv_sha256):
            raise ValueError("csv_sha256 must be a lowercase SHA-256 digest")
        return cls(
            _name(value["series"], "series"), _name(value["model"], "model"),
            _name(value["candidate"], "candidate"),
            _name(value["feature_set"], "feature_set"), seed, csv_sha256,
            as_of, target, horizon,
            _finite(value["predicted_log_return"], "predicted_log_return"),
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
            forecast.feature_set, forecast.seed, forecast.horizon_bars)


def _sort_key(key: tuple[object, ...]) -> tuple[object, ...]:
    return (*key[:4], -1 if key[4] is None else key[4], key[5])


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
              costs: Costs, initial_cash: float, always_up: bool
              ) -> tuple[Trade, ...]:
    trades, cash, unavailable_before = [], initial_cash, -1
    for forecast, as_of, target in aligned:
        # A signal at the prior exit close may enter at the following open.
        if as_of < unavailable_before or \
           not always_up and forecast.predicted_log_return <= 0.0:
            continue
        signal = None if always_up else forecast.predicted_log_return
        trade = _execute(cash, as_of + 1, target, bars, costs, signal)
        trades.append(trade)
        cash, unavailable_before = trade.cash_after, target
    return tuple(trades)


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
                  initial_cash: float, costs: Costs) -> dict[str, object]:
    initial_cash = _finite(initial_cash, "initial_cash")
    if initial_cash <= 0.0:
        raise ValueError("initial_cash must be positive")
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
        name, _, _, _, _, horizon = key
        bars = series[str(name)]
        if any(forecast.csv_sha256 != bars.sha256 for forecast in groups[key]):
            raise ValueError("forecast CSV hash does not match the supplied series")
        aligned = _align(groups[key], bars, indexes[str(name)])
        grid = tuple((item.as_of, item.target_time) for item, _, _ in aligned)
        grid_key = (name, horizon)
        if grid_key in grids and grids[grid_key] != grid:
            raise ValueError("model groups must share one evaluation grid")
        grids[grid_key], aligned_groups[key] = grid, aligned
    results = []
    for key in sorted(groups, key=_sort_key):
        name, model, candidate, feature_set, seed, horizon = key
        bars, aligned = series[str(name)], aligned_groups[key]
        start, end = aligned[0][1] + 1, aligned[-1][2]
        model_trades = _schedule(aligned, bars, costs, initial_cash, False)
        always_up = _schedule(aligned, bars, costs, initial_cash, True)
        buy_hold = (_execute(initial_cash, start, end, bars, costs),)
        results.append({
            "series": name, "model": model, "candidate": candidate,
            "feature_set": feature_set, "seed": seed, "horizon_bars": horizon,
            "evaluation": {
                "forecasts": len(aligned),
                "first_as_of": aligned[0][0].as_of,
                "first_entry_time": bars.timestamps[start],
                "last_target_time": aligned[-1][0].target_time,
            },
            "strategies": {
                "forecast_long_cash": _summary(
                    bars, start, end, model_trades, costs, initial_cash,
                ),
                "always_up": _summary(
                    bars, start, end, always_up, costs, initial_cash,
                ),
                "buy_and_hold": _summary(
                    bars, start, end, buy_hold, costs, initial_cash,
                ),
                "cash": _summary(bars, start, end, (), costs, initial_cash),
            },
        })
    return {
        "schema": 1,
        "protocol": {
            "performance_kind": "hypothetical holdout backtest",
            "position": "long or cash; fractional shares; no leverage",
            "sizing": "100% of available equity per entry",
            "signal": "long when predicted_log_return > 0",
            "forecast_target": "close[t + horizon] / close[t]",
            "traded_return": "close[t + horizon] / open[t + 1]",
            "entry": "next bar open after the completed as_of bar",
            "exit": "close horizon_bars after the as_of bar",
            "overlap": "ignore forecasts made before an open position exits",
            "mark_to_market": "cost-adjusted liquidation value at each bar close",
            "period_timezone": "UTC",
            "period_scope": "first and last periods may be partial",
            "allocation": "each result independently starts with initial_cash",
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
    parser.add_argument("--initial-cash", type=float, default=100.0)
    parser.add_argument("--spread-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--fee-bps", type=float, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        require_disjoint(
            [args.predictions, *(path for _, path in args.series)], [args.report],
        )
        paths = dict(args.series)
        if len(paths) != len(args.series):
            raise ValueError("series names must be unique")
        costs = Costs(args.spread_bps, args.slippage_bps, args.fee_bps)
        bars = {name: load_bars(path) for name, path in paths.items()}
        ledger_hash = _sha256(args.predictions)
        forecasts = read_forecasts(args.predictions)
        if _sha256(args.predictions) != ledger_hash:
            raise ValueError("prediction ledger changed while it was being read")
        report = run_backtests(
            forecasts, bars, args.initial_cash, costs,
        )
        if any(_sha256(Path(item.path)) != item.sha256 for item in bars.values()):
            raise ValueError("CSV changed during the backtest")
        report["prediction_ledger"] = {
            "path": str(args.predictions), "sha256": ledger_hash,
            "records": len(forecasts),
        }
        write_report(args.report, report)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"report": str(args.report),
                      "results": len(report["results"])}, sort_keys=True))


if __name__ == "__main__":
    main()
