# Forward Interval Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the paired-MSE gain that the fixed 60-session SPY-residual
forward test's 20-session interval can detect with approximately 80% marginal
power, without reading a future label or changing the frozen experiment.

**Architecture:** Extend the existing stock-macro circular-block sampler so it
can draw a predetermined number of sessions over an explicit market-session
grid, while preserving all current complete-panel interval results. Missing
stock-session cells retain zero counts; they are never imputed. Add one
authenticated analyzer mode that derives the already frozen direction-gated
candidate on the completed calibration phase, centers its paired losses, and
publishes a development-only sensitivity report. The mode authenticates the
tracked forward config, derives every experiment constant from that contract,
and never imports the forward bundle, Massive client, Torch runtime, price
reconstruction, or backtester.

**Tech Stack:** Python 3.12 standard library, existing residual authentication
and stock-macro circular-block primitives, direct Python test scripts,
GitButler.

## Global Constraints

- Apply the Karpathy Guidelines and Ponytail discipline: test first, keep the
  diff surgical, reuse current primitives, and add no speculative abstraction.
- Keep the frozen candidate, scale `0.4029492434939931`, universe, 60-session
  window, 20-session decision block, 10,000 replicates, seed `20260725`, gates,
  and optional-stopping prohibition unchanged.
- Freeze and hash the tracked forward config. Bind the analyzer, gate, forward
  contract, cross-sectional seed, and circular-block sampler sources to the
  implementation commit.
- Read only the authenticated historical calibration truth, predictions, and
  completed-as-of SPY regime labels.
- Do not fetch Massive data, inspect a partial forward feature, open a future
  label, import Torch, reconstruct a price, execute a policy, or run a new
  `$100` backtest.
- Treat stocks as one fixed panel and the ordered market calendar as the
  resampling axis. Use the same circular block starts and authenticated
  missingness mask for every stock; never compress separated common dates,
  impute a missing cell, or treat 30-minute rows or stocks as independent.
- Label the output `development-planning-only`. It is marginal sensitivity for
  each paired interval, not the power of the five-gate conjunction and not an
  expected future improvement.
- Keep every residual execution, forward, backtest, universe-expansion, and
  trading lock false.
- Write generated reports only below ignored `reports/`; do not track market
  data, credentials, model state, caches, or generated reports.
- Preserve unrelated working-tree changes in `Makefile`, `docs/training.md`,
  and the existing experiment JSON files.

---

### Task 1: Generalize the Existing Circular-Block Sampler

**Files:**

- Modify: `tools/universe_scaling.py:235`
- Test: `tests/python/test_universe_scaling.py:128`

**Interfaces:**

- Consumes:
  `Mapping[str, Mapping[str, Sequence[float]]]` with finite observations and
  either its nonempty common date grid or an explicit ordered union grid.
- Produces:
  `circular_block_means(values, block_days, *, session_dates=None,
  sample_days=None, replicates=BOOTSTRAP_REPLICATES,
  seed=BOOTSTRAP_SEED) -> tuple[float, ...]`.
- Preserves:
  `circular_block_interval(values, block_days, replicates, seed) ->
  tuple[float, float]` bit-for-bit for all currently valid inputs.

- [ ] **Step 1: Write the failing resampling tests**

Import `circular_block_means` beside `circular_block_interval`. In
`tests/python/test_universe_scaling.py`, add this independent reproduction of
the pre-extraction loop:

```python
def reference_block_means(
    blocks: dict[str, dict[str, tuple[float, ...]]],
    width: int,
    sample_days: int,
    replicates: int,
    seed: int,
) -> tuple[float, ...]:
    """Reproduce the original circular-block loop as a test oracle."""
    dates = tuple(next(iter(blocks.values())))
    generator, samples = Random(seed), []
    for _ in range(replicates):
        selected = []
        while len(selected) < sample_days:
            start = generator.randrange(len(dates))
            selected.extend(
                (start + offset) % len(dates) for offset in range(width)
            )
        selected = selected[:sample_days]
        samples.append(fmean(
            sum(
                sum(blocks[name][dates[index]]) for index in selected
            ) / sum(
                len(blocks[name][dates[index]]) for index in selected
            )
            for name in sorted(blocks)
        ))
    return tuple(sorted(samples))
```

Use it in `test_bootstrap_and_effective_count()` and add:

```python
    extended = circular_block_means(
        blocks, 5, sample_days=60, replicates=200, seed=7,
    )
    assert extended == reference_block_means(blocks, 5, 60, 200, 7)
    complete = {
        name: dict(tuple(values.items())[:20])
        for name, values in blocks.items()
    }
    for width in (5, 10, 20):
        samples = reference_block_means(complete, width, 20, 200, 7)
        assert circular_block_interval(complete, width, 200, 7) == (
            samples[int(0.025 * 199)], samples[int(0.975 * 199)],
        )
    raises(
        circular_block_means, blocks, 5,
        sample_days=4, replicates=10, seed=7,
    )
```

Extract the existing hand-calculated loop into `reference_block_means()` and
use it for the 21-date tail cases, the 60-session extension, and the
nonconstant 20-date no-remainder cases above. These are exact regression
oracles for the pre-extraction algorithm.

- [ ] **Step 2: Run the test and verify the new import fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_universe_scaling.py
```

Expected: nonzero exit with an import error for
`circular_block_means`.

- [ ] **Step 3: Extract the sorted bootstrap samples**

Replace the body split at `circular_block_interval()` with:

```python
def circular_block_means(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    block_days: int,
    *,
    sample_days: int | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    """Resample one stock-macro mean over shared circular date blocks."""
    dates = _common_dates(values)
    count = len(dates) if sample_days is None else sample_days
    if type(block_days) is not int or type(count) is not int or \
       type(replicates) is not int or type(seed) is not int or \
       not 1 <= block_days <= min(len(dates), count) or replicates < 2:
        raise ValueError("bootstrap parameters are invalid")
    daily = tuple(
        tuple((sum(values[name][day]), len(values[name][day]))
              for day in dates)
        for name in sorted(values)
    )
    full_blocks, remainder = divmod(count, block_days)

    def blocks(stock: Sequence[tuple[float, int]], width: int) -> tuple[
        tuple[float, int], ...
    ]:
        return tuple((
            sum(stock[(start + offset) % len(dates)][0]
                for offset in range(width)),
            sum(stock[(start + offset) % len(dates)][1]
                for offset in range(width)),
        ) for start in range(len(dates)))

    prepared = tuple((
        blocks(stock, block_days),
        blocks(stock, remainder) if remainder else (),
    ) for stock in daily)
    generator, samples = Random(seed), []
    for _ in range(replicates):
        starts = tuple(
            generator.randrange(len(dates)) for _ in range(full_blocks)
        )
        tail = generator.randrange(len(dates)) if remainder else None
        samples.append(fmean(
            (
                sum(full[start][0] for start in starts) +
                (partial[tail][0] if tail is not None else 0.0)
            ) / (
                sum(full[start][1] for start in starts) +
                (partial[tail][1] if tail is not None else 0)
            )
            for full, partial in prepared
        ))
    return tuple(sorted(samples))


def circular_block_interval(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    block_days: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap one paired stock-macro mean over its observed date count."""
    samples = circular_block_means(
        values, block_days, replicates=replicates, seed=seed,
    )
    return (
        samples[int(0.025 * (replicates - 1))],
        samples[int(0.975 * (replicates - 1))],
    )
```

This is an extraction, not a second bootstrap implementation. With
`sample_days=None`, `count == len(dates)`, so the generator calls and arithmetic
remain identical to the current interval function.

- [ ] **Step 4: Run the focused regression test**

Run the command from Step 2.

Expected:

```text
universe scaling tests passed
```

- [ ] **Step 5: Keep Task 1 with the sensitivity checkpoint**

Do not commit Task 1 alone. Its only new caller is Task 2, so retain both tasks
in one coherent implementation checkpoint.

### Task 2: Compute Fixed Marginal Interval Sensitivity

**Files:**

- Modify: `tools/analyze_spy_residual_shrinkage.py:173`
- Modify: `tools/analyze_spy_residual_shrinkage.py:677`
- Test: `tests/python/test_spy_residual_shrinkage.py`

**Interfaces:**

- Consumes:
  the current `_paired_mse_metrics()` truth/prediction structures and the
  existing calibration-phase SPY labels.
- Produces:
  `_paired_mse_days(...) -> tuple[daily_gain, daily_reference_loss]`,
  `direction_gated_predictions(...) -> dict[str, tuple[float, ...]]`, and
  `interval_sensitivity(...) -> dict[str, object]`.
- Reuses:
  `_union_dates()`, `_macro()`, `circular_block_means()`,
  `SPY_DIRECTION_SCALE`, `CROSS_SECTION_SEED`, and
  `BOOTSTRAP_REPLICATES`.

- [ ] **Step 1: Write failing math and boundary tests**

Import `direction_gated_predictions` and `interval_sensitivity` in
`tests/python/test_spy_residual_shrinkage.py`, then add:

```python
def test_direction_gate_uses_only_bound_regime_labels() -> None:
    predictions = {"A": (1.0, -2.0), "B": (3.0, -4.0)}
    regimes = {
        "A": ("negative", "nonnegative"),
        "B": ("nonnegative", "negative"),
    }
    assert direction_gated_predictions(predictions, regimes) == {
        "A": (0.0, -2.0 * analyzer.SPY_DIRECTION_SCALE),
        "B": (3.0 * analyzer.SPY_DIRECTION_SCALE, 0.0),
    }
    rejects(
        direction_gated_predictions, predictions,
        {"A": ("negative",), "B": regimes["B"]},
    )
    rejects(
        direction_gated_predictions, predictions,
        {"B": regimes["B"], "A": regimes["A"]},
    )


def test_interval_sensitivity_uses_fixed_session_blocks() -> None:
    gains = {
        name: {
            f"2026-01-{day:02d}": (float(day + index),)
            for day in range(1, 22)
        }
        for index, name in enumerate(("A", "B"))
    }
    losses = {
        name: {day: (100.0,) for day in values}
        for name, values in gains.items()
    }
    observed = interval_sensitivity(
        gains, losses, target_sessions=60, block_sessions=20,
        replicates=200, seed=7,
    )
    assert observed["source_session_count"] == 21
    assert observed["target_session_count"] == 60
    assert observed["block_sessions"] == 20
    assert observed["replicates"] == 200
    assert observed["seed"] == 7
    assert observed["lower_tail_probability"] == 0.025
    assert observed["marginal_power"] == 0.8
    assert observed["reference_mean_squared_error"] == 100.0
    dates, center = tuple(gains["A"]), 11.5
    generator, samples = Random(7), []
    for _ in range(200):
        starts = tuple(generator.randrange(len(dates)) for _ in range(3))
        selected = tuple(
            (start + offset) % len(dates)
            for start in starts for offset in range(20)
        )
        samples.append(fmean(
            fmean(
                gains[name][dates[index]][0] - center
                for index in selected
            )
            for name in gains
        ))
    samples.sort()
    critical = -samples[int(0.025 * 199)]
    detectable = critical - samples[int(0.20 * 199)]
    assert observed["critical_mean_gain"] == critical
    assert observed["minimum_detectable_mean_gain"] == detectable
    assert observed["minimum_detectable_fraction_of_reference_mse"] == \
        detectable / 100.0
    shifted = {
        name: {
            day: tuple(value + 100.0 for value in values)
            for day, values in by_day.items()
        }
        for name, by_day in gains.items()
    }
    translated = interval_sensitivity(
        shifted, losses, target_sessions=60, block_sessions=20,
        replicates=200, seed=7,
    )
    assert isclose(
        translated["descriptive_development_mean_gain"],
        observed["descriptive_development_mean_gain"] + 100.0,
    )
    for key in ("critical_mean_gain", "minimum_detectable_mean_gain"):
        assert isclose(translated[key], observed[key], abs_tol=1e-12)
    rejects(
        interval_sensitivity, gains, losses,
        target_sessions=19, block_sessions=20, replicates=200, seed=7,
    )
    for changed in (
        {**gains, "B": dict(tuple(gains["B"].items())[:-1])},
        {**gains, "B": {
            **gains["B"], "2026-01-22": (1.0,),
        }},
        {**gains, "B": dict(reversed(tuple(gains["B"].items())))},
    ):
        rejects(
            interval_sensitivity, changed, losses,
            target_sessions=60, block_sessions=20,
            replicates=200, seed=7,
        )
```

Call both tests from `main()`.

- [ ] **Step 2: Run the analyzer tests and verify the imports fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_shrinkage.py
```

Expected: nonzero exit with import errors for the two new functions.

- [ ] **Step 3: Expose daily paired squared-error components once**

Add `_paired_mse_days()` immediately before `_paired_mse_metrics()` and make
the existing metric function consume it:

```python
def _paired_mse_days(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    candidate: str,
    reference: str,
) -> tuple[
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, dict[str, tuple[float, ...]]],
]:
    """Group paired MSE gains and reference losses by target date."""
    forecast = {
        **predictions,
        "zero": {
            series: (0.0,) * len(rows) for series, rows in truth.items()
        },
    }
    if candidate not in forecast or reference not in forecast:
        raise ValueError("paired residual model is missing")
    gains, losses = {}, {}
    for series, rows in truth.items():
        gain_days: dict[str, list[float]] = {}
        loss_days: dict[str, list[float]] = {}
        for row, left, right in zip(
            rows, forecast[candidate][series], forecast[reference][series],
            strict=True,
        ):
            day = _day(row.target)
            candidate_loss = (row.value - left) ** 2
            reference_loss = _finite(
                (row.value - right) ** 2, "reference residual MSE",
            )
            gain_days.setdefault(day, []).append(_finite(
                reference_loss - candidate_loss, "paired residual MSE gain",
            ))
            loss_days.setdefault(day, []).append(reference_loss)
        gains[series] = {
            day: tuple(values) for day, values in gain_days.items()
        }
        losses[series] = {
            day: tuple(values) for day, values in loss_days.items()
        }
    return gains, losses
```

Replace `_paired_mse_metrics()`'s duplicated forecast and grouping block with:

```python
    daily, _ = _paired_mse_days(
        truth, predictions, candidate, reference,
    )
```

Keep its common-date, per-stock, interval, win, loss, and tie calculations
unchanged.

- [ ] **Step 4: Implement the fixed gate and sensitivity equation**

The authenticated replay exposed a structured unbalanced panel. The unseen
cohort has 42 market sessions: 10 stocks have all 538 targets, while KRYS has
447 targets after two missing 30-minute bars propagate through the frozen
history and horizon. KRYS is absent on six sessions and partial on three. The
36 all-stock dates form two separated runs, so compressing them would make
nonadjacent market sessions appear adjacent.

Import `_union_dates`, `_macro`, and `circular_block_means` from
`tools.universe_scaling`, and import `gate_mean_predictions` from
`tools.spy_residual_gate`. Add:

```python
def direction_gated_predictions(
    predictions: Mapping[str, Sequence[float]],
    regimes: Mapping[str, Sequence[str]],
) -> dict[str, tuple[float, ...]]:
    """Apply the frozen causal SPY-direction gate without refitting."""
    if not isinstance(predictions, Mapping) or \
       not isinstance(regimes, Mapping) or \
       tuple(predictions) != tuple(regimes):
        raise ValueError("direction-gate series order changed")
    return {
        series: gate_mean_predictions(values, regimes[series])
        for series, values in predictions.items()
    }


def interval_sensitivity(
    gains: Mapping[str, Mapping[str, Sequence[float]]],
    reference_losses: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    target_sessions: int,
    block_sessions: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Estimate one interval's detectable mean shift from centered history."""
    dates = _union_dates(gains)
    if not isinstance(reference_losses, Mapping) or \
       dates != _union_dates(reference_losses) or \
       tuple(gains) != tuple(reference_losses) or any(
        tuple(gains[name]) != tuple(reference_losses[name])
        for name in gains
    ) or any(
        tuple(by_day) != tuple(day for day in dates if day in by_day)
        for by_day in gains.values()
    ) or not any(
        tuple(by_day) == dates for by_day in gains.values()
    ) or any(
        len(gains[name][day]) != len(reference_losses[name][day])
        for name in gains for day in gains[name]
    ) or any(
        _day(day) != day for day in dates
    ):
        raise ValueError("sensitivity stock-session grid changed")
    sessions = tuple(map(len, gains.values()))
    observations = tuple(
        sum(map(len, by_day.values())) for by_day in gains.values()
    )
    mean_gain = _finite(_macro(gains, dates), "development mean gain")
    reference_mse = _finite(
        _macro(reference_losses, dates), "reference residual MSE",
    )
    if reference_mse <= 0.0:
        raise ValueError("sensitivity reference MSE must be positive")
    centered = {
        name: {
            day: tuple(value - mean_gain for value in values)
            for day, values in by_day.items()
        }
        for name, by_day in gains.items()
    }
    samples = circular_block_means(
        centered, block_sessions, session_dates=dates,
        sample_days=target_sessions, replicates=replicates, seed=seed,
    )
    critical = -samples[int(0.025 * (replicates - 1))]
    detectable = critical - samples[int(0.20 * (replicates - 1))]
    if not isfinite(detectable) or detectable <= 0.0:
        raise ValueError("minimum detectable gain is invalid")
    return {
        "block_sessions": block_sessions,
        "critical_mean_gain": critical,
        "descriptive_development_mean_gain": mean_gain,
        "lower_tail_probability": 0.025,
        "marginal_power": 0.8,
        "minimum_detectable_fraction_of_reference_mse":
            detectable / reference_mse,
        "minimum_detectable_mean_gain": detectable,
        "reference_mean_squared_error": reference_mse,
        "replicates": replicates,
        "seed": seed,
        "source_maximum_observations_per_stock": max(observations),
        "source_maximum_sessions_per_stock": max(sessions),
        "source_minimum_observations_per_stock": min(observations),
        "source_minimum_sessions_per_stock": min(sessions),
        "source_missing_stock_session_count":
            len(gains) * len(dates) - sum(sessions),
        "source_observation_count": sum(observations),
        "source_session_count": len(dates),
        "source_stock_count": len(gains),
        "target_session_count": target_sessions,
    }
```

Here `sessions` and `observations` are the per-stock counts computed before
centering. The report therefore exposes the mask's breadth without exposing
prices, returns, forecasts, or trades.

The centered bootstrap distribution \(M\) estimates sampling error for a
60-session mean. With critical shift
\(c=-Q_{0.025}(M)\), the mean shift with 80% marginal power is
\(c-Q_{0.20}(M)\). This is computed separately for each reference; it does not
model the other three forward gates.

- [ ] **Step 5: Run both focused test scripts**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_universe_scaling.py
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_shrinkage.py
```

Expected: both scripts print their `passed` summaries and exit zero.

### Task 3: Publish One Authenticated Development-Only Report

**Files:**

- Modify: `tools/analyze_spy_residual_shrinkage.py:860`
- Modify: `tools/analyze_spy_residual_shrinkage.py:1085`
- Modify: `tools/analyze_spy_residual_shrinkage.py:1124`
- Test: `tests/python/test_spy_residual_shrinkage.py`

**Interfaces:**

- Consumes:
  only `states[1]`, `phases[1]`, the authenticated `ResidualLease`, current
  analysis bindings, the hash-bound forward config, and the implementation
  commit.
- Produces:
  `reports/h13-spy-residual-20260725-01-sensitivity/sensitivity.json`.
- Preserves:
  existing default shrinkage mode and `--alignment` mode.

- [ ] **Step 1: Write the failing report and orchestration tests**

Add a report-boundary test that calls `_sensitivity_report()` with two
synthetic comparisons and asserts:

```python
    assert report["evidence_role"] == "development-planning-only"
    assert report["decision"] == {
        "candidate_or_protocol_change_authorized": False,
        "joint_test_power_estimated": False,
        "output_role": "paired-mse-interval-sensitivity-only",
    }
    assert report["truth_phases_read"] == ["calibration"]
    assert report["locks"] == expected_forward_protocol()["locks"]
    assert all(value is False for value in report["locks"].values())
    assert not {"prices", "returns", "trades", "backtest"} & report.keys()
```

Add an orchestration test with patched `_phase_market_regimes`,
`_phase_truth`, `_publish`, and `_validate_published`. Pass a calibration
state/phase, assert only that phase is consumed, and assert the published
report contains exactly the `zero` and `unchanged-five-seed-mean`
comparisons. Assert the sensitivity call receives `60`, `20`, `10_000`, and
`20_260_725` from `expected_forward_protocol()`. Reject a missing
`forward_config` binding and every mutated forward protocol.

Extend the CLI test to parse `--sensitivity`, reject
`--alignment --sensitivity`, and verify the summary:

```python
{
    "comparisons": [
        "unchanged-five-seed-mean",
        "zero",
    ],
    "mode": "sensitivity",
    "status": "analyzed",
}
```

- [ ] **Step 2: Run the analyzer test and verify the new symbols fail**

Run the Task 2 analyzer-test command.

Expected: nonzero exit because `_sensitivity_report()` and
`_analyze_sensitivity()` do not exist.

- [ ] **Step 3: Build the bounded report**

Import `FORWARD_CONFIG`, `expected_forward_protocol`, and
`validate_forward_protocol`. Extend `ANALYSIS_SOURCE_PATHS` with
`tools/spy_residual_forward_contract.py`, `tools/universe_cross_section.py`,
and `tools/universe_scaling.py`, then add:

```python
def _sensitivity_report(
    inputs: Mapping[str, object],
    protocol: Mapping[str, object],
    comparisons: Mapping[str, Mapping[str, object]],
    implementation_commit: str,
) -> dict[str, object]:
    """Bind planning sensitivity without changing the forward contract."""
    frozen = validate_forward_protocol(protocol)
    references = tuple(frozen["metrics"]["references"])
    if inputs.get("forward_config") != _binding_value(FORWARD_CONFIG):
        raise ValueError("sensitivity forward config changed")
    if tuple(comparisons) != references:
        raise ValueError("sensitivity comparisons changed")
    candidate = frozen["candidate"]
    return _json_mapping({
        "candidate": {
            "gate": dict(candidate["gate"]),
            "model": candidate["model"],
            "name": candidate["name"],
            "source_phase": "calibration",
        },
        "decision": {
            "candidate_or_protocol_change_authorized": False,
            "joint_test_power_estimated": False,
            "output_role": "paired-mse-interval-sensitivity-only",
        },
        "evidence_role": "development-planning-only",
        "inputs": dict(inputs),
        "integrity": {"implementation_commit": implementation_commit},
        "locks": dict(frozen["locks"]),
        "paired_squared_error": dict(comparisons),
        "schema": 1,
        "truth_phases_read": ["calibration"],
    })
```

- [ ] **Step 4: Add the authenticated calibration-only mode**

Add `_analyze_sensitivity()` beside `_analyze_alignment()`:

```python
def _analyze_sensitivity(
    state: _PhaseRows,
    phase: AuthenticatedPhase,
    lease: ResidualLease,
    inputs: Mapping[str, object],
    protocol: Mapping[str, object],
    implementation_commit: str,
    result_path: Path,
    directory: int,
    output_identity: tuple[int, int],
    verify: Callable[[], None],
) -> Mapping[str, object]:
    """Publish future-label-blind sensitivity from calibration history."""
    if state.source.phase != "calibration" or \
       phase.source != state.source or not callable(verify):
        raise ValueError("sensitivity phase inputs are invalid")
    frozen = validate_forward_protocol(protocol)
    candidate = frozen["candidate"]["name"]
    bootstrap = frozen["metrics"]["bootstrap"]
    verify()
    regimes = _phase_market_regimes(state, lease)
    truth, _ = _phase_truth(state, lease)
    unchanged = phase.predictions[MODEL]
    family = {
        MODEL: unchanged,
        candidate: direction_gated_predictions(unchanged, regimes),
    }
    comparisons = {}
    for name, reference in zip(
        frozen["metrics"]["references"], ("zero", MODEL), strict=True,
    ):
        gains, losses = _paired_mse_days(
            truth, family, candidate, reference,
        )
        comparisons[name] = interval_sensitivity(
            gains, losses,
            target_sessions=frozen["forward_window"][
                "target_session_count"
            ],
            block_sessions=bootstrap["decision_block_sessions"],
            replicates=bootstrap["replicates"],
            seed=bootstrap["seed"],
        )
    del truth
    result = _sensitivity_report(
        inputs, frozen, comparisons, implementation_commit,
    )
    _publish(result_path, result, directory, verify)
    identities = _validate_published(result_path, result)
    with freeze_inputs((result_path,)) as frozen:
        if not _exact_json(
            read_canonical_json(frozen[0].snapshot), result,
        ):
            raise ValueError("sensitivity result changed")
        verify()
        _verify_single_link_inputs(identities, "sensitivity result")
        verify_frozen(frozen)
        if _directory_members(
            result_path.parent, (result_path.name,),
        ) != output_identity:
            raise ValueError("sensitivity result topology changed")
    return result
```

In `analyze_residual_shrinkage()`, freeze
`ROOT / FORWARD_CONFIG.path` only in sensitivity mode, require its
`FileBinding` to equal `FORWARD_CONFIG`, parse it with
`validate_forward_protocol()`, and add the binding to report inputs. Add keyword
`sensitivity: bool = False`, require both mode flags to be exact booleans, and
reject `alignment and sensitivity`. Select:

```python
suffix = (
    "alignment" if alignment else
    "sensitivity" if sensitivity else
    "shrinkage"
)
```

For sensitivity, create only `sensitivity.json` and call:

```python
return _analyze_sensitivity(
    states[1], phases[1], lease, inputs, protocol,
    implementation_commit, result_path, directory,
    output_identity, verify,
)
```

This indexes the already authenticated calibration phase and never accepts a
future-data path.

- [ ] **Step 5: Make the CLI modes mutually exclusive**

In `parse_args()`, use:

```python
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--alignment", action="store_true")
mode.add_argument("--sensitivity", action="store_true")
```

Pass both booleans to `analyze_residual_shrinkage()`. In `main()`, emit the
sensitivity summary before the existing alignment/default branches:

```python
summary = {
    "comparisons": sorted(report["paired_squared_error"]),
    "mode": "sensitivity",
    "status": "analyzed",
} if arguments.sensitivity else (
    {
        "mode": "alignment",
        "status": "analyzed",
        "unclipped_scale":
            report["diagnostic"]["global"]["unclipped_scale"],
    } if arguments.alignment else {
        "later_residual_holdout_preregistration_warranted":
            report["decision"][
                "later_residual_holdout_preregistration_warranted"
            ],
        "scale": report["fit"]["scale"],
        "status": "analyzed",
    }
)
```

- [ ] **Step 6: Run focused tests**

Run both commands from Task 2 Step 5.

Expected: both tests pass.

### Task 4: Verify, Checkpoint, and Run the Diagnostic

**Files:**

- Verify only:
  `tools/universe_scaling.py`,
  `tools/analyze_spy_residual_shrinkage.py`,
  `tests/python/test_universe_scaling.py`,
  `tests/python/test_spy_residual_shrinkage.py`
- Generate but do not track:
  `reports/h13-spy-residual-20260725-01-sensitivity/sensitivity.json`

**Interfaces:**

- Consumes: the signed implementation checkpoint SHA.
- Produces: one signed local GitButler checkpoint and one ignored,
  source-authenticated development report.

- [ ] **Step 1: Run syntax, focused, and aggregate gates**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B -m py_compile \
  tools/universe_scaling.py tools/analyze_spy_residual_shrinkage.py \
  tests/python/test_universe_scaling.py \
  tests/python/test_spy_residual_shrinkage.py
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_universe_scaling.py
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_shrinkage.py
make -B \
  PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  check
```

Expected: compile exits zero, both focused scripts pass, all C suites pass,
and all registered Python suites pass.

- [ ] **Step 2: Review the exact four-file implementation diff**

Run `but diff`, inspect only the four implementation/test file changes, and
confirm:

- no forward source, experiment JSON, report, model, data, credential, or
  unrelated working-tree hunk is selected;
- the old `circular_block_interval()` fixtures are exactly unchanged;
- sensitivity reads calibration only;
- no return, price, policy, or trade field enters the report.

- [ ] **Step 3: Create a signed local implementation checkpoint**

Use the selected-change GitButler fast path. Pass the comma-joined, exact IDs
printed by the preceding `but diff` to `--changes`, with branch
`feat/forward-interval-sensitivity` and message
`feat(evaluation): quantify forward interval sensitivity`.

Set author and committer to:

```text
enkyuan <yuan.enkng@gmail.com>
```

Sign with the configured ED25519 key. Do not push or land.

- [ ] **Step 4: Verify the checkpoint identity and signature**

Use GitButler workspace output to identify the new 40-character SHA, then run
`git verify-commit` and `git show -s --format=fuller` against that exact SHA.

Expected: a valid SSH signature for fingerprint
`SHA256:mnKfNNJJUV5ELrMXZCwdu6v/PusXBeAi8tWUnRXmWMQ`, with enkyuan as both
author and committer.

- [ ] **Step 5: Produce the ignored authenticated report**

Run the following command with the exact signed implementation SHA emitted by
GitButler substituted as the value of `--implementation-commit`:

```bash
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -I -S -B tools/analyze_spy_residual_shrinkage.py \
  experiments/h13-spy-residual-20260725-01-attempt.json \
  --implementation-commit "$IMPLEMENTATION_SHA" \
  --sensitivity
```

Expected:

```json
{"comparisons":["unchanged-five-seed-mean","zero"],"mode":"sensitivity","status":"analyzed"}
```

Open the generated report read-only and record both MDE values and ratios.
Do not use them to alter the frozen forward test.

- [ ] **Step 6: Obtain independent correctness and scope review**

Ask a read-only reviewer to inspect the exact checkpoint for:

- preservation of existing bootstrap outputs;
- correct centered 60-session/20-session marginal-power equation;
- stock-panel/session dependence;
- calibration-only truth access;
- false execution/trading locks;
- absence of protocol changes and return claims.

Resolve P0-P2 findings in the same unpublished checkpoint, rerun all gates,
and verify the replacement signature. Leave the final checkpoint local.

## Mathematical and Literature Basis

For stock \(i\), market session \(t\), and observed intraday target \(k\), the
paired squared-error gain is:

\[
g_{itk} =
L(y_{itk},\hat y_{\text{reference},itk}) -
L(y_{itk},\hat y_{\text{candidate},itk}).
\]

Positive values favor the candidate. If \(n_{it}\) is the number of observed
targets in one stock-session cell, the existing stock-macro estimand is:

\[
\hat\mu =
\frac{1}{S}\sum_{i=1}^{S}
\frac{\sum_t\sum_{k=1}^{n_{it}}g_{itk}}
     {\sum_t n_{it}}.
\]

This observation-weights rows within each stock, then weights the 11 stocks
equally. Center every observed gain by the single macro mean,
\(x_{itk}=g_{itk}-\hat\mu\); per-stock centering would erase real
cross-sectional heterogeneity.

Serial dependence requires dependence-preserving resampling. Let \(D\) be the
ordered 42-session market grid and retain each authenticated zero-count
stock-session cell. For replicate \(b\), select the same multiset \(I_b\) of
60 sessions for every stock using three 20-session circular blocks:

\[
M_b =
\frac{1}{S}\sum_{i=1}^{S}
\frac{\sum_{t\in I_b}\sum_{k=1}^{n_{it}}x_{itk}}
     {\sum_{t\in I_b}n_{it}}.
\]

Every possible block is preflighted to keep each stock denominator positive;
there is no imputation, independent stock sampling, or rejection-redraw. Then
define:

\[
c=-Q_{0.025}(M), \qquad
\operatorname{MDE}_{80}=c-Q_{0.20}(M).
\]

Under a symmetric normal approximation this becomes:

\[
\operatorname{MDE}_{80}
\approx
(z_{0.975}+z_{0.80})SE(\bar g)
\approx
2.8016\,SE(\bar g).
\]

The implementation uses the empirical block distribution, not the normal
shortcut. The 42-session source contains only about two 20-session block
lengths, and extending it to 60 sessions assumes the historical vector process
is sufficiently stable. The result is therefore a planning sensitivity
estimate, not new information or evidence that the candidate will pass.

Primary references:

- Diebold and Mariano, forecast-loss comparison with serial dependence:
  <https://www.nber.org/papers/t0169>.
- Politis and Romano, circular block resampling for stationary dependent data:
  <https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data>.
- Newey and West, heteroskedasticity/autocorrelation-consistent long-run
  variance:
  <https://www.nber.org/papers/t0055>.
- Giacomini and White, forecast comparison using information observable at
  forecast time:
  <https://doi.org/10.1111/j.1468-0262.2006.00718.x>.
- White, specification search and data-snooping control:
  <https://doi.org/10.1111/1468-0262.00152>.

This sensitivity report cannot answer whether the candidate will pass, whether
all five gates have 80% joint power, or whether any forecast can earn a net
return. Those questions require the untouched completed forward labels and,
only after a separate authorization, an execution-aware absolute-return
experiment.
