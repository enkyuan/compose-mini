# Horizon-13 Seed-Disagreement Abstention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select a calibration-only abstention penalty from the existing five
seeded horizon-13 predictions after the stationary feature candidate failed.

**Architecture:** Keep each forecast's arithmetic seed mean as
`predicted_log_return`, and attach an in-memory scheduling signal equal to that
mean minus a selected multiple of population seed disagreement. Jointly select
the disagreement multiplier and existing safety threshold on calibration P&L,
freeze both in strict policy schema 3, and replay them without changing any
ledger, experiment, model, or training contract.

**Tech Stack:** Python 3.12+ standard library (`statistics.fmean` and
`statistics.pstdev`), strict JSON, existing backtest and policy-selection
tools, procedural Python tests, GitButler.

## Rationale

The five seeded Transformer predictions already exist at every calibration
timestamp. Their population standard deviation is a cheap member-disagreement
heuristic that may identify timestamps where the arithmetic mean is less useful.
This follows the practical ensemble motivation in the
[Deep Ensembles paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html),
but this implementation is **not calibrated uncertainty or confidence**. It
does not establish probabilistic coverage, epistemic uncertainty, or forecast
calibration.

For seed predictions `p_1[t] ... p_K[t]`, define:

```text
mu[t] = fmean(p_1[t], ..., p_K[t])
d[t] = pstdev(p_1[t], ..., p_K[t])
decision_signal[t] = mu[t] - disagreement_lambda * d[t]
```

The scheduler compares `decision_signal` with the unchanged threshold:

```text
threshold = costs.break_even_log_return + safety_bps / 10_000
```

Reports and trade records continue to expose `predicted_log_return = mu`.
`decision_signal` exists only in memory and never becomes a ledger field.

## Evidence and Global Constraints

- The 2026-07-23 feature calibration selected `raw-17`.
  `stationary-16` was 1.0101% worse on macro validation return MAE and won only
  1 of 6 series-fold buckets.
- Reuse the ignored
  `reports/executable-h13-feature-calibration.json` and its schema-3,
  9,495-record calibration ledger. Do not train or regenerate predictions.
- Keep the selected `raw-17` Transformer, seeds `[7, 19, 31, 43, 61]`, AAPL,
  MSFT, SPY, horizon 13, `executable-return-v1`, calibration boundaries, and
  costs unchanged.
- This is a **member disagreement heuristic**, never calibrated uncertainty or
  confidence.
- Add no dependency, model, candidate, seed, training run, experiment schema,
  ledger schema, fingerprint field, or historical test.
- Do not modify experiment or training configs, `tools/experiment.py`,
  `tools/train.py`, experiment tests, or generated reports.
- Modify only `tools/backtest.py`, `tools/select_policy.py`,
  `tests/python/test_backtest.py`, `tests/python/test_policy.py`, and
  `docs/training.md`.
- `disagreement_lambda == 0.0` must avoid `pstdev` and multiplication and
  preserve exact aggregation, scheduling, trade, and P&L arithmetic.
- A one-member seeded ensemble has population disagreement `0.0`.
  A deterministic `(None,)` stream rejects nonzero disagreement.
- Any nonzero disagreement value with `ensemble_seeds == False` is invalid,
  even when the underlying forecasts happen to carry seed values.
- Policy schema 2 remains accepted with its exact old fields and implies
  `disagreement_lambda == 0.0`. A legacy selector call still writes schema 2;
  an explicit canonical disagreement grid writes strict schema 3.
- Backtest report schema stays 2. Forecast ledger schema stays 3.
- Data through `2026-07-21` remains development-only. Do not inspect or run the
  old historical test interval.
- Success still requires externally pre-registering the complete schema-3
  policy hash and a boundary against later, previously unavailable labels.
- Use one implementation branch, `enkyuan/h13-seed-disagreement`, stacked
  directly above `enkyuan/h13-seed-disagreement-plan`.
- Keep Tasks 1, 2, and 3 as coherent local commits. A guarded local publisher
  may land those checkpoints separately in dependency order.
- Do not push, pull, land, or open a pull request.

## File Map and Locked Interfaces

| File | Responsibility |
| --- | --- |
| `tools/backtest.py` | In-memory signal, schema-2/3 policy validation, replay |
| `tools/select_policy.py` | Canonical joint lambda/safety calibration grid |
| `tests/python/test_backtest.py` | Arithmetic, alignment, compatibility, replay |
| `tests/python/test_policy.py` | Grid size/order, selection, tie, strict validation |
| `docs/training.md` | Diagnostic commands, result, and future-label boundary |

Lock these signatures before implementing:

```text
Forecast.decision_signal: float | None = None

policy_disagreement_lambda(policy: Mapping[str, object]) -> float

_aggregate_seeds(
    forecasts: Sequence[Forecast],
    expected_seeds: Sequence[int] | None = None,
    disagreement_lambda: float = 0.0,
) -> tuple[
    tuple[Forecast, ...],
    dict[tuple[object, ...], tuple[int | None, ...]],
]

run_backtests(
    forecasts: Sequence[Forecast],
    series: Mapping[str, Bars],
    initial_cash: float,
    costs: Costs,
    safety_bps: float = 0.0,
    ensemble_seeds: bool = False,
    expected_seeds: Sequence[int] | None = None,
    cash_only: bool = False,
    disagreement_lambda: float = 0.0,
) -> dict[str, object]

select_policy(
    report: Mapping[str, object],
    forecasts: Sequence[Forecast],
    series: Mapping[str, object],
    costs: Costs,
    safety_values: Sequence[float],
    initial_cash: float,
    model: str,
    report_path: Path,
    report_hash: str,
    ledger_path: Path,
    ledger_hash: str,
    source_records: int,
    disagreement_values: Sequence[float] | None = None,
) -> dict[str, object]
```

The `run_backtests` argument is last to preserve every existing positional
caller. The optional selector argument is also last, so every existing direct
call remains valid and retains schema-2 safety-only behavior.

---

### Task 1: Add the in-memory decision signal and dual policy schemas

**Files:**

- Modify: `tools/backtest.py:41-340,494-548,560-727,849-1007`
- Modify: `tests/python/test_backtest.py:1-383`

**Interfaces:**

- Consumes: strict ledger parsing, complete seed alignment, `fmean`, scheduling,
  policy replay, and report schema 2.
- Produces: optional `Forecast.decision_signal`,
  `policy_disagreement_lambda()`, schema-2 compatibility, strict schema 3, and
  replayed `disagreement_lambda`.

- [ ] **Step 1: Write the failing arithmetic and scheduling tests**

Extend `verify_ensemble()` in `tests/python/test_backtest.py`. Import
`_aggregate_seeds`, then use binary-exact members and assertions:

```text
members = (
    replace(predictions[0], seed=3, predicted_log_return=0.0),
    replace(predictions[0], seed=7, predicted_log_return=0.5),
)

lambda_zero, _ = _aggregate_seeds(
    members, expected_seeds=(3, 7), disagreement_lambda=0.0,
)
assert lambda_zero[0].predicted_log_return == 0.25
assert lambda_zero[0].decision_signal == 0.25

penalized, _ = _aggregate_seeds(
    members, expected_seeds=(3, 7), disagreement_lambda=0.5,
)
assert penalized[0].predicted_log_return == 0.25
assert penalized[0].decision_signal == 0.125

one_member, _ = _aggregate_seeds(
    members[:1], expected_seeds=(3,), disagreement_lambda=1.0,
)
assert one_member[0].decision_signal == one_member[0].predicted_log_return
```

For `(0.0, 0.5)`, `mu == 0.25`, population disagreement is exactly `0.25`,
and lambda `0.5` produces exactly `0.125`; all values are binary-exact.

Use `unittest.mock.patch("tools.backtest.pstdev")` around the lambda-zero call
and assert it was not called. Compare default and explicit lambda-zero
`results`, trades, final equity, turnover, and scheduling counts recursively.
Do not compare a pre-change serialized report: protocol metadata intentionally
gains the selected lambda and corrected signal definition.

Add a scheduling case where the mean exceeds the threshold but the penalized
signal does not. Assert no trade at nonzero lambda. In the trading case, assert
the trade's `predicted_log_return` is the exact mean, never the penalized
decision signal.

Add rejections for:

- a missing or extra seed at one timestamp;
- one seed stream with a different `as_of` or `target_time`;
- negative, boolean, infinite, and NaN lambdas;
- a direct `run_backtests(..., disagreement_lambda=0.5)` call with
  `ensemble_seeds=False`;
- a diagnostic CLI call with `--disagreement-lambda 0.5` but without
  `--ensemble-seeds`;
- `disagreement_lambda=0.5` on a deterministic `seed=None` stream.

The existing incomplete-grid and alignment assertions remain; extend them
rather than create parallel fixtures.

- [ ] **Step 2: Run the backtest test and confirm failure**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_backtest.py
```

Expected: failure because `Forecast` and `_aggregate_seeds()` do not yet expose
the decision signal or disagreement multiplier.

- [ ] **Step 3: Add the minimal in-memory signal**

In `tools/backtest.py`:

1. Import `pstdev` beside `fmean`.
2. Add `decision_signal: float | None = None` as the final `Forecast` field.
   Do not add it to `LEDGER_V1_FIELDS`, `LEDGER_V2_FIELDS`, or
   `LEDGER_V3_FIELDS`.
3. Add one private accessor so counting and scheduling cannot diverge:

```text
def _decision_signal(forecast: Forecast) -> float:
    return (
        forecast.predicted_log_return
        if forecast.decision_signal is None
        else forecast.decision_signal
    )
```

4. Validate `disagreement_lambda` as a finite, non-boolean, nonnegative float
   in `run_backtests()`.
5. Before any aggregation, reject nonzero disagreement when
   `ensemble_seeds` is false. This direct API guard also governs diagnostic CLI
   calls; do not wait for seed inspection to reject it.
6. Pass it into `_aggregate_seeds()`.
7. For every complete timestamp tuple, compute `mu` once with `fmean`.
   Use this exact branch:

```text
decision = mu
if disagreement_lambda != 0.0:
    if seeds == (None,):
        raise ValueError(
            "deterministic forecasts do not support seed disagreement"
        )
    decision = mu - disagreement_lambda * pstdev(
        item.predicted_log_return for item in items
    )
```

8. Replace the first seed member with `seed=None`,
   `predicted_log_return=mu`, and `decision_signal=decision`.
9. Make both signal coverage and `_schedule()` compare `_decision_signal(item)`
   with the existing threshold.
10. Continue passing `forecast.predicted_log_return` to `_execute()` so trade
   records retain `mu`.

The `disagreement_lambda != 0.0` branch is deliberate. Do not reduce it to
unconditional `pstdev` or `mu - lambda * d`; lambda zero is the compatibility
path.

- [ ] **Step 4: Record the heuristic without changing report schema**

Keep the report's top-level `"schema": 2`. Add only:

```text
"disagreement_lambda": disagreement_lambda,
```

Replace the existing `protocol["signal"]` text with this precedence:

```text
"cash" if cash_only else
(
    "long when arithmetic mean seed prediction minus disagreement_lambda "
    "times population seed disagreement exceeds the threshold"
    if ensemble_seeds else
    "long when predicted_log_return exceeds the threshold"
)
```

`cash_only` must remain the highest-priority representation. Retain
`"seed_aggregation": "arithmetic mean per timestamp"` because stored forecasts
and trade values remain the mean.

Do not add a protocol key named `decision_signal`, a per-result field, a trade
field, or a ledger field. Remove the old ensemble signal wording rather than
leaving contradictory definitions in the report.

- [ ] **Step 5: Write failing schema and replay tests**

In `tests/python/test_backtest.py`, construct one exact existing schema-2 policy
from the current fields and assert:

```text
assert validate_policy(policy_v2) == policy_v2
assert policy_disagreement_lambda(policy_v2) == 0.0
```

Construct a seeded schema-3 policy by adding:

```text
"schema": 3,
"disagreement_lambda": 0.5,
```

Its long trials must contain the exact lambda-major, safety-minor product:

```text
[
    (disagreement_lambda, safety_bps)
    for disagreement_lambda in (0.0, 0.5, 1.0)
    for safety_bps in (0.0, 3.0, 6.0, 10.0)
]
```

Append cash with both values null, choose the matching `0.5` winner, and assert
the helper returns `0.5`.

Reject each schema-3 mutation independently:

- missing or extra top-level fields;
- boolean, negative, NaN, or infinite lambda;
- cash with a numeric lambda;
- long with a null lambda;
- a selected top-level lambda that differs from the winning trial;
- one whole lambda row removed;
- one whole safety column removed;
- one extra lambda row;
- one extra safety column;
- a duplicated pair or any reordered pair;
- a deterministic policy with any nonzero long-trial or selected lambda.

Keep one schema-2 mutation test proving that adding the new field to schema 2 is
rejected. Backward compatibility means exact old fields, not permissive fields.

No policy-mode CLI fixture exists yet. Add `verify_policy_cli()` that creates
all trust inputs in its temporary directory:

1. Write a schema-3 test ledger for one complete two-seed Transformer grid.
2. Hash the ledger and create a schema-6
   `selection-calibration-and-test` experiment with matching fingerprint,
   model fingerprints, policy authorization, ledger metadata, test records,
   and one-series test boundary.
3. Create and hash an exact schema-2 policy, an exact schema-3 long policy, and
   an exact schema-3 cash policy. Rewrite the experiment's policy hash for each
   invocation so `validate_test_experiment()` receives a valid authorization.
4. Invoke `backtest_main()` with the real ledger, policy, experiment, and CSV
   paths. Patch `run_backtests()` and `write_report()` only for the two
   long-policy calls that inspect the forwarded lambda. Run the schema-3 cash
   policy without either patch and read its real report.

Assert these separate valid replays:

```text
schema 2 long -> called.kwargs["disagreement_lambda"] == 0.0
schema 3 long -> called.kwargs["disagreement_lambda"] == 0.5
schema 3 cash -> report["protocol"]["disagreement_lambda"] == 0.0
schema 3 cash -> report["protocol"]["signal"] == "cash"
```

For the real cash report, assert every `forecast_long_cash` strategy has zero
trades and final equity equal to initial cash. This verifies schema-3 cash
replay and report representation rather than relying only on a mock. Then
invoke valid policy mode with an explicit diagnostic override of zero:

```text
--disagreement-lambda 0
policy mode does not accept diagnostic overrides
```

Explicit zero is still an override and must be rejected. Keep the existing
model, cost, safety, and `--ensemble-seeds` override rejection coverage. This
fixture proves schema-2 replay, schema-3 long replay, schema-3 cash replay, and
override rejection through the actual CLI boundary.

- [ ] **Step 6: Run the focused test and confirm failure**

Run the Step 2 command again.

Expected: failure because policy schema 3 and policy replay are not implemented.

- [ ] **Step 7: Implement exact schema-2/schema-3 validation**

In `tools/backtest.py`:

1. Rename the current field constants to `POLICY_V2_FIELDS` and
   `TRIAL_V2_FIELDS`.
2. Define:

```text
POLICY_V3_FIELDS = POLICY_V2_FIELDS | {"disagreement_lambda"}
TRIAL_V3_FIELDS = TRIAL_V2_FIELDS | {"disagreement_lambda"}
POLICY_SAFETY_GRID = (0.0, 3.0, 6.0, 10.0)
SEEDED_DISAGREEMENT_GRID = (0.0, 0.5, 1.0)
DETERMINISTIC_DISAGREEMENT_GRID = (0.0,)
```

3. Select exact field sets by integer schema. Accept only 2 or 3.
4. Add:

```text
def policy_disagreement_lambda(
    policy: Mapping[str, object],
) -> float:
    value = (
        0.0 if policy["schema"] == 2
        else policy["disagreement_lambda"]
    )
    return 0.0 if value is None else float(value)
```

5. Pass the policy schema into `_validate_trial()`. Schema-2 long trials imply
   lambda zero and cash implies null only for comparison. Schema-3 cash requires
   null; schema-3 long requires a finite nonnegative number.
6. For schema 3, choose the required lambda constant from whether the model is
   seeded. Require long-trial pairs to equal exactly:

```text
[
    (disagreement_lambda, safety_bps)
    for disagreement_lambda in required_disagreement_grid
    for safety_bps in POLICY_SAFETY_GRID
]
```

   Require exactly one cash trial last. Never infer an allowed rectangle from
   policy contents: removing a complete row or column must still fail.
7. Seeded schema 3 therefore has 12 long trials plus cash. Deterministic
   schema 3 has four lambda-zero long trials plus cash. Any extra, missing,
   duplicate, reordered, malformed, or nonfinite value fails.
8. Match the selected action, safety, and lambda to `select_trial(trials)`.
   Leave schema-2 selection behavior unchanged.
9. Add `--disagreement-lambda` to the diagnostic backtest CLI with default
   `None`. In diagnostic mode, convert `None` to `0.0`. In policy mode, reject
   any supplied value and replay `policy_disagreement_lambda(policy)`.
10. Pass the final value as the last `run_backtests()` argument.

Do not alter `minimum_predicted_log_return`; it remains costs plus safety only.

- [ ] **Step 8: Implement the locked tie-break**

Extend `select_trial()` without changing its first three keys:

```text
return max(
    trials,
    key=lambda item: (
        item["objective"],
        -item["mean_gross_turnover"],
        math.inf
        if item["safety_bps"] is None
        else item["safety_bps"],
        -policy_trial_disagreement_lambda(item),
    ),
)
```

The private trial helper returns `0.0` for old schema-2 long trials and for cash.
The safety key makes cash safest (`math.inf`); the final key chooses the
smallest lambda only after objective, turnover, and safety all tie.

- [ ] **Step 9: Run focused and standard checks**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_backtest.py
make -B PYTHON="$PRIMARY_PYTHON" check
```

Expected: `backtest tests passed`, then all C and standard Python suites pass.

- [ ] **Step 10: Create the backtest/policy checkpoint**

Inspect once with `but diff`. Commit only `tools/backtest.py` and
`tests/python/test_backtest.py` by copying their change IDs into:

```sh
but commit enkyuan/h13-seed-disagreement -c \
  -m "feat(backtest): add seed disagreement signal" \
  --changes "$CHANGE_IDS"
```

Read the returned workspace state. If it does not show the new branch directly
above `enkyuan/h13-seed-disagreement-plan`, run exactly:

```sh
but move \
  enkyuan/h13-seed-disagreement \
  enkyuan/h13-seed-disagreement-plan
```

Do not include selector, policy-test, documentation, or generated artifacts in
this checkpoint.

---

### Task 2: Select the joint disagreement and safety grid

**Files:**

- Modify: `tools/select_policy.py:247-410`
- Modify: `tests/python/test_policy.py:1-318`

**Interfaces:**

- Consumes: Task 1 `run_backtests(..., disagreement_lambda=...)`, schema-3
  validation, and locked tie-breaking.
- Produces: unchanged schema-2 policies for legacy safety-only calls, or strict
  schema-3 policies for an explicit canonical joint grid.

- [ ] **Step 1: Write the failing canonical-grid tests**

First call the existing `choose()` and `select_policy()` fixtures without a new
argument. Assert they still return exact schema-2 field and trial sets, and that
`policy_disagreement_lambda(policy) == 0.0`.

Then extend `choose()` with an optional final test argument:

```text
disagreement_values: tuple[float, ...] | None = None
```

Pass it as the final keyword argument to `select_policy()`. Existing direct
calls must remain untouched.

For an explicit seeded Transformer call, pass safety
`(0.0, 3.0, 6.0, 10.0)` and disagreement `(0.0, 0.5, 1.0)`, then assert:

```text
assert policy["schema"] == 3
assert len(policy["threshold_trials"]) == 13
assert [
    (trial["disagreement_lambda"], trial["safety_bps"])
    for trial in policy["threshold_trials"][:-1]
] == [
    (disagreement_lambda, safety_bps)
    for disagreement_lambda in (0.0, 0.5, 1.0)
    for safety_bps in (0.0, 3.0, 6.0, 10.0)
]
assert policy["threshold_trials"][-1]["action"] == "cash"
assert policy["threshold_trials"][-1]["disagreement_lambda"] is None
```

For explicit `last_close`, pass the same exact safety grid and disagreement
`(0.0,)`; assert exactly five trials: four lambda-zero long trials followed by
cash.

Add argument/grid rejection tests for:

- any explicit schema-3 safety grid other than `(0.0, 3.0, 6.0, 10.0)`;
- any explicit seeded lambda grid other than `(0.0, 0.5, 1.0)`;
- any explicit deterministic lambda grid other than `(0.0,)`;
- reordered, duplicated, empty, boolean, negative, NaN, or infinite values;
- any nonzero disagreement value for a deterministic model;
- omitting the CLI flag, which must remain valid and produce schema 2.

- [ ] **Step 2: Add the abstention fixture**

Add one two-seed, two-timestamp calibration fixture where:

- timestamp 1 has a profitable, low-disagreement mean and remains tradable;
- timestamp 2 has a losing positive mean with enough disagreement that
  lambda `1.0` moves its decision signal below threshold;
- lambda zero trades both timestamps and has lower terminal log growth;
- lambda one trades only the first timestamp and wins.

Use zero costs and horizon 1 so hand calculations are exact. Assert:

```text
assert policy["action"] == "long_above"
assert policy["disagreement_lambda"] == 1.0
assert policy["safety_bps"] == 0.0
assert policy["threshold_trials"][0]["disagreement_lambda"] == 0.0
```

Invoke it with the exact schema-3 safety and seeded-lambda constants; do not
introduce a test-only grid.

Also assert the selected trial has strictly greater objective than every
lambda-zero long trial. This fixture proves abstention changes scheduling, not
the stored mean or execution arithmetic.

- [ ] **Step 3: Add direct tie-break tests**

Use `select_trial()` with synthetic valid-shaped mappings. Lock this precedence
in separate assertions:

1. maximum objective;
2. minimum mean gross turnover;
3. maximum safety, with cash treated as `math.inf`;
4. minimum disagreement lambda.

Include a final exact tie between lambda `0.5` and `1.0`; assert `0.5` wins.
Do not rely on list order for any tie assertion.

- [ ] **Step 4: Run the policy test and confirm failure**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_policy.py
```

Expected: legacy schema-2 assertions pass, then the explicit disagreement call
fails because the selector does not yet accept the optional grid.

- [ ] **Step 5: Implement the joint grid**

In `tools/select_policy.py`:

1. Import the three canonical grid constants from `tools.backtest`.
2. Append
   `disagreement_values: Sequence[float] | None = None` to the end of
   `select_policy()`. Do not insert it among existing positional parameters.
3. When it is `None`, run the current safety-only loop unchanged, omit lambda
   from every trial and the policy, and emit schema 2. This is the legacy path.
4. When it is explicit, require safety to equal `POLICY_SAFETY_GRID` and
   disagreement to equal the exact seeded or deterministic constant. Reject
   before running a backtest when either sequence differs.
5. Extend `_trial()` so only schema-3 calls include
   `"disagreement_lambda"`. Build those long trials in lambda-major,
   safety-minor order:

```text
trials = [
    _trial(
        run_backtests(
            selected,
            series,
            initial_cash,
            costs,
            safety_bps,
            ensemble_seeds=True,
            expected_seeds=seeds,
            disagreement_lambda=disagreement_lambda,
        ),
        safety_bps,
        disagreement_lambda,
    )
    for disagreement_lambda in disagreement_values
    for safety_bps in safety_values
]
```

6. Append one schema-3 cash trial with both `safety_bps` and
   `disagreement_lambda` null.
7. Write schema 3 and top-level `"disagreement_lambda"` only on the explicit
   path.
8. Leave `minimum_predicted_log_return` equal to break-even plus safety in both
   schemas.
9. Return `validate_policy(policy)` as today.

- [ ] **Step 6: Require canonical CLI values**

Add the optional selector option:

```text
parser.add_argument(
    "--disagreement-lambda",
    nargs="+",
    type=float,
)
```

Omission passes `None` and keeps legacy schema-2 behavior. An explicit value
passes the tuple unchanged and must match the exact seeded or deterministic
research grid; do not sort, deduplicate, or normalize it. An explicit
schema-3 call also requires the exact safety grid. This rejects incomplete,
extra, duplicate, reordered, boolean, negative, and nonfinite grids.

Include the selected lambda in final JSON only for schema 3:

```text
if policy["schema"] == 3:
    summary["disagreement_lambda"] = policy["disagreement_lambda"]
```

Retain the existing schema-2 summary fields exactly when the option is omitted.

- [ ] **Step 7: Run focused and aggregate policy checks**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_backtest.py
"$PRIMARY_PYTHON" tests/python/test_policy.py
make -B PYTHON="$PRIMARY_PYTHON" check
```

Expected: both focused scripts and every standard suite pass.

- [ ] **Step 8: Create the selector checkpoint**

Inspect once with `but diff`. Commit only `tools/select_policy.py` and
`tests/python/test_policy.py` on the existing implementation branch:

```sh
but commit enkyuan/h13-seed-disagreement \
  -m "feat(training): select seed disagreement penalty" \
  --changes "$CHANGE_IDS"
```

Trust the returned workspace state. Do not create a second implementation
branch or include Task 3 documentation.

---

### Task 3: Document and run the calibration-only decision

**Files:**

- Modify: `docs/training.md:262-380`
- Create in one new ignored `reports/h13-seed-disagreement.XXXXXX/` directory:
  `policy-v3.json`
- Create in that same directory only for a long policy:
  `calibration-backtest.json`

**Interfaces:**

- Consumes: existing feature calibration report/ledger/CSVs and the Task 2
  schema-3 selector.
- Produces: one calibration-only pass/fail decision and future-label
  preregistration instructions. It produces no trained or versioned artifact.

- [ ] **Step 1: Update selector and replay documentation**

In `docs/training.md`, explain that schema 3 adds a member-disagreement
heuristic, not calibrated uncertainty or confidence. Document:

```text
decision_signal =
  mean(seed predicted_log_return)
  - disagreement_lambda
    * population_pstdev(seed predicted_log_return)
```

State that policy schema 2 replays as lambda zero, policy schema 3 freezes the
selected value, and `predicted_log_return` in reports remains the mean.

Update every selector example:

- seeded `transformer` and `mlp` use
  `--disagreement-lambda 0 0.5 1`;
- deterministic `linear` uses `--disagreement-lambda 0`.

Those explicit-grid examples emit schema 3. Rename every policy output and
later `--policy` reference in that workflow consistently:

```text
reports/executable-h13-transformer-policy-v3.json
reports/executable-h13-mlp-policy-v3.json
reports/executable-h13-linear-policy-v3.json
```

Change the surrounding prose from "schema-2 policies" to "schema-3 policies".
Do not rename unrelated experiment or final-report files merely because their
names contain `v2`.

State separately that existing `*-policy-v2.json` files remain exact schema-2
inputs and replay unchanged with implied lambda zero. Do not imply that those
historical files were rewritten or upgraded in place.

- [ ] **Step 2: Run the selector against existing ignored artifacts**

Create one fresh ignored run directory, define the existing series, and run no
training:

```zsh
export RUN_DIR="$(mktemp -d reports/h13-seed-disagreement.XXXXXX)"
policy_path="$RUN_DIR/policy-v3.json"
backtest_path="$RUN_DIR/calibration-backtest.json"
test ! -e "$policy_path"
test ! -e "$backtest_path"

series=(
  AAPL=data/aapl-30m.csv
  MSFT=data/msft-30m.csv
  SPY=data/spy-30m.csv
)

python tools/select_policy.py \
  reports/executable-h13-feature-calibration.json \
  reports/executable-h13-feature-calibration.jsonl \
  "$policy_path" \
  "${series[@]}" \
  --model transformer \
  --safety-bps 0 3 6 10 \
  --disagreement-lambda 0 0.5 1 \
  --initial-cash 100 \
  --spread-bps 1 \
  --slippage-bps 1 \
  --fee-bps 0
```

Require:

```sh
jq -e '
  .schema == 3 and
  .model == "transformer" and
  .candidate == "raw-17" and
  .seeds == [7, 19, 31, 43, 61] and
  .series == ["AAPL", "MSFT", "SPY"] and
  (.threshold_trials | length) == 13
' "$policy_path"
```

Expected: `true`. Stop if the directory or either output path already existed,
or if the contract check fails. Do not reuse a prior ignored policy path.

- [ ] **Step 3: Materialize per-series calibration evidence**

If the selected action is cash, skip this command and continue to Step 4; the
gate will fail with zero per-series execution. Otherwise:

```zsh
safety_bps=$(
  jq -er 'select(.action == "long_above") | .safety_bps' \
    "$policy_path"
)
disagreement_lambda=$(
  jq -er 'select(.action == "long_above") | .disagreement_lambda' \
    "$policy_path"
)

test ! -e "$backtest_path"
python tools/backtest.py \
  reports/executable-h13-feature-calibration.jsonl \
  "$backtest_path" \
  "${series[@]}" \
  --model transformer \
  --ensemble-seeds \
  --safety-bps "$safety_bps" \
  --disagreement-lambda "$disagreement_lambda" \
  --initial-cash 100 \
  --spread-bps 1 \
  --slippage-bps 1 \
  --fee-bps 0
```

This is a diagnostic calibration backtest. Do not pass `--policy` or
`--experiment-report`; those options are reserved for authorized test replay.
Do not substitute an older ignored backtest from another run directory.

- [ ] **Step 4: Apply the promotion gate once**

Run this complete script:

```python
from hashlib import sha256
from pathlib import Path
import json
import os

run_dir = Path(os.environ["RUN_DIR"])
policy_path = run_dir / "policy-v3.json"
backtest_path = run_dir / "calibration-backtest.json"
feature_report_path = Path(
    "reports/executable-h13-feature-calibration.json"
)
feature_ledger_path = Path(
    "reports/executable-h13-feature-calibration.jsonl"
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


policy = json.loads(policy_path.read_text())
feature_report = json.loads(feature_report_path.read_text())
feature_report_hash = file_sha256(feature_report_path)
feature_ledger_hash = file_sha256(feature_ledger_path)
with feature_ledger_path.open("rb") as file:
    feature_ledger_records = sum(bool(line.strip()) for line in file)

policy_report = policy["calibration_report"]
policy_ledger = policy["calibration_prediction_ledger"]
report_ledger = feature_report.get("calibration_prediction_ledger")
report_protocol = feature_report.get("protocol")
expected_series = ["AAPL", "MSFT", "SPY"]
source_bindings = {
    "fresh_run_paths": (
        policy_path.parent == run_dir
        and backtest_path.parent == run_dir
    ),
    "calibration_report_provenance": (
        Path(policy_report["path"]) == feature_report_path
        and policy_report["sha256"] == feature_report_hash
    ),
    "calibration_ledger_provenance": (
        Path(policy_ledger["path"]) == feature_ledger_path
        and policy_ledger["sha256"] == feature_ledger_hash
        and policy_ledger["source_records"] == feature_ledger_records
        and isinstance(report_ledger, dict)
        and report_ledger.get("schema") == 3
        and report_ledger.get("sha256") == feature_ledger_hash
        and report_ledger.get("records") == feature_ledger_records
    ),
    "calibration_protocol": (
        feature_report.get("schema") == 6
        and isinstance(report_protocol, dict)
        and report_protocol.get("phase") == "selection-and-calibration"
        and report_protocol.get("target_kind") == policy["target_kind"]
        and report_protocol.get("target_horizon_bars")
        == policy["horizon_bars"]
        and feature_report.get("test") == []
    ),
    "calibration_selection": (
        feature_report["selection"][policy["model"]]["candidate"]
        == policy["candidate"]
        and feature_report["sweep"]["seeds"] == policy["seeds"]
        and [
            item["name"] for item in feature_report["series"]
        ] == expected_series
        and any(
            item["name"] == policy["candidate"]
            and item["feature_set"] == policy["feature_set"]
            for item in feature_report["sweep"]["candidates"]
        )
    ),
}
if not all(source_bindings.values()):
    print(
        json.dumps(
            {
                "bindings": source_bindings,
                "error": "stale or mismatched calibration source",
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(2)

trials = policy["threshold_trials"]
selected = next(
    trial
    for trial in trials
    if trial["action"] == policy["action"]
    and trial["safety_bps"] == policy["safety_bps"]
    and trial["disagreement_lambda"] == policy["disagreement_lambda"]
)
best_lambda_zero = max(
    trial["objective"]
    for trial in trials
    if trial["action"] == "long_above"
    and trial["disagreement_lambda"] == 0.0
)

if policy["action"] != "long_above":
    result = {
        "selected_trial": selected,
        "best_lambda_zero_objective": best_lambda_zero,
        "checks": {"long_policy": False},
        "promote_seed_disagreement": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(1)

backtest = json.loads(backtest_path.read_text())
ledger = backtest.get("prediction_ledger")
protocol = backtest.get("protocol")
results = backtest.get("results")
policy_ledger = policy["calibration_prediction_ledger"]

result_names = (
    [item.get("series") for item in results]
    if isinstance(results, list)
    and all(isinstance(item, dict) for item in results)
    else []
)
bindings = {
    **source_bindings,
    "prediction_ledger": (
        isinstance(ledger, dict)
        and all(
            ledger.get(field) == policy_ledger[field]
            for field in (
                "sha256",
                "source_records",
                "selected_records",
            )
        )
    ),
    "protocol": (
        isinstance(protocol, dict)
        and protocol.get("initial_cash") == policy["initial_cash"]
        and protocol.get("safety_bps") == policy["safety_bps"]
        and protocol.get("disagreement_lambda")
        == policy["disagreement_lambda"]
        and protocol.get("target_kind") == policy["target_kind"]
        and protocol.get("split") == "calibration"
        and protocol.get("costs")
        == {
            "full_spread_bps": policy["costs"]["spread_bps"],
            "slippage_bps_per_side": policy["costs"][
                "slippage_bps"
            ],
            "fee_bps_per_side": policy["costs"]["fee_bps"],
        }
    ),
    "unique_exact_series": (
        policy["series"] == expected_series
        and len(result_names) == len(expected_series)
        and sorted(result_names) == expected_series
        and len(set(result_names)) == len(result_names)
    ),
    "result_contract": (
        isinstance(results, list)
        and all(
            item.get("model") == policy["model"]
            and item.get("candidate") == policy["candidate"]
            and item.get("feature_set") == policy["feature_set"]
            and item.get("horizon_bars") == policy["horizon_bars"]
            and item.get("target_kind") == policy["target_kind"]
            and item.get("seeds") == policy["seeds"]
            and item.get("seed_aggregation") == "arithmetic_mean"
            and item.get("split") == "calibration"
            and item.get("fold") is None
            for item in results
        )
    ),
}
if not all(bindings.values()):
    print(
        json.dumps(
            {
                "bindings": bindings,
                "error": "stale or mismatched calibration backtest",
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(2)

per_series = {
    item["series"]: {
        "trade_count": item["strategies"][
            "forecast_long_cash"
        ]["trade_count"],
        "execution_coverage": item["strategies"][
            "forecast_long_cash"
        ]["execution_coverage"],
    }
    for item in results
}

checks = {
    "long_policy": True,
    "selected_lambda_nonzero": (
        policy["disagreement_lambda"] is not None
        and policy["disagreement_lambda"] > 0.0
    ),
    "objective_strictly_above_best_lambda_zero": (
        selected["objective"] > best_lambda_zero
    ),
    "total_trades_at_least_30": selected["trade_count"] >= 30,
    "backtest_trade_count_matches_trial": (
        sum(item["trade_count"] for item in per_series.values())
        == selected["trade_count"]
    ),
    "every_series_trades": all(
        item["trade_count"] > 0 for item in per_series.values()
    ),
    "every_series_execution_coverage_positive": all(
        item["execution_coverage"] > 0.0
        for item in per_series.values()
    ),
}
result = {
    "selected_trial": selected,
    "best_lambda_zero_objective": best_lambda_zero,
    "bindings": bindings,
    "per_series": per_series,
    "checks": checks,
    "promote_seed_disagreement": all(checks.values()),
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["promote_seed_disagreement"] else 1)
```

The fresh run directory prevents output reuse. Before reading execution
metrics, the script recomputes both actual calibration-input hashes, binds them
and the ledger record count to policy/report provenance, checks the calibration
protocol and selection, then binds the new backtest's ledger, parameters,
series, and result contracts. Any stale, reused, or mismatched input or output
exits 2 and cannot contribute evidence. Promotion requires every remaining
check. Do not weaken a threshold after reading the result.

- [ ] **Step 5: Record the decision without overclaiming**

Append the exact Step 4 JSON values to the horizon-13 development-result
subsection in `docs/training.md`.

On failure:

- retain the current lambda-zero behavior and schema-2 policy as the operative
  development choice;
- describe schema 3 and ignored outputs as diagnostic only;
- do not run the old historical test or retrain.

On success:

- describe only a calibration-selected member-disagreement heuristic;
- do not run the old historical test;
- externally pre-register the full schema-3 policy hash and a boundary against
  labels strictly after `2026-07-21` before evaluation.

In both cases, keep the generated policy and backtest ignored.

- [ ] **Step 6: Run focused, full, and training checks**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_backtest.py
"$PRIMARY_PYTHON" tests/python/test_policy.py
make -B PYTHON="$PRIMARY_PYTHON" check
make \
  PYTHON="/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python" \
  check-training
```

Expected: both focused scripts pass, all C and standard Python suites pass, and
both PyTorch integration suites pass. No test command may read a historical
holdout report.

- [ ] **Step 7: Create the documentation/evidence checkpoint**

Inspect once with `but diff`. Confirm generated `reports/` files do not appear,
then commit only `docs/training.md` on the existing branch:

```sh
but commit enkyuan/h13-seed-disagreement \
  -m "docs(training): record seed disagreement calibration" \
  --changes "$CHANGE_IDS"
```

Trust the returned workspace state. Do not include a config, ledger, report,
policy, backtest, or model artifact.

---

## Final Review and Checkpoint Boundaries

- [ ] Confirm the implementation diff contains only the five permitted files.
- [ ] Confirm no `decision_signal` key appears in a ledger, protocol, result,
  or trade record.
- [ ] Confirm schema-3 cash replay writes `protocol.signal == "cash"` and a
  zero-trade cash report.
- [ ] Confirm schema-2 policy fixtures still validate byte-for-byte unchanged.
- [ ] Confirm selector omission remains schema 2 and explicit exact grids emit
  schema 3.
- [ ] Confirm the fixed seeded grid is 12 long trials plus cash and rejects
  missing or extra whole rows and columns.
- [ ] Confirm the fixed deterministic grid is 4 lambda-zero trials plus cash.
- [ ] Confirm `predicted_log_return` in every trade remains the arithmetic mean.
- [ ] Confirm lambda zero avoids `pstdev` and preserves aggregation, scheduling,
  trades, and P&L arithmetic.
- [ ] Confirm nonzero lambda without seed aggregation fails through both API and
  diagnostic CLI.
- [ ] Confirm policy mode rejects all direct lambda, model, cost, seed, and
  threshold overrides.
- [ ] Confirm the calibration gate was evaluated once against only the existing
  feature-calibration inputs in a fresh run directory after recomputed source
  hashes and every binding check passed.
- [ ] Confirm no historical test, retraining, generated artifact, push, pull,
  landing, or pull request occurred.

Expected local history, oldest to newest:

1. `enkyuan/h13-seed-disagreement-plan`
   - `docs(training): plan seed-aware abstention`
2. `enkyuan/h13-seed-disagreement`
   - `feat(backtest): add seed disagreement signal`
   - `feat(training): select seed disagreement penalty`
   - `docs(training): record seed disagreement calibration`

The local publisher may land the three implementation commits separately, but
must preserve this order and all normal active-work, signature, dependency, and
target-movement guards.
