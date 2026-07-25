# Universe Coverage and Shared-Portfolio Implementation Plan

> **For implementation:** execute each checkpoint in order. Keep credentials,
> market data, reports, model artifacts, and caches untracked.

**Goal:** Repair the development-only unseen-stock coverage gate without
rewriting the point-in-time selection, then evaluate scaling forecasts through
one execution-aware `$100` portfolio rather than independent per-stock
accounts.

**Architecture:** Preserve the original selection as historical provenance.
Apply one strict, hash-bound overlay whose deterministic rule replaces the
structurally unavailable ENLC member with the first same-stratum reserve from
the already-frozen selection order. Fetch and inspect only development
coverage before rebinding the scaling attempt. Keep the existing single-series
backtester unchanged; add forward-state forecast ledgers and a separate
portfolio engine that carries one cash balance across phases. Reject protected
rows until a later authorization is frozen.

**Tech stack:** Python 3.12 standard library, PyTorch, Massive REST API, Make,
GitButler.

---

## Evidence and scope

- Frozen selection report:
  `reports/universe-selection-20260724-06/selection.json`
  (`9f484ca3e7f44d329179b54b9c1655957d0ad385240b59c2a569ecc76fd32737`)
- Frozen base manifest:
  `reports/universe-selection-20260724-06/manifests/liquid-common-55.json`
  (`61819afe2729682180d361094793bbff0d0ba13909d04f9dbb838d5233f9e5ff`)
- ENLC is selected at zero-based master rank `49`, liquidity stratum `5`.
  Its observed bars end on `2025-01-30`; refetching cannot create later bars
  after the security stopped trading.
- The deterministic reserve rule is:
  `first-same-stratum-eligible-not-selected-by-within-stratum-rank`.
  It selects AAON at within-stratum rank `11`, with composite FIGI
  `BBG000C2LZP3` and share-class FIGI `BBG001S6CZK0`.
- A partial AAON response was inspected before this plan was frozen. It is
  exploratory evidence only. Ignore it; the signed rule, not that
  response, determines the replacement. Any resulting benchmark estimates
  performance conditional on development observability and is not
  survivorship-free confirmation.
- Do not splice OKE prices into ENLC. That crosses security identities and
  requires a separate corporate-action portfolio contract.
- Historical opportunities `[4955, 5505)` are already locally available and
  were inspected by earlier experiments. Keep them logically closed during
  this work, but do not call them confirmatory. Genuine confirmation requires
  future labels that did not exist when the final policy was registered.

Bind both files the overlay reads and the complete 77-file selection tree. The
file bindings determine the transformation; the tree binding preserves the raw
selection-source provenance and rejects output inside the frozen package.

## Checkpoint 1: Freeze and apply the coverage overlay

### Files

- Create:
  `universes/liquid-common-55-coverage-v2.example.json`
- Create:
  `tools/apply_universe_coverage_overlay.py`
- Modify:
  `tests/python/test_massive.py`

### Policy schema

The policy must contain exactly:

```json
{
  "base_manifest": {
    "path": "reports/universe-selection-20260724-06/manifests/liquid-common-55.json",
    "sha256": "61819afe2729682180d361094793bbff0d0ba13909d04f9dbb838d5233f9e5ff"
  },
  "declared_on": "2026-07-24",
  "failed_member": {
    "master_rank": 49,
    "stratum": 5,
    "ticker": "ENLC"
  },
  "purpose": "Replace one structurally unavailable unseen member for development-only coverage.",
  "replacement": {
    "composite_figi": "BBG000C2LZP3",
    "share_class_figi": "BBG001S6CZK0",
    "ticker": "AAON"
  },
  "replacement_rule": "first-same-stratum-eligible-not-selected-by-within-stratum-rank",
  "schema": 1,
  "scope": "development-coverage-only-conditional-on-observability",
  "selection": {
    "path": "reports/universe-selection-20260724-06/selection.json",
    "sha256": "9f484ca3e7f44d329179b54b9c1655957d0ad385240b59c2a569ecc76fd32737"
  },
  "selection_tree": {
    "files": 77,
    "root": "reports/universe-selection-20260724-06",
    "sha256": "bd9366ec5b040555e8b05ae932447b01b97d57e51832c9d5503059fc9119db24"
  }
}
```

The CLI accepts only `POLICY OUTPUT`. It has no bar, report, model, metric, or
reserved-data argument.

### Public interface

The implementation exposes:

```text
replacement_candidate(
    selection: Mapping[str, object],
    failed_ticker: str,
) -> Mapping[str, object]

revised_manifest(
    base: Mapping[str, object],
    failed: Mapping[str, object],
    replacement: Mapping[str, object],
    *,
    purpose: str,
    declared_on: str,
) -> dict[str, object]

apply_overlay(
    policy_path: Path,
    output_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, object]
```

`replacement_candidate()` requires one selected failed member and chooses the
unique minimum `within_stratum_rank` among same-stratum
`eligible-not-selected` candidates. `revised_manifest()` changes exactly the
failed master-rank entry, applies the overlay's purpose/declaration, and leaves
every other field and series entry unchanged.
`apply_overlay()`:

1. freezes and strictly parses the policy;
2. resolves canonical repository-relative inputs;
3. binds every regular file in the selection tree and rejects output within it;
4. freezes the selection and base manifest;
5. verifies both file bindings, the tree digest, source archives, cohort
   bindings, and semantic list hashes;
6. checks all 55 manifest entries against the frozen master order;
7. checks the declared failure and replacement identities against the
   deterministic result;
8. validates the revised value through `UniverseManifest`;
9. revalidates all frozen inputs and file identities immediately before an
   exclusive write; and
10. returns exactly the mapping written to the absent output.

Use existing `freeze_inputs()`, `verify_frozen()`, `file_sha256()`,
`write_json_exclusive()`, and `UniverseManifest.read()`. Add no abstraction or
dependency.

### Tests

Add focused procedural tests to the already-registered Massive suite:

- the production policy selects AAON and changes only rank `49`;
- ranks `0..48` and the `11/22/33` prefixes remain exact;
- the output parses through `UniverseManifest.read()`;
- an arbitrary replacement, wrong stratum, rejected candidate, changed input
  hash, changed base/master order, extra policy field, symlink input, or
  existing output is rejected;
- no output exists after any pre-publication failure.

Run:

```sh
$PRIMARY tests/python/test_massive.py
$PRIMARY -m py_compile tools/apply_universe_coverage_overlay.py
make -B PYTHON="$PRIMARY" check
```

Create one signed local checkpoint:
`feat(data): bind universe coverage overlay`. Do not push.

## Checkpoint 2: Fetch and measure the revised development universe

Generate the manifest only after checkpoint 1 is signed:

```sh
$PRIMARY tools/apply_universe_coverage_overlay.py \
  universes/liquid-common-55-coverage-v2.example.json \
  reports/universe-coverage-20260724-01/liquid-common-55.json

$PRIMARY tools/fetch_universe.py \
  reports/universe-coverage-20260724-01/liquid-common-55.json \
  data/liquid-common-55-20260724-03 \
  reports/liquid-common-55-20260724-03-fetch.json \
  --session-calendar \
  universes/us-equities-core-2024-07-22_2026-07-21.json \
  --requests-per-minute 5
```

The fetch must start from a fresh path and fetch all 55 series. Do not merge a
single downloaded CSV into the old report. Generated manifest, CSVs, and fetch
report remain ignored.

Measure only fold-0, fold-1, and calibration row availability. Expected
success is:

- core-11 remains complete;
- AAON has nonzero calibration rows;
- all 11 unseen names have nonzero calibration rows;
- coverage is derived through a timestamp-only reader; no protected OHLCV
  value is converted into a label, prediction, metric, or equity value.

If AAON fails, stop. A later v3 policy must predeclare POR before fetching it.

## Checkpoint 3: Rebind and run the scaling benchmark

### Files

- Modify: `tools/universe_scaling_contract.py`
- Modify: `tools/universe_scaling_inputs.py`
- Modify: `tools/arm_universe_scaling.py`
- Modify: `tests/python/test_universe_scaling_driver.py`
- Modify: `tests/python/test_universe_scaling_arm.py`

Replace path construction under one selection root with explicit
size-to-`FileBinding` values. Keep the original `11/22/33` manifests and bind
the generated overlay only for `55`.

Make `common_coverage()` measurement-only and timestamp-only: it must derive
phase counts and core/unseen availability without constructing `Bars` or
comparing them to hard-coded missing names. Move exact overlay-policy and
manifest hashes, expected phase coverage, update budgets, fetch paths, and
fit-count assertions into the armer, where a concrete attempt is frozen.

Run focused driver/armer tests, the aggregate gate, and optional Torch gate.
The bound runner and finalizer execute the zero, ridge, MLP, unconditioned
panel Transformer, conditioned panel Transformer, and local Transformer using
stock-balanced loss and five fixed seeds. DLinear requires its own later
checkpoint and must not be added to this frozen attempt implicitly.

## Checkpoint 4: Produce forward-state development forecasts

The scaling checkpoint selector currently chooses a model state using the same
phase it predicts. Those ledgers are valid model-development diagnostics, but
they are not forward-clean trading inputs. Do not pass them to a portfolio
engine as out-of-sample evidence.

Do not add a mode to the immutable scaling runner. Arm a separate forward
attempt only after its bound scaling outcome has status `pass`:

1. Consume the canonical fold-0 `selected_checkpoint` record for each exact
   `(question, mode, cohort, model, seed)` identity; do not retrain fold-0 to
   rediscover it.
2. Reinitialize deterministically, train on fold-1's training range for exactly
   that frozen checkpoint count and phase-specific updates per checkpoint,
   never evaluate fold-1 during fitting, then predict fold-1 once.
3. After fold-1 outcomes are complete, freeze the canonical fold-1
   `selected_checkpoint` record for calibration.
4. Reinitialize deterministically, train on calibration's training range for
   exactly that checkpoint count, never evaluate calibration during fitting,
   then predict calibration once.

A PASS-bound calibration schedule may be derived early for attempt hashing,
but derivation is not execution authorization. The runner must authenticate a
terminal fold-1 phase outcome that binds all five fits and every expected
prediction, with no extras, before materializing or consuming calibration
work. A schedule, checkpoint selection, or detached ledger closure is
insufficient.

After the scaling finalizer is terminal, add one no-validation primitive beside
`fit_epochs()` in `tools/train.py`:

```python
def fit_exact_updates(
    model: nn.Module, loader: DataLoader, updates: int,
    learning_rate: float, weight_decay: float, device: torch.device,
) -> int:
    """Fit one frozen update budget without validation or checkpoint selection."""
    if type(updates) is not int or updates < 1 or len(loader) != updates:
        raise ValueError("fixed-update loader does not match its budget")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    model.train()
    batches = iter(loader)
    for _ in range(updates):
        try:
            batch = next(batches)
        except StopIteration as error:
            raise ValueError(
                "fixed-update loader ended before its budget"
            ) from error
        _train_batch(model, batch, optimizer, device)
    return updates
```

The target phase trains for
`selected_checkpoint * target_budget.updates_per_checkpoint`; fold-1 therefore
uses `321` updates per checkpoint and calibration uses `368`. Reuse
`_neural_model()`, `_stock_uniform_loader()`, `_phase_data()`,
`model_fingerprint()`, and the existing fit, prediction, and market-truth
validators. Do not modify `experiment.py` or `run_universe_scaling.py`. Keep the
forward attempt, one-shot runner, terminal finalizer, and focused tests in
separate `universe_forward` files so the completed scaling evidence remains
immutable.

Every forward ledger record must bind the passing scaling outcome, its
prior-phase selection record, fit fingerprint, source/runtime/input hashes,
exact update count, phase, cohort, model, seed, series, and timestamp grid.
Mutating fold-1 validation or outcome labels must not change the fold-1 model
state; mutating calibration validation or outcome labels must not change the
calibration state. Mutating a phase's training labels must change its state.
Test exact update count, early loader exhaustion, non-finite loss rejection,
deterministic reinitialization, target-phase budget selection, and validation
and outcome label invariance before arming a forward attempt.

The first portfolio candidate is fixed before its results:

- question: `cohort-scaling`
- mode: `fixed-update`
- cohort: `55`
- model: `panel_transformer`
- seeds: `7, 19, 31, 43, 61`
- series: every development-evaluable manifest member, in manifest order

Cash and always-up require no model ledger. Any later forecast-model control
requires its own complete forward ledger. Records from another question, mode,
cohort, model, seed set, phase, or provenance identity must be rejected rather
than silently dropped.

## Checkpoint 5: Add one shared `$100` development portfolio

Keep the legacy per-series behavior unchanged. A behavior-preserving refactor
may move its execution-price and cash arithmetic into one shared pure helper;
do not duplicate that math. Add:

- `tools/portfolio_backtest.py`
- `tools/select_portfolio_policy.py`
- focused tests under `tests/python/`

After the scaling finalizer is terminal, add this public arithmetic seam beside
`Costs` and make the legacy `_execute()` adapter call it:

```python
@dataclass(frozen=True, slots=True)
class LongExecution:
    shares: float
    entry_execution_price: float
    exit_execution_price: float
    entry_notional: float
    exit_notional: float
    cash_after: float


def execute_long(
    cash: float, reference_price: float, outcome_price: float, costs: Costs,
) -> LongExecution:
    entry = reference_price * (1.0 + costs.impact)
    exit_ = outcome_price * (1.0 - costs.impact)
    shares = cash / (entry * (1.0 + costs.fee))
    entry_notional, exit_notional = shares * entry, shares * exit_
    return LongExecution(
        shares, entry, exit_, entry_notional, exit_notional,
        exit_notional * (1.0 - costs.fee),
    )
```

Reject non-finite or nonpositive cash, reference price, and outcome price before
performing this arithmetic; keep `Costs` validation as the single authority for
cost parameters.

The portfolio engine consumes validated, market-truth-joined opportunities:

```python
@dataclass(frozen=True, slots=True)
class PortfolioOpportunity:
    phase: str
    series: str
    manifest_rank: int
    as_of: str
    entry_time: str
    target_time: str
    reference_price: float
    outcome_price: float
    prediction_mean: float | None
    prediction_pstdev: float | None
```

`manifest_rank` is one-based: it equals `1 +` the zero-based master-manifest
index. The coverage overlay's `master_rank` remains zero-based and must never
enter a forward or portfolio record unchanged.

Keep authorization and truth access as separate API stages.
`validate_portfolio_phase()` consumes only the requested phase and action,
available prediction metadata, the passing forward contract, and the
digest-bound master manifest. It must reject protected or unbound phases and
provenance mismatches before receiving any price or outcome accessor. For
trading actions it also rejects manifest-map mismatches and incomplete seeds
when forecast metadata is present. Only its validated result may be passed with
such an accessor to `join_portfolio_truth()`, which constructs
`PortfolioOpportunity` values. Test the phase gate with an accessor that raises
if called.

Implement
`run_phase(opportunities: Sequence[PortfolioOpportunity], initial_cash: float,
costs: Costs, *, phase: str, action: str, safety_bps: float = 0.0,
disagreement_lambda: float = 0.0) -> dict[str, object]` only after the forward
ledger schema is frozen. The forward-ledger validator aggregates the exact
five-seed set once; policy trials reuse `prediction_mean` and
`prediction_pstdev`. `action` accepts exactly `long_above`, `always_up`, or
`cash`.

### Frozen execution contract

Use the existing `Costs` math with:

- full spread: `1` basis point;
- slippage: `1` basis point per side;
- proportional fee: `0`;
- cash yield: `0`;
- fractional long-or-cash positions, no leverage, shorting, tax, or minimum
  commission, and gross exposure at most `1`;
- invest `100%` of available cash in the single ranked winner.

This is a `$100` diagnostic, not a capacity model. Assume adjusted,
regular-session 30-minute bars, no separate corporate-action cash flow, and
notional too small to consume a material share of bar volume. Model both legs
as complete fills: a market proxy at the adjusted next-bar open plus impact,
followed by a market-on-close proxy at the adjusted target-bar close minus
impact. Omit latency, order rejection, partial fills, participation limits,
and size-driven impact until a separately frozen capacity experiment.

For impact

\[
i = \frac{\text{spread bps}}{20000}
  + \frac{\text{slippage bps}}{10000},
\]

the round-trip break-even threshold is

\[
\log\left(
\frac{(1+i)(1+f)}{(1-i)(1-f)}
\right)
+ \frac{\text{safety bps}}{10000}.
\]

`safety_bps` is an additive log-return buffer. With the frozen costs,
per-side impact is `1.5` basis points and the round-trip break-even is about
`3` log basis points, so the four safety values produce thresholds near
`3`, `6`, `9`, and `13` log basis points.

Aggregate all five seeds at one
`(phase, series, as_of, entry_time, target_time)` and score:

\[
s = \operatorname{mean}(\hat r)
  - \lambda\operatorname{pstdev}(\hat r).
\]

The population standard deviation measures disagreement among the five fixed
seeds; it is not calibrated predictive uncertainty. `lambda` is dimensionless
because both terms are log returns. Reject incomplete seed sets. Enter only
when `s` strictly exceeds break-even. The signal is known after `as_of`; enter
at `entry_time` open and exit at `target_time` close. Require:

```text
as_of < entry_time <= target_time
actual_return := log(outcome_price / reference_price)
```

The forward ledger has its own schema; it is not legacy backtest schema `2`.
Join it to the scaling finalizer's bound market truth before executing trades.
Derive `actual_return` exactly once from that joined truth; do not accept it
from a second field or file. The truth join rejects non-finite or nonpositive
reference and outcome prices before returning any opportunity.

For either trading action, validate every metadata record's provenance, phase,
and canonical UTC timestamps at the pre-join stage. Rows for one series may
repeat its rank over time. Build the distinct `series -> manifest_rank` map,
require one stable rank per series and no rank shared by two series, and require
the whole map to equal the digest-bound, development-evaluable master-manifest
map exactly. A unique but permuted map is invalid. Reject duplicate opportunity
identities and duplicate `(series, entry_time)` pairs. `run_phase()` requires
one explicit phase, groups entries, and processes those groups in chronological
order regardless of input order.

All actions require finite positive `initial_cash`. `long_above` requires
finite nonnegative `safety_bps` and `disagreement_lambda`, finite prediction
means, and finite nonnegative population deviations. `always_up` and `cash`
require both policy parameters to remain zero. `always_up` requires prediction
fields to be `None`. Both trading actions require nonempty opportunities;
`cash` requires an empty opportunity sequence and returns without consulting
predictions or prices. Its early path validates only phase authorization, bound
forward-contract and manifest identity, cash and parameter invariants, and the
empty sequence; it performs no map or truth join.

If `entry_time <= active.target_time`, reject the whole entry group because the
position remains active through the target close. Otherwise, rank
threshold-passing candidates by descending score and then ascending manifest
rank. Manifest ranks are unique, so equal scores deterministically choose the
lower rank; no ticker tie-break is reachable in valid input.

### Frozen policy selection

Use the Cartesian grid:

- safety basis points: `(0, 3, 6, 10)`
- disagreement penalty `lambda`: `(0, 0.5, 1)`
- cash: an explicit no-trade trial

Register exactly `12` forecast trials plus cash, for `13` trials total, before
reading results. Do not expand this grid after inspection. Select by highest
terminal log growth and then lower turnover. Cash wins an exact
growth-and-turnover tie against a forecast trial; among forecast trials, prefer
higher safety and then smaller lambda.

For each fold-1 selection trial, define gross turnover as the sum of entry and
exit notionals divided by that trial's initial `$100`. Fold-0 has no
forward-clean ledger and contributes neither portfolio-policy selection nor
evidence equity. Before any fold-1 truth is read, freeze its evidence rule as
`long_above` with `safety_bps = 0` and `disagreement_lambda = 0`. Start one
evidence account at `$100` under that rule and require every fold-1 position to
close before its boundary.

After fold-1 closes, evaluate the registered 13 trials on fold-1 from separate
`$100` selection accounts and select the rule for calibration. Carry the
evidence account's actual fold-1 terminal cash unchanged through the zero-yield
embargo into calibration. The selected rule controls only future calibration
actions; it cannot replace or rewind the fold-1 cash path. Combined evidence
turnover sums both phases' notionals and divides once by the original `$100`.
No position may cross a policy or phase boundary. Selection and control
accounts are counterfactual replays: they are neither simultaneously funded nor
pooled with the one carried `$100` evidence account, and selection-account cash
is discarded after ranking.

The comparison controls are:

- cash at `$100`, zero yield, carried across both evidence phases;
- always-up, which enters the lowest-manifest-rank eligible genuine bar whenever
  idle on the same phase-evaluable opportunity grid, under the same collision,
  holding-period, cost, and cross-phase cash-carry rules.

Omit buy-and-hold until an exact multi-stock basket, missing-bar, rebalance, and
exit contract is separately frozen.

The curve contains initial equity and post-exit realized-cash events only.
At event `t`, compute realized-event drawdown as
`1 - C_t / max(C_u for u <= t)`. Count rejections only as
`below_threshold`, `not_selected`, or `position_active`; invalid input raises
instead, and count each rejected opportunity rather than each group. Report
terminal equity, realized-event drawdown explicitly labeled as neither
bar-close nor intratrade risk, turnover, completed trades, phase coverage, and
those rejection counts. Do not synthesize daily/weekly/monthly equity or
compare this drawdown with legacy bar-close curves.

Every report fixes `evidence_role` to
`hypothetical-development-execution`, `cash_yield` to `0`, and `risk_scope` to
`realized-exit-events-only`. Do not annualize results or label them expected,
projected, live, deployable, or independently confirmed.

Rejection precedence is fixed. When a position is active, count every member of
the entry group as `position_active` without reading scores. When idle under
`long_above`, count each threshold failure as `below_threshold`, each passing
nonwinner as `not_selected`, and do not count the winner. Under `always_up`,
count every nonwinner as `not_selected`; under `cash`, all three counts are
zero.

Test invalid and non-finite execution inputs, costed and frictionless execution,
strict break-even comparison, exact 13-trial registration, equal-score
lower-rank selection, swapped-rank-map and duplicate-series-entry rejection,
entry-at-exit collision rejection, input-order invariance, action-specific
numeric validation, cross-phase cash carry, cash and always-up without model
records, protected phase rejection before truth access, and the absence of
period or bar-close risk claims. One zero-cost synthetic path must prove two
trades compound `$100 -> $110 -> $121`.

The finite grid and chronological selection/evidence split limit, but do not
eliminate, data-snooping risk; the result remains development evidence. This
scope follows the concerns in the
[backtest-overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
and [Reality Check](https://doi.org/10.1111/1468-0262.00152) literature.
Do not claim a
[Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
or formal probability of overfitting from this sparse realized-event path.

The fixed-impact, full-fill model is appropriate only for the registered
`$100` diagnostic. Before increasing notional, freeze spread/quote inputs,
volume participation, nonlinear temporary and permanent impact, latency,
partial-fill, and implementation-shortfall accounting. These extensions are
motivated by
[Almgren--Chriss](https://doi.org/10.21314/JOR.2001.041) and
[Perold](https://www.jstor.org/stable/j.ctv1mjqtwg.16), but are deliberately
outside this checkpoint.

### Frozen promotion gate

Freeze these predicates before reading portfolio results, then evaluate them
against the terminal portfolio outcome:

- exact candidate axes and complete five-seed forward ledgers;
- no protected rows and no position crossing a phase boundary;
- at least `8` completed trades in each evidence phase and `20` total;
- positive net log return in fold-1 and calibration separately;
- final equity above both `$100` cash and the costed always-up control.

This is a new scaling gate, not the historical H13 per-stock gate. Failure
of any predicate keeps the protected block closed.

Tests must prove:

- future-phase label mutation cannot change an earlier model or policy;
- fold-0 contributes neither portfolio-policy selection nor evidence equity;
- the fixed fold-1 evidence rule is bound before truth access and is unchanged
  by the fold-1-selected calibration portfolio policy;
- fold-1 terminal cash is calibration initial cash exactly;
- a divergent synthetic path proves calibration receives the fixed-rule
  evidence cash, not the winning selection trial's cash, and that every
  selection-account balance is discarded after ranking;
- a separate divergent path proves cash and always-up each start at `$100`,
  carry only their own balances across phases, and cannot alter evidence cash;
- equal-score lower-rank selection, exact manifest-map validation, duplicate
  rejection, seed completeness, costs, and returns are exact;
- an entry at the same timestamp as the prior target is rejected;
- exposure never exceeds one and positions never overlap or cross boundaries;
- unrelated phase/provenance/question/mode/cohort/model records are rejected;
- sparse reports contain no bar-close, period-return, or intratrade claim;
- protected rows are rejected before any outcome accessor is called;
- promotion-gate boundaries fail closed at `7` versus `8` trades in either
  phase, `19` versus `20` total trades, zero phase log return, and final-equity
  equality with either cash or always-up; every failed predicate keeps the
  protected block closed.

## Protected and future-confirmation boundary

The already-inspected historical block may later be replayed once as a labeled
exploratory check, but cannot restore confirmatory status. A separate reviewed
authorization must bind:

- immutable forecast and portfolio outcomes plus every promotion-gate value;
- the complete final policy grid and selected rule;
- all engine, ledger, source, runtime, environment, and input hashes;
- a timestamp-only digest over every protected
  `(series, as_of, entry_time, target_time)` before any price outcome is read;
- a final checkpoint index chosen from calibration evidence;
- deterministic refits for every seed on all development rows, exact update
  counts, serialized-state hashes or reproducible fit fingerprints;
- fresh, absent prediction/report/outcome paths and no CLI overrides.

Before any exploratory replay, reconstruct and verify every final fit. One
runner must exclusively write a terminal outcome on success, failure, or
interruption; a retry is a new exploratory attempt.

For genuine future confirmation, register the complete authorization manifest
as a user-signed commit on the remote repository before those future labels
exist. If the user has not explicitly authorized that publication, leave the
future test closed. A local file alone cannot prove that copied reruns did not
occur.
