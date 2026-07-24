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

```python
def replacement_candidate(
    selection: Mapping[str, object],
    failed_ticker: str,
) -> Mapping[str, object]: ...


def revised_manifest(
    base: Mapping[str, object],
    failed: Mapping[str, object],
    replacement: Mapping[str, object],
    *,
    purpose: str,
    declared_on: str,
) -> dict[str, object]: ...


def apply_overlay(
    policy_path: Path,
    output_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, object]: ...
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
Before arming, complete Tasks 2 through 4 of
`2026-07-24-universe-scaling-execution.md`; the referenced runner and finalizer
do not yet exist. Compare only the already-bound zero, ridge, MLP,
unconditioned panel Transformer, and conditioned panel Transformer using
stock-balanced loss and five fixed seeds. DLinear requires its own later
checkpoint and must not be added to this frozen attempt implicitly.

## Checkpoint 4: Produce forward-state development forecasts

The scaling checkpoint selector currently chooses a model state using the same
phase it predicts. Those ledgers are valid model-development diagnostics, but
they are not forward-clean trading inputs. Do not pass them to a portfolio
engine as out-of-sample evidence.

Add a forward-state mode to the scaling runner:

1. Train/evaluate fold-0 only to choose one checkpoint index per exact
   `(question, mode, cohort, model, seed)` identity.
2. Reinitialize deterministically, train on fold-1's training range for exactly
   that frozen index and phase-specific updates-per-checkpoint, never evaluate
   fold-1 during fitting, then predict fold-1 once.
3. After fold-1 outcomes are complete, use fold-1 only to choose the checkpoint
   index for calibration.
4. Reinitialize deterministically, train on calibration's training range for
   exactly that index, never evaluate calibration during fitting, then predict
   calibration once.

Every forward ledger record must bind its prior-phase selection record, fit
fingerprint, source/runtime/input hashes, exact update count, phase, cohort,
model, seed, series, and timestamp grid. Mutating fold-1 labels must not change
the fold-1 model state; mutating calibration labels must not change the
calibration state.

The first portfolio candidate is fixed before its results:

- question: `cohort-scaling`
- mode: `fixed-update`
- cohort: `55`
- model: `panel_transformer`
- seeds: `7, 19, 31, 43, 61`
- series: every development-evaluable manifest member, in manifest order

Controls use their own separately complete forward ledgers; records from
another question, mode, cohort, model, seed set, phase, or provenance identity
must be rejected rather than silently dropped.

## Checkpoint 5: Add one shared `$100` development portfolio

Do not change `tools/backtest.py`; it intentionally resets capital per
series/fold. Add:

- `tools/portfolio_backtest.py`
- `tools/select_portfolio_policy.py`
- focused tests under `tests/python/`

### Frozen execution contract

Use the existing `Costs` math with:

- full spread: `1` basis point;
- slippage: `1` basis point per side;
- proportional fee: `0`;
- cash yield: `0`;
- fractional shares, no leverage, and gross exposure at most `1`.

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

Aggregate all five seeds at one
`(phase, series, as_of, entry_time, target_time)` and score:

\[
s = \operatorname{mean}(\hat r)
  - \lambda\operatorname{pstdev}(\hat r).
\]

Reject incomplete seed sets. Enter only when `s` strictly exceeds break-even.
The signal is known after `as_of`; enter at `entry_time` open and exit at
`target_time` close. Require:

```text
as_of < entry_time <= target_time
actual_return == log(outcome_price / reference_price)
```

At one entry time, rank threshold-passing candidates by descending score, then
frozen manifest rank, then ticker. A position remains active through the target
close, so reject every candidate with
`candidate.entry_time <= active.target_time`.

### Frozen policy selection

Use the Cartesian grid:

- safety basis points: `(0, 3, 6, 10)`
- disagreement penalty `lambda`: `(0, 0.5, 1)`
- cash: an explicit no-trade trial

Select by highest terminal log growth, then lower turnover, higher safety, and
smaller lambda. Fold-0 is selection-only and contributes no evidence equity.
Start the evidence account at `$100` on fold-1 using the fold-0-selected rule.
Require all fold-1 positions closed before its boundary. Select the calibration
rule from fold-1 after fold-1 closes, then carry fold-1 terminal cash unchanged
through the zero-yield embargo into calibration. No position may cross a
policy or phase boundary.

The comparison controls are:

- cash at `$100`, zero yield;
- always-up, which enters the lowest-manifest-rank eligible genuine bar whenever
  idle, under the same collision, holding-period, and cost rules.

Omit buy-and-hold until an exact multi-stock basket, missing-bar, rebalance, and
exit contract is separately frozen.

The curve contains initial equity and post-exit realized-cash events only.
Report terminal equity, realized-event drawdown explicitly labeled as neither
bar-close nor intratrade risk, turnover, completed trades, phase coverage, and
rejection counts. Do not synthesize daily/weekly/monthly equity or compare this
drawdown with legacy bar-close curves.

### Frozen promotion gate

Before reading portfolio results, require all of:

- exact candidate axes and complete five-seed forward ledgers;
- no protected rows and no position crossing a phase boundary;
- at least `8` completed trades in each evidence phase and `20` total;
- positive net log return in fold-1 and calibration separately;
- final equity above both `$100` cash and the costed always-up control.

This is a new scaling gate, not the historical H13 per-stock gate. Failure
keeps the protected block closed.

Tests must prove:

- future-phase label mutation cannot change an earlier model or policy;
- fold-0 contributes zero evidence equity;
- fold-1 terminal cash is calibration initial cash exactly;
- simultaneous ties, seed completeness, manifest order, costs, and returns are
  exact;
- an entry at the same timestamp as the prior target is rejected;
- exposure never exceeds one and positions never overlap or cross boundaries;
- unrelated phase/provenance/question/mode/cohort/model records are rejected;
- sparse reports contain no bar-close, period-return, or intratrade claim;
- protected rows are rejected before any outcome accessor is called.

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
