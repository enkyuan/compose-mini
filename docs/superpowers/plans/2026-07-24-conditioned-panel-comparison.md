# Series-Conditioned Panel Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one immutable, calibration-only comparison that determines
whether a zero-initialized series embedding materially improves the shared
Transformer on the existing AAPL/MSFT/SPY horizon-13 panel.

**Architecture:** Keep the experiment producer unchanged and teach the frozen
panel analyzer to recognize two exact profiles: the completed one-panel
profile and a new two-panel comparison profile. The new profile pairs the
conditioned and unconditioned models inside one source/runtime closure,
preserves every stock's training-only scaler, and adds dependence-aware
calibration diagnostics before the one-shot driver may run.

**Tech Stack:** Python 3.12+, PyTorch through the existing offline uv runtime,
standard-library statistics and bootstrap code, procedural Python tests, Make,
GitButler.

## Global Constraints

- Follow `karpathy-guidelines`, `ponytail`, repository `AGENTS.md`, and
  GitButler-only version-control writes.
- Keep code direct, DRY, deterministic, and proportional to two exact
  comparison profiles; do not build a general experiment framework.
- Preserve the completed legacy panel config, attempt, outcome, evidence, and
  schema-1 analysis behavior byte-for-byte.
- Reuse `experiments/executable-h13-panel-inputs.json` unchanged.
- Use only AAPL, MSFT, and SPY in their declared order; raw-17; horizon 13;
  folds 2; seeds 7, 19, 31, 43, and 61.
- Keep feature and target scalers per-series and training-only.
- Keep the new model experiment-only and calibration-only. Do not change
  Artifact V1, the C runtime, policy models, test authorization, or backtest
  authorization.
- Never inspect reserved-test labels or emit test predictions in this plan.
- Do not fetch data, access Massive, install dependencies, or create a second
  data universe.
- Generated reports, ledgers, models, CSVs, bytecode, caches, and credentials
  remain ignored and untracked.
- Create signed local checkpoints authored by
  `enkyuan <yuan.enkng@gmail.com>`. Do not push, land, or open a pull request.
- Existing uncommitted `Makefile` and `docs/training.md` hunks belong to other
  work; do not absorb or rewrite them.

---

## File and Responsibility Map

- Modify `tools/analyze_panel.py`: select one of two exact profiles, validate
  the conditioned report, compute paired and dependence-aware evidence, and
  preserve legacy output.
- Modify `tools/panel_contract.py`: share exact profile, selected-source-tree,
  command-construction, and Torch-observation primitives between validation,
  finalization, and arming.
- Modify `tests/python/test_panel_analysis.py`: parameterize the existing
  synthetic fixture and prove both profiles, ordering, integrity, and gate
  boundaries.
- Modify `tools/finalize_panel_attempt.py`: accept analysis schema 2 only for
  the exact 207-run comparison while preserving legacy schema 1.
- Modify `tools/run_panel_attempt.py`: bind the one-shot driver to the new
  reviewed config/attempt/run/outcome and 207-run cap.
- Add `tools/arm_panel_attempt.py`: exclusively construct a validated attempt
  from selected source paths and the unchanged historical runtime identity.
- Modify `tests/python/test_panel_driver.py`: prove the new exact bindings and
  retain every exactly-once, signal, cleanup, symlink, and runtime test.
- Add `experiments/executable-h13-conditioned-panel.example.json`: the exact
  seven-model comparison sweep.
- Add `experiments/executable-h13-conditioned-panel-attempt.json` only after
  implementation review: the immutable one-shot manifest.
- Add `experiments/executable-h13-conditioned-panel-outcome.json` only through
  the finalizer after the run.
- Add `docs/experiments/h13-conditioned-panel-20260724-01.md` only after the
  run: concise immutable evidence, never generated report contents.

## Locked Estimands

Let \(U\) be `panel_transformer`, \(C\) be
`conditioned_panel_transformer`, \(S=3\) stocks, \(F=2\) folds, and \(K=5\)
seeds.

For validation return MAE \(a^V_{m,s,f,k}\):

```text
d[s,f,k] = a_validation[C,s,f,k] - a_validation[U,s,f,k]

A_validation[m] =
  (1 / (S * F * K)) * sum_s,f,k a_validation[m,s,f,k]

R_validation =
  1 - A_validation[C] / A_validation[U]
```

For calibration prediction \(p[m,s,k,t]\), ensemble seeds before metrics:

```text
p_bar[m,s,t] = (1 / K) * sum_k p[m,s,k,t]

A_calibration[m,s] =
  (1 / T) * sum_t abs(p_bar[m,s,t] - actual[s,t])

A_calibration[m] = (1 / S) * sum_s A_calibration[m,s]
R_calibration = 1 - A_calibration[C] / A_calibration[U]
```

Same-seed pairing is required because the conditioned model starts as the
exact unconditioned function: its embedding is zero-initialized and added
before the encoder. Pairing removes much of the shared initialization and
data-order noise.

Every MAE used as a relative-improvement denominator must be finite and
strictly positive, including validation, full calibration, leave-one-seed-out,
bootstrap, per-stock unconditioned close MAE, and per-stock zero-return close
MAE. A zero or non-finite denominator is an integrity error, never a zero
improvement.

The minimum practical improvement is fixed at one percent:

```text
R_validation >= 0.01
R_calibration >= 0.01
```

That threshold is predeclared from existing admissible evidence. A one-percent
improvement over the frozen unconditioned result is approximately 0.96 return
basis points on validation and 1.16 return basis points on calibration; it is
large enough to clear the current best validation MLP and calibration
zero-return references. It is not a profitability claim.

The 30 paired validation units are not IID: seeds reuse labels, stocks share
market regimes, and only two temporal folds exist. Report stock-, fold-, and
seed-axis means and population dispersions; do not report
`standard_deviation / sqrt(30)` as a confidence interval.

The existing 20-of-30 win threshold is retained as a robustness count. Under
the hypothetical assumptions that all 30 signs were independent,
exchangeable, tie-free Bernoulli draws with 50-percent win probability:

```text
P(Binomial(30, 0.5) >= 20) = 0.04936857335269451
```

This is only an independence-based combinatorial reference. The actual signs
share labels, regimes, folds, and seeds, so `0.049368...` is not a valid
experiment p-value or significance guarantee. Ties are never wins and make
the displayed tie-free reference inapplicable.

## Locked Calibration Bootstrap

Use one deterministic paired moving-block-bootstrap sensitivity set:

```text
replicates = 10_000
PRNG seed = 20_260_724
block lengths = 13, 29, and 39 target rows
quantile = nearest-rank 2.5th percentile
reported gate statistic = minimum lower percentile across block lengths
```

The three predeclared lengths cover the 13-bar output horizon, the 29-row
mechanical dependence span induced by the 17-bar input window plus that
horizon, and one longer sensitivity case. For each block length, reset a
`random.Random(20_260_724)` instance, sample non-circular block starts
uniformly from `0..T-block`, append complete consecutive blocks until at
least `T` indexes exist, then truncate to `T`. Use the same sampled indexes
for both models and all three stocks within every replicate. This preserves
paired model comparisons and observed cross-stock dependence without
pretending the forecast horizon alone determines the dependence length.

The lower percentile is:

```python
def _lower_025(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.025 * len(ordered)) - 1]
```

This calibration period has already been reused for model development and
helped motivate both series conditioning and the one-percent threshold.
Bootstrap percentiles are therefore conditional descriptive stability
diagnostics, not confirmatory confidence intervals, nominal 95-percent
coverage, or Type-I-error-controlled tests. They are conditional on these
three stocks, five seeds, and this period; they do not estimate unseen-ticker
or future-regime uncertainty. Only the untouched reserved test can later
provide confirmation, after a separate frozen policy and explicit
authorization.

## Locked Work Accounting

```text
local models:
  5 Transformer seeds + 1 linear + 5 MLP seeds
  + 1 rolling mean + 1 zero return = 13 runs per stock/phase

two panel models:
  5 + 5 = 10 series-equivalent runs per stock/phase

series-equivalent runs:
  3 stocks * (13 + 10) * (2 validation folds + 1 calibration)
  = 207

physical panel fits:
  2 panel models * 5 seeds * (2 folds + 1 calibration)
  = 30
```

## Frozen Promotion Gates

Every gate is conjunctive. Equality fails strict comparisons.

1. `validation_macro_mae`
   - \(R_validation >= 0.01\).
   - Conditioned macro return MAE is strictly below unconditioned panel, local
     Transformer, MLP, linear, rolling mean, and zero return.
2. `validation_paired`
   - Mean `conditioned - unconditioned` delta is strictly negative.
   - At least 20 of 30 deltas are negative.
   - Every stock-axis mean is strictly negative.
   - Both fold-axis means are strictly negative.
   - At least four of five seed-axis means are strictly negative.
3. `calibration_macro_mae`
   - \(R_calibration >= 0.01\).
   - For every predeclared block length, compute the moving-block-bootstrap
     2.5th percentile of \(R_calibration\); their minimum is at least 0.01.
   - All five leave-one-seed-out ensembles have positive relative improvement.
   - Conditioned macro return MAE is strictly below every comparator listed in
     gate 1.
4. `calibration_per_stock_zero`
   - For AAPL, MSFT, and SPY separately, conditioned return MAE is strictly
     below both unconditioned panel and zero return.
5. `calibration_direction`
   - Conditioned macro direction is strictly above unconditioned panel and the
     macro majority-sign reference.
   - Every stock is at least 50 percent and strictly above its unconditioned
     direction.
   - For each direction statistic, compute a paired-bootstrap 2.5th
     percentile at every predeclared block length; the minimum
     conditioned-minus-unconditioned value is strictly positive.
   - The minimum conditioned-minus-resampled-majority value is strictly
     positive. Recompute each stock's majority sign from that replicate's
     sampled actuals.
6. `calibration_close_mae`
   - Mean per-stock relative close-MAE improvement over zero return is
     strictly positive.
   - Mean per-stock relative close-MAE improvement over the unconditioned
     panel is strictly positive.

No gate, seed subset, block-length set, threshold, or comparator may change
after the run starts. A failure is a valid negative result and permanently
closes this attempt without test access or a `$100` backtest.

---

## Task 1: Review and Checkpoint This Plan

**Files:**

- Add: `docs/superpowers/plans/2026-07-24-conditioned-panel-comparison.md`

**Interfaces:**

- Consumes: reviewed implementation checkpoint `5084605`; completed legacy
  panel evidence and input manifest.
- Produces: the immutable engineering and statistical contract for Tasks 2-6.

- [ ] **Step 1: Self-review the complete plan**

Check every Global Constraint, profile, model order, count, gate, run path, and
terminal rule. Run the placeholder scan from Step 3; it must return no matches.

- [ ] **Step 2: Request independent reviews**

Require:

- engineering review: profile selection, legacy compatibility, exact ordering,
  source/runtime closure, finalizer/driver transitions, and run accounting;
- methodology review: paired units, one-percent effect, sign-test meaning,
  block bootstrap, leave-one-seed-out ensembles, calibration reuse, and
  no-test/no-backtest boundary.

Resolve every Important finding in this file and re-review.

- [ ] **Step 3: Verify and checkpoint**

Run:

```zsh
rg -n 'T[B]D|T[O]DO|implement l[a]ter|similar t[o]' \
  docs/superpowers/plans/2026-07-24-conditioned-panel-comparison.md
```

Expected: no matches.

Commit only this plan on
`enkyuan/conditioned-panel-comparison-plan`, stacked directly above
`enkyuan/conditioned-panel`:

```text
docs(training): plan conditioned panel comparison
```

Do not push.

## Task 2: Add Exact Analyzer Profiles and Comparison Evidence

**Files:**

- Modify: `tools/analyze_panel.py`
- Modify: `tools/panel_contract.py`
- Modify: `tests/python/test_panel_analysis.py`
- Modify: `tools/finalize_panel_attempt.py`

**Interfaces:**

- Consumes: report schema 6, ledger schema 3, `PanelAttempt`,
  `PanelInputs`, and the exact producer ordering in `tools/experiment.py`.
- Produces: shared `PanelProfile`, `panel_profile`, schema-1 legacy analysis,
  schema-2 conditioned analysis, and the six frozen gate results.

- [ ] **Step 1: Write profile and ordering tests**

Parameterize the existing `PanelFixture`; do not clone it. Add fixture fields
for `profile`, `models`, `panel_models`, `candidate`, `reference`,
`expected_runs`, `expected_panel_fits`, and `analysis_schema`.

Require these exact profiles:

```python
LEGACY_MODELS = (*LOCAL_MODELS, "panel_transformer")
COMPARISON_MODELS = (
    *LOCAL_MODELS,
    "panel_transformer",
    "conditioned_panel_transformer",
)
```

Tests must prove:

- legacy remains 162/15, schema 1, globally lexical fingerprints, and has no
  `panel_conditioning`;
- one deterministic legacy fixture's complete normalized analysis JSON equals
  a checked-in expected recursive dictionary, protecting every schema-1 key
  and value from generic refactors;
- comparison is 207/30, schema 2, and requires:

```json
{
  "model": "conditioned_panel_transformer",
  "kind": "learned-series-embedding",
  "series_order": ["AAPL", "MSFT", "SPY"],
  "initialization": "zeros",
  "application": "additive-before-encoder"
}
```

- comparison validation/calibration/ledger order is local records unchanged,
  then all unconditioned panel records, then all conditioned panel records,
  each in bound series/fold/seed/target order;
- comparison fingerprints are local lexical fingerprints followed by panel
  model, bound series, and configured seed order;
- partial, extra, missing, or reordered profiles fail;
- missing/reordered conditioned records, ledger rows, fingerprints, seeds, or
  conditioning metadata fail;
- shared fit telemetry and selected epochs are validated independently for
  each panel model;
- local records, fingerprints, and ledger lines still reproduce the frozen
  baseline exactly;
- neither panel model enters policy, selection, replay, or backtest support;
- `test` remains empty.

The legacy expected dictionary is built from that fixture's exact resolved
paths and computed hashes. Do not replace paths/hashes with broad wildcard
tokens or normalize any nonvolatile model, protocol, metric, gate, or
provenance value.

Run with the bundled Python. Expected before production changes: failure on
the missing conditioned profile.

- [ ] **Step 2: Add the minimum shared profile type**

Add to `tools/panel_contract.py`:

```python
@dataclass(frozen=True)
class PanelProfile:
    models: tuple[str, ...]
    panel_models: tuple[str, ...]
    candidate: str
    reference: str
    expected_runs: int
    expected_panel_fits: int
    analysis_schema: int
```

Define only:

```python
LEGACY_PROFILE = PanelProfile(
    (*LOCAL_MODELS, "panel_transformer"),
    ("panel_transformer",),
    "panel_transformer", "transformer", 162, 15, 1,
)
COMPARISON_PROFILE = PanelProfile(
    (*LOCAL_MODELS, "panel_transformer",
      "conditioned_panel_transformer"),
    ("panel_transformer", "conditioned_panel_transformer"),
    "conditioned_panel_transformer", "panel_transformer", 207, 30, 2,
)
PROFILES = (LEGACY_PROFILE, COMPARISON_PROFILE)
```

`expected_panel_sweep(profile)` returns the complete exact sweep dictionary,
and `panel_profile(config)` accepts only exact decoded equality with one of
those two dictionaries, including ordered models. Store the chosen profile on
`BoundPanel`. Analyzer, arming tool, and finalizer must all call these shared
functions. Pass the profile to command/attempt validation, report validation,
record grids, metrics, gates, and analysis serialization. No component may
infer a profile from run counts alone.

`panel_contract.py` must remain standard-library-only and inside the existing
narrow finalizer closure. Encode the target kind as the contract literal
`"executable-return-v1"`; do not import `data_v1.py`, PyTorch, the analyzer, or
another unbound module. Analyzer tests must assert that this literal equals
`EXECUTABLE_RETURN_TARGET`. Preserve type-sensitive exact JSON equality, so
`1`, `1.0`, and `True` are not interchangeable.

- [ ] **Step 3: Share exact arming and validation primitives**

In `tools/panel_contract.py`, add:

```python
def selected_source_tree(
    root: Path, paths: Sequence[str],
) -> SourceTree:
```

Resolve `root`, require every declared relative path exactly once, reject
symlinks/non-regular files, hash only those files in sorted relative-path
order, and use the existing `_tree_digest`. Keep `source_tree(root)` unchanged
for full runtime-package hashing.

Move the current Torch observation subprocess into:

```python
def observe_torch(
    torch_argv: Sequence[str], root: Path,
) -> TorchIdentity:
```

Have the analyzer call this shared helper. Add:

```python
def expected_panel_commands(
    attempt_path: Path,
    input_manifest_path: str,
    config_path: str,
    baseline_report_path: str,
    baseline_ledger_path: str,
    outputs: Mapping[str, str],
    inputs: PanelInputs,
    profile: PanelProfile,
) -> Mapping[str, tuple[str, ...]]:
```

It constructs the exact
validate/preflight/experiment/analyze/finalizer arrays from explicit bound
paths, outputs, ordered `PanelInputs.series`, and
`profile.expected_runs`. The analyzer and Task 3 arming tool must call the
same helper.

Tests must prove selected trees contain exactly `SOURCE_PATHS` or
`FINALIZER_SOURCE_PATHS`, reject a missing/symlinked/duplicate path, and have
the documented sorted `path + NUL + sha256 + LF` digest.

- [ ] **Step 4: Generalize exact report validation**

Add the conditioned model to `SEEDED_MODELS`. Change
`_validation_keys(names, panel_models)` and
`_calibration_keys(names, panel_models)` to loop over the ordered tuple.

For legacy fingerprints, retain the exact global lexical key:

```python
(model, series, -1 if seed is None else seed)
```

For comparison fingerprints, expect:

```python
[
    *sorted(local_keys, key=legacy_key),
    *(
        (model, series, seed)
        for model in profile.panel_models
        for series in bound_series
        for seed in SEEDS
    ),
]
```

Require shared validation telemetry for every
`(panel_model, fold, seed)` and shared calibration epochs for every
`(panel_model, seed)`.

`_expected_protocol` must preserve the legacy object exactly and append only
the exact producer `panel_conditioning` object for the comparison profile.

- [ ] **Step 5: Add generic paired metrics**

Keep legacy analysis keys unchanged. Schema 2 emits:

```text
candidate_model = conditioned_panel_transformer
reference_model = panel_transformer
paired_candidate_minus_reference
```

For the 30 deltas, add one helper that groups by stock, fold, and seed and
returns:

```python
{
    "count": len(values),
    "mean": fmean(values),
    "stddev": pstdev(values),
    "minimum": min(values),
    "maximum": max(values),
}
```

Do not change the existing `_stats` helper because it is part of exact legacy
report validation.

- [ ] **Step 6: Preserve seed predictions and add robustness metrics**

Extend `_validate_ledger` to return the validated per-seed prediction mapping
alongside actuals and full ensembles. Compute five leave-one-seed-out
ensembles from the exact declared seed tuple.

Add standard-library moving-block helpers:

```python
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_724
BOOTSTRAP_BLOCKS = (13, 29, 39)


def _block_indexes(
    size: int, block: int, rng: random.Random,
) -> tuple[int, ...]:
    if size < block:
        raise ValueError("calibration grid is shorter than one block")
    result: list[int] = []
    while len(result) < size:
        start = rng.randrange(size - block + 1)
        result.extend(range(start, start + block))
    return tuple(result[:size])
```

Run the same bootstrap loop independently for each block length, resetting
`random.Random(BOOTSTRAP_SEED)` at the start of each length. For every sampled
index tuple, compute candidate/reference macro MAEs, macro direction
accuracies, and the resampled per-stock majority-sign macro from the same rows
and append:

```text
1 - candidate_macro_mae / reference_macro_mae
candidate_macro_direction - reference_macro_direction
candidate_macro_direction - resampled_majority_macro
```

Require every reference-MAE denominator to be finite and strictly positive for
the full ensemble, each leave-one-seed-out ensemble, and every bootstrap
replicate. Report each block length's three lower percentiles and their
componentwise minima. Use `_lower_025` from the locked definition. Do not
bootstrap models independently or resample stocks independently.

- [ ] **Step 7: Implement the six exact gates**

Make candidate/reference explicit in every schema-2 diagnostic. Implement the
Frozen Promotion Gates exactly. Keep the legacy gate computation and nested
field names unchanged under `LEGACY_PROFILE`.

The schema-2 protocol must record:

```json
{
  "candidate_model": "conditioned_panel_transformer",
  "reference_model": "panel_transformer",
  "minimum_relative_mae_improvement": 0.01,
  "calibration_role": "reused-development-diagnostic",
  "uncertainty_interpretation": "conditional-descriptive-not-confirmatory",
  "bootstrap": {
    "kind": "paired-noncircular-moving-block",
    "block_rows": [13, 29, 39],
    "replicates": 10000,
    "seed": 20260724,
    "lower_percentile": 0.025,
    "gate_aggregation": "minimum-lower-percentile-across-block-lengths"
  }
}
```

This metadata is additive to the existing analysis protocol, not the
experiment report protocol.

The schema-2 top-level fields remain exactly:

```text
schema, status, inputs, protocol, validation, calibration, gates
```

Its `protocol` contains the existing exact keys:

```text
candidate, models, seeds, series, folds, fold_fraction,
target_horizon_bars, target_kind, series_equivalent_runs,
physical_panel_fits, validation_pair, calibration_ensemble,
macro_unit, majority_reference
```

plus exactly:

```text
candidate_model, reference_model, minimum_relative_mae_improvement,
calibration_role, uncertainty_interpretation, bootstrap
```

Schema-2 `validation` has exactly:

```text
macro_return_mae
paired_candidate_minus_reference
```

`macro_return_mae` has one finite value for each of the seven ordered models.
`paired_candidate_minus_reference` has exactly:

```text
candidate_model, reference_model, relative_improvement,
mean_delta, wins, ties, losses, by_stock, by_fold, by_seed
```

Each axis maps only the exact declared stock, fold, or seed keys to:

```text
count, mean, stddev, minimum, maximum
```

The exact JSON string keys are:

```python
STOCK_KEYS = ("AAPL", "MSFT", "SPY")
FOLD_KEYS = ("0", "1")
SEED_KEYS = ("7", "19", "31", "43", "61")
COMPARATOR_KEYS = (
    "unconditioned_panel", "local_transformer", "mlp", "linear",
    "rolling_mean", "zero_return",
)
```

`by_stock`, `by_fold`, and `by_seed` values are always the five-field axis
statistics object above, never bare means.

Schema-2 `calibration` has exactly:

```text
macro_return_mae, macro_direction_accuracy, macro_majority_direction,
relative_improvement_vs_reference, leave_one_seed_out,
bootstrap, mean_candidate_close_relative_improvement_over_zero,
mean_candidate_close_relative_improvement_over_reference, per_stock
```

`leave_one_seed_out` has exactly `SEED_KEYS`, each mapping to:

```text
relative_improvement
```

as a one-field object, not a bare number. `bootstrap` has exactly:

```text
by_block_rows
mae_relative_improvement_lower_025
direction_candidate_minus_reference_lower_025
direction_candidate_minus_majority_lower_025
```

`by_block_rows` has exactly the string keys `"13"`, `"29"`, and `"39"`.
Each maps to exactly the same three lower-percentile scalar fields. The three
top-level scalars are the componentwise minima of those per-block values and
are the values consumed by the gates.

Each `per_stock` entry has exactly:

```text
samples, models, majority_direction, zero_return_return_mae,
zero_return_close_mae, candidate_close_relative_improvement_over_zero,
candidate_close_relative_improvement_over_reference
```

`models` contains all seven exact model keys, each with exactly
`return_mae`, `direction_accuracy`, and `close_mae`.

Each per-stock `majority_direction` has exactly:

```text
p_up, p_down, p_flat, reference
```

Schema-2 `gates` has the existing six names plus `all_pass`. Require these
exact fields:

```text
validation_macro_mae:
  pass, candidate_model, reference_model, candidate, comparators, margin,
  relative_improvement, required_relative_improvement

validation_paired:
  pass, candidate_model, reference_model, mean_delta, wins, ties, losses,
  required_wins, by_stock, by_fold, by_seed, required_improving_seeds

calibration_macro_mae:
  pass, candidate_model, reference_model, candidate, comparators, margin,
  relative_improvement, required_relative_improvement,
  bootstrap_lower_025, leave_one_seed_out

calibration_per_stock_zero:
  pass, candidate_model, reference_model, per_stock

calibration_direction:
  pass, candidate_model, reference_model, candidate_macro, reference_macro,
  majority_macro, reference_margin, majority_margin,
  reference_bootstrap_lower_025, majority_bootstrap_lower_025, per_stock

calibration_close_mae:
  pass, candidate_model, reference_model,
  mean_relative_improvement_over_zero,
  mean_relative_improvement_over_reference
```

Both macro-gate `comparators` objects have exactly `COMPARATOR_KEYS`, each
mapping to one finite MAE. `validation_paired` reuses the exact full axis maps,
not reduced means.

Each `calibration_per_stock_zero.per_stock[stock]` has exactly:

```text
candidate_return_mae, reference_return_mae, zero_return_mae,
reference_margin, zero_margin, reference_pass, zero_pass
```

Each `calibration_direction.per_stock[stock]` has exactly:

```text
candidate, reference, minimum, reference_margin, minimum_margin,
reference_pass, minimum_pass
```

`calibration_macro_mae.leave_one_seed_out` reuses the exact five one-field
objects from `calibration.leave_one_seed_out`.

`all_pass` must equal the conjunction of the six gate booleans. No extra or
missing nested field is accepted.

- [ ] **Step 8: Bind finalizer schema to the exact profile**

In `_validate_provenance`, read the already frozen, attempt-bound config and
select `panel_profile(config)`, which proves the complete exact sweep rather
than model names or counts alone. Require attempt counts, exact shared command
profile, schema, candidate, reference, model order, practical threshold,
bootstrap contract, and every nested schema-2 field set to match that profile.
A forged object containing only six booleans must fail.

Add one self-contained `validate_panel_analysis(value, profile)` contract
function in `panel_contract.py`. Both the analyzer, immediately before writing,
and the finalizer must call it. It validates every recursive field/type,
finite range, exact key set, and then recomputes each gate Boolean from the
reported numeric diagnostics:

```text
validation macro:
  relative >= 0.01 and candidate < every comparator

validation paired:
  mean < 0, wins >= 20, all stock/fold means < 0,
  at least four seed means < 0

calibration macro:
  relative >= 0.01, candidate < every comparator,
  every per-block lower is finite, the reported bootstrap lower equals their
  minimum and is >= 0.01, every leave-one-out relative > 0

calibration per stock:
  every reference_margin > 0 and zero_margin > 0

calibration direction:
  both macro margins > 0, every per-block lower is finite, both reported
  bootstrap lowers equal their per-block minima and are > 0,
  every reference_margin > 0 and minimum_margin >= 0

calibration close:
  both mean relative improvements > 0
```

It also recomputes `all_pass`. Contradictory numeric values with `pass=true`
are integrity errors. This shared validator does not recompute predictions or
bootstrap samples; the bound analyzer source produces those diagnostics, while
the validator prevents structural or Boolean semantic forgery.

Keep outcome schema 1, top-level analysis fields, terminal transitions, and
the six gate names unchanged. Add finalizer tests for legacy pass/failure,
comparison pass/failure, swapped candidate/reference, altered bootstrap
parameters, partial/reordered profiles, unknown counts, missing fields, extra
fields, contradictory numeric/Boolean gate values, and schema/profile
mismatch.

- [ ] **Step 9: Prove the mathematical boundaries**

Tests must cover:

- conditioned/unconditioned equality fails both practical MAE gates;
- 0.9999-percent improvement fails and one percent passes;
- 19 wins fail and 20 wins pass, with ties excluded;
- one nonnegative stock or fold mean fails;
- three improving seed means fail and four pass;
- seed averaging occurs before calibration metrics;
- every leave-one-seed-out relative improvement must be positive;
- identical bootstrap draws are used across models/stocks;
- the fixed bootstrap seed is deterministic;
- one minimal 39-row fixture executes the unpatched three-block,
  10,000-replicate path and asserts all nine per-block lower percentiles plus
  the exact three componentwise minima;
- a separate greater-than-39-row fixture with a stubbed sequence of starts
  proves end-to-end randomized block selection is nondegenerate for every
  declared block length;
- 2.5th-percentile MAE equality at one percent passes, while direction
  equality at zero fails;
- `_block_indexes` with a stub RNG returns exact contiguous blocks for 13,
  29, and 39 rows, truncates to the requested length, never wraps, and rejects
  sizes below the selected block;
- `_lower_025` over 10,000 distinct ordered values returns the 250th order
  statistic;
- five asymmetric seed streams produce the five exact leave-one-seed-out
  predictions and relative improvements, proving named-seed exclusion and
  prediction averaging before MAE;
- one asymmetric `3 * 2 * 5` delta cube produces exact ordered stock/fold/seed
  counts, means, population standard deviations, minima, and maxima;
- zero reference-MAE denominators fail for validation, full calibration,
  leave-one-out, bootstrap, unconditioned close, and zero-return close
  relative improvements;
- resampled majority direction is recomputed from each replicate's sampled
  actuals and uses the same paired indexes;
- one per-stock reference or zero-return failure fails the gate;
- direction and both close-MAE references are independently enforced.

Run:

```zsh
/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_panel_analysis.py
```

Expected: `panel analysis tests passed`.

Commit Task 2 with its tests on
`enkyuan/conditioned-panel-comparison`, stacked above the plan:

```text
feat(training): analyze conditioned panel evidence
```

Do not push.

## Task 3: Freeze the New Config and One-Shot Driver

**Files:**

- Add: `experiments/executable-h13-conditioned-panel.example.json`
- Add: `tools/arm_panel_attempt.py`
- Modify: `tools/run_panel_attempt.py`
- Modify: `tests/python/test_panel_analysis.py`
- Modify: `tests/python/test_panel_driver.py`

**Interfaces:**

- Consumes: the exact reviewed comparison profile, existing input manifest and
  baseline paths, shared panel-contract primitives, and the established
  exactly-once driver/finalizer state machine.
- Produces: one exact unarmed comparison surface and a deterministic,
  no-clobber arming command for Task 5.

- [ ] **Step 1: Add the exact config**

Use:

```json
{
  "alignment_horizon_bars": 13,
  "batch_size": 128,
  "candidates": [
    {
      "feature_set": "ohlcv",
      "ff_dim": 32,
      "heads": 2,
      "layers": 1,
      "learning_rate": 0.0003,
      "mlp_dim": 32,
      "model_dim": 16,
      "name": "raw-17",
      "ridge": 0.001,
      "rolling_window": 8,
      "seq_len": 17,
      "weight_decay": 0.0001
    }
  ],
  "epochs": 100,
  "fold_fraction": 0.1,
  "folds": 2,
  "models": [
    "transformer",
    "linear",
    "mlp",
    "rolling_mean",
    "last_close",
    "panel_transformer",
    "conditioned_panel_transformer"
  ],
  "patience": 10,
  "seeds": [7, 19, 31, 43, 61],
  "target_horizon_bars": 13,
  "target_kind": "executable-return-v1"
}
```

Assert exact decoded equality and `expected_runs(sweep, 3) == 207`.

- [ ] **Step 2: Write failing driver-binding tests**

Require:

```text
ATTEMPT = experiments/executable-h13-conditioned-panel-attempt.json
INPUTS = experiments/executable-h13-panel-inputs.json
CONFIG = experiments/executable-h13-conditioned-panel.example.json
BASELINE_REPORT = reports/executable-h13-calibration.json
BASELINE_LEDGER = reports/executable-h13-calibration.jsonl
RUN_DIR = reports/h13-conditioned-panel-20260724-01
OUTCOME = experiments/executable-h13-conditioned-panel-outcome.json
SERIES = AAPL, MSFT, SPY in that order
```

The experiment argv must include `--calibration-only` and
`--max-runs 207`; it must not include `--predictions`, `--policy`,
`--authorization`, `--test`, or a backtest command.

Retain every existing test for exactly-once stages, finalization, signals,
descendant cleanup, symlink rejection, fresh run setup, and primary runtime.

- [ ] **Step 3: Repoint only immutable driver constants**

Change only `ATTEMPT`, `CONFIG`, `RUN_DIR`, `OUTCOME`, and the run cap.
Explicitly retain the existing `INPUTS`, `BASELINE_REPORT`,
`BASELINE_LEDGER`, and ordered `SERIES`. Do not duplicate the driver, alter its
process lifecycle, or generalize it to execute arbitrary manifest commands.
The completed legacy outcome already binds the exact historical driver source
hash.

- [ ] **Step 4: Add a deterministic no-clobber arming tool**

`tools/arm_panel_attempt.py` accepts:

```text
OUTPUT
--runtime-template LEGACY_ATTEMPT
--input-manifest INPUTS
--config CONFIG
--implementation-commit 40_HEX
--run-id RUN_ID
--run-dir RUN_DIR
--outcome OUTCOME
```

It must:

1. Parse the historical template with `PanelAttempt.read`.
2. Read the input manifest and exact config; select the exact shared
   `PanelProfile`, never counts alone.
3. Require the provided 40-character implementation commit.
4. Require the supplied input-manifest binding to equal the historical
   attempt's `input_manifest` binding exactly. Require its baseline report and
   ledger bindings to equal both the historical attempt and parsed
   `PanelInputs`. Freeze and validate the three declared CSVs against that
   unchanged manifest.
5. Require fresh output, run directory, cache prefix, and outcome paths.
6. Validate the template primary Python and uv bindings live.
7. Call shared `observe_torch`; require byte-for-byte equality with the
   template's `primary_python`, `uv`, `torch_argv`, and complete
   `torch_probe`. The cache-prefix path is the only intentional runtime
   environment difference.
8. Build source and finalizer closures with `selected_source_tree` over the
   exact declared path tuples.
9. Bind the frozen input/config/baseline hashes, exact profile counts,
   environment, outputs, and shared `expected_panel_commands`.
10. Serialize canonical JSON with `write_json_exclusive`; never overwrite.
11. Parse the created file with `PanelAttempt.read` and require its fields to
    equal the constructed values.

Before hashing, freeze the template, input manifest, config, baselines, CSVs,
every selected source/finalizer file, and all runtime executables using
`freeze_inputs` plus `regular_file_identities`. Compute manifest hashes from
the frozen snapshots. Open and anchor the nonsymlink output parent with a
directory descriptor/identity. Pass that descriptor and a `before_link`
callback to `write_json_exclusive`; immediately before publication, the
callback must recheck every frozen input and regular identity, both selected
trees, the complete live runtime identity, all required absent paths, and the
output-parent identity.

Add synthetic tests for deterministic output, selected tree membership,
runtime drift, Torch drift, historical input/baseline mismatch, stale source
hashes, invalid commit length, profile/count/command mismatch, present
run/output paths, mutation immediately before link, symlinked or replaced
output parent, and no-clobber publication. The tool must not invoke Git or run
an experiment.

`tools/arm_panel_attempt.py` is pre-execution tooling. Do not add it to
`SOURCE_PATHS` or `FINALIZER_SOURCE_PATHS`; the runtime closure remains exact.

- [ ] **Step 5: Verify and checkpoint**

Run:

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3

"$PRIMARY" tests/python/test_panel_driver.py
"$PRIMARY" tests/python/test_panel_analysis.py
```

Expected: both suites pass.

Commit on the existing comparison branch:

```text
chore(training): freeze conditioned panel comparison
```

Do not push.

## Task 4: Verify and Independently Review the Unarmed Comparison

**Files:** No production file is added by this task.

**Interfaces:**

- Consumes: Tasks 2-3 implementation checkpoints.
- Produces: reviewed evidence that the comparison is safe to arm.

- [ ] **Step 1: Run focused and aggregate gates**

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
TORCH=(/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python)

"$PRIMARY" tests/python/test_panel_analysis.py
"$PRIMARY" tests/python/test_panel_driver.py
"${TORCH[@]}" tests/python/test_experiment.py
"${TORCH[@]}" tests/python/test_training.py
make -B PYTHON="$PRIMARY" check
```

Expected:

- all nine C suites pass;
- C/Python e2e remains within five ULP;
- analyzer, driver, experiment, and training suites pass;
- only the known optional-NumPy Torch warning may remain.

- [ ] **Step 2: Audit scope**

Confirm:

- no policy/backtest model set contains either panel model;
- no generated CSV/report/model, credential, bytecode, or cache is tracked;
- the old config/attempt/outcome/evidence hashes are unchanged;
- the new run directory, attempt, and outcome are absent;
- the existing uncommitted `Makefile` and `docs/training.md` hunks remain
  outside the comparison commits.

- [ ] **Step 3: Request two independent reviews**

Require:

- code/integrity review: exact profiles, legacy behavior, ordering, shared-fit
  attribution, selected-source arming, runtime-template equality, driver
  lifecycle, finalizer provenance, scope;
- methodology review: paired units, practical threshold, axis robustness,
  seed ensembles, bootstrap algorithm, gate inequalities, and calibration-only
  interpretation.

Fix every Important finding in the owning unpublished checkpoint, rerun the
full gates, and re-review. Do not arm before both reviews approve.

## Task 5: Arm One Durable Comparison Attempt

**Files:**

- Add: `experiments/executable-h13-conditioned-panel-attempt.json`

**Interfaces:**

- Consumes: fully committed/reviewed source closure, exact config, unchanged
  input manifest/baseline, and the existing runtime identities.
- Produces: one immutable schema-1 `PanelAttempt` for
  `h13-conditioned-panel-20260724-01`.

- [ ] **Step 1: Require fresh outputs**

The following must be absent:

```text
reports/h13-conditioned-panel-20260724-01
experiments/executable-h13-conditioned-panel-outcome.json
```

Use `but show enkyuan/conditioned-panel-comparison` and record the exact
40-character reviewed tip containing both Tasks 2 and 3. GitButler must show
the analyzer, panel contract, finalizer, arming tool, one-shot driver, both
tests, and new config in that committed stack with no uncommitted hunk in any
`SOURCE_PATHS` file. That exact tip is `implementation_commit`; the earlier
Task 2 commit is invalid.

- [ ] **Step 2: Build the canonical attempt with the reviewed tool**

Set `IMPLEMENTATION_COMMIT` to the literal 40-character GitButler value from
Step 1, then run:

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3

"$PRIMARY" tools/arm_panel_attempt.py \
  experiments/executable-h13-conditioned-panel-attempt.json \
  --runtime-template experiments/executable-h13-panel-attempt.json \
  --input-manifest experiments/executable-h13-panel-inputs.json \
  --config experiments/executable-h13-conditioned-panel.example.json \
  --implementation-commit "$IMPLEMENTATION_COMMIT" \
  --run-id h13-conditioned-panel-20260724-01 \
  --run-dir reports/h13-conditioned-panel-20260724-01 \
  --outcome experiments/executable-h13-conditioned-panel-outcome.json
```

The tool must compute every current selected-source SHA and abort if the
primary Python, uv, Torch argv, or complete Torch probe differs from
`experiments/executable-h13-panel-attempt.json`. Do not bind a newly observed
runtime after drift.

Bind exactly:

```text
schema = 1
run_id = h13-conditioned-panel-20260724-01
status = armed
run_dir = reports/h13-conditioned-panel-20260724-01
implementation_commit = reviewed comparison implementation commit
input_manifest = experiments/executable-h13-panel-inputs.json
config = experiments/executable-h13-conditioned-panel.example.json
baseline_report = reports/executable-h13-calibration.json
baseline_ledger = reports/executable-h13-calibration.jsonl
expected_equivalent_runs = 207
expected_panel_fits = 30
outcome = experiments/executable-h13-conditioned-panel-outcome.json
```

Bind the current ordered selected source/finalizer trees, the byte-identical
historical primary-Python/uv/Torch identities, no-bytecode environment, exact
shared analyzer/experiment/finalizer argv arrays, and the three run-directory
output paths.

The experiment command must end with:

```text
--device cpu
--calibration-only
--calibration-predictions reports/h13-conditioned-panel-20260724-01/calibration.jsonl
--max-runs 207
```

- [ ] **Step 3: Validate the real manifest without consuming preflight**

Run:

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=reports/h13-conditioned-panel-20260724-01/.pycache \
"$PRIMARY" tools/analyze_panel.py validate-attempt \
  experiments/executable-h13-conditioned-panel-attempt.json \
  experiments/executable-h13-panel-inputs.json \
  experiments/executable-h13-conditioned-panel.example.json \
  reports/executable-h13-calibration.json \
  reports/executable-h13-calibration.jsonl \
  AAPL=data/aapl-30m.csv \
  MSFT=data/msft-30m.csv \
  SPY=data/spy-30m.csv
```

Expected canonical result:

```json
{"mode": "validate-attempt", "status": "valid"}
```

- [ ] **Step 4: Review and checkpoint**

Independently review every manifest path, hash, count, command, runtime, output
absence, and source-tree entry. Commit only the manifest on
`enkyuan/conditioned-panel-comparison-attempt`, stacked above the comparison
implementation:

```text
chore(training): arm conditioned panel comparison
```

After this commit, never modify or re-arm the attempt. Any binding change
produces one finalized failure.

## Task 6: Run Once, Finalize, and Record Evidence

**Files:**

- Add through finalizer:
  `experiments/executable-h13-conditioned-panel-outcome.json`
- Add after finalization:
  `docs/experiments/h13-conditioned-panel-20260724-01.md`

**Interfaces:**

- Consumes: the immutable armed attempt.
- Produces: one terminal outcome and concise evidence; never a policy or
  backtest.

- [ ] **Step 1: Execute exactly once**

```zsh
/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3 \
  tools/run_panel_attempt.py
```

Preflight, experiment, and analyzer each run at most once. The finalizer runs
exactly once for every catchable terminal path. Never manually invoke a stage
after a failure. The attempt is permanently consumed when the first driver
process starts, even if an uncatchable failure prevents outcome creation.
Never delete a partial run directory or outcome to create a retry.

- [ ] **Step 2: Honor terminal state**

- Exit 0 / `pass`: record gates and stop. A separate reviewed policy plan and
  explicit one-time reserved-test authorization are still required before a
  `$100` backtest.
- Exit 3 / `gate-failure`: record the negative result and permanently stop
  this attempt.
- Any integrity, signal, setup, experiment, finalizer, host-loss, or
  uncatchable failure: preserve every file, do not rerun, and request user
  direction.

- [ ] **Step 3: Write concise evidence**

The evidence document must state:

- exact command, run ID, input/config/attempt/output hashes, 207 equivalents,
  and 30 physical fits;
- conditioned and unconditioned validation/calibration MAE and relative
  improvement;
- 30-pair wins/ties/losses and stock/fold/seed summaries;
- leave-one-seed-out results, all per-block bootstrap lower percentiles, and
  their gate-controlling minima;
- per-stock return, direction, and close-MAE comparisons;
- every frozen gate and final status;
- explicit calibration-only, calibration-reused-for-development,
  no-test/no-policy/no-backtest language.

Do not copy the generated report or ledger into tracked files.

- [ ] **Step 4: Verify and checkpoint**

Re-run the focused and aggregate gates from Task 4. Scan tracked changes for
credentials and generated artifacts.

Commit only the outcome and evidence document on
`enkyuan/conditioned-panel-comparison-evidence`, stacked above the armed
attempt:

```text
docs(training): record conditioned panel comparison
```

Do not push or land.

## Final Audit

- [ ] Legacy schema-1 panel analysis remains exact.
- [ ] New analysis uses schema 2 and identifies candidate/reference explicitly.
- [ ] The exact seven-model order, 207 equivalents, and 30 fits are bound.
- [ ] Both shared models use the same inputs, code, runtime, folds, and seeds.
- [ ] Conditioning IDs follow bound AAPL/MSFT/SPY order.
- [ ] All scalers remain per-series and training-only.
- [ ] No window crosses series and no test label is read.
- [ ] Paired validation and seed-ensemble calibration math is exact.
- [ ] Bootstrap rows are paired across models/stocks and use fixed parameters.
- [ ] The attempt and every output path are immutable and no-clobber.
- [ ] A gate failure cannot trigger policy selection or a `$100` backtest.
- [ ] Generated artifacts and credentials remain untracked.
- [ ] Every checkpoint is local, enkyuan-authored, and unpushed.
