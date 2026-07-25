# Universe Scaling Failure Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the terminal universe-scaling failure into bounded development
diagnostics that can identify whether the model lacks stock-selection signal,
temporal context, or optimization quality.

**Architecture:** Authenticate the existing terminal failure without opening a
forward-evaluation path. Analyze complete execution groups with pure,
cross-section-preserving statistics. Then compare frozen context lengths and
simple controls on identical cells and equal budgets. A later untouched period
may be armed only after this development family is frozen.

This plan narrows the conditional work in
`2026-07-25-post-scaling-improvements.md` after run
`h13-universe-scaling-20260724-01` failed. It supersedes that plan's
PASS-dependent forward-first implementation order.

**Tech Stack:** Python 3.12, PyTorch, canonical JSON, existing universe-scaling
statistics, `unittest.mock`, Make, GitButler.

## Global Constraints

- Treat `gate-failure` as terminal. Do not reinterpret it as a partial pass.
- Do not run forward refits, join protected outcomes, or execute a portfolio.
- Keep the current `$100` policy in cash; at the project's zero-yield cash
  assumption, its balance remains exactly `$100`.
- Do not fetch more Massive symbols. The largest panel improved over its
  11-stock form but still lost to zero-return, local-Transformer, and MLP
  controls.
- Task 1 reads only the canonical failure outcome and summary. Tasks 2-4 may
  read attempt-bound CSV development prefixes through an inclusive,
  timestamp-exact bounded reader. Never decode a row past the phase cutoff or
  inspect reserved labels.
- Compare models on the exact same `(series, as_of, entry, target)` cells.
- Fit every scaler on the retained training prefix only.
- Resample complete trading-day stock vectors so contemporaneous and serial
  dependence stay coupled.
- Keep generated data, reports, model artifacts, credentials, and caches
  untracked.
- Export the modern runtime once before running non-Torch commands:

  ```sh
  export PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
  ```

  The macOS Python 3.9 runtime is unsupported.

## Evidence Lock

- Outcome: terminal `gate-failure`, stage `analysis`, exit `3`.
- Summary SHA-256:
  `22607ed864f0074bbc2e1fa7447d48ffce23003beb43a0d70a7e0060e6f57b2f`.
- Unseen 44-stock panel return MAE: `0.023390185790695007`.
- Unseen local, MLP, and zero-return MAE: `0.02242239935259441`,
  `0.022845733093927702`, and `0.022496063286870118`.
- Unseen direction: `49.3043%`, below the `56.4420%` majority-sign control.
- Breadth relative to the 11-stock panel improved on unseen stocks, but the
  panel still failed the absolute controls. The next diagnostic must therefore
  separate common-market timing from stock selection before adding symbols.

---

## Task 1: Authenticate the terminal failure

**Files:**

- Modify: `tools/universe_forward_contract.py`
- Modify: `tests/python/test_universe_forward_contract.py`

- [ ] **Step 1: Write the failing contract test**

```python
def test_reads_only_exact_terminal_failure(directory: Path) -> None:
    outcome, summary = write_failure_fixture(directory)
    failure = read_scaling_failure(outcome, root=directory.resolve())
    assert failure.summary.sha256 == sha256(summary)
    assert "unseen_mae_improvement" in failure.failed_gates
    raises(ValueError, read_scaling_failure, forged_pass(outcome),
           root=directory.resolve())
```

Also prove rejection of a changed outcome hash, a nonzero status/exit mismatch,
an unbound summary, a summary with `gates.all_pass = true`, a path outside the
repository, and any trading lock set to true.

- [ ] **Step 2: Run the test to verify the missing module**

Run:

```sh
"$PYTHON" tests/python/test_universe_forward_contract.py
```

Expected: import failure for `read_scaling_failure`.

- [ ] **Step 3: Generalize the existing terminal parser**

Change the private outcome parser to accept one exact expected terminal state:

```python
def _outcome(
    value: object, expected: FileBinding, root: Path,
    terminal: tuple[str, int] = ("pass", 0),
) -> tuple[
    str, FileBinding, FileBinding, FileBinding, FileBinding, str, FileBinding,
]:
```

Keep `read_passing_scaling_outcome()` on the default. The failure reader passes
`("gate-failure", 3)`. This preserves one path, hash, timestamp, output,
self-record, and finalizer-integrity implementation.

Apply the same pattern to `_summary()`:

```python
def _summary(
    value: object,
    bindings: tuple[FileBinding, FileBinding, FileBinding],
    status: str = "pass",
) -> tuple[str, ...]:
```

Validate every named gate as a strict Boolean. Return the sorted failed names;
the passing caller requires that tuple to be empty, while the failure caller
requires it to be nonempty. Keep the input bindings and no-trading locks in
this one parser.

- [ ] **Step 4: Implement the smallest immutable result**

```python
@dataclass(frozen=True, slots=True)
class ScalingFailure:
    run_id: str
    outcome: FileBinding
    summary: FileBinding
    failed_gates: tuple[str, ...]


def read_scaling_failure(
    outcome: FileBinding, *, root: Path,
) -> ScalingFailure:
    """Authenticate one exact terminal development failure."""
```

Reuse `_object`, `FileBinding`, `_regular_inputs`, `freeze_inputs`,
`verify_frozen`, and the existing field sets in
`tools.universe_forward_contract`.
Require `status == "gate-failure"`, `stage == "analysis"`, `exit == 3`,
`gates.all_pass is False`, at least one failed named gate, and unchanged
no-trading locks. Return names only; do not copy metric payloads into a second
schema.

- [ ] **Step 5: Verify the focused and aggregate gates**

Run:

```sh
"$PYTHON" tests/python/test_universe_forward_contract.py
make -B PYTHON="$PYTHON" check
```

Expected: both pass.

- [ ] **Step 6: Commit the checkpoint**

```sh
but diff
```

Use the selected-change fast path to create
`enkyuan/scaling-failure-contract` with message
`feat(training): authenticate scaling failure`. Pass only the contract and test
hunk IDs returned by `but diff`.

---

## Task 2: Measure stock-selection signal separately from market direction

**Files:**

- Create: `tools/universe_cross_section.py`
- Test: `tests/python/test_universe_cross_section.py`
- Modify: `Makefile`

- [ ] **Step 1: Write failing mathematical tests**

Use two complete dates with three stocks for literal hand-computed statistics,
plus a separate 20-date fixture for bootstrap coverage. Prove:

- subtracting each execution group's mean from truth and prediction;
- `R²_XS = 1` for perfect centered ranks and `0` for the zero centered
  prediction;
- paired centered absolute loss is negative when the candidate helps;
- constant truth gives an unavailable `R²_XS`;
- incomplete or duplicate group membership fails;
- the date-block bootstrap receives whole stock vectors; and
- effective breadth is reported with raw breadth, exclusions, and reason.

Run:

```sh
"$PYTHON" tests/python/test_universe_cross_section.py
```

Expected: failure because the module does not exist.

- [ ] **Step 2: Implement one validated cell type**

```python
@dataclass(frozen=True, slots=True)
class CrossSectionCell:
    day: str
    group: str
    series: str
    actual: float
    predicted: float
```

Validate nonempty identifiers, finite numbers, unique `(group, series)` pairs,
and the exact same ordered series set for every group.

- [ ] **Step 3: Implement the diagnostics**

```python
@dataclass(frozen=True, slots=True)
class CrossSectionResult:
    r2: float | None
    paired_mean: float
    interval: tuple[float, float]
    eligible_spearman_groups: int
    excluded_spearman_groups: int
    mean_spearman: float | None
    effective_breadth: EffectiveCount
```

For group \(g\), center both vectors:

\[
\tilde y_{i,g}=y_{i,g}-\bar y_g,\qquad
\tilde{\hat y}_{i,g}=\hat y_{i,g}-\bar{\hat y}_g.
\]

Compute:

\[
R^2_{\mathrm{XS}}
=1-\frac{\sum_{g,i}(\tilde y_{i,g}-\tilde{\hat y}_{i,g})^2}
{\sum_{g,i}\tilde y_{i,g}^2}
\]

and

\[
\Delta^{\mathrm{XS}}_{i,g}
=|\tilde y_{i,g}-\tilde{\hat y}_{i,g}|-|\tilde y_{i,g}|.
\]

Reuse `circular_block_interval()` for the paired loss and
`effective_count()` for variance-equivalent breadth. Implement average ranks
locally in one short helper; exclude a group from Spearman when either centered
vector is constant. Do not let Spearman availability alter the loss gate.
Pass seed `20_260_725` explicitly to each bootstrap call.

Report effective breadth as

\[
M_{\mathrm{eff}}
=\frac{M\,\operatorname{tr}(\Sigma)}
{\mathbf 1^\top\Sigma\mathbf 1},
\]

where \(\Sigma\) is the covariance of aligned daily loss differentials. This is
a variance-equivalent diagnostic, not a literal independent sample count.

- [ ] **Step 4: Freeze the development-only gate**

The diagnostic passes only when:

- the \(R^2_{\mathrm{XS}}\) denominator is positive;
- `R²_XS > 0`;
- the maximum 97.5% upper bound over block lengths `5`, `10`, and `20` is
  below zero; and
- every group was complete before outcome access.

This gate explains signal type. It cannot authorize trading or universe
expansion because it uses already inspected development phases.

- [ ] **Step 5: Run focused and aggregate checks**

```sh
"$PYTHON" tests/python/test_universe_cross_section.py
"$PYTHON" tests/python/test_universe_analysis.py
make -B PYTHON="$PYTHON" check
```

Expected: all pass.

- [ ] **Step 6: Commit the checkpoint**

```sh
but diff
```

Use the selected-change fast path to create
`enkyuan/cross-sectional-diagnostic` with message
`feat(training): measure stock-selection signal`. Pass only the module, test,
and Makefile hunk IDs returned by `but diff`.

---

## Task 3: Freeze the context-length comparison before fitting

**Files:**

- Create: `tools/context_diagnostic_contract.py`
- Create: `experiments/executable-h13-context.example.json`
- Test: `tests/python/test_context_diagnostic_contract.py`
- Modify: `Makefile`

- [ ] **Step 1: Write the failing family test**

Require exactly:

```python
HISTORY_LENGTHS = (17, 34, 68)
MODELS = ("global_ridge", "global_mlp", "panel_transformer")
SEEDS = (7, 19, 31, 43, 61)
```

Prove that reordering, omitting, or adding an axis fails; destinations must be
absent; inputs must be regular files with bound hashes; and all candidates must
share one target-cell digest per phase.

- [ ] **Step 2: Implement canonical parsing and arming**

The contract binds:

- the exact training, validation, and later evaluation cutoffs;
- the shared target-cell digest;
- target `executable-return-v1` and horizon `13`;
- current feature recipe and per-stock train-only scalers;
- identical neural checkpoint grid, optimizer, and update budget within each
  model comparison;
- deterministic ridge penalty and rows;
- five neural seeds;
- phase-scoped fit, prediction-ledger, and immutable-receipt paths;
- output paths and no-bytecode environment; and
- the complete candidate family before any target-phase labels are read.

Do not include patching, RevIN, ticker embeddings, a probabilistic head, or
extra model sizes.

- [ ] **Step 3: Verify the immutable family**

Run:

```sh
"$PYTHON" tests/python/test_context_diagnostic_contract.py
make -B PYTHON="$PYTHON" check
```

Expected: all pass and every destination remains absent.

- [ ] **Step 4: Commit the checkpoint**

```sh
but diff
```

Use the selected-change fast path to create
`enkyuan/context-diagnostic-contract` with message
`feat(training): freeze context comparison`. Pass only the module, config,
test, and Makefile hunk IDs returned by `but diff`.

---

## Task 4: Execute equal-budget context diagnostics

**Files:**

- Create: `tools/run_context_diagnostic.py`
- Create: `tools/finalize_context_diagnostic.py`
- Test: `tests/python/test_context_diagnostic_driver.py`
- Modify: `Makefile`

- [ ] **Step 1: Write driver failure-path tests**

Prove one-shot execution, atomic terminal outcome publication, exact exit-code
mapping, cleanup after interruption, no second run, and no prediction before
all same-phase candidate fits are complete. Also prove the target-phase label
accessor stays disabled until an immutable receipt binds the complete
same-phase prediction ledger.

- [ ] **Step 2: Reuse existing training primitives**

Use `feature_lookback()` for each history length, the existing chronological
sample rows, train-only scaler fit, ridge/MLP/panel fitters, prediction-ledger
validators, stock-macro metrics, and complete-day bootstrap. Do not fork their
math or serialization.

For each neural model and seed, select the checkpoint using length `17` on the
prior phase only. Reinitialize all three histories, give them that exact update
budget, and publish predictions on the shared target cells before reading
target-phase outcomes. Ridge fits once per phase with its frozen penalty.

After fold-1 prediction, atomically publish a receipt that binds the fit
closure, prediction-ledger SHA-256, target-cell digest, candidate family, and
source tree. Only a successfully revalidated receipt may enable the fold-1
label accessor used to select calibration's checkpoint. Apply the same receipt
boundary to calibration before any calibration label access. An interruption
after label access is terminal; never delete a receipt or rerun that phase.

- [ ] **Step 3: Evaluate without post-hoc rescue**

Primary comparison: paired stock-macro return MAE. A longer history advances
only when its maximum 97.5% bootstrap upper bound against history `17` is below
zero in both development phases. If both pass, select `34`. Direction and close
MAE are descriptive and cannot overturn the primary result.

More epochs or width are allowed only when a pre-frozen training diagnostic
shows validation loss still improving at the final checkpoint without a
widening train-validation gap.

- [ ] **Step 4: Run all gates**

```sh
"$PYTHON" tests/python/test_context_diagnostic_driver.py
uv run --offline --with torch python \
  tests/python/test_context_diagnostic_driver.py
make -B PYTHON="$PYTHON" check
```

Expected: focused and aggregate tests pass without reading reserved labels.

- [ ] **Step 5: Commit the checkpoint**

```sh
but diff
```

Use the selected-change fast path to create
`enkyuan/context-diagnostic-runner` with message
`feat(training): compare temporal context`. Pass only the runner, finalizer,
test, and Makefile hunk IDs returned by `but diff`.

---

## Task 5: Preserve a genuinely later decision boundary

**Files:**

- Modify: `docs/training.md`
- Create: `docs/experiments/h13-context-diagnostic-20260725-01.md`

- [ ] **Step 1: Record only authenticated development findings**

Report every frozen candidate, exact cells, hashes, update counts, seeds,
point estimates, block intervals, cross-sectional diagnostics, and failures.
Label them `development-diagnostic-not-forward-clean`.

- [ ] **Step 2: Decide the next model change mechanically**

- Positive cross-sectional signal plus a longer-context win: test one
  predeclared temporal architecture on a later untouched block.
- Positive cross-sectional signal without a context win: test training
  diagnostics before model capacity.
- No cross-sectional signal: add market/sector/context features or reformulate
  the target before changing the Transformer.
- Transformer loss no better than ridge/MLP/zero: keep the simple control and
  do not promote a Transformer policy.

- [ ] **Step 3: Arm later evidence only after the family is frozen**

Use data whose labels were unavailable throughout Tasks 1-4. Bind its exact
universe, dates, prediction schedule, costs, and candidate family before
materializing labels. Only a later PASS may unlock the existing forward-ledger
and `$100` portfolio path.

- [ ] **Step 4: Verify and commit documentation**

```sh
make -B PYTHON="$PYTHON" check
but diff
```

Use the selected-change fast path to create
`enkyuan/context-diagnostic-evidence` with message
`docs(training): record context diagnostics`. Pass only the training-doc hunk
and evidence-file IDs returned by `but diff`.

Expected: documentation matches canonical artifacts and the aggregate gate
passes.
