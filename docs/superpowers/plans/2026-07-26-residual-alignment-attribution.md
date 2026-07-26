# Residual Alignment Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or execute each checked task inline with an
> independent review gate.

**Goal:** Explain the failed global residual scale with fold-1-only
stock and causal market-direction sufficient statistics.

**Architecture:** Extend the existing authenticated shrinkage analyzer with
one explicit alignment mode. Reuse its completed-run authentication, ensemble,
truth reconstruction, exclusive publication, and execution locks. Partition
the original pooled estimator without fitting, selecting, or backtesting a new
model.

**Tech Stack:** Python 3.12 standard library, existing compose-mini
authentication and block-free diagnostic primitives, GitButler.

## Global Constraints

- Keep the existing shrinkage CLI and reports unchanged.
- Read residual truth from `fold-1` only; never regenerate calibration truth.
- Use the arithmetic mean of the existing five `panel_transformer` seeds.
- Persist aggregate sufficient statistics only—never prices, predictions,
  truth rows, or per-row regime labels.
- Keep every absolute-price, forward-clean, universe, trading, and backtest
  lock false.
- Keep generated reports ignored, mode `0600`, single-link, and uncommitted.
- Do not push or land the checkpoint.

---

## Frozen Diagnostic

For residual truth \(z\), ensemble prediction \(p\), and cell \(g\), report

\[
A_g=\sum_{(i,t)\in g}z_{i,t}p_{i,t},\qquad
B_g=\sum_{(i,t)\in g}p_{i,t}^2,\qquad
C_g=\sum_{(i,t)\in g}z_{i,t}^2.
\]

The unconstrained and existing restricted slopes are

\[
\lambda_g^{*}=
\begin{cases}
\text{unavailable},&B_g=0,\\
A_g/B_g,&B_g>0,
\end{cases}
\qquad
\lambda_g=\operatorname{clip}_{[0,1]}(\lambda_g^{*}).
\]

The market state uses exactly the existing 17 completed input bars, whose
indices are `as_of - 16` through `as_of`:

\[
m_t=\log
\left(
\frac{C^{SPY}_{as\_of}}
     {C^{SPY}_{as\_of-16}}
\right).
\]

Label the cell `negative` when \(m_t<0\), otherwise `nonnegative`. This is a
causal partition with a fixed zero threshold; it is not a learned regime
model.

Require the global count and all three sufficient statistics to reconcile
with the sums across:

1. stock;
2. market direction; and
3. stock × market direction.

Floating regrouping uses a strict finite tolerance; it must never change a
reported value to force equality. Every cell remains descriptive and
selection-ineligible, including cells with a positive slope.

## Literature Basis

- Campbell and Thompson motivate predeclared nonnegative restrictions for
  weak return forecasts:
  <https://academic.oup.com/rfs/article-abstract/21/4/1509/1567518>.
- Clark and West show that the nested zero-forecast comparison is driven by
  the same \(zp\) alignment term:
  <https://www.sciencedirect.com/science/article/pii/S0304407606000960>.
- Gu, Kelly, and Xiu retain chronological validation, seed ensembles, and the
  zero-return out-of-sample benchmark:
  <https://academic.oup.com/rfs/article/33/5/2223/5758276>.
- Giacomini and White require regime instruments to be observable at forecast
  time:
  <https://doi.org/10.1111/j.1468-0262.2006.00718.x>.
- White's Reality Check prohibits treating the best post-hoc stock or regime
  cell as a prespecified winner:
  <https://doi.org/10.1111/1468-0262.00152>.

The first implementation intentionally omits volatility buckets, quantile
searches, optimized seed weights, confidence claims, sign reversal, and
per-cell deployment. Those additions change the candidate family and require
a separate frozen attempt.

### Task 1: Pure Alignment and Causal Direction Math

**Files:**

- Modify: `tools/analyze_spy_residual_shrinkage.py`
- Test: `tests/python/test_spy_residual_shrinkage.py`

**Interfaces:**

- Produces:
  `_fit_pairs(pairs: Sequence[tuple[float, float]]) -> ShrinkageFit`
- Produces:
  `_market_regimes(bars: Sequence[float], as_of: Sequence[int]) -> dict[int, str]`
- Produces:
  `alignment_diagnostic(truth, predictions, regimes) -> dict[str, object]`

- [ ] **Step 1: Add failing tests for partitioning and causal direction**

Test a panel containing positive and negative stock alignment. Assert that
global \(A,B,C,n\) reconcile across stock, regime, and joint partitions.
Assert that 17 bars use endpoints `0` and `16`, that changing a later,
unread bar cannot be supplied, and that nonpositive, nonfinite, short, or
misaligned prefixes fail.

- [ ] **Step 2: Run the focused test and confirm the missing interface**

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -B \
  tests/python/test_spy_residual_shrinkage.py
```

Expected: failure because the alignment interfaces do not exist.

- [ ] **Step 3: Extract the existing pair fit and add the partition**

`fit_zero_anchored_scale()` must delegate its arithmetic to `_fit_pairs()`.
`alignment_diagnostic()` must call `_pairs()` once, retain its validated
ordered values, partition them without deduplicating timestamps, and report:

```python
{
    "global": segment,
    "by_stock": {series: segment},
    "by_market_regime": {
        "negative": segment,
        "nonnegative": segment,
    },
    "by_stock_and_market_regime": {
        series: {
            "negative": segment,
            "nonnegative": segment,
        },
    },
}
```

Each segment contains `observation_count`, `numerator`, `prediction_square_sum`,
`truth_square_sum`, `unclipped_scale`, `scale`, and
`selection_eligible: false`. Empty joint cells remain present with zero sums
and unavailable constrained and unconstrained slopes; no bucket is silently
dropped.

- [ ] **Step 4: Run the focused test**

Expected: `SPY residual shrinkage tests passed`.

### Task 2: Authenticated Fold-1 Alignment Mode

**Files:**

- Modify: `tools/analyze_spy_residual_shrinkage.py`
- Test: `tests/python/test_spy_residual_shrinkage.py`

**Interfaces:**

- Produces:
  `_phase_market_regimes(state: _PhaseRows, lease: ResidualLease) ->`
  `Mapping[str, tuple[str, ...]]`
- Produces:
  `_analyze_alignment(...) -> Mapping[str, object]`
- Extends:
  `analyze_residual_shrinkage(..., alignment: bool = False)`
- Extends CLI with: `--alignment`

- [ ] **Step 1: Add failing ordering and report-boundary tests**

Patch the market-state reader, truth reader, and exclusive publisher. Assert
the order is causal state → fold-1 truth → publication and that calibration
truth is never requested. Assert the report binds the exact analyzer commit,
uses `development-post-hoc-not-forward-clean`, exposes no row data, and keeps
all execution locks false.

- [ ] **Step 2: Reuse the existing authenticated closure**

After `_completed_run()` and `_collect_inputs()` verify the completed run,
alignment mode must pass only `states[0]` and `phases[0]` to
`_analyze_alignment()`. `_phase_market_regimes()` reads the authenticated SPY
prefix only through the last fold-1 `as_of`, derives every label, and verifies
the lease before and after that read.

- [ ] **Step 3: Publish one exclusive report**

Write only:

```text
reports/h13-spy-residual-20260725-01-alignment/alignment.json
```

Reuse `_create_output_directory()`, `_publish()`, and
`_validate_published()`. Freeze and re-read the same inode, verify the live
lease and directory topology, and return success only while the report is the
directory's sole member.

- [ ] **Step 4: Preserve the default CLI**

Without `--alignment`, behavior and stdout remain unchanged. With the flag,
stdout contains only mode, status, and the global unclipped scale.

- [ ] **Step 5: Run focused and related tests**

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -B \
  tests/python/test_spy_residual_shrinkage.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -B \
  tests/python/test_finalize_spy_residual.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -B \
  tests/python/test_run_spy_residual.py
```

Expected: all tests pass.

### Task 3: Verify, Checkpoint, and Execute Once

**Files:**

- Create: `docs/superpowers/plans/2026-07-26-residual-alignment-attribution.md`
- Verify: `tools/analyze_spy_residual_shrinkage.py`
- Verify: `tests/python/test_spy_residual_shrinkage.py`

- [ ] **Step 1: Run the aggregate gate**

```sh
make -B PYTHON="$PYTHON" check
```

Expected: every C and Python suite passes.

- [ ] **Step 2: Create one signed local GitButler checkpoint**

Commit only the plan, analyzer, and focused test on
`feat/residual-alignment`, stacked directly above
`enkyuan/residual-shrinkage`:

```text
feat(training): attribute residual alignment
```

Do not push or land.

- [ ] **Step 3: Verify identity, signature, and ancestry**

Require exact author and committer
`enkyuan <yuan.enkng@gmail.com>`, the expected ED25519 fingerprint, and the
shrinkage checkpoint as direct parent.

- [ ] **Step 4: Execute the committed analyzer exactly once**

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -S -B \
  tools/analyze_spy_residual_shrinkage.py \
  experiments/h13-spy-residual-20260725-01-attempt.json \
  --implementation-commit "$COMMIT" \
  --alignment
```

Expected: exit zero and one ignored, mode-`0600`, single-link report.

- [ ] **Step 5: Select no model from the decomposition**

Interpret the pattern only as a design diagnosis:

- broad negative alignment favors changing target/features or abandoning this
  residual family;
- mixed stock alignment may motivate one separately preregistered
  series-conditioned hypothesis;
- mixed causal direction alignment may motivate one separately preregistered
  soft-regime hypothesis.

No observed cell may be inverted, excluded, scaled, traded, or backtested.

## Stop Condition

This diagnostic ends when the report is authenticated and interpreted. It
does not unlock a residual holdout, absolute-return reconstruction, or the
`$100` backtest. A later backtest requires a separately frozen model to beat
zero residual on genuinely untouched data under the existing policy and
execution-cost contract.
