# Completed Residual Evidence Replay Implementation Plan

> **For agentic workers:** Implement test-first and keep the strict execution
> leases unchanged. Stop on any semantic, data, runtime, or artifact mismatch.

**Goal:** Let the interval-sensitivity analyzer read one completed,
hash-bound calibration after unrelated source files evolve, without claiming
that current code is the historical execution code.

**Architecture:** Preserve `authenticate_context_attempt()` and
`authenticate_residual_attempt()` as exact live execution leases. Add a
separate calibration-evidence path that validates each published historical
source tree against its recorded commit, rederives the same data and phase
semantics with a frozen current closure, validates recorded runtime metadata,
and binds that current interpretation closure to the sensitivity
implementation commit. The report records both provenance layers and exposes
no refit, forward, backtest, or trading authority.

**Tech Stack:** Python 3.12 standard library, existing frozen-input and
source-tree primitives, direct Python test scripts, GitButler.

## Why This Is Required

The residual attempt pins commit `0bc33956ddbff9f706d1341b77f01e71a0b07496`
and its original source tree. The live GitButler workspace now differs in
`tools/backtest.py`, `tools/float32.py`, and `tools/universe_scaling.py`.
The strict lease correctly rejects that drift. A relocated checkout is not a
valid substitute because both attempts bind the authoritative repository root.

The sensitivity report is a new interpretation of completed evidence, not a
continuation of the old execution. It therefore needs two explicit claims:

1. the historical artifacts still match their historical commits and trees;
2. every current reader, parser, derivation, gate, and resampling helper matches
   the current analysis commit.

## Non-Negotiable Constraints

- Do not change, parameterize, or add a bypass flag to either public strict
  execution lease.
- Do not rewrite attempts, roots, receipts, hashes, source trees, or terminal
  outcomes.
- Do not fall back to evidence replay after a strict-lease failure. The caller
  must select the sensitivity-only path explicitly.
- Allow only source-tree bytes to differ. Reject changed phase definitions,
  universe order, grids, data/config/benchmark bindings, runtimes, Torch
  identity, predictions, evaluations, receipts, truth authorization, or
  terminal outcome.
- Bind the full current interpretation closure to the current implementation
  commit, not only the analyzer entry point.
- Expose calibration evidence only. Permit only hash-bound historical
  calibration OHLC reads needed to derive residual returns and causal SPY
  regimes; never serialize raw prices or returns into the report. Do not read
  fold-1 truth, forward inputs, Torch model state, policies, trades, or P&L,
  and do not invoke Massive or any other network fetch.
- Keep every execution, forward, backtest, and trading lock false.
- Preserve unrelated working-tree changes and do not push.

---

### Task 1: Lock the Failure and the Calibration-Only Boundary

**Files:**

- Modify: `tests/python/test_spy_residual_shrinkage.py`

- [ ] **Step 1: Add routing tests**

Assert shrinkage and alignment use only the strict residual authenticator,
while sensitivity uses only the completed-calibration authenticator. An
authentication failure must never select another path.

- [ ] **Step 2: Add capability-boundary tests**

Require a new immutable `CompletedCalibrationEvidence` value to expose only:

- the exact calibration `ContextPhase` and `ResidualPhaseInput`;
- ordered calibration truth, transformer-mean predictions, and causal regimes;
- historical scaling-runner/scaling-finalizer/context/residual and
  current-interpretation provenance;
- a named `verify()` method.

Assert it is not a `ContextLease` or `ResidualLease`, is not callable, and
contains no frozen snapshots, arbitrary readers, training inputs, forward
inputs, runtime objects, or benchmark handles.

- [ ] **Step 3: Add top-level tamper tests**

At the analyzer entry point, mutate the residual attempt, outcome, every
fold-1/calibration fit/prediction/receipt/access/evaluation artifact, run
directory membership, each historical commit/tree, and each current source
binding one at a time. Every case must fail before output-directory creation.

- [ ] **Step 4: Add source-closure tests**

Define a sensitivity-only source list. Audit its recursive repository imports,
require exact equality with the bound files, and reject new unbound imports.
Some validation modules share parsers with fetch and training entry points, so
bind those imported bytes too; assert the sensitivity execution path never
invokes a network fetch, model training, Torch, backtest, trading, or
forward-input execution.

- [ ] **Step 5: Run the red test**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_shrinkage.py
```

Expected: nonzero exit because the calibration-evidence type and authenticator
do not exist.

### Task 2: Share Data-Only Derivation Without Weakening Execution

**Files:**

- Modify: `tools/spy_residual_controller.py`
- Modify: `tools/analyze_spy_residual_shrinkage.py`
- Test: `tests/python/test_spy_residual_shrinkage.py`

- [ ] **Step 1: Extract snapshot-based row derivation**

Extract the data-only body below `_collect_inputs()`. It accepts exact frozen
config/manifest/calendar/stock/SPY snapshots and a verifier, and returns only
phase rows. It neither imports nor constructs a strict lease.

- [ ] **Step 2: Keep the strict wrapper exact**

Keep `_collect_inputs()` accepting and checking the real `ContextLease`, then
delegate to the data-only helper. Place the shared phase implementations in
one pure `residual_phase_evidence.py` helper, bind that helper in the strict
analysis closure, and keep the strict `_phase_truth()` and
`_phase_market_regimes()` wrappers accepting only a real `ResidualLease`.

- [ ] **Step 3: Run controller and analyzer tests**

Run the controller suite and the Task 1 command. Existing strict tests must
remain green.

### Task 3: Authenticate the Completed Historical Closure

**Files:**

- Create: `tools/residual_calibration_evidence.py`
- Test: `tests/python/test_spy_residual_shrinkage.py`

- [ ] **Step 1: Validate terminal evidence first**

Reuse the completed-run validators for the residual and context runs. Require
both phases' five artifacts, truth-access receipts, evaluations, terminal
outcomes, and directory topology to authenticate before calibration truth is
constructed.

- [ ] **Step 2: Validate archived producer provenance**

Validate the scaling runner/finalizer, context, and residual commit/tree pairs
with `_validate_commit()`. Recorded runtime and Torch identities remain
historical metadata; never resolve a live Torch executable or package.

- [ ] **Step 3: Reconstruct source semantics from frozen data**

Freeze the scaling attempt/failure/fits/predictions/summary, fetch report,
manifests, calendar, context config, all 55 stock CSVs, residual config, and
SPY report/CSV. Rederive the context master/phases and residual phase bindings;
require exact equality with the published attempts.

- [ ] **Step 4: Run the analyzer test**

Expected: remaining failures concern only immutable evidence assembly and
sensitivity routing.

### Task 4: Assemble Immutable Calibration Evidence

**Files:**

- Modify: `tools/residual_calibration_evidence.py`
- Test: `tests/python/test_spy_residual_shrinkage.py`

- [ ] **Step 1: Select calibration structurally**

Require exactly one phase named `calibration`; reject missing, duplicate,
reordered, or fold-1 state. Validate its phase binding and select only the
authenticated panel-transformer five-seed mean.

- [ ] **Step 2: Derive only calibration labels**

Using the snapshot helpers from Task 2, derive causal SPY regimes, reverify,
then reconstruct calibration residual truth. Normalize it through the existing
truth-grid validator and require identical series order and vector lengths.
Never construct fold-1 truth.

- [ ] **Step 3: Yield the narrow evidence value**

Yield immutable series plus historical/current provenance and `verify()`.
Reverify all files, inodes, and directory memberships before yield, during
publication, and on context-manager exit.

- [ ] **Step 4: Run the focused test**

Expected: evidence and tamper tests pass before analyzer routing.

### Task 5: Bind Dual Provenance and Sensitivity-Only Use

**Files:**

- Modify: `tools/analyze_spy_residual_shrinkage.py`
- Test: `tests/python/test_spy_residual_shrinkage.py`

- [ ] **Step 1: Bind the sensitivity-only current closure**

Define `SENSITIVITY_SOURCE_PATHS` as the exact recursive repository import
closure. Freeze it and validate those bytes against the supplied current
implementation commit. The only strict-closure addition is the pure shared
phase helper from Task 2; strict behavior remains unchanged.

- [ ] **Step 2: Select the evidence path explicitly**

Choose the analyzer-private `_authenticate_completed_calibration()` only when
sensitivity is true. It yields a publication session containing the narrow
evidence value and separately held report metadata. Shrinkage and alignment
continue through `authenticate_residual_attempt()`. Never catch authentication
errors to switch paths.

- [ ] **Step 3: Record both provenances**

Add this integrity shape to the sensitivity report:

```json
{
  "current_analysis": {
    "implementation_commit": "<current commit>",
    "source_tree_sha256": "<digest of analysis_sources>"
  },
  "historical_execution": {
    "scaling_runner": {
      "implementation_commit": "<scaling-runner commit>",
      "source_tree_sha256": "<scaling-runner tree digest>"
    },
    "scaling_finalizer": {
      "implementation_commit": "<scaling-finalizer commit>",
      "source_tree_sha256": "<scaling-finalizer tree digest>"
    },
    "context": {
      "implementation_commit": "<context commit>",
      "source_tree_sha256": "<context tree digest>"
    },
    "residual": {
      "implementation_commit": "<residual commit>",
      "source_tree_sha256": "<residual tree digest>"
    }
  }
}
```

Keep the existing planning-only decision, calibration-only truth declaration,
and all-false locks.

- [ ] **Step 4: Reject provenance drift**

Add exact tests for the integrity object. Reject a changed historical commit,
tree digest, current implementation commit, source list, or source-tree
digest.

- [ ] **Step 5: Run the sensitivity and adjacent suites**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_shrinkage.py

PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_gate.py

PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_forward_contract.py
```

Expected: all pass.

### Task 6: Verify, Checkpoint, and Run the Report

**Files:**

- Create: `tools/residual_calibration_evidence.py`
- Create: `tools/residual_phase_evidence.py`
- Create: `tests/python/test_spy_residual_evidence.py`
- Modify: `tools/context_cross_section.py`
- Modify: `tools/analyze_context_cross_section.py`
- Modify: `tools/relative_context_contract.py`
- Modify: `tools/spy_residual_controller.py`
- Modify: `tools/analyze_spy_residual_shrinkage.py`
- Modify: `tests/python/test_spy_residual_shrinkage.py`
- Generate ignored:
  `reports/h13-spy-residual-20260725-01-sensitivity/sensitivity.json`

- [ ] **Step 1: Run focused and aggregate checks**

Run the Task 5 suites, the context/residual armer suites, all adjacent forward
suites, the complete replay/tamper integration:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -B tests/python/test_spy_residual_evidence.py
```

Then run:

```bash
make -B \
  PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  check
```

- [ ] **Step 2: Review the source diff**

Verify:

- strict execution leases have identical public behavior;
- only sensitivity imports the completed-evidence path;
- current analysis sources cover the complete transitive interpretation path;
- no fallback flag, Torch import, forward input, Massive call, price, return,
  trade, policy, or P&L field was introduced;
- unrelated dirty files remain uncommitted.

- [ ] **Step 3: Amend the unpublished sensitivity checkpoint**

Amend the coherent repair into `feat/forward-interval-sensitivity`, preserve
`enkyuan <yuan.enkng@gmail.com>` as author and committer, and verify its ED25519
signature. Do not push.

- [ ] **Step 4: Run the exact signed report**

Use the new exact 40-character implementation commit:

```bash
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -I -S -B tools/analyze_spy_residual_shrinkage.py \
  experiments/h13-spy-residual-20260725-01-attempt.json \
  --implementation-commit <signed-commit> \
  --sensitivity
```

Expected:

```json
{"comparisons":["unchanged-five-seed-mean","zero"],"mode":"sensitivity","status":"analyzed"}
```

- [ ] **Step 5: Audit the ignored report**

Verify both paired-MSE comparisons, 60 target sessions, 20-session blocks,
10,000 replicates, seed `20260725`, exact source/config bindings, both
provenance layers (one current interpreter and four historical producers),
calibration-only truth, planning-only decision, and all-false locks. Confirm
that no price, return, policy, trade, P&L, or new `$100` backtest field exists.

## Stop Conditions

Stop without publishing if:

- any historical producer commit/tree pair does not match;
- current derivation changes any phase, grid, universe, data, runtime, Torch,
  prediction, evaluation, receipt, truth-authorization, or outcome field;
- the full current source closure does not match the supplied current commit;
- the canonical report directory already exists;
- implementation requires a strict-lease bypass or fallback;
- any forward label, model refit, Torch execution, market fetch, or trading
  path would be opened.

The completed-evidence replay authenticates a historical calibration for
development planning. It does not make the sensitivity estimate forward
evidence, authorize a candidate/protocol change, or open the `$100` account.
