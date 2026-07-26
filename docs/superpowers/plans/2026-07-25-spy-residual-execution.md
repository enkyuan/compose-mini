# SPY-Residual Execution Implementation Plan

> **For agentic workers:** Execute tasks in order. An `armed` attempt is valid
> only after Tasks 1–6 are complete, checked, and committed.

**Goal:** Run one authenticated, development-only comparison of stock-minus-SPY
residual forecasts while preserving chronology, source provenance, and every
lock in the frozen protocol.

**Architecture:** Build an additive residual path around the successful
`h13-context-diagnostic-20260725-03` run. Reuse its live authenticated lease for
the 55 stocks, calendar, runtimes, and Torch package; add an independently
authenticated SPY bundle and residual-only controller, runtime, finalizer,
runner, and armer. Do not modify any file in `CONTEXT_SOURCE_PATHS`, because
doing so would invalidate the successful source attempt.

**Tech Stack:** Python 3.12+, PyTorch 2.x, existing exact-JSON and frozen-input
helpers, GitButler.

## Fixed Boundaries

- Source attempt:
  `experiments/h13-context-diagnostic-20260725-03-attempt.json`,
  SHA-256
  `700d4e27ccd714e6156522be22515c9b3b04aa97dbdd6f09fd199e13463c1394`.
- Source outcome:
  `reports/h13-context-diagnostic-20260725-03/outcome.json`,
  SHA-256
  `bc33d4c86afeab4d7273215a81f2f701c68ff1a251fcb9935508098677063040`.
- Residual profile:
  `experiments/executable-h13-spy-residual.example.json`,
  SHA-256
  `cd5103fa93835222ae789a228ff776765c23bd7d0de6a2200c1c610ec557af19`.
- SPY fetch report:
  `data/spy-residual-20260725/fetch.json`,
  SHA-256
  `024e710102f866a3ffcd89ae22688d333f2736ed99b086f03680f380f3fbbaf6`.
- SPY CSV:
  `data/spy-residual-20260725/spy.csv`,
  SHA-256
  `ce8de54c6fddac96d2866687e97cea2367579051c9da5b360ad4ccda53c1ed2`.
- Session calendar:
  `universes/us-equities-core-2024-07-22_2026-07-21.json`,
  SHA-256
  `b1e0835a60624a67e21f7941ac00ece6c488937989560bbd4d0333afd869e5f8`.
- Source phases remain ordered `fold-1`, then `calibration`; their update
  budgets remain `302` and `349` per checkpoint.
- Each phase has exactly 11 fits: one global ridge, five global MLP seeds, and
  five panel-Transformer seeds.
- The residual target is
  \(z_{i,t}=r^{stock}_{i,t}-r^{SPY}_t\), with fixed beta \(1\).
- The zero residual is a baseline, never a fitted model.
- Both controls receive stock-only inputs. Only the panel Transformer receives
  the final completed SPY feature row.
- Outputs remain residual-only and development-only. They do not authorize an
  absolute forecast, a trade, a forward-clean claim, or a `$100` backtest.
- Generated attempts, reports, data, models, credentials, and caches remain
  untracked.

---

### Task 1: Freeze Source and Benchmark Metadata

**Files:**

- Modify: `.gitignore`
- Modify: `tools/relative_context_contract.py`
- Modify: `tests/python/test_relative_context_contract.py`

**Interfaces:**

- Add exact `FileBinding` constants for the source attempt, source outcome,
  residual profile, SPY fetch report, SPY CSV, and session calendar.
- Add
  `expected_source_context_outcome() -> dict[str, object]`.
- Add
  `validate_source_context_outcome(value: object) -> Mapping[str, object]`.
- Add
  `expected_spy_fetch_report(repository_root: Path) -> dict[str, object]`.
- Add
  `validate_spy_fetch_report(value: object, repository_root: Path) -> Mapping[str, object]`.

- [x] Add narrow ignore patterns for
  `/experiments/h13-spy-residual-*-attempt.json` and
  `/experiments/h13-spy-residual-*-outcome.json`.
- [x] Test the exact source bindings, selected history `17`, development-only
  evidence role, source-attempt link, phase order, and integrity hashes.
- [x] Test the exact SPY identity, adjusted request, split-adjusted return
  basis, absolute live CSV/calendar paths, `5,534` rows, `428` sessions, and
  zero-missing-bin audit.
- [x] Reject every missing, extra, aliased, or changed field and every
  reordered phase or evidence sequence.
- [x] Keep the validator Torch-free and independent of ignored local files by
  testing fresh expected values rather than reading calibration truth.
- [x] Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
```

- [x] Create a signed local checkpoint:

```text
feat(training): bind residual calibration inputs
```

Do not arm an attempt in this task.

---

### Task 2: Derive Causal Residual Phase Inputs

**Files:**

- Modify: `tools/relative_context_contract.py`
- Modify: `tools/relative_context.py`
- Create: `tools/spy_residual_controller.py`
- Modify: `tests/python/test_relative_context.py`
- Create: `tests/python/test_spy_residual_controller.py`

**Interfaces:**

- Add `MarketContextForwardWindows(stock, spy)` to
  `tools.relative_context`. It accepts two aligned
  `ForwardFeatureWindows` instances and returns only
  `(stock_features, spy_features)`; it contains no targets.
- Add immutable `ResidualForwardSeries` with fields `series: str`,
  `stock: ForwardFeatureWindows`,
  `market: MarketContextForwardWindows`, and
  `samples: tuple[SampleRows, ...]`.
- Add immutable `ResidualPreparedPhase` with fields
  `source: ContextPhase`, `binding: ResidualPhaseInput`,
  `training: tuple[tuple[str, TrainingData], ...]`, and
  `forward: tuple[ResidualForwardSeries, ...]`.
- Add the Torch-free immutable `ResidualTruthRow` to
  `tools.relative_context_contract`, with fields `as_of: str`, `entry: str`,
  `target: str`, and `value: float`. The finalizer imports it without importing
  the Torch-backed controller.
- Add
  `derive_residual_phases(context: ContextAttempt, lease: ContextLease, spy_csv: FrozenInput) -> tuple[ResidualPhaseInput, ...]`.
- Add
  `prepare_residual_phase(context: ContextAttempt, source_phase: ContextPhase, phase: ResidualPhaseInput, lease: ContextLease, spy_csv: FrozenInput) -> tuple[ResidualPreparedPhase, Callable[[], Mapping[str, tuple[ResidualTruthRow, ...]]]]`.

- [x] Reuse `context_all_phase_rows()` for stocks and build one canonical SPY
  `SessionSamples` sequence.
- [x] Reuse `align_spy_rows()` for every stock and phase. Require full equality
  of `(as_of, entry, target)` triples; never split SPY independently.
- [x] Recompute aggregate training and evaluation grid digests and require
  equality with the authenticated source `ContextPhase`.
- [x] Hash each stock and aligned SPY prefix only through the last training
  target. Build all 55 ordered `ResidualScalerInput` records per phase.
- [x] Prove evaluation-label changes do not alter the scaler-input digest,
  while any training-prefix, row-count, order, or grid change does.
- [x] Keep arming label-free: timestamp and byte-prefix hashing is allowed;
  numeric bar parsing, tensor preparation, training, prior prediction reads,
  and backtesting are not.
- [x] Build each stock and SPY forward window from bytes ending at the final
  `as_of` row. Neither `entry` nor `target` bytes may enter a forward dataset.
- [ ] Call the returned truth reader only after the run claim, fit ledger,
  prediction ledger, and prediction receipt are durable, frozen, and
  revalidated. Task 6 owns this runner-level gate; Task 2 only defers the read.
- [x] Add a per-prediction metamorphic test: for each evaluation sample,
  change rows strictly after that sample's own `as_of` and compare only that
  sample's stock input, SPY-conditioned input, and encoded prediction. Each
  must remain byte-identical; later overlapping samples may legitimately use
  a row after the earlier sample's `as_of`.
- [x] Run:

```sh
$PYTHON tests/python/test_relative_context_inputs.py
$TORCH_PYTHON tests/python/test_relative_context.py
$TORCH_PYTHON tests/python/test_spy_residual_controller.py
```

---

### Task 3: Implement the 11-Fit Residual Runtime

**Files:**

- Modify: `tools/relative_context_contract.py`
- Create: `tools/spy_residual_runtime.py`
- Create: `tests/python/test_spy_residual_runtime.py`

**Interfaces:**

- Add immutable `ResidualFitEvidence` with fields `fit: ContextFit`,
  `provenance_id: str`, `state_fingerprint: str`, and
  `training_loss: float`.
- Add immutable `ResidualPredictionEvidence` with fields
  `prediction: ContextPrediction`, `fit_provenance_id: str`,
  `state_fingerprint: str`, and `values: tuple[float, ...]`.
- Add
  `expected_residual_predictions(master: Sequence[str], phase: ContextPhase) -> tuple[ContextPrediction, ...]`.
- Add
  `ResidualRuntime(prepared: ResidualPreparedPhase, device: torch.device, runtime_sha256: str)`.
- Add
  `ResidualRuntime.fit_one(fit: ContextFit) -> tuple[str, float, object]`.
- Add
  `ResidualRuntime.predict_one(prediction: ContextPrediction, model: object) -> tuple[float, ...]`.
- Add exact record serializers and validators for the 11 fit records and
  `11 × 11` ordered evaluation prediction records in each phase.

- [x] Reuse the existing ridge, MLP, stock-balanced loader, optimizer-update,
  loss, and fingerprint primitives without modifying their source files.
- [x] Fit the ridge once per phase and each neural model once for each frozen
  seed `7/19/31/43/61`.
- [x] Reuse the selected history-17 checkpoint counts from the authenticated
  source phase; do not reselect epochs on calibration labels.
- [x] Feed stock-only residual windows to ridge and MLP. Feed
  `MarketContextWindows` only to `MarketContextTransformer`.
- [x] Models train in each stock's training-standardized residual units.
  Before float32 prediction encoding, convert each evaluation stock back to
  raw residual returns with
  `raw = scaled * target_scale + target_mean`. Bind that stock's training-only
  scaler digest into prediction provenance; Task 4 consumes raw units only.
- [x] Require every prediction vector to match the source evaluation grid,
  series order, observation count, model/seed axes, and finite float32
  encoding.
- [x] Test fit order, update counts, deterministic seeds, model input
  asymmetry, a tiny synthetic learnable residual, and inverse scaling with a
  nonzero target mean and nonunit target scale.
- [x] Run:

```sh
$TORCH_PYTHON tests/python/test_relative_context.py
$TORCH_PYTHON tests/python/test_spy_residual_runtime.py
```

---

### Task 4: Finalize Residual Metrics Without Trading Claims

**Files:**

- Create: `tools/finalize_spy_residual.py`
- Create: `tests/python/test_finalize_spy_residual.py`

**Interfaces:**

- Add
  `evaluate_residual_phase(master: Sequence[str], source: ContextPhase, phase: ResidualPhaseInput, evidence: Sequence[ResidualPredictionEvidence], truth: Mapping[str, Sequence[ResidualTruthRow]]) -> Mapping[str, object]`.
- Add
  `finalize_residual_run(attempt: FileBinding, phases: Sequence[ResidualPhaseInput], evaluations: Mapping[str, Mapping[str, object]], phase_inputs: Sequence[Mapping[str, object]], source_tree_sha256: str) -> Mapping[str, object]`.

- [x] Ensemble each neural model by arithmetic-mean prediction across five
  seeds before primary metrics.
- [x] Reparse `phase` against `source`, require its source/grid hashes to
  match, and derive evaluation stock order, counts, and per-series grids only
  from the authenticated `ContextPhase`.
- [x] Compute pooled raw residual \(R^2\) against zero:

```text
1 - sum((z - prediction)^2) / sum(z^2)
```

- [x] Compute paired absolute-error gain as:

```text
abs(z - reference) - abs(z - candidate)
```

  Average within stock, then equally across stocks. Positive values favor the
  candidate.
- [x] Use shared circular date blocks of `5`, `10`, and `20` days, `10,000`
  replicates, seed `20260725`, and equal-tailed 95% intervals only for paired
  absolute-error gains.
- [x] Compute secondary timestamp-centered cross-sectional \(R^2\) and mean
  valid-timestamp Spearman RankIC.
- [x] For centered cross-sectional \(R^2\), center truth and predictions within
  each exact common target timestamp, pool centered squared error and centered
  truth squares, and reject a zero pooled denominator.
- [x] For RankIC, assign average ranks to ties and use Pearson correlation of
  ranks. Skip timestamps with fewer than two stocks or a constant truth or
  prediction rank vector; report the valid count and reject if it is zero.
- [x] Define a bootstrap date as the UTC target-date prefix `target[:10]`.
  Draw one circular sequence of dates and apply the identical multiplicities
  to every stock, model, and comparison before recomputing stock-macro gains.
- [x] Report population standard deviation across seed predictions per
  observation, then stock-macro average it on the common grid.
- [x] Preserve the profile locks verbatim in every phase evaluation and
  terminal outcome.
- [x] Reject non-finite values, changed grids, incomplete seed/model closure,
  or a zero denominator rather than silently substituting a metric.
- [x] Emit one exact phase object containing only `schema`, `phase`,
  `evidence_role`, `target_kind`, `observation_count`, `stock_count`,
  `timestamp_count`, `primary`, `secondary`, `seed_dispersion`, `locks`, and
  `integrity`. The terminal object contains only `schema`, `evidence_role`,
  `inputs`, `phases`, `decision`, `locks`, and `integrity`.
- [x] Add named tests for hand-calculated pooled \(R^2\), stock-macro
  weighting, positive/negative paired gains, average-rank ties, constant-rank
  timestamp exclusion, shared circular-date sampling, seed population
  deviation, common-grid enforcement, and every zero-denominator rejection.
- [x] Verify that permuting rows, stocks, models, seeds, or dates is rejected.
- [x] Run:

```sh
$PYTHON tests/python/test_finalize_spy_residual.py
```

---

### Task 5: Define Attempt and Prediction-Receipt Contracts

**Files:**

- Modify: `tools/relative_context_contract.py`
- Modify: `tests/python/test_relative_context_contract.py`

**Interfaces:**

- Add
  `ResidualAttempt.read(path: Path, logical_path: Path, repository_root: Path, context: ContextAttempt) -> ResidualAttempt`
  with exactly `schema`, `status`, `attempt_path`, `run_id`, `run_dir`,
  `implementation_commit`, `source`, `config`, `benchmark`, `phases`,
  `source_tree`, `primary_python`, `torch_argv`, `torch_probe`, and
  `environment`.
- Define the final `RESIDUAL_SOURCE_PATHS` strings. Parsing synthetic attempts
  is allowed, but no live attempt may be emitted until every path exists and
  Task 6 authenticates it.
- Add immutable `ResidualReceipt` with fields `phase: str`,
  `attempt: FileBinding`, `fits: FileBinding`, `predictions: FileBinding`,
  `source_phase_sha256: str`, `residual_phase_sha256: str`,
  `evaluation_grid_sha256: str`, `source_tree_sha256: str`,
  `run_identity: tuple[int, int]`, `fit_count: int`, and
  `prediction_count: int`.
- Add `ResidualReceipt.parse(value: object) -> ResidualReceipt`,
  `ResidualReceipt.value() -> dict[str, object]`, and
  `ResidualReceipt.validate(source: ContextPhase, phase: ResidualPhaseInput, attempt: FileBinding, fits: FileBinding, predictions: FileBinding, source_tree_sha256: str, run_identity: tuple[int, int]) -> None`.
- Add
  `expected_residual_command(attempt_path: Path) -> tuple[str, str]`, returning
  only `("tools/run_spy_residual.py", attempt_path)`.

- [ ] Require receipt counts of exactly 11 fits and 121 prediction vectors;
  require every file path to be distinct and repository-relative.
- [ ] Test exact attempt and receipt round trips. Reject unknown fields,
  changed status/schema/run ID/path/environment/source/config/benchmark/tree,
  wrong source phase, wrong residual phase, wrong grid, wrong run inode,
  changed ledger bindings, Boolean counts, and any count other than 11/121.
- [ ] Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
```

---

### Task 6: Add the Runner and Arm the Complete Executable Closure

**Files:**

- Create: `tools/run_spy_residual.py`
- Create: `tools/arm_spy_residual.py`
- Create: `tests/python/test_run_spy_residual.py`
- Create: `tests/python/test_spy_residual_armer.py`

**Interfaces:**

- Add immutable `ResidualRunClaim` with fields `root: Path`, `path: Path`,
  `directory_identity: tuple[int, int]`, `attempt_path: Path`,
  `attempt_binding: FileBinding`, `attempt_identity: tuple[int, int]`, and a
  private ordered completed-phase map.
- Add type aliases
  `ResidualFitOne = Callable[[ContextFit], tuple[str, float, object]]` and
  `ResidualPredictOne = Callable[[ContextPrediction, object], tuple[float, ...]]`.
- Add `execute_residual_attempt(path: Path) -> Mapping[str, object]`.
- Add
  `execute_residual_phase(claim: ResidualRunClaim, attempt: ResidualAttempt, source: ContextPhase, phase: ResidualPhaseInput, fit_one: ResidualFitOne, predict_one: ResidualPredictOne, read_truth: Callable[[], Mapping[str, tuple[ResidualTruthRow, ...]]], verify: Callable[[], None]) -> Mapping[str, object]`.
- Add
  `authenticate_residual_attempt(attempt: ResidualAttempt) -> Iterator[ResidualLease]`.
- Add
  `arm(output: Path, implementation_commit: str, run_id: str) -> ResidualAttempt`.
- Add immutable `ResidualLease` with fields `context: ContextLease`,
  `benchmark: tuple[tuple[str, FrozenInput], ...]`, and a private callable
  verifier; calling the lease revalidates both closures.

- [ ] Require exact isolated, safe-path, no-site, no-bytecode Python launch.
- [ ] Claim the run directory atomically before any residual truth access.
- [ ] For each phase, publish ordered fit and label-free prediction ledgers,
  freeze both ledgers, publish and revalidate `ResidualReceipt`, then publish
  the truth-access record and invoke the truth reader.
- [ ] Publish the evaluation only after truth access. The terminal outcome
  binds the prediction receipt, truth-access record, and evaluation.
- [ ] Finalize the terminal outcome only after both receipts authenticate.
- [ ] On failure, publish a terminal integrity outcome that never implies a
  completed metric or authorization.
- [ ] Test interruption after every durable boundary. A partial phase may
  never be mistaken for a completed phase.
- [ ] Define `RESIDUAL_SOURCE_PATHS` as the sorted union of
  `CONTEXT_SOURCE_PATHS`, `ANALYSIS_SOURCE_PATHS`, and every residual armer,
  contract, input, adapter, controller, runtime, finalizer, and runner file
  executed by the run.
- [ ] Parse and authenticate the exact `-03` attempt through
  `ContextAttempt.read()` and `authenticate_context_attempt()`.
- [ ] Freeze the source attempt, outcome, run directory, and every bound phase
  fit, prediction, receipt, truth-access, and evaluation artifact. Reuse
  `analyze_context_cross_section._completed_run()` against that frozen closure
  and exact run-directory inode, discard its decoded predictions immediately,
  and expose only the authenticated `ContextAttempt` and `ContextLease` to the
  residual controller.
- [ ] Include `tools/analyze_context_cross_section.py`,
  `tools/context_cross_section.py`, and `tools/universe_cross_section.py` in
  the residual source tree because the source-run authenticator executes them.
- [ ] Source evidence authentication may decode the already-accessed `-03`
  fit/prediction ledgers and evaluations. Pass none of those values to
  residual phase derivation, and prove changing decoded source predictions
  cannot change a residual phase binding.
- [ ] Freeze the exact SPY bundle directory and require only `fetch.json` and
  `spy.csv`; reject symlinked parents/leaves, external hardlinks, aliases,
  changed inodes, changed bytes, and extra entries.
- [ ] Require the benchmark calendar path and hash to equal the live context
  lease calendar exactly.
- [ ] Require primary Python, Torch argv, and Torch probe to equal the
  authenticated source attempt. Preserve the source environment keys and
  bytecode policy, but derive
  `PYTHONPYCACHEPREFIX=reports/<residual-run-id>/.pycache`; never reuse the
  source run's cache path.
- [ ] Validate that `implementation_commit` is a real commit containing every
  declared source file at the exact frozen byte hash.
- [ ] Derive both residual phase bindings before publication without reading
  residual numeric values or labels.
- [ ] Publish the attempt exclusively. Hold the parent directory descriptor,
  capture the writer-owned inode, fsync, reopen with `O_NOFOLLOW`, require the
  same device/inode, regular type, mode, and single link, and validate bytes
  through that descriptor.
- [ ] Clean up only a still-owned, single-link temporary inode. Never unlink a
  replacement or externally hardlinked path.
- [ ] Existing attempt, run directory, or outcome blocks all work before input
  discovery.
- [ ] Test source/data/runtime mutations and post-publication replacement by
  symlink, same-byte new inode, and hardlink. Each must fail or burn the
  attempt; none may report `armed`.
- [ ] Test that arming never reads residual numeric bars, residual truth,
  residual predictions, or model state and never imports Torch, trains,
  evaluates a residual model, or backtests. The only permitted prediction and
  evaluation reads are the frozen `-03` artifacts consumed by
  `_completed_run()`.
- [ ] Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
$PYTHON tests/python/test_spy_residual_armer.py
$TORCH_PYTHON tests/python/test_relative_context.py
$TORCH_PYTHON tests/python/test_spy_residual_controller.py
$TORCH_PYTHON tests/python/test_spy_residual_runtime.py
$PYTHON tests/python/test_finalize_spy_residual.py
$TORCH_PYTHON tests/python/test_run_spy_residual.py
make -B PYTHON=$PYTHON check
```

- [ ] Create a signed local checkpoint containing the complete runnable
  closure:

```text
feat(training): execute authenticated residual calibration
```

- [ ] Verify author, committer, and ED25519 signature. Do not push.

---

### Task 7: Create and Execute One Ignored Attempt

- [ ] Invoke the armer with the exact signed implementation commit from Task 6.
- [ ] Re-authenticate the published attempt and prove all source, source
  evidence, benchmark, calendar, stock data, phase grids, runtimes, and Torch
  package bindings still match.
- [ ] Execute the sole derived runner command once.
- [ ] Authenticate every phase receipt and terminal outcome.
- [ ] Report raw residual \(R^2\), stock-macro MAE gains and confidence
  intervals, centered cross-sectional \(R^2\), RankIC, seed dispersion,
  training loss, and runtime.
- [ ] Compare all models with the zero residual and global ridge baselines.
- [ ] Keep the attempt and all output artifacts ignored and uncommitted.

## Stop Condition

The residual experiment ends with model-comparison evidence only. Do not
reconstruct absolute returns or run the `$100` trading simulation from these
outputs. If the residual Transformer earns a credible calibration gain, write
a separate plan that freezes an absolute-return reconstruction rule, an
execution threshold, costs, non-overlap policy, and genuinely later holdout
before any backtest.
