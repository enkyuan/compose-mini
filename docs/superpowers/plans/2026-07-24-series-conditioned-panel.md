# Series-Conditioned Panel Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` (recommended) or `executing-plans` to implement
> this plan task by task. Follow `karpathy-guidelines`, `ponytail`, repository
> `AGENTS.md`, and the local-only GitButler checkpoint boundaries below.

**Goal:** Add an experiment-only learned series identity embedding so the
shared panel Transformer can represent stock-specific behavior without
changing the five-feature inference artifact or opening reserved test labels.

**Architecture:** Keep each stock's training-only feature and target scalers.
Wrap each already-formed window with its ordered integer series ID, add one
zero-initialized learned embedding to every token after the existing input
projection and positional encoding, and retain the same encoder blocks and
scalar head. Compare this conditioned model with the existing unconditioned
panel model in later frozen calibration work.

**Tech:** Python 3.12+, existing PyTorch dependency, procedural Python tests,
Make, and GitButler. Add no dependency and change no C code.

**Delivery boundary:** This plan covers the model path and deterministic
synthetic verification only. Make signed local checkpoints authored by
`enkyuan <yuan.enkng@gmail.com>`. Do not fetch Massive data, use credentials,
change the universe, create generated reports/models, inspect test labels,
authorize a policy, run a `$100` backtest, push, land, or open a pull request.

## Why This Change Comes Before More Stocks

The frozen AAPL/MSFT/SPY panel experiment was a valid gate failure:

- the unconditioned panel validation return MAE was `0.00964785`, behind the
  MLP at `0.00957891` and the zero-return reference at `0.00962274`;
- it won only 13 of 30 paired stock/fold/seed comparisons against the local
  Transformer, below the predeclared 20-win gate;
- its calibration return MAE was `0.01156928`, behind zero return at
  `0.01149941`;
- its calibration direction accuracy was `47.92%`, below the `52.98%`
  per-stock majority-sign reference.

The result rejects the current unconditioned pooling method, not all shared
models. The pooled-ridge diagnostic improved macro return MAE by about `2.73%`
over separate ridge fits, so some cross-stock structure may be transferable.
The smallest unresolved question is whether erasing stock identity causes
negative transfer.

The single authorized Massive recovery attempt failed before producing a CSV
or report. Its directory remains empty and the attempt is consumed. Universe
expansion therefore remains a separate future experiment; this plan neither
retries the request nor edits the conflicted universe work.

## Mathematical Contract

For stock \(i\), window-end time \(t\), and independently standardized target
\(z_{i,t}\), ordinary concatenation minimizes the sample-weighted objective:

```text
L_concat(theta) =
  (1 / sum_i N_i) * sum_i sum_t (
    z[i,t] - f(theta, X[i,t])
  )^2

L_macro(theta) =
  (1 / S) * sum_i ((1 / N_i) * sum_t loss[i,t])
```

The frozen histories have equal \(N_i\), so `L_concat == L_macro` for this
experiment. That equality does not hold for unequal histories.

When two normalized windows have the same values, the current model must
produce the same prediction even when their stocks have different conditional
means. The conditional variance decomposes as:

```text
Var(z | X) =
  E[Var(z | X, i) | X]
  + Var(E[z | X, i] | X)
```

The second term is between-stock variation that an identity-blind model cannot
remove. Add an ordered learned embedding \(E_i\):

```text
h[0,i,t] = X[i,t] @ W[input] + position[t] + E[i]
prediction[i,t] = head(encoder(h[0,i,t]))
```

The new parameter count is `series_count * model_dim`: 48 parameters for
three stocks at model dimension 16. Only stock \(i\)'s samples contribute a
data gradient to `E[i]`; all stocks contribute data gradients to the shared
projection, encoder, and head:

```text
dL_data / dE[i] = sum of token-level gradients from stock i only
dL_data / dtheta_shared = sum of gradients from all stocks
```

AdamW weight decay remains unchanged and can update a nonzero embedding row
even on an optimizer step whose batch contains no sample from that stock.

Initialize every embedding row to zero. At initialization the conditioned
model is exactly the existing unconditioned function, which isolates the
effect of learning identity and prevents an initialization change from
confounding the comparison.

The current three histories are aligned and equal length, so ordinary
concatenation already implements equal-stock weighting. Do not add a sampler
in this checkpoint. When unequal histories are supported later, preserve the
macro objective by sampling a stock uniformly and then a timestamp uniformly:

```text
L_macro = (1 / S) * sum_i ((1 / N_i) * sum_t loss[i,t])
```

## Locked Design

- Name the new model `conditioned_panel_transformer`.
- Keep `panel_transformer` as the unconditioned comparator.
- Encode identity as an integer lookup, not a one-hot feature.
- Bind row `i` of the embedding to series `i` in the ordered CLI/bound
  execution input.
- Keep the outer dataset sample contract at four fields. For conditioned
  datasets, the first field is `(window, series_id)`.
- Let the training loop accept either one tensor or a sequence of tensors.
  Keep one shared forward helper; do not duplicate train/evaluate functions.
- Add optional context after the input projection and positional encoding in
  `ForecastTransformer.forward`.
- Keep `Config.in_dim == 5`, the artifact schema, weight export, C runtime,
  prediction ledger schema, optimizer, loss, scalers, and inverse scaling
  unchanged.
- Support known series only. An unseen ticker has no learned embedding and
  requires a later cold-start design or retraining.
- Do not add sector embeddings, per-stock output heads, a larger Transformer,
  PCGrad, a new sampler, or a new optimizer.
- Do not change the stock universe and conditioning in the same first
  experiment; doing both would make any difference unattributable.

## File Map

| File | Change | Purpose |
| --- | --- | --- |
| `tools/train.py` | Modify | Optional token context and one generic batch forward path |
| `tools/experiment.py` | Modify | Ordered series IDs, embedding wrapper, and generalized panel loops |
| `tests/python/test_training.py` | Modify | Core tensor/nested-batch and context invariants |
| `tests/python/test_experiment.py` | Modify | Identity isolation, ordering, run accounting, and determinism |

No manifest, analyzer, driver, report, model artifact, dataset, documentation
outside this plan, or C source changes in this checkpoint.

## Checkpoint 0: Save the Reviewed Plan

- [ ] Run `but diff`, select only this new plan's change ID, and commit it on
  `enkyuan/conditioned-panel-plan`:

```text
docs(training): plan series-conditioned panel
```

- [ ] Stop after the GitButler fast-path commit and report the returned
  workspace state. Do not include the unrelated `Makefile` or
  `docs/training.md` changes.
- [ ] Before Task 1, stack a new implementation branch
  `enkyuan/conditioned-panel` above the reviewed plan.

## Task 1: Specify Core Context and Batch Behavior

**Files:**

- Modify: `tests/python/test_training.py`

- [ ] Add a failing test that calls the existing `ForecastTransformer` with a
  normal `[batch, sequence, 5]` tensor and confirms its output remains
  `[batch]`.
- [ ] Add a failing test that sends `(values, context)` through `mean_loss`,
  `train_epoch`, and `evaluate`, proving PyTorch's default collator shape is
  accepted without a second training loop.
- [ ] Add a failing exact-equivalence test:

```python
plain = model(values)
conditioned = model(values, torch.zeros(batch, model_dim))
assert torch.equal(plain, conditioned)
```

- [ ] Add a failing isolation test showing a nonzero context changes only the
  sample receiving it.
- [ ] Import `DataLoader`, `Dataset`, `mean_loss`, `train_epoch`, and `evaluate`
  in `tests/python/test_training.py`; keep the synthetic nested dataset local
  to one `verify_conditioned_batches()` helper.
- [ ] Call `verify_conditioned_batches()` from that file's existing `main()`.
- [ ] Run:

```zsh
TORCH=(/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python)
"${TORCH[@]}" tests/python/test_training.py
```

Expected: fail only on the missing optional-context/nested-input behavior.

## Task 2: Add the Minimal Shared Forward Path

**Files:**

- Modify: `tools/train.py`

- [ ] Change the model signature to:

```python
def forward(
    self, values: torch.Tensor, context: torch.Tensor | None = None,
) -> torch.Tensor:
```

- [ ] Form the hidden state once:

```python
hidden = values @ self.embed_W + self.position
if context is not None:
    hidden = hidden + context.unsqueeze(1)
```

The branch is the only new model behavior. Broadcasting applies one identity
vector to every token in that sample.

- [ ] Add one private helper:

```python
def _model_output(
    model: nn.Module,
    values: torch.Tensor | Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    inputs = (values,) if isinstance(values, torch.Tensor) else tuple(values)
    return model(*(item.to(device) for item in inputs))
```

Use the already imported `Sequence`. Do not add speculative conversion or
error wrapping; the DataLoader contract supplies tensors.

- [ ] Route `mean_loss`, `train_epoch`, and `evaluate` through
  `_model_output`.
- [ ] In `train_epoch`, use the target batch length for loss accumulation
  because a nested first field does not have one unambiguous tensor length.
- [ ] Re-run `tests/python/test_training.py`.
- [ ] Run the artifact and C/Python parity tests already exercised by
  `make check` and confirm the unconditioned export remains byte-compatible.

## Task 3: Specify Ordered Series Conditioning

**Files:**

- Modify: `tests/python/test_experiment.py`

- [ ] Add a small synthetic test for `_SeriesDataset`:
  the last sample of series 0 has ID 0 and the first sample of series 1 has ID
  1, with windows and the remaining three fields unchanged.
- [ ] Assert `SeriesTransformer` owns exactly one embedding with shape
  `[3, model_dim]` and all-zero initial values.
- [ ] Copy one base Transformer's state into a conditioned model and assert
  their outputs are exactly equal before training.
- [ ] Change only embedding row 1 and assert ID 1 output changes while ID 0
  output remains exactly equal.
- [ ] Assert the embedding row order is the bound input order
  `("A", "B", "C")`; alphabetic resorting is forbidden.
- [ ] Generalize `_panel_selected_epochs` test evidence by model name and
  confirm records for the two panel models cannot satisfy each other's epoch
  selection.
- [ ] Extend run accounting:

```text
existing local models                       117 equivalents
unconditioned panel Transformer              45 equivalents
conditioned panel Transformer                45 equivalents
total                                        207 equivalents

physical panel fits:
  2 panel models * (2 folds + 1 calibration) * 5 seeds
  = 30 fits
```

- [ ] Assert panel test authorization is still rejected when either panel model
  is present.
- [ ] Assert a repeated synthetic run produces the same report and calibration
  ledger.
- [ ] Import `PANEL_MODEL_SET`, `_SeriesDataset`, `SeriesTransformer`, and any
  generalized private helpers beside the existing experiment imports. Put the
  new assertions in `verify_series_conditioning()` and call it from the
  existing `main()`.
- [ ] Update `panel_fixture` so `expected_panel_fits` is:

```python
len(PANEL_MODEL_SET.intersection(sweep.models)) * len(sweep.seeds) * (
    len(sweep.candidates) * sweep.folds + 1
)
```

- [ ] Run:

```zsh
TORCH=(/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python)
"${TORCH[@]}" tests/python/test_experiment.py
```

Expected: fail only on the missing conditioned model and generalized
orchestration.

## Task 4: Implement Experiment-Only Identity Conditioning

**Files:**

- Modify: `tools/experiment.py`

- [ ] Preserve deterministic presentation order:

```python
PANEL_MODELS = ("panel_transformer", "conditioned_panel_transformer")
PANEL_MODEL_SET = frozenset(PANEL_MODELS)
```

Use the tuple for loops/report order and the set for membership checks.
Extend `TRANSFORMERS`, `NEURAL`, and `MODELS` without changing default local
models.
- [ ] Inside `_run_experiment`, derive the exact active order once:

```python
panel_models = tuple(model for model in PANEL_MODELS if model in sweep.models)
```

Use `panel_models` for training loops, report ordering, and physical-fit
accounting. `requested_models` is empty in calibration-only execution and must
not control panel work.

- [ ] Add a minimal dataset adapter:

```python
class _SeriesDataset(Dataset):
    def __init__(self, dataset: Dataset, series_id: int) -> None:
        self.dataset, self.series_id = dataset, series_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[object, ...]:
        values, *rest = self.dataset[index]
        return (values, self.series_id), *rest
```

Import `Dataset` beside `ConcatDataset`. Do not copy windows or targets.

- [ ] Add:

```python
class SeriesTransformer(nn.Module):
    def __init__(self, config: Config, series_count: int) -> None:
        super().__init__()
        self.model = ForecastTransformer(config)
        self.series = nn.Embedding(series_count, config.model_dim)
        nn.init.zeros_(self.series.weight)

    def forward(
        self, values: torch.Tensor, series_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(values, self.series(series_id))
```

The wrapper is experiment-only and leaves artifact export unaware of series
identity.

- [ ] Generalize `_neural_model` and `_fit_neural` with a defaulted
  `series_count: int = 0`. Construct `SeriesTransformer` only for
  `conditioned_panel_transformer`; preserve every existing caller.
- [ ] Replace `_panel_data(members)` with
  `_panel_data(members, conditioned=False)`. When conditioned, wrap every
  split of member `i` in `_SeriesDataset(..., i)` before concatenation.
- [ ] Add one reusable adapter for both pooled training and per-stock
  evaluation:

```python
def _conditioned(data: DataSplits, series_id: int) -> DataSplits:
    return DataSplits(*(
        _SeriesDataset(getattr(data, name), series_id)
        for name in ("train", "validation", "test")
    ))
```

Have `_panel_data` concatenate `_conditioned(member, i)` splits when requested;
do not teach `TrainingData` about experiment-only IDs.
- [ ] Change `_panel_selected_epochs` to accept `model_name`; filter exact
  model evidence.
- [ ] Loop over `panel_models` for validation and calibration instead of
  copying either orchestration path. Pass `len(names)` only to the
  conditioned model and use conditioned pooled data only for it.
- [ ] Build conditioned per-stock loaders with `_conditioned(data, series_id)`
  during validation and calibration evaluation. Keep the original
  `TrainingData` beside each loader for its scaler, fingerprint, timestamps,
  and prediction metadata. A conditioned model must never receive a bare
  window tensor.
- [ ] Sort panel validation records by `panel_models` order, candidate order,
  series, fold, and seed. Sort panel calibration records and ledgers by
  `panel_models` order, series, seed, and target time as applicable. Assert
  that order in the deterministic orchestration test.
- [ ] Compute every `(panel_model, seed)` selected epoch before calling
  `fit_epochs` for any panel calibration model. Add a test with invalid
  evidence for the later model and assert no calibration fit occurred.
- [ ] Count equivalent runs with existing `expected_runs`. Change bound
  physical panel fits to:

```text
number of active panel models
* number of seeds
* (number of candidates * folds + 1)
```

- [ ] When the conditioned model is present, add exact protocol metadata:

```json
"panel_conditioning": {
  "model": "conditioned_panel_transformer",
  "kind": "learned-series-embedding",
  "series_order": ["A", "B", "C"],
  "initialization": "zeros",
  "application": "additive-before-encoder"
}
```

Derive `series_order` from the already validated bound execution order.
Do not introduce a new report schema solely for this additive metadata.
- [ ] Re-run `tests/python/test_training.py` and
  `tests/python/test_experiment.py`.

## Task 5: Verify the Whole Unchanged Boundary

- [ ] Run focused standard-library suites:

```zsh
PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$PYTHON" tests/python/test_panel_analysis.py
"$PYTHON" tests/python/test_panel_driver.py
```

- [ ] Run the full repository gate:

```zsh
make -B PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 check
```

- [ ] Run the optional Torch gate with the repository's existing uv/Python
  command:

```zsh
make check-training
```

If this shell's default Python lacks PyTorch, use the exact uv-backed command
already declared by the Make target; do not install or update dependencies.

- [ ] Confirm all nine C suites, artifact validation, data tests, e2e parity,
  experiment tests, panel analyzer/driver tests, and the optional training
  smoke pass.
- [ ] Confirm the existing C e2e prediction remains within its established
  five-ULP parity boundary.
- [ ] Scan the diff for credentials, generated CSVs, reports, and model files.
  None may be tracked.

## Task 6: Review and Make Local Checkpoints

- [ ] Commit the four implementation/test files together after all gates pass:

```text
feat(training): condition shared panel by series
```

- [ ] Request an independent read-only diff review for correctness, leakage,
  determinism, run accounting, and scope.
- [ ] Amend only the unpublished local implementation checkpoint if review
  finds a fix that belongs to it.
- [ ] Report branch, commit, author/committer identity, GitButler's available
  signature evidence, exact tests, and any unrelated shared-worktree failures.
- [ ] Do not push or land.

## Next-Phase Boundary

After implementation and independent review, write a separate frozen comparison
plan before any live work. That later plan may compare
`panel_transformer` with `conditioned_panel_transformer` on the exact existing
AAPL/MSFT/SPY inputs, source/runtime closure, seeds, gates, and calibration-only
boundary. It must allocate 207 series-equivalent runs and 30 physical panel
fits.

That later phase must also update and re-review `tools/analyze_panel.py`, its
fixtures, and the immutable attempt contract. They currently require the old
model set, 162 equivalent runs, 15 physical fits, and exact legacy protocol.
Task 5 above proves backward compatibility; it does not claim conditioned
end-to-end attempt readiness.

Only if the conditioned model passes every predeclared validation and
calibration gate may another checkpoint select a policy and request a
one-time reserved-test/backtest authorization. More epochs, a larger universe,
larger architecture, multi-task gradient methods, or another Massive request
are separate hypotheses and must not be folded into that comparison.
