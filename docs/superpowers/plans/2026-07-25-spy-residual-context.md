# SPY-Residual Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or execute each task inline with its stated
> checks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest reusable adapter for a stable market-relative
target and strictly completed SPY context without changing the sealed
experiment or inference paths.

**Architecture:** Keep the absolute-return runtime and completed context run
unchanged. Build an experiment-only adapter that subtracts the aligned SPY
executable return from each stock's executable return, fits residual scalers
on training rows only, and pairs each stock window with its completed SPY
window. A separate plan will bind and run this adapter after its invariants are
proven in isolation. The adapter feeds training and scaled residual loss only;
it deliberately omits stock prices so the absolute-return evaluator cannot
misinterpret residuals as executable returns.

**Tech Stack:** Python 3.12+, PyTorch, existing context contracts and
GitButler.

## Global Constraints

- The stock target remains
  `log(close[target] / open[entry])`; the experiment target is stock target
  minus the exactly aligned SPY target.
- SPY inputs stop at each opportunity's completed `as_of` bar. Target-period
  SPY prices are labels and never model inputs.
- Matching `as_of_ordinal` values are necessary but not sufficient to prove
  timestamp identity. The later runner must authenticate one common calendar
  and timestamp-grid hash before constructing either preparation.
- Fit every stock residual mean and scale from that stock's retained training
  rows only.
- Keep history `17`, horizon `13`, the existing update budgets, five seeds,
  stock-balanced sampling, phases, and point-in-time stock manifests fixed.
- Do not use ticker embeddings for unseen-stock claims. Do not call liquidity
  strata sectors.
- Generated data, attempts, reports, models, credentials, and caches remain
  untracked.
- Do not run or authorize a `$100` backtest from residual predictions alone.
  Absolute forecasts require a separately frozen SPY forecast plus the
  residual forecast.

---

### Task 1: Pure Residual and Completed-Market Adapter

**Files:**

- Create: `tools/relative_context.py`
- Create: `tests/python/test_relative_context.py`

**Interfaces:**

- Consumes: two aligned, indexed `TrainingData` values whose target kind is
  `executable-return-v1`.
- Produces:
  `spy_residual_data(stock: TrainingData, spy: TrainingData) -> TrainingData`
  and `MarketContextTransformer(config: Config)`.
- Alignment means both preparations use the same feature set, horizon, split
  coordinates, sequence length, and ordered `SampleRows.as_of_ordinal`
  values. Raw row numbers remain series-local and are not compared. The
  caller is responsible for the authenticated common-grid precondition.

- [x] **Step 1: Write the failing residual tests**

Implement these exact cases with a four-sample indexed fixture, two retained
training rows, and raw stock/SPY returns reconstructed from independent
training-fitted scalers:

- `verify_residual_targets`: assert all four raw residuals, training-only
  residual mean/population scale, stock feature reuse, and unchanged scalers
  after changing only later SPY labels.
- `verify_alignment_guards`: reject mismatched ordinals, horizon, feature set,
  split coordinates, shared preparation tensors, non-executable targets,
  non-finite target scalers, and zero residual scale.
- `verify_causal_windows`: mutate SPY feature rows after the first completed
  window and assert its nested input is unchanged.
- `verify_context_model`: assert zero-context identity; different completed
  context; no effect from earlier SPY rows; rejection of batch, sequence,
  feature, and dtype broadcasting; and `mean_loss` plus backward smoke tests.
- Assert each dataset item has only nested model inputs and a residual target;
  references and outcomes are absent by design.

```python
residual = spy_residual_data(stock, spy)
actual = (
    residual.train.stock.targets
    * residual.target_scale
    + residual.target_mean
)
torch.testing.assert_close(actual, stock_returns - spy_returns)

with torch.no_grad():
    torch.testing.assert_close(
        contextual(stock_windows, torch.zeros_like(spy_windows)),
        contextual.model(stock_windows),
    )
```

- [x] **Step 2: Run the test and verify it fails**

Run:

```sh
$PYTHON tests/python/test_relative_context.py
```

Expected: failure because `tools.relative_context` does not exist.

- [x] **Step 3: Implement the minimal adapter**

Use one training-only dataset wrapper and the existing Transformer:

```python
SPY_RESIDUAL_TARGET = "spy-residual-executable-return-v1"


class MarketContextWindows(Dataset):
    def __init__(self, stock: Windows, spy: Windows) -> None:
        if not _aligned(stock, spy):
            raise ValueError("stock and SPY windows are not aligned")
        self.stock, self.spy = stock, spy

    def __len__(self) -> int:
        return len(self.stock)

    def __getitem__(self, index: int) -> tuple[object, ...]:
        stock, target, *_ = self.stock[index]
        return (stock, self.spy[index][0]), target


class MarketContextTransformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.model = ForecastTransformer(config)

    def forward(
        self, stock: torch.Tensor, spy: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            isinstance(stock, torch.Tensor)
            and isinstance(spy, torch.Tensor)
            and stock.ndim == 3
            and stock.shape == spy.shape
            and stock.shape[1:] == (
                self.model.config.seq_len,
                self.model.config.in_dim,
            )
        )
        if not expected:
            raise ValueError("stock and SPY inputs have invalid shapes")
        return self.model(stock, spy[:, -1] @ self.model.embed_W)
```

`_aligned()` compares sequence length, split coordinates, feature width, and
the full ordered `as_of_ordinal` grid. `spy_residual_data()` additionally
requires one indexed preparation per series, `executable-return-v1`, equal
feature sets and horizons, and one nonempty training split. It reconstructs
the two full raw target tensors from their training-fitted scalers, subtracts
SPY elementwise, fits a new finite positive residual scale from the stock
training slice only, reuses the stock feature storage and split coordinates,
and wraps the three rebuilt residual-target splits with
`MarketContextWindows`.

Each series' three source splits must share the same feature, target,
reference, and outcome tensors plus the same indexed sample metadata.
Preparations must retain the `prepare_rows()` CPU-float32 contract and
zero-dimensional target scalers; this prevents implicit dtype promotion and
shape broadcasting.
`MarketContextWindows` does not inherit `Windows`, so `tail_training_data()`
rejects it instead of silently stripping context. Its two-field samples work
with `mean_loss()` and training loops but fail closed in `evaluate()`, whose
absolute close-price metrics require references and outcomes.

- [x] **Step 4: Run focused checks**

Run:

```sh
$PYTHON tests/python/test_relative_context.py
$PYTHON tests/python/test_training.py
```

Expected: both pass.

- [x] **Step 5: Create a signed local checkpoint**

Commit only the two files as:

```text
feat(training): build causal SPY residual inputs
```

Do not push.

## Next Checkpoint Boundary

After Task 1 passes, write a separate plan for the authenticated experiment
contract, runner, and analyzer. That plan must freeze history `17`, horizon
`13`, models `global_ridge`, `pooled_mlp`, and `panel_transformer`, seeds
`7/19/31/43/61`, exact SPY bindings, and the current chronological phases.

No `$100` backtest follows automatically. A passing stock-selection diagnostic
must first be composed with a separately frozen absolute SPY forecast:

\[
\widehat r_{i,t}=\widehat r_{\mathrm{SPY},t}
                 +\widehat z_{i,t}.
\]

Only that absolute forecast may enter the existing costed policy path.
