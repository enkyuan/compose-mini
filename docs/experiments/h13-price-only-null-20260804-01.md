# Price-only forecasting null result

Purpose: consolidate three independent horizon-13 studies that each tried to
forecast returns from price history alone, and record that all three failed to
clear the zero-change benchmark. This is a negative result kept as evidence so
the levers below are not re-attempted without new inputs.

## Protocol

- Date: 2026-08-04.
- Common target: `executable-return-v1`, `log(close[t+13] / open[t+1])`,
  horizon 13 bars, 30-minute bars.
- Common benchmark: zero-change (`last_close`), i.e. predicting no move.
- Common metric: mean validation scaled-MSE (lower is better) and direction
  accuracy (0.5 is a coin flip); residual study adds R^2 versus zero.
- Scope: calibration/validation only. No test labels, policy selection, or
  backtest was consumed in any of the three studies.
- Inputs: OHLCV only, in every study.

## Finding

No framing beat the zero-change benchmark by a usable margin, and simpler
models matched or beat the Transformer throughout.

**1. Absolute per-ticker (55-ticker universe, `liquid-common-55-20260724-02`).**
Each ticker trained in isolation. The model beat the naive close-MAE on 6 of
55 tickers, all by <=0.3%; mean direction accuracy `0.490` (below a coin
flip); mean close-MAE ratio `1.11` (11% worse than predicting no change).

**2. Pooled forward-return panel (AAPL, MSFT, SPY).** Walk-forward, 5 seeds,
5 model families. Validation scaled-MSE: mlp `0.926`, transformer `0.932`,
last_close `0.940`, linear `0.979`, rolling_mean `2.436`. The Transformer beat
naive by `0.9%` and lost to a plain MLP; best direction accuracy `0.535`.

**3. SPY-residual (11 stocks, market move removed).** Predicting stock return
minus aligned SPY return. Pooled residual R^2 versus zero was negative for
every model: panel_transformer `-0.0058`, global_mlp `-0.0412`,
global_ridge `-0.0494`; Spearman rank IC ~`0.002`. The zero-anchored shrinkage
follow-up produced a paired-MAE gain of `~0.00009` on 6 of 11 stocks, i.e. no
recoverable signal.

**4. Tier A engineered features (this study).** Framing 2's exact protocol with
one added candidate: `stationary-v1` plus rolling realized volatility, volume
z-score, and intraday range percent (8 features, OHLCV-derived, no new data).
Engineered features did not help and degraded the flexible models:

| Model | raw val-MSE | tier-a val-MSE | delta |
| --- | --- | --- | --- |
| transformer | 0.9447 | 0.9785 | +0.0338 worse |
| mlp | 0.9399 | 0.9564 | +0.0165 worse |
| linear | 0.9933 | 0.9712 | -0.0221 better |
| rolling_mean | 2.4645 | 2.4645 | 0 |
| last_close | 0.9535 | (n/a) | benchmark |

The Transformer moved from marginally beating naive (`0.9447 < 0.9535`) to
losing to it (`0.9785 > 0.9535`), and its direction accuracy fell `0.534` ->
`0.515`. Only the linear model improved, from worse-than-naive to roughly tied,
consistent with the extra features adding variance the neural models overfit
rather than signal they exploited. (Absolute values differ from framing 2
because this is a separate run of the same protocol; the raw-17 column is the
in-run baseline and last_close `0.9535` its in-run benchmark, so raw-vs-tier-a
is the only within-run comparison.)

## Interpretation

Four levers inside "price history only" — model architecture, target framing
(absolute, forward, market-residual), universe breadth (11 -> 55), and
OHLCV-derived engineered features — have each been tried and each failed to
clear the zero-change benchmark by a usable margin. The wall is the input, not
the model. The next untried lever is exogenous data (options-implied
volatility, factor/sector exposures, cross-asset signals), which requires a
data source not currently wired to the fetcher.

## Provenance

- Absolute universe: `data/liquid-common-55-20260724-02/` (55 tickers),
  trained per-ticker with `tools/train.py` defaults.
- Panel: `reports/executable-h13-calibration.json`
  (`validation_summary`, `selection`).
- Residual: `reports/h13-spy-residual-20260725-01/calibration-evaluation.json`;
  shrinkage `reports/h13-spy-residual-20260725-01-shrinkage/shrinkage.json`.
- Tier A: sweep and report under the session scratchpad; feature set
  `tier-a-v1` added to `tools/train.py`; width unlocked in
  `tools/experiment.py` and `tools/artifact_v1.py` (V1 export remains
  OHLCV-only). Run: `tools/experiment.py --calibration-only`,
  horizon 13, seeds 7/19/31/43/61, 2 folds.
