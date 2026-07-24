# Conditioned panel calibration postmortem

Purpose: preserve the terminal evidence from
`h13-conditioned-panel-20260724-01` without treating an integrity failure as
an accepted gate result.

## Protocol

- Date: 2026-07-24
- Status: `analysis-integrity-failure`; the one-shot attempt was not rerun.
- Series: AAPL, MSFT, and SPY; target: `executable-return-v1`; horizon: 13 bars.
- Comparison: stock-conditioned versus unconditioned panel Transformer.
- Work: 207 series-equivalent runs and 30 physical panel fits.
- Scope: calibration only. No test labels, policy selection, or backtest.

## Finding

The report and prediction ledger are complete, but the analyzer rejected two
equivalent floating-point aggregations:

- Mean of 30 paired deltas: `0.000007881824439170873`
- Difference of macro means: `0.000007881824439169224`
- Absolute discrepancy: `0.0000000000000000016483261153568685`

The discrepancy is less than one ULP at the source-MAE scale, but the validator
measured tolerance at the much smaller, cancellation-affected result. No
analysis report was published.

Read-only recomputation is negative: validation return MAE was
`0.009655729344231383` conditioned versus `0.009647847519792214`
unconditioned, or `-0.081695%` relative improvement, with 12 wins and 18
losses. Calibration return MAE was `0.011575864860953704` conditioned versus
`0.011569278040265514` unconditioned, or `-0.056934%` relative improvement.
These values are postmortem evidence, not a formal gate result.

## Provenance

- Attempt SHA-256:
  `22f802ade058d33a6c1c50c2062486a62cc9e7923794f9b9ced9ce51d02ddda7`
- Experiment SHA-256:
  `79de195a7cbcb1e139cef6181261f3b36c188bdc842a0fd18c88c592357a1940`
- Calibration-ledger SHA-256:
  `04575ee2ef517e29b523e888ea4099ca5d0dee4e915d373e71440b8197ed48aa`
- Terminal outcome:
  `experiments/executable-h13-conditioned-panel-outcome.json`
