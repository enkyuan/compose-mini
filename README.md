# compose-mini

`compose-mini` is a staged C11 inference runtime for a pretrained,
encoder-only Transformer. Its target is the next completed bar's close from a
fixed window of completed OHLCV history.

The repository is currently a scaffold. It does not yet load model artifacts
or produce forecasts, and training and trading decisions are outside its
scope. See [docs/forecasting-contract.md](docs/forecasting-contract.md) for the
intended system boundary and delivery stages.
