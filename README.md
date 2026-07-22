# compose-mini

`compose-mini` is a staged C11 inference runtime for a pretrained,
encoder-only Transformer. Its target is the next completed bar's close from a
fixed window of completed OHLCV history.

The core encoder, scalar head, versioned artifact loader, chronological CSV
windowing, and JSONL inference CLI are implemented and tested. External model
training and trading decisions remain outside this runtime. See
[docs/forecasting-contract.md](docs/forecasting-contract.md) for the system
boundary and delivery stages.

```sh
make check
bin/transformer MODEL CSV INSTRUMENT INTERVAL FINAL_TARGET_TIME
```
