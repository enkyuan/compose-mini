# Point-in-Time Common-Stock Universe Expansion Implementation Plan

> **For Codex:** Use `subagent-driven-development` to execute this plan one
> task at a time. Follow `karpathy-guidelines`, `ponytail`, repository
> `AGENTS.md`, and the local-only GitButler checkpoint boundaries below.

**Goal:** Test whether the existing horizon-13 Transformer generalizes across
11 predeclared U.S. common stocks without opening the reserved historical test
boundary.

**Architecture:** Add a strict manifest-driven Massive fetcher, retain the
manifest order through fetching and calibration, reuse the existing
single-series experiment and policy primitives, replay selected policies only
as calibration resubstitution diagnostics, and use one tested analyzer to
validate provenance before computing metrics or gates.

**Tech:** Python 3.12 standard library, existing PyTorch experiment tools,
procedural Python tests, Make, GitButler.

**Delivery boundary:** Four signed local checkpoints only. Do not push, pull,
land, open a PR, commit credentials, or commit generated data/results.

## Locked Research Contract

### Honest universe declaration

- The universe is manually declared and checkpointed on **2026-07-23**.
- `eligibility_date` is **2024-07-22**. It is the Massive reference lookup date
  and the first requested aggregate date, not the date the symbols were chosen.
- Manual hand-selection of recognizable stocks remains selection-biased and
  survivorship-biased. Point-in-time ticker eligibility does not remove either
  bias.
- Generic manifests require unique safe tickers and allow repeated nonempty
  strata.
- The tracked manifest is stricter: exactly 11 tickers and 11 unique strata.
- A provider/data failure stops the experiment. Do not replace a failed or
  poorly performing ticker inside this checkpoint.

### Fixed calibration workload

The single `raw-17` candidate uses five seeds for Transformer and MLP, and one
deterministic run for each other model:

```text
seeded and deterministic runs per fold = 5 + 5 + 1 + 1 + 1 = 13
walk-forward validation fits           = 13 * 2 folds       = 26
final calibration fits                 = 13                 = 13
fits per stock                         = 26 + 13             = 39
total fits                             = 39 * 11 stocks      = 429
```

The live experiment is calibration-only. It emits calibration predictions but
must not emit test predictions or accept a policy authorization.

### Ordering contracts

- Manifest, fetch report, CSV arguments, experiment report, and calibration
  series retain manifest order.
- Policies retain the existing alphabetical series order.
- Calibration replay first validates policy and manifest series sets and
  counts, then constructs the bars mapping in that policy's order.
- The analyzer requires sorted policy series and the same order in each
  policy replay report.

### Forecast-development metrics and gates

For seeded models, first average predicted log returns across the exact five
seeds within each `(stock, as_of, target_time)` group. Do not average losses
from seed records and call that a paired stock/timestamp comparison.

For stock \(i\), timestamp \(t\), model \(m\):

```text
prediction[m,i,t] = arithmetic mean of seed predictions, if seeded
actual[i,t]       = log(close[t + 13] / open[t + 1])
absolute error    = abs(prediction[m,i,t] - actual[i,t])
```

Each stock is one macro unit regardless of its row count:

```text
return_MAE[m,i]       = mean_t absolute_error[m,i,t]
macro_return_MAE[m]   = mean_i return_MAE[m,i]
direction[m,i]        = mean_t sign(prediction[m,i,t]) == sign(actual[i,t])
macro_direction[m]    = mean_i direction[m,i]
```

Derive the majority-sign reference from unique actual calibration targets,
never duplicated seed rows:

```text
p_up[i]   = count(actual > 0) / count(actual)
p_down[i] = count(actual < 0) / count(actual)
p_flat[i] = count(actual = 0) / count(actual)
majority_direction[i] = max(p_up[i], p_down[i], p_flat[i])
macro_majority_direction = mean_i majority_direction[i]
```

Reconstruct close MAE from the executable reference open and predicted return:

```text
predicted_close = open[t + 1] * exp(predicted_log_return)
zero_close      = open[t + 1]
relative_close_improvement[i]
  = (MAE_zero_close[i] - MAE_transformer_close[i]) / MAE_zero_close[i]
```

A zero or nonfinite zero-return denominator is an integrity error. The only
development gates are:

1. Transformer macro return MAE is strictly below each of MLP, linear,
   rolling mean, and last close.
2. Transformer macro direction accuracy is strictly above
   `macro_majority_direction`.
3. Mean per-stock `relative_close_improvement` is strictly greater than zero.

Policy objective, selected action, trade count, signal coverage, execution
coverage, terminal equity, bootstrap intervals, and \(N_\text{eff}\) are never
success criteria.

### Descriptive paired uncertainty

For each baseline, report per-stock paired return-MAE deltas:

```text
delta[baseline,i] = return_MAE[baseline,i] - return_MAE[transformer,i]
wins              = count(delta > 0)
ties              = count(delta == 0)
losses            = count(delta < 0)
```

Also report a deterministic circular moving-date-block bootstrap confidence
interval for the macro paired absolute-error delta:

- align on ordered unique calibration trading dates;
- sample five-date circular blocks with replacement;
- concatenate and truncate to the original date count;
- recompute each stock's paired mean, then the stock macro mean;
- use 10,000 replicates and RNG seed `20260723`;
- report the sorted 2.5% and 97.5% empirical endpoints.

This interval describes date-clustered uncertainty. It is not independent-seed
evidence, is not a promotion claim, and does not change the gates.

### Policy resubstitution diagnostics

Selected-policy `$100` results reuse the same calibration predictions that
selected the policy. Label them **descriptive calibration resubstitution**.
For Transformer, MLP, and linear, report per-series terminal equity for:

- `forecast_long_cash`;
- `cash`;
- `buy_and_hold`;
- `always_up`.

For each strategy report arithmetic and geometric mean terminal equity across
stocks. For `forecast_long_cash`, additionally report excess mean log growth
versus each baseline:

```text
mean_i(log(final_forecast[i] / 100))
  - mean_i(log(final_baseline[i] / 100))
```

Do not project these values into future daily, weekly, monthly, or annual
returns.

### Cross-series effective count

Use aligned daily `forecast_long_cash` strategy returns from one policy replay.
Remove any stock with exactly zero variance, record its name, and form the
sample covariance matrix \(S\) over the remaining stocks:

```text
N_eff = N * trace(S) / (1' S 1)
```

Return `null` plus a deterministic reason when fewer than two aligned dates
remain, fewer than two nonconstant stocks remain, or the denominator is
nonpositive/nonfinite. \(N_\text{eff}\) can exceed \(N\) under diversification.
It is descriptive, never multiplied by temporal effective sample size, and
never used as a gate.

## Final File Map

| Checkpoint | Files | Purpose |
| --- | --- | --- |
| Plan | `docs/superpowers/plans/2026-07-23-universe-expansion.md` | Reviewed execution contract |
| Task 1 | `universes/liquid-common-11.json` | Tracked universe and declaration metadata |
| Task 1 | `experiments/executable-h13-universe.example.json` | Fixed 429-fit sweep |
| Task 1 | `tests/python/test_experiment.py` | Exact tracked manifest/config assertions |
| Task 1/4 | `docs/training.md` | Pre-result gates, then concise evidence |
| Task 2 | `tools/fetch_universe.py` | Frozen-manifest, point-in-time multi-stock fetch |
| Task 2 | `tests/python/test_massive.py` | Fetch, path, mutation, URL, and secret tests |
| Task 3 | `tools/replay_calibration.py` | Tested selected-policy calibration replay |
| Task 3 | `tools/analyze_universe.py` | Provenance validator and metric/gate analyzer |
| Task 3 | `tests/python/test_universe_analysis.py` | Synthetic semantic analyzer/replay tests |
| Task 3 | `Makefile` | Register the standard-library semantic test |
| Task 4 | ignored `reports/h13-universe.*/*` | Live data, ledgers, policies, replays, analysis |

Do not change the C runtime, model architecture, training loop, artifact
schema, prediction ledger schema, policy schema, or reserved test protocol.

## Task 1: Predeclare the Universe, Sweep, and Gates

**Files:**

- Create: `universes/liquid-common-11.json`
- Create: `experiments/executable-h13-universe.example.json`
- Modify: `tests/python/test_experiment.py`
- Modify: `docs/training.md`

### Step 1: Add failing tracked-contract tests

Add a small `verify_universe_contract(root: Path)` test helper that:

- loads both JSON files;
- requires manifest keys exactly
  `schema`, `purpose`, `declared_on`, `eligibility_date`, `start`, `end`,
  `interval_minutes`, `adjusted`, `session`, and `series`;
- asserts `declared_on == "2026-07-23"`;
- asserts `eligibility_date == start == "2024-07-22"`;
- asserts 11 unique safe tickers in the exact order below;
- asserts 11 unique nonempty strata for this tracked manifest;
- decodes the sweep through `Sweep.read`;
- asserts one `raw-17`/`ohlcv` candidate, horizons 13/13,
  `executable-return-v1`, two folds, five exact seeds, and five exact models;
- asserts `expected_runs(sweep, 11) == 429`.

Call the helper from the existing test entry point. Keep the generic repeated
stratum behavior in `test_massive.py`; do not encode it in the tracked-file
test.

Run:
```sh
TORCH_PYTHON=(/Users/Enkang.Yuan1/.local/bin/uv run \
  --offline --with torch python)
"${TORCH_PYTHON[@]}" tests/python/test_experiment.py
```

Expected: fail because the two tracked JSON files do not exist.

### Step 2: Add the exact tracked manifest

Create `universes/liquid-common-11.json`:

```json
{
  "adjusted": true,
  "declared_on": "2026-07-23",
  "eligibility_date": "2024-07-22",
  "end": "2026-07-21",
  "interval_minutes": 30,
  "purpose": "Benchmark the fixed horizon-13 raw-17 stack across one manually selected U.S. common stock per sector stratum.",
  "schema": 1,
  "series": [
    {"stratum": "information-technology", "ticker": "AAPL"},
    {"stratum": "financials", "ticker": "JPM"},
    {"stratum": "energy", "ticker": "XOM"},
    {"stratum": "health-care", "ticker": "JNJ"},
    {"stratum": "consumer-staples", "ticker": "PG"},
    {"stratum": "consumer-discretionary", "ticker": "AMZN"},
    {"stratum": "industrials", "ticker": "CAT"},
    {"stratum": "utilities", "ticker": "NEE"},
    {"stratum": "materials", "ticker": "LIN"},
    {"stratum": "real-estate", "ticker": "PLD"},
    {"stratum": "communication-services", "ticker": "GOOGL"}
  ],
  "session": "regular",
  "start": "2024-07-22"
}
```

Order is contractual. Do not reorder or substitute symbols after seeing data
or metrics.

### Step 3: Add the exact sweep

Create `experiments/executable-h13-universe.example.json`:

```json
{
  "alignment_horizon_bars": 13,
  "batch_size": 128,
  "candidates": [{
    "feature_set": "ohlcv", "ff_dim": 32, "heads": 2, "layers": 1,
    "learning_rate": 0.0003, "mlp_dim": 32, "model_dim": 16,
    "name": "raw-17", "ridge": 0.001, "rolling_window": 8, "seq_len": 17,
    "weight_decay": 0.0001
  }],
  "epochs": 100,
  "fold_fraction": 0.1,
  "folds": 2,
  "models": ["transformer", "linear", "mlp", "rolling_mean", "last_close"],
  "patience": 10,
  "seeds": [7, 19, 31, 43, 61],
  "target_horizon_bars": 13,
  "target_kind": "executable-return-v1"
}
```

Do not add candidates or tune values in this universe-only experiment.

### Step 4: Document the pre-result protocol

Add a short pre-result subsection to `docs/training.md` that states:

- declaration/checkpoint date, eligibility/reference date, and both biases;
- exact 11-stock manifest and 429-fit calibration-only command shape;
- the three forecast-development gates verbatim;
- policy results, bootstrap CI, and \(N_\text{eff}\) are descriptive only;
- no reserved test labels are opened;
- any threshold change requires a new plan before rerunning analysis.

Do not include a result placeholder that could be mistaken for evidence.

### Step 5: Run focused tests and checkpoint

Run:
```sh
"${TORCH_PYTHON[@]}" tests/python/test_experiment.py
```

Expected: pass, including `expected_runs(..., 11) == 429`.

Run `but diff`, select only the four Task 1 files, and create:

```sh
but commit enkyuan/universe-expansion \
  -c -m "feat(training): define common-stock universe benchmark" \
  --changes <task-1-change-ids>
```

Stack `enkyuan/universe-expansion` directly above
`enkyuan/universe-expansion-plan`. Verify the returned commit signature before
Task 2.

## Task 2: Build the Strict Universe Fetcher

**Files:**

- Create: `tools/fetch_universe.py`
- Modify: `tests/python/test_massive.py`

### Step 1: Lock the small public surface in tests

Use the existing dependency-injection style. Lock:

```text
SeriesSpec(ticker: str, stratum: str)
UniverseManifest(
    schema, purpose, declared_on, eligibility_date, start, end,
    interval_minutes, adjusted, session, series
)
UniverseManifest.read(path: Path) -> UniverseManifest
reference_url(ticker: str, eligibility_date: date) -> str
fetch_universe(
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    key: str | None = None,
    requester: Requester = request_json,
) -> dict[str, object]
```

CLI:

```text
python tools/fetch_universe.py MANIFEST OUTPUT_DIR REPORT
```

The real CLI reads `ROOT / ".env"` with the existing `api_key()` helper. The
report must never contain an authenticated URL or credential.

### Step 2: Add red manifest and no-request path tests

Add tests for:

- required exact top-level and per-series fields;
- valid ISO dates, `declared_on >= eligibility_date`, and
  `eligibility_date == start <= end`;
- interval 1..59, adjusted boolean, regular session, and nonempty text;
- safe unique tickers, while repeated generic strata are accepted;
- the tracked manifest still has 11 unique strata;
- missing, malformed, symlinked, or non-regular manifest rejected.

Before the key or requester can run, require all final targets to be absent,
including broken symlinks:

- `output_dir`, `report_path`, and each derived `<ticker-lower>-30m.csv`;
- equality or either ancestor/descendant relationship between normalized
  `output_dir` and `report_path`.

Use `os.path.lexists()` for broken links and `Path.resolve(strict=False)` plus
parent membership for nesting. Patch `api_key` and `requester` to fail if
reached. Cover existing/symlinked/broken output or report paths, equal paths,
both nesting directions, and a derived CSV collision.

Run:
```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: fail because `tools/fetch_universe.py` does not exist.

### Step 3: Add reference and aggregate request tests

For every manifest entry, require this reference request:

```text
GET https://api.massive.com/v3/reference/tickers/{ticker}
    ?date={eligibility_date}
```

Validate response status and one result with exact requested ticker,
`active == true`, `market == "stocks"`, `locale == "us"`, `type == "CS"`,
and `currency_name == "usd"`.

Capture every actual authorized request in the fake requester. For the first
aggregate request per ticker:

- strip only `apiKey`;
- assert its URL path equals the report's aggregate path;
- parse and assert its query equals the report query exactly;
- require adjusted, ascending sort, and limit 50000;
- require the manifest date range and interval in the path;
- assert no report string contains `apiKey`, `MASSIVE_API_KEY`, or the fake key.

Reuse, rather than duplicate, `aggregate_url`, `fetch_bars`, `regular_bars`,
`write_csv`, `read_csv`, and `file_sha256`.

### Step 4: Add mutation and atomic-report tests

Add deterministic injected-mutation tests:

1. Mutate the source manifest after its frozen snapshot is parsed and the first
   request starts. Expect failure before a final report.
2. Mutate the first written CSV after its original hash is recorded but before
   final verification. Expect failure before a final report.

Also assert:

- the manifest bytes are frozen before parsing or fetching;
- the report records the frozen manifest SHA-256;
- manifest order is retained in the report;
- each CSV is read back through the strict parser;
- every reported row count and SHA-256 matches the final file;
- source manifest and all final CSV hashes are rechecked immediately before
  atomic report replacement;
- a failed run never creates the final report;
- pagination cycles, untrusted hosts, malformed bars, and provider failures
  still fail closed through existing helpers.

### Step 5: Implement the minimum fetch flow

Keep orchestration linear:

```text
validate target paths without reading the key
freeze manifest bytes
parse frozen manifest
derive and revalidate all CSV targets
read key if not injected
for each entry in manifest order:
    validate point-in-time reference identity
    build sanitized aggregate request contract
    fetch/paginate, regular-session filter, atomic CSV write/readback
    record row/session counts and CSV hash
verify original manifest still matches frozen hash
verify every final CSV still matches its recorded hash
atomically write report
```

The report schema is intentionally narrow:

```text
schema, purpose, declared_on, eligibility_date, start, end,
interval_minutes, adjusted, session,
manifest {path, sha256},
series [{ticker, stratum,
         reference {path, query, active, market, locale, type, currency_name},
         aggregate {path, query},
         csv {path, rows, sessions, source_rows, sha256}}]
```

Store query objects, not URL strings. Never store headers, `apiKey`, next-page
URLs, or raw provider payloads.

Implementation references:

- [Massive ticker overview](https://massive.com/docs/rest/stocks/tickers/ticker-overview)
- [Massive custom bars](https://massive.com/docs/rest/stocks/aggregates/custom-bars)

### Step 6: Run checks and amend the Task 2 checkpoint

Run:
```sh
"$PRIMARY_PYTHON" tests/python/test_massive.py
make -B PYTHON="$PRIMARY_PYTHON" check
```

Expected: all universe fetch tests, existing Massive tests, nine C suites, and
standard Python suites pass.

Run `but diff`; commit only the fetcher and its tests onto
`enkyuan/universe-expansion`:

```sh
but commit enkyuan/universe-expansion \
  -m "feat(data): fetch point-in-time stock universes" \
  --changes <task-2-change-ids>
```

Do not include generated CSVs, `.env`, or reports.

## Task 3: Add Tested Replay and Analysis Tools

**Files:**

- Create: `tools/replay_calibration.py`
- Create: `tools/analyze_universe.py`
- Create: `tests/python/test_universe_analysis.py`
- Modify: `Makefile`

No live request or 429-fit run occurs in this task.

### Step 1: Register a failing semantic test

Add `test_universe_analysis.py` to `PYTHON_TEST`, not `PYTORCH_TEST`; its
fixtures are tiny JSON/JSONL/CSV files and do not train models.

The procedural test must create a complete synthetic run directory:

```text
csv/<ticker-lower>-30m.csv
fetch-report.json
experiment.json
calibration.jsonl
policy-{transformer,mlp,linear}.json
backtest-{transformer,mlp,linear}.json
```

Run:
```sh
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
```

Expected: fail because both tools are absent.

### Step 2: Define the calibration replay CLI

Use:

```text
python tools/replay_calibration.py \
  MANIFEST RUN_DIR MODEL OUTPUT
```

Support only `transformer`, `mlp`, and `linear`. The tool must:

- require a fresh, absent, non-symlink output;
- freeze manifest, fetch report, experiment, calibration ledger, chosen policy,
  and all CSVs before parsing;
- validate hashes and the experiment/policy calibration contract;
- require manifest/fetch/experiment order equality;
- require `policy["series"] == sorted(manifest_tickers)`;
- compare series sets and counts before creating bars;
- create the bars mapping in policy order;
- select exact model/candidate/feature/horizon calibration predictions;
- ensemble seeded predictions with the policy's exact seed set;
- replay the selected safety/disagreement/cash action with existing
  `run_backtests`;
- preserve `forecast_long_cash`, `cash`, `buy_and_hold`, and `always_up`;
- label the result `descriptive-calibration-resubstitution`;
- add direct `{path, sha256}` provenance for policy and experiment, plus ledger
  path/hash/source/selected record counts;
- verify all frozen sources immediately before atomic output.

Do not weaken `tools/backtest.py` policy mode or disguise calibration records as
test records.

### Step 3: Define the analyzer CLI and integrity boundary

Use:

```text
python tools/analyze_universe.py \
  MANIFEST CONFIG RUN_DIR OUTPUT
```

The output must be fresh, absent, non-symlinked, and disjoint from every input.
Freeze all required files once, parse only snapshots, verify original sources
again immediately before atomic output.

Before metrics, validate:

- manifest and config hashes against fetch and experiment provenance;
- exact manifest/fetch/experiment series order and 11 unique tickers/strata;
- 429-run protocol, raw-17, horizons, target kind, folds, models, and seeds;
- every CSV path, hash, row count, chronology, and report request contract;
- calibration ledger schema/hash/record count and exact prediction grid;
- no test records and an empty experiment test result;
- every policy's sorted series and direct calibration report/ledger hashes;
- each policy's `test_grid`, `model_fingerprints`, and
  `calibration_fingerprint` exactly equal the corresponding experiment
  contract, not merely the same sets;
- each replay's direct policy path/hash and experiment path/hash;
- each replay's series/result order equals that policy order;
- exact `$100` initial cash and costs `spread=1`, `slippage=1`, `fee=0`;
- all numeric inputs are finite and all expected timestamps align to CSV bars.

Any missing, extra, stale, reordered, aliased, mutated, or mismatched artifact
is an integrity error and must exit with a code other than 0 or 3 without
writing a final analysis.

### Step 4: Lock analyzer semantics with synthetic tests

Create small fixtures whose expected values are hand-computable. Assert:

- seed averaging precedes stock/timestamp pairing and stocks receive equal
  macro weight;
- unique actuals produce sign proportions/majority, and every gate is strict;
- paired stock deltas/counts and seeded date-block bootstrap are exact;
- bootstrap and complete four-strategy policy summaries never affect status;
- covariance \(N_\text{eff}\) matches a hand calculation, names zero-variance
  exclusions, returns every null reason, permits \(N_\text{eff}>N\), and has no
  temporal ESS field.

Mutation tests independently alter each input hash; policy order, test grid,
fingerprints, and calibration fingerprint; replay policy/experiment
provenance and result order; and a prediction/actual timestamp.

Each mutation must fail as integrity failure, not valid gate failure.

### Step 5: Emit a compact, fresh analysis report

Keep implementation decomposed into validation, reconstruction, metrics, and
serialization helpers. The report needs only:

```text
schema, status, inputs {all direct paths and hashes},
protocol {ordering, seed_ensemble, macro_unit, majority_reference,
          bootstrap, policy_evidence, n_eff_formula},
forecast {per_model macro return MAE/direction, per_stock metrics,
          majority p_up/p_down/p_flat, close relative improvement,
          paired deltas/counts, date-block bootstrap CI},
policy_resubstitution {four terminal equities, arithmetic/geometric
                       aggregates, forecast excess mean log growth},
n_eff {value or null, included, excluded, reason},
gates {return_mae, direction, close_mae, all_pass}
```

Exit `0` when the report is valid and all gates pass. Exit `3` when the report
is valid and at least one gate fails. All other nonzero exits are integrity or
runtime failures.

### Step 6: Run focused and aggregate checks

Run:
```sh
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
"$PRIMARY_PYTHON" tests/python/test_massive.py
make -B PYTHON="$PRIMARY_PYTHON" check
TORCH_PYTHON=(/Users/Enkang.Yuan1/.local/bin/uv run \
  --offline --with torch python)
make PYTHON="${TORCH_PYTHON[*]}" check-training
```

Expected: semantic, fetch, standard, C, and optional PyTorch suites pass.
Run `but diff`; commit only the four Task 3 files onto
`enkyuan/universe-expansion`:

```sh
but commit enkyuan/universe-expansion \
  -m "feat(training): analyze common-stock calibration" \
  --changes <task-3-change-ids>
```

Verify the signed checkpoint before any live provider request.
## Task 4: Run One Live Calibration and Record Evidence

**Files:**

- Modify: `docs/training.md`
- Create only ignored files under one fresh `reports/h13-universe.*` directory

This task requires the real `MASSIVE_API_KEY` in the process or ignored `.env`.
Never print, echo, interpolate into a command trace, or commit the key.

### Step 1: Confirm ignored output boundaries

Run:
```zsh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
TORCH_PYTHON=(/Users/Enkang.Yuan1/.local/bin/uv run \
  --offline --with torch python)
RUN_DIR=$(mktemp -d reports/h13-universe.XXXXXX)
CSV_DIR="$RUN_DIR/csv"
FETCH_REPORT="$RUN_DIR/fetch-report.json"
```

Confirm `.gitignore` already excludes `reports/`, `data/`, `models/`, `.env`,
and bytecode. If an exclusion is missing, stop and create a separate reviewed
checkpoint before fetching.

### Step 2: Fetch exactly once

Run:
```zsh
"$PRIMARY_PYTHON" tools/fetch_universe.py \
  universes/liquid-common-11.json "$CSV_DIR" "$FETCH_REPORT"
```

Do not retry into the same paths. A provider failure records no final report
and ends this experiment attempt.

Perform a real-key no-leak check without printing the key:

```zsh
"$PRIMARY_PYTHON" -c '
import pathlib, sys
from tools.fetch_massive import api_key

root = pathlib.Path(sys.argv[1])
secret = api_key(pathlib.Path(".env")).encode()
leaked = any(secret in path.read_bytes() for path in root.rglob("*")
             if path.is_file())
print("credential scan:", "fail" if leaked else "pass")
raise SystemExit(1 if leaked else 0)
' "$RUN_DIR"
```

This prints only `pass` or `fail`, never the secret or a matching line. A
failure stops before training.

### Step 3: Run the one 429-fit calibration

Build `SERIES_ARGS` from the manifest in its declared order and assert each
path/hash matches the fetch report. Then run:

```zsh
"${TORCH_PYTHON[@]}" tools/experiment.py \
  experiments/executable-h13-universe.example.json \
  "$RUN_DIR/experiment.json" \
  "${SERIES_ARGS[@]}" \
  --device cpu \
  --calibration-only \
  --calibration-predictions "$RUN_DIR/calibration.jsonl" \
  --max-runs 429
```

Do not pass `--predictions` or `--policy`. Confirm the resulting report says
`run_count == 429`, phase `selection-and-calibration`, and has no test results.

### Step 4: Select and replay three policies

For each model, pass CSVs in manifest order to policy selection; the policy
itself must serialize alphabetically sorted series:

```zsh
for model in transformer mlp linear; do
  disagreement=(0)
  [[ "$model" == transformer || "$model" == mlp ]] &&
    disagreement=(0 0.5 1)

  "${TORCH_PYTHON[@]}" tools/select_policy.py \
    "$RUN_DIR/experiment.json" \
    "$RUN_DIR/calibration.jsonl" \
    "$RUN_DIR/policy-$model.json" \
    "${SERIES_ARGS[@]}" \
    --model "$model" \
    --safety-bps 0 3 6 10 \
    --disagreement-lambda "${disagreement[@]}" \
    --initial-cash 100 \
    --spread-bps 1 --slippage-bps 1 --fee-bps 0

  "$PRIMARY_PYTHON" tools/replay_calibration.py \
    universes/liquid-common-11.json \
    "$RUN_DIR" "$model" "$RUN_DIR/backtest-$model.json"
done
```

These are descriptive resubstitution replays. Do not treat a profitable policy
as independent evidence.

### Step 5: Invoke the analyzer exactly once

The shell accepts only the two valid-analysis exits, then checks status:

```zsh
set +e
"$PRIMARY_PYTHON" tools/analyze_universe.py \
  universes/liquid-common-11.json \
  experiments/executable-h13-universe.example.json \
  "$RUN_DIR" "$RUN_DIR/analysis.json"
analysis_exit=$?
set -e

case "$analysis_exit" in
  0) expected_status=pass ;;
  3) expected_status=gate-failure ;;
  *) echo "analysis integrity failure" >&2; exit "$analysis_exit" ;;
esac

"$PRIMARY_PYTHON" -c '
import json, pathlib, sys
actual = json.loads(pathlib.Path(sys.argv[1]).read_text())["status"]
raise SystemExit(0 if actual == sys.argv[2] else
                 "analysis exit/status mismatch")
' "$RUN_DIR/analysis.json" "$expected_status"
```

Do not rerun the analyzer after observing the outcome. Any integrity failure
invalidates the run; any gate failure is a valid negative result.

### Step 6: Record concise evidence only

Append to `docs/training.md`:

- run date and direct SHA-256 values for manifest, config, fetch report,
  experiment, ledger, three policies, three replays, and analysis;
- exact 429-fit command contract and analyzer exit/status;
- three gate values and pass/fail outcomes without threshold changes;
- per-model macro return MAE/direction;
- majority-sign `p_up/p_down/p_flat` summary;
- mean close-MAE relative improvement;
- paired per-stock delta win/tie/loss counts and descriptive bootstrap CIs;
- descriptive policy terminal-equity aggregates and excess mean log growth;
- descriptive \(N_\text{eff}\), exclusions, or null reason;
- explicit language that policy results are calibration resubstitution,
  bootstrap is date-clustered uncertainty, and manual universe construction is
  selection/survivorship biased;
- confirmation that no reserved historical test labels were opened.

Do not commit generated reports. Documentation is a transcription of the one
validated analysis, not a second analysis.

### Step 7: Run final checks and create docs-only evidence checkpoint

Run:
```zsh
"$PRIMARY_PYTHON" tests/python/test_massive.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
"${TORCH_PYTHON[@]}" tests/python/test_experiment.py
make -B PYTHON="$PRIMARY_PYTHON" check
make PYTHON="${TORCH_PYTHON[*]}" check-training
```

Run `but diff`. Confirm no `.env`, `reports/`, CSV, ledger, policy, model,
analysis, or bytecode file appears. Commit only `docs/training.md`:

```sh
but commit enkyuan/universe-expansion \
  -m "docs(training): record common-stock calibration" \
  --changes <training-doc-change-id>
```

## Final Review Checklist

- [ ] Plan is at most 900 lines and has no embedded analyzer implementation.
- [ ] Declaration/eligibility dates and manual selection/survivorship bias are
  explicit; generic strata may repeat, but tracked tickers/strata are 11/11.
- [ ] All target collisions and symlinks fail before key/network; frozen
  manifest/CSV mutation prevents the report; request/report contracts match.
- [ ] The real-key scan emits only pass/fail.
- [ ] Manifest order reaches experiment; sorted policy order reaches replay.
- [ ] The registered semantic test covers every hash, exact policy contract,
  seed averaging, descriptive bootstrap/policy metrics, and covariance
  \(N_\text{eff}\), including null and \(N_\text{eff}>N\) cases.
- [ ] Only strict return-MAE, direction, and relative close-MAE gates determine
  0/pass versus 3/gate-failure; other nonzero exits mean integrity failure.
- [ ] Task order and all four signed local checkpoint boundaries were followed.
- [ ] No push, pull, landing, PR, credential, or generated artifact occurred.

Expected local history:

1. Plan branch: `docs(training): plan common-stock universe expansion`.
2. Implementation branch: define benchmark; fetch universes; analyze
   calibration; record calibration.

After each checkpoint, verify:

```sh
git verify-commit <sha>
git show -s \
  --format='commit=%H%nparent=%P%nauthor=%an <%ae>%ncommitter=%cn <%ce>' \
  <sha>
```

Require author/committer `enkyuan <yuan.enkng@gmail.com>` and:

```text
Good "git" signature for enkyuan with ED25519 key
SHA256:mnKfNNJJUV5ELrMXZCwdu6v/PusXBeAi8tWUnRXmWMQ
```

Report each SHA, parent, checks, clean workspace, and `push/landing: none`.
