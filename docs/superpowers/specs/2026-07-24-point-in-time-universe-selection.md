# Point-in-Time Massive Universe Selection Specification

This specification fixes the research boundary, mathematical definitions, and
implementation roadmap. Executable plans under `docs/superpowers/plans/`
translate one independently reviewable subproject at a time into exact code
steps.

**Goal:** Replace the manually chosen 11-stock research universe with a
deterministic, point-in-time Massive selection that emits nested
11/22/33/55-stock cohorts without inspecting model outcomes.

**Architecture:** Freeze a tracked selection policy, obtain dated common-stock
reference data and pre-model liquidity observations from Massive, reduce them
with decimal arithmetic into one deterministic master order, and emit
hash-bound schema-1 fetch manifests plus an audit report. Selection remains
separate from fetching minute bars, training, promotion, and backtesting.

**Tech:** Python 3.12 standard library, existing Massive request and file
primitives, procedural Python tests, Make, GitButler.

**Delivery boundary:** Signed local checkpoints only. Do not push, pull, land,
open a PR, commit credentials, or commit generated source archives, datasets,
models, and reports. Do not modify the existing conflicted
`enkyuan/universe-expansion` branch.

## Why This Is a Separate Subproject

The existing 11 names were manually selected. Dated ticker validation proves
that each symbol existed, but it cannot remove selection or survivorship bias.
This plan changes only how membership is declared.

The next subproject will generalize calibration analysis to arbitrary manifest
sizes. It must not be mixed into this selector because structural run accounting
and scientific promotion policy are separate contracts.

The historical anchor below predates this policy declaration. The resulting
study is computationally leakage-safe, but remains retrospective research
rather than confirmatory evidence. A future confirmatory run requires
externally timestamping the same policy before its model interval begins.

## Frozen Selection Policy

Create `universes/liquid-common-ladder.example.json` with exactly:

```json
{
  "adjusted": true,
  "anchor_date": "2024-10-31",
  "cohort_sizes": [11, 22, 33, 55],
  "declared_on": "2026-07-24",
  "end": "2026-07-21",
  "formation_end": "2024-10-31",
  "formation_start": "2024-08-01",
  "interval_minutes": 30,
  "liquidity_strata": 5,
  "minimum_coverage": "0.95",
  "minimum_formation_sessions": 60,
  "minimum_median_close_usd": "5",
  "minimum_median_dollar_volume_usd": "25000000",
  "primary_cohort_size": 55,
  "purpose": "Select nested point-in-time U.S. common-stock cohorts before fetching model data.",
  "schema": 1,
  "selection_seed": "compose-mini-massive-universe-v1",
  "session": "regular",
  "start": "2024-11-01"
}
```

The three-month formation interval keeps the example executable inside
Massive's current two-year entry-tier history while reserving all model rows
strictly after the anchor. The parser must support other valid dates and
thresholds; these exact values are the tracked research declaration.

`primary_cohort_size=55` is fixed before results because the largest cohort is
the intended robustness estimand. Smaller prefixes are scaling diagnostics,
not alternate opportunities to pass.

## Massive Source Contract

Fetch all dated candidates through every trusted-host page:

```text
GET /v3/reference/tickers
  ?active=true
  &date=2024-10-31
  &limit=1000
  &market=stocks
  &order=asc
  &sort=ticker
  &type=CS
```

Fetch one unadjusted, non-OTC daily market summary for each weekday in the
formation interval:

```text
GET /v2/aggs/grouped/locale/us/market/stocks/{date}
  ?adjusted=false
  &include_otc=false
```

The API key is attached only after `authorized_url()` verifies
`api.massive.com`. Stored request contracts and pagination links must never
contain `apiKey`.

Require:

- status `OK`;
- no pagination cycle;
- strictly increasing, unique tickers across reference pages;
- one successful response for every requested weekday;
- at least 60 successful, nonempty formation sessions;
- unique ticker rows within each formation session.

Archive only canonical fields used by selection. Ticker archives retain
`ticker`, `active`, `market`, `locale`, `type`, `currency_name`,
`primary_exchange`, `composite_figi`, and `share_class_figi`. Daily archives
retain `ticker`, `c`, `v`, and `vw`. Decimal values are stored as normalized
strings. Every archive is fresh, canonical JSON with a SHA-256 recorded in the
final report.

## Eligibility and Selection Math

Use `Decimal(str(value))` for every selection calculation. The existing JSON
transport first decodes provider numbers as binary64; this contract avoids any
additional binary arithmetic but does not claim byte-exact recovery of the
provider's JSON decimal token.

For each dated reference candidate, require:

```text
active == true
market == "stocks"
locale == "us"
type == "CS"
currency_name == "usd"
primary_exchange in {"XNAS", "XNYS", "XASE"}
composite_figi is nonempty
share_class_figi is nonempty
```

For a candidate \(i\) over the \(D\) observed formation sessions:

```text
observed_i       = count(days with finite positive c, v, and vw)
coverage_i       = observed_i / D
dollar_volume_it = volume_it * vwap_it
median_close_i   = median(close_it)
median_dv_i      = median(dollar_volume_it)
```

Keep the candidate only when:

```text
coverage_i     >= 0.95
median_close_i >= $5
median_dv_i    >= $25,000,000
```

Missing observations are absent, never converted to zero. No forward filling,
interpolation, replacement, or result-aware substitution is permitted.

Use `share_class_figi` as the security identity. If multiple eligible listings
share it, keep the listing minimizing:

```text
(-median_dv, composite_figi, ticker)
```

Rank the remaining \(M\) representatives globally by:

```text
(-median_dv, share_class_figi, composite_figi, ticker)
```

For zero-based liquidity rank \(r\) and \(K=5\) strata:

```text
stratum(r) = floor(K * r / M) + 1
```

Within each stratum order by:

```text
(
  sha256(selection_seed + "\0" + share_class_figi),
  composite_figi,
  ticker
)
```

Build the master list by round-robin traversal of strata 1 through 5. Every
declared cohort is an exact prefix. Therefore, for a prefix of size \(N\), each
stratum contributes either \(\lfloor N/5\rfloor\) or
\(\lceil N/5\rceil\) members. Fail if any stratum cannot supply the largest
prefix.

The hash order prevents recognizable-name hand selection. The liquidity
stratification prevents the smallest prefix from becoming only mega-cap names.

## Outputs and Integrity Boundary

`tools/select_universe.py POLICY OUTPUT_DIR` creates a fresh output directory:

```text
OUTPUT_DIR/
  selection.json
  sources/
    tickers-0001.json
    ...
    daily-YYYY-MM-DD.json
  manifests/
    liquid-common-11.json
    liquid-common-22.json
    liquid-common-33.json
    liquid-common-55.json
```

Each generated fetch manifest remains schema 1 for compatibility with
`tools/fetch_universe.py`:

```json
{
  "adjusted": true,
  "declared_on": "2026-07-24",
  "eligibility_date": "2024-10-31",
  "end": "2026-07-21",
  "interval_minutes": 30,
  "purpose": "...",
  "schema": 1,
  "series": [{"stratum": "liquidity-1", "ticker": "..."}],
  "session": "regular",
  "start": "2024-11-01"
}
```

The selection report binds:

- frozen policy path and SHA-256;
- the selector and its local request/file dependency paths and SHA-256 values;
- sanitized request contracts;
- each canonical source path, SHA-256, and record count;
- total formation sessions;
- every candidate's identity, statistics, decision, and ordered rejection
  reasons;
- share-class deduplication decisions;
- liquidity rank, stratum, within-stratum rank, and master rank;
- master-list SHA-256;
- each cohort's ordered members, member-list SHA-256, manifest path, and
  manifest SHA-256.

Require `cohort_N == master[:N]` before writing the final report. Recheck the
policy, local source closure, and every output hash immediately before
publishing `selection.json`. The report is the completion marker; a partial
directory without it is invalid.

`fetch_universe.py` currently conflates eligibility date with aggregate start.
Relax only that relationship from `eligibility_date == start` to
`eligibility_date <= start`; retain all other date checks and schema-1
compatibility.

## Scaling and Training Math

Massive can paginate far beyond 55 listings. Fifty-five is a controlled first
expansion, not an API ceiling.

The current local-model sweep performs:

```text
26 validation fits per stock + 13 calibration fits per stock = 39N fits
```

Thus the four cohort workloads would be:

```text
N=11:  429 fits
N=22:  858 fits
N=33: 1,287 fits
N=55: 2,145 fits
```

Because the present fixed sweep contains only the `raw-17` candidate and fits
each stock independently, one 55-stock ledger can be filtered exactly into all
four prefixes. Running four overlapping experiments would perform 4,719 fits,
of which 2,574 are redundant:

```text
1 - 2,145 / 4,719 = 54.5% avoided work
```

This reuse ceases to be valid if a later sweep selects among candidates using
prefix-level aggregate metrics or if a global model pools stocks during
training. The former must freeze the 55-stock winner for every diagnostic
prefix or perform prefix-specific calibration; the latter must be trained
separately for each declared cohort.

Adding stocks improves breadth, not guaranteed predictability. Under equal
stock weights, macro loss is:

```text
L_N = (1 / N) * sum_i mean_t |prediction_it - actual_it|
```

If stock errors have average cross-correlation \(\bar{\rho}\), a useful
equal-correlation approximation is:

```text
N_eff ~= N / (1 + (N - 1) * rho_bar)
SE(L_N) scales as 1 / sqrt(N_eff)
```

At \(N=55\), \(\bar{\rho}=0.20\) implies \(N_eff\approx4.6\), not 55. This is
why the report must retain per-stock losses, correlation diagnostics, and
date-block uncertainty rather than treating every ticker as independent.

No `$100` return may be reported from this selection layer. Reserved-test
trading remains closed until the later generalized calibration gate passes.

## Final File Map

| Checkpoint | Files | Purpose |
| --- | --- | --- |
| Plan | `docs/superpowers/plans/2026-07-24-point-in-time-universe-selection.md` | Frozen contract and tasks |
| Policy | `universes/liquid-common-ladder.example.json` | Pre-result dates, thresholds, cohorts, and seed |
| Selector | `tools/select_universe.py` | Massive retrieval, decimal selection, provenance, and manifests |
| Fetch compatibility | `tools/fetch_universe.py` | Permit eligibility before aggregate start |
| Tests | `tests/python/test_massive.py` | Policy, math, source, secret, output, and compatibility tests |

Do not modify `tools/experiment.py`, panel code, the C runtime, model
architecture, loss, split, prediction ledger, policy schema, analyzer, reserved
test protocol, `Makefile`, or `docs/training.md` in this subproject.

## Task 1: Lock Policy Parsing and Pure Selection

**Files:**

- Create: `universes/liquid-common-ladder.example.json`
- Create: `tools/select_universe.py`
- Modify: `tests/python/test_massive.py`

### Interfaces

Implement these immutable value boundaries:

```python
@dataclass(frozen=True)
class Reference:
    ticker: str
    active: bool | None
    market: str | None
    locale: str | None
    type: str | None
    currency_name: str | None
    primary_exchange: str | None
    composite_figi: str | None
    share_class_figi: str | None


@dataclass(frozen=True)
class DailyRow:
    ticker: str
    close: Decimal
    volume: Decimal
    vwap: Decimal


@dataclass(frozen=True)
class Candidate:
    reference: Reference
    observed: int
    coverage: Decimal
    median_close: Decimal
    median_dollar_volume: Decimal
    rejection_reasons: tuple[str, ...]
    liquidity_rank: int | None = None
    stratum: int | None = None
    within_stratum_rank: int | None = None
    master_rank: int | None = None


@dataclass(frozen=True)
class Selection:
    candidates: tuple[Candidate, ...]
    master: tuple[Candidate, ...]


@dataclass(frozen=True)
class SelectionPolicy:
    schema: int
    purpose: str
    declared_on: date
    anchor_date: date
    formation_start: date
    formation_end: date
    start: date
    end: date
    interval_minutes: int
    adjusted: bool
    session: str
    cohort_sizes: tuple[int, ...]
    primary_cohort_size: int
    selection_seed: str
    liquidity_strata: int
    minimum_formation_sessions: int
    minimum_coverage: Decimal
    minimum_median_close_usd: Decimal
    minimum_median_dollar_volume_usd: Decimal

    @classmethod
    def read(cls, path: Path) -> SelectionPolicy: ...


def select_candidates(
    policy: SelectionPolicy,
    references: Sequence[Reference],
    sessions: Sequence[tuple[date, Sequence[DailyRow]]],
) -> Selection: ...
```

`candidates` is ticker-sorted and contains every dated reference record exactly
once, including rejected and FIGI-deduplicated records. `master` contains only
representatives, in final cohort order. A deduplicated listing uses rejection
reason `duplicate-share-class-figi`; all reason strings and their evaluation
order are constants in the module.

### Step 1: Add failing policy tests

Add tests that:

- decode the exact tracked policy;
- reject missing, extra, and duplicate fields;
- reject booleans where integers are required;
- reject noncanonical dates and decimal strings;
- require
  `formation_start <= formation_end == anchor_date < start <= end`;
- require a declaration date on or after the anchor;
- require positive, strictly increasing, unique cohort sizes;
- require `primary_cohort_size == cohort_sizes[-1]`;
- require `2 <= liquidity_strata <= cohort_sizes[-1]`;
- require thresholds in their valid ranges.

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected: fail because the policy parser does not exist.

### Step 2: Implement the smallest strict policy object

Add an immutable `SelectionPolicy` dataclass with one `read()` constructor.
Keep one small local duplicate-key hook; do not import the private hook from
`fetch_universe.py`. Reuse the existing ticker pattern, frozen-input, hashing,
JSON, API-key, trusted-host, and request-gate primitives rather than creating
parallel infrastructure.

Keep decimal thresholds as `Decimal`. Convert them to strings only at JSON
boundaries.

### Step 3: Add failing pure-selection tests

Build a tiny synthetic universe spanning five liquidity bands. Test:

- exact coverage, median, and dollar-volume calculations;
- eligibility and ordered rejection reasons;
- exclusion of missing/nonpositive/nonfinite observations;
- share-class FIGI deduplication;
- deterministic hash order;
- round-robin master order;
- exact 11/22/33/55 prefixes;
- per-prefix stratum counts differing by at most one;
- identical output under reordered provider input;
- failure before partial selection when fewer than 55 candidates qualify.

### Step 4: Implement pure selection

Use small pure helpers for:

- canonical decimal parsing;
- median and coverage;
- metadata eligibility;
- FIGI representative selection;
- liquidity quantiles;
- seeded within-stratum order;
- round-robin prefix construction.

No helper may read environment variables, files, clocks, or network state.

### Step 5: Run the focused test

Use the command from Step 1. Expected: pass.

### Step 6: Review and checkpoint

Run an independent spec review and code-quality review. Amend only findings
that belong to this task, rerun the focused test, and create one signed local
GitButler checkpoint:

```text
feat(data): define point-in-time universe selection
```

## Task 2: Add Massive Retrieval and Canonical Provenance

**Files:**

- Modify: `tools/select_universe.py`
- Modify: `tests/python/test_massive.py`

### Interfaces

Add:

```python
@dataclass(frozen=True)
class SourceArchive:
    name: str
    request: Mapping[str, object]
    records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SourceBundle:
    references: tuple[Reference, ...]
    sessions: tuple[tuple[date, tuple[DailyRow, ...]], ...]
    archives: tuple[SourceArchive, ...]


def fetch_sources(
    policy: SelectionPolicy,
    key: str,
    requester: Requester,
    before_request: Callable[[], None],
) -> SourceBundle: ...
```

Each archive serializes exactly:

```json
{
  "records": [],
  "request": {"path": "/...", "query": {}},
  "schema": 1
}
```

Reference archives use names `tickers-0001`, `tickers-0002`, and so on. Daily
archives use `daily-YYYY-MM-DD`. `request.query` contains sorted public query
parameters and never `apiKey`. `records` are normalized and sorted before
entering `SourceArchive`.

### Step 1: Add failing offline transport tests

With a fake requester, cover:

- exact reference and daily-summary request contracts;
- multi-page reference traversal;
- trusted-host enforcement and pagination-cycle rejection;
- status, shape, ordering, duplicate, and required-field rejection;
- every weekday requested exactly once;
- empty weekdays excluded from the session denominator;
- failed weekdays aborting instead of disappearing;
- one shared rate gate across all request types;
- canonical archives containing no API key;
- no API key in exceptions or report values.

### Step 2: Implement one shared transport path

Reuse `request_gate()`, `request_json()`, `authorized_url()`, and `api_key()`.
Keep URL builders pure. Keep the request loop generic enough for fake offline
tests, but do not add a client class or dependency.

Normalize provider values immediately into the canonical source records used by
the pure selector. Never persist the authorized URL.

### Step 3: Run the focused test

Use the Task 1 test command. Expected: pass.

### Step 4: Review and checkpoint

Run independent spec and code-quality reviews, then create one signed local
GitButler checkpoint:

```text
feat(data): archive Massive universe sources
```

## Task 3: Emit Bound Manifests and Preserve Fetch Compatibility

**Files:**

- Modify: `tools/select_universe.py`
- Modify: `tools/fetch_universe.py`
- Modify: `tests/python/test_massive.py`

### Interfaces

The orchestration boundary is:

```python
def select_universe(
    policy_path: Path,
    output_dir: Path,
    *,
    key: str | None = None,
    requester: Requester | None = None,
    requests_per_minute: int = 0,
) -> dict[str, object]: ...
```

The returned mapping is exactly the value written to `selection.json`. Its
top-level fields are:

```text
schema, purpose, declared_on, anchor_date, formation_start, formation_end,
start, end, primary_cohort_size, policy, source_closure, sources,
formation_sessions, candidates, master, master_sha256, cohorts
```

`cohorts` is keyed by decimal size strings. Each value contains exactly
`size`, `primary`, `members`, `members_sha256`, `manifest`, and
`manifest_sha256`. `members` records ticker, both FIGIs, and liquidity stratum;
the generated manifest projects only ticker and `liquidity-{stratum}`.

### Step 1: Add failing output tests

Test that:

- the output directory and final report must not already exist or alias inputs;
- each cohort is exactly the declared master prefix;
- generated manifests preserve selected order and liquidity strata;
- generated manifests use anchor date for eligibility and a later model start;
- every path and SHA-256 in the report matches fresh output bytes;
- policy/source/output mutation before final publication is rejected;
- a failure leaves no `selection.json`;
- all output JSON rejects nonfinite numbers;
- schema-1 historical manifests remain byte-compatible at the parser boundary;
- `eligibility_date < start` is accepted while
  `eligibility_date > start` is rejected.

### Step 2: Implement fresh output publication

Use `write_json_exclusive()`, `freeze_inputs()`, `verify_frozen()`,
`file_sha256()`, and `require_disjoint()`. Freeze the policy, selector,
`fetch_massive.py`, and `files.py` as the local source closure. Keep generated
data under the one ignored output directory. Publish `selection.json` last.

Change only the one `fetch_universe.py` date relationship required by the
generated manifests.

### Step 3: Run focused and aggregate gates

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py

make -B \
  PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  check
```

Then run the optional Torch gates because the shared Python suite imports no
Torch:

```sh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
  tests/python/test_experiment.py

/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
  tests/python/test_training.py
```

Expected: all pass.

### Step 4: Review and checkpoint

Run independent spec and code-quality reviews. Confirm unrelated
`Makefile`/`docs/training.md` changes remain uncommitted. Create one signed
local GitButler checkpoint:

```text
feat(data): emit nested universe manifests
```

## Task 4: Live Selection Is a Separate Evidence Step

Do not run this task until Tasks 1-3 are reviewed, green, signed, and the API
key is available to the process without printing it.

Use a new ignored output directory and the account's actual rate limit:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tools/select_universe.py \
  universes/liquid-common-ladder.example.json \
  reports/liquid-common-selection-20260724-01 \
  --requests-per-minute 5
```

After completion:

1. verify every reported hash;
2. scan all outputs for `apiKey` and the in-memory key without printing either;
3. independently recompute eligibility, ranks, strata, and prefixes from the
   canonical archives;
4. record the output directory, report SHA-256, selected names, exclusions,
   call counts, and elapsed time;
5. do not fetch minute bars or train in this task.

The next reviewed plan will first bind the selected FIGI identities through
post-anchor ticker events, then bind the selected 55-member manifest to one
2,145-fit calibration run, derive the 11/22/33/55 prefix metrics from its
ledger, and keep the reserved `$100` backtest closed unless the predeclared
primary calibration gate passes.
