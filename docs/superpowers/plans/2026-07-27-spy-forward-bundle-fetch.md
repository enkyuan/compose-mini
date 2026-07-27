# SPY Residual Forward Bundle Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the one complete, atomic Massive data bundle accepted by the
preregistered SPY-residual forward runner without exposing a label or changing
the frozen candidate.

**Architecture:** Reuse the existing Massive request, pacing, calendar, CSV,
hashing, and exclusive-rename primitives. Move only the generic atomic
directory publication code out of the SPY benchmark fetcher, then add one
fixed-purpose fetcher that derives all dates, tickers, filenames, and report
fields from the frozen forward contract. The fetcher rejects execution before
the final session closes and publishes nothing unless all 12 series have the
exact complete 30-minute grid.

**Tech Stack:** Python 3.12+ standard library, Massive aggregates API, existing
session-calendar and file-integrity helpers, direct Python test scripts,
GitButler.

## Global Constraints

- Apply the Karpathy Guidelines and Ponytail discipline: test first, reuse
  existing primitives, and keep the diff surgical.
- Do not train, refit, tune, fetch real forward data, open truth, reconstruct a
  price, select a policy, backtest, or trade.
- Keep the frozen candidate, scale, seeds, universe order, 60 target sessions,
  20-session decision block, gates, and false execution locks unchanged.
- Bind the tracked calendar at
  `universes/us-equities-core-forward-2026-05-19_2026-08-18.json` to SHA-256
  `997d751a5a2ae8b2c51f4b500bd27ec94155359e211d1f9cfef198d5f156c362`.
- Fetch exactly `KRYS,TGT,STM,SSNC,NWL,AAON,GEV,SWKS,BMRN,ACI,HUN,SPY`
  as adjusted 30-minute regular-session aggregates.
- Require the exact calendar grid for every series. Missing or extra bins fail
  without changing the dates, universe, or output target.
- Do not make a real request before the 2026-08-18 final regular-session bar is
  complete. Tests use synthetic responses and an injected completed timestamp.
- Publish `fetch.json` and all 12 CSVs as one no-replace directory rename.
  Failure must leave no partial final bundle.
- Never put `MASSIVE_API_KEY` in a URL contract, report, CSV, exception, or
  tracked file.
- Keep generated data, credentials, reports, caches, and model state ignored.
- Preserve unrelated working-tree changes in `Makefile`, `docs/training.md`,
  and the existing experiment JSON files.

---

### Task 1: Produce the Strict Atomic Forward Bundle

**Files:**

- Create: `tools/atomic_bundle.py`
- Modify: `tools/fetch_benchmark.py`
- Create: `tools/fetch_spy_residual_forward.py`
- Modify: `tests/python/test_massive.py`
- Create: `tests/python/test_fetch_spy_residual_forward.py`

**Interfaces:**

- Consumes:
  `publish_directory(stage: Path, target: Path, verify: Callable[[], None])`,
  `fetch_bars()`, `request_gate()`, `scan_regular_bars()`,
  `session_grid_audit()`, `write_csv()`, the tracked forward calendar, and the
  frozen `FORWARD_UNIVERSE`.
- Produces:
  `fetch_forward_bundle(bundle: Path, *, env_file: Path = ROOT / ".env",
  key: str | None = None, requester: Requester | None = None,
  requests_per_minute: int = 5,
  current_time: datetime | None = None) -> Mapping[str, object]`.
- Preserves:
  `fetch_benchmark()` behavior and its atomic-publication failure semantics.
- Publishes:
  one absolute bundle directory containing `fetch.json`,
  `krys-30m.csv`, `tgt-30m.csv`, `stm-30m.csv`, `ssnc-30m.csv`,
  `nwl-30m.csv`, `aaon-30m.csv`, `gev-30m.csv`, `swks-30m.csv`,
  `bmrn-30m.csv`, `aci-30m.csv`, `hun-30m.csv`, and `spy-30m.csv`.

- [ ] **Step 1: Write the failing synthetic bundle tests**

Create `tests/python/test_fetch_spy_residual_forward.py`. Build one synthetic
Massive aggregate for every `expected_bins()` item in the tracked forward
calendar and route responses by the ticker encoded in each aggregate URL.
The success test must:

```python
report = fetch_forward_bundle(
    bundle,
    key="fake-secret",
    requester=requester,
    requests_per_minute=0,
    current_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
)
assert [item["ticker"] for item in report["series"]] == [
    *FORWARD_UNIVERSE, "SPY",
]
assert all(item["csv"]["rows"] == 819 for item in report["series"])
assert "fake-secret" not in (bundle / "fetch.json").read_text()
with _bound_future(FORWARD_CALENDAR_PATH, bundle, FORWARD_CALENDAR[2]):
    pass
```

Also require:

```python
rejects(
    fetch_forward_bundle,
    early_bundle,
    key="fake-secret",
    requester=unreachable,
    requests_per_minute=0,
    current_time=datetime(2026, 7, 27, tzinfo=timezone.utc),
)
assert not early_bundle.exists()

rejects(
    fetch_forward_bundle,
    incomplete_bundle,
    key="fake-secret",
    requester=missing_one_expected_bar,
    requests_per_minute=0,
    current_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
)
assert not incomplete_bundle.exists()
```

Patch the publication helper to fail after staging and assert that neither the
bundle nor a staging directory remains. Add an isolated import test proving
that importing the fetcher does not import Torch, training, experiment, runner,
or finalizer modules.

- [ ] **Step 2: Run the new test and verify the fetcher import fails**

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -I -B tests/python/test_fetch_spy_residual_forward.py
```

Expected: nonzero exit because `tools.fetch_spy_residual_forward` does not
exist.

- [ ] **Step 3: Extract the generic atomic directory publisher**

Move the benchmark fetcher's path identity, symlink rejection, identity
verification, and exclusive directory rename into `tools/atomic_bundle.py`.
The public publication function must retain the existing ambiguous-rename
handling:

```python
def publish_directory(
    stage: Path,
    target: Path,
    verify: Callable[[], None],
) -> Identity:
    """Fsync and exclusively rename one verified directory into place."""
    if stage.parent != target.parent:
        raise ValueError("staged and final bundles must share one parent")
    parent_fd = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        identity = path_identity(stage, "staged bundle", directory=True)
        verify()
        stage_fd = os.open(
            stage.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        failure = None
        try:
            rename_noreplace(
                parent_fd, stage.name, parent_fd, target.name,
            )
        except OSError as error:
            failure = error
        source = entry(parent_fd, stage.name)
        result = entry(parent_fd, target.name)
        committed = (
            rename_may_have_committed(failure)
            and source is None
            and result == (identity, stat.S_IFDIR)
            and entry(parent_fd, stage.name) is None
        )
        if not committed:
            if failure is not None:
                raise failure
            raise OSError("bundle publication failed")
        os.fsync(parent_fd)
        return identity
    finally:
        os.close(parent_fd)
```

Keep the benchmark fetcher's public API and validation unchanged. Update its
tests only where mocks must target `tools.atomic_bundle` after the extraction.

- [ ] **Step 4: Implement the fixed fetcher**

In `tools/fetch_spy_residual_forward.py`:

1. Resolve the forward calendar path and verify its exact SHA-256 before
   reading the API key.
2. Derive the expected timestamp tuple from `expected_bins()` and compute the
   ready time as the final expected bar start plus 30 minutes.
3. Reject a naive timestamp or a time earlier than readiness before invoking
   the requester.
4. Fetch the 12 tickers in frozen order through one shared request gate.
5. Require every aggregate page to report the requested ticker and
   `adjusted: true`.
6. Retain regular-session bars, require exact timestamp equality, and write
   each strict CSV into one private staging directory.
7. Build exactly the schema reconstructed by
   `arm_spy_residual_forward._validate_bundle_report()`:

```python
report = {
    "adjusted": True,
    "calendar": {
        "path": str(FORWARD_CALENDAR_PATH),
        "sha256": calendar_input.sha256,
    },
    "end": str(calendar.end),
    "interval_minutes": INTERVAL_MINUTES,
    "provider": "massive",
    "purpose": "Authenticate the fixed SPY-residual forward holdout.",
    "schema": 1,
    "series": records,
    "session": "regular",
    "start": str(calendar.start),
}
```

Each record must contain the secret-free aggregate path/query and:

```python
{
    "csv": {
        "path": str(bundle / f"{ticker.lower()}-30m.csv"),
        "rows": len(expected_times),
        "session_audit": {
            "expected_bins": len(expected_times),
            "missing_bins": 0,
            "scope": "all-expected-session-bins",
        },
        "sha256": csv_input.sha256,
        "source_rows": len(source),
    },
    "ticker": ticker,
}
```

Freeze the staged CSVs and report, revalidate every input and inode, then call
`publish_directory()`. Reopen the final bundle and verify exact membership,
hashes, identities, and report bytes before returning.

- [ ] **Step 5: Run focused tests**

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -I -B tests/python/test_fetch_spy_residual_forward.py
PYTHONDONTWRITEBYTECODE=1 \
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -I -B tests/python/test_massive.py
```

Expected: both scripts print their `passed` summaries and exit zero.

- [ ] **Step 6: Run the forward and repository gates**

Run the forward contract, gate, input, armer, runner, and finalizer scripts
directly, then:

```sh
make -B \
  PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  check
```

Use the existing Python 3.14/PyTorch environment for PyTorch-dependent forward
tests. Do not execute the real forward runner or fetcher.

- [ ] **Step 7: Create one signed local checkpoint**

Use `but diff`, then commit only this plan's files on
`feat/spy-forward-bundle-fetch` with:

```sh
but commit feat/spy-forward-bundle-fetch -c \
  -m "feat(data): fetch strict forward bundle" \
  --changes <plan-change-ids>
```

Do not push or land.
