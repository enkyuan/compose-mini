# Bound Universe Manifest Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task by task.

**Goal:** Publish one fresh, hash-bound universe-selection package containing
canonical Massive sources, nested schema-1 fetch manifests, and an auditable
final selection report.

**Architecture:** Keep JSON construction pure. One orchestration boundary
freezes the policy and source closure, retrieves and selects entirely in
memory, writes immutable package members through pinned directory descriptors,
revalidates every binding, then publishes `selection.json` as the sole
completion marker.

**Tech Stack:** Python 3.12 standard library, existing Massive transport,
existing `tools.files` integrity primitives, procedural Python tests,
GitButler.

---

## Constraints

- Modify only `tools/files.py`, `tools/select_universe.py`,
  `tools/fetch_universe.py`, and `tests/python/test_massive.py`.
- Preserve unrelated `Makefile` and `docs/training.md` changes.
- Keep generated outputs, credentials, datasets, and models ignored and
  uncommitted.
- Do not fetch minute bars, train, select a model, promote a policy, open the
  reserved test, or run the `$100` backtest.
- Use the existing retry-aware Massive transport and file-integrity helpers;
  add no dependency and no publisher class.
- Keep each source archive's frozen schema exactly
  `{schema, request, records}`. Record the ordered retained session dates in
  the final report. An empty daily archive whose date appears there is a
  raw-nonempty response filtered to zero rows; one absent from that list is a
  truly empty provider response.
- Archive well-formed provider ticker text exactly, but admit only canonical
  runtime tickers into references and formation rows. Do not normalize or
  alias unsupported symbols: Massive currently returns the mixed-case
  `ECGw` record from a dated `type=CS` query.
- Publish only into a path that did not exist when the command started. A
  failure before completion may leave an ignored partial directory; retry with
  a new directory. Normal rollback removes `selection.json`, while definitive
  rollback I/O failure preserves it as explicitly reported ambiguous evidence.
  The package is committed only after the marker is exact, all private
  temporary files are gone, exact membership is restored, and the root
  directory is synced.
- Create one reviewed, signed, local checkpoint:
  `feat(data): emit nested universe manifests`. Do not push or land it.

## Public Interface

Add to `tools/select_universe.py`:

```python
def select_universe(
    policy_path: Path,
    output_dir: Path,
    *,
    key: str | None = None,
    requester: Requester | None = None,
    requests_per_minute: int = 0,
) -> dict[str, object]: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...


def main() -> None: ...
```

Make the file directly executable before importing `tools.*`, using the same
bootstrap as `fetch_universe.py`:

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
```

The CLI is:

```text
select_universe.py POLICY OUTPUT_DIR [--requests-per-minute N]
```

The returned mapping must exactly equal the JSON value written to
`OUTPUT_DIR/selection.json`.

## Canonical Values

Serialize JSON with the existing `write_json_exclusive()` representation:
UTF-8, `allow_nan=False`, two-space indentation, sorted keys, and one trailing
newline. For semantic list hashes, use the same representation:

```python
def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
```

`master_sha256` and `members_sha256` hash those standalone JSON-array bytes.
File hashes use `file_sha256()` over the actual file bytes. Emit every
selection `Decimal` as a finite, normalized fixed-point string, never a JSON
float.

Each source file is `sources/<archive.name>.json` and contains exactly:

```json
{
  "records": [],
  "request": {"path": "/...", "query": {}},
  "schema": 1
}
```

Each item in the report's `sources` list contains exactly:

```text
name, path, sha256, records, formation_session
```

`records` is the integer number of canonical records, not the records
themselves. `sources` preserves `SourceBundle.archives` order. Generated paths
are POSIX paths relative to the output directory.

`formation_session` is `null` for a reference page, `false` for a truly empty
daily response, and `true` for every date in `SourceBundle.sessions`,
including a filtered-empty session. This flag is derived from the bundle; it
does not alter the frozen archive schema.

Each candidate report value is flat and contains exactly:

```text
ticker, active, market, locale, type, currency_name, primary_exchange,
composite_figi, share_class_figi, observed, coverage, median_close_usd,
median_dollar_volume_usd, rejection_reasons, decision,
share_class_representative, liquidity_rank, stratum, within_stratum_rank,
master_rank
```

Decisions are:

- `selected` for a master member;
- `eligible-not-selected` for an unrejected representative beyond the master
  prefix;
- `rejected` for any nonempty rejection list.

`share_class_representative` is the retained representative's ticker, the
winner's ticker for `duplicate-share-class-figi`, or `null` when rejection
occurred before share-class deduplication.

`master` and every cohort `members` list contain exactly:

```text
ticker, composite_figi, share_class_figi, stratum
```

The final report's top-level fields are exactly:

```text
schema, purpose, declared_on, anchor_date, formation_start, formation_end,
start, end, primary_cohort_size, policy, source_closure, sources,
formation_sessions, candidates, master, master_sha256, cohorts
```

`formation_sessions` is the ordered ISO-date list from
`SourceBundle.sessions`. `policy` contains exactly `path` and `sha256`.
`policy.path` is the resolved root-relative POSIX path when the resolved input
is under `ROOT`; otherwise it is the resolved absolute POSIX path.

`source_closure` is an ordered list of objects containing exactly `path` and
`sha256`, in this fixed order:

```text
tools/select_universe.py
tools/fetch_massive.py
tools/files.py
```

Its paths are root-relative POSIX paths.

`cohorts` is keyed by the declared decimal size strings in policy order. Each
value contains exactly:

```text
size, primary, members, members_sha256, manifest, manifest_sha256
```

Every `members` value must equal `master[:size]` by direct equality.

Each generated schema-1 manifest is
`manifests/liquid-common-<size>.json` and contains exactly:

```text
schema, purpose, declared_on, eligibility_date, start, end,
interval_minutes, adjusted, session, series
```

It uses the policy anchor as `eligibility_date`, the later policy `start` for
model data, and projects each member to:

```json
{"stratum": "liquidity-1", "ticker": "AAPL"}
```

## Task 1: Preserve Schema-1 Eligibility Compatibility

**Files:**

- Modify: `tests/python/test_massive.py`
- Modify: `tools/fetch_universe.py`

- [ ] Extend `test_manifest_contract()` with three cases:
  `eligibility_date == start` remains byte-compatible, an earlier eligibility
  date parses, and a later eligibility date fails.
- [ ] Run the focused test and record RED at the earlier-date case.
- [ ] Change only the date relationship to:

  ```python
  declared_on < eligibility_date or eligibility_date > start or start > end
  ```

- [ ] Update only invalid fixtures that intentionally depended on inequality.
- [ ] Rerun the focused test and record GREEN.

## Task 2: Build Pure Report and Manifest Values

**Files:**

- Modify: `tests/python/test_massive.py`
- Modify: `tools/select_universe.py`

Add only these minimal pure helpers:

```python
def _decimal_text(value: Decimal) -> str: ...
def _canonical_sha256(value: object) -> str: ...
def _member_value(candidate: Candidate) -> dict[str, object]: ...
def _candidate_value(
    candidate: Candidate,
    representative_ticker: str | None,
) -> dict[str, object]: ...
def _manifest_value(
    policy: SelectionPolicy,
    members: Sequence[Mapping[str, object]],
) -> dict[str, object]: ...
def _source_archive_value(archive: SourceArchive) -> dict[str, object]: ...
def _source_binding_value(
    archive: SourceArchive,
    formation_sessions: frozenset[str],
    sha256: str,
) -> dict[str, object]: ...
```

- [ ] Test exact keys, normalized decimal strings, all three decisions,
  duplicate-to-representative binding, and nonfinite rejection.
- [ ] Test exact member projection, ordered master prefixes, exact manifest
  fields, and liquidity-stratum projection.
- [ ] Test semantic hashes under reordered mapping insertion and rejection of
  NaN, infinity, and nonfinite `Decimal`.
- [ ] Test the source flag derivation for provider-empty and filtered-empty
  archives while keeping source files at exactly three top-level fields.
- [ ] Record import RED, implement the helpers directly, and record GREEN.

## Task 3: Publish a Fresh Bound Package

**Files:**

- Modify: `tests/python/test_massive.py`
- Modify: `tools/select_universe.py`

### Transport

Use exactly one gate per physical request:

```python
def _transport(
    requester: Requester | None,
    requests_per_minute: int,
) -> tuple[Requester, Callable[[], None]]:
    gate = request_gate(requests_per_minute)
    if requester is None:
        return (
            lambda url: request_json(url, before_request=gate),
            lambda: None,
        )
    return requester, gate
```

Then call:

```python
transport, before_request = _transport(requester, requests_per_minute)
bundle = fetch_sources(
    policy,
    api_key(ROOT / ".env") if key is None else key,
    transport,
    before_request,
)
```

Test the live branch with a fake `request_json()` that invokes its callback
twice: the gate count must be two, not three. Test that an injected requester
gets one logical gate.

### Fresh-output algorithm

- [ ] Validate the policy and three source-closure paths as distinct,
  nonsymlink regular files; record `(st_dev, st_ino)`.
- [ ] Reject the original, absolute-normalized, and resolved output target if
  any exists. Reject an output that aliases an input, a source path, or its
  own completion marker through `require_disjoint()`.
- [ ] Require the normalized output parent to exist as a nonsymlink directory.
  Open and identity-bind that parent *before retrieval*. All later root
  creation happens by basename relative to this pinned descriptor. Recheck
  that the named parent still resolves to the recorded identity before
  publication.
- [ ] Freeze `(policy_path, *SOURCE_PATHS)` and parse the policy snapshot.
- [ ] Retrieve sources and select candidates entirely in memory before
  creating the output directory.
- [ ] Create the root with `os.mkdir(output_name, dir_fd=parent_fd)`. Open it
  relative to the pinned parent with `O_DIRECTORY | O_NOFOLLOW`; create and
  similarly open `sources` and `manifests` relative to that descriptor.
- [ ] `fsync(parent_fd)` after root creation so the package path itself is
  crash-durable. A failure occurs before marker publication and leaves only a
  markerless partial directory.
- [ ] Write every source and manifest with `write_json_exclusive()` and the
  corresponding pinned directory descriptor.
- [ ] Immediately inspect every member through its pinned child descriptor.
  Require a nonsymlink regular file with link count one and record
  `(st_dev, st_ino, stat.S_IFMT(st_mode), st_nlink)`. Hash member bytes through
  a descriptor opened relative to that same pinned directory; do not trust a
  later path traversal.
- [ ] Hash actual files and construct the JSON-native report.
- [ ] `fsync()` both child directories and the root before constructing the
  completion marker.
- [ ] In the final `before_link` callback:
  - call `verify_frozen()`;
  - recheck all original input identities;
  - recheck the named output parent against its pinned descriptor;
  - recheck root and child identities against their open descriptors;
  - recheck every source and manifest's recorded identity, regular-file type,
    and link count through its pinned child descriptor;
  - require exact permanent directory membership, ignoring only the writer's
    private `.selection.json.<token>.tmp`;
  - require `selection.json` absent;
  - rehash every source and manifest;
  - reassert every cohort prefix and semantic hash.
- [ ] Publish `selection.json` through the pinned root descriptor with an
  atomic no-replace rename that consumes the private temporary name. Reconcile
  an ambiguous syscall error from the writer-bound inode: accept only an
  absent source plus the exact destination inode with link count one. Preserve
  the temporary on every definite failure.
- [ ] After the rename, require exact marker bytes, no temporary residue,
  exact root/child membership, and all previously recorded identities and
  hashes. Then `fsync()` the root. This successful sync is the commit point.
- [ ] On any post-link validation, cleanup, or sync failure, quarantine the
  marker through the pinned root descriptor, sync that rename, and raise.
  Tests must exercise this cleanup path. Return the report only after the full
  commit point succeeds.
- [ ] Close all descriptors in `finally`.

Never recursively clean a failed package. This preserves failure evidence and
avoids deleting an unrelated path if a parent was substituted.

### Completion-marker integrity addendum

Adversarial review expands this task to `tools/files.py` only for the shared
exclusive writer; `tools/fetch_universe.py` remains unchanged by this addendum.
The writer must bind its private name to the `(st_dev, st_ino)` captured from
the descriptor it created, expose that binding to the selector, and recheck it
before publication. The selector must not infer ownership by scanning the
directory in `before_link`.

Successful publication uses Darwin `renameatx_np(RENAME_EXCL)` or Linux
`renameat2(RENAME_NOREPLACE)` and fails closed when neither primitive is
available. It never falls back to an overwrite-capable rename or to
link-then-unlink. A definite failure retains the private temporary as evidence;
an ambiguous error succeeds only when the source is absent and the destination
is the writer-bound regular inode with link count one.

Public rollback uses the same no-replace primitive to move `selection.json`
into a fresh, pinned quarantine directory before inspecting it. An owned
marker remains quarantined in a failed package; a foreign marker is restored
with another no-replace rename and never unlinked. Normal rollback removes the
publisher-owned public completion name. If both rollback attempts fail without
moving it, the command raises `universe marker rollback failed` and preserves
all evidence, including the public marker; no filesystem protocol can promise
namespace reversal after definitive I/O failure. A failed or interrupted
command therefore never makes marker existence sufficient: consumers also
require full package validation and a successful producing command.

Add RED cases that replace the writer's private name before `before_link` and
replace the public marker immediately before rollback. GREEN requires both
foreign inodes and their bytes to survive, no foreign entry to be accepted as
the completion marker, and every existing publication test to remain green.

### Offline integration tests

Use a one-session policy and at least 60 valid fake references so all four
cohorts are exercised cheaply. Verify:

- exact root, `sources/`, and `manifests/` membership;
- report bytes parse to the returned mapping;
- every reported path and file hash;
- exact source schemas and formation-session flags;
- every manifest parses through `UniverseManifest.read()`;
- exact master and cohort semantic hashes and prefix equality;
- exact policy and source-closure bindings;
- absence of `apiKey`, the fake key, NaN, and infinity from every output;
- `selection.json` is the last permanent write.

Patch the final exclusive write and mutate one condition per test: policy,
source closure, source archive, manifest, output file identity, directory
identity, or undeclared permanent membership. Every case must fail without a
`selection.json`.

Also simulate an exception after marker publication and a root-sync failure.
An injected ambiguous rename error succeeds only when the exact writer-bound
inode committed; a definite rename failure preserves its private temporary and
leaves no completion marker. Sync or validation failure uses the pinned-marker
quarantine path. Test both normal removal and definitive double-rollback
failure, which raises explicitly and preserves the owned marker as ambiguous
failure evidence. Also simulate parent-sync failure after root creation and
require a markerless partial directory.

Also reject an existing directory or file, symlinks including broken links,
lexically normalized aliases, input aliases, and substituted output parents.
Test the actual script entrypoint, not only imported `parse_args()`, so the
`ROOT` bootstrap is covered. Network or selection failure before creation
leaves no directory; later pre-commit failure may leave a markerless partial
directory.

### CLI

- [ ] Test exact parsed arguments and nonnegative rate validation through
  `request_gate()`.
- [ ] `main()` catches only `OSError` and `ValueError`, exits with a fixed
  secret-free message, and prints sorted one-line JSON on success.
- [ ] Record RED, implement the smallest orchestration and CLI, and record
  GREEN.

## Task 4: Verify, Review, and Checkpoint

- [ ] Run:

  ```sh
  /Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
    tests/python/test_massive.py
  ```

- [ ] Run:

  ```sh
  make -B \
    PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
    check
  ```

- [ ] Run the optional Torch gates:

  ```sh
  /Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
    tests/python/test_experiment.py

  /Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
    tests/python/test_training.py
  ```

- [ ] Obtain independent specification and code-quality approval with no
  Critical or Important findings.
- [ ] Confirm only the three task files are dirty and unrelated `Makefile` and
  `docs/training.md` remain untouched.
- [ ] Commit the plan separately, then create the signed local implementation
  checkpoint `feat(data): emit nested universe manifests` on the existing
  point-in-time universe stack.
- [ ] Verify author, committer, ED25519 signature, and exact checks. Do not
  push or land.

## Later Live Evidence Step

Only after Tasks 1-4 are signed and green, use a new ignored directory:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tools/select_universe.py \
  universes/liquid-common-ladder.example.json \
  reports/liquid-common-selection-20260724-01 \
  --requests-per-minute 5
```

Verify every hash, scan outputs for secret material without printing the key,
and independently recompute eligibility, strata, rankings, and prefixes from
the canonical sources. Stop before minute-bar fetching or training. A later
reviewed plan will bind the selected 55-name manifest to the fixed benchmark
and report the predeclared 11/22/33/55 learning curve without selecting the
best cohort after seeing outcomes.
