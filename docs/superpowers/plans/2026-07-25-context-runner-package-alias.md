# Context Runner Package Alias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make direct-script context runs and their finalizer share one
authenticated runner module, claim registry, and terminal-outcome state.

**Architecture:** After the bootstrap authenticates and imports the real
`tools` package, register the active `__main__` module under
`tools.run_context_diagnostic`. Recheck that identity before reading an attempt
and immediately before importing the finalizer.

**Tech Stack:** Python 3.12/3.14 standard library, existing context lifecycle
tests, GitButler.

Use these exact interpreters for every command below:

```zsh
PRIMARY="$HOME/.cache/codex-runtimes/codex-primary-runtime/"\
"dependencies/python/bin/python3.12"
TORCH="$HOME/.local/share/uv/python/"\
"cpython-3.14.6-macos-aarch64-none/bin/python3.14"
```

## Global Constraints

- Do not change model families, histories, seeds, update budgets, or metrics.
- Preserve the completed `20260725-02` phase artifacts and terminal failure.
- Do not expose truth before the existing receipt barrier.
- Do not modify unrelated Makefile, training documentation, or universe-run
  evidence.
- Create signed local GitButler checkpoints; do not push.

---

### Task 1: Bind direct execution to the canonical runner module

**Files:**
- Modify: `tools/run_context_diagnostic.py`
- Test: `tests/python/test_context_diagnostic_driver.py`

**Interfaces:**
- Consumes: the authenticated `tools` package created by `_bootstrap_main()`.
- Produces: one shared `RunClaim` type, `_CLAIMS` registry, and terminal-state
  namespace for the runner and finalizer.

- [ ] **Step 1: Write the failing isolated alias test**

```python
def test_runner_package_alias_shares_finalizer_state() -> None:
    script = (
        "import importlib.util,sys; "
        f"root={str(ROOT)!r}; "
        f"path={str(ROOT / 'tools/run_context_diagnostic.py')!r}; "
        "sys.path.append(root); "
        "spec=importlib.util.spec_from_file_location("
        "'context_runner_script',path); "
        "runner=importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name]=runner; "
        "spec.loader.exec_module(runner); "
        "runner._register_package_alias(); "
        "import tools.finalize_context_diagnostic as finalizer; "
        "assert finalizer.RunClaim is runner.RunClaim; "
        "assert finalizer.publish_context_outcome.__globals__['_CLAIMS'] "
        "is runner._CLAIMS"
    )
    result = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Verify the test fails for the observed defect**

Run:

```zsh
"$PRIMARY" tests/python/test_context_diagnostic_driver.py
```

Expected: `AttributeError` because the script-loaded runner has no package
alias and the finalizer would otherwise load a second runner module.

- [ ] **Step 3: Register and verify the authenticated alias**

```python
_PACKAGE_NAME = "tools.run_context_diagnostic"

def _register_package_alias() -> None:
    module = sys.modules[__name__]
    if _PACKAGE_NAME in sys.modules:
        raise ValueError("context runner package alias is unsafe")
    sys.modules[_PACKAGE_NAME] = module

def _require_package_alias() -> None:
    if sys.modules.get(_PACKAGE_NAME) is not sys.modules[__name__]:
        raise ValueError("context runner package alias changed")
```

Call `_require_package_alias()` before attempt parsing and again immediately
before importing `tools.finalize_context_diagnostic`.

- [ ] **Step 4: Run focused lifecycle checks**

Run:

```zsh
"$PRIMARY" tests/python/test_context_diagnostic_driver.py
"$TORCH" tests/python/test_context_diagnostic_driver.py
```

Expected: both print `context diagnostic driver tests passed`.

### Task 2: Verify and checkpoint the fix

**Files:**
- Modify: `docs/superpowers/plans/2026-07-25-context-runner-package-alias.md`
- Modify: `tests/python/test_context_diagnostic_driver.py`
- Modify: `tools/run_context_diagnostic.py`

**Interfaces:**
- Consumes: Task 1's canonical runner identity.
- Produces: one signed local checkpoint suitable for a fresh attempt.

- [ ] **Step 1: Run the aggregate gate**

Run:

```zsh
make -B PYTHON="$PRIMARY" check
```

Expected: all nine C suites and every registered Python suite pass.

- [ ] **Step 2: Create and verify the local checkpoint**

Use GitButler to commit only the three files above on
`fix/context-runner-alias`. Verify the enkyuan author, committer, and ED25519
signature. Do not push.

### Task 3: Replay the frozen diagnostic once

**Files:**
- Create: `experiments/h13-context-diagnostic-20260725-03-attempt.json`
- Create: `reports/h13-context-diagnostic-20260725-03/`

**Interfaces:**
- Consumes: the exact signed Task 2 commit and unchanged frozen inputs.
- Produces: one terminal development-only context decision or a preserved
  terminal integrity failure.

- [ ] **Step 1: Arm a fresh `-03` attempt**

Set `IMPLEMENTATION_COMMIT` to the full SHA printed by Task 2's GitButler
commit, confirm both destinations are absent, and run:

```zsh
"$PRIMARY" -I -S -B tools/arm_context_diagnostic.py \
  experiments/h13-context-diagnostic-20260725-03-attempt.json \
  --implementation-commit "$IMPLEMENTATION_COMMIT" \
  --run-id h13-context-diagnostic-20260725-03 \
  --primary-python "$PRIMARY" \
  --torch-python "$TORCH"
```

Never reuse `-01` or `-02`.

- [ ] **Step 2: Execute the exact one-shot runner**

Run:

```zsh
/usr/bin/env -i \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=reports/h13-context-diagnostic-20260725-03/.pycache \
  "$TORCH" -I -S -B tools/run_context_diagnostic.py \
  experiments/h13-context-diagnostic-20260725-03-attempt.json
```

Monitor only process health, artifact existence, and fit/prediction ledger
counts until the terminal outcome is durable.

- [ ] **Step 3: Evaluate the terminal decision**

If the outcome is complete, report the selected history and paired uncertainty
gates as development evidence. Do not authorize a shared-$100 backtest unless
the separate forward-clean contract passes; otherwise retain exactly $100 in
cash.
