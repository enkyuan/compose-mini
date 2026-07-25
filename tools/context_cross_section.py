"""Measure stock-selection signal in one sealed context phase."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json

from tools.context_diagnostic_contract import (
    HISTORY_LENGTHS, PRIMARY_MODEL, ContextPhase,
    ContextPredictionEvidence, context_phase_sha256,
)
from tools.finalize_context_diagnostic import (
    ContextTruthRow, _ensembles, _truth,
)
from tools.universe_cross_section import (
    CROSS_SECTION_SEED, CrossSectionCell, cross_section_diagnostics,
)
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES,
)
from tools.universe_scaling_contract import timestamp_grid_sha256


def _group(row: ContextTruthRow) -> tuple[str, str, str]:
    return row.as_of, row.entry_time, row.target_time


def evaluate_context_cross_section(
    master: Sequence[str], phase: ContextPhase,
    evidence: Sequence[ContextPredictionEvidence],
    truth: Mapping[str, Sequence[ContextTruthRow]], history: int,
    groups: Sequence[tuple[str, str, str]],
    *, block_days: Sequence[int] = BOOTSTRAP_BLOCK_DAYS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = CROSS_SECTION_SEED,
) -> dict[str, object]:
    """Evaluate the selected Transformer after removing each group mean."""
    if type(history) is not int or history not in HISTORY_LENGTHS:
        raise ValueError("context cross-sectional history is invalid")
    rows = _truth(phase, truth)
    predictions = _ensembles(master, phase, evidence)
    names = tuple(rows)
    common = tuple(groups)
    if not common or len(set(common)) != len(common):
        raise ValueError("context cross-sectional groups are invalid")
    grid_sha256, required = timestamp_grid_sha256(common), set(common)
    indexed = {}
    for series in names:
        values = {}
        for row, predicted in zip(
            rows[series],
            predictions[PRIMARY_MODEL, history, series],
            strict=True,
        ):
            key = _group(row)
            if key in values:
                raise ValueError("context cross-sectional group is duplicated")
            values[key] = row.actual_return, predicted
        indexed[series] = values
    if any(not required.issubset(values) for values in indexed.values()):
        raise ValueError("context cross-sectional group is incomplete")
    expected = tuple(
        (
            target[:10],
            json.dumps(
                (phase.phase, as_of, entry, target),
                separators=(",", ":"),
            ),
        )
        for as_of, entry, target in common
    )
    result = cross_section_diagnostics(
        tuple(
            CrossSectionCell(
                key[2][:10], group, series, *indexed[series][key],
            )
            for key, (_, group) in zip(common, expected, strict=True)
            for series in names
        ),
        names,
        expected_groups=expected,
        block_days=block_days,
        replicates=replicates,
        seed=seed,
    )
    diagnostic = asdict(result)
    diagnostic["intervals"] = {
        str(item.block_days): (item.lower, item.upper)
        for item in result.intervals
    }
    return {
        "diagnostic": diagnostic,
        "evidence_role": "development-post-hoc-not-forward-clean",
        "group_grid_sha256": grid_sha256,
        "history": history,
        "model": PRIMARY_MODEL,
        "phase": phase.phase,
        "phase_sha256": context_phase_sha256(phase),
        "schema": 1,
    }
