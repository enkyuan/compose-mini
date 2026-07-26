#!/usr/bin/env python3
"""Orchestrate one authenticated SPY-residual forward candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in tuple(map(os.path.realpath, sys.path)):
    sys.path.append(str(ROOT))

from tools.arm_spy_residual import _directory_members
from tools.arm_spy_residual_forward import (
    FORWARD_RUN_DIR, ForwardLease, ForwardRunContext,
)
from tools.panel_contract import (
    _absent, _directory_identity, _open_directory, mkdir_nofollow,
)
from tools.spy_residual_forward_inputs import (
    CandidateLedger, SpyResidualForwardInputs, TruthReader,
)

@dataclass(frozen=True, slots=True)
class ForwardRunClaim:
    """Bind one process to a newly created canonical run directory."""

    path: Path
    identity: tuple[int, int]


def _require_isolated_execution() -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or \
       not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "forward runner requires isolated bytecode-free Python",
        )


def _claim_run() -> ForwardRunClaim:
    _absent(FORWARD_RUN_DIR, "forward run directory")
    parent, parent_identity = _open_directory(FORWARD_RUN_DIR.parent)
    try:
        identity = mkdir_nofollow(FORWARD_RUN_DIR)
        os.fsync(parent)
        if _directory_identity(FORWARD_RUN_DIR.parent) != parent_identity or \
           _directory_members(FORWARD_RUN_DIR, ()) != identity:
            raise ValueError("forward run claim changed")
    finally:
        os.close(parent)
    return ForwardRunClaim(FORWARD_RUN_DIR, identity)


def publish_forward_candidate(
    lease: ForwardLease, context: ForwardRunContext,
) -> tuple[CandidateLedger, TruthReader]:
    """Stream all cutoffs and stop before the deferred truth reader."""
    session, read_truth = lease._prepare(context)
    result: SpyResidualForwardInputs | CandidateLedger = session.current()
    while isinstance(result, SpyResidualForwardInputs):
        result = session.submit(result)
    if type(result) is not CandidateLedger:
        raise ValueError("forward candidate publication changed")
    return result, read_truth
