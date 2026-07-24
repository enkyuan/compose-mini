#!/usr/bin/env python3
"""Apply a hash-bound development-coverage overlay to a frozen universe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fetch_massive import TICKER
from tools.fetch_universe import MANIFEST_FIELDS, SERIES_FIELDS, UniverseManifest
from tools.files import (
    file_sha256, freeze_inputs, require_disjoint, verify_frozen, write_json,
    write_json_exclusive,
)
from tools.panel_contract import regular_file_identities

POLICY_FIELDS = {
    "schema", "purpose", "declared_on", "scope", "selection",
    "selection_tree", "base_manifest", "failed_member", "replacement_rule",
    "replacement",
}
BINDING_FIELDS = {"path", "sha256"}
TREE_FIELDS = {"root", "files", "sha256"}
FAILED_FIELDS = {"ticker", "master_rank", "stratum"}
REPLACEMENT_FIELDS = {"ticker", "composite_figi", "share_class_figi"}
SELECTION_FIELDS = {
    "schema", "purpose", "declared_on", "anchor_date", "formation_start",
    "formation_end", "start", "end", "primary_cohort_size", "policy",
    "source_closure", "sources", "formation_sessions", "candidates", "master",
    "master_sha256", "cohorts",
}
CANDIDATE_FIELDS = {
    "ticker", "active", "market", "locale", "type", "currency_name",
    "primary_exchange", "composite_figi", "share_class_figi", "observed",
    "coverage", "median_close_usd", "median_dollar_volume_usd",
    "rejection_reasons", "decision", "share_class_representative",
    "liquidity_rank", "stratum", "within_stratum_rank", "master_rank",
}
MEMBER_FIELDS = {
    "ticker", "composite_figi", "share_class_figi", "stratum",
}
COHORT_FIELDS = {
    "size", "primary", "members", "members_sha256", "manifest",
    "manifest_sha256",
}
RULE = "first-same-stratum-eligible-not-selected-by-within-stratum-rank"
SCOPE = "development-coverage-only-conditional-on-observability"


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("coverage overlay contains a duplicate field")
        value[name] = item
    return value


def _constant(_: str) -> object:
    raise ValueError("coverage overlay JSON constants must be finite")


def _object(
    value: object, fields: set[str], label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(byte not in "0123456789abcdef" for byte in text):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return text


def _ticker(value: object, label: str) -> str:
    text = _text(value, label)
    if TICKER.fullmatch(text) is None:
        raise ValueError(f"{label} is invalid")
    return text


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _relative(value: object, label: str) -> Path:
    text = _text(value, label)
    path = Path(text)
    if path.is_absolute() or path.as_posix() != text or \
       not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a canonical relative path")
    return path


def _binding(value: object, label: str) -> tuple[Path, str]:
    item = _object(value, BINDING_FIELDS, label)
    return (
        _relative(item["path"], f"{label}.path"),
        _sha256(item["sha256"], f"{label}.sha256"),
    )


def _tree_binding(value: object) -> tuple[Path, int, str]:
    item = _object(value, TREE_FIELDS, "coverage overlay selection_tree")
    return (
        _relative(item["root"], "coverage overlay selection_tree.root"),
        _integer(item["files"], "coverage overlay selection_tree.files", 1),
        _sha256(
            item["sha256"], "coverage overlay selection_tree.sha256",
        ),
    )


def _regular(path: Path, label: str) -> None:
    if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a nonsymlink regular file")


def _bound_directory(root: Path, relative: Path, label: str) -> Path:
    base = root.resolve(strict=True)
    path = base / relative
    if path.resolve(strict=True) != path or path.is_symlink() or \
       not path.is_dir():
        raise ValueError(f"{label} must be a nonsymlink repository directory")
    return path


def _bound_path(root: Path, relative: Path, label: str) -> Path:
    base = root.resolve(strict=True)
    path = base / relative
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} must remain inside the repository")
    _regular(path, label)
    return path


def _tree_state(
    root: Path,
) -> tuple[tuple[Path, ...], dict[str, str], str]:
    entries = []
    for path in root.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        kind = stat.S_IFMT(metadata.st_mode)
        if kind == stat.S_IFREG:
            relative = path.relative_to(root).as_posix()
            entries.append((relative, file_sha256(path)))
        elif kind != stat.S_IFDIR:
            raise ValueError("selection tree contains a nonregular member")
    entries.sort()
    digest = hashlib.sha256()
    for relative, sha256 in entries:
        digest.update(
            relative.encode("utf-8") + b"\0" +
            sha256.encode("ascii") + b"\n"
        )
    by_path = dict(entries)
    ordered_paths = tuple(root / relative for relative in by_path)
    return ordered_paths, by_path, digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = (
        json.dumps(
            value, allow_nan=False, indent=2, sort_keys=True,
        ).encode("utf-8") + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def _policy(value: object) -> Mapping[str, object]:
    item = _object(value, POLICY_FIELDS, "coverage overlay policy")
    if type(item["schema"]) is not int or item["schema"] != 1 or \
       item["scope"] != SCOPE or item["replacement_rule"] != RULE:
        raise ValueError("coverage overlay policy contract is invalid")
    _text(item["purpose"], "coverage overlay purpose")
    declared = _text(item["declared_on"], "coverage overlay declared_on")
    try:
        parsed = date.fromisoformat(declared)
    except ValueError as error:
        raise ValueError("coverage overlay declared_on is invalid") from error
    if str(parsed) != declared:
        raise ValueError("coverage overlay declared_on is invalid")
    _binding(item["selection"], "coverage overlay selection")
    _tree_binding(item["selection_tree"])
    _binding(item["base_manifest"], "coverage overlay base_manifest")
    failed = _object(
        item["failed_member"], FAILED_FIELDS, "coverage overlay failed_member",
    )
    _ticker(failed["ticker"], "coverage overlay failed ticker")
    _integer(failed["master_rank"], "coverage overlay failed master_rank")
    _integer(failed["stratum"], "coverage overlay failed stratum", 1)
    replacement = _object(
        item["replacement"], REPLACEMENT_FIELDS,
        "coverage overlay replacement",
    )
    _ticker(replacement["ticker"], "coverage overlay replacement ticker")
    _text(
        replacement["composite_figi"],
        "coverage overlay replacement composite_figi",
    )
    _text(
        replacement["share_class_figi"],
        "coverage overlay replacement share_class_figi",
    )
    return item


def _candidates(selection: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    _object(selection, SELECTION_FIELDS, "selection report")
    if type(selection["schema"]) is not int or selection["schema"] != 1 or \
       type(selection["primary_cohort_size"]) is not int:
        raise ValueError("selection report contract is invalid")
    raw = selection["candidates"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("selection candidates are invalid")
    values = tuple(
        _object(item, CANDIDATE_FIELDS, f"selection candidate[{index}]")
        for index, item in enumerate(raw)
    )
    tickers = tuple(
        _ticker(item["ticker"], "selection candidate ticker")
        for item in values
    )
    if len(set(tickers)) != len(tickers):
        raise ValueError("selection candidates must have unique tickers")
    return values


def _candidate(
    candidates: Sequence[Mapping[str, object]], ticker: str,
) -> Mapping[str, object]:
    matches = tuple(item for item in candidates if item["ticker"] == ticker)
    if len(matches) != 1:
        raise ValueError("coverage overlay member is not a unique candidate")
    return matches[0]


def _replacement_candidate(
    candidates: Sequence[Mapping[str, object]],
    failed: Mapping[str, object],
) -> Mapping[str, object]:
    stratum = _integer(failed["stratum"], "failed candidate stratum", 1)
    if failed["decision"] != "selected" or \
       failed["rejection_reasons"] != [] or \
       type(failed["master_rank"]) is not int:
        raise ValueError("failed member must be selected")
    eligible = []
    for item in candidates:
        if item["decision"] == "eligible-not-selected" and \
           item["rejection_reasons"] == [] and \
           item["master_rank"] is None and item["stratum"] == stratum:
            rank = _integer(
                item["within_stratum_rank"],
                "replacement within_stratum_rank",
            )
            _text(item["composite_figi"], "replacement composite_figi")
            _text(item["share_class_figi"], "replacement share_class_figi")
            eligible.append((rank, item))
    if not eligible or len({rank for rank, _ in eligible}) != len(eligible):
        raise ValueError("same-stratum replacement order is invalid")
    return min(eligible, key=lambda pair: pair[0])[1]


def replacement_candidate(
    selection: Mapping[str, object],
    failed_ticker: str,
) -> Mapping[str, object]:
    """Choose the frozen first unused member of the failed member's stratum."""
    candidates = _candidates(selection)
    failed = _candidate(
        candidates, _ticker(failed_ticker, "failed ticker"),
    )
    return _replacement_candidate(candidates, failed)


def revised_manifest(
    base: Mapping[str, object],
    failed: Mapping[str, object],
    replacement: Mapping[str, object],
    *,
    purpose: str,
    declared_on: str,
) -> dict[str, object]:
    """Replace exactly one manifest entry while preserving every other value."""
    _object(base, MANIFEST_FIELDS, "base manifest")
    raw = base["series"]
    if not isinstance(raw, list) or len(raw) != 55 or \
       any(not isinstance(item, Mapping) or set(item) != SERIES_FIELDS
           for item in raw):
        raise ValueError("base manifest series are invalid")
    rank = _integer(failed.get("master_rank"), "failed master_rank")
    stratum = _integer(failed.get("stratum"), "failed stratum", 1)
    failed_ticker = _ticker(failed.get("ticker"), "failed ticker")
    replacement_ticker = _ticker(
        replacement.get("ticker"), "replacement ticker",
    )
    expected = {"stratum": f"liquidity-{stratum}", "ticker": failed_ticker}
    if rank >= len(raw) or dict(raw[rank]) != expected or \
       replacement.get("stratum") != stratum or \
       replacement_ticker in {
           item["ticker"] for item in raw if isinstance(item, Mapping)
       }:
        raise ValueError("manifest replacement does not match the selection")
    series = [dict(item) for item in raw]
    series[rank] = {
        "stratum": f"liquidity-{stratum}", "ticker": replacement_ticker,
    }
    result = {**base, "series": series}
    result["purpose"] = _text(purpose, "overlay manifest purpose")
    value = _text(declared_on, "overlay manifest declared_on")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("overlay manifest declared_on is invalid") from error
    if str(parsed) != value:
        raise ValueError("overlay manifest declared_on is invalid")
    result["declared_on"] = value
    return result


def _validate_selection(
    selection: Mapping[str, object],
    base: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    tree_files: Mapping[str, str],
    selection_path: str,
    manifest_path: str,
    manifest_sha256: str,
) -> None:
    raw = selection["master"]
    series = base["series"]
    if not isinstance(raw, list) or len(raw) != 55 or \
       selection["primary_cohort_size"] != len(raw) or \
       not isinstance(series, list) or len(series) != len(raw):
        raise ValueError("selection master is invalid")
    if selection["master_sha256"] != _canonical_sha256(raw):
        raise ValueError("selection master hash is invalid")
    members = tuple(
        _object(item, MEMBER_FIELDS, f"selection master[{index}]")
        for index, item in enumerate(raw)
    )
    expected = []
    for index, item in enumerate(members):
        ticker = _ticker(item["ticker"], "selection master ticker")
        stratum = _integer(
            item["stratum"], "selection master stratum", 1,
        )
        candidate = _candidate(candidates, ticker)
        if candidate["decision"] != "selected" or \
           candidate["master_rank"] != index or \
           candidate["stratum"] != stratum or any(
               candidate[name] != _text(
                   item[name], f"selection master {name}",
               )
               for name in ("composite_figi", "share_class_figi")
           ):
            raise ValueError("selection master candidate is inconsistent")
        expected.append({
            "stratum": f"liquidity-{stratum}", "ticker": ticker,
        })
    selected = {}
    for candidate in candidates:
        if candidate["decision"] != "selected":
            continue
        rank = _integer(
            candidate["master_rank"], "selected candidate master_rank",
        )
        _integer(candidate["stratum"], "selected candidate stratum", 1)
        if candidate["rejection_reasons"] != [] or \
           rank >= len(expected) or rank in selected:
            raise ValueError("selected candidate set is inconsistent")
        selected[rank] = candidate
    if set(selected) != set(range(len(expected))) or any(
        selected[index]["ticker"] != item["ticker"]
        for index, item in enumerate(expected)
    ):
        raise ValueError("selected candidate set does not match the master")
    if len({item["ticker"] for item in expected}) != len(expected):
        raise ValueError("selection master tickers must be unique")
    if series != expected:
        raise ValueError("base manifest does not match the selection master")

    raw_cohorts = selection["cohorts"]
    if not isinstance(raw_cohorts, Mapping) or \
       set(raw_cohorts) != {"11", "22", "33", "55"}:
        raise ValueError("selection cohorts are invalid")
    package_paths = {selection_path}
    for size in (11, 22, 33, 55):
        cohort = _object(
            raw_cohorts[str(size)], COHORT_FIELDS,
            f"selection cohort {size}",
        )
        members = cohort["members"]
        path = _relative(
            cohort["manifest"], f"selection cohort {size} manifest",
        ).as_posix()
        digest = _sha256(
            cohort["manifest_sha256"],
            f"selection cohort {size} manifest_sha256",
        )
        if cohort["size"] != size or \
           cohort["primary"] is not (size == 55) or \
           members != raw[:size] or \
           cohort["members_sha256"] != _canonical_sha256(members) or \
           tree_files.get(path) != digest:
            raise ValueError("selection cohort binding is invalid")
        package_paths.add(path)
        if size == 55 and (
            path != manifest_path or digest != manifest_sha256
        ):
            raise ValueError("base manifest binding is invalid")

    sources = selection["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("selection sources are invalid")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"selection source[{index}] is invalid")
        path = _relative(
            source.get("path"), f"selection source[{index}].path",
        ).as_posix()
        digest = _sha256(
            source.get("sha256"), f"selection source[{index}].sha256",
        )
        if path in package_paths or tree_files.get(path) != digest:
            raise ValueError("selection source binding is invalid")
        package_paths.add(path)
    if set(tree_files) != package_paths:
        raise ValueError("selection package membership is invalid")


def apply_overlay(
    policy_path: Path,
    output_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, object]:
    """Validate one immutable overlay and publish its revised manifest."""
    _regular(policy_path, "coverage overlay policy")
    if os.path.lexists(output_path):
        raise ValueError("coverage overlay output must not exist")
    with freeze_inputs((policy_path,)) as (frozen_policy,):
        policy = _policy(_read_json(frozen_policy.snapshot, "overlay policy"))
        selection_binding = _binding(
            policy["selection"], "coverage overlay selection",
        )
        manifest_binding = _binding(
            policy["base_manifest"], "coverage overlay base_manifest",
        )
        tree_binding = _tree_binding(policy["selection_tree"])
        tree_root = _bound_directory(
            root, tree_binding[0], "selection tree",
        )
        selection_path = _bound_path(
            root, selection_binding[0], "selection report",
        )
        manifest_path = _bound_path(
            root, manifest_binding[0], "base manifest",
        )
        if tree_root not in selection_path.parents or \
           tree_root not in manifest_path.parents:
            raise ValueError("overlay inputs must belong to the selection tree")
        resolved_output = Path(os.path.abspath(output_path)).resolve(
            strict=False,
        )
        if resolved_output == tree_root or tree_root in resolved_output.parents:
            raise ValueError("overlay output must be outside the selection tree")
        tree_paths, tree_files, tree_sha256 = _tree_state(tree_root)
        if (len(tree_paths), tree_sha256) != tree_binding[1:]:
            raise ValueError("selection tree does not match its binding")
        identities = regular_file_identities((policy_path, *tree_paths))
        require_disjoint(
            (policy_path, selection_path, manifest_path), (output_path,),
        )
        with freeze_inputs(
            (selection_path, manifest_path),
        ) as (frozen_selection, frozen_manifest):
            if frozen_selection.sha256 != selection_binding[1] or \
               frozen_manifest.sha256 != manifest_binding[1]:
                raise ValueError("coverage overlay input hash changed")
            selection = _read_json(
                frozen_selection.snapshot, "selection report",
            )
            base = _read_json(frozen_manifest.snapshot, "base manifest")
            UniverseManifest.read(frozen_manifest.snapshot)
            candidates = _candidates(selection)
            _validate_selection(
                selection, base, candidates, tree_files,
                selection_path.relative_to(tree_root).as_posix(),
                manifest_path.relative_to(tree_root).as_posix(),
                manifest_binding[1],
            )
            failed_value = _object(
                policy["failed_member"], FAILED_FIELDS,
                "coverage overlay failed_member",
            )
            failed = _candidate(candidates, failed_value["ticker"])
            if any(
                failed[name] != failed_value[name]
                for name in FAILED_FIELDS
            ):
                raise ValueError("failed member does not match the selection")
            replacement = _replacement_candidate(candidates, failed)
            declared = _object(
                policy["replacement"], REPLACEMENT_FIELDS,
                "coverage overlay replacement",
            )
            if any(replacement[name] != declared[name]
                   for name in REPLACEMENT_FIELDS):
                raise ValueError(
                    "replacement does not match the frozen selection rule",
                )
            result = revised_manifest(
                base, failed, replacement,
                purpose=policy["purpose"],
                declared_on=policy["declared_on"],
            )
            with tempfile.TemporaryDirectory(
                prefix="compose-mini-overlay-",
            ) as directory:
                validation = Path(directory) / "manifest.json"
                write_json(validation, result)
                UniverseManifest.read(validation)

            def verify() -> None:
                for path, label in (
                    (policy_path, "coverage overlay policy"),
                    (selection_path, "selection report"),
                    (manifest_path, "base manifest"),
                ):
                    _regular(path, label)
                current_paths, current_files, current_sha256 = \
                    _tree_state(tree_root)
                if current_paths != tree_paths or \
                   current_files != tree_files or \
                   current_sha256 != tree_sha256 or \
                   regular_file_identities(
                       (policy_path, *current_paths),
                   ) != identities:
                    raise ValueError(
                        "selection tree changed during the command",
                    )
                verify_frozen(
                    (frozen_policy, frozen_selection, frozen_manifest),
                )

            write_json_exclusive(output_path, result, before_link=verify)
            return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, metavar="POLICY")
    parser.add_argument("output", type=Path, metavar="OUTPUT")
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    try:
        result = apply_overlay(arguments.policy, arguments.output)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
