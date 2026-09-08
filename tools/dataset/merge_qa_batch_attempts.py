#!/usr/bin/env python3
"""Merge original QA batch attempts with a later rerun and rebuild coverage including failed-episode accounting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.batch_coverage import (  # noqa: E402
    CODEX_WORKTREE_PREFIX,
    build_batch_coverage,
    rewrite_codex_worktree_path,
    write_batch_coverage,
)

DEFAULT_ORIGINAL = REPOSITORY / "tmp/qa_pilot46_background_20260907_v1"
DEFAULT_RERUN = REPOSITORY / "tmp/qa_pilot46_rerun_20260907_v1"
DEFAULT_DRY = REPOSITORY / "tmp/gc_scaleup_dryrun_7x50_20260907/scaleup_dry_run_summary.json"


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_runner_module():
    path = Path(__file__).resolve().parent / "run_qa_batch.py"
    spec = importlib.util.spec_from_file_location("avengine_qa_batch_runner_merge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_batch_manifest(path: Path | None) -> dict[str, Any]:
    """Accept a raw batch_manifest or the wrapper written next to outcomes.json."""
    if path is None:
        return {}
    payload = _load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("manifest"), Mapping):
        inner = payload["manifest"]
        if isinstance(inner.get("episodes"), list):
            return dict(inner)
    if isinstance(payload, Mapping) and isinstance(payload.get("episodes"), list):
        return dict(payload)
    raise ValueError(f"not a batch manifest: {path}")


def resolve_manifest(original_root: Path, manifest_path: Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path))
    original_root = Path(original_root)
    candidates.append(original_root / "manifest.json")
    outcomes_path = original_root / "outcomes.json"
    if outcomes_path.is_file():
        outcomes = _load_json(outcomes_path)
        recorded = outcomes.get("manifest_path")
        if isinstance(recorded, str) and recorded:
            candidates.append(Path(recorded))
    for candidate in candidates:
        if candidate.is_file():
            return load_batch_manifest(candidate)
    return {}


def family_class(episode_id: str) -> tuple[str, str]:
    eid = episode_id
    if "apartment_" in eid:
        family = "apartment"
    elif "authored_" in eid:
        family = "authored"
    elif "kujiale_" in eid:
        family = "kujiale"
    elif "mp3d_" in eid:
        family = "mp3d"
    elif "hm3d_" in eid:
        family = "hm3d"
    else:
        family = "unknown"
    if "_human_human" in eid or eid.endswith("_single_active"):
        classes = "human"
    elif "_animal_animal" in eid:
        classes = "animal"
    elif "_device_device" in eid:
        classes = "device"
    elif "_human_animal" in eid:
        classes = "human+animal"
    elif "_human_device" in eid:
        classes = "human+device"
    elif "_animal_device" in eid:
        classes = "animal+device"
    else:
        classes = "unknown"
    return family, classes


def _add_asset_id(ids: list[str], seen: set[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in seen:
        seen.add(value)
        ids.append(value)


def asset_ids_from_entry(entry: Mapping[str, Any] | None, rec: Mapping[str, Any] | None = None) -> list[str]:
    """Collect intended asset IDs from the manifest row or saved request. Do not invent IDs."""
    ids: list[str] = []
    seen: set[str] = set()
    entry = entry or {}
    rec = rec or {}
    for row in entry.get("source_assignments") or []:
        if isinstance(row, Mapping):
            _add_asset_id(ids, seen, row.get("asset_id"))
    request = entry.get("request")
    if not isinstance(request, Mapping):
        request = rec.get("request")
    if isinstance(request, Mapping):
        for item in request.get("source_asset_ids") or []:
            _add_asset_id(ids, seen, item)
    request_path = rec.get("request_path")
    if not ids and isinstance(request_path, str) and Path(request_path).is_file():
        try:
            saved = _load_json(request_path)
        except (OSError, json.JSONDecodeError):
            saved = None
        if isinstance(saved, Mapping):
            for item in saved.get("source_asset_ids") or []:
                _add_asset_id(ids, seen, item)
    return ids


def room_id_from_entry(entry: Mapping[str, Any] | None, rec: Mapping[str, Any] | None = None) -> str | None:
    entry = entry or {}
    rec = rec or {}
    for value in (entry.get("room_id"), (entry.get("request") or {}).get("room_id") if isinstance(entry.get("request"), Mapping) else None, rec.get("room_id")):
        if isinstance(value, str) and value:
            return value
    request_path = rec.get("request_path")
    if isinstance(request_path, str) and Path(request_path).is_file():
        try:
            saved = _load_json(request_path)
        except (OSError, json.JSONDecodeError):
            saved = None
        if isinstance(saved, Mapping) and isinstance(saved.get("room_id"), str) and saved["room_id"]:
            return saved["room_id"]
    return None


def _review_path(attempt_root: Path) -> Path | None:
    for cand in (attempt_root / "episode/batch_review/review.json", attempt_root / "batch_review/review.json"):
        if cand.is_file():
            return cand
    return None


def _facts_questions(attempt_root: Path) -> tuple[Path | None, Path | None]:
    delivery = attempt_root / "episode/delivery"
    facts, questions = delivery / "facts.json", delivery / "questions.json"
    if facts.is_file() and questions.is_file():
        return facts, questions
    return None, None


def _episode_output_root(rec: Mapping[str, Any]) -> Path | None:
    raw = rec.get("episode_output_root")
    if isinstance(raw, str) and raw:
        return Path(raw)
    attempt = rec.get("attempt_root")
    if isinstance(attempt, str) and attempt:
        return Path(attempt) / "episode"
    return None


def backfill_failed_record(
    rec: Mapping[str, Any],
    *,
    manifest_entry: Mapping[str, Any] | None,
    runner: Any,
) -> dict[str, Any]:
    """Fill room_id / asset_ids / failure_stage / gap_state from the manifest and attempt artifacts."""
    out = deepcopy(dict(rec))
    entry = dict(manifest_entry or {})
    if not isinstance(out.get("room_id"), str) or not out.get("room_id"):
        room_id = room_id_from_entry(entry, out)
        if room_id:
            out["room_id"] = room_id
    if not isinstance(out.get("asset_ids"), list) or not out.get("asset_ids"):
        asset_ids = asset_ids_from_entry(entry, out)
        if asset_ids:
            out["asset_ids"] = asset_ids
    family, pair = family_class(str(out.get("episode_id") or ""))
    out["family"] = entry.get("room_family") or out.get("family") or family
    classes = entry.get("requested_source_classes")
    if isinstance(classes, list) and classes and all(isinstance(value, str) for value in classes):
        out["class_pair"] = "+".join(dict.fromkeys(classes))
    else:
        out.setdefault("class_pair", entry.get("class_pair") or pair)
    if out.get("status") == "delivered":
        out["gap_state"] = "produced"
        return out
    reason_code = out.get("reason_code")
    if reason_code in {"preallocation_gap", "preallocation_deficit"}:
        out["failure_stage"] = out.get("failure_stage") or "planning"
        out["gap_state"] = "evidence_missing_or_unsampled"
        out["failure_reason"] = out.get("failure_reason") or out.get("reason")
        out["reason_code"] = reason_code or "preallocation_gap"
        return out
    already = (
        isinstance(out.get("failure_stage"), str)
        and out.get("failure_stage")
        and isinstance(out.get("gap_state"), str)
        and out.get("gap_state") in {"interface_not_implemented", "evidence_missing_or_unsampled"}
        and reason_code not in {None, "controller_exit"}
    )
    if already:
        return out
    episode_root = _episode_output_root(out)
    attempt_root = Path(out["attempt_root"]) if out.get("attempt_root") else None
    stderr_path = Path(out["stderr_log"]) if out.get("stderr_log") and Path(out["stderr_log"]).is_file() else None
    stdout_path = Path(out["stdout_log"]) if out.get("stdout_log") and Path(out["stdout_log"]).is_file() else None
    if attempt_root is not None:
        if stderr_path is None and (attempt_root / "stderr.log").is_file():
            stderr_path = attempt_root / "stderr.log"
        if stdout_path is None and (attempt_root / "stdout.log").is_file():
            stdout_path = attempt_root / "stdout.log"
    if episode_root is None:
        return out
    classified = runner.classify_controller_failure(
        episode_output_root=episode_root,
        stderr_path=stderr_path,
        stdout_path=stdout_path,
        returncode=out.get("controller_returncode"),
    )
    out.update(classified)
    out["reason"] = classified["failure_reason"]
    return out


def load_attempt_outcomes(root: Path) -> dict[str, Any]:
    """Read a runner batch or a previously merged selection without changing it."""
    for name in ("outcomes.json", "merged_episodes.json"):
        path = Path(root) / name
        if path.is_file():
            payload = _load_json(path)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("episodes"), list):
                raise ValueError(f"invalid attempt outcomes: {path}")
            return dict(payload)
    summary_path = Path(root) / "rerender_summary.json"
    if summary_path.is_file():
        payload = _load_json(summary_path)
        if not isinstance(payload, list):
            raise ValueError(f"invalid rerender summary: {summary_path}")
        rows = []
        for raw in payload:
            episode_root = Path(raw["dest"])
            row = {"episode_id": raw["episode_id"], "attempt": episode_root.parent.name,
                   "episode_output_root": str(episode_root), "attempt_root": str(episode_root.parent),
                   "status": "delivered" if raw.get("status") == "ok" else "failed",
                   "controller_returncode": raw.get("finalize_returncode"),
                   "request_path": str(episode_root / "request.json"),
                   "stderr_log": str(episode_root / "attempt_03_finalize.log"),
                   "historical_rerender_summary": deepcopy(raw),
                   "historical_rerender_summary_path": str(summary_path),
                   "command_record_status": "not_recorded_in_rerender_summary"}
            review_path = episode_root / "batch_review/review.json"
            if review_path.is_file():
                row["review"] = _load_json(review_path)
            rows.append(row)
        return {"episodes": rows, "source": str(summary_path)}
    raise FileNotFoundError(f"no supported attempt summary in {root}")


def _attempt_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    # Commands describe historical execution. Never rebase them to the current tree.
    result = {key: deepcopy(row[key]) for key in (
        "attempt", "attempt_root", "episode_output_root", "status", "command",
        "request_path", "facts_path", "questions_path", "producer", "outcome_path",
        "failure_stage", "failure_reason", "reason", "reason_code", "gap_state",
        "exception_type", "controller_returncode", "stderr_log", "stdout_log", "failure_details",
        "failure_classification", "classification_status", "classification_unknown", "diagnostic",
        "historical_rerender_summary", "historical_rerender_summary_path", "command_record_status",
    ) if key in row}
    audit = (row.get("review") or {}).get("audit") if isinstance(row.get("review"), Mapping) else None
    if isinstance(audit, Mapping) and "command" in audit:
        result["review_audit_command"] = deepcopy(audit["command"])
    return result


def merge_episode_records(
    original_outcomes: Mapping[str, Any],
    rerun_outcomes: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    runner: Any | None = None,
) -> list[dict[str, Any]]:
    """Prefer rerun rows when present; backfill failure accounting on every undelivered cell."""
    runner = runner or load_runner_module()
    entries = {row["episode_id"]: row for row in (manifest or {}).get("episodes", []) if isinstance(row, Mapping)}
    orig_eps = {e["episode_id"]: e for e in original_outcomes["episodes"]}
    rerun_eps = {e["episode_id"]: e for e in rerun_outcomes.get("episodes", [])}
    extras = [eid for eid in rerun_eps if eid not in orig_eps]
    if any(eid not in entries for eid in extras):
        raise ValueError("supplemental rerun Episodes must be declared in the manifest")
    merged: list[dict[str, Any]] = []
    for eid in [*orig_eps, *extras]:
        row = orig_eps.get(eid)
        if eid in rerun_eps:
            rec = deepcopy(rerun_eps[eid])
            rec["attempt"] = rec.get("attempt") or "attempt_02"
            if row is not None:
                rec["supersedes_attempt_01"] = row.get("supersedes_attempt_01") or row.get("attempt_root")
                history = deepcopy(row.get("prior_attempts") or [])
                history.append(_attempt_snapshot(row))
                rec["prior_attempts"] = history
                rec["supersedes_attempt"] = row.get("attempt_root")
        else:
            rec = deepcopy(row)
            rec["attempt"] = rec.get("attempt") or "attempt_01"
        attempt_root = Path(rec["attempt_root"]) if rec.get("attempt_root") else None
        if attempt_root is not None:
            facts, questions = _facts_questions(attempt_root)
            rec["facts_path"] = str(facts) if facts else rec.get("facts_path")
            rec["questions_path"] = str(questions) if questions else rec.get("questions_path")
        rec["evidence_path"] = rec.get("facts_path") or rec.get("outcome_path") or rec.get("attempt_root")
        rec = backfill_failed_record(rec, manifest_entry=entries.get(eid), runner=runner)
        merged.append(rec)
    return merged


def failed_coverage_records(merged: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for rec in merged:
        if rec.get("status") == "delivered":
            continue
        gap_state = rec.get("gap_state")
        if gap_state not in {"interface_not_implemented", "evidence_missing_or_unsampled"}:
            continue
        room_id = rec.get("room_id")
        asset_ids = rec.get("asset_ids") or []
        if not isinstance(room_id, str) or not room_id:
            continue
        if not isinstance(asset_ids, list) or not asset_ids:
            continue
        records.append({
            "episode_id": rec.get("episode_id"),
            "room_id": room_id,
            "asset_ids": list(asset_ids),
            "gap_state": gap_state,
            "failure_stage": rec.get("failure_stage"),
            "failure_reason": rec.get("failure_reason") or rec.get("reason"),
            **{key: deepcopy(rec[key]) for key in (
                "reason_code", "failure_code", "exception_type", "failure_classification",
                "classification_unknown", "classification_status", "diagnostic",
            ) if key in rec},
        })
    return records


def production_registry_paths(repository: Path) -> dict[str, str]:
    repo = Path(repository).resolve()
    return {
        "asset_inventory": str((repo / "tmp/qa_generalized_sampler_review_20260906_v1/full_source_scope_inventory.json").resolve()),
        "room_catalog": str((repo / "examples/rooms/packages/catalog.json").resolve()),
        "runtime_registry": str((repo / "examples/runtime/source_asset_runtime_profiles.json").resolve()),
    }


def _remap_payload_paths(payload: dict[str, Any], *, repository: Path) -> dict[str, Any]:
    repo = Path(repository).resolve()
    defaults = production_registry_paths(repo)
    out = deepcopy(payload)
    out["asset_inventory"] = rewrite_codex_worktree_path(
        out.get("asset_inventory", defaults["asset_inventory"]),
        repository=repo,
        fallback=Path(defaults["asset_inventory"]),
    )
    out["room_catalog"] = rewrite_codex_worktree_path(
        out.get("room_catalog", defaults["room_catalog"]),
        repository=repo,
        fallback=Path(defaults["room_catalog"]),
    )
    out["runtime_registry"] = rewrite_codex_worktree_path(
        out.get("runtime_registry", defaults["runtime_registry"]),
        repository=repo,
        fallback=Path(defaults["runtime_registry"]),
    )
    for key in ("asset_inventory", "room_catalog", "runtime_registry"):
        value = out.get(key)
        if isinstance(value, str) and CODEX_WORKTREE_PREFIX in value:
            out[key] = defaults[key]
    return out


def build_coverage_manifest(
    merged: list[Mapping[str, Any]],
    *,
    original_inputs: Mapping[str, Any],
    rerun_inputs: Mapping[str, Any] | None,
    repository: Path,
    original_root: Path,
    rerun_root: Path,
) -> dict[str, Any]:
    orig_by_id = {e["episode_id"]: e for e in original_inputs.get("episodes", []) if isinstance(e, Mapping)}
    rerun_by_id = {e["episode_id"]: e for e in (rerun_inputs or {}).get("episodes") or [] if isinstance(e, Mapping)}
    coverage_episodes = []
    for rec in merged:
        eid = rec["episode_id"]
        if rec.get("status") != "delivered":
            continue
        src = deepcopy(rerun_by_id.get(eid) or orig_by_id.get(eid) or {})
        if rec.get("facts_path") and rec.get("questions_path"):
            src.update({
                "episode_id": eid,
                "room_id": rec.get("room_id"),
                "family": rec.get("family"),
                "facts": rec["facts_path"],
                "questions": rec["questions_path"],
            })
        elif not src:
            src = None
        if src is not None:
            coverage_episodes.append(src)
    coverage_manifest = _remap_payload_paths(dict(original_inputs), repository=Path(repository))
    coverage_manifest["episodes"] = coverage_episodes
    coverage_manifest["failed_episodes"] = failed_coverage_records(merged)
    coverage_manifest["_merged_from"] = {"original": str(original_root), "rerun": str(rerun_root)}
    return coverage_manifest


def attach_exposure_gates(
    merged: list[dict[str, Any]],
    *,
    output_root: Path,
    apply_gate: bool,
) -> None:
    if not apply_gate:
        return
    from avengine.qa.exposure_gate import apply_exposure_gate
    for rec in merged:
        if rec.get("status") != "delivered":
            continue
        attempt_root = Path(rec["attempt_root"]) if rec.get("attempt_root") else None
        if attempt_root is None:
            continue
        review = _review_path(attempt_root)
        gate = None
        if review:
            payload = _load_json(review)
            gate = payload.get("exposure_gate")
        if gate is None or (isinstance(gate, dict) and gate.get("status") is None):
            sidecar = output_root / "exposure_gate_offline" / f"{rec['episode_id']}.json"
            dummy = {"episode_id": rec["episode_id"], "status": "delivered"}
            gated = apply_exposure_gate(dummy, attempt_root / "episode")
            gate = gated.get("exposure_gate")
            _write_json(sidecar, gated)
            rec["exposure_gate_offline"] = str(sidecar)
        rec["exposure_gate"] = gate
        rec["exposure_gate_status"] = None if not isinstance(gate, dict) else gate.get("status")


def merge_attempts(
    *,
    original_root: Path,
    rerun_root: Path,
    output_root: Path,
    repository: Path = REPOSITORY,
    manifest_path: Path | None = None,
    dry_run_summary: Path | None = DEFAULT_DRY,
    apply_exposure: bool = True,
    build_coverage: bool = True,
    runner: Any | None = None,
    later_attempt_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    original_root = Path(original_root)
    rerun_root = Path(rerun_root)
    output_root = Path(output_root)
    repository = Path(repository).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing existing merge output: {output_root}")
    orig = load_attempt_outcomes(original_root)
    rerun = load_attempt_outcomes(rerun_root)
    manifest = resolve_manifest(original_root, manifest_path)
    runner = runner or load_runner_module()
    merged = merge_episode_records(orig, rerun, manifest=manifest, runner=runner)
    for later_root in later_attempt_roots:
        merged = merge_episode_records(
            {"episodes": merged}, load_attempt_outcomes(Path(later_root)),
            manifest=manifest, runner=runner,
        )
    output_root.mkdir(parents=True, exist_ok=False)
    attach_exposure_gates(merged, output_root=output_root, apply_gate=apply_exposure)
    counts = Counter(r["status"] for r in merged)
    gate_counts = Counter(r.get("exposure_gate_status") for r in merged if r["status"] == "delivered")
    table = {
        "schema": "avengine_qa_pilot46_merged_v1",
        "original_root": str(original_root),
        "rerun_root": str(rerun_root),
        "later_attempt_roots": [str(path) for path in later_attempt_roots],
        "episode_denominator": len(merged),
        "status_counts": dict(counts),
        "delivered_exposure_gate": dict(gate_counts),
        "episodes": merged,
    }
    _write_json(output_root / "merged_episodes.json", table)
    entries = {entry["episode_id"]: entry for entry in manifest.get("episodes", [])}
    current_commands = []
    for record in merged:
        entry = entries.get(record["episode_id"], {})
        request_path = entry.get("request_path")
        if not request_path:
            continue
        replay_root = repository / "tmp" / (output_root.name + "_current_replay") / record["episode_id"]
        current_commands.append({
            "episode_id": record["episode_id"],
            "status": "proposed_not_executed",
            "command": [sys.executable, str(repository / "tools/studio/run_qa_episode.py"),
                        "--request", str(request_path), "--output", str(replay_root)],
            "cwd": str(repository),
            "pythonpath": f"{repository / 'src'}:{repository / 'tmp/native_python_addons_v1'}",
            "request_path": str(request_path),
            "historical_commands_preserved_in": "merged_episodes.json",
        })
    _write_json(output_root / "current_replay_commands.json", current_commands)

    orig_inputs_path = original_root / "summary/coverage_inputs.json"
    rerun_inputs_path = rerun_root / "summary/coverage_inputs.json"
    orig_inputs = _load_json(orig_inputs_path) if orig_inputs_path.is_file() else {"episodes": []}
    rerun_inputs = _load_json(rerun_inputs_path) if rerun_inputs_path.is_file() else {}
    coverage_manifest = build_coverage_manifest(
        merged,
        original_inputs=orig_inputs,
        rerun_inputs=rerun_inputs,
        repository=repository,
        original_root=original_root,
        rerun_root=rerun_root,
    )
    _write_json(output_root / "coverage_inputs.json", coverage_manifest)
    _write_json(output_root / "failed_episodes.json", coverage_manifest["failed_episodes"])

    paths = {}
    states: Counter[str] = Counter()
    if build_coverage:
        coverage = build_batch_coverage(coverage_manifest, repository=repository)
        paths = write_batch_coverage(coverage, output_root / "coverage")
        states = Counter(row.get("state") for row in coverage.get("rows", []))
        _write_json(output_root / "coverage_state_counts.json", dict(states))

    family_class_clips: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in merged:
        if rec.get("status") == "delivered" and rec.get("exposure_gate_status") == "pass":
            fam, pair = rec["family"], rec["class_pair"]
            for token in ("human", "animal", "device"):
                if token in pair or (pair == "human" and token == "human"):
                    family_class_clips[(fam, token)].append(rec["episode_id"])

    dry = _load_json(dry_run_summary) if dry_run_summary and Path(dry_run_summary).is_file() else {}
    cells = [
        {
            "episode_id": r["episode_id"],
            "family": r.get("family"),
            "class_pair": r.get("class_pair"),
            "status": r.get("status"),
            "attempt": r.get("attempt"),
            "failure_stage": r.get("failure_stage"),
            "gap_state": r.get("gap_state"),
            "room_id": r.get("room_id"),
            "asset_ids": r.get("asset_ids"),
        }
        for r in merged
    ]
    question_counts: Counter[str] = Counter()
    missing_question_files = []
    for record in merged:
        if record.get("status") != "delivered":
            continue
        question_path = record.get("questions_path")
        if not question_path or not Path(question_path).is_file():
            missing_question_files.append(record["episode_id"])
            continue
        question_counts.update(item["qa_id"] for item in _load_json(Path(question_path)).get("items", []))
    qa_ids = [f"QA-{number:02d}" for number in range(1, 25)]
    summary = {
        "questions": {
            "valid_item_count": sum(question_counts.values()),
            "covered_type_count": sum(bool(question_counts[qa]) for qa in qa_ids),
            "count_by_qa": {qa: question_counts[qa] for qa in qa_ids},
            "missing_qa_types": [qa for qa in qa_ids if not question_counts[qa]],
            "missing_question_files": missing_question_files,
        },
        "merged_status_counts": dict(counts),
        "delivered_exposure_gate": dict(gate_counts),
        "coverage_row_states": dict(states),
        "coverage_paths": paths,
        "failed_episode_coverage_count": len(coverage_manifest["failed_episodes"]),
        "family_class_gated_clips": {f"{k[0]}|{k[1]}": v for k, v in sorted(family_class_clips.items())},
        "cells": cells,
        "scaleup_dry_run": {
            "path": str(dry_run_summary) if dry_run_summary else None,
            "requested_episode_count": dry.get("requested_episode_count"),
            "repeat_deficit_count": dry.get("repeat_deficit_count"),
            "preallocation_gap_counts": dry.get("preallocation_gap_counts"),
            "off_screen_portrait_count": dry.get("off_screen_portrait_count"),
            "crosstab_min_distinct": (dry.get("class_pair_condition_group_crosstab") or {}).get(
                "min_distinct_groups_per_class_pair"
            ),
            "crosstab_meets_acceptance": (dry.get("class_pair_condition_group_crosstab") or {}).get("meets_acceptance"),
        },
        "failed_or_blocked": [
            {
                "episode_id": r["episode_id"],
                "status": r.get("status"),
                "attempt": r.get("attempt"),
                "failure_stage": r.get("failure_stage"),
                "gap_state": r.get("gap_state"),
                "room_id": r.get("room_id"),
                "asset_ids": r.get("asset_ids"),
                "reason": r.get("failure_reason") or r.get("reason"),
                "reason_code": r.get("reason_code"),
                "failure_classification": r.get("failure_classification"),
                "diagnostic": deepcopy(r.get("diagnostic")),
                "classification_unknown": r.get("classification_unknown"),
                "evidence_path": r.get("attempt_root"),
            }
            for r in merged
            if r.get("status") != "delivered"
        ],
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--rerun", type=Path, default=DEFAULT_RERUN)
    parser.add_argument("--later-attempt-root", type=Path, action="append", default=[],
                        help="Overlay another attempt batch in supplied order; preserve prior commands")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run-summary", type=Path, default=DEFAULT_DRY)
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-exposure-gate", action="store_true")
    args = parser.parse_args()
    summary = merge_attempts(
        original_root=args.original,
        rerun_root=args.rerun,
        output_root=args.output,
        repository=args.repository,
        manifest_path=args.manifest,
        dry_run_summary=args.dry_run_summary,
        apply_exposure=not args.skip_exposure_gate,
        build_coverage=not args.skip_coverage,
        later_attempt_roots=args.later_attempt_root,
    )
    printable = {k: summary[k] for k in summary if k not in {"family_class_gated_clips", "cells"}}
    print(json.dumps(printable, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
