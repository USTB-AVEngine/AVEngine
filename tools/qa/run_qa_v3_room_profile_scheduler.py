#!/usr/bin/env python3
"""Room-centric QA-v3 scene x profile scheduler.

For every registered scene, attempt every requested question profile with an
independent fresh output. A failed pair never suppresses another profile in
the same room. The scheduler reads geometry/configuration evidence only; it
does not read model scores, missing-modality probes, or downstream outcomes.

Finite randomized search failure is reported as not_found_within_budget.
scene_infeasible is reserved for an explicit exhaustive proof in a producer
manifest and is never inferred from repeated failure alone.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from design_qa_v3_scene_batch import main as design_scene_batch_main  # noqa: E402
from design_qa_v3_extended_profile import main as design_extended_main  # noqa: E402
from design_qa_v3_offscreen_identity import (  # noqa: E402
    main as design_offscreen_identity_main,
)


PAIR_STATUSES = (
    "generated",
    "not_found_within_budget",
    "scene_infeasible",
    "pixel_rejected",
    "pipeline_error",
    "profile_not_implemented",
    "resource_unavailable",
    "not_scheduled",
)


from scene_sampler import read_scene_config
from qa_v3_request import normalize_answer_forms, plan_room_questions, read_qa_params


@dataclass
class SceneSpec:
    source_path: Path
    scene_id: str
    config: dict | None
    load_error: str | None = None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"identifier has no safe path component: {value!r}")
    return safe


def _load_scene_specs(paths: list[Path]) -> list[SceneSpec]:
    specs = []
    seen_ids = set()
    seen_safe = set()
    for path in paths:
        try:
            value = read_scene_config(path)
            if not isinstance(value, dict):
                raise ValueError("scene config must be a JSON object")
            scene_id = str(value.get("scene_id") or "").strip()
            if not scene_id:
                raise ValueError("scene config has no non-empty scene_id")
            spec = SceneSpec(path.resolve(), scene_id, value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            scene_id = f"invalid_{path.stem}"
            spec = SceneSpec(
                path.resolve(), scene_id, None,
                f"{type(exc).__name__}: {exc}")
        safe = _safe_component(scene_id)
        if scene_id in seen_ids or safe in seen_safe:
            raise ValueError(f"duplicate or path-colliding scene id: {scene_id}")
        seen_ids.add(scene_id)
        seen_safe.add(safe)
        specs.append(spec)
    if not specs:
        raise ValueError("at least one scene config is required")
    return specs


def _load_profile_catalog(path: Path) -> dict[str, dict]:
    value = _read_json(path)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("profile catalog must be a JSON list or profile object")
    catalog = {}
    safe_ids = set()
    for profile in value:
        if not isinstance(profile, dict):
            raise ValueError("every profile must be an object")
        profile_id = str(profile.get("id") or "").strip()
        if not profile_id or profile_id in catalog:
            raise ValueError(
                f"profile ids must be non-empty and unique: {profile_id!r}")
        safe = _safe_component(profile_id)
        if safe in safe_ids:
            raise ValueError(f"profile ids collide as paths: {profile_id!r}")
        safe_ids.add(safe)
        catalog[profile_id] = profile
    if not catalog:
        raise ValueError("profile catalog is empty")
    return catalog


def _requested_profiles(values: list[str] | None,
                        catalog: dict[str, dict]) -> list[str]:
    requested = list(values or catalog)
    if not requested:
        raise ValueError("at least one profile must be requested")
    if len(requested) != len(set(requested)):
        raise ValueError(f"requested profiles are not unique: {requested}")
    safe = [_safe_component(value) for value in requested]
    if len(safe) != len(set(safe)):
        raise ValueError("requested profile ids collide as paths")
    return requested


def _point_ids(value, *, owner: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{owner} must be a list of point IDs")
    result = []
    for index, point_id in enumerate(value):
        if not isinstance(point_id, str) or not point_id:
            raise ValueError(
                f"{owner}[{index}] must be a non-empty string point ID"
            )
        result.append(point_id)
    if len(result) != len(set(result)):
        raise ValueError(f"{owner} contains duplicate point IDs")
    return result


def _pixel_point_ids(pixel_result: dict) -> list[str] | None:
    fields = (
        "point_ids",
        "candidate_point_ids",
        "candidate_ids",
    )
    present = [
        (field, pixel_result[field])
        for field in fields
        if field in pixel_result
    ]
    if not present:
        return None
    normalized = [
        (field, _point_ids(value, owner=f"pixel result {field}"))
        for field, value in present
    ]
    first = normalized[0][1]
    for field, value in normalized[1:]:
        if value != first:
            raise ValueError(
                f"pixel result point ID fields disagree: "
                f"{normalized[0][0]} versus {field}"
            )
    return first


def _pixel_outcome_ids(
    pixel_result: dict,
) -> tuple[list[str], list[str]] | None:
    present = [
        field for field in ("passed_point_ids", "rejected_point_ids")
        if field in pixel_result
    ]
    if not present:
        return None
    if len(present) != 2:
        raise ValueError(
            "pixel result must provide both passed_point_ids and "
            "rejected_point_ids"
        )
    passed = _point_ids(
        pixel_result["passed_point_ids"],
        owner="pixel result passed_point_ids",
    )
    rejected = _point_ids(
        pixel_result["rejected_point_ids"],
        owner="pixel result rejected_point_ids",
    )
    overlap = sorted(set(passed) & set(rejected))
    if overlap:
        raise ValueError(
            f"pixel passed/rejected point IDs overlap: {overlap}"
        )
    if _pixel_point_ids(pixel_result) is None:
        raise ValueError(
            "pixel result with per-point outcomes must also provide point_ids"
        )
    return passed, rejected


def _manifest_point_ids(manifest: dict) -> list[str] | None:
    for field in ("generated_main_point_ids", "generated_point_ids",
                  "main_point_ids"):
        if field in manifest:
            return _point_ids(
                manifest[field], owner=f"batch manifest {field}"
            )
    for field in ("records", "selected", "candidates"):
        records = manifest.get(field)
        if records is None:
            continue
        if not isinstance(records, list):
            raise ValueError(f"batch manifest {field} must be a list")
        result = []
        for index, record in enumerate(records):
            if isinstance(record, str):
                result.append(record)
                continue
            if not isinstance(record, dict):
                raise ValueError(
                    f"batch manifest {field}[{index}] must be an object"
                )
            if record.get("variant") not in (None, "main"):
                continue
            point_id = record.get("point_id")
            if point_id is None:
                raise ValueError(
                    f"batch manifest {field}[{index}] has no point_id"
                )
            result.append(point_id)
        return _point_ids(result, owner=f"batch manifest {field}")
    return None


def _generated_main_point_ids(manifest: dict, batch_root: Path) -> list[str] | None:
    declared = _manifest_point_ids(manifest)
    if declared is not None:
        return declared
    facts_path = batch_root / "facts.jsonl"
    if not facts_path.is_file():
        return None
    result = []
    with facts_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid facts.jsonl row {facts_path}:{line_no}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"facts.jsonl row {facts_path}:{line_no} is not an object"
                )
            if record.get("variant") not in (None, "main"):
                continue
            point_id = record.get("point_id")
            if point_id is None:
                raise ValueError(
                    f"facts.jsonl row {facts_path}:{line_no} has no point_id"
                )
            result.append(point_id)
    return _point_ids(result, owner=f"facts.jsonl {facts_path}")


def _load_pixel_results(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    value = _read_json(path)
    rows = value.get("results") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("pixel results must be a list or {results: [...]}")
    result = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["profile_id"]))
        if key in result:
            raise ValueError(f"duplicate pixel result for {key}")
        attempted = int(row.get("attempted", 0))
        passed = int(row.get("passed", 0))
        rejected = int(row.get("rejected", 0))
        if min(attempted, passed, rejected) < 0 or passed + rejected != attempted:
            raise ValueError(f"pixel counts do not close for {key}")
        _pixel_point_ids(row)
        _pixel_outcome_ids(row)
        result[key] = dict(row)
    return result


def classify_manifest(manifest: dict, pixel_result: dict | None = None) -> dict:
    counts = manifest.get("counts") or {}
    search = manifest.get("search") or {}
    candidate_value = counts.get("geometry_candidates")
    if candidate_value is None:
        candidate_value = counts.get("candidates", 0)
    candidates = int(candidate_value)
    requested = int(counts.get("cells_requested", 0))
    rejected = int(counts.get("rejected", 0))
    if (min(requested, candidates, rejected) < 0
            or candidates + rejected != requested):
        raise ValueError(
            "batch counts do not close: "
            f"requested={requested}, generated={candidates}, rejected={rejected}")
    quota_shortfall = max(0, requested - candidates)
    quota_status = ("empty" if candidates == 0 else
                    "filled" if quota_shortfall == 0 else "partial")
    record = {
        "attempt_status": None,
        "requested_cells": requested,
        "quota_status": quota_status,
        "quota_shortfall": quota_shortfall,
        "geometry_candidates": candidates,
        "generation_rejected": rejected,
        "evaluated_combinations": int(
            search.get("combinations_evaluated", 0)),
        "search_budget_exhausted": int(search.get("budget_exhausted", 0)),
        "rejection_reasons": dict(search.get("by_reason") or {}),
        "evidence_class": manifest.get(
            "evidence_class", "geometry_candidate"),
        "qualification_claim": False,
        "pixel": {"status": "not_run"},
    }
    proof = manifest.get("feasibility_proof") or {}
    if candidates == 0:
        resources = manifest.get("resource_status") or {}
        if (resources.get("status") == "unavailable"
                and resources.get("method") == "registry_preflight"):
            record["attempt_status"] = "resource_unavailable"
            record["resource_status"] = resources
            record["evidence_class"] = "resource_unavailable"
        elif (proof.get("status") == "infeasible"
                and proof.get("method") == "exhaustive"):
            record["attempt_status"] = "scene_infeasible"
            record["infeasibility_proof"] = proof
        else:
            record["attempt_status"] = "not_found_within_budget"
            record["boundary"] = (
                "No candidate was found under this search budget; this is not "
                "proof that the scene can never support the profile.")
        return record

    if pixel_result is not None:
        pixel = {
            "status": "complete" if pixel_result.get(
                "complete_for_geometry_candidates") else "partial",
            "attempted": int(pixel_result.get("attempted", 0)),
            "passed": int(pixel_result.get("passed", 0)),
            "rejected": int(pixel_result.get("rejected", 0)),
            "rejection_reasons": dict(
                pixel_result.get("rejection_reasons") or {}),
        }
        if (min(pixel["attempted"], pixel["passed"], pixel["rejected"]) < 0
                or pixel["passed"] + pixel["rejected"] != pixel["attempted"]
                or pixel["attempted"] > candidates):
            raise ValueError(
                "pixel counts do not close within geometry candidates: "
                f"attempted={pixel['attempted']}, passed={pixel['passed']}, "
                f"rejected={pixel['rejected']}, candidates={candidates}")
        pixel_ids = _pixel_point_ids(pixel_result)
        outcome_ids = _pixel_outcome_ids(pixel_result)
        generated_ids = _manifest_point_ids(manifest)
        if outcome_ids is not None:
            passed_ids, rejected_ids = outcome_ids
            pixel["passed_point_ids"] = passed_ids
            pixel["rejected_point_ids"] = rejected_ids
            record["passed_point_ids"] = passed_ids
            record["rejected_point_ids"] = rejected_ids
            union_ids = set(passed_ids) | set(rejected_ids)
        else:
            union_ids = None
            record["passed_point_ids"] = None
            record["rejected_point_ids"] = None
        if generated_ids is None:
            pixel.update({
                "status": "unverified",
                "identity_status": "batch_candidate_ids_missing",
                "qualification_blocked": True,
            })
        elif len(generated_ids) != candidates:
            raise ValueError(
                "batch generated main point IDs do not match geometry "
                f"candidate count: ids={len(generated_ids)}, "
                f"candidates={candidates}")
        elif pixel_ids is None:
            pixel.update({
                "status": "unverified",
                "identity_status": "legacy_pixel_counts_only",
                "qualification_blocked": True,
            })
        else:
            generated_set = set(generated_ids)
            pixel_set = set(pixel_ids)
            if pixel["attempted"] != len(pixel_ids):
                raise ValueError(
                    "pixel attempted count does not match point ID count: "
                    f"attempted={pixel['attempted']}, ids={len(pixel_ids)}")
            if not pixel_set <= generated_set:
                raise ValueError(
                    "pixel point IDs are not generated main candidate IDs: "
                    f"extra={sorted(pixel_set - generated_set)}")
            if outcome_ids is None:
                pixel.update({
                    "status": "unverified",
                    "identity_status": "legacy_pixel_outcomes_missing",
                    "qualification_blocked": True,
                })
            else:
                if len(passed_ids) != pixel["passed"]:
                    raise ValueError(
                        "pixel passed count does not match passed_point_ids: "
                        f"passed={pixel['passed']}, ids={len(passed_ids)}")
                if len(rejected_ids) != pixel["rejected"]:
                    raise ValueError(
                        "pixel rejected count does not match rejected_point_ids: "
                        f"rejected={pixel['rejected']}, ids={len(rejected_ids)}")
                if union_ids != pixel_set:
                    raise ValueError(
                        "pixel passed/rejected IDs do not match point_ids: "
                        f"missing={sorted(pixel_set - union_ids)}, "
                        f"extra={sorted(union_ids - pixel_set)}")
                if pixel["status"] == "complete" and pixel_set != generated_set:
                    raise ValueError(
                        "complete pixel result point IDs do not cover generated "
                        f"main candidates: missing={sorted(generated_set - pixel_set)}"
                    )
                pixel["identity_status"] = "verified"
                if pixel["status"] == "complete":
                    if pixel["passed"] == 0:
                        record["attempt_status"] = "pixel_rejected"
                        record["evidence_class"] = "pixel_rejected"
                        record["pixel"] = pixel
                        return record
                    record["evidence_class"] = "pixel_qualified_candidate"
        record["pixel"] = pixel
    record["attempt_status"] = "generated"
    return record


def _invoke_pair(*, scene_config: Path, profile_config: Path,
                 params: Path, batch_root: Path, cells: int, seed: str,
                 snapshot_content: str) -> dict:
    profile_value = _read_json(profile_config)
    if isinstance(profile_value, list):
        if len(profile_value) != 1 or not isinstance(profile_value[0], dict):
            raise ValueError(
                "scheduler profile snapshot must contain exactly one profile"
            )
        profile = profile_value[0]
    elif isinstance(profile_value, dict):
        profile = profile_value
    else:
        raise ValueError("scheduler profile snapshot must be an object or one-item list")
    backend = profile.get("execution_backend", "scene")
    if backend is None:
        backend = "scene"
    if not isinstance(backend, str) or not backend:
        raise ValueError("profile execution_backend must be a non-empty string")
    common = [
        "--scene-config", str(scene_config),
        "--params", str(params),
        "--out-root", str(batch_root),
        "--cells", str(cells),
        "--seed", seed,
        "--snapshot-content", snapshot_content,
    ]
    if backend == "offscreen_identity":
        argv = ["--profile", str(profile_config), *common]
        producer = design_offscreen_identity_main
    elif backend == "extended":
        argv = ["--profiles", str(profile_config), *common]
        producer = design_extended_main
    elif backend == "scene":
        argv = ["--profiles", str(profile_config), *common]
        producer = design_scene_batch_main
    else:
        raise ValueError(f"unsupported profile execution_backend: {backend!r}")
    code = producer(argv)
    manifest_path = batch_root / "batch_manifest.json"
    # The offscreen producer uses exit 1 to report a bounded search with no
    # candidates, but still writes its manifest.  Let classify_manifest apply
    # the same quota/shortfall rules as the other producers.
    if code != 0 and not (
        backend == "offscreen_identity" and manifest_path.is_file()
    ):
        raise RuntimeError(
            f"{backend} design producer returned {code}"
        )
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{backend} design producer wrote no batch_manifest.json"
        )
    return _read_json(manifest_path)


PairRunner = Callable[..., dict]


def _attempt_record(scene_id: str, profile_id: str, pair_dir: Path,
                    requested_cells: int) -> dict:
    return {
        "scene_id": scene_id,
        "profile_id": profile_id,
        "attempt_status": None,
        "requested_cells": requested_cells,
        "quota_status": "not_run",
        "quota_shortfall": requested_cells,
        "pair_output": str(pair_dir.resolve()),
        "qualification_claim": False,
    }


def run_scheduler(*, scene_specs: list[SceneSpec],
                  profile_catalog: dict[str, dict],
                  requested_profiles: list[str], params_value: dict,
                  params_source: Path, out_root: Path, cells: int | dict[str, int], seed: str,
                  snapshot_content: str,
                  pixel_results: dict[tuple[str, str], dict],
                  runner: PairRunner = _invoke_pair,
                  request_plan: dict | None = None) -> dict:
    cells_by_profile = (dict(cells) if isinstance(cells, dict)
                        else {pid: cells for pid in requested_profiles})
    if set(cells_by_profile) != set(requested_profiles) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in cells_by_profile.values()):
        raise ValueError("cells must give a non-negative integer for every requested profile")
    inputs_root = out_root / "inputs"
    rooms_root = out_root / "rooms"
    inputs_root.mkdir(parents=True)
    rooms_root.mkdir()
    params_snapshot = inputs_root / "params.json"
    _write_json(params_snapshot, params_value)

    profile_snapshots = {}
    for profile_id, profile in profile_catalog.items():
        profile_path = inputs_root / "profiles" / (
            _safe_component(profile_id) + ".json")
        backend = profile.get("execution_backend", "scene")
        snapshot = profile if backend == "offscreen_identity" else [profile]
        _write_json(profile_path, snapshot)
        profile_snapshots[profile_id] = profile_path

    rows = []
    for scene in scene_specs:
        scene_safe = _safe_component(scene.scene_id)
        scene_config = scene.config or {}
        scene_asset_id = str(
            scene_config.get("scene_asset_id") or scene.scene_id)
        route_domain = scene_config.get("route_domain")
        backend = scene_config.get("backend")
        room_root = rooms_root / scene_safe
        room_root.mkdir()
        scene_snapshot = inputs_root / "scenes" / (scene_safe + ".json")
        if scene.config is not None:
            _write_json(scene_snapshot, scene.config)
        room_rows = []
        for profile_id in requested_profiles:
            pair_dir = room_root / "profiles" / _safe_component(profile_id)
            pair_dir.mkdir(parents=True)
            requested_cells = cells_by_profile[profile_id]
            record = _attempt_record(
                scene.scene_id, profile_id, pair_dir, requested_cells)
            record.update({
                "scene_asset_id": scene_asset_id,
                "route_domain": route_domain,
                "backend": backend,
                "execution_backend": (
                    profile_catalog.get(profile_id, {}).get(
                        "execution_backend", "scene")
                    if profile_id in profile_catalog else None
                ),
            })
            if profile_id not in profile_catalog:
                record.update({
                    "attempt_status": "profile_not_implemented",
                    "detail": (
                        "Requested profile is absent from the implemented "
                        "profile catalog."),
                    "evidence_class": "not_run",
                })
            elif requested_cells == 0:
                record.update({
                    "attempt_status": "not_scheduled",
                    "quota_status": "filled", "quota_shortfall": 0,
                    "geometry_candidates": 0,
                    "detail": "This profile received zero candidates within the question budget.",
                    "evidence_class": "not_run",
                })
            elif scene.load_error is not None:
                record.update({
                    "attempt_status": "pipeline_error",
                    "detail": scene.load_error,
                    "evidence_class": "not_run",
                })
            else:
                stdout = io.StringIO()
                stderr = io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout):
                        with contextlib.redirect_stderr(stderr):
                            batch_manifest = runner(
                                scene_config=scene_snapshot,
                                profile_config=profile_snapshots[profile_id],
                                params=params_snapshot,
                                batch_root=pair_dir / "batch",
                                cells=requested_cells,
                                seed=(
                                    f"{seed}|{scene.scene_id}|{profile_id}"),
                                snapshot_content=snapshot_content)
                    batch_root = pair_dir / "batch"
                    generated_ids = _generated_main_point_ids(
                        batch_manifest, batch_root)
                    if generated_ids is not None:
                        # Enrich the scheduler's in-memory classification and
                        # matrix row without rewriting the producer-owned
                        # batch manifest.
                        batch_manifest = dict(batch_manifest)
                        batch_manifest["generated_main_point_ids"] = generated_ids
                        record["generated_main_point_ids"] = generated_ids
                    record.update(classify_manifest(
                        batch_manifest,
                        pixel_results.get((scene.scene_id, profile_id))))
                    record["batch_manifest"] = str(
                        (batch_root / "batch_manifest.json").resolve())
                except Exception as exc:
                    record.update({
                        "attempt_status": "pipeline_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "evidence_class": "not_run",
                    })
                if stdout.getvalue():
                    (pair_dir / "runner_stdout.txt").write_text(
                        stdout.getvalue(), encoding="utf-8")
                if stderr.getvalue():
                    (pair_dir / "runner_stderr.txt").write_text(
                        stderr.getvalue(), encoding="utf-8")
            if record["attempt_status"] not in PAIR_STATUSES:
                raise AssertionError(f"unknown pair status: {record}")
            _write_json(pair_dir / "attempt_manifest.json", record)
            rows.append(record)
            room_rows.append(record)

        room_manifest = {
            "schema": "qa_v3_room_profile_attempt_manifest_v1",
            "status": "research_dev",
            "qualification_claim": False,
            "scene_id": scene.scene_id,
            "scene_config_source": str(scene.source_path),
            "requested_profiles": requested_profiles,
            "scene_asset_id": scene_asset_id,
            "route_domain": route_domain,
            "backend": backend,
            "attempted_all_requested_profiles": (
                len(room_rows) == len(requested_profiles)),
            "counts_by_status": dict(Counter(
                row["attempt_status"] for row in room_rows)),
            "counts_by_quota_status": dict(Counter(
                row["quota_status"] for row in room_rows)),
            "attempts": room_rows,
        }
        _write_json(room_root / "room_attempt_manifest.json", room_manifest)

    status_counts = Counter(row["attempt_status"] for row in rows)
    matrix = {
        "schema": "qa_v3_scene_profile_matrix_v1",
        "status": (
            "completed_with_pipeline_errors"
            if status_counts["pipeline_error"] else "completed"),
        "qualification_claim": False,
        "boundary": (
            "Room-centric research/dev feasibility attempts. generated may "
            "still mean geometry_candidate when pixel evidence is not supplied; "
            "this matrix is not question admission or modality certification."),
        "selection_boundary": (
            "Every requested profile is attempted independently per scene. "
            "No model score, modality probe, or downstream question outcome is "
            "read or used to switch profiles."),
        "inputs": {
            "scene_configs": [str(spec.source_path) for spec in scene_specs],
            "profile_catalog": {
                profile_id: str(profile_snapshots[profile_id].resolve())
                for profile_id in profile_catalog},
            "params_source": str(params_source.resolve()),
            "params_snapshot": str(params_snapshot.resolve()),
            "cells_per_scene_profile": cells,
            "seed": seed,
        },
        "scene_count": len(scene_specs),
        "requested_profiles": requested_profiles,
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "scene_asset_id": str(
                    (scene.config or {}).get("scene_asset_id") or scene.scene_id),
                "route_domain": (scene.config or {}).get("route_domain"),
                "backend": (scene.config or {}).get("backend"),
            } for scene in scene_specs],
        "expected_matrix_cells": (
            len(scene_specs) * len(requested_profiles)),
        "observed_matrix_cells": len(rows),
        "attempted_every_requested_profile_per_scene": (
            len(rows) == len(scene_specs) * len(requested_profiles)),
        "counts_by_status": dict(status_counts),
        "counts_by_quota_status": dict(Counter(
            row["quota_status"] for row in rows)),
        "per_scene": {
            scene.scene_id: dict(Counter(
                row["attempt_status"] for row in rows
                if row["scene_id"] == scene.scene_id))
            for scene in scene_specs},
        "per_profile": {
            profile_id: dict(Counter(
                row["attempt_status"] for row in rows
                if row["profile_id"] == profile_id))
            for profile_id in requested_profiles},
        "matrix": rows,
    }
    if request_plan is not None:
        matrix["question_request"] = request_plan
        matrix["designed_questions_per_scene"] = {
            scene.scene_id: sum(row.get("geometry_candidates", 0)
                               for row in rows if row["scene_id"] == scene.scene_id)
            * request_plan["forms_per_candidate"]
            for scene in scene_specs}
        matrix["question_count_boundary"] = (
            "Counts refer to designed question forms; rendered media and admission remain separate.")
    _write_json(out_root / "scene_profile_matrix.json", matrix)
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", action="append", required=True,
                        type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--requested-profile", action="append")
    parser.add_argument("--params", required=True, type=Path)
    quota = parser.add_mutually_exclusive_group()
    quota.add_argument("--cells-per-pair", type=int,
                       help="explicit legacy candidate count for every scene/profile pair")
    quota.add_argument("--question-budget", type=int,
                       help="final question-form budget per room; defaults to params ITEMS_PER_ROOM_DEFAULT")
    parser.add_argument("--answer-form", action="append", help="mcq or open")
    parser.add_argument("--profile-weights", type=Path,
                        help="JSON mapping from requested profile ids to non-negative weights")
    parser.add_argument("--plan-only", action="store_true",
                        help="write the resolved request without running geometric search")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--pixel-results", type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--snapshot-content", default=(
        "/data/avengine_external/ue-assets/"
        "actor_content_registry_v9_20260823T033709Z/cpp/unreal_projects/"
        "SpearSim/Content"))
    args = parser.parse_args(argv)
    # Repository tmp may be a declared symlink to external output storage.
    args.out_root = args.out_root.resolve()

    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    if args.cells_per_pair is not None and args.cells_per_pair <= 0:
        parser.error("--cells-per-pair must be positive")

    profile_catalog = _load_profile_catalog(args.profiles)
    requested = _requested_profiles(
        args.requested_profile, profile_catalog)
    params_value = read_qa_params(args.params)
    if not isinstance(params_value, dict):
        parser.error("--params must contain a JSON object")
    scene_specs = _load_scene_specs(args.scene_config)
    pixel_results = _load_pixel_results(args.pixel_results)
    weights = (_read_json(args.profile_weights) if args.profile_weights is not None
               else params_value.get("PROFILE_WEIGHTS"))
    if isinstance(weights, dict):
        unknown = set(weights) - set(profile_catalog)
        if unknown:
            parser.error(f"profile weights name unknown profiles: {sorted(unknown)}")
        weights = {pid: weights[pid] for pid in requested if pid in weights}
    forms = args.answer_form or params_value.get("ANSWER_FORMS_DEFAULT")
    request_plan = None
    if args.cells_per_pair is None:
        request_plan = plan_room_questions(
            requested, params_value, question_budget=args.question_budget,
            answer_forms=forms, profile_weights=weights)
        cells = request_plan["cells"]
    else:
        if weights is not None:
            parser.error("profile weights and --cells-per-pair describe different allocation modes")
        cells = args.cells_per_pair
        if forms is not None:
            forms = normalize_answer_forms(forms)
            request_plan = plan_room_questions(
                requested, params_value,
                question_budget=cells * len(requested) * len(forms), answer_forms=forms)
    if request_plan is not None:
        params_value = dict(params_value, ANSWER_FORMS_DEFAULT=request_plan["answer_forms"])
    if args.plan_only:
        if request_plan is None:
            parser.error("--plan-only requires explicit or configured answer forms")
        args.out_root.mkdir(parents=True)
        plan = {"status": "planned", "per_room": request_plan,
                "scene_ids": [scene.scene_id for scene in scene_specs],
                "planned_question_count": len(scene_specs) * request_plan["planned_question_count"]}
        _write_json(args.out_root / "question_request.json", plan)
        print(json.dumps(plan, ensure_ascii=False))
        return 0

    args.out_root.mkdir(parents=True)
    matrix = run_scheduler(
        scene_specs=scene_specs,
        profile_catalog=profile_catalog,
        requested_profiles=requested,
        params_value=params_value,
        params_source=args.params,
        out_root=args.out_root,
        cells=cells,
        seed=args.seed,
        snapshot_content=args.snapshot_content,
        pixel_results=pixel_results, request_plan=request_plan)
    print(json.dumps({
        "out": str(args.out_root),
        "status": matrix["status"],
        "scene_count": matrix["scene_count"],
        "matrix_cells": matrix["observed_matrix_cells"],
        "counts_by_status": matrix["counts_by_status"],
    }, ensure_ascii=False))
    return 1 if matrix["counts_by_status"].get("pipeline_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
