#!/usr/bin/env python3
"""Join native main/GateB pixel evidence for an F2 identity candidate.

The binder is deliberately evidence-only. It consumes the point's declared
route windows and an intervention descriptor, checks the two native truth
documents, and writes a fresh joined report plus a copied fact. It does not
modify the source fact or make dataset/admission decisions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
import sys
from pathlib import Path
from typing import Any


SCHEMA = "avengine_qa_v3_f2_offscreen_identity_pixel_join_v1"
PIXEL_EVIDENCE_SCHEMA = "qa_v3_current_timeline_native_pixel_probe_v1"
VISIBLE_STATES = {"visible_clear", "visible_occluded"}


class F2PixelJoinError(ValueError):
    """Evidence or fact input is structurally invalid."""


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise F2PixelJoinError(f"cannot read JSON {path}: {exc}") from exc


def _write_fresh(path: Path, value: Any) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise F2PixelJoinError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _require_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise F2PixelJoinError(f"{owner} must be an object")
    return value


def _resolve_ref(raw: Any, *, base: Path, owner: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise F2PixelJoinError(f"{owner} must be a non-empty path")
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise F2PixelJoinError(f"{owner} is missing: {path}")
    return path


def _truth_document(value: Any, *, path: Path) -> Mapping[str, Any]:
    value = _require_mapping(value, owner=f"pixel truth {path}")
    for key in ("pixel_visibility_truth", "pixel_truth", "truth"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested
    return value


def _timeline_content(value: Any, *, owner: str) -> Any:
    value = _require_mapping(value, owner=owner)
    normalized = copy.deepcopy(dict(value))
    # v8/v9 may point at different copies of the same selection file.  This
    # is the only timeline field whose path is intentionally non-semantic.
    normalized.pop("actor_selection", None)
    return normalized


def _evidence_source_match(
    evidence_path: Path,
    wrapper: Mapping[str, Any],
    *,
    expected_selection: Path,
    expected_timeline: Path,
    label: str,
) -> dict[str, Any]:
    inputs = _require_mapping(
        wrapper.get("inputs"), owner=f"{label} evidence.inputs"
    )
    matches: dict[str, Any] = {}
    for field, expected, timeline in (
        ("actor_selection", expected_selection, False),
        ("timeline", expected_timeline, True),
    ):
        actual = _resolve_ref(
            inputs.get(field),
            base=evidence_path.parent,
            owner=f"{label} evidence.inputs.{field}",
        )
        actual_value = _read(actual)
        expected_value = _read(expected)
        if timeline:
            content_equal = (
                _timeline_content(
                    actual_value, owner=f"{label} evidence timeline"
                )
                == _timeline_content(
                    expected_value, owner=f"{label} declared timeline"
                )
            )
        else:
            content_equal = actual_value == expected_value
        matches[field] = {
            "evidence_path": str(actual),
            "declared_path": str(expected),
            "path_equal": actual == expected,
            "content_equal": content_equal,
            "matched": content_equal,
        }
    artifacts = wrapper.get("artifacts")
    truth_path = None
    truth_equal = True
    if isinstance(artifacts, Mapping) and artifacts.get("truth") is not None:
        truth_path = _resolve_ref(
            artifacts.get("truth"),
            base=evidence_path.parent,
            owner=f"{label} evidence.artifacts.truth",
        )
        truth = _truth_document(_read(truth_path), path=truth_path)
        inline = wrapper.get("pixel_visibility")
        if inline is not None:
            inline_truth = _truth_document(
                inline, path=evidence_path
            )
            truth_equal = dict(inline_truth) == dict(truth)
    matches["truth_payload"] = {
        "evidence_path": str(truth_path) if truth_path is not None else None,
        "inline_matches_artifact": truth_equal,
        "matched": truth_equal,
    }
    matches["matched"] = all(
        value.get("matched") is True
        for value in matches.values()
    )
    return matches


def _load_evidence(
    path: Path,
    *,
    expected_selection: Path,
    expected_timeline: Path,
    label: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    wrapper = _require_mapping(_read(path), owner=f"{label} pixel evidence")
    if (
        wrapper.get("schema") != PIXEL_EVIDENCE_SCHEMA
        or wrapper.get("status") != "pass"
    ):
        raise F2PixelJoinError(
            f"{label} evidence must have schema {PIXEL_EVIDENCE_SCHEMA!r} "
            "and status='pass'"
        )
    source_match = _evidence_source_match(
        path,
        wrapper,
        expected_selection=expected_selection,
        expected_timeline=expected_timeline,
        label=label,
    )
    artifacts = _require_mapping(
        wrapper.get("artifacts"), owner=f"{label} evidence.artifacts"
    )
    truth_path = _resolve_ref(
        artifacts.get("truth"),
        base=path.parent,
        owner=f"{label} evidence.artifacts.truth",
    )
    truth = _truth_document(_read(truth_path), path=truth_path)
    if not isinstance(wrapper.get("pixel_visibility"), Mapping):
        raise F2PixelJoinError(
            f"{label} evidence must carry a pixel_visibility truth wrapper"
        )
    return truth, {
        "path": str(path),
        "truth_path": str(truth_path),
        "source_match": source_match,
    }


def _unique_ints(value: Any, *, owner: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise F2PixelJoinError(f"{owner} must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise F2PixelJoinError(
                f"{owner}[{index}] must be a non-negative integer"
            )
        result.append(int(item))
    if len(result) != len(set(result)):
        raise F2PixelJoinError(f"{owner} contains duplicate frame indices")
    return result


def _frame_indices(truth: Mapping[str, Any], *, owner: str) -> list[int]:
    return _unique_ints(truth.get("frame_indices"), owner=f"{owner}.frame_indices")


def _camera_signature(truth: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for key in ("camera_contract_id", "resolution_hw", "camera_pose_ids"):
        if key in truth:
            signature[key] = copy.deepcopy(truth[key])
    for key in (
        "camera_poses",
        "camera_pose",
        "camera",
        "camera_pose_by_frame",
    ):
        if key in truth:
            signature[key] = copy.deepcopy(truth[key])
            break
    if not signature:
        raise F2PixelJoinError(
            f"{owner} lacks camera identity metadata "
            "(camera_contract_id/camera_pose_ids/camera pose)"
        )
    return signature


def _identity_signature(
    truth: Mapping[str, Any], *, owner: str
) -> dict[str, dict[str, Any]]:
    instances = truth.get("per_instance")
    if not isinstance(instances, Mapping) or not instances:
        raise F2PixelJoinError(f"{owner}.per_instance must be a non-empty object")
    result = {}
    identity_fields = (
        "source_slot_id",
        "target_id",
        "instance_id",
        "entity_instance_id",
        "actor_id",
        "object_id",
        "target_instance_id",
    )
    for slot, record in instances.items():
        record = _require_mapping(record, owner=f"{owner}.per_instance.{slot}")
        token = {
            key: copy.deepcopy(record[key])
            for key in identity_fields
            if key in record
        }
        if not token:
            token = {"per_instance_key": str(slot)}
        result[str(slot)] = token
    return result


def _frames_by_slot(
    truth: Mapping[str, Any], *, owner: str
) -> dict[str, dict[int, Mapping[str, Any]]]:
    instances = truth.get("per_instance")
    if not isinstance(instances, Mapping) or not instances:
        raise F2PixelJoinError(f"{owner}.per_instance must be a non-empty object")
    result = {}
    for slot, record in instances.items():
        record = _require_mapping(record, owner=f"{owner}.per_instance.{slot}")
        frames = record.get("frames")
        if not isinstance(frames, list):
            raise F2PixelJoinError(
                f"{owner}.per_instance.{slot}.frames must be a list"
            )
        by_index: dict[int, Mapping[str, Any]] = {}
        for index, frame in enumerate(frames):
            frame = _require_mapping(
                frame, owner=f"{owner}.per_instance.{slot}.frames[{index}]"
            )
            raw_index = frame.get("frame_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise F2PixelJoinError(
                    f"{owner}.per_instance.{slot}.frames[{index}] has invalid frame_index"
                )
            if raw_index in by_index:
                raise F2PixelJoinError(
                    f"{owner}.per_instance.{slot} repeats frame {raw_index}"
                )
            by_index[int(raw_index)] = frame
        result[str(slot)] = by_index
    return result


def _route_windows(
    fact: Mapping[str, Any],
) -> dict[str, dict[str, list[int]]]:
    geometry = _require_mapping(fact.get("geometry"), owner="fact.geometry")
    reports = geometry.get("route_reports")
    if not isinstance(reports, list) or not reports:
        raise F2PixelJoinError("fact.geometry.route_reports must be a non-empty list")
    result: dict[str, dict[str, list[int]]] = {}
    for index, report in enumerate(reports):
        report = _require_mapping(
            report, owner=f"fact.geometry.route_reports[{index}]"
        )
        slot = report.get("source_slot_id")
        if not isinstance(slot, str) or not slot:
            raise F2PixelJoinError(
                "each route_report must declare source_slot_id; "
                f"route_reports[{index}] has no usable source_slot_id"
            )
        if slot in result:
            raise F2PixelJoinError(
                f"fact.geometry.route_reports repeats source_slot_id {slot!r}"
            )
        windows = {}
        for phase in ("early", "late"):
            phase_record = _require_mapping(
                report.get(phase),
                owner=f"fact.geometry.route_reports[{index}].{phase}",
            )
            windows[phase] = _unique_ints(
                phase_record.get("frames"),
                owner=(
                    f"fact.geometry.route_reports[{index}].{phase}.frames"
                ),
            )
        result[slot] = windows
    return result


def _selection_assets(path: Path, *, owner: str) -> dict[str, str]:
    value = _require_mapping(_read(path), owner=owner)
    actors = value.get("actors")
    if not isinstance(actors, list) or not actors:
        raise F2PixelJoinError(f"{owner} has no actors")
    result = {}
    for index, actor in enumerate(actors):
        actor = _require_mapping(actor, owner=f"{owner}.actors[{index}]")
        slot = actor.get("source_slot_id")
        if not isinstance(slot, str) or not slot:
            raise F2PixelJoinError(
                f"{owner}.actors[{index}] has no source_slot_id"
            )
        identity = actor.get("asset_id") or actor.get("actor_id")
        if not isinstance(identity, str) or not identity:
            raise F2PixelJoinError(
                f"{owner}.actors[{index}] has no asset_id or actor_id"
            )
        if slot in result:
            raise F2PixelJoinError(f"{owner} repeats source_slot_id {slot!r}")
        result[slot] = identity
    return result


def _appearance_binding(
    fact: Mapping[str, Any], intervention_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    artifacts = _require_mapping(fact.get("artifacts"), owner="fact.artifacts")
    main_raw = artifacts.get("selection") or artifacts.get("actor_selection")
    main_selection = _resolve_ref(
        main_raw, base=Path(fact["_source_dir"]), owner="fact main selection"
    )
    main_timeline = _resolve_ref(
        artifacts.get("timeline"),
        base=Path(fact["_source_dir"]),
        owner="fact main timeline",
    )
    descriptor = _require_mapping(
        _read(intervention_path),
        owner=f"visual intervention {intervention_path}",
    )
    gateb_selection = _resolve_ref(
        descriptor.get("actor_selection"),
        base=intervention_path.parent,
        owner="visual intervention actor_selection",
    )
    gateb_timeline = _resolve_ref(
        descriptor.get("timeline"),
        base=intervention_path.parent,
        owner="visual intervention timeline",
    )
    gateb_endpoints = _resolve_ref(
        descriptor.get("source_endpoints"),
        base=intervention_path.parent,
        owner="visual intervention source_endpoints",
    )
    main_assets = _selection_assets(main_selection, owner="main actor selection")
    gateb_assets = _selection_assets(
        gateb_selection, owner="GateB actor selection"
    )
    if set(main_assets) != set(gateb_assets):
        raise F2PixelJoinError(
            "main/GateB actor selections have different source_slot_id sets"
        )
    if sorted(main_assets.values()) != sorted(gateb_assets.values()):
        raise F2PixelJoinError(
            "GateB actor selection does not preserve the main appearance set"
        )
    changed = [
        slot for slot in main_assets
        if main_assets[slot] != gateb_assets[slot]
    ]
    if not changed:
        raise F2PixelJoinError(
            "GateB actor selection does not exchange any appearance binding"
        )
    if len(changed) != len(main_assets):
        raise F2PixelJoinError(
            "GateB actor selection changes only part of the appearance binding"
        )
    main_appearance = fact.get("appearance_by_slot")
    expected_gateb_appearance = None
    if isinstance(main_appearance, Mapping):
        if set(map(str, main_appearance)) != set(main_assets):
            raise F2PixelJoinError(
                "fact appearance_by_slot does not cover actor selection slots"
            )
        asset_to_appearance = {
            main_assets[slot]: main_appearance[slot] for slot in main_assets
        }
        if len(asset_to_appearance) != len(main_assets):
            raise F2PixelJoinError(
                "main actor selection assets are not unique for appearance binding"
            )
        expected_gateb_appearance = {
            slot: asset_to_appearance[gateb_assets[slot]]
            for slot in gateb_assets
        }
    return {
        "main_assets_by_slot": main_assets,
        "gateb_assets_by_slot": gateb_assets,
        "changed_slots": changed,
        "main_appearance_by_slot": copy.deepcopy(main_appearance),
        "expected_gateb_appearance_by_slot": expected_gateb_appearance,
        "intervention": copy.deepcopy(dict(descriptor)),
    }, {
        "main_selection": main_selection,
        "main_timeline": main_timeline,
        "gateb_selection": gateb_selection,
        "gateb_timeline": gateb_timeline,
        "gateb_source_endpoints": gateb_endpoints,
    }


def _numeric_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return False
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-9)
    return a == b


def _geometry_signature(frame: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    fields = (
        "state",
        "target_pixels",
        "visible_pixels",
        "visible_fraction",
        "occlusion_fraction",
        "target_bbox_xyxy_px",
        "target_centroid_xy_px",
    )
    missing = [key for key in fields if key not in frame]
    if missing:
        raise F2PixelJoinError(
            f"{owner} lacks native geometry fields {missing}"
        )
    return {key: copy.deepcopy(frame[key]) for key in fields}


def _same_geometry(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            if not all(_numeric_equal(x, y) for x, y in zip(a, b)):
                return False
        elif not _numeric_equal(a, b):
            return False
    return True


def _footprint(frame: Mapping[str, Any], *, owner: str) -> int:
    for key in ("target_pixels", "native_footprint_pixels", "footprint_pixels"):
        if key not in frame:
            continue
        value = frame[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise F2PixelJoinError(f"{owner}.{key} must be a non-negative integer")
        return int(value)
    raise F2PixelJoinError(f"{owner} lacks native footprint pixel count")


def _visible_pixels(frame: Mapping[str, Any], *, owner: str) -> int:
    value = frame.get("visible_pixels")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F2PixelJoinError(
            f"{owner}.visible_pixels must be a non-negative integer"
        )
    return int(value)


def _evaluate_windows(
    truth: Mapping[str, Any],
    routes: Mapping[str, Mapping[str, list[int]]],
    *,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    by_slot = _frames_by_slot(truth, owner=label)
    evaluations: dict[str, Any] = {}
    reasons: list[str] = []
    for slot, phases in routes.items():
        if slot not in by_slot:
            reasons.append(f"{label}.{slot}.missing_target_identity")
            continue
        evaluations[slot] = {}
        for phase, indices in phases.items():
            rows = []
            for frame_index in indices:
                frame = by_slot[slot].get(frame_index)
                owner = f"{label}.{slot}.{phase}.frame_{frame_index}"
                if frame is None:
                    reasons.append(f"{owner}.missing")
                    continue
                state = frame.get("state")
                footprint = _footprint(frame, owner=owner)
                visible = _visible_pixels(frame, owner=owner)
                if phase == "early":
                    ok = state == "out_of_view" and footprint == 0 and visible == 0
                    reason = (
                        "out_of_view"
                        if ok
                        else (
                            "not_out_of_view"
                            if state != "out_of_view"
                            else "native_footprint_nonzero"
                        )
                    )
                else:
                    ok = (
                        state in VISIBLE_STATES
                        and footprint > 0
                        and visible > 0
                    )
                    reason = "in_view" if ok else "not_in_view"
                rows.append({
                    "frame_index": frame_index,
                    "state": state,
                    "native_footprint_pixels": footprint,
                    "visible_pixels": visible,
                    "reason": reason,
                    "passed": ok,
                })
                if not ok:
                    reasons.append(f"{owner}.{reason}")
            evaluations[slot][phase] = rows
    return evaluations, reasons


def _compare_truths(
    main: Mapping[str, Any],
    gateb: Mapping[str, Any],
    routes: Mapping[str, Mapping[str, list[int]]],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    checks: dict[str, Any] = {}
    main_frames = _frame_indices(main, owner="main pixel truth")
    gateb_frames = _frame_indices(gateb, owner="GateB pixel truth")
    checks["frame_indices_same"] = main_frames == gateb_frames
    if not checks["frame_indices_same"]:
        reasons.append("frame_indices_differ")
    main_camera = _camera_signature(main, owner="main pixel truth")
    gateb_camera = _camera_signature(gateb, owner="GateB pixel truth")
    checks["camera_identity_same"] = main_camera == gateb_camera
    if not checks["camera_identity_same"]:
        reasons.append("camera_identity_differ")
    main_identity = _identity_signature(main, owner="main pixel truth")
    gateb_identity = _identity_signature(gateb, owner="GateB pixel truth")
    checks["target_identity_range_same"] = main_identity == gateb_identity
    if not checks["target_identity_range_same"]:
        reasons.append("target_identity_range_differ")
    route_slots = set(routes)
    checks["route_slots_in_truth"] = (
        route_slots <= set(main_identity) and route_slots <= set(gateb_identity)
    )
    if not checks["route_slots_in_truth"]:
        reasons.append("route_slot_missing_from_truth")
    main_eval, main_reasons = _evaluate_windows(main, routes, label="main")
    gateb_eval, gateb_reasons = _evaluate_windows(gateb, routes, label="gateB")
    reasons.extend(main_reasons)
    reasons.extend(gateb_reasons)
    checks["main_windows"] = main_eval
    checks["gateb_windows"] = gateb_eval
    checks["window_checks_pass"] = not main_reasons and not gateb_reasons
    geometry_checks = []
    main_by_slot = _frames_by_slot(main, owner="main pixel truth")
    gateb_by_slot = _frames_by_slot(gateb, owner="GateB pixel truth")
    if route_slots <= set(main_by_slot) and route_slots <= set(gateb_by_slot):
        for slot, phases in routes.items():
            for phase, indices in phases.items():
                for frame_index in indices:
                    left = main_by_slot[slot].get(frame_index)
                    right = gateb_by_slot[slot].get(frame_index)
                    if left is None or right is None:
                        continue
                    same = _same_geometry(
                        _geometry_signature(
                            left,
                            owner=f"main.{slot}.{phase}.frame_{frame_index}",
                        ),
                        _geometry_signature(
                            right,
                            owner=f"gateB.{slot}.{phase}.frame_{frame_index}",
                        ),
                    )
                    geometry_checks.append({
                        "slot": slot,
                        "phase": phase,
                        "frame_index": frame_index,
                        "same": same,
                    })
                    if not same:
                        reasons.append(
                            f"gateB_geometry_differs.{slot}.{phase}.frame_{frame_index}"
                        )
    checks["gateb_geometry_checks"] = geometry_checks
    checks["gateb_geometry_same"] = all(
        row["same"] for row in geometry_checks
    ) if geometry_checks else False
    if not checks["gateb_geometry_same"]:
        reasons.append("gateB_geometry_not_same_as_main")
    return checks, reasons


def join(
    fact: Mapping[str, Any],
    main_pixel_path: str | Path,
    intervention_path: str | Path,
    gateb_pixel_path: str | Path,
) -> dict[str, Any]:
    fact = _require_mapping(fact, owner="fact")
    if not fact.get("point_id") or not fact.get("scene_id"):
        raise F2PixelJoinError("fact needs non-empty point_id and scene_id")
    main_pixel_path = Path(main_pixel_path).expanduser().resolve()
    intervention_path = Path(intervention_path).expanduser().resolve()
    gateb_pixel_path = Path(gateb_pixel_path).expanduser().resolve()
    fact_copy = copy.deepcopy(dict(fact))
    fact_copy["_source_dir"] = str(
        Path(fact.get("_source_dir") or ".").resolve()
    )
    if not fact_copy["_source_dir"] or fact_copy["_source_dir"] == str(Path(".").resolve()):
        fact_copy["_source_dir"] = str(main_pixel_path.parent.resolve())
    routes = _route_windows(fact_copy)
    appearance, appearance_paths = _appearance_binding(
        fact_copy, intervention_path
    )
    main_truth, main_evidence = _load_evidence(
        main_pixel_path,
        expected_selection=appearance_paths["main_selection"],
        expected_timeline=appearance_paths["main_timeline"],
        label="main",
    )
    gateb_truth, gateb_evidence = _load_evidence(
        gateb_pixel_path,
        expected_selection=appearance_paths["gateb_selection"],
        expected_timeline=appearance_paths["gateb_timeline"],
        label="GateB",
    )
    checks, reasons = _compare_truths(main_truth, gateb_truth, routes)
    checks["source_match"] = {
        "main": main_evidence["source_match"],
        "gateB": gateb_evidence["source_match"],
    }
    for label, source in (("main", main_evidence), ("GateB", gateb_evidence)):
        if not source["source_match"]["matched"]:
            reasons.append(f"{label}_evidence_source_mismatch")
    passed = not reasons
    report = {
        "schema": SCHEMA,
        "status": "pass" if passed else "pixel_rejected",
        "evidence_class": (
            "pixel_qualified_candidate" if passed else "pixel_rejected"
        ),
        "qualification_claim": False,
        "point_id": str(fact["point_id"]),
        "scene_id": str(fact["scene_id"]),
        "profile_id": fact.get("profile_id"),
        "rejection_reasons": reasons,
        "checks": checks,
        "appearance_binding": appearance,
        "inputs": {
            "main_fact": fact.get("_source_path"),
            "main_pixel_evidence": str(main_pixel_path),
            "main_pixel_truth": main_evidence["truth_path"],
            "visual_intervention": str(intervention_path),
            "gateb_pixel_evidence": str(gateb_pixel_path),
            "gateb_pixel_truth": gateb_evidence["truth_path"],
            "resolved_artifacts": {
                key: str(value) for key, value in appearance_paths.items()
            },
        },
        "boundary": (
            "Native main/GateB pixel identity join only. The result remains a "
            "research candidate and does not establish dataset admission, "
            "human recognisability, or general answerability."
        ),
    }
    joined_fact = copy.deepcopy(dict(fact))
    joined_fact.pop("_source_path", None)
    joined_fact.pop("_source_dir", None)
    joined_fact["qualification_claim"] = False
    joined_fact["truth_status"] = (
        "native_pixel_joined_research_candidate"
        if passed else "native_pixel_rejected_research_candidate"
    )
    joined_fact["pixel_visibility"] = {
        "schema": SCHEMA,
        "status": report["status"],
        "evidence_paths": report["inputs"],
        "appearance_binding": appearance,
        "route_windows": routes,
    }
    return {
        "schema": SCHEMA,
        "status": "research_candidate",
        "pixel_join_status": report["status"],
        "evidence_class": report["evidence_class"],
        "qualification_claim": False,
        "joined_fact": joined_fact,
        "report": report,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact", required=True, type=Path)
    parser.add_argument(
        "--main-pixel-truth",
        "--main-pixel-evidence",
        dest="main_pixel_truth",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--visual-intervention", "--intervention",
        dest="intervention",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--gateb-pixel-truth",
        "--gateb-pixel-evidence",
        dest="gateb_pixel_truth",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite output: {args.output}", file=sys.stderr)
        return 2
    try:
        fact_path = args.fact.expanduser().resolve()
        fact = _require_mapping(_read(fact_path), owner=f"fact {fact_path}")
        fact = dict(fact)
        fact["_source_path"] = str(fact_path)
        fact["_source_dir"] = str(fact_path.parent)
        result = join(
            fact,
            args.main_pixel_truth,
            args.intervention,
            args.gateb_pixel_truth,
        )
        _write_fresh(args.output, result)
    except F2PixelJoinError as exc:
        print(f"F2 pixel join refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "status": result["pixel_join_status"],
        "point_id": result["report"]["point_id"],
        "rejection_reasons": result["report"]["rejection_reasons"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
