#!/usr/bin/env python3
"""Validate and publish the CPU-only strict two-human row7 v2 overlay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = (
    REPOSITORY / "tools/qa/build_strict_two_human_expansion_preflight.py"
)
PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "strict_two_human_expansion_preflight", PREFLIGHT_PATH
)
if PREFLIGHT_SPEC is None or PREFLIGHT_SPEC.loader is None:
    raise RuntimeError(f"cannot import {PREFLIGHT_PATH}")
PREFLIGHT = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT)

OVERLAY_SCHEMA = "avengine_native_strict_two_human_row_revision_overlay_v1"
OUTPUT_SCHEMA = "avengine_native_strict_two_human_row_revision_preflight_v1"
EXPECTED_SCOPE = [
    "episode_id",
    "camera_pose",
    "camera_floor_point_provenance",
    "actors.source1.root_translation_m",
    "actors.source1.actor_yaw_ue_deg",
    "actors.source1.floor_point_provenance",
    "actors.source2.root_translation_m",
    "actors.source2.actor_yaw_ue_deg",
    "actors.source2.floor_point_provenance",
]
TOP_LEVEL_KEYS = {
    "schema",
    "revision_id",
    "status",
    "base_plan",
    "base_plan_sha256",
    "v1_rejection",
    "v1_rejection_sha256",
    "target_row_id",
    "replacement_scope",
    "immutable_contract",
    "replacement",
    "rationale",
    "execution_boundary",
}
ACTOR_GEOMETRY_KEYS = {
    "source_slot_id",
    "root_translation_m",
    "actor_yaw_ue_deg",
    "floor_point_provenance",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPOSITORY / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ref_key(value: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(value["scenario_id"]),
        int(value["frame_index"]),
        str(value["actor_id"]),
    )


def _actor_contract(actor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: actor[key]
        for key in (
            "role",
            "source_slot_id",
            "identity_key",
            "expected_screen_side",
            "voice_policy",
        )
    }


def apply_overlay(
    base_plan: Mapping[str, Any], overlay: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    """Materialize only the explicitly allowed row geometry and Episode ID."""

    revised = deepcopy(base_plan)
    matches = [
        index
        for index, row in enumerate(revised["rows"])
        if row.get("row_id") == overlay.get("target_row_id")
    ]
    if len(matches) != 1:
        raise ValueError("target row must resolve exactly once")
    row_index = matches[0]
    row = revised["rows"][row_index]
    replacement = overlay["replacement"]
    row["episode_id"] = replacement["episode_id"]
    row["camera_pose"] = deepcopy(replacement["camera_pose"])
    row["camera_floor_point_provenance"] = deepcopy(
        replacement["camera_floor_point_provenance"]
    )
    geometry_by_slot = {
        actor["source_slot_id"]: actor for actor in replacement["actors"]
    }
    for actor in row["actors"]:
        geometry = geometry_by_slot[actor["source_slot_id"]]
        for field in (
            "root_translation_m",
            "actor_yaw_ue_deg",
            "floor_point_provenance",
        ):
            actor[field] = deepcopy(geometry[field])
    return revised, row_index


def validate_overlay(
    overlay: Mapping[str, Any],
    base_plan: Mapping[str, Any],
    rejection: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[str]:
    """Return all overlay errors without publishing or using a GPU."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(set(overlay) == TOP_LEVEL_KEYS, "overlay top-level fields drift")
    require(overlay.get("schema") == OVERLAY_SCHEMA, "overlay schema mismatch")
    require(
        overlay.get("status") == "cpu_geometry_revision_proposal",
        "overlay status mismatch",
    )
    require(
        overlay.get("replacement_scope") == EXPECTED_SCOPE,
        "replacement scope drift",
    )
    require(
        overlay.get("target_row_id") == "strict_07_female_construction_right",
        "target row mismatch",
    )
    boundary = overlay.get("execution_boundary", {})
    require(boundary.get("formal_scene_count") == 0, "formal count must remain zero")
    require(
        boundary.get("qualification_claim") is False,
        "qualification claim must remain false",
    )
    require(
        boundary.get("gpu_or_rir_executed") is False,
        "GPU/RIR execution forbidden in overlay atom",
    )
    require(
        boundary.get("exact_rir_required_before_sparse") is True,
        "exact RIR must remain required",
    )
    require(
        boundary.get("single_sparse_gate_required") is True,
        "single sparse gate must remain required",
    )
    require(
        boundary.get("automatic_retry_allowed") is False,
        "automatic retry must remain forbidden",
    )
    require(
        boundary.get("base_plan_mutation_allowed") is False,
        "base plan mutation must remain forbidden",
    )

    row_matches = [
        (index, row)
        for index, row in enumerate(base_plan.get("rows", []))
        if row.get("row_id") == overlay.get("target_row_id")
    ]
    require(len(row_matches) == 1, "base target row must resolve exactly once")
    if len(row_matches) != 1:
        return errors
    row_index, base_row = row_matches[0]
    require(row_index == 6, "row7 index drift")
    require(
        base_row.get("episode_id")
        == "rocketbox_female_construction__strict_two_human_right_v1",
        "base row7 Episode drift",
    )
    require(
        base_plan.get("evidence", {}).get("rejected_row7_v1")
        == overlay.get("v1_rejection"),
        "base plan rejection pointer mismatch",
    )
    require(rejection.get("status") == "rejected", "v1 must remain rejected")
    require(rejection.get("decision") == "fail", "v1 decision must remain fail")
    require(
        rejection.get("row_id") == base_row.get("row_id")
        and rejection.get("episode_id") == base_row.get("episode_id"),
        "v1 rejection row/Episode mismatch",
    )
    require(
        rejection.get("target_gate", {}).get("observed_visible_fraction", 1.0)
        < 0.8,
        "v1 rejection target failure disappeared",
    )
    require(
        rejection.get("formal_scene_count") == 0
        and rejection.get("qualification_claim") is False,
        "v1 rejection claim boundary drift",
    )

    thresholds = base_plan.get("projection_and_native_thresholds", {})
    immutable = overlay.get("immutable_contract", {})
    require(
        thresholds.get("target_visible_fraction_minimum") == 0.8
        and immutable.get("target_visible_fraction_minimum") == 0.8,
        "target visibility threshold must remain 0.8",
    )
    require(
        thresholds.get("distractor_visible_fraction_minimum") == 0.5
        and immutable.get("distractor_visible_fraction_minimum") == 0.5,
        "distractor visibility threshold must remain 0.5",
    )
    require(
        immutable.get("identity_pair") == base_row.get("identity_pair") == "F/C",
        "identity pair drift",
    )
    require(
        immutable.get("sparse_gate_frame_index")
        == base_plan.get("timeline", {}).get("sparse_gate_frame_index")
        == 15,
        "sparse frame drift",
    )
    if len(base_row.get("actors", [])) == 2:
        require(
            immutable.get("target") == _actor_contract(base_row["actors"][0]),
            "target identity/side/voice drift",
        )
        require(
            immutable.get("distractor") == _actor_contract(base_row["actors"][1]),
            "distractor identity/side/voice drift",
        )
    else:
        errors.append("base row7 must contain exactly two actors")

    replacement = overlay.get("replacement", {})
    require(
        replacement.get("episode_id")
        == "rocketbox_female_construction__strict_two_human_right_v2",
        "replacement Episode mismatch",
    )
    require(
        replacement.get("episode_id") != base_row.get("episode_id"),
        "replacement Episode must differ from v1",
    )
    geometries = replacement.get("actors", [])
    require(
        isinstance(geometries, list) and len(geometries) == 2,
        "replacement must contain exactly two actor geometries",
    )
    if isinstance(geometries, list) and len(geometries) == 2:
        require(
            [actor.get("source_slot_id") for actor in geometries]
            == ["source1", "source2"],
            "replacement actor slot order mismatch",
        )
        for geometry in geometries:
            require(
                set(geometry) == ACTOR_GEOMETRY_KEYS,
                f"{geometry.get('source_slot_id')} replacement fields drift",
            )

    used_refs: set[tuple[str, int, str]] = set()
    for row in base_plan.get("rows", [])[1:]:
        used_refs.add(_ref_key(row["camera_floor_point_provenance"]))
        used_refs.update(
            _ref_key(actor["floor_point_provenance"])
            for actor in row.get("actors", [])
        )
    try:
        revised_refs = [
            _ref_key(replacement["camera_floor_point_provenance"]),
            *[
                _ref_key(actor["floor_point_provenance"])
                for actor in geometries
            ],
        ]
    except (KeyError, TypeError, ValueError):
        revised_refs = []
        errors.append("replacement provenance is malformed")
    require(len(set(revised_refs)) == 3, "three revised provenance tuples required")
    require(
        all(reference not in used_refs for reference in revised_refs),
        "revised provenance must be unused by the base plan",
    )

    if errors:
        return errors
    before = deepcopy(base_plan)
    try:
        revised, revised_index = apply_overlay(base_plan, overlay)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"overlay materialization failed: {exc}"]
    require(base_plan == before, "overlay mutated the base plan in memory")
    require(revised_index == row_index, "materialized row index drift")
    for index, row in enumerate(base_plan["rows"]):
        if index != row_index:
            require(revised["rows"][index] == row, f"base row {index + 1} changed")
    for old_actor, new_actor in zip(
        base_row["actors"], revised["rows"][row_index]["actors"], strict=True
    ):
        require(
            _actor_contract(old_actor) == _actor_contract(new_actor),
            f"{old_actor['source_slot_id']} immutable actor contract changed",
        )
    plan_errors = PREFLIGHT.validate_plan(revised, registry)
    errors.extend(f"materialized plan: {message}" for message in plan_errors)
    return errors


def build(overlay_path: Path, output: Path) -> Path:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    overlay = _load(overlay_path)
    base_path = _resolve(str(overlay["base_plan"]))
    rejection_path = _resolve(str(overlay["v1_rejection"]))
    if _sha256(base_path) != overlay.get("base_plan_sha256"):
        raise RuntimeError("base plan content drift")
    if _sha256(rejection_path) != overlay.get("v1_rejection_sha256"):
        raise RuntimeError("v1 rejection content drift")
    base_plan = _load(base_path)
    rejection = _load(rejection_path)
    registry = _load(_resolve(str(base_plan["evidence"]["runtime_registry"])))
    errors = validate_overlay(overlay, base_plan, rejection, registry)
    if errors:
        raise RuntimeError("overlay validation failed:\n- " + "\n- ".join(errors))

    revised, row_index = apply_overlay(base_plan, overlay)
    output.mkdir(parents=True)
    effective_plan_path = output / "effective_plan.json"
    effective_plan_path.write_text(
        json.dumps(revised, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    base_preflight_path = PREFLIGHT.build(
        effective_plan_path, output / "expansion_preflight"
    )

    suite_path = _resolve(str(revised["source_floor_point_suite"]))
    suite = _load(suite_path)
    suite_by_id = {scenario["scenario_id"]: scenario for scenario in suite["scenarios"]}
    row = revised["rows"][row_index]
    references = [
        ("camera_floor", row["camera_floor_point_provenance"]),
        *[
            (actor["role"], actor["floor_point_provenance"])
            for actor in row["actors"]
        ],
    ]
    readbacks: list[dict[str, Any]] = []
    for role, reference in references:
        planned, record = PREFLIGHT._source_point(
            suite_by_id, suite_path.parent, reference
        )
        expected = (
            [row["camera_pose"]["translation_m"][0], 0.4,
             row["camera_pose"]["translation_m"][2]]
            if role == "camera_floor"
            else next(
                actor["root_translation_m"]
                for actor in row["actors"]
                if actor["role"] == role
            )
        )
        if max(abs(float(planned[i]) - float(expected[i])) for i in range(3)) > 1e-6:
            raise RuntimeError(f"{role} revised provenance position mismatch")
        record["planned_role"] = role
        readbacks.append(record)

    thresholds = revised["projection_and_native_thresholds"]
    catalog = revised["approved_identity_catalog"]
    projections: list[dict[str, Any]] = []
    for actor in row["actors"]:
        root = [float(value) for value in actor["root_translation_m"]]
        offset = catalog[actor["identity_key"]]["mouth_offset_from_root_m"]
        mouth = [root[i] + float(offset[i]) for i in range(3)]
        depth, lateral, x_fraction, y_fraction = PREFLIGHT._project(
            row["camera_pose"],
            mouth,
            thresholds["horizontal_fov_deg"],
            thresholds["resolution_hw"],
        )
        envelope = [
            PREFLIGHT._project(
                row["camera_pose"],
                [root[0], root[1] + height, root[2]],
                thresholds["horizontal_fov_deg"],
                thresholds["resolution_hw"],
            )[3]
            for height in thresholds[
                "conservative_actor_vertical_envelope_from_root_m"
            ]
        ]
        projections.append(
            {
                "role": actor["role"],
                "source_slot_id": actor["source_slot_id"],
                "identity_key": actor["identity_key"],
                "depth_m": depth,
                "camera_right_m": lateral,
                "mouth_xy_fraction": [x_fraction, y_fraction],
                "vertical_envelope_y_fraction": envelope,
            }
        )

    cameras = [candidate["camera_pose"]["translation_m"] for candidate in revised["rows"]]
    revised_camera = cameras[row_index]
    camera_distances = [
        {
            "row_id": revised["rows"][index]["row_id"],
            "distance_m": PREFLIGHT._distance_xz(revised_camera, camera),
        }
        for index, camera in enumerate(cameras)
        if index != row_index
    ]
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "pass_cpu_geometry_revision_pending_exact_rir_and_single_sparse_gate",
        "revision_id": overlay["revision_id"],
        "base_plan": str(base_path),
        "v1_rejection": str(rejection_path),
        "effective_plan": str(effective_plan_path),
        "expansion_preflight": str(base_preflight_path),
        "target_row_id": row["row_id"],
        "replacement_episode_id": row["episode_id"],
        "replacement_scope": overlay["replacement_scope"],
        "unchanged_row_ids": [
            candidate["row_id"]
            for index, candidate in enumerate(base_plan["rows"])
            if index != row_index
        ],
        "base_plan_rows_unchanged": True,
        "immutable_contract": overlay["immutable_contract"],
        "native_occupied_point_readbacks": readbacks,
        "projections": projections,
        "camera_cluster_distances": camera_distances,
        "minimum_camera_cluster_distance_m": min(
            item["distance_m"] for item in camera_distances
        ),
        "rationale": overlay["rationale"],
        "formal_scene_count": 0,
        "qualification_claim": False,
        "gpu_or_rir_executed": False,
        "next_required_steps": [
            "build_new_row7_v2_exact_two_endpoint_rir_and_binaural_audio",
            "run_exactly_one_row7_v2_f15_native_sparse_gate_on_idle_physical_gpu1",
            "require_target_visible_fraction_at_least_0.8_before_any_row8_execution",
        ],
    }
    result_path = output / "preflight.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=(
            REPOSITORY
            / "examples/qa/native_strict_two_human_row7_v2_revision_overlay.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.overlay.resolve(), args.output.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
