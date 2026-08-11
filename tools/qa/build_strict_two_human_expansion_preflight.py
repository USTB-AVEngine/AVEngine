#!/usr/bin/env python3
"""Validate and publish the CPU-only strict two-human eight-row plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "avengine_native_strict_two_human_expansion_plan_v1"
OUTPUT_SCHEMA = "avengine_native_strict_two_human_expansion_preflight_v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPOSITORY / path


def _distance_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _camera_basis(camera: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float], float]:
    q = [float(value) for value in camera["rotation_xyzw"]]
    if len(q) != 4:
        raise ValueError("camera quaternion must contain four values")
    norm = math.sqrt(sum(value * value for value in q))
    if not math.isclose(norm, 1.0, abs_tol=1e-9):
        raise ValueError("camera quaternion must be normalized")
    if abs(q[0]) > 1e-9 or abs(q[2]) > 1e-9:
        raise ValueError("strict expansion camera must be yaw-only")
    yaw_rad = 2.0 * math.atan2(q[1], q[3])
    yaw_deg = math.degrees(yaw_rad) % 360.0
    declared = float(camera["habitat_yaw_deg"]) % 360.0
    if _angle_delta_deg(yaw_deg, declared) > 1e-8:
        raise ValueError("camera quaternion/yaw mismatch")
    forward = (-math.sin(yaw_rad), -math.cos(yaw_rad))
    right = (-forward[1], forward[0])
    return forward, right, yaw_deg


def _profile_by_id(registry: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    matches = [asset for asset in registry.get("assets", []) if asset.get("asset_id") == asset_id]
    if len(matches) != 1:
        raise ValueError(f"runtime profile must resolve exactly once: {asset_id}")
    return matches[0]


def _project(
    camera: Mapping[str, Any],
    world_point: Sequence[float],
    horizontal_fov_deg: float,
    resolution_hw: Sequence[int],
) -> tuple[float, float, float, float]:
    forward, right, _ = _camera_basis(camera)
    origin = [float(value) for value in camera["translation_m"]]
    delta_x = float(world_point[0]) - origin[0]
    delta_z = float(world_point[2]) - origin[2]
    depth = delta_x * forward[0] + delta_z * forward[1]
    lateral = delta_x * right[0] + delta_z * right[1]
    if depth <= 0.0:
        raise ValueError("projected point is behind camera")
    tan_horizontal = math.tan(math.radians(horizontal_fov_deg) / 2.0)
    tan_vertical = tan_horizontal * float(resolution_hw[0]) / float(resolution_hw[1])
    x_fraction = 0.5 + lateral / (2.0 * depth * tan_horizontal)
    y_fraction = 0.5 - (
        (float(world_point[1]) - origin[1]) / (2.0 * depth * tan_vertical)
    )
    return depth, lateral, x_fraction, y_fraction


def validate_plan(plan: Mapping[str, Any], registry: Mapping[str, Any]) -> list[str]:
    """Return all deterministic plan errors without publishing or using a GPU."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(plan.get("schema") == PLAN_SCHEMA, "plan schema mismatch")
    require(plan.get("status") == "cpu_preflight_pending", "plan status must be pending")
    require(plan.get("formal_scene_count") == 0, "formal scene count must remain zero")
    require(plan.get("qualification_claim") is False, "qualification claim must remain false")
    require(plan.get("paper_catalog_mutation_allowed") is False, "paper catalog mutation forbidden")
    require(
        plan.get("target_only_actor_map")
        == {"source1": "source1_actor", "source2": "source2_actor"},
        "target-only actor map mismatch",
    )
    execution = plan.get("execution_policy", {})
    require(execution.get("gpu_or_rir_allowed_in_this_atom") is False, "GPU/RIR forbidden in CPU atom")
    require(execution.get("per_row_exact_rir_required_before_sparse") is True, "exact RIR gate missing")
    require(execution.get("native_combined_room_clearance_required") is True, "native clearance gate missing")
    gpu = plan.get("gpu_policy", {})
    require(gpu.get("physical_gpu_index") == 1, "physical GPU must be 1")
    require(gpu.get("graphics_adapter_argument") == 1, "graphics adapter must be 1")
    require(gpu.get("required_idle_compute_process_count") == 0, "GPU idle gate missing")
    require(gpu.get("forbidden_physical_gpu_indices") == [0, 3], "GPU 0/3 must be forbidden")

    timeline = plan.get("timeline", {})
    require(timeline.get("frame_count") == 75, "frame count must be 75")
    require(timeline.get("frame_rate_hz") == 15, "frame rate must be 15 Hz")
    require(timeline.get("sparse_gate_frame_index") == 15, "sparse gate must be f15")
    require(timeline.get("target_speech_start_sample") == 7467, "speech start sample drift")
    require(timeline.get("target_speech_start_frame") == 7, "speech start frame drift")
    require(
        timeline.get("target_speech_duration_policy") == "full_dry_asset",
        "full dry speech policy required",
    )

    catalog = plan.get("approved_identity_catalog", {})
    require(set(catalog) == {"M", "F", "C"}, "approved catalog must contain exactly M/F/C")
    feasibility = _load_json(_resolve(str(plan["evidence"]["feasibility_report"])))
    approved = set(feasibility.get("approved_adults", []))
    excluded = {
        item.get("identity_id") for item in feasibility.get("excluded_adults", [])
    }
    child_ids = set(feasibility.get("children", {}).get("identity_ids", []))
    for key, identity in catalog.items():
        identity_id = identity.get("original_identity_id")
        require(identity_id in approved, f"identity {key} is not approved")
        require(identity_id not in excluded, f"identity {key} is excluded")
        require(identity_id not in child_ids, f"identity {key} is a child")
        try:
            profile = _profile_by_id(registry, str(identity.get("runtime_asset_id")))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        alias = registry.get("aliases", {}).get(identity.get("runtime_asset_alias"), {})
        require(alias.get("asset_id") == identity.get("runtime_asset_id"), f"identity {key} alias mismatch")
        require(alias.get("revision") == identity.get("runtime_revision"), f"identity {key} alias revision mismatch")
        require(profile.get("revision") == identity.get("runtime_revision"), f"identity {key} profile revision mismatch")
        require(profile.get("entity_class") == "articulated_human", f"identity {key} is not human")
        require(profile.get("realized_attributes", {}).get("life_stage") == "adult", f"identity {key} is not adult")
        require(profile.get("admission_state") == "research", f"identity {key} must remain research")
        require(identity.get("listening_review") == "pending_research_only", f"identity {key} listening boundary mismatch")
        expected_window = {"M": [7, 31], "F": [7, 50], "C": [7, 50]}[key]
        require(
            identity.get("expected_speech_frame_window_inclusive")
            == expected_window,
            f"identity {key} speech window mismatch",
        )
        require(
            isinstance(identity.get("ue_import_manifest"), str),
            f"identity {key} UE import authority missing",
        )
        anchors = profile.get("emitter_anchors", [])
        mouth = [anchor for anchor in anchors if anchor.get("anchor_id") == "mouth"]
        require(len(mouth) == 1, f"identity {key} mouth anchor must resolve once")
        if len(mouth) == 1:
            require(mouth[0].get("offset_m") == identity.get("mouth_offset_from_root_m"), f"identity {key} mouth offset mismatch")

    rows = plan.get("rows")
    require(isinstance(rows, list) and len(rows) == 8, "exactly eight rows required")
    if not isinstance(rows, list) or len(rows) != 8:
        return errors
    require([row.get("identity_pair") for row in rows] == plan.get("identity_pair_sequence"), "identity pair sequence mismatch")
    require([row.get("target_expected_screen_side") for row in rows] == plan.get("target_side_sequence"), "target side sequence mismatch")
    require(sum(row.get("target_expected_screen_side") == "left" for row in rows) == 4, "four left targets required")
    require(sum(row.get("target_expected_screen_side") == "right" for row in rows) == 4, "four right targets required")
    require(rows[0].get("status") == "pass_existing_sparse_canary", "row1 must bind passed canary")
    require(
        all(row.get("status") == "cpu_pose_candidate_only_pending_combined_native_sparse" for row in rows[1:]),
        "rows2-8 must remain CPU-only candidates",
    )

    thresholds = plan.get("projection_and_native_thresholds", {})
    resolution = thresholds.get("resolution_hw", [])
    hfov = float(thresholds.get("horizontal_fov_deg", 0.0))
    safe = thresholds.get("mouth_safe_fraction_open_interval", [])
    dead_zone = float(thresholds.get("screen_side_dead_zone_fraction", 0.0))
    minimum_x_separation = float(thresholds.get("minimum_projected_x_separation_fraction", 0.0))
    minimum_actor_separation = float(thresholds.get("minimum_actor_horizontal_separation_m", 0.0))
    envelope = thresholds.get("conservative_actor_vertical_envelope_from_root_m", [])
    require(
        float(thresholds.get("target_visible_fraction_minimum", -1.0)) == 0.8,
        "target visible-fraction minimum must remain 0.8",
    )
    require(
        float(thresholds.get("distractor_visible_fraction_minimum", -1.0)) == 0.5,
        "distractor visible-fraction minimum must remain 0.5",
    )
    camera_signatures: set[tuple[float, ...]] = set()
    actor_signatures: set[tuple[float, ...]] = set()
    camera_translations: list[Sequence[float]] = []
    for row_index, row in enumerate(rows):
        actors = row.get("actors", [])
        require(len(actors) == 2, f"row {row_index + 1} must contain two actors")
        if len(actors) != 2:
            continue
        target, distractor = actors
        require(target.get("role") == "target", f"row {row_index + 1} target order mismatch")
        require(distractor.get("role") == "distractor", f"row {row_index + 1} distractor order mismatch")
        require(target.get("source_slot_id") == "source1", f"row {row_index + 1} target slot mismatch")
        require(distractor.get("source_slot_id") == "source2", f"row {row_index + 1} distractor slot mismatch")
        require(target.get("identity_key") != distractor.get("identity_key"), f"row {row_index + 1} identities must differ")
        require(row.get("identity_pair") == f"{target.get('identity_key')}/{distractor.get('identity_key')}", f"row {row_index + 1} pair mismatch")
        require(target.get("voice_policy") == "speaking", f"row {row_index + 1} target must speak")
        require(distractor.get("voice_policy") == "silent", f"row {row_index + 1} distractor must be silent")
        target_side = row.get("target_expected_screen_side")
        opposite = "right" if target_side == "left" else "left"
        require(target.get("expected_screen_side") == target_side, f"row {row_index + 1} target side mismatch")
        require(distractor.get("expected_screen_side") == opposite, f"row {row_index + 1} distractor side mismatch")

        camera = row.get("camera_pose", {})
        try:
            _camera_basis(camera)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {row_index + 1} camera: {exc}")
            continue
        camera_translation = camera.get("translation_m", [])
        camera_translations.append(camera_translation)
        camera_signatures.add(tuple(round(float(v), 9) for v in [*camera_translation, *camera["rotation_xyzw"]]))
        actor_signatures.add(tuple(round(float(v), 9) for actor in actors for v in actor["root_translation_m"]))

        projected: list[tuple[float, float]] = []
        for actor in actors:
            identity = catalog.get(actor.get("identity_key"), {})
            root = [float(value) for value in actor.get("root_translation_m", [])]
            mouth_offset = [float(value) for value in identity.get("mouth_offset_from_root_m", [])]
            if len(root) != 3 or len(mouth_offset) != 3:
                errors.append(f"row {row_index + 1} actor geometry length mismatch")
                continue
            require(abs(mouth_offset[0]) <= 1e-12 and abs(mouth_offset[2]) <= 1e-12, f"row {row_index + 1} nonvertical mouth offset unsupported")
            mouth = [root[i] + mouth_offset[i] for i in range(3)]
            try:
                _, _, x_fraction, y_fraction = _project(camera, mouth, hfov, resolution)
                _, _, _, envelope_a = _project(camera, [root[0], root[1] + float(envelope[0]), root[2]], hfov, resolution)
                _, _, _, envelope_b = _project(camera, [root[0], root[1] + float(envelope[1]), root[2]], hfov, resolution)
            except (IndexError, TypeError, ValueError) as exc:
                errors.append(f"row {row_index + 1} projection: {exc}")
                continue
            require(float(safe[0]) < x_fraction < float(safe[1]), f"row {row_index + 1} mouth x outside safe frame")
            require(float(safe[0]) < y_fraction < float(safe[1]), f"row {row_index + 1} mouth y outside safe frame")
            require(all(float(safe[0]) < value < float(safe[1]) for value in (envelope_a, envelope_b)), f"row {row_index + 1} vertical envelope outside safe frame")
            side = actor.get("expected_screen_side")
            if side == "left":
                require(x_fraction < 0.5 - dead_zone, f"row {row_index + 1} left actor in dead zone")
            elif side == "right":
                require(x_fraction > 0.5 + dead_zone, f"row {row_index + 1} right actor in dead zone")
            else:
                errors.append(f"row {row_index + 1} actor side invalid")
            projected.append((x_fraction, y_fraction))
            expected_yaw = -math.degrees(
                math.atan2(
                    float(camera_translation[0]) - root[0],
                    float(camera_translation[2]) - root[2],
                )
            )
            require(
                _angle_delta_deg(expected_yaw, float(actor.get("actor_yaw_ue_deg", 9999.0)))
                <= float(thresholds.get("actor_facing_yaw_tolerance_deg", 0.0)),
                f"row {row_index + 1} actor yaw does not face camera",
            )
        if len(projected) == 2:
            require(abs(projected[0][0] - projected[1][0]) >= minimum_x_separation, f"row {row_index + 1} projected actors too close")
        require(_distance_xz(actors[0]["root_translation_m"], actors[1]["root_translation_m"]) >= minimum_actor_separation, f"row {row_index + 1} actor roots too close")

    require(len(camera_signatures) == 8, "eight distinct camera poses required")
    require(len(actor_signatures) == 8, "eight distinct actor-pose pairs required")
    minimum_camera_separation = min(
        _distance_xz(camera_translations[i], camera_translations[j])
        for i in range(len(camera_translations))
        for j in range(i + 1, len(camera_translations))
    )
    require(
        minimum_camera_separation
        >= float(thresholds.get("minimum_camera_translation_cluster_separation_m", 0.0)),
        "camera translation clusters are not separated enough",
    )
    require(
        len(camera_translations)
        >= int(thresholds.get("minimum_camera_translation_cluster_count", 0)),
        "camera translation cluster count too small",
    )
    return errors


def _package_fragment(unreal_object_path: str) -> str:
    return unreal_object_path.split(".", 1)[0].lower().replace(
        "/game/", "spearsim/content/"
    )


def _source_point(
    suite_by_id: Mapping[str, Any],
    suite_root: Path,
    provenance: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    scenario_id = str(provenance["scenario_id"])
    frame_index = int(provenance["frame_index"])
    actor_id = str(provenance["actor_id"])
    scenario = suite_by_id.get(scenario_id)
    if not isinstance(scenario, Mapping):
        raise TypeError(f"source scenario missing: {scenario_id}")
    frames = scenario.get("plan", {}).get("frames", [])
    frame_matches = [frame for frame in frames if frame.get("frame_index") == frame_index]
    if len(frame_matches) != 1:
        raise ValueError(f"source frame does not resolve once: {scenario_id} f{frame_index}")
    actor_matches = [
        actor
        for actor in frame_matches[0].get("actor_states", [])
        if actor.get("actor_id") == actor_id
    ]
    if len(actor_matches) != 1:
        raise ValueError(f"source actor does not resolve once: {scenario_id} f{frame_index} {actor_id}")
    planned = [float(value) for value in actor_matches[0]["translation_m"]]
    readback_path = suite_root / scenario_id / "runtime_readbacks.json"
    readback = _load_json(readback_path)
    readback_matches = [
        frame
        for frame in readback.get("actor_roots", {}).get(actor_id, [])
        if frame.get("frame_index") == frame_index
    ]
    if len(readback_matches) != 1:
        raise ValueError(f"source runtime readback missing: {scenario_id} f{frame_index} {actor_id}")
    location_cm = [float(value) for value in readback_matches[0]["location_cm"]]
    observed = [location_cm[0] / 100.0, location_cm[2] / 100.0, location_cm[1] / 100.0]
    drift = max(abs(planned[i] - observed[i]) for i in range(3))
    if drift > 1e-6:
        raise ValueError(f"source runtime readback drift: {scenario_id} f{frame_index} {actor_id}")
    return planned, {
        "scenario_id": scenario_id,
        "frame_index": frame_index,
        "actor_id": actor_id,
        "runtime_readback": str(readback_path),
        "maximum_location_drift_m": drift,
        "status": "pass_native_occupied_floor_point",
    }


def build(plan_path: Path, output: Path) -> Path:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    plan = _load_json(plan_path)
    registry = _load_json(_resolve(str(plan["evidence"]["runtime_registry"])))
    errors = validate_plan(plan, registry)
    if errors:
        raise RuntimeError("plan validation failed:\n- " + "\n- ".join(errors))

    ledger = _load_json(_resolve(str(plan["evidence"]["passed_canary_ledger"])))
    if not (
        ledger.get("status") == "pass"
        and ledger.get("formal_scene_count") == 0
        and ledger.get("qualification_claim") is False
        and ledger.get("episode_id") == plan["rows"][0]["episode_id"]
    ):
        raise RuntimeError("row1 passed-canary ledger boundary mismatch")

    package_text = _resolve(str(plan["evidence"]["current_package_manifest"])).read_text(
        encoding="utf-8"
    )
    package_lower = package_text.lower()
    identity_readiness: dict[str, Any] = {}
    for key, identity in plan["approved_identity_catalog"].items():
        profile = _profile_by_id(registry, str(identity["runtime_asset_id"]))
        unreal = profile["runtime_backends"]["spear_unreal"]
        required = [
            unreal["blueprint_class_path"],
            unreal["idle_animation"],
            unreal["walking_animation"],
        ]
        missing = [value for value in required if _package_fragment(value) not in package_lower]
        if missing:
            raise RuntimeError(f"identity {key} package objects missing: {missing}")
        identity_readiness[key] = {
            "original_identity_id": identity["original_identity_id"],
            "runtime_profile": "pass_exactly_once",
            "package_blueprint_idle_walking": "pass",
            "listening_review": identity["listening_review"],
            "formal_admission": False,
        }

    suite_path = _resolve(str(plan["source_floor_point_suite"]))
    suite = _load_json(suite_path)
    suite_by_id = {scenario["scenario_id"]: scenario for scenario in suite["scenarios"]}
    suite_root = suite_path.parent
    point_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    for index, row in enumerate(plan["rows"]):
        if index == 0:
            row_records.append(
                {
                    "row_id": row["row_id"],
                    "status": "pass_existing_sparse_canary",
                    "native_combined_clearance": "pass_retained_canary_only",
                    "exact_rir": "pass_retained_canary_only",
                }
            )
            continue
        expected_camera_floor = [
            float(row["camera_pose"]["translation_m"][0]),
            0.4,
            float(row["camera_pose"]["translation_m"][2]),
        ]
        observed_camera_floor, camera_record = _source_point(
            suite_by_id, suite_root, row["camera_floor_point_provenance"]
        )
        if max(abs(expected_camera_floor[i] - observed_camera_floor[i]) for i in range(3)) > 1e-6:
            raise RuntimeError(f"row {row['row_id']} camera floor provenance mismatch")
        camera_record["row_id"] = row["row_id"]
        camera_record["planned_role"] = "camera_floor"
        point_records.append(camera_record)
        for actor in row["actors"]:
            observed, actor_record = _source_point(
                suite_by_id, suite_root, actor["floor_point_provenance"]
            )
            planned = [float(value) for value in actor["root_translation_m"]]
            if max(abs(planned[i] - observed[i]) for i in range(3)) > 1e-6:
                raise RuntimeError(f"row {row['row_id']} actor floor provenance mismatch")
            actor_record["row_id"] = row["row_id"]
            actor_record["planned_role"] = actor["role"]
            point_records.append(actor_record)
        row_records.append(
            {
                "row_id": row["row_id"],
                "status": "pass_cpu_geometry_pending_exact_rir_and_native_sparse",
                "native_occupied_floor_points": "pass_3_of_3",
                "native_combined_clearance": "pending_sparse_required",
                "native_bbox_depth_target_only_readback": "pending_sparse_required",
                "exact_rir": "pending_required",
            }
        )

    cameras = [row["camera_pose"]["translation_m"] for row in plan["rows"]]
    minimum_camera_separation = min(
        _distance_xz(cameras[i], cameras[j])
        for i in range(len(cameras))
        for j in range(i + 1, len(cameras))
    )
    output.mkdir(parents=True)
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates",
        "claim_boundary": plan["claim_boundary"],
        "plan_id": plan["plan_id"],
        "plan_record": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256(plan_path),
        },
        "row_count": 8,
        "left_target_count": 4,
        "right_target_count": 4,
        "camera_translation_cluster_count": 8,
        "minimum_camera_translation_separation_m": minimum_camera_separation,
        "native_occupied_floor_point_count": len(point_records),
        "identity_readiness": identity_readiness,
        "rows": row_records,
        "occupied_floor_point_evidence": point_records,
        "formal_scene_count": 0,
        "qualification_claim": False,
        "gpu_or_rir_executed": False,
        "next_required_steps": [
            "build_per_row_exact_two_human_RIR",
            "run_rows_2_to_8_one_at_a_time_on_idle_physical_GPU1",
            "require_native_RGB_metric_depth_two_target_only_runtime_readback_and_projection_gates",
            "keep_all_three_candidate_voices_research_only_until_listening_review",
        ],
    }
    result_path = output / "preflight.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.plan.resolve(), args.output.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
