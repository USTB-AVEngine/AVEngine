#!/usr/bin/env python3
"""Build one CPU-only camera-pan/both-static full75 geometry candidate.

The candidate reuses an exact native two-root pair and the qualified common
camera translation.  Only the camera yaw changes.  Metric-depth screening is
performed in world-ray space from the unchanged camera center, while the
planned yaw path is used for every-frame projection/envelope checks.  This
tool does not authorize materialization, RIR work, or GPU capture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

import build_strict_two_human_distractor_moves_v2_candidate as base


FRAME_COUNT = 75
YAW_START_DEG = 52.0
YAW_END_DEG = 58.0
MINIMUM_YAW_SPAN_DEG = 5.9
MAXIMUM_STATIC_DRIFT_M = 1.0e-6
MINIMUM_MIDLINE_DEAD_ZONE_FRACTION = 0.01
MINIMUM_FRAME_EDGE_MARGIN_FRACTION = 0.05
TARGET_ROOT = list(base.NATIVE_HUMAN_ANCHORS[0])
DISTRACTOR_ROOT = list(base.STATIC_TARGET_ROOT)
SOURCE_SCENARIO = base.SOURCE_SCENARIO
CAMERA_CLUSTER = base.CAMERA_CLUSTER
EPISODE_ID = "strict2h_dynamic_canary_04_camera_pan_both_static_v2"
CANDIDATE_REVISION = "camera_pan_v2_0589_right_target_yaw52_58_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def yaw_path() -> list[float]:
    return [
        YAW_START_DEG + (YAW_END_DEG - YAW_START_DEG) * frame_index / (FRAME_COUNT - 1)
        for frame_index in range(FRAME_COUNT)
    ]


def project(
    point: Sequence[float], camera_yaw_deg: float, height_m: float = 0.0
) -> tuple[float, float, float]:
    yaw = math.radians(camera_yaw_deg)
    forward = (-math.sin(yaw), -math.cos(yaw))
    right = (-forward[1], forward[0])
    dx = float(point[0]) - base.CAMERA[0]
    dz = float(point[2]) - base.CAMERA[2]
    depth = dx * forward[0] + dz * forward[1]
    lateral = dx * right[0] + dz * right[1]
    tan_horizontal = math.tan(math.radians(base.HFOV_DEG) / 2.0)
    tan_vertical = tan_horizontal * base.RESOLUTION_HW[0] / base.RESOLUTION_HW[1]
    x_fraction = 0.5 + lateral / (2.0 * depth * tan_horizontal)
    y_fraction = 0.5 - (float(point[1]) + height_m - base.CAMERA[1]) / (
        2.0 * depth * tan_vertical
    )
    return depth, x_fraction, y_fraction


def static_drift(path: Sequence[Sequence[float]]) -> float:
    return max(math.dist(path[0], point) for point in path)


def facing_to_midpoint_camera(root: Sequence[float]) -> list[float]:
    dx = base.CAMERA[0] - float(root[0])
    dz = base.CAMERA[2] - float(root[2])
    norm = math.hypot(dx, dz)
    require(norm > 0.0, "actor root equals camera center")
    return [dx / norm, 0.0, dz / norm]


def projection_metrics(
    target_path: Sequence[Sequence[float]],
    distractor_path: Sequence[Sequence[float]],
    camera_yaws: Sequence[float],
) -> dict[str, Any]:
    target_centers = [
        project(root, camera_yaw)
        for root, camera_yaw in zip(target_path, camera_yaws, strict=True)
    ]
    distractor_centers = [
        project(root, camera_yaw)
        for root, camera_yaw in zip(distractor_path, camera_yaws, strict=True)
    ]
    target_envelopes = [
        project(point, camera_yaw)
        for root, camera_yaw in zip(target_path, camera_yaws, strict=True)
        for point in base.cylinder_points(root)
    ]
    distractor_envelopes = [
        project(point, camera_yaw)
        for root, camera_yaw in zip(distractor_path, camera_yaws, strict=True)
        for point in base.cylinder_points(root)
    ]
    center_separations = [
        abs(target[1] - distractor[1])
        for target, distractor in zip(target_centers, distractor_centers, strict=True)
    ]
    target_x_range = [
        min(value[1] for value in target_envelopes),
        max(value[1] for value in target_envelopes),
    ]
    distractor_x_range = [
        min(value[1] for value in distractor_envelopes),
        max(value[1] for value in distractor_envelopes),
    ]
    return {
        "frame_count": FRAME_COUNT,
        "camera_yaw_span_deg": max(camera_yaws) - min(camera_yaws),
        "target_center_x_fraction_range": [
            min(value[1] for value in target_centers),
            max(value[1] for value in target_centers),
        ],
        "distractor_center_x_fraction_range": [
            min(value[1] for value in distractor_centers),
            max(value[1] for value in distractor_centers),
        ],
        "target_center_depth_m_range": [
            min(value[0] for value in target_centers),
            max(value[0] for value in target_centers),
        ],
        "distractor_center_depth_m_range": [
            min(value[0] for value in distractor_centers),
            max(value[0] for value in distractor_centers),
        ],
        "target_all75_2m_cylinder_x_fraction_range": target_x_range,
        "target_all75_2m_cylinder_y_fraction_range": [
            min(value[2] for value in target_envelopes),
            max(value[2] for value in target_envelopes),
        ],
        "distractor_all75_2m_cylinder_x_fraction_range": distractor_x_range,
        "distractor_all75_2m_cylinder_y_fraction_range": [
            min(value[2] for value in distractor_envelopes),
            max(value[2] for value in distractor_envelopes),
        ],
        "minimum_projected_center_x_separation_fraction": min(center_separations),
        "target_right_midline_dead_zone_fraction": target_x_range[0] - 0.5,
        "distractor_left_midline_dead_zone_fraction": 0.5 - distractor_x_range[1],
        "minimum_frame_edge_margin_fraction": min(
            target_x_range[0],
            1.0 - target_x_range[1],
            distractor_x_range[0],
            1.0 - distractor_x_range[1],
        ),
        "minimum_actor_horizontal_separation_m": min(
            math.hypot(
                float(target[0]) - float(distractor[0]),
                float(target[2]) - float(distractor[2]),
            )
            for target, distractor in zip(target_path, distractor_path, strict=True)
        ),
    }


def _range_inside_unit_interval(value: Sequence[float]) -> bool:
    return 0.0 <= float(value[0]) <= float(value[1]) <= 1.0


def build_documents(
    depth_path: Path, masks_path: Path, readback_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    camera_yaws = yaw_path()
    target_path = [list(TARGET_ROOT) for _ in range(FRAME_COUNT)]
    distractor_path = [list(DISTRACTOR_ROOT) for _ in range(FRAME_COUNT)]
    target_drift = static_drift(target_path)
    distractor_drift = static_drift(distractor_path)
    projection = projection_metrics(target_path, distractor_path, camera_yaws)

    require(
        projection["camera_yaw_span_deg"] >= MINIMUM_YAW_SPAN_DEG,
        "camera yaw span is below the mechanism gate",
    )
    require(target_drift <= MAXIMUM_STATIC_DRIFT_M, "target root drifted")
    require(distractor_drift <= MAXIMUM_STATIC_DRIFT_M, "distractor root drifted")
    require(
        projection["target_right_midline_dead_zone_fraction"]
        >= MINIMUM_MIDLINE_DEAD_ZONE_FRACTION,
        "right target envelope entered the midline dead zone",
    )
    require(
        projection["distractor_left_midline_dead_zone_fraction"]
        >= MINIMUM_MIDLINE_DEAD_ZONE_FRACTION,
        "left distractor envelope entered the midline dead zone",
    )
    require(
        projection["minimum_frame_edge_margin_fraction"]
        >= MINIMUM_FRAME_EDGE_MARGIN_FRACTION,
        "actor envelope entered the frame-edge dead zone",
    )
    for key in (
        "target_all75_2m_cylinder_x_fraction_range",
        "target_all75_2m_cylinder_y_fraction_range",
        "distractor_all75_2m_cylinder_x_fraction_range",
        "distractor_all75_2m_cylinder_y_fraction_range",
    ):
        require(_range_inside_unit_interval(projection[key]), f"{key} left frame")

    normal_depth, actor_footprints, depth_authority = base.load_depth_authority(
        depth_path, masks_path, readback_path
    )
    target_corridor = base.corridor_metrics(target_path, normal_depth, actor_footprints)
    distractor_corridor = base.corridor_metrics(
        distractor_path, normal_depth, actor_footprints
    )
    require(target_corridor["status"] == "pass", "target depth corridor failed")
    require(
        distractor_corridor["status"] == "pass",
        "distractor depth corridor failed",
    )

    target_forward = facing_to_midpoint_camera(TARGET_ROOT)
    distractor_forward = facing_to_midpoint_camera(DISTRACTOR_ROOT)
    camera_ue_yaws = [-90.0 - value for value in camera_yaws]
    row: dict[str, Any] = {
        "execution_order": 4,
        "episode_id": EPISODE_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "mechanism": "camera_pan_both_static",
        "target_side": "right",
        "target": {
            "content_id": "cremad_ieo_v1",
            "frame_index_map": [base.NATIVE_HUMAN_FRAME_RANGE[0]] * FRAME_COUNT,
            "identity_id": "rocketbox_adults_male_adult_01",
            "identity_key": "M",
            "listening_review": "pending",
            "rights_status": "review_required",
            "root_path_m": target_path,
            "runtime_asset_id": "rocketbox_human_male_adult_01_m5_1_candidate",
            "runtime_revision": "native_runtime_ue_v3",
            "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
            "source_actor_id": base.NATIVE_HUMAN_ACTOR,
            "source_slot_id": "source1",
            "path_provenance": {
                "method": "exact_native_root_held_all75_v1",
                "native_source_scenario_id": SOURCE_SCENARIO,
                "native_source_actor_id": base.NATIVE_HUMAN_ACTOR,
                "native_source_frame_index": base.NATIVE_HUMAN_FRAME_RANGE[0],
                "original_species": "human",
                "replacement_identity_species": "human",
                "fresh_pan_pixels_required": True,
            },
            "per_frame_anatomical_forward_habitat_world": [target_forward]
            * FRAME_COUNT,
            "facing_policy": "face_camera_midpoint_then_hold_all75",
            "per_frame_action_phase": [0.0] * FRAME_COUNT,
            "speech_frame_window_inclusive": [7, 31],
            "speech_sample_count": 25626,
            "transcript": "It's eleven o'clock.",
            "voice_id": "cremad_actor_1001",
            "voice_policy": "speaking",
        },
        "distractor": {
            "frame_index_map": [base.NATIVE_DOG_FRAME] * FRAME_COUNT,
            "identity_id": "rocketbox_adults_female_adult_01",
            "identity_key": "F",
            "listening_review": "not_applicable_silent",
            "root_path_m": distractor_path,
            "runtime_asset_id": ("lead_b_rocketbox_adults_female_adult_01_original_v1"),
            "runtime_revision": "native_runtime_ue_v1",
            "source_actor_id": base.NATIVE_DOG_ACTOR,
            "source_slot_id": "source2",
            "path_provenance": {
                "method": "exact_native_root_held_all75_v1",
                "native_source_scenario_id": SOURCE_SCENARIO,
                "native_source_actor_id": base.NATIVE_DOG_ACTOR,
                "native_source_frame_index": base.NATIVE_DOG_FRAME,
                "original_species": "dog",
                "replacement_identity_species": "human",
                "fresh_pan_pixels_required": True,
            },
            "per_frame_anatomical_forward_habitat_world": [distractor_forward]
            * FRAME_COUNT,
            "facing_policy": "face_camera_midpoint_then_hold_all75",
            "per_frame_action_phase": [0.0] * FRAME_COUNT,
            "voice_policy": "silent",
        },
        "native_source_scenario_id": SOURCE_SCENARIO,
        "camera_cluster_id": CAMERA_CLUSTER,
        "camera": {
            "translation_m": list(base.CAMERA),
            "translation_path_m": [list(base.CAMERA)] * FRAME_COUNT,
            "yaw_path_deg": camera_yaws,
            "ue_yaw_path_deg": camera_ue_yaws,
            "horizontal_fov_deg": base.HFOV_DEG,
            "provenance": {
                "source_suite": base.SOURCE_SUITE,
                "midpoint_fresh_capture_episode": (
                    "strict2h_dynamic_canary_02_distractor_moves_v2"
                ),
                "coordinate_contract": "UE_yaw_deg=-90-Habitat_yaw_deg",
            },
        },
        "motion_preflight": {
            "status": "pass",
            "mechanism": "camera_pan_both_static",
            "target": {
                "expected_moving": False,
                "maximum_root_displacement_m": target_drift,
                "maximum_forward_angular_drift_deg": 0.0,
                "unique_root_positions_at_1mm": 1,
                "status": "pass",
            },
            "distractor": {
                "expected_moving": False,
                "maximum_root_displacement_m": distractor_drift,
                "maximum_forward_angular_drift_deg": 0.0,
                "unique_root_positions_at_1mm": 1,
                "status": "pass",
            },
            "camera": {
                "expected_moving": True,
                "translation_drift_m": 0.0,
                "yaw_span_deg": projection["camera_yaw_span_deg"],
                "unique_yaws_at_1e4_deg": len(
                    {round(value, 4) for value in camera_yaws}
                ),
                "status": "pass",
            },
        },
        "projection_preflight": projection,
        "depth_corridor_preflight": {
            "authority": depth_authority,
            "world_ray_orientation_invariance": (
                "camera translation is fixed; each actor-envelope world ray and its environment range are orientation-invariant. "
                "The planned yaw path is checked separately for all75 viewport containment."
            ),
            "static_target_all75": target_corridor,
            "static_distractor_all75": distractor_corridor,
        },
        "source_suite": base.SOURCE_SUITE,
        "suite_plan": "PENDING_CAMERA_PAN_V2_CPU_MATERIALIZATION",
        "exact_rir_plan": "PENDING_CAMERA_PAN_V2_EXACT_RIR_PLAN",
        "binaural_audio": "PENDING_CAMERA_PAN_V2_BINAURAL_RENDER",
        "gpu_launch_authorized": False,
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "formal": False,
        "qualification_claim": False,
        "status": "pass_cpu_geometry_only_fresh_pan_pixels_pending",
    }
    preflight = {
        "schema": (
            "avengine_native_strict_two_human_dynamic_full75_canary_preflight_v1"
        ),
        "status": "pass_cpu_geometry_only_fresh_pan_pixels_pending",
        "dynamic_canary_count": 1,
        "dynamic_canary_gpu_pass_count": 0,
        "single_room_mechanism_pilot_authorized": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "unique_source_scenario_count": 1,
        "unique_camera_cluster_count": 1,
        "target_side_counts": {"right": 1},
        "canaries": [row],
    }
    receipt = {
        "schema": (
            "avengine_native_strict_two_human_camera_pan_v2_cpu_geometry_receipt_v1"
        ),
        "status": "go_cpu_geometry_only_fresh_pan_pixels_still_required",
        "candidate_decision": "GO_CPU_GEOMETRY_ONLY",
        "candidate": row,
        "claim_boundary": (
            "Exact held native roots, all75 pinhole projection/envelopes, and fixed-center metric-depth world-ray screening pass. "
            "They do not prove fresh pan-sequence F/M pixels, physics collisions, foot-ground contact, or runtime camera readback."
        ),
        "motion_contract": row["motion_preflight"],
        "projection_preflight": projection,
        "depth_corridor_preflight": row["depth_corridor_preflight"],
        "retained_midpoint_pixel_evidence": {
            "frame": (
                "/data/jzy/code/AVEngine-lead-a/tmp/lead_a_strict_two_human_full_episode_batch_v1/"
                "dynamic_distractor_moves_v2_capture_attempt_01/rgb_frames/frame_000000.png"
            ),
            "camera_yaw_habitat_deg": base.CAMERA_HABITAT_YAW_DEG,
            "review_result": "fresh F/M identities full-body visible at the planned pan midpoint",
            "use_boundary": "midpoint diagnostic only; not endpoint or all75 pan pixel truth",
        },
        "strict_native_acceptance_gate": {
            "target_speech_window_inclusive": [7, 31],
            "target_minimum_visible_pixels_during_speech": 10000,
            "target_minimum_visible_fraction_during_speech": 0.8,
            "distractor_minimum_visible_pixels_all75": 5000,
            "distractor_minimum_visible_fraction_all75": 0.5,
            "runtime_camera_yaw_readback_required_all75": True,
            "status": "pending_fresh_native_pan_capture",
        },
        "camera_cluster_scope": {
            "camera_cluster_id": CAMERA_CLUSTER,
            "mechanism_canary_only": True,
            "independent_episode_claim": False,
            "reason": (
                "reuses the qualified common-camera center and is not counted toward the 100 unique clusters"
            ),
        },
        "dynamic_canary_side_balance_if_accepted": {
            "target_moves": "left",
            "distractor_moves": "left",
            "both_move": "right",
            "camera_pan_both_static": "right",
            "counts": {"left": 2, "right": 2},
            "status": "pass_2_left_2_right",
        },
        "single_attempt_policy": {
            "maximum_gpu_attempts_for_this_candidate": 1,
            "gpu_attempts_used": 0,
            "on_any_native_gate_failure": (
                "freeze rejected receipt and advance to a new candidate; do not tune against pixels"
            ),
        },
        "acoustic_state_expectation": {
            "source_frame_uses": 150,
            "target_unique_rir_states": 75,
            "distractor_unique_rir_states": 75,
            "total_unique_rir_states": 150,
            "reason": "listener orientation changes every frame even though both sources are static",
            "exact_rir_required_before_gpu": True,
            "status": "not_executed",
        },
        "next_authorized_step": (
            "await root review; no materialization, RIR, or GPU is authorized by this receipt"
        ),
        "gpu_launch_authorized": False,
        "formal_episode_count": 0,
    }
    return preflight, receipt


def _screen_box(root: Sequence[float], camera_yaw: float) -> tuple[int, int, int, int]:
    points = [project(point, camera_yaw) for point in base.cylinder_points(root)]
    width = base.RESOLUTION_HW[1]
    height = base.RESOLUTION_HW[0]
    xs = [round(value[1] * (width - 1)) for value in points]
    ys = [round(value[2] * (height - 1)) for value in points]
    return min(xs), min(ys), max(xs), max(ys)


def build_overlay(
    midpoint_frame: Path, output_path: Path, receipt: dict[str, Any]
) -> None:
    require(midpoint_frame.is_file(), "fresh midpoint frame missing")
    background = Image.open(midpoint_frame).convert("RGB")
    require(background.size == (1280, 720), "midpoint frame size drift")
    draw = ImageDraw.Draw(background, "RGBA")
    font = ImageFont.load_default()
    for root, color, label in (
        (TARGET_ROOT, (255, 60, 60, 255), "static speaking target M"),
        (DISTRACTOR_ROOT, (60, 170, 255, 255), "static silent distractor F"),
    ):
        box = _screen_box(root, base.CAMERA_HABITAT_YAW_DEG)
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 4, box[1] + 4), label, fill=color, font=font)
    draw.rectangle((6, 6, 690, 58), fill=(0, 0, 0, 180))
    draw.text(
        (14, 13),
        "Fresh F/M midpoint frame at yaw 55 deg - not all75 pan truth",
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (14, 33),
        "planned camera pan: 52 -> 58 deg; both actor transforms held static",
        fill=(255, 255, 255),
        font=font,
    )

    panel = Image.new("RGB", (640, 720), (247, 248, 250))
    panel_draw = ImageDraw.Draw(panel, "RGBA")
    projection = receipt["projection_preflight"]
    camera_yaws = yaw_path()
    target_centers = [project(TARGET_ROOT, value)[1] for value in camera_yaws]
    distractor_centers = [project(DISTRACTOR_ROOT, value)[1] for value in camera_yaws]

    def plot_point(frame_index: int, x_fraction: float) -> tuple[int, int]:
        return (
            55 + round(frame_index / (FRAME_COUNT - 1) * 540),
            610 - round(x_fraction * 460),
        )

    panel_draw.rectangle((55, 150, 595, 610), outline=(90, 90, 90), width=2)
    middle_y = plot_point(0, 0.5)[1]
    panel_draw.line((55, middle_y, 595, middle_y), fill=(80, 80, 80), width=2)
    panel_draw.line(
        [plot_point(i, value) for i, value in enumerate(target_centers)],
        fill=(220, 45, 45),
        width=5,
    )
    panel_draw.line(
        [plot_point(i, value) for i, value in enumerate(distractor_centers)],
        fill=(35, 125, 230),
        width=5,
    )
    panel_draw.text((20, 18), "CPU camera-pan projection audit", fill=(20, 20, 20))
    panel_draw.text(
        (20, 46),
        f"yaw span: {projection['camera_yaw_span_deg']:.3f} deg / 75 unique yaws",
        fill=(20, 20, 20),
    )
    panel_draw.text(
        (20, 70),
        "red RIGHT target 2m envelope: "
        f"{projection['target_all75_2m_cylinder_x_fraction_range'][0]:.3f}-"
        f"{projection['target_all75_2m_cylinder_x_fraction_range'][1]:.3f}",
        fill=(180, 35, 35),
    )
    panel_draw.text(
        (20, 94),
        "blue LEFT distractor 2m envelope: "
        f"{projection['distractor_all75_2m_cylinder_x_fraction_range'][0]:.3f}-"
        f"{projection['distractor_all75_2m_cylinder_x_fraction_range'][1]:.3f}",
        fill=(25, 95, 190),
    )
    depth = receipt["depth_corridor_preflight"]
    panel_draw.text(
        (20, 640),
        "depth min clearance: target "
        f"+{depth['static_target_all75']['minimum_depth_clearance_m']:.3f}m; "
        "distractor "
        f"+{depth['static_distractor_all75']['minimum_depth_clearance_m']:.3f}m",
        fill=(20, 120, 55),
    )
    panel_draw.text(
        (20, 670),
        "fresh all75 pan pixels/runtime readback: PENDING",
        fill=(150, 80, 10),
    )
    combined = Image.new("RGB", (1920, 720), (255, 255, 255))
    combined.paste(background, (0, 0))
    combined.paste(panel, (1280, 0))
    combined.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-depth", type=Path, required=True)
    parser.add_argument("--actor-masks", type=Path, required=True)
    parser.add_argument("--camera-readbacks", type=Path, required=True)
    parser.add_argument("--midpoint-frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    output.mkdir(parents=True)
    try:
        preflight, receipt = build_documents(
            args.metric_depth.resolve(),
            args.actor_masks.resolve(),
            args.camera_readbacks.resolve(),
        )
        write_json(output / "camera_pan_v2_preflight.json", preflight)
        write_json(output / "cpu_geometry_receipt.json", receipt)
        build_overlay(
            args.midpoint_frame.resolve(),
            output / "camera_pan_v2_projection_depth_overlay.png",
            receipt,
        )
    except Exception as exc:
        write_json(
            output / "failure_receipt.json",
            {
                "schema": "avengine_camera_pan_v2_cpu_geometry_failure_v1",
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    projection = receipt["projection_preflight"]
    depth = receipt["depth_corridor_preflight"]
    print(
        "CAMERA_PAN_V2_CPU_GEOMETRY_GO "
        f"yaw_span_deg={projection['camera_yaw_span_deg']:.6f} "
        f"target_x_min={projection['target_all75_2m_cylinder_x_fraction_range'][0]:.6f} "
        f"distractor_x_max={projection['distractor_all75_2m_cylinder_x_fraction_range'][1]:.6f} "
        f"target_clearance_m={depth['static_target_all75']['minimum_depth_clearance_m']:.6f} "
        f"distractor_clearance_m={depth['static_distractor_all75']['minimum_depth_clearance_m']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
