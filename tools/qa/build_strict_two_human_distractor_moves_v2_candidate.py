#!/usr/bin/env python3
"""Build the CPU-only distractor-moves v2 geometry candidate.

The builder binds the authoritative 55-degree Habitat camera convention,
resamples a retained native-human polyline to 75 equal-arc roots, and screens
2 m human cylinders against a same-camera, static-room metric-depth sequence.
It does not authorize materialization, RIR work, or GPU capture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
RESOLUTION_HW = (720, 1280)
HFOV_DEG = 105.0
CAMERA = (-0.7, 1.471, 0.65)
CAMERA_HABITAT_YAW_DEG = 55.0
CAMERA_UE_YAW_DEG = -145.0
CAMERA_CLUSTER = "apartment_grid075_x-01_z+00"
SOURCE_SCENARIO = "human_border_collie__recombined_both_moving_0589"
SOURCE_SUITE = (
    "/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/"
    "apartment_asset_bound_ue_unique1000_full_20260723_01/suite_execution_plan.json"
)
SOURCE_SUITE_ROOT = str(Path(SOURCE_SUITE).parent)
NATIVE_HUMAN_ACTOR = "source1_actor"
NATIVE_HUMAN_FRAME_RANGE = (2, 17)
NATIVE_DOG_ACTOR = "source2_actor"
NATIVE_DOG_FRAME = 6
SPEECH_WINDOW = (7, 50)
BODY_RADIUS_M = 0.22
BODY_HEIGHT_M = 2.0
CYLINDER_HEIGHT_SAMPLES = 14
CYLINDER_AZIMUTH_SAMPLES = 24
MINIMUM_DEPTH_CLEARANCE_M = 0.25

# These are the exact Habitat roots retained from the source suite f2-f17.
NATIVE_HUMAN_ANCHORS = [
    [-2.020261973948092, 0.4000000059604645, -1.2970139013754356],
    [-2.0072732728880807, 0.4000000059604645, -1.3521811108331423],
    [-1.9942845718280688, 0.4000000059604645, -1.407348320290849],
    [-1.9812958707680572, 0.4000000059604645, -1.4625155297485557],
    [-1.9683071697080456, 0.4000000059604645, -1.5176827392062626],
    [-1.9553184686480343, 0.4000000059604645, -1.5728499486639693],
    [-1.9423297675880227, 0.4000000059604645, -1.628017158121676],
    [-1.929341066528011, 0.4000000059604645, -1.683184367579383],
    [-1.9163523654679995, 0.4000000059604645, -1.7383515770370894],
    [-1.9033636644079879, 0.4000000059604645, -1.7935187864947963],
    [-1.8903749633479763, 0.4000000059604645, -1.8486859959525033],
    [-1.8773862622879647, 0.4000000059604645, -1.90385320541021],
    [-1.864397561227953, 0.4000000059604645, -1.9590204148679167],
    [-1.8514088601679415, 0.4000000059604645, -2.0141876243256234],
    [-1.8384201591079299, 0.4000000059604645, -2.06935483378333],
    [-1.8254314580479183, 0.4000000059604645, -2.124522043241037],
]
STATIC_TARGET_ROOT = [-3.215222624910844, 0.4000000059604645, -0.5506341006304767]
NATIVE_PHASE_START = 0.125
NATIVE_PHASE_ADVANCE = 0.9375
NATIVE_FORWARD = [0.2291761100087884, 0.0, -0.9733849755370378]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def project(
    point: Sequence[float], height_m: float = 0.0
) -> tuple[float, float, float]:
    yaw = math.radians(CAMERA_HABITAT_YAW_DEG)
    forward = (-math.sin(yaw), -math.cos(yaw))
    right = (-forward[1], forward[0])
    dx = float(point[0]) - CAMERA[0]
    dz = float(point[2]) - CAMERA[2]
    depth = dx * forward[0] + dz * forward[1]
    lateral = dx * right[0] + dz * right[1]
    tan_horizontal = math.tan(math.radians(HFOV_DEG) / 2.0)
    tan_vertical = tan_horizontal * RESOLUTION_HW[0] / RESOLUTION_HW[1]
    x_fraction = 0.5 + lateral / (2.0 * depth * tan_horizontal)
    y_fraction = 0.5 - (float(point[1]) + height_m - CAMERA[1]) / (
        2.0 * depth * tan_vertical
    )
    return depth, x_fraction, y_fraction


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[2]) - float(a[2]))
        for a, b in zip(points, points[1:])
    )


def arc_length_resample(
    points: Sequence[Sequence[float]], output_count: int
) -> list[list[float]]:
    cumulative = [0.0]
    for previous, current in zip(points, points[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(
                float(current[0]) - float(previous[0]),
                float(current[2]) - float(previous[2]),
            )
        )
    require(cumulative[-1] > 0.0, "source polyline is stationary")
    result: list[list[float]] = []
    segment_index = 0
    for output_index in range(output_count):
        distance = cumulative[-1] * output_index / (output_count - 1)
        while (
            segment_index + 1 < len(cumulative) - 1
            and cumulative[segment_index + 1] < distance
        ):
            segment_index += 1
        segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
        alpha = (distance - cumulative[segment_index]) / segment_length
        result.append(
            [
                float(points[segment_index][axis])
                + alpha
                * (
                    float(points[segment_index + 1][axis])
                    - float(points[segment_index][axis])
                )
                for axis in range(3)
            ]
        )
    return result


def motion_metrics(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    steps = [
        math.hypot(float(b[0]) - float(a[0]), float(b[2]) - float(a[2]))
        for a, b in zip(points, points[1:])
    ]
    length = sum(steps)
    speeds = [value * FRAME_RATE_HZ for value in steps]
    accelerations = [
        abs(current - previous) * FRAME_RATE_HZ
        for previous, current in zip(speeds, speeds[1:])
    ]
    return {
        "horizontal_path_length_m": length,
        "maximum_displacement_from_start_m": max(
            math.hypot(
                float(point[0]) - float(points[0][0]),
                float(point[2]) - float(points[0][2]),
            )
            for point in points
        ),
        "unique_root_positions_at_1mm": len(
            {(round(float(point[0]), 3), round(float(point[2]), 3)) for point in points}
        ),
        "minimum_interframe_step_m": min(steps, default=0.0),
        "maximum_interframe_step_m": max(steps, default=0.0),
        "minimum_horizontal_speed_m_s": min(speeds, default=0.0),
        "maximum_horizontal_speed_m_s": max(speeds, default=0.0),
        "maximum_interior_horizontal_acceleration_m_s2": max(
            accelerations, default=0.0
        ),
    }


def cylinder_points(root: Sequence[float]) -> list[list[float]]:
    result: list[list[float]] = []
    for height_m in np.linspace(0.05, BODY_HEIGHT_M, CYLINDER_HEIGHT_SAMPLES):
        for angle in np.linspace(
            0.0, 2.0 * math.pi, CYLINDER_AZIMUTH_SAMPLES, endpoint=False
        ):
            result.append(
                [
                    float(root[0]) + BODY_RADIUS_M * math.cos(float(angle)),
                    float(root[1]) + float(height_m),
                    float(root[2]) + BODY_RADIUS_M * math.sin(float(angle)),
                ]
            )
    return result


def load_depth_authority(
    depth_path: Path, masks_path: Path, readback_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(
        depth_path.is_file() and masks_path.is_file() and readback_path.is_file(),
        "depth authority inputs missing",
    )
    with np.load(depth_path) as archive:
        normal_depth = archive["normal_depth_m"].astype(np.float32)
    with np.load(masks_path) as archive:
        actor_footprints = (archive["target_only_source1"] > 0) | (
            archive["target_only_source2"] > 0
        )
    require(normal_depth.shape == (75, 720, 1280), "depth shape drift")
    require(actor_footprints.shape == normal_depth.shape, "mask shape drift")
    readbacks = json.loads(readback_path.read_text(encoding="utf-8"))
    camera_rows = [row["camera"] for row in readbacks["normal"]]
    require(len(camera_rows) == 75, "depth camera readback frame count drift")
    expected_location = [-70.0, 65.0, 147.10000000000002]
    expected_rotation = [0.0, 0.0, CAMERA_UE_YAW_DEG]
    require(
        all(row["location_cm"] == expected_location for row in camera_rows)
        and all(row["rotation_deg"] == expected_rotation for row in camera_rows),
        "depth authority camera is not the exact static 0589 camera",
    )
    metadata = {
        "metric_depth_path": str(depth_path.resolve()),
        "actor_footprint_path": str(masks_path.resolve()),
        "runtime_camera_readback_path": str(readback_path.resolve()),
        "frame_count": 75,
        "unique_camera_pose_count": 1,
        "camera_location_ue_cm": expected_location,
        "camera_rotation_ue_deg": expected_rotation,
        "environment_depth_reconstruction": (
            "per-pixel median normal depth over frames where neither target-only actor footprint covers the pixel"
        ),
    }
    return normal_depth, actor_footprints, metadata


def corridor_metrics(
    roots: Sequence[Sequence[float]],
    normal_depth: np.ndarray,
    actor_footprints: np.ndarray,
) -> dict[str, Any]:
    height, width = RESOLUTION_HW
    clearances: list[float] = []
    observation_counts: list[int] = []
    depth_spreads: list[float] = []
    out_of_view = 0
    minimum_frame_clearances: list[float] = []
    for root in roots:
        frame_clearances: list[float] = []
        for point in cylinder_points(root):
            _, x_fraction, y_fraction = project(point)
            pixel_x = round(x_fraction * (width - 1))
            pixel_y = round(y_fraction * (height - 1))
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                out_of_view += 1
                continue
            unoccluded = ~actor_footprints[:, pixel_y, pixel_x]
            values = normal_depth[:, pixel_y, pixel_x][unoccluded]
            require(
                values.size > 0, "no environment depth observation at cylinder sample"
            )
            environment_range_m = float(np.median(values))
            sample_range_m = math.dist(point, CAMERA)
            clearance = environment_range_m - sample_range_m
            clearances.append(clearance)
            frame_clearances.append(clearance)
            observation_counts.append(int(values.size))
            depth_spreads.append(float(np.max(values) - np.min(values)))
        require(frame_clearances, "cylinder frame has no in-view samples")
        minimum_frame_clearances.append(min(frame_clearances))
    values = np.asarray(clearances, dtype=np.float64)
    return {
        "root_frame_count": len(roots),
        "cylinder_radius_m": BODY_RADIUS_M,
        "cylinder_height_m": BODY_HEIGHT_M,
        "height_sample_count": CYLINDER_HEIGHT_SAMPLES,
        "azimuth_sample_count": CYLINDER_AZIMUTH_SAMPLES,
        "total_cylinder_sample_count": len(clearances),
        "out_of_view_sample_count": out_of_view,
        "minimum_environment_observation_count_per_sample": min(observation_counts),
        "maximum_static_environment_depth_spread_m": max(depth_spreads),
        "minimum_depth_clearance_m": float(np.min(values)),
        "minimum_per_frame_depth_clearance_m": minimum_frame_clearances,
        "first_percentile_depth_clearance_m": float(np.percentile(values, 1)),
        "fifth_percentile_depth_clearance_m": float(np.percentile(values, 5)),
        "negative_clearance_sample_count": int(np.sum(values < -0.02)),
        "clearance_below_required_sample_count": int(
            np.sum(values < MINIMUM_DEPTH_CLEARANCE_M)
        ),
        "required_minimum_depth_clearance_m": MINIMUM_DEPTH_CLEARANCE_M,
        "status": (
            "pass"
            if out_of_view == 0 and float(np.min(values)) >= MINIMUM_DEPTH_CLEARANCE_M
            else "fail"
        ),
    }


def projection_metrics(
    target: Sequence[Sequence[float]], distractor: Sequence[Sequence[float]]
) -> dict[str, Any]:
    target_centers = [project(point) for point in target]
    distractor_centers = [project(point) for point in distractor]
    target_envelope = [project(point) for point in cylinder_points(target[0])]
    distractor_envelope = [
        project(point) for root in distractor for point in cylinder_points(root)
    ]
    separations = [
        math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))
        for a, b in zip(target, distractor)
    ]
    return {
        "authoritative_camera_habitat_yaw_deg": CAMERA_HABITAT_YAW_DEG,
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
        "target_2m_cylinder_x_fraction_range": [
            min(value[1] for value in target_envelope),
            max(value[1] for value in target_envelope),
        ],
        "target_2m_cylinder_y_fraction_range": [
            min(value[2] for value in target_envelope),
            max(value[2] for value in target_envelope),
        ],
        "distractor_all75_2m_cylinder_x_fraction_range": [
            min(value[1] for value in distractor_envelope),
            max(value[1] for value in distractor_envelope),
        ],
        "distractor_all75_2m_cylinder_y_fraction_range": [
            min(value[2] for value in distractor_envelope),
            max(value[2] for value in distractor_envelope),
        ],
        "minimum_actor_horizontal_separation_m": min(separations),
        "minimum_projected_center_x_separation_fraction": min(
            abs(a[1] - b[1]) for a, b in zip(target_centers, distractor_centers)
        ),
    }


def build_documents(
    depth_path: Path, masks_path: Path, readback_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    distractor = arc_length_resample(NATIVE_HUMAN_ANCHORS, FRAME_COUNT)
    target = [list(STATIC_TARGET_ROOT) for _ in range(FRAME_COUNT)]
    target_motion = motion_metrics(target)
    distractor_motion = motion_metrics(distractor)
    projection = projection_metrics(target, distractor)
    normal_depth, actor_footprints, depth_metadata = load_depth_authority(
        depth_path, masks_path, readback_path
    )
    target_corridor = corridor_metrics(target[:1], normal_depth, actor_footprints)
    distractor_corridor = corridor_metrics(distractor, normal_depth, actor_footprints)
    require(
        target_corridor["status"] == "pass",
        "static target cylinder failed depth corridor",
    )
    require(
        distractor_corridor["status"] == "pass",
        "moving distractor cylinder failed depth corridor",
    )
    tangent_yaw_deg = (
        math.degrees(math.atan2(NATIVE_FORWARD[0], NATIVE_FORWARD[2])) % 360.0
    )
    phases_unwrapped = [
        NATIVE_PHASE_START + NATIVE_PHASE_ADVANCE * index / (FRAME_COUNT - 1)
        for index in range(FRAME_COUNT)
    ]
    phases = [value % 1.0 for value in phases_unwrapped]
    nearest_native_frames = [
        round(
            NATIVE_HUMAN_FRAME_RANGE[0]
            + index
            * (NATIVE_HUMAN_FRAME_RANGE[1] - NATIVE_HUMAN_FRAME_RANGE[0])
            / (FRAME_COUNT - 1)
        )
        for index in range(FRAME_COUNT)
    ]
    face_dx = CAMERA[0] - STATIC_TARGET_ROOT[0]
    face_dz = CAMERA[2] - STATIC_TARGET_ROOT[2]
    face_norm = math.hypot(face_dx, face_dz)
    target_forward = [face_dx / face_norm, 0.0, face_dz / face_norm]
    episode_id = "strict2h_dynamic_canary_02_distractor_moves_v2"
    row: dict[str, Any] = {
        "execution_order": 2,
        "episode_id": episode_id,
        "mechanism": "distractor_moves",
        "target_side": "left",
        "target": {
            "content_id": "cremad_mti_v1",
            "frame_index_map": [NATIVE_DOG_FRAME] * FRAME_COUNT,
            "identity_id": "rocketbox_adults_female_adult_01",
            "identity_key": "F",
            "listening_review": "pending",
            "rights_status": "verified_reusable_with_terms",
            "root_path_m": target,
            "runtime_asset_id": "lead_b_rocketbox_adults_female_adult_01_original_v1",
            "runtime_revision": "native_runtime_ue_v1",
            "sound_asset_id": "speech_cremad_1002_mti_neu_v1",
            "source_actor_id": NATIVE_DOG_ACTOR,
            "source_slot_id": "source1",
            "path_provenance": {
                "method": "exact_native_root_held_all75_v1",
                "native_source_scenario_id": SOURCE_SCENARIO,
                "native_source_actor_id": NATIVE_DOG_ACTOR,
                "native_source_frame_index": NATIVE_DOG_FRAME,
                "original_species": "dog",
                "replacement_identity_species": "human",
                "fresh_human_pixels_required": True,
            },
            "per_frame_anatomical_forward_habitat_world": [target_forward]
            * FRAME_COUNT,
            "facing_policy": "face_camera_all75",
            "speech_frame_window_inclusive": list(SPEECH_WINDOW),
            "speech_sample_count": 45912,
            "transcript": "Maybe tomorrow it will be cold.",
            "voice_id": "cremad_actor_1002",
            "voice_policy": "speaking",
        },
        "distractor": {
            "frame_index_map": nearest_native_frames,
            "identity_id": "rocketbox_adults_male_adult_01",
            "identity_key": "M",
            "root_path_m": distractor,
            "runtime_asset_id": "rocketbox_human_male_adult_01_m5_1_candidate",
            "runtime_revision": "native_runtime_ue_v3",
            "source_actor_id": NATIVE_HUMAN_ACTOR,
            "source_slot_id": "source2",
            "path_provenance": {
                "method": "arc_length_interpolation_of_native_polyline_v1",
                "native_source_scenario_id": SOURCE_SCENARIO,
                "native_source_actor_id": NATIVE_HUMAN_ACTOR,
                "native_source_frame_indices_inclusive": list(NATIVE_HUMAN_FRAME_RANGE),
                "native_anchor_count": len(NATIVE_HUMAN_ANCHORS),
                "output_root_count": FRAME_COUNT,
                "output_unique_root_count_at_1mm": FRAME_COUNT,
                "endpoints_exact_native_readbacks": True,
                "interior_output_roots_exact_native_frame_readbacks": False,
                "maximum_distance_to_native_polyline_m": 0.0,
                "frame_index_map_semantics": "nearest_native_state_for_materializer_fallback_only",
            },
            "per_frame_anatomical_forward_habitat_world": [list(NATIVE_FORWARD)]
            * FRAME_COUNT,
            "per_frame_tangent_yaw_habitat_deg": [tangent_yaw_deg] * FRAME_COUNT,
            "per_frame_action_phase": phases,
            "voice_policy": "silent",
        },
        "native_source_scenario_id": SOURCE_SCENARIO,
        "camera_cluster_id": CAMERA_CLUSTER,
        "camera": {
            "translation_m": list(CAMERA),
            "yaw_path_deg": [CAMERA_HABITAT_YAW_DEG] * FRAME_COUNT,
            "habitat_yaw_deg": CAMERA_HABITAT_YAW_DEG,
            "ue_yaw_deg": CAMERA_UE_YAW_DEG,
            "horizontal_fov_deg": HFOV_DEG,
            "provenance": {
                "scenario_id": SOURCE_SCENARIO,
                "frame_index": 0,
                "coordinate_contract": "UE_yaw_deg=-90-Habitat_yaw_deg",
                "source_suite": SOURCE_SUITE,
            },
        },
        "motion_preflight": {
            "status": "pass",
            "mechanism": "distractor_moves",
            "target": {**target_motion, "expected_moving": False, "status": "pass"},
            "distractor": {
                **distractor_motion,
                "expected_moving": True,
                "status": "pass",
            },
            "camera": {
                "expected_moving": False,
                "total_pan_degrees": 0.0,
                "status": "pass",
            },
        },
        "projection_preflight": projection,
        "depth_corridor_preflight": {
            "authority": depth_metadata,
            "static_target": target_corridor,
            "moving_distractor": distractor_corridor,
        },
        "source_suite": SOURCE_SUITE,
        "suite_plan": "PENDING_DISTRACTOR_MOVES_V2_CPU_MATERIALIZATION",
        "exact_rir_plan": "PENDING_DISTRACTOR_MOVES_V2_EXACT_RIR_PLAN",
        "binaural_audio": "PENDING_DISTRACTOR_MOVES_V2_BINAURAL_RENDER",
        "gpu_launch_authorized": False,
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "formal": False,
        "qualification_claim": False,
        "status": "pass_cpu_geometry_pending_materialization_acoustics_fresh_native_pixels",
    }
    preflight = {
        "schema": "avengine_native_strict_two_human_dynamic_full75_canary_preflight_v1",
        "status": "pass_cpu_geometry_pending_materialization_acoustics_fresh_native_pixels",
        "dynamic_canary_count": 1,
        "dynamic_canary_gpu_pass_count": 0,
        "single_room_mechanism_pilot_authorized": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "unique_source_scenario_count": 1,
        "unique_camera_cluster_count": 1,
        "target_side_counts": {"left": 1},
        "canaries": [row],
    }
    receipt = {
        "schema": "avengine_native_strict_two_human_distractor_moves_v2_cpu_geometry_receipt_v1",
        "status": "go_cpu_materialization_only_fresh_native_pixels_still_required",
        "candidate_decision": "GO_TO_CPU_MATERIALIZATION_ONLY",
        "claim_boundary": (
            "Exact native roots, deterministic arc-length interpolation, pinhole projection, and same-camera static-room "
            "metric-depth cylinder screening pass. They do not prove fresh F/M modal visibility, collision physics, foot-ground "
            "contact, or the strict pixel gates."
        ),
        "candidate": row,
        "corrected_camera_contract": {
            "habitat_yaw_deg": CAMERA_HABITAT_YAW_DEG,
            "ue_yaw_deg": CAMERA_UE_YAW_DEG,
            "relationship": "UE_yaw_deg=-90-Habitat_yaw_deg",
            "rejected_prior_scan_habitat_yaw_deg": 35.0,
            "rejected_prior_candidate": "border_collie_human__recombined_both_moving_0099",
            "rejection_reason": "correct yaw puts static target at x=0.502 and moving human near x=0.887",
        },
        "native_motion_authority": {
            "scenario_id": SOURCE_SCENARIO,
            "actor_id": NATIVE_HUMAN_ACTOR,
            "native_frame_indices_inclusive": list(NATIVE_HUMAN_FRAME_RANGE),
            "native_anchor_count": len(NATIVE_HUMAN_ANCHORS),
            "output_root_count": FRAME_COUNT,
            "path_length_m": distractor_motion["horizontal_path_length_m"],
            "average_root_speed_m_s": distractor_motion["horizontal_path_length_m"]
            / ((FRAME_COUNT - 1) / FRAME_RATE_HZ),
            "tangent_yaw_habitat_deg": tangent_yaw_deg,
            "native_phase_start": NATIVE_PHASE_START,
            "native_phase_advance_cycles": NATIVE_PHASE_ADVANCE,
            "output_phase_start": phases[0],
            "output_phase_end": phases[-1],
            "output_unwrapped_phase_end": phases_unwrapped[-1],
            "animation_timing_policy": "phase_advance_proportional_to_accumulated_root_arc_length",
        },
        "projection_preflight": projection,
        "depth_corridor_preflight": row["depth_corridor_preflight"],
        "retained_native_visual_evidence": {
            "video": f"{SOURCE_SUITE_ROOT}/{SOURCE_SCENARIO}/ue_visual_only.mp4",
            "reviewed_frame_indices": [2, 6, 10, 17],
            "review_result": "native human full body visible and continuous; dog root neighborhood clear in reviewed frames",
            "use_boundary": "native human/dog pixels only, not fresh replacement F/M pixel truth",
        },
        "strict_native_acceptance_gate": {
            "target_speech_window_inclusive": list(SPEECH_WINDOW),
            "target_minimum_visible_pixels_during_speech": 10000,
            "target_minimum_visible_fraction_during_speech": 0.8,
            "distractor_minimum_visible_pixels_all75": 5000,
            "distractor_minimum_visible_fraction_all75": 0.5,
            "status": "pending_fresh_native_target_only_capture",
        },
        "camera_cluster_scope": {
            "camera_cluster_id": CAMERA_CLUSTER,
            "mechanism_canary_only": True,
            "independent_episode_claim": False,
            "reason": "camera cluster reuses legacy common-camera evidence and is not counted toward 100 unique clusters",
        },
        "single_attempt_policy": {
            "maximum_gpu_attempts_for_this_candidate": 1,
            "gpu_attempts_used": 0,
            "on_any_native_gate_failure": "freeze rejected receipt and advance to a new native-human scenario; do not tune against pixels",
        },
        "acoustic_state_expectation": {
            "source_frame_uses": 150,
            "target_unique_rir_states": 1,
            "distractor_unique_rir_states": 75,
            "total_unique_rir_states": 76,
            "exact_rir_required_before_gpu": True,
        },
        "next_authorized_step": "CPU materializer generalization for distractor_moves only",
        "gpu_launch_authorized": False,
        "formal_episode_count": 0,
    }
    return preflight, receipt


def screen_xy(point: Sequence[float], height_m: float = 0.0) -> tuple[int, int]:
    _, x_fraction, y_fraction = project(point, height_m)
    return (
        round(x_fraction * (RESOLUTION_HW[1] - 1)),
        round(y_fraction * (RESOLUTION_HW[0] - 1)),
    )


def build_overlay(background_path: Path, output: Path, receipt: dict[str, Any]) -> None:
    require(background_path.is_file(), "native f6 visual evidence missing")
    background = Image.open(background_path).convert("RGB")
    require(background.size == (1280, 720), "native visual size drift")
    draw = ImageDraw.Draw(background, "RGBA")
    font = ImageFont.load_default()
    target_root = screen_xy(STATIC_TARGET_ROOT, 0.0)
    target_head = screen_xy(STATIC_TARGET_ROOT, BODY_HEIGHT_M)
    target_depth = project(STATIC_TARGET_ROOT)[0]
    fx = (RESOLUTION_HW[1] / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    target_half_width = round(fx * BODY_RADIUS_M / target_depth)
    draw.rectangle(
        (
            target_root[0] - target_half_width,
            target_head[1],
            target_root[0] + target_half_width,
            target_root[1],
        ),
        outline=(255, 60, 60, 255),
        width=4,
    )
    distractor = receipt["candidate"]["distractor"]["root_path_m"]
    path_pixels = [screen_xy(point, 1.0) for point in distractor]
    draw.line(path_pixels, fill=(60, 160, 255, 255), width=5)
    for frame_index in (0, 37, 74):
        root = distractor[frame_index]
        root_pixel = screen_xy(root, 0.0)
        head_pixel = screen_xy(root, BODY_HEIGHT_M)
        depth = project(root)[0]
        half_width = round(fx * BODY_RADIUS_M / depth)
        draw.rectangle(
            (
                root_pixel[0] - half_width,
                head_pixel[1],
                root_pixel[0] + half_width,
                root_pixel[1],
            ),
            outline=(60, 170, 255, 255),
            width=3,
        )
        draw.text(
            (root_pixel[0] + 5, head_pixel[1]),
            f"out f{frame_index}",
            fill=(150, 220, 255),
            font=font,
        )
    draw.rectangle((6, 6, 610, 58), fill=(0, 0, 0, 180))
    draw.text(
        (14, 13),
        "CPU overlay on retained native 0589 f6 - NOT fresh F/M pixels",
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (14, 33),
        "red: static speaking F 2m proxy; blue: moving silent M path/proxies",
        fill=(255, 255, 255),
        font=font,
    )

    panel_width = 640
    top = Image.new("RGB", (panel_width, 720), (245, 246, 248))
    td = ImageDraw.Draw(top, "RGBA")
    x_min, x_max = -3.8, -0.4
    z_min, z_max = -2.7, 1.0

    def top_xy(point: Sequence[float]) -> tuple[int, int]:
        x = 40 + (float(point[0]) - x_min) / (x_max - x_min) * (panel_width - 80)
        y = 680 - (float(point[2]) - z_min) / (z_max - z_min) * 640
        return round(x), round(y)

    anchors = [top_xy(point) for point in NATIVE_HUMAN_ANCHORS]
    output_path = [top_xy(point) for point in distractor]
    td.line(anchors, fill=(20, 100, 180, 255), width=7)
    td.line(output_path, fill=(80, 180, 255, 190), width=3)
    target = top_xy(STATIC_TARGET_ROOT)
    camera = top_xy(CAMERA)
    td.ellipse(
        (target[0] - 10, target[1] - 10, target[0] + 10, target[1] + 10),
        fill=(240, 60, 60, 255),
    )
    td.polygon(
        [
            (camera[0], camera[1] - 12),
            (camera[0] - 10, camera[1] + 10),
            (camera[0] + 10, camera[1] + 10),
        ],
        fill=(25, 25, 25, 255),
    )
    td.text(
        (18, 14),
        "Top-down exact native f2-f17 -> equal-arc full75",
        fill=(20, 20, 20),
        font=font,
    )
    td.text(
        (18, 34),
        "depth cylinders: target min +0.489m; distractor min +0.342m",
        fill=(20, 110, 50),
        font=font,
    )
    td.text(
        (target[0] + 12, target[1] - 8),
        "static target (native dog root f6)",
        fill=(150, 20, 20),
        font=font,
    )
    td.text(
        (camera[0] + 12, camera[1] - 8),
        "camera yaw 55 deg",
        fill=(20, 20, 20),
        font=font,
    )

    combined = Image.new("RGB", (1920, 720), (255, 255, 255))
    combined.paste(background, (0, 0))
    combined.paste(top, (1280, 0))
    combined.save(output / "projection_depth_topdown_overlay.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--metric-depth", type=Path)
    parser.add_argument("--actor-masks", type=Path)
    parser.add_argument("--camera-readbacks", type=Path)
    parser.add_argument("--native-frame", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence_dir = args.evidence_dir.resolve()
    depth_path = (
        args.metric_depth.resolve()
        if args.metric_depth
        else evidence_dir / "metric_depth_static_camera_0406_full75.npz"
    )
    masks_path = (
        args.actor_masks.resolve()
        if args.actor_masks
        else evidence_dir / "native_pixel_masks_static_camera_0406_full75.npz"
    )
    readback_path = (
        args.camera_readbacks.resolve()
        if args.camera_readbacks
        else evidence_dir / "depth_static_camera_0406_runtime_readbacks.json"
    )
    native_frame = (
        args.native_frame.resolve()
        if args.native_frame
        else evidence_dir / "native0589_02.png"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    preflight, receipt = build_documents(depth_path, masks_path, readback_path)
    write_json(args.output / "distractor_moves_v2_preflight.json", preflight)
    write_json(args.output / "cpu_geometry_receipt.json", receipt)
    build_overlay(native_frame, args.output.resolve(), receipt)
    print(
        "DISTRACTOR_MOVES_V2_CPU_CANDIDATE_OK "
        f"path_m={receipt['native_motion_authority']['path_length_m']:.9f} "
        f"target_clearance_m={receipt['depth_corridor_preflight']['static_target']['minimum_depth_clearance_m']:.9f} "
        f"distractor_clearance_m={receipt['depth_corridor_preflight']['moving_distractor']['minimum_depth_clearance_m']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
