#!/usr/bin/env python3
"""Bind one exact SPEAR Apartment RGB render to Habitat Topdown v3 and audio.

The SPEAR scenario spec remains the trajectory/audio identity.  This bridge
converts its historical Apartment SSOT coordinates into Habitat world
coordinates, validates the source tags against UE runtime readback, checks
every source center against the retained Apartment NavMesh, and emits one
five-second side-by-side listening video.  It never substitutes Habitat RGB.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import wave
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.capture.review import (
    SourceOverlayTrack,
    compose_annotated_frames,
    encode_annotated_review,
)
from avengine.m7.sensor_rig import (
    M7SensorRigError,
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
)
from avengine.optional_backends.spear_apartment import (
    POSITION_TOLERANCE_CM,
    ROTATION_TOLERANCE_DEGREES,
    wrap_angle_difference_degrees,
)
from avengine.optional_backends.spear_visual import (
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)
from avengine.m6x.geometry import (
    RuntimeObstacleMap,
    evaluate_source_center_gate,
)
from avengine.m6x.raster_pathfinder import RasterPathfinder
from avengine.m6x.topdown import TOPDOWN_SCHEMA, render_runtime_topdown_frames


SCHEMA = "avengine_spear_apartment_habitat_topdown_review_v1"
SSOT_TO_HABITAT = np.asarray(
    (
        (1.0, 0.0, 0.0, -1.2),
        (0.0, 0.0, 1.0, 0.271),
        (0.0, -1.0, 0.0, 0.8),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


class SpearApartmentReviewError(RuntimeError):
    """The requested RGB, trajectory, Topdown and audio binding is invalid."""


def ssot_points_to_habitat(points_m: Any) -> np.ndarray:
    """Convert legacy Apartment SSOT XYZ points to Habitat world XYZ."""

    try:
        points = np.asarray(points_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpearApartmentReviewError(
            "SSOT points must be a finite [point,3] array"
        ) from exc
    if (
        points.ndim != 2
        or points.shape[0] < 1
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise SpearApartmentReviewError(
            "SSOT points must be a finite [point,3] array"
        )
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    return np.ascontiguousarray((SSOT_TO_HABITAT @ homogeneous.T).T[:, :3])


def source_center_paths_from_spec(
    spec: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return generic source1/source2 emitter-center paths in Habitat world."""

    sources = spec.get("sources")
    if not isinstance(sources, list) or len(sources) < 1:
        raise SpearApartmentReviewError("scenario spec has no sources")
    paths: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            raise SpearApartmentReviewError("scenario source must be an object")
        try:
            roots = np.asarray(source["trajectory_m"], dtype=np.float64)
            height = float(source["audio_source_height_offset_m"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SpearApartmentReviewError(
                f"source{index} lacks a valid trajectory/emitter height"
            ) from exc
        if (
            roots.ndim != 2
            or roots.shape[0] < 1
            or roots.shape[1] != 3
            or not np.all(np.isfinite(roots))
            or not np.isfinite(height)
            or height < 0.0
        ):
            raise SpearApartmentReviewError(
                f"source{index} lacks a valid trajectory/emitter height"
            )
        if frame_count is None:
            frame_count = int(roots.shape[0])
        elif roots.shape[0] != frame_count:
            raise SpearApartmentReviewError("source trajectory frame counts differ")
        centers_ssot = roots.copy()
        centers_ssot[:, 2] += height
        paths[f"source{index}"] = ssot_points_to_habitat(centers_ssot)
    return paths


def _motion_headings(paths: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    headings: dict[str, np.ndarray] = {}
    for source_id, points in paths.items():
        horizontal = np.asarray(points[:, (0, 2)], dtype=np.float64)
        delta = np.empty_like(horizontal)
        if len(horizontal) == 1:
            raise SpearApartmentReviewError(
                f"{source_id} needs an authored heading when stationary"
            )
        delta[0] = horizontal[1] - horizontal[0]
        delta[-1] = horizontal[-1] - horizontal[-2]
        if len(horizontal) > 2:
            delta[1:-1] = horizontal[2:] - horizontal[:-2]
        norms = np.linalg.norm(delta, axis=1)
        if np.any(norms <= 1.0e-9):
            raise SpearApartmentReviewError(
                f"{source_id} contains a zero-motion heading"
            )
        headings[source_id] = np.ascontiguousarray(delta / norms[:, None])
    return headings


def _obstacle_map(root: Path) -> tuple[RuntimeObstacleMap, RasterPathfinder]:
    record = load_json(root / "feasible_region.json")["obstacle_authority"]
    navmesh = np.load(
        root / "feasible_region_source1.npz", allow_pickle=False
    )["navmesh_mask"]
    pathfinder = RasterPathfinder(
        navmesh,
        bounds_m=record["bounds_m"],
        floor_height_m=float(record["floor_height_m"]),
    )
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.ascontiguousarray(navmesh),
        bounds_m=tuple(
            tuple(float(value) for value in row) for row in record["bounds_m"]
        ),
        floor_height_m=float(record["floor_height_m"]),
        meters_per_pixel=float(record["meters_per_pixel"]),
        rigid_obstacles=tuple(record.get("rigid_obstacles", ())),
        authority=str(record["authority"]),
        claim_boundary=str(record["claim_boundary"]),
        rigid_obstacles_baked_into_navmesh=bool(
            record.get("rigid_obstacles_baked_into_navmesh", False)
        ),
        _pathfinder=pathfinder,
    )
    return obstacle_map, pathfinder


def _decode_video_rgb(
    path: Path, *, width: int, height: int, frame_count: int
) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=120.0,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0 or stderr:
        raise SpearApartmentReviewError("UE RGB decode failed: " + stderr)
    expected = frame_count * height * width * 3
    if len(completed.stdout) != expected:
        raise SpearApartmentReviewError(
            "UE RGB frame count or dimensions differ from the scenario spec"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        frame_count, height, width, 3
    )


def _audio_contract(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_count = handle.getnframes()
            sample_width = handle.getsampwidth()
    except (OSError, wave.Error) as exc:
        raise SpearApartmentReviewError(f"cannot read binaural WAV: {exc}") from exc
    expected = spec.get("audio_config")
    if not isinstance(expected, Mapping):
        raise SpearApartmentReviewError("scenario lacks audio_config")
    if (
        channels != int(expected["output_channels"])
        or sample_rate != int(expected["sample_rate_hz"])
        or sample_count != int(expected["n_samples"])
    ):
        raise SpearApartmentReviewError(
            "binaural WAV channels/rate/sample count differ from the scenario spec"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_count": sample_count,
        "sample_width_bytes": sample_width,
        "duration_seconds": sample_count / sample_rate,
    }


def _validate_visual_binding(
    spec: Mapping[str, Any], metadata_path: Path
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    sources = spec.get("sources")
    runtime_sources = metadata.get("sources")
    if not isinstance(sources, list) or not isinstance(runtime_sources, list):
        raise SpearApartmentReviewError("spec/runtime source lists are missing")
    expected_tags = [source.get("tag") for source in sources]
    runtime_tags = [source.get("tag") for source in runtime_sources]
    if runtime_tags != expected_tags:
        raise SpearApartmentReviewError(
            f"UE runtime source tags differ: expected {expected_tags}, got {runtime_tags}"
        )
    warmup = metadata.get("capture_warmup")
    if not isinstance(warmup, Mapping) or warmup.get("status") != "passed":
        raise SpearApartmentReviewError("UE capture warmup did not pass")
    direction = metadata.get("rig_direction_evidence")
    if not isinstance(direction, Mapping) or any(
        not isinstance(direction.get(tag), Mapping)
        or direction[tag].get("status") != "passed"
        for tag in expected_tags
    ):
        raise SpearApartmentReviewError("UE source direction gate did not pass")
    return {
        "source_tags": expected_tags,
        "capture_warmup": dict(warmup),
        "direction_status_by_tag": {
            tag: direction[tag]["status"] for tag in expected_tags
        },
    }


def _validate_camera_readback_series(
    records: Any,
    *,
    owner: str,
    expected_pose_hashes: tuple[str, ...],
    expected_positions_cm: np.ndarray,
    expected_yaws_deg: np.ndarray,
) -> dict[str, Any]:
    """Validate one complete series of actual UE camera-root readbacks."""

    expected_frame_count = len(expected_pose_hashes)
    if not isinstance(records, list) or len(records) != expected_frame_count:
        raise SpearApartmentReviewError(
            f"{owner} must contain exactly {expected_frame_count} camera readbacks"
        )
    position_errors: list[float] = []
    roll_errors: list[float] = []
    pitch_errors: list[float] = []
    yaw_errors: list[float] = []
    checked_pose_hash_count = 0
    for frame_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SpearApartmentReviewError(
                f"{owner}[{frame_index}] must be an object"
            )
        if record.get("frame_index") != frame_index:
            raise SpearApartmentReviewError(
                f"{owner} frame order changed at frame {frame_index}"
            )
        try:
            location = np.asarray(record["location_cm"], dtype=np.float64)
            rotation = np.asarray(record["rotation_deg"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SpearApartmentReviewError(
                f"{owner} lacks actual camera transform at frame {frame_index}"
            ) from exc
        if (
            location.shape != (3,)
            or rotation.shape != (3,)
            or not np.all(np.isfinite(location))
            or not np.all(np.isfinite(rotation))
        ):
            raise SpearApartmentReviewError(
                f"{owner} lacks actual camera transform at frame {frame_index}"
            )
        position_error = float(
            np.max(np.abs(location - expected_positions_cm[frame_index]))
        )
        yaw_error = abs(
            wrap_angle_difference_degrees(
                float(rotation[2]),
                float(expected_yaws_deg[frame_index]),
            )
        )
        roll_error = abs(
            wrap_angle_difference_degrees(float(rotation[0]), 0.0)
        )
        pitch_error = abs(
            wrap_angle_difference_degrees(float(rotation[1]), 0.0)
        )
        position_errors.append(position_error)
        roll_errors.append(roll_error)
        pitch_errors.append(pitch_error)
        yaw_errors.append(yaw_error)
        if (
            position_error > POSITION_TOLERANCE_CM
            or roll_error > ROTATION_TOLERANCE_DEGREES
            or pitch_error > ROTATION_TOLERANCE_DEGREES
            or yaw_error > ROTATION_TOLERANCE_DEGREES
        ):
            raise SpearApartmentReviewError(
                "UE actual camera readback differs from "
                f"SensorRigTrajectory at frame {frame_index}"
            )
        pose_hash = record.get("expected_pose_hash")
        if pose_hash is not None:
            if pose_hash != expected_pose_hashes[frame_index]:
                raise SpearApartmentReviewError(
                    "UE expected pose hash differs from "
                    f"SensorRigTrajectory at frame {frame_index}"
                )
            checked_pose_hash_count += 1
    if checked_pose_hash_count not in (0, expected_frame_count):
        raise SpearApartmentReviewError(
            f"{owner} only partially binds SensorRigTrajectory pose hashes"
        )
    return {
        "frame_count": expected_frame_count,
        "maximum_position_error_cm": max(position_errors),
        "maximum_roll_error_deg": max(roll_errors),
        "maximum_pitch_error_deg": max(pitch_errors),
        "maximum_yaw_error_deg": max(yaw_errors),
        "checked_pose_hash_count": checked_pose_hash_count,
    }


def _validate_ue_camera_readback_binding(
    metadata_path: Path,
    *,
    expected_pose_hashes: tuple[str, ...],
    listener_positions_m: np.ndarray,
    listener_yaws_deg: np.ndarray,
) -> dict[str, Any]:
    """Cross-check actual UE camera readbacks against the formal rig."""

    metadata = load_json(metadata_path)
    root_readback = metadata.get("root_readback")
    root_camera = (
        root_readback.get("camera")
        if isinstance(root_readback, Mapping)
        else None
    )
    if (
        not isinstance(root_camera, Mapping)
        or root_camera.get("status") != "pass"
    ):
        raise SpearApartmentReviewError(
            "UE root_readback.camera status did not pass"
        )
    documents: list[tuple[str, Mapping[str, Any], Path]] = [
        ("visual_metadata", metadata, metadata_path)
    ]
    referenced_readbacks = metadata.get("runtime_readbacks")
    readback_paths: list[Path] = []
    if isinstance(referenced_readbacks, str) and referenced_readbacks:
        readback_paths.append(
            (metadata_path.parent / referenced_readbacks).resolve()
        )
    sibling_readbacks = metadata_path.parent / "runtime_readbacks.json"
    if sibling_readbacks.is_file():
        readback_paths.append(sibling_readbacks.resolve())
    seen_paths = {metadata_path.resolve()}
    for path in readback_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.is_file():
            raise SpearApartmentReviewError(
                f"declared UE runtime readback is missing: {path}"
            )
        readback = load_json(path)
        if not isinstance(readback, Mapping):
            raise SpearApartmentReviewError(
                f"UE runtime readback must be an object: {path}"
            )
        documents.append(("runtime_readbacks", readback, path))

    expected_positions_cm = np.asarray(
        [
            habitat_point_to_apartment_ue_cm(position.tolist())
            for position in listener_positions_m
        ],
        dtype=np.float64,
    )
    expected_yaws_deg = np.asarray(
        [camera_ue_yaw_degrees(yaw) for yaw in listener_yaws_deg],
        dtype=np.float64,
    )
    checked: list[dict[str, Any]] = []
    for document_id, document, path in documents:
        candidates: list[tuple[str, Any]] = [
            ("camera_root", document.get("camera_root")),
            ("camera_readbacks", document.get("camera_readbacks")),
        ]
        embedded = document.get("runtime_readbacks")
        if isinstance(embedded, Mapping):
            candidates.extend(
                (
                    (
                        "runtime_readbacks.camera_root",
                        embedded.get("camera_root"),
                    ),
                    (
                        "runtime_readbacks.camera_readbacks",
                        embedded.get("camera_readbacks"),
                    ),
                )
            )
        for candidate_id, records in candidates:
            if records is None:
                continue
            gate = _validate_camera_readback_series(
                records,
                owner=f"{document_id}.{candidate_id}",
                expected_pose_hashes=expected_pose_hashes,
                expected_positions_cm=expected_positions_cm,
                expected_yaws_deg=expected_yaws_deg,
            )
            checked.append(
                {
                    "document": document_id,
                    "field": candidate_id,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    **gate,
                }
            )

    if not checked:
        raise SpearApartmentReviewError(
            "UE per-frame actual camera readback series is missing"
        )
    return {
        "status": "pass",
        "root_readback_camera_status": "pass",
        "checked_actual_readback_series_count": len(checked),
        "checked_frame_count": len(expected_pose_hashes),
        "maximum_position_error_cm": max(
            record["maximum_position_error_cm"] for record in checked
        ),
        "maximum_roll_error_deg": max(
            record["maximum_roll_error_deg"] for record in checked
        ),
        "maximum_pitch_error_deg": max(
            record["maximum_pitch_error_deg"] for record in checked
        ),
        "maximum_yaw_error_deg": max(
            record["maximum_yaw_error_deg"] for record in checked
        ),
        "position_tolerance_cm": POSITION_TOLERANCE_CM,
        "yaw_tolerance_deg": ROTATION_TOLERANCE_DEGREES,
        "sources": checked,
    }


def _display_label(source: Mapping[str, Any], source_id: str) -> str:
    asset_id = source.get("asset_id")
    if isinstance(asset_id, str) and asset_id:
        return asset_id.replace("_", " ").title()
    return source_id


def build_review(
    *,
    spec_path: Path,
    ue_video_path: Path,
    visual_metadata_path: Path,
    audio_path: Path,
    feasibility_root: Path,
    output_path: Path,
    sensor_rig_trajectory_path: Path | None = None,
) -> Path:
    """Build one no-overwrite UE + Topdown v3 + binaural review."""

    paths = (
        spec_path,
        ue_video_path,
        visual_metadata_path,
        audio_path,
        feasibility_root / "feasible_region.json",
        feasibility_root / "feasible_region_source1.npz",
        *(
            (sensor_rig_trajectory_path,)
            if sensor_rig_trajectory_path is not None
            else ()
        ),
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SpearApartmentReviewError(f"required inputs are missing: {missing}")
    evidence_path = output_path.with_suffix(".evidence.json")
    if output_path.exists() or evidence_path.exists():
        raise SpearApartmentReviewError(
            f"refusing to replace review output: {output_path}"
        )

    spec = load_json(spec_path)
    render = spec.get("render_config")
    camera_configs = spec.get("camera_configs")
    mic = spec.get("mic")
    if (
        not isinstance(render, Mapping)
        or not isinstance(camera_configs, list)
        or len(camera_configs) != 1
        or not isinstance(camera_configs[0], Mapping)
        or not isinstance(mic, Mapping)
    ):
        raise SpearApartmentReviewError("scenario render/camera/listener is invalid")
    frame_count = int(render["n_frames"])
    fps = int(render["fps"])
    width = int(render["width"])
    height = int(render["height"])
    if frame_count != 75 or fps != 15:
        raise SpearApartmentReviewError("Apartment review must use 75 frames at 15 fps")

    visual_binding = _validate_visual_binding(spec, visual_metadata_path)
    audio = _audio_contract(audio_path, spec)
    center_paths = source_center_paths_from_spec(spec)
    if any(len(points) != frame_count for points in center_paths.values()):
        raise SpearApartmentReviewError("source paths differ from the video clock")
    headings = _motion_headings(center_paths)
    listener = ssot_points_to_habitat([mic["pos_m"]])[0]
    listener_yaw = (float(mic["yaw_deg"]) - 90.0) % 360.0
    listener_positions_m_by_frame = None
    listener_yaws_deg_by_frame = None
    sensor_rig_binding = None
    ue_camera_readback_binding = None
    if sensor_rig_trajectory_path is not None:
        try:
            sensor_rig_trajectory = load_json(sensor_rig_trajectory_path)
            rig_poses = m7_sensor_rig_pose_series(sensor_rig_trajectory)
            sensor_rig_binding = {
                **m7_sensor_rig_binding(sensor_rig_trajectory),
                "path": str(sensor_rig_trajectory_path.resolve()),
                "file_sha256": sha256_file(sensor_rig_trajectory_path),
            }
        except (M7SensorRigError, OSError, ValueError) as exc:
            raise SpearApartmentReviewError(
                f"SensorRigTrajectory validation failed: {exc}"
            ) from exc
        if (
            len(rig_poses.pose_hashes) != frame_count
            or sensor_rig_trajectory.get("frame_count") != frame_count
            or sensor_rig_trajectory.get("frame_rate_hz") != fps
        ):
            raise SpearApartmentReviewError(
                "SensorRigTrajectory clock differs from the review video"
            )
        listener = rig_poses.positions_m[0]
        listener_yaw = float(rig_poses.yaws_deg[0])
        listener_positions_m_by_frame = rig_poses.positions_m
        listener_yaws_deg_by_frame = rig_poses.yaws_deg
        ue_camera_readback_binding = _validate_ue_camera_readback_binding(
            visual_metadata_path,
            expected_pose_hashes=rig_poses.pose_hashes,
            listener_positions_m=rig_poses.positions_m,
            listener_yaws_deg=rig_poses.yaws_deg,
        )
    hfov = float(camera_configs[0]["fov_deg"])

    obstacle_map, pathfinder = _obstacle_map(feasibility_root)
    center_gate = evaluate_source_center_gate(
        pathfinder,
        obstacle_map,
        center_paths,
        maximum_floor_snap_xz_m=0.02,
        minimum_navmesh_clearance_m=0.0,
        minimum_rigid_clearance_m=0.0,
    )
    if center_gate["status"] != "pass":
        raise SpearApartmentReviewError(
            "one or more transformed source centers are outside the Apartment NavMesh"
        )
    activity = {
        source_id: np.ones(frame_count, dtype=np.bool_)
        for source_id in center_paths
    }
    sources = spec["sources"]
    labels = {
        source_id: _display_label(sources[index], source_id)
        for index, source_id in enumerate(center_paths)
    }
    colors = {
        source_id: color
        for source_id, color in zip(
            center_paths,
            ((42, 210, 220), (250, 120, 70), (167, 121, 255)),
            strict=False,
        )
    }
    if set(colors) != set(center_paths):
        raise SpearApartmentReviewError("review supports at most three sources")

    topdown = render_runtime_topdown_frames(
        obstacle_map,
        center_paths,
        listener_position_m=listener,
        listener_yaw_deg=listener_yaw,
        camera_hfov_degrees=hfov,
        listener_positions_m_by_frame=listener_positions_m_by_frame,
        listener_yaws_deg_by_frame=listener_yaws_deg_by_frame,
        source_activity_by_frame=activity,
        source_heading_xz_by_frame=headings,
        source_labels=labels,
        source_colors=colors,
        size_wh=(640, 480),
    )
    rgb = _decode_video_rgb(
        ue_video_path,
        width=width,
        height=height,
        frame_count=frame_count,
    )
    tracks = tuple(
        SourceOverlayTrack(
            source_id=source_id,
            label=labels[source_id],
            asset_class=str(sources[index].get("asset_class", "unknown")),
            sound_class=str(sources[index].get("audio_lookup", "unknown")),
            color_rgb=colors[source_id],
            positions_m=center_paths[source_id],
            current_event_by_frame=("simultaneous_source_audio",) * frame_count,
            active_by_frame=(True,) * frame_count,
            true_flags=("simultaneous", str(sources[index].get("kind", "unknown"))),
            center_clearance_m=np.asarray(
                [
                    frame["navmesh_clearance_m"]
                    for frame in center_gate["sources"][source_id]["frames"]
                ],
                dtype=np.float64,
            ),
        )
        for index, source_id in enumerate(center_paths)
    )
    frames = compose_annotated_frames(
        main_rgb=rgb,
        topdown_rgb=topdown,
        tracks=tracks,
        clip_id=str(spec.get("scenario_id", spec_path.stem)),
        room_id=str(spec.get("room_backend", "spear_apartment_0000")),
        review_stage_label="SPEAR/UE RGB + Habitat Topdown v3",
        listener_position_m=listener,
        listener_yaw_deg=listener_yaw,
        listener_positions_m_by_frame=listener_positions_m_by_frame,
        listener_yaws_deg_by_frame=listener_yaws_deg_by_frame,
        aggregate_true_flags=(
            "all_sources_active",
            "ue_rgb",
            "topdown_v3",
            "exact_binaural_bound",
        ),
        audio_diagnostic_by_frame=(
            "exact source WAV; 2ch native-HRTF binaural; audio is 360 deg",
        )
        * frame_count,
        center_gate_pass=True,
        fps=fps,
    )
    media = encode_annotated_review(
        frames,
        output_path,
        fps=fps,
        audio_path=audio_path,
    )
    concise_gate = {
        source_id: {
            "status": record["status"],
            "minimum_navmesh_clearance_m": record[
                "minimum_navmesh_clearance_m"
            ],
            "failed_frame_indices": record["failed_frame_indices"],
        }
        for source_id, record in center_gate["sources"].items()
    }
    write_json(
        evidence_path,
        {
            "schema": SCHEMA,
            "status": "pass",
            "scenario_id": spec.get("scenario_id"),
            "spec": {
                "path": str(spec_path.resolve()),
                "sha256": sha256_file(spec_path),
            },
            "ue_rgb": {
                "path": str(ue_video_path.resolve()),
                "sha256": sha256_file(ue_video_path),
                **visual_binding,
            },
            "coordinate_transform": {
                "equation": "[x,y,z] -> [-1.2+x,0.271+z,0.8-y]",
                "listener_position_m": listener.tolist(),
                "listener_yaw_deg": listener_yaw,
                "listener_pose_mode": (
                    "sensor_rig_trajectory_v1"
                    if sensor_rig_binding is not None
                    else "legacy_fixed_mic"
                ),
                "camera_hfov_degrees": hfov,
            },
            "topdown": {
                "schema": TOPDOWN_SCHEMA,
                "obstacle_map": obstacle_map.summary(),
                "source_center_gate": concise_gate,
                "source_ids": list(center_paths),
                "source_tags": visual_binding["source_tags"],
            },
            "audio": audio,
            "media": dict(media),
            **(
                {
                    "sensor_rig_trajectory": sensor_rig_binding,
                    "ue_camera_readback_binding": (
                        ue_camera_readback_binding
                    ),
                }
                if sensor_rig_binding is not None
                else {}
            ),
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--ue-video", type=Path, required=True)
    parser.add_argument("--visual-metadata", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-rig-trajectory", type=Path)
    args = parser.parse_args()
    output = build_review(
        spec_path=args.spec.resolve(),
        ue_video_path=args.ue_video.resolve(),
        visual_metadata_path=args.visual_metadata.resolve(),
        audio_path=args.audio.resolve(),
        feasibility_root=args.feasibility_root.resolve(),
        output_path=args.output.resolve(),
        sensor_rig_trajectory_path=(
            args.sensor_rig_trajectory.resolve()
            if args.sensor_rig_trajectory is not None
            else None
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
