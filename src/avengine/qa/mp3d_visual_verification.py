#!/usr/bin/env python3
"""Verify an observed native MP3D visual capture batch.

The verifier is intentionally independent of Habitat and GPU execution.  It
reads the native capture receipt, observed frame records and RGB/depth/semantic
arrays.  A planned timeline may be supplied as an expected pose source, but a
planned route centre is never accepted as an observed emitter position.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


NATIVE_CAPTURE_SCHEMA = "avengine_mp3d_multi_actor_native_capture_v1"
VERIFICATION_SCHEMA = "qa_v3_visual_batch_verification_v1"
ROOT_POSITION_ATOL_M = 2.0e-6
CAMERA_POSITION_ATOL_M = 2.0e-6
EMITTER_POSITION_ATOL_M = 2.0e-6
ROTATION_MATRIX_ATOL = 2.0e-6


class MP3DVisualVerificationError(ValueError):
    """An observed MP3D capture is incomplete or internally inconsistent."""


def _read_json(path: Path, *, owner: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MP3DVisualVerificationError(f"{owner} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MP3DVisualVerificationError(f"cannot read {owner}: {error}") from error
    if not isinstance(value, Mapping):
        raise MP3DVisualVerificationError(f"{owner} must be a JSON object")
    return dict(value)


def _finite_vector(value: Any, *, owner: str, length: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise MP3DVisualVerificationError(f"{owner} must be a finite vector")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise MP3DVisualVerificationError(f"{owner} must be a finite vector") from error
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise MP3DVisualVerificationError(f"{owner} must be a finite vector")
    return np.ascontiguousarray(result)


def _finite_matrix(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise MP3DVisualVerificationError(f"{owner} must be a finite 4x4 matrix") from error
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise MP3DVisualVerificationError(f"{owner} must be a finite 4x4 matrix")
    return np.ascontiguousarray(result)


def _transform_matrix(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, Mapping) and "translation_m" in value:
        translation = _finite_vector(
            value.get("translation_m"), owner=f"{owner}.translation_m", length=3
        )
        quaternion = _finite_vector(
            value.get("rotation_xyzw"), owner=f"{owner}.rotation_xyzw", length=4
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-15:
            raise MP3DVisualVerificationError(
                f"{owner}.rotation_xyzw must be nonzero"
            )
        x, y, z, w = quaternion / norm
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), translation[0]],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), translation[1]],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return _finite_matrix(value, owner=owner)


def _artifact_path(point_dir: Path, receipt: Mapping[str, Any], name: str) -> Path:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MP3DVisualVerificationError("capture receipt has no artifacts object")
    raw = artifacts.get(name)
    if not isinstance(raw, str) or not raw:
        raise MP3DVisualVerificationError(
            f"capture receipt has no observed {name} artifact"
        )
    candidate = point_dir / raw
    if candidate.is_symlink():
        raise MP3DVisualVerificationError(
            f"capture artifact must not be a symlink: {candidate}"
        )
    path = candidate.resolve()
    try:
        path.relative_to(point_dir.resolve())
    except ValueError as error:
        raise MP3DVisualVerificationError(
            f"capture artifact escapes its output directory: {path}"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise MP3DVisualVerificationError(f"capture artifact is missing: {path}")
    return path


def _load_array(point_dir: Path, receipt: Mapping[str, Any], name: str) -> np.ndarray:
    path = _artifact_path(point_dir, receipt, name)
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise MP3DVisualVerificationError(
            f"cannot read observed {name} array: {error}"
        ) from error
    return value


def _camera_agent(frame: Mapping[str, Any], *, frame_index: int) -> Mapping[str, Any]:
    snapshot = frame.get("camera_readback")
    if not isinstance(snapshot, Mapping):
        raise MP3DVisualVerificationError(
            f"frame {frame_index} has no observed camera_readback"
        )
    agent = snapshot.get("agent")
    if not isinstance(agent, Mapping):
        raise MP3DVisualVerificationError(
            f"frame {frame_index} camera_readback has no observed agent pose"
        )
    _transform_matrix(agent, owner=f"frame {frame_index} camera agent")
    return agent


def _m1_camera(m1_request: Mapping[str, Any] | None) -> np.ndarray | None:
    if m1_request is None:
        return None
    rig = m1_request.get("primary_camera_rig")
    if not isinstance(rig, Mapping):
        raise MP3DVisualVerificationError("M1 request has no primary_camera_rig")
    transform = rig.get("world_from_rig")
    if transform is None:
        raise MP3DVisualVerificationError("M1 request has no world_from_rig")
    return _transform_matrix(transform, owner="M1 primary camera rig")


def _load_expected_m1(
    point_dir: Path,
    *,
    case_manifest: Mapping[str, Any] | Path | None,
    m1_request: Mapping[str, Any] | Path | None,
) -> Mapping[str, Any] | None:
    if m1_request is not None:
        if isinstance(m1_request, Mapping):
            return m1_request
        return _read_json(Path(m1_request).expanduser(), owner="M1 request")
    if case_manifest is not None:
        case = (
            dict(case_manifest)
            if isinstance(case_manifest, Mapping)
            else _read_json(Path(case_manifest).expanduser(), owner="case manifest")
        )
        raw = case.get("m1_request_path")
        if isinstance(raw, str) and raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path(case_manifest).resolve().parent / path if isinstance(case_manifest, (str, Path)) else point_dir / path
            return _read_json(path.resolve(), owner="case M1 request")
    for name in ("m1_capture_request.json", "habitat_m1_capture_request.json"):
        path = point_dir / name
        if path.is_file() and not path.is_symlink():
            return _read_json(path, owner="M1 request")
    return None


def _validate_observed_arrays(
    point_dir: Path,
    receipt: Mapping[str, Any],
    *,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    arrays = {
        name: _load_array(point_dir, receipt, name)
        for name in ("rgb", "depth", "semantic")
    }
    rgb, depth, semantic = arrays["rgb"], arrays["depth"], arrays["semantic"]
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != frame_count or rgb.shape[-1] != 3:
        raise MP3DVisualVerificationError(
            f"observed RGB must be uint8 [frames,height,width,3], got {rgb.dtype} {rgb.shape}"
        )
    if depth.ndim != 3 or depth.shape[0] != frame_count:
        raise MP3DVisualVerificationError(
            f"observed depth must be [frames,height,width], got {depth.shape}"
        )
    if semantic.ndim != 3 or semantic.shape[0] != frame_count:
        raise MP3DVisualVerificationError(
            f"observed semantic must be [frames,height,width], got {semantic.shape}"
        )
    resolution = (int(rgb.shape[1]), int(rgb.shape[2]))
    if depth.shape[1:] != resolution or semantic.shape[1:] != resolution:
        raise MP3DVisualVerificationError(
            "observed RGB/depth/semantic arrays are not co-registered"
        )
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise MP3DVisualVerificationError("observed depth contains invalid values")
    if not np.issubdtype(semantic.dtype, np.integer):
        raise MP3DVisualVerificationError("observed semantic must have an integer dtype")
    return rgb, depth, semantic, resolution


def _receipt_actor_order(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    actors = receipt.get("actors")
    if not isinstance(actors, list) or not actors:
        raise MP3DVisualVerificationError(
            "native MP3D receipt must declare at least one actor"
        )
    slots: list[str] = []
    endpoints: set[str] = set()
    for index, actor in enumerate(actors):
        if not isinstance(actor, Mapping):
            raise MP3DVisualVerificationError(f"receipt actors[{index}] must be an object")
        slot = actor.get("source_slot_id")
        endpoint = actor.get("source_endpoint_id")
        if not isinstance(slot, str) or not slot:
            raise MP3DVisualVerificationError(f"receipt actors[{index}] has no source slot")
        if not isinstance(endpoint, str) or not endpoint:
            raise MP3DVisualVerificationError(f"receipt actors[{index}] has no source endpoint")
        if slot in slots or endpoint in endpoints:
            raise MP3DVisualVerificationError(
                "native MP3D receipt actor slots/endpoints must be unique"
            )
        slots.append(slot)
        endpoints.add(endpoint)
    return tuple(slots)


def verify_point(
    point_id: str,
    point_dir: str | Path,
    expected_frames: int | None = None,
    *,
    case_manifest: Mapping[str, Any] | Path | None = None,
    m1_request: Mapping[str, Any] | Path | None = None,
) -> dict[str, Any]:
    """Verify one native MP3D capture from observed frame records and arrays."""

    raw_root = Path(point_dir).expanduser()
    if raw_root.is_symlink():
        raise MP3DVisualVerificationError(
            f"native capture root must not be a symlink: {raw_root}"
        )
    root = raw_root.resolve()
    receipt = _read_json(root / "research_receipt.json", owner="native capture receipt")
    if receipt.get("schema") not in {None, NATIVE_CAPTURE_SCHEMA}:
        raise MP3DVisualVerificationError(
            "receipt is not an MP3D native capture receipt"
        )
    if receipt.get("artifact_role") != "observed_native_habitat_capture":
        raise MP3DVisualVerificationError(
            "receipt is not marked as an observed native Habitat capture"
        )
    if receipt.get("research_only") is not True or receipt.get("episode_counted") is not False:
        raise MP3DVisualVerificationError(
            "native MP3D visual verification requires research-only, uncounted evidence"
        )
    capture = receipt.get("capture")
    if not isinstance(capture, Mapping):
        raise MP3DVisualVerificationError("native capture receipt has no capture block")
    frame_count = capture.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise MP3DVisualVerificationError("capture frame_count must be a positive integer")
    if expected_frames is not None and frame_count != expected_frames:
        raise MP3DVisualVerificationError(
            f"capture frame_count {frame_count} differs from expected {expected_frames}"
        )
    if capture.get("native_habitat_started") is not True:
        raise MP3DVisualVerificationError("native Habitat execution was not recorded")
    completed = capture.get("completed_frame_count")
    if completed is not None and completed != frame_count:
        raise MP3DVisualVerificationError("capture completed_frame_count is incomplete")
    fps = capture.get("frame_rate_hz")
    ticks = capture.get("ticks_per_frame")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or isinstance(ticks, bool)
        or not isinstance(ticks, int)
        or ticks < 1
    ):
        raise MP3DVisualVerificationError("capture clock is invalid")
    records = _read_json(root / "frame_records.json", owner="observed frame records")
    frames = records.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise MP3DVisualVerificationError(
            "observed frame_records must contain exactly the receipt frame count"
        )
    if records.get("artifact_role") in {
        "planned_frame_records_not_observed_capture",
        "planned_timeline_not_native_capture",
    }:
        raise MP3DVisualVerificationError(
            "planned frame records cannot be used as observed capture evidence"
        )
    if records.get("source_endpoint_ids") is not None and not isinstance(
        records.get("source_endpoint_ids"), list
    ):
        raise MP3DVisualVerificationError("observed source_endpoint_ids must be a list")
    actor_slots = _receipt_actor_order(receipt)
    expected_actors = {
        str(actor["source_slot_id"]): actor
        for actor in receipt["actors"]
    }
    endpoint_ids = tuple(
        actor.get("source_endpoint_id") for actor in receipt["actors"]
    )
    recorded_ids = records.get("source_endpoint_ids")
    if recorded_ids is not None and recorded_ids != list(endpoint_ids):
        raise MP3DVisualVerificationError(
            "observed frame_records source endpoint order differs from the receipt"
        )
    receipt_inputs = receipt.get("inputs")
    if isinstance(receipt_inputs, Mapping):
        if case_manifest is None and receipt_inputs.get("case_manifest"):
            case_manifest = Path(str(receipt_inputs["case_manifest"]))
        if m1_request is None and receipt_inputs.get("m1_request"):
            m1_request = Path(str(receipt_inputs["m1_request"]))
    expected_m1 = _load_expected_m1(
        root, case_manifest=case_manifest, m1_request=m1_request
    )
    expected_camera = _m1_camera(expected_m1)
    rgb, depth, semantic, resolution = _validate_observed_arrays(
        root, receipt, frame_count=frame_count
    )

    maximums = {
        "actor_root_error_m": 0.0,
        "actor_root_error_cm": 0.0,
        "actor_rotation_matrix_error": 0.0,
        "emitter_position_error_m": 0.0,
        "camera_position_error_m": 0.0,
        "camera_position_error_cm": 0.0,
        "camera_rotation_matrix_error": 0.0,
        "semantic_pixel_count_error": 0,
    }
    previous_world_time: float | None = None
    observed_slots: tuple[str, ...] | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise MP3DVisualVerificationError(f"observed frame {index} must be an object")
        if frame.get("frame_index") != index:
            raise MP3DVisualVerificationError(
                f"observed frame indices must be contiguous from zero (frame {index})"
            )
        if frame.get("pts_ticks") != index * ticks:
            raise MP3DVisualVerificationError(
                f"observed frame {index} PTS differs from the capture clock"
            )
        camera = _camera_agent(frame, frame_index=index)
        camera_matrix = _transform_matrix(camera, owner=f"frame {index} camera agent")
        snapshot = frame.get("camera_readback")
        observed_time = snapshot.get("world_time_seconds")
        if isinstance(observed_time, (int, float)) and not isinstance(observed_time, bool):
            if not math.isfinite(float(observed_time)):
                raise MP3DVisualVerificationError(f"frame {index} world time is non-finite")
            current_world_time = float(observed_time)
            if (
                previous_world_time is not None
                and current_world_time < previous_world_time - 1.0e-12
            ):
                raise MP3DVisualVerificationError(
                    "Habitat world time moved backwards during capture"
                )
            previous_world_time = current_world_time
        if expected_camera is not None:
            camera_error = float(np.linalg.norm(
                camera_matrix[:3, 3] - expected_camera[:3, 3]
            ))
            camera_rotation_error = float(np.max(np.abs(
                camera_matrix[:3, :3] - expected_camera[:3, :3]
            )))
            maximums["camera_position_error_m"] = max(
                maximums["camera_position_error_m"], camera_error
            )
            maximums["camera_position_error_cm"] = (
                maximums["camera_position_error_m"] * 100.0
            )
            maximums["camera_rotation_matrix_error"] = max(
                maximums["camera_rotation_matrix_error"],
                camera_rotation_error,
            )
            if (
                camera_error > CAMERA_POSITION_ATOL_M
                or camera_rotation_error > ROTATION_MATRIX_ATOL
            ):
                raise MP3DVisualVerificationError(
                    f"frame {index} camera readback differs from the declared M1 rig"
                )

        actor_readbacks = frame.get("actor_readbacks")
        if not isinstance(actor_readbacks, list) or len(actor_readbacks) != len(actor_slots):
            raise MP3DVisualVerificationError(
                f"frame {index} actor_readbacks does not cover the receipt actors"
            )
        frame_slots: list[str] = []
        source_positions = frame.get("source_positions_m")
        if not isinstance(source_positions, list) or len(source_positions) != len(actor_slots):
            raise MP3DVisualVerificationError(
                f"frame {index} source_positions_m does not cover observed actors"
            )
        for actor_index, actor in enumerate(actor_readbacks):
            if not isinstance(actor, Mapping):
                raise MP3DVisualVerificationError(
                    f"frame {index} actor_readbacks[{actor_index}] must be an object"
                )
            slot = actor.get("source_slot_id")
            if slot not in actor_slots or slot in frame_slots:
                raise MP3DVisualVerificationError(
                    f"frame {index} actor slots differ from the receipt"
                )
            frame_slots.append(slot)
            expected_actor = expected_actors[slot]
            for field in ("source_endpoint_id", "actor_id", "asset_id"):
                expected_value = expected_actor.get(field)
                if (
                    expected_value is not None
                    and actor.get(field) != expected_value
                ):
                    raise MP3DVisualVerificationError(
                        f"frame {index} {slot} {field} differs from the receipt"
                    )
            root_matrix = _transform_matrix(
                actor.get("world_from_skin_root"),
                owner=f"frame {index} {slot} observed root",
            )
            emitter = _finite_vector(
                actor.get("emitter_world_position_m"),
                owner=f"frame {index} {slot} observed emitter",
                length=3,
            )
            source = _finite_vector(
                source_positions[actor_index],
                owner=f"frame {index} source_positions_m[{actor_index}]",
                length=3,
            )
            emitter_error = float(np.max(np.abs(source - emitter)))
            maximums["emitter_position_error_m"] = max(
                maximums["emitter_position_error_m"], emitter_error
            )
            if emitter_error > EMITTER_POSITION_ATOL_M:
                raise MP3DVisualVerificationError(
                    f"frame {index} source_positions_m is not the observed emitter "
                    f"position for {slot}"
                )
            declared_count = actor.get("semantic_pixel_count")
            if (
                isinstance(declared_count, bool)
                or not isinstance(declared_count, int)
                or declared_count < 0
            ):
                raise MP3DVisualVerificationError(
                    f"frame {index} {slot} semantic_pixel_count is invalid"
                )
            semantic_id = actor.get("semantic_id")
            if isinstance(semantic_id, bool) or not isinstance(semantic_id, int) or semantic_id < 0:
                raise MP3DVisualVerificationError(
                    f"frame {index} {slot} semantic_id is invalid"
                )
            actual_count = int(np.count_nonzero(semantic[index] == semantic_id))
            count_error = abs(actual_count - declared_count)
            maximums["semantic_pixel_count_error"] = max(
                maximums["semantic_pixel_count_error"], count_error
            )
            if count_error:
                raise MP3DVisualVerificationError(
                    f"frame {index} {slot} semantic pixel count differs from observed array"
                )
            planned = actor.get("planned_world_from_skin_root")
            if planned is not None:
                planned_matrix = _transform_matrix(
                    planned, owner=f"frame {index} {slot} planned root"
                )
                root_error = float(np.linalg.norm(
                    root_matrix[:3, 3] - planned_matrix[:3, 3]
                ))
                root_rotation_error = float(np.max(np.abs(
                    root_matrix[:3, :3] - planned_matrix[:3, :3]
                )))
                maximums["actor_root_error_m"] = max(
                    maximums["actor_root_error_m"], root_error
                )
                maximums["actor_root_error_cm"] = maximums["actor_root_error_m"] * 100.0
                maximums["actor_rotation_matrix_error"] = max(
                    maximums["actor_rotation_matrix_error"],
                    root_rotation_error,
                )
                if (
                    root_error > ROOT_POSITION_ATOL_M
                    or root_rotation_error > ROTATION_MATRIX_ATOL
                ):
                    raise MP3DVisualVerificationError(
                        f"frame {index} {slot} observed root differs from planned apply root"
                    )
        if tuple(frame_slots) != actor_slots:
            raise MP3DVisualVerificationError(
                f"frame {index} actor order differs from the receipt"
            )
        if observed_slots is None:
            observed_slots = tuple(frame_slots)
        elif observed_slots != tuple(frame_slots):
            raise MP3DVisualVerificationError("actor order changed between observed frames")

    return {
        "point_id": point_id,
        "frame_count": frame_count,
        "frame_rate_hz": fps,
        "resolution_hw": list(resolution),
        "actor_count": len(actor_slots),
        "animation_status": "not_applicable",
        "status": "pass",
        **{f"maximum_{key}": value for key, value in maximums.items()},
    }


def _selection_point_ids(selection: Mapping[str, Any]) -> list[str]:
    records = selection.get("selected", selection.get("records"))
    if not isinstance(records, list):
        raise MP3DVisualVerificationError(
            "selection manifest needs selected or records entries"
        )
    ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("point_id"), str):
            raise MP3DVisualVerificationError(
                f"selection record {index} has no point_id"
            )
        ids.append(record["point_id"])
    if len(ids) != len(set(ids)):
        raise MP3DVisualVerificationError("selection contains duplicate point ids")
    return ids


def verify_batch(
    selection_manifest: str | Path | Mapping[str, Any] | None = None,
    visual_root: str | Path | None = None,
    expected_frames: int | None = None,
    *,
    case_manifests: Mapping[str, Mapping[str, Any] | Path] | None = None,
    m1_requests: Mapping[str, Mapping[str, Any] | Path] | None = None,
) -> dict[str, Any]:
    """Verify one or many observed native MP3D captures.

    For convenience, passing only one path treats it as the capture directory.
    A selection manifest may list 'selected' or 'records' point ids just
    like the existing QA visual verifier.
    """

    if visual_root is None:
        visual_root = selection_manifest
        selection_manifest = None
    if visual_root is None:
        raise MP3DVisualVerificationError("visual_root is required")
    raw_root = Path(visual_root).expanduser()
    if raw_root.is_symlink():
        raise MP3DVisualVerificationError(
            f"visual root must not be a symlink: {raw_root}"
        )
    root = raw_root.resolve()
    if not root.is_dir():
        raise MP3DVisualVerificationError(f"visual root must be a directory: {root}")

    selection: Mapping[str, Any] | None
    selection_path: Path | None = None
    if selection_manifest is None:
        selection = None
    elif isinstance(selection_manifest, Mapping):
        selection = selection_manifest
    else:
        raw_selection = Path(selection_manifest).expanduser()
        if raw_selection.is_symlink():
            raise MP3DVisualVerificationError(
                f"selection manifest must not be a symlink: {raw_selection}"
            )
        selection_path = raw_selection.resolve()
        selection = _read_json(selection_path, owner="selection manifest")

    if (root / "frame_records.json").is_file() and (root / "research_receipt.json").is_file():
        point_ids = [root.name]
        point_dirs = {root.name: root}
    else:
        point_ids = (
            _selection_point_ids(selection)
            if selection is not None
            else sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and (path / "frame_records.json").is_file()
                and (path / "research_receipt.json").is_file()
            )
        )
        if not point_ids:
            raise MP3DVisualVerificationError("visual root contains no native capture points")
        point_dirs = {point_id: root / point_id for point_id in point_ids}
        actual = {
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink()
            and (path / "frame_records.json").is_file()
            and (path / "research_receipt.json").is_file()
        }
        if selection is not None and actual != set(point_ids):
            raise MP3DVisualVerificationError(
                "visual point coverage differs from the selection manifest: "
                f"missing={sorted(set(point_ids) - actual)}, extra={sorted(actual - set(point_ids))}"
            )

    results = []
    for point_id in point_ids:
        case = case_manifests.get(point_id) if case_manifests is not None else None
        m1 = m1_requests.get(point_id) if m1_requests is not None else None
        results.append(
            verify_point(
                point_id,
                point_dirs[point_id],
                expected_frames,
                case_manifest=case,
                m1_request=m1,
            )
        )

    def maximum(field: str) -> float | int | None:
        return max((result[field] for result in results), default=None)

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": (
            "observed native Habitat MP3D visual/readback engineering "
            "verification only; no question, acoustic or dataset admission"
        ),
        "inputs": {
            "selection_manifest": None if selection_path is None else str(selection_path),
            "visual_root": str(root),
            "expected_frames_per_point": expected_frames,
        },
        "counts": {
            "selected_points": len(point_ids),
            "verified_points": len(results),
            "verified_frames": sum(int(result["frame_count"]) for result in results),
            "failures": 0,
        },
        "maxima": {
            "actor_root_error_m": maximum("maximum_actor_root_error_m"),
            "actor_root_error_cm": maximum("maximum_actor_root_error_cm"),
            "actor_rotation_matrix_error": maximum("maximum_actor_rotation_matrix_error"),
            "emitter_position_error_m": maximum("maximum_emitter_position_error_m"),
            "camera_position_error_m": maximum("maximum_camera_position_error_m"),
            "camera_position_error_cm": maximum("maximum_camera_position_error_cm"),
            "camera_rotation_matrix_error": maximum("maximum_camera_rotation_matrix_error"),
            "semantic_pixel_count_error": maximum("maximum_semantic_pixel_count_error"),
        },
        "points": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", "--capture-root", dest="visual_root", required=True, type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--m1-request", type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    try:
        summary = verify_batch(
            args.selection_manifest,
            args.visual_root,
            args.expected_frames,
            case_manifests=(
                None
                if args.case_manifest is None
                else {args.visual_root.resolve().name: args.case_manifest}
            ),
            m1_requests=(
                None
                if args.m1_request is None
                else {args.visual_root.resolve().name: args.m1_request}
            ),
        )
    except MP3DVisualVerificationError as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False))
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": summary["counts"],
                "maxima": summary["maxima"],
                "out": str(args.out.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
