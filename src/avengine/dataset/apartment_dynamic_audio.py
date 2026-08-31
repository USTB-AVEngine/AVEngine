"""Bind the current Apartment UE capture to the dynamic research-audio core.

The UE-native capture records per-frame actor anchor poses in centimeters.
The legacy glTF-import transform is ``U = 100 * (H.x, H.z, H.y)`` (see
``habitat_point_to_apartment_ue_cm``), so world meters recover as
``H = (U.x / 100, U.z / 100, U.y / 100)``. Anchor poses ride at the actor
root; the acoustic emitters use the declared per-slot mouth/muzzle heights
from the fixed-apartment anchor library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    EPISODE_FRAME_COUNT,
)

# Mouth/muzzle probe heights declared by the fixed-apartment anchor library.
APARTMENT_EMITTER_HEIGHTS_M = {"source1": 1.63, "source2": 0.45}
APARTMENT_SLOT_ENDPOINTS = {
    "source1": "m6x_human0_mouth",
    "source2": "m6x_dog0_muzzle",
}


def derive_slot_bindings(
    actor_selection: Mapping,
    source_asset_registry: Mapping,
    endpoint_registry: Mapping,
) -> tuple[dict[str, str], dict[str, float]]:
    """Per-slot endpoint ids and emitter heights from the executed actor
    selection, replacing the legacy human+beagle constants for arbitrary
    registered pairs (e.g. two humans or two dogs).

    Fail-closed: every selected actor must resolve to exactly one endpoint
    whose binding matches the actor's instance id, asset id and the asset's
    default emitter anchor; heights come from that anchor's measured offset.
    """

    actors = actor_selection.get("actors")
    if not isinstance(actors, list) or not actors:
        raise CurrentMP3DDynamicAudioError("actor selection has no actors")
    assets = {
        record.get("asset_id"): record
        for record in source_asset_registry.get("assets", [])
        if isinstance(record, Mapping)
    }
    endpoints = [
        entry
        for entry in endpoint_registry.get("source_endpoints", [])
        if isinstance(entry, Mapping)
    ]
    slot_endpoints: dict[str, str] = {}
    emitter_heights: dict[str, float] = {}
    for actor in actors:
        slot = actor.get("source_slot_id")
        asset_id = actor.get("asset_id")
        instance_id = actor.get("legacy_timeline_actor_id")
        record = assets.get(asset_id)
        if record is None:
            raise CurrentMP3DDynamicAudioError(
                f"actor selection references unregistered asset {asset_id}"
            )
        anchor_id = record.get("default_emitter_anchor_id")
        anchor = next(
            (
                item
                for item in record.get("emitter_anchors", [])
                if isinstance(item, Mapping) and item.get("anchor_id") == anchor_id
            ),
            None,
        )
        if anchor is None or len(anchor.get("offset_m", [])) != 3:
            raise CurrentMP3DDynamicAudioError(
                f"asset {asset_id} has no usable default emitter anchor"
            )
        matches = [
            entry
            for entry in endpoints
            if entry.get("binding", {}).get("entity_asset_id") == asset_id
            and entry.get("binding", {}).get("entity_instance_id") == instance_id
            and entry.get("binding", {}).get("emitter_anchor_id") == anchor_id
        ]
        if len(matches) != 1:
            raise CurrentMP3DDynamicAudioError(
                f"expected exactly one endpoint for {instance_id}/{asset_id}/"
                f"{anchor_id}, found {len(matches)}"
            )
        slot_endpoints[slot] = str(matches[0]["source_endpoint_id"])
        emitter_heights[slot] = float(anchor["offset_m"][1])
    if set(slot_endpoints) != {"source1", "source2"}:
        raise CurrentMP3DDynamicAudioError(
            "actor selection must bind exactly source1 and source2"
        )
    return slot_endpoints, emitter_heights


def apartment_ue_point_to_world_m(point_ue_cm) -> list[float]:
    """Invert the legacy glTF-import transform ``U = 100 * (H.x, H.z, H.y)``."""

    values = [float(value) for value in point_ue_cm]
    if len(values) != 3 or not all(np.isfinite(values)):
        raise CurrentMP3DDynamicAudioError("UE points must be finite 3-vectors")
    x, y, z = values
    return [x / 100.0, z / 100.0, y / 100.0]


def _frames(visual_capture_dir: str | Path) -> list:
    records_path = Path(visual_capture_dir).resolve() / "frame_records.json"
    if not records_path.is_file():
        raise CurrentMP3DDynamicAudioError(
            f"visual capture is missing frame_records.json: {records_path}"
        )
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != EPISODE_FRAME_COUNT:
        raise CurrentMP3DDynamicAudioError(
            "frame_records must carry exactly the 75 episode frames"
        )
    return frames


def load_ue_anchor_trajectories(
    visual_capture_dir: str | Path,
    *,
    slot_endpoints: Mapping[str, str] = APARTMENT_SLOT_ENDPOINTS,
    emitter_heights_m: Mapping[str, float] = APARTMENT_EMITTER_HEIGHTS_M,
) -> dict[str, list[list[float]]]:
    """World-meter emitter trajectories from the UE anchor pose records."""

    if set(slot_endpoints) != set(emitter_heights_m):
        raise CurrentMP3DDynamicAudioError(
            "slot endpoints and emitter heights must cover the same slots"
        )
    trajectories: dict[str, list[list[float]]] = {
        endpoint: [] for endpoint in slot_endpoints.values()
    }
    for index, frame in enumerate(_frames(visual_capture_dir)):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
            raise CurrentMP3DDynamicAudioError(
                "frame_records indices must be contiguous from zero"
            )
        anchors = frame.get("actor_anchor_poses")
        if not isinstance(anchors, Mapping):
            raise CurrentMP3DDynamicAudioError(
                "each frame must record actor_anchor_poses"
            )
        for slot, endpoint in slot_endpoints.items():
            pose = anchors.get(slot)
            if not isinstance(pose, Mapping) or "location_cm" not in pose:
                raise CurrentMP3DDynamicAudioError(
                    f"frame {index} is missing the {slot} anchor pose"
                )
            point = apartment_ue_point_to_world_m(pose["location_cm"])
            point[1] = float(emitter_heights_m[slot])
            trajectories[endpoint].append(point)
    return trajectories


def captured_static_camera_world_m(
    visual_capture_dir: str | Path,
) -> tuple[list[float], float]:
    """Return the static camera world position and its UE yaw in degrees."""

    frames = _frames(visual_capture_dir)
    first = frames[0].get("camera_pose")
    if not isinstance(first, Mapping):
        raise CurrentMP3DDynamicAudioError("frame 0 must record camera_pose")
    for index, frame in enumerate(frames):
        pose = frame.get("camera_pose")
        if pose != first:
            raise CurrentMP3DDynamicAudioError(
                f"the capture camera moves at frame {index}; a static listener "
                "pose cannot represent it"
            )
    world = apartment_ue_point_to_world_m(first["location_cm"])
    yaw = float(first["rotation_deg"][2])
    return world, yaw


def listener_ue_yaw_deg(listener_orientation_wxyz) -> float:
    """UE yaw (degrees) implied by the habitat listener orientation.

    The habitat camera looks down ``-Z`` and the legacy glTF-import transform
    maps UE ``(x, y)`` to habitat ``(x, z)``, so a yaw-only listener faces
    ``(cos(yaw), 0, sin(yaw))`` in world meters. Raises when the orientation
    carries pitch or roll, because a single UE yaw cannot represent it.
    """

    values = [float(v) for v in listener_orientation_wxyz]
    if len(values) != 4:
        raise CurrentMP3DDynamicAudioError("listener orientation must be wxyz")
    w, x, y, z = values
    axis = np.array([x, y, z])
    forward = np.array([0.0, 0.0, -1.0])
    rotated = forward + 2.0 * np.cross(
        axis, np.cross(axis, forward) + w * forward
    )
    if abs(float(rotated[1])) > 1.0e-6:
        raise CurrentMP3DDynamicAudioError(
            "the listener orientation is not yaw-only: forward vector "
            f"{rotated.tolist()} leaves the horizontal plane"
        )
    return float(np.degrees(np.arctan2(rotated[2], rotated[0])))


def assert_listener_matches_capture_yaw(
    listener_orientation_wxyz, camera_ue_yaw_deg: float, *, tolerance_deg: float = 1.0e-3
) -> float:
    """Fail closed when the M1 listener faces elsewhere than the capture camera.

    The renderer takes the listener **orientation** from the M1 request while
    the video camera yaw comes from the capture; only the position was
    cross-checked, so a per-point camera yaw would rotate the picture while
    leaving the binaural rendering untouched — silently, and in a spatial
    audio benchmark. This closes that gap.
    """

    listener_yaw = listener_ue_yaw_deg(listener_orientation_wxyz)
    gap = abs((listener_yaw - float(camera_ue_yaw_deg) + 180.0) % 360.0 - 180.0)
    if gap > tolerance_deg:
        raise CurrentMP3DDynamicAudioError(
            "the capture camera yaw does not match the M1 listener "
            f"orientation: capture {camera_ue_yaw_deg} deg vs request "
            f"{listener_yaw:.6f} deg (gap {gap:.6f} deg); the video would "
            "rotate while the binaural audio does not"
        )
    return listener_yaw
