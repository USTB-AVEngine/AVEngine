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

from avengine.m5.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    EPISODE_FRAME_COUNT,
)

# Mouth/muzzle probe heights declared by the fixed-apartment anchor library.
APARTMENT_EMITTER_HEIGHTS_M = {"source1": 1.63, "source2": 0.45}
APARTMENT_SLOT_ENDPOINTS = {
    "source1": "m6x_human0_mouth",
    "source2": "m6x_dog0_muzzle",
}


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
