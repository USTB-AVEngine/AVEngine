"""UE executor's observed centimeter/Z-up to neutral meter/Y-up exchange."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.capture.neutral_readback import (
    COORDINATE_FRAME, SCHEMA, observed_motion, validate_clock, vector3, write_neutral_readback,
)


def ue_cm_to_neutral_m(value: Any) -> list[float]:
    return (vector3(value, "UE location_cm")[[0, 2, 1]] / 100.0).tolist()


def neutral_m_to_ue_cm(value: Any) -> list[float]:
    return (vector3(value, "neutral position_m")[[0, 2, 1]] * 100.0).tolist()


def ue_rotator_to_neutral_basis(value: Any) -> dict[str, list[float]]:
    """Preserve the native RLR/unified_catalog optical-basis convention."""
    roll, pitch, yaw = np.radians(vector3(value, "UE rotation_deg"))
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    forward = np.array([cp * cy, cp * sy, sp])[[0, 2, 1]]
    right0 = np.array([-sy, cy, 0.0])
    up0 = np.array([-sp * cy, -sp * sy, cp])
    up = (-right0 * math.sin(roll) + up0 * math.cos(roll))[[0, 2, 1]]
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    forward /= np.linalg.norm(forward)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return {key: axis.tolist() for key, axis in (("forward", forward), ("right", right), ("up", up))}


def neutral_from_ue_readbacks(readbacks: Mapping[str, Any], plan: Mapping[str, Any], *,
                              source_readbacks: str) -> dict:
    clock = validate_clock(plan["clock"])
    for key, value in readbacks.get("clock", {}).items():
        if key in clock and value != clock[key]:
            raise ValueError(f"UE observed clock differs from plan: {key}")
    actor_tracks, emitter_tracks = readbacks["actors"], readbacks["emitters"]
    if set(actor_tracks) != set(emitter_tracks):
        raise ValueError("UE root and emitter entity sets differ")
    camera = []
    for record in readbacks["camera"]:
        i = record["frame_index"]
        camera.append({"frame_index": i, "pts_ticks": i * clock["ticks_per_frame"],
                       "position_m": ue_cm_to_neutral_m(record["location_cm"]),
                       "basis": ue_rotator_to_neutral_basis(record["rotation_deg"])})
    entities = {}
    for aid, records in actor_tracks.items():
        emitters = emitter_tracks[aid]
        if len(records) != len(emitters):
            raise ValueError(f"UE root/emitter frame counts differ for {aid}")
        roots = [ue_cm_to_neutral_m(record["location_cm"]) for record in records]
        moving = observed_motion(roots, clock["frame_rate_hz"])
        converted = []
        for record, emitter, root, motion in zip(records, emitters, roots, moving, strict=True):
            i = record["frame_index"]
            if emitter["frame_index"] != i:
                raise ValueError(f"UE root/emitter frame identity differs for {aid}")
            converted.append({"frame_index": i, "pts_ticks": i * clock["ticks_per_frame"],
                              "root": root, "emitter": ue_cm_to_neutral_m(emitter["location_cm"]),
                              "moving": motion})
        entities[aid] = converted
    return {"schema": SCHEMA, "clock": clock, "coordinate_frame": dict(COORDINATE_FRAME),
            "camera": camera, "entities": entities,
            "producer": {"module": __name__, "renderer": "ue_spear",
                         "source_readbacks": [source_readbacks],
                         "world_transform": "ue_xyz_cm_to_xzy_m_v1",
                         "movement": "observed_root_forward_difference_gt_0.05_mps; last repeats previous"}}


def write_ue_neutral_readback(capture_dir: Path, plan: Mapping[str, Any],
                              output_path: Path | None = None) -> dict:
    source = Path(capture_dir) / "frame_readbacks.json"
    data = neutral_from_ue_readbacks(json.loads(source.read_text()), plan,
                                    source_readbacks=str(source.resolve()))
    path = output_path or Path(capture_dir) / "neutral_readback.json"
    write_neutral_readback(path, data, plan=plan)
    return data
