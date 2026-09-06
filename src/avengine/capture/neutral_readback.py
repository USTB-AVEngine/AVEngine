"""Renderer-neutral observed positions, camera basis and authoritative clock."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "avengine_neutral_readback_v1"
COORDINATE_FRAME = {"linear_unit": "meter", "up_axis": "+Y", "handedness": "right"}


def validate_clock(clock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the supplied plan clock; never derive an independent duration."""
    required = ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count",
                "time_base_hz", "ticks_per_frame")
    if not isinstance(clock, Mapping):
        raise ValueError("clock must be a mapping")
    for key in required:
        value = clock.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"clock.{key} must be a positive finite number")
        if key != "frame_rate_hz" and int(value) != value:
            raise ValueError(f"clock.{key} must be an integer")
    fps = Fraction(str(clock["frame_rate_hz"]))
    if fps * int(clock["ticks_per_frame"]) != int(clock["time_base_hz"]):
        raise ValueError("clock tick and frame rates disagree")
    duration = Fraction(int(clock["frame_count"]), 1) / fps
    # Short native canaries can end between sample boundaries. The existing
    # plan clock rounds once to the nearest sample; retain that declared count.
    if abs(duration * int(clock["sample_rate_hz"]) - int(clock["sample_count"])) > Fraction(1, 2):
        raise ValueError("clock frame and sample counts disagree")
    for name in ("clip_seconds", "duration_seconds"):
        if name in clock and abs(float(clock[name]) - float(duration)) > 1e-9:
            raise ValueError(f"clock.{name} disagrees with frame count")
    return deepcopy(dict(clock))


def vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return result


def validate_neutral_readback(data: Mapping[str, Any], *, plan: Mapping[str, Any] | None = None) -> dict:
    if data.get("schema") != SCHEMA:
        raise ValueError("unsupported NeutralReadback schema")
    frame = data.get("coordinate_frame", {})
    if any(frame.get(key) != value for key, value in COORDINATE_FRAME.items()):
        raise ValueError("NeutralReadback must be meter, +Y up and right handed")
    clock = validate_clock(data.get("clock"))
    if plan is not None:
        expected = validate_clock(plan["clock"])
        for key in ("frame_count", "frame_rate_hz", "sample_count", "sample_rate_hz",
                    "time_base_hz", "ticks_per_frame"):
            if clock[key] != expected[key]:
                raise ValueError(f"readback clock differs from plan: {key}")
    count = int(clock["frame_count"])
    camera = data.get("camera")
    entities = data.get("entities")
    if not isinstance(camera, list) or len(camera) != count:
        raise ValueError("camera readbacks do not cover clock")
    if not isinstance(entities, Mapping) or not entities:
        raise ValueError("entity readbacks are required")
    series = {"camera": camera, **{f"entity:{key}": value for key, value in entities.items()}}
    for label, records in series.items():
        if not isinstance(records, list) or len(records) != count:
            raise ValueError(f"{label} readbacks do not cover clock")
        for i, record in enumerate(records):
            if record.get("frame_index") != i or record.get("pts_ticks") != i * clock["ticks_per_frame"]:
                raise ValueError(f"{label} has missing, reordered or mistimed frame {i}")
            if label == "camera":
                vector3(record.get("position_m"), "camera.position_m")
                basis = record.get("basis", {})
                forward, right, up = (vector3(basis.get(k), f"camera.basis.{k}")
                                      for k in ("forward", "right", "up"))
                axes = np.column_stack((right, up, -forward))
                if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(axes), 1, atol=1e-5):
                    raise ValueError("camera basis must be right-handed and orthonormal")
            else:
                vector3(record.get("root"), f"{label}.root")
                vector3(record.get("emitter"), f"{label}.emitter")
                if not isinstance(record.get("moving"), bool):
                    raise ValueError(f"{label}.moving must be an observed boolean")
    if not isinstance(data.get("producer"), Mapping) or not data["producer"].get("source_readbacks"):
        raise ValueError("readback producer and source_readbacks are required")
    return {"status": "pass", "frame_count": count, "entity_ids": list(entities),
            "coordinate_frame": dict(COORDINATE_FRAME)}


def observed_motion(positions: Any, frame_rate_hz: float) -> list[bool]:
    """Reuse unified_catalog's forward difference and 0.05 m/s motion rule."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("observed root positions must be [frame,3]")
    if len(values) < 2:
        raise ValueError("at least two observed frames are needed to infer movement")
    speed = np.linalg.norm(np.diff(values, axis=0), axis=1) * frame_rate_hz
    return (np.r_[speed, speed[-1]] > 0.05).tolist()


def write_neutral_readback(path: Path, data: Mapping[str, Any], *, plan: Mapping[str, Any]) -> dict:
    validation = validate_neutral_readback(data, plan=plan)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return validation
