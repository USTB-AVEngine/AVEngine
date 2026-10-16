"""Explicit, hash-bound GLB action-duration normalization."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

from avengine.assets.glb import (
    GlbError,
    decode_accessor,
    extract_actions,
    load_glb,
    parse_glb,
)
from avengine.assets.glb_write import build_glb


class ActionTimingError(ValueError):
    """An action cannot be retimed without an ambiguous accessor edit."""


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create one output without following/replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _objects(value: Any, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ActionTimingError(f"{owner} must be an array of objects")
    return value


def _append_scalar(
    document: dict[str, Any], binary: bytearray, values: np.ndarray
) -> int:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ActionTimingError("retimed timestamp accessor is invalid")
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    packer = struct.Struct("<f")
    for value in array:
        binary.extend(packer.pack(float(value)))
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    if not isinstance(views, list) or not isinstance(accessors, list):
        raise ActionTimingError("bufferViews/accessors must be arrays")
    view_index = len(views)
    views.append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
    )
    accessor_index = len(accessors)
    accessors.append(
        {
            "bufferView": view_index,
            "componentType": 5126,
            "count": len(array),
            "type": "SCALAR",
            "min": [float(np.min(array))],
            "max": [float(np.max(array))],
        }
    )
    return accessor_index


def retime_glb_actions(
    source_path: str | Path,
    output_path: str | Path,
    *,
    durations_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Scale selected action clocks while preserving every sampled pose value."""

    source_resolved = Path(source_path).resolve()
    output_argument = Path(output_path)
    if output_argument.exists() or output_argument.is_symlink():
        raise ActionTimingError(f"refusing to replace output: {output_argument}")
    output_resolved = output_argument.resolve()
    if source_resolved == output_resolved:
        raise ActionTimingError("output must not overwrite the source GLB")
    if not durations_seconds:
        raise ActionTimingError("at least one action duration is required")
    try:
        source = load_glb(source_resolved)
        actions = extract_actions(source)
    except GlbError as exc:
        raise ActionTimingError(f"invalid input GLB: {exc}") from exc
    by_name = {action.name: action for action in actions}
    missing = sorted(set(durations_seconds) - set(by_name))
    if missing:
        raise ActionTimingError(f"action(s) not found: {missing}")
    source_animations = _objects(source.json.get("animations", []), "animations")
    action_names_by_index = {action.animation_index: action.name for action in actions}
    unsupported_samplers = [
        f"{action_names_by_index[animation_index]}/sampler[{sampler_index}]="
        f"{sampler.get('interpolation', 'LINEAR')}"
        for animation_index, animation in enumerate(source_animations)
        for sampler_index, sampler in enumerate(
            _objects(
                animation.get("samplers"),
                f"animation {action_names_by_index[animation_index]!r}.samplers",
            )
        )
        if sampler.get("interpolation", "LINEAR") not in {"LINEAR", "STEP"}
    ]
    if unsupported_samplers:
        raise ActionTimingError(
            f"unsupported retiming interpolation(s) {unsupported_samplers}; "
            "CUBICSPLINE tangent values would require an explicit time-domain "
            "conversion, so only LINEAR and STEP are accepted"
        )
    for name, duration in durations_seconds.items():
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not np.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise ActionTimingError(
                f"duration for {name!r} must be finite and positive"
            )

    document = copy.deepcopy(source.json)
    buffers = _objects(document.get("buffers", []), "buffers")
    if len(buffers) != 1:
        raise ActionTimingError("retiming requires one embedded buffer")
    declared_length = buffers[0].get("byteLength")
    if (
        isinstance(declared_length, bool)
        or not isinstance(declared_length, int)
        or declared_length <= 0
    ):
        raise ActionTimingError("buffers[0].byteLength is invalid")
    binary = bytearray(source.binary[:declared_length])
    animations = _objects(document.get("animations", []), "animations")
    records: list[dict[str, Any]] = []
    for name, desired_raw in durations_seconds.items():
        action = by_name[name]
        desired = float(desired_raw)
        starts = [channel.timestamps_seconds[0] for channel in action.channels]
        ends = [channel.timestamps_seconds[-1] for channel in action.channels]
        clip_start = min(starts)
        clip_end = max(ends)
        source_duration = clip_end - clip_start
        if source_duration <= 0.0:
            raise ActionTimingError(f"action {name!r} has zero duration")
        factor = desired / source_duration
        raw_animation = animations[action.animation_index]
        samplers = _objects(raw_animation.get("samplers"), "animation.samplers")
        accessor_remap: dict[int, int] = {}
        for sampler in samplers:
            source_accessor = sampler.get("input")
            if isinstance(source_accessor, bool) or not isinstance(
                source_accessor, int
            ):
                raise ActionTimingError(
                    "animation sampler input must be an accessor index"
                )
            if source_accessor not in accessor_remap:
                try:
                    decoded = decode_accessor(source, source_accessor)
                except GlbError as exc:
                    raise ActionTimingError(
                        f"invalid timestamp accessor: {exc}"
                    ) from exc
                if decoded.element_type != "SCALAR":
                    raise ActionTimingError(
                        "animation sampler input must be FLOAT SCALAR"
                    )
                values = np.asarray(decoded.scalars, dtype=np.float64)
                retimed = clip_start + (values - clip_start) * factor
                accessor_remap[source_accessor] = _append_scalar(
                    document, binary, retimed
                )
            sampler["input"] = accessor_remap[source_accessor]
        records.append(
            {
                "action": name,
                "source_clip_start_seconds": clip_start,
                "source_clip_end_seconds": clip_end,
                "source_duration_seconds": source_duration,
                "output_duration_seconds_requested": desired,
                "time_scale_output_over_source": factor,
                "pose_output_accessors_changed": False,
                "timestamp_accessor_remap": {
                    str(old): new for old, new in sorted(accessor_remap.items())
                },
            }
        )
    buffers[0]["byteLength"] = len(binary)
    payload = build_glb(document, binary)
    try:
        verified = parse_glb(payload)
        verified_actions = {action.name: action for action in extract_actions(verified)}
    except GlbError as exc:
        raise ActionTimingError(f"output readback failed: {exc}") from exc
    for name, desired_raw in durations_seconds.items():
        action = verified_actions[name]
        starts = [channel.timestamps_seconds[0] for channel in action.channels]
        ends = [channel.timestamps_seconds[-1] for channel in action.channels]
        actual = max(ends) - min(starts)
        if abs(actual - float(desired_raw)) > 1.0e-5:
            raise ActionTimingError(
                f"retimed action duration readback differs: {name} {actual:.9g}"
            )
    try:
        _write_exclusive(output_resolved, payload)
    except OSError as exc:
        raise ActionTimingError(
            f"failed to create output exclusively: {output_resolved}: {exc}"
        ) from exc
    return {
        "schema": "avengine_m2_action_timing_normalization_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_resolved),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output_resolved),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "actions": records,
        "notes": [
            "Only animation input timestamp accessors are replaced; pose output accessors are unchanged.",
            "The explicit time scale requires species-motion review and does not qualify the asset.",
        ],
    }
