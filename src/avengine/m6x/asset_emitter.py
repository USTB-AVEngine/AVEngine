"""Simple asset-specific emitter offsets applied to reusable root paths.

Navigation trajectories name only ``source1`` and ``source2``.  A dataset
episode binds each slot to a concrete visual asset and one constant emitter
offset in that asset's final, scaled root frame.  The resulting world-space
point path can be sent directly to RLR without inspecting a skeleton or
requiring mouth animation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m5_1.mixed_capture import MixedCaptureError, trajectory_world_matrices
from avengine.m6x.room_feasibility import (
    SOURCE_SLOTS,
    TrajectoryBank,
    TrajectoryEpisode,
)


ASSET_EMITTER_BINDING_SET_SCHEMA = "avengine_asset_emitter_binding_set_v1"
ASSET_EMITTER_BINDING_REPORT_SCHEMA = "avengine_asset_emitter_binding_report_v1"


class AssetEmitterBindingError(ValueError):
    """A concrete asset cannot be bound to a generic source-slot route."""


def _nonempty(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetEmitterBindingError(f"{owner} must be a non-empty string")
    return value.strip()


def _vector(value: Any, length: int, *, owner: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise AssetEmitterBindingError(f"{owner} must contain {length} numbers")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise AssetEmitterBindingError(
            f"{owner} must contain {length} numbers"
        ) from exc
    if len(items) != length:
        raise AssetEmitterBindingError(f"{owner} must contain {length} numbers")
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise AssetEmitterBindingError(f"{owner}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise AssetEmitterBindingError(f"{owner}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _points(value: Any, *, owner: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssetEmitterBindingError(f"{owner} must be finite [frame,3]") from exc
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise AssetEmitterBindingError(f"{owner} must be finite [frame,3]")
    if len(points) < 2:
        raise AssetEmitterBindingError(f"{owner} must contain at least two frames")
    return np.ascontiguousarray(points)


@dataclass(frozen=True)
class AssetEmitterBinding:
    """One concrete visual asset's constant point-emitter placement."""

    source_slot_id: str
    asset_id: str
    semantic_anchor_id: str
    emitter_offset_m: tuple[float, float, float]
    local_anatomical_forward_axis: tuple[float, float, float]

    def record(self) -> dict[str, Any]:
        return {
            "source_slot_id": self.source_slot_id,
            "asset_id": self.asset_id,
            "semantic_anchor_id": self.semantic_anchor_id,
            "emitter_offset_m": list(self.emitter_offset_m),
            "local_anatomical_forward_axis": list(self.local_anatomical_forward_axis),
            "offset_space": "final_scaled_asset_root",
        }


@dataclass(frozen=True)
class BoundEmitterPaths:
    """World emitter paths and compact evidence for one bound episode."""

    paths_m: Mapping[str, np.ndarray]
    actor_world_matrices: Mapping[str, np.ndarray]
    report: Mapping[str, Any]


def validate_asset_emitter_binding_set(
    value: Mapping[str, Any],
) -> dict[str, AssetEmitterBinding]:
    """Validate the deliberately small per-asset emitter configuration."""

    if (
        not isinstance(value, Mapping)
        or value.get("schema") != ASSET_EMITTER_BINDING_SET_SCHEMA
    ):
        raise AssetEmitterBindingError(
            f"binding schema must be {ASSET_EMITTER_BINDING_SET_SCHEMA}"
        )
    raw_bindings = value.get("bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != len(SOURCE_SLOTS):
        raise AssetEmitterBindingError(
            "bindings must contain exactly source1 and source2"
        )
    bindings: dict[str, AssetEmitterBinding] = {}
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise AssetEmitterBindingError(f"bindings[{index}] must be an object")
        source_slot = _nonempty(
            raw.get("source_slot_id"), owner=f"bindings[{index}].source_slot_id"
        )
        if source_slot not in SOURCE_SLOTS or source_slot in bindings:
            raise AssetEmitterBindingError(
                "bindings must identify source1 and source2 exactly once"
            )
        offset_space = raw.get("offset_space", "final_scaled_asset_root")
        if offset_space != "final_scaled_asset_root":
            raise AssetEmitterBindingError(
                "emitter_offset_m must use final_scaled_asset_root space"
            )
        forward = _vector(
            raw.get("local_anatomical_forward_axis"),
            3,
            owner=f"bindings[{index}].local_anatomical_forward_axis",
        )
        horizontal_norm = math.hypot(forward[0], forward[2])
        if abs(forward[1]) > 1.0e-12 or horizontal_norm <= 1.0e-12:
            raise AssetEmitterBindingError(
                "local anatomical forward must be a nonzero horizontal axis"
            )
        normalized_forward = (
            forward[0] / horizontal_norm,
            0.0,
            forward[2] / horizontal_norm,
        )
        bindings[source_slot] = AssetEmitterBinding(
            source_slot_id=source_slot,
            asset_id=_nonempty(
                raw.get("asset_id"), owner=f"bindings[{index}].asset_id"
            ),
            semantic_anchor_id=_nonempty(
                raw.get("semantic_anchor_id"),
                owner=f"bindings[{index}].semantic_anchor_id",
            ),
            emitter_offset_m=_vector(
                raw.get("emitter_offset_m"),
                3,
                owner=f"bindings[{index}].emitter_offset_m",
            ),
            local_anatomical_forward_axis=normalized_forward,
        )
    if set(bindings) != set(SOURCE_SLOTS):
        raise AssetEmitterBindingError(
            "bindings must contain exactly source1 and source2"
        )
    return {source_slot: bindings[source_slot] for source_slot in SOURCE_SLOTS}


def _fallback_toward_listener(
    root_path_m: np.ndarray, listener_position_m: np.ndarray
) -> tuple[float, float]:
    delta = (
        listener_position_m[(0, 2),]
        - root_path_m[
            0,
            (0, 2),
        ]
    )
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-12:
        return (0.0, -1.0)
    return (float(delta[0] / norm), float(delta[1] / norm))


def materialize_asset_emitter_paths(
    bindings: Mapping[str, AssetEmitterBinding],
    *,
    source_root_paths_m: Mapping[str, Any],
    source_fallback_forwards_xz: Mapping[str, Any],
) -> BoundEmitterPaths:
    """Apply one constant local emitter offset to each source-slot root path."""

    if set(bindings) != set(SOURCE_SLOTS):
        raise AssetEmitterBindingError(
            "bindings must contain exactly source1 and source2"
        )
    if set(source_root_paths_m) != set(SOURCE_SLOTS):
        raise AssetEmitterBindingError(
            "source_root_paths_m must contain exactly source1 and source2"
        )
    if set(source_fallback_forwards_xz) != set(SOURCE_SLOTS):
        raise AssetEmitterBindingError(
            "fallback forwards must contain exactly source1 and source2"
        )
    paths: dict[str, np.ndarray] = {}
    matrices_by_slot: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    frame_count: int | None = None
    for source_slot in SOURCE_SLOTS:
        binding = bindings[source_slot]
        if not isinstance(binding, AssetEmitterBinding):
            raise AssetEmitterBindingError(
                f"{source_slot} binding must be an AssetEmitterBinding"
            )
        roots = _points(
            source_root_paths_m[source_slot], owner=f"{source_slot} root path"
        )
        if frame_count is None:
            frame_count = len(roots)
        elif len(roots) != frame_count:
            raise AssetEmitterBindingError("source root paths must have equal length")
        fallback = _vector(
            source_fallback_forwards_xz[source_slot],
            2,
            owner=f"{source_slot} fallback forward",
        )
        if math.hypot(*fallback) <= 1.0e-12:
            raise AssetEmitterBindingError(
                f"{source_slot} fallback forward must be nonzero"
            )
        try:
            matrices = trajectory_world_matrices(
                roots,
                local_forward_axis=binding.local_anatomical_forward_axis,
                fallback_forward_xz=fallback,
            )
        except MixedCaptureError as exc:
            raise AssetEmitterBindingError(str(exc)) from exc
        offset = np.asarray(binding.emitter_offset_m, dtype=np.float64)
        emitter_path = np.einsum("nij,j->ni", matrices[:, :3, :3], offset)
        emitter_path += matrices[:, :3, 3]
        horizontal_offset = np.linalg.norm(
            emitter_path[:, (0, 2)] - roots[:, (0, 2)], axis=1
        )
        root_is_static = bool(
            np.allclose(roots[:, (0, 2)], roots[0, (0, 2)], rtol=0.0, atol=1.0e-12)
        )
        emitter_is_static = bool(
            np.allclose(
                emitter_path,
                emitter_path[0],
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        if root_is_static and not emitter_is_static:
            raise AssetEmitterBindingError(
                f"{source_slot} static root produced a moving emitter"
            )
        paths[source_slot] = np.ascontiguousarray(emitter_path)
        matrices_by_slot[source_slot] = matrices
        reports[source_slot] = {
            **binding.record(),
            "frame_count": len(roots),
            "minimum_world_height_m": float(np.min(emitter_path[:, 1])),
            "maximum_world_height_m": float(np.max(emitter_path[:, 1])),
            "minimum_horizontal_root_offset_m": float(np.min(horizontal_offset)),
            "maximum_horizontal_root_offset_m": float(np.max(horizontal_offset)),
            "root_motion": "static" if root_is_static else "moving",
            "emitter_motion": "static" if emitter_is_static else "moving",
        }
    return BoundEmitterPaths(
        paths_m=paths,
        actor_world_matrices=matrices_by_slot,
        report={
            "schema": ASSET_EMITTER_BINDING_REPORT_SCHEMA,
            "status": "pass",
            "method": "constant_asset_root_offset",
            "mouth_animation_required": False,
            "skeleton_lookup_required": False,
            "sources": reports,
        },
    )


def bind_asset_emitters_to_bank(
    bank: TrajectoryBank,
    bindings: Mapping[str, AssetEmitterBinding],
    *,
    listener_position_m: Sequence[float],
) -> tuple[TrajectoryBank, dict[str, Any]]:
    """Bind one concrete two-asset pairing to every route in a trajectory bank."""

    if not isinstance(bank, TrajectoryBank):
        raise AssetEmitterBindingError("bank must be a TrajectoryBank")
    listener = np.asarray(listener_position_m, dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise AssetEmitterBindingError("listener_position_m must be finite [3]")
    episodes: list[TrajectoryEpisode] = []
    height_ranges: dict[str, list[float]] = {
        source_slot: [math.inf, -math.inf] for source_slot in SOURCE_SLOTS
    }
    for episode in bank.episodes:
        root_paths = {
            source_slot: _points(
                episode.source_root_paths_m[source_slot],
                owner=f"{episode.episode_id} {source_slot} root path",
            )
            for source_slot in SOURCE_SLOTS
        }
        fallbacks = {
            source_slot: _fallback_toward_listener(root_paths[source_slot], listener)
            for source_slot in SOURCE_SLOTS
        }
        bound = materialize_asset_emitter_paths(
            bindings,
            source_root_paths_m=root_paths,
            source_fallback_forwards_xz=fallbacks,
        )
        statistics = dict(episode.statistics)
        statistics["asset_emitter_binding"] = {
            source_slot: dict(bound.report["sources"][source_slot])
            for source_slot in SOURCE_SLOTS
        }
        episodes.append(
            TrajectoryEpisode(
                episode_id=episode.episode_id,
                motion_case=episode.motion_case,
                source_root_paths_m=root_paths,
                source_center_paths_m=bound.paths_m,
                statistics=statistics,
            )
        )
        for source_slot in SOURCE_SLOTS:
            source_report = bound.report["sources"][source_slot]
            height_ranges[source_slot][0] = min(
                height_ranges[source_slot][0],
                float(source_report["minimum_world_height_m"]),
            )
            height_ranges[source_slot][1] = max(
                height_ranges[source_slot][1],
                float(source_report["maximum_world_height_m"]),
            )
    bound_bank = TrajectoryBank(
        episodes=tuple(episodes),
        frame_count=bank.frame_count,
        frame_rate_hz=bank.frame_rate_hz,
        seed=bank.seed,
    )
    return bound_bank, {
        "schema": ASSET_EMITTER_BINDING_REPORT_SCHEMA,
        "status": "pass",
        "method": "constant_asset_root_offset",
        "mouth_animation_required": False,
        "skeleton_lookup_required": False,
        "episode_count": len(episodes),
        "listener_position_m": listener.tolist(),
        "bindings": [bindings[source_slot].record() for source_slot in SOURCE_SLOTS],
        "world_height_range_m_by_source": {
            source_slot: height_ranges[source_slot] for source_slot in SOURCE_SLOTS
        },
    }


__all__ = [
    "ASSET_EMITTER_BINDING_REPORT_SCHEMA",
    "ASSET_EMITTER_BINDING_SET_SCHEMA",
    "AssetEmitterBinding",
    "AssetEmitterBindingError",
    "BoundEmitterPaths",
    "bind_asset_emitters_to_bank",
    "materialize_asset_emitter_paths",
    "validate_asset_emitter_binding_set",
]
