"""Build scene-agnostic camera-framing inputs from sampled actor envelopes.

This adapter connects asset-local CPU skinning to the generic camera framing
solver.  It does not select a room, camera, or dataset revision, and it makes
no live-renderer or continuous-containment claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from avengine.actor_envelope import build_action_envelope, materialize_world_aabb
from avengine.assets.glb import load_glb
from avengine.assets.skinning import compile_skinning


ACTOR_FRAMING_INPUT_SCHEMA = "avengine_actor_framing_inputs_v1"
CANONICAL_WORLD_FRAME = "avengine_world_right_handed_y_up_m"


class ActorFramingError(ValueError):
    """Actor bindings or frame states cannot form strict framing inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActorFramingError(message)


def _mapping(value: object, *, owner: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{owner} must be an object")
    return value


def _sequence(value: object, *, owner: str) -> Sequence[Any]:
    _require(
        not isinstance(value, (str, bytes)) and isinstance(value, Sequence),
        f"{owner} must be an array",
    )
    return value


def _text(value: object, *, owner: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()), f"{owner} must be non-empty"
    )
    return value.strip()


def _finite(value: object, *, owner: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, Real),
        f"{owner} must be finite",
    )
    result = float(value)
    _require(math.isfinite(result), f"{owner} must be finite")
    return 0.0 if result == 0.0 else result


def _vec(value: object, length: int, *, owner: str) -> tuple[float, ...]:
    items = _sequence(value, owner=owner)
    _require(len(items) == length, f"{owner} must contain {length} values")
    return tuple(
        _finite(item, owner=f"{owner}[{index}]") for index, item in enumerate(items)
    )


def _scale(value: object, *, owner: str) -> tuple[float, float, float]:
    if not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        result = _vec(value, 3, owner=owner)
    else:
        scalar = _finite(value, owner=owner)
        result = (scalar, scalar, scalar)
    _require(all(component > 0.0 for component in result), f"{owner} must be positive")
    return result  # type: ignore[return-value]


def _world_from_asset(state: Mapping[str, Any], *, owner: str) -> list[list[float]]:
    translation = np.asarray(
        _vec(state.get("translation_m"), 3, owner=f"{owner}.translation_m"),
        dtype=np.float64,
    )
    quaternion = np.asarray(
        _vec(state.get("rotation_xyzw"), 4, owner=f"{owner}.rotation_xyzw"),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    _require(abs(norm - 1.0) <= 1.0e-5, f"{owner}.rotation_xyzw must be unit length")
    quaternion /= norm
    x, y, z, w = quaternion
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    scale = np.asarray(_scale(state.get("scale", 1.0), owner=f"{owner}.scale"))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix.tolist()


def _normalize_bindings(values: object) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for ordinal, raw in enumerate(_sequence(values, owner="actor_bindings")):
        value = _mapping(raw, owner=f"actor_bindings[{ordinal}]")
        actor_id = _text(
            value.get("actor_id"), owner=f"actor_bindings[{ordinal}].actor_id"
        )
        _require(actor_id not in result, f"duplicate actor binding: {actor_id}")
        path_value = _text(
            value.get("source_asset_path"),
            owner=f"actor binding {actor_id}.source_asset_path",
        )
        try:
            source_path = Path(path_value).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ActorFramingError(
                f"actor binding {actor_id} source asset does not exist"
            ) from error
        _require(
            source_path.is_file(),
            f"actor binding {actor_id} source asset is not a regular file",
        )
        raw_skin_index = value.get("skin_index", 0)
        _require(
            not isinstance(raw_skin_index, bool)
            and isinstance(raw_skin_index, Integral)
            and int(raw_skin_index) >= 0,
            f"actor binding {actor_id}.skin_index must be nonnegative",
        )
        action_map = _mapping(
            value.get("action_name_by_action_id"),
            owner=f"actor binding {actor_id}.action_name_by_action_id",
        )
        normalized_action_map = {
            _text(action_id, owner=f"actor binding {actor_id} action ID"): _text(
                action_name,
                owner=f"actor binding {actor_id} action name",
            )
            for action_id, action_name in action_map.items()
        }
        _require(normalized_action_map, f"actor binding {actor_id} action map is empty")
        chain = _mapping(
            value.get("coordinate_chain"),
            owner=f"actor binding {actor_id}.coordinate_chain",
        )
        from_frame = _text(
            chain.get("from_frame"), owner=f"actor binding {actor_id} from_frame"
        )
        to_frame = _text(
            chain.get("to_frame"), owner=f"actor binding {actor_id} to_frame"
        )
        _require(
            to_frame == CANONICAL_WORLD_FRAME,
            f"actor binding {actor_id} must target canonical world",
        )
        operations = [
            _text(item, owner=f"actor binding {actor_id} coordinate operation")
            for item in _sequence(
                chain.get("operations"), owner=f"actor binding {actor_id} operations"
            )
        ]
        _require(operations, f"actor binding {actor_id} operations are empty")
        result[actor_id] = {
            "actor_id": actor_id,
            "asset_id": _text(
                value.get("asset_id"), owner=f"actor binding {actor_id}.asset_id"
            ),
            "asset_revision": _text(
                value.get("asset_revision"),
                owner=f"actor binding {actor_id}.asset_revision",
            ),
            "source_asset_path": str(source_path),
            "skin_index": int(raw_skin_index),
            "action_name_by_action_id": normalized_action_map,
            "coordinate_chain": {
                "from_frame": from_frame,
                "to_frame": to_frame,
                "operations": operations,
            },
        }
    _require(result, "actor_bindings cannot be empty")
    return result


def _normalize_frames(
    values: object,
    *,
    actor_ids: set[str],
    expected_frame_count: int | None,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for ordinal, raw_frame in enumerate(_sequence(values, owner="frame_states")):
        frame = _mapping(raw_frame, owner=f"frame_states[{ordinal}]")
        raw_index = frame.get("frame_index")
        _require(
            not isinstance(raw_index, bool) and isinstance(raw_index, Integral),
            f"frame_states[{ordinal}].frame_index must be an integer",
        )
        frame_index = int(raw_index)
        _require(
            frame_index not in seen_indices, f"duplicate frame_index: {frame_index}"
        )
        seen_indices.add(frame_index)
        states: dict[str, dict[str, Any]] = {}
        for state_ordinal, raw_state in enumerate(
            _sequence(
                frame.get("actor_states"), owner=f"frame {frame_index}.actor_states"
            )
        ):
            state = _mapping(
                raw_state, owner=f"frame {frame_index} actor state {state_ordinal}"
            )
            actor_id = _text(
                state.get("actor_id"), owner=f"frame {frame_index} actor_id"
            )
            _require(
                actor_id in actor_ids,
                f"frame {frame_index} contains unknown actor {actor_id}",
            )
            _require(
                actor_id not in states,
                f"frame {frame_index} duplicates actor {actor_id}",
            )
            states[actor_id] = {
                "actor_id": actor_id,
                "action_id": _text(
                    state.get("action_id"),
                    owner=f"frame {frame_index} {actor_id}.action_id",
                ),
                "translation_m": list(
                    _vec(
                        state.get("translation_m"),
                        3,
                        owner=f"frame {frame_index} {actor_id}.translation_m",
                    )
                ),
                "rotation_xyzw": list(
                    _vec(
                        state.get("rotation_xyzw"),
                        4,
                        owner=f"frame {frame_index} {actor_id}.rotation_xyzw",
                    )
                ),
                "scale": list(
                    _scale(
                        state.get("scale", 1.0),
                        owner=f"frame {frame_index} {actor_id}.scale",
                    )
                ),
            }
        _require(
            set(states) == actor_ids,
            f"frame {frame_index} must contain every bound actor exactly once",
        )
        frames.append({"frame_index": frame_index, "actor_states": states})
    _require(frames, "frame_states cannot be empty")
    frames.sort(key=lambda item: item["frame_index"])
    indices = [item["frame_index"] for item in frames]
    _require(
        indices == list(range(len(frames))),
        "frame indices must be contiguous and start at zero",
    )
    if expected_frame_count is not None:
        _require(
            not isinstance(expected_frame_count, bool)
            and isinstance(expected_frame_count, Integral)
            and int(expected_frame_count) > 0,
            "expected_frame_count must be positive",
        )
        _require(
            len(frames) == int(expected_frame_count),
            "frame count differs from expected_frame_count",
        )
    return frames


def build_actor_framing_frames(
    *,
    actor_bindings: object,
    frame_states: object,
    sample_rate_hz: float = 120.0,
    padding_m: float = 0.02,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    """Build strict world AABBs consumable by :mod:`avengine.camera_framing`."""

    bindings = _normalize_bindings(actor_bindings)
    frames = _normalize_frames(
        frame_states,
        actor_ids=set(bindings),
        expected_frame_count=expected_frame_count,
    )
    coverage: dict[tuple[str, str], list[int]] = {}
    for frame in frames:
        for actor_id, state in frame["actor_states"].items():
            action_id = state["action_id"]
            _require(
                action_id in bindings[actor_id]["action_name_by_action_id"],
                f"actor {actor_id} action {action_id!r} has no source action binding",
            )
            coverage.setdefault((actor_id, action_id), []).append(frame["frame_index"])

    envelopes: dict[tuple[str, str], Any] = {}
    envelope_records: dict[str, dict[str, Any]] = {}
    for actor_id, binding in bindings.items():
        compiled = compile_skinning(
            load_glb(binding["source_asset_path"]), skin_index=binding["skin_index"]
        )
        per_action: dict[str, Any] = {}
        for bound_actor_id, action_id in sorted(coverage):
            if bound_actor_id != actor_id:
                continue
            action_name = binding["action_name_by_action_id"][action_id]
            envelope = build_action_envelope(
                compiled,
                action_name,
                sample_rate_hz=sample_rate_hz,
                padding_m=padding_m,
                source_asset_path=binding["source_asset_path"],
            )
            envelopes[(actor_id, action_id)] = envelope
            per_action[action_id] = {
                "source_action_name": action_name,
                "covered_frame_indices": list(coverage[(actor_id, action_id)]),
                "envelope": envelope.to_manifest_record(),
            }
        envelope_records[actor_id] = {
            "actor_id": actor_id,
            "asset_id": binding["asset_id"],
            "asset_revision": binding["asset_revision"],
            "source_asset_path": binding["source_asset_path"],
            "skin_index": binding["skin_index"],
            "actions": per_action,
        }

    framing_frames: list[dict[str, Any]] = []
    for frame in frames:
        frame_index = frame["frame_index"]
        actor_aabbs: dict[str, Any] = {}
        for actor_id, state in frame["actor_states"].items():
            binding = bindings[actor_id]
            action_id = state["action_id"]
            envelope = envelopes[(actor_id, action_id)]
            world = materialize_world_aabb(
                envelope.padded_bounds,
                _world_from_asset(state, owner=f"frame {frame_index} actor {actor_id}"),
            )
            authority_stem = f"{actor_id}/{binding['asset_id']}/{binding['asset_revision']}/{action_id}"
            actor_aabbs[actor_id] = {
                "minimum_m": list(world.bounds.minimum),
                "maximum_m": list(world.bounds.maximum),
                "action_id": action_id,
                "bounds_authority": {
                    "status": "pass",
                    "authority_id": f"{authority_stem}/sampled-envelope-v2",
                    "source": "sampled_cpu_skinning_envelope",
                    "asset_id": binding["asset_id"],
                    "revision_id": binding["asset_revision"],
                    "action_scope": action_id,
                    "source_action_name": binding["action_name_by_action_id"][
                        action_id
                    ],
                    "source_asset_path": binding["source_asset_path"],
                    "continuous_containment_claim": False,
                    "live_renderer_bounds_pending": True,
                },
                "coordinate_chain": {
                    "status": "pass",
                    "authority_id": f"{authority_stem}/coordinate-chain",
                    **deepcopy(binding["coordinate_chain"]),
                },
                "action_coverage": {
                    "status": "pass",
                    "authority_id": f"{authority_stem}/frame-coverage",
                    "action_id": action_id,
                    "covered_frame_indices": list(coverage[(actor_id, action_id)]),
                },
            }
        framing_frames.append({"frame_index": frame_index, "actor_aabbs": actor_aabbs})

    return {
        "schema": ACTOR_FRAMING_INPUT_SCHEMA,
        "status": "pass_cpu_sampled_planning_envelopes",
        "frame_count": len(framing_frames),
        "actor_ids": sorted(bindings),
        "actor_envelopes": envelope_records,
        "frames": framing_frames,
        "qualification": {
            "state": "planning_only",
            "qualification_claim": False,
            "formal_episode_count": 0,
            "native_ue_bounds_pending": True,
            "native_pixel_framing_pending": True,
            "continuous_containment_claim": False,
        },
    }
