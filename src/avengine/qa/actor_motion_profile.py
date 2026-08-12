"""CPU-only, source-bound actor motion profiles.

The profile is intentionally data driven: it binds a proposed candidate, the
selected row that preceded it, and the materialized base suite without knowing
anything about a particular room or mechanism name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.m6x.room_feasibility import (
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)

PROFILE_SCHEMA = "avengine_actor_motion_profile_v1"
FRAME_SCHEMA = "avengine_actor_motion_profile_frame_v1"


class ActorMotionProfileError(ValueError):
    """A profile or one of its immutable authorities is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActorMotionProfileError(message)


def _source_binding(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    json_pointer: str,
) -> dict[str, Any]:
    source_path = Path(path).resolve()
    _require(source_path.is_file(), f"authority is not a file: {source_path}")
    return {
        "path": str(source_path),
        "document_sha256": sha256_file(source_path),
        "json_pointer": json_pointer,
        "canonical_value_sha256": canonical_json_sha256(value),
        "value": deepcopy(dict(value)),
    }


def _candidate(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    authorities = profile.get("authorities")
    _require(isinstance(authorities, Mapping), "authorities are missing")
    binding = authorities.get("candidate")
    _require(isinstance(binding, Mapping), "candidate authority is missing")
    value = binding.get("value")
    _require(isinstance(value, Mapping), "candidate value is missing")
    return value


def materialize_profile_frames(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return canonical, hash-bound frames from a validated profile."""

    candidate = _candidate(profile)
    frames = candidate.get("frames")
    _require(isinstance(frames, list) and bool(frames), "candidate frames are missing")
    result: list[dict[str, Any]] = []
    for frame_index, source in enumerate(frames):
        _require(isinstance(source, Mapping), f"frame {frame_index} is not an object")
        _require(
            source.get("frame_index") == frame_index, "frame indices are not exact"
        )
        core = {
            "schema": FRAME_SCHEMA,
            "frame_index": frame_index,
            "pts_ticks": source.get("pts_ticks"),
            "actor_states": deepcopy(source.get("actor_states")),
        }
        result.append({**core, "canonical_frame_sha256": canonical_json_sha256(core)})
    return result


def source_center_paths(profile: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    """Derive emitter centers by adding declared offsets to candidate roots."""

    candidate = _candidate(profile)
    declarations = candidate.get("actor_declarations")
    actors = candidate.get("actors")
    _require(isinstance(declarations, Mapping), "actor declarations are missing")
    _require(isinstance(actors, Mapping) and bool(actors), "actors are missing")
    result: dict[str, list[list[float]]] = {}
    for slot, actor in sorted(actors.items()):
        _require(isinstance(actor, Mapping), f"actor {slot!r} is invalid")
        actor_id = actor.get("actor_id")
        declaration = declarations.get(actor_id)
        _require(
            isinstance(declaration, Mapping), f"declaration for {actor_id!r} is missing"
        )
        offset = declaration.get("emitter_offset_m")
        roots = actor.get("root_path_m")
        _require(
            isinstance(offset, Sequence)
            and len(offset) == 3
            and isinstance(roots, Sequence),
            f"source-center inputs for {slot!r} are invalid",
        )
        result[str(slot)] = [
            [float(root[axis]) + float(offset[axis]) for axis in range(3)]
            for root in roots
        ]
    return result


def _rir_expectation(profile: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _candidate(profile)
    frame_count = int(candidate["frame_count"])
    episode_id = str(candidate["candidate_episode_id"])
    centers = source_center_paths(profile)
    roots = {
        str(slot): np.asarray(actor["root_path_m"], dtype=np.float64)
        for slot, actor in candidate["actors"].items()
    }
    episode = TrajectoryEpisode(
        episode_id=episode_id,
        motion_case=str(candidate["mechanism"]),
        source_root_paths_m=roots,
        source_center_paths_m={
            slot: np.asarray(path, dtype=np.float64) for slot, path in centers.items()
        },
        statistics={},
    )
    bank = TrajectoryBank(
        episodes=(episode,),
        frame_count=frame_count,
        frame_rate_hz=int(candidate["frame_rate_hz"]),
        seed=0,
    )
    base_suite = profile["authorities"]["base_suite"]["value"]
    scenario = base_suite["scenarios"][0]
    frames = scenario["plan"]["frames"]
    positions = [frame["camera_state"]["habitat_position_m"] for frame in frames]
    orientations = []
    for frame in frames:
        xyzw = frame["camera_state"]["world_from_rig"]["rotation_xyzw"]
        orientations.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    plan = build_rir_job_plan(
        bank,
        listener_positions_m_by_episode={episode_id: positions},
        listener_orientations_wxyz_by_episode={episode_id: orientations},
        stride_frames=1,
    )
    return {
        "builder": "avengine.m6x.room_feasibility.build_rir_job_plan",
        "stride_frames": plan["stride_frames"],
        "requested_pair_state_count": plan["requested_pair_state_count"],
        "unique_rir_job_count": plan["unique_rir_job_count"],
        "canonical_plan_sha256": canonical_json_sha256(plan),
    }


def build_actor_motion_profile(
    *,
    candidate_path: str | Path,
    candidate: Mapping[str, Any],
    old_preflight_path: str | Path,
    selected_old_row: Mapping[str, Any],
    base_suite_path: str | Path,
    base_suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a generic immutable profile from three supplied authorities."""

    core: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "status": "pass_cpu_bound_actor_motion_profile",
        "qualification_claim": False,
        "formal_episode_count": 0,
        "authorities": {
            "candidate": _source_binding(candidate_path, candidate, json_pointer=""),
            "selected_old_row": _source_binding(
                old_preflight_path, selected_old_row, json_pointer="/canaries/0"
            ),
            "base_suite": _source_binding(base_suite_path, base_suite, json_pointer=""),
        },
    }
    core["frames"] = materialize_profile_frames(core)
    core["rir_expectation"] = _rir_expectation(core)
    profile = {**core, "profile_content_sha256": canonical_json_sha256(core)}
    validate_actor_motion_profile(profile)
    return profile


def validate_actor_motion_profile(profile: Mapping[str, Any]) -> None:
    """Fail closed on source drift and basic frame/profile inconsistency."""

    _require(profile.get("schema") == PROFILE_SCHEMA, "profile schema is invalid")
    _require(
        profile.get("qualification_claim") is False, "qualification claim is forbidden"
    )
    _require(
        profile.get("formal_episode_count") == 0, "formal episode count must be zero"
    )
    core = dict(profile)
    declared_hash = core.pop("profile_content_sha256", None)
    _require(
        declared_hash == canonical_json_sha256(core), "profile content hash mismatch"
    )
    authorities = profile.get("authorities")
    _require(isinstance(authorities, Mapping), "authorities are missing")
    for name, pointer in (
        ("candidate", ""),
        ("selected_old_row", "/canaries/0"),
        ("base_suite", ""),
    ):
        binding = authorities.get(name)
        _require(isinstance(binding, Mapping), f"{name} authority is missing")
        path = Path(str(binding.get("path", "")))
        _require(path.is_file(), f"{name} authority file is missing")
        _require(
            binding.get("document_sha256") == sha256_file(path),
            f"{name} file hash drift",
        )
        document = load_json(path)
        actual = document if pointer == "" else document["canaries"][0]
        _require(binding.get("json_pointer") == pointer, f"{name} JSON pointer drift")
        _require(binding.get("value") == actual, f"{name} bound value drift")
        _require(
            binding.get("canonical_value_sha256") == canonical_json_sha256(actual),
            f"{name} value hash drift",
        )
    _require(
        profile.get("frames") == materialize_profile_frames(profile),
        "frame materialization drift",
    )
    _require(
        profile.get("rir_expectation") == _rir_expectation(profile),
        "RIR expectation drift",
    )
