"""Generic two-point-source room evaluation planning.

The trajectory slots, dry sound classes and optional visible assets are
independent choices.  This module selects a balanced subset of a room's
existing source-center trajectories and assigns two distinct sound classes
without changing the geometry or pretending those classes are visual assets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m6x.room_feasibility import (
    RIR_JOB_PLAN_SCHEMA,
    SOURCE_SLOTS,
    TRAJECTORY_BANK_SCHEMA,
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)
from avengine.sensor_rig_trajectory import validate_sensor_rig_trajectory


ROOM_EVALUATION_PLAN_SCHEMA = "avengine_room_evaluation_plan_v1"
ROOM_SOUND_ASSIGNMENTS_SCHEMA = "avengine_room_sound_class_assignments_v1"
DEFAULT_SOUND_CLASSES = ("dog barking", "cat meowing", "human speech")
AZIMUTH_REGIONS = ("front", "right", "rear", "left")
_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class RoomEvaluationError(ValueError):
    """A trajectory bank cannot form the requested evaluation closure."""


@dataclass(frozen=True)
class RoomEvaluationPlan:
    trajectory_bank: Mapping[str, Any]
    rir_job_plan: Mapping[str, Any]
    sound_assignments: Mapping[str, Any]
    summary: Mapping[str, Any]


def build_static_source_trajectory_bank(
    source_positions_m: Mapping[str, Sequence[float]],
    *,
    frame_count: int,
    frame_rate_hz: int,
    episode_id: str = "static_sources_000",
    seed: int = 0,
) -> dict[str, Any]:
    """Materialize one two-source bank on an existing visual/sensor clock."""

    if (
        not isinstance(source_positions_m, Mapping)
        or set(source_positions_m) != set(SOURCE_SLOTS)
    ):
        raise RoomEvaluationError(
            "static source positions must contain exactly source1/source2"
        )
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
        or isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise RoomEvaluationError("static source trajectory clock or seed is invalid")
    normalized: dict[str, np.ndarray] = {}
    for slot in SOURCE_SLOTS:
        try:
            position = np.asarray(source_positions_m[slot], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RoomEvaluationError(
                f"static {slot} position must be finite xyz data"
            ) from exc
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise RoomEvaluationError(
                f"static {slot} position must be finite xyz data"
            )
        normalized[slot] = np.repeat(position[None, :], frame_count, axis=0)
    episode = TrajectoryEpisode(
        episode_id=validate_episode_id(episode_id),
        motion_case="static_static",
        source_root_paths_m=normalized,
        source_center_paths_m=normalized,
        statistics={
            "source_motion": "fixed_world_positions",
            "static_source_positions_m": {
                slot: normalized[slot][0].tolist() for slot in SOURCE_SLOTS
            },
        },
    )
    return TrajectoryBank(
        episodes=(episode,),
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        seed=seed,
    ).record(include_paths=True)


def validate_episode_id(value: Any) -> str:
    """Return one portable episode ID that is safe as a filename component."""

    if not isinstance(value, str) or not _EPISODE_ID.fullmatch(value):
        raise RoomEvaluationError(
            "episode_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,127}"
        )
    return value


def _points(value: Any, *, frame_count: int, owner: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RoomEvaluationError(f"{owner} must contain finite [frame,3] data") from exc
    if points.shape != (frame_count, 3) or not np.all(np.isfinite(points)):
        raise RoomEvaluationError(f"{owner} must contain finite [frame,3] data")
    return np.ascontiguousarray(points)


def _load_bank(value: Mapping[str, Any]) -> TrajectoryBank:
    if value.get("schema") != TRAJECTORY_BANK_SCHEMA:
        raise RoomEvaluationError(
            f"trajectory bank schema must be {TRAJECTORY_BANK_SCHEMA}"
        )
    frame_count = value.get("frame_count")
    frame_rate_hz = value.get("frame_rate_hz")
    seed = value.get("seed")
    raw_episodes = value.get("episodes")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
        or isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not isinstance(raw_episodes, list)
        or not raw_episodes
    ):
        raise RoomEvaluationError("trajectory bank header is invalid")
    episodes: list[TrajectoryEpisode] = []
    seen: set[str] = set()
    for ordinal, raw in enumerate(raw_episodes):
        if not isinstance(raw, Mapping):
            raise RoomEvaluationError(f"episodes[{ordinal}] must be an object")
        episode_id = validate_episode_id(raw.get("episode_id"))
        motion_case = raw.get("motion_case")
        if (
            episode_id in seen
            or not isinstance(motion_case, str)
            or not motion_case
        ):
            raise RoomEvaluationError("trajectory episode identity is invalid")
        raw_roots = raw.get("source_root_paths_m")
        raw_centers = raw.get("source_center_paths_m")
        if not isinstance(raw_roots, Mapping) or not isinstance(raw_centers, Mapping):
            raise RoomEvaluationError(f"{episode_id} lacks source paths")
        if set(raw_roots) != set(SOURCE_SLOTS) or set(raw_centers) != set(SOURCE_SLOTS):
            raise RoomEvaluationError(f"{episode_id} must contain source1/source2")
        roots = {
            slot: _points(
                raw_roots[slot], frame_count=frame_count, owner=f"{episode_id} {slot} root"
            )
            for slot in SOURCE_SLOTS
        }
        centers = {
            slot: _points(
                raw_centers[slot],
                frame_count=frame_count,
                owner=f"{episode_id} {slot} center",
            )
            for slot in SOURCE_SLOTS
        }
        statistics = raw.get("statistics", {})
        if not isinstance(statistics, Mapping):
            raise RoomEvaluationError(f"{episode_id} statistics are invalid")
        episodes.append(
            TrajectoryEpisode(
                episode_id=episode_id,
                motion_case=motion_case,
                source_root_paths_m=roots,
                source_center_paths_m=centers,
                statistics=dict(statistics),
            )
        )
        seen.add(episode_id)
    if value.get("episode_count") != len(episodes):
        raise RoomEvaluationError("trajectory bank episode_count differs")
    return TrajectoryBank(
        episodes=tuple(episodes),
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        seed=seed,
    )


def _listener_pose(
    position_m: Sequence[float], orientation_wxyz: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    try:
        position = np.asarray(position_m, dtype=np.float64)
        orientation = np.asarray(orientation_wxyz, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RoomEvaluationError("listener pose must be finite xyz/wxyz data") from exc
    if (
        position.shape != (3,)
        or orientation.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(orientation))
        or not math.isclose(
            float(np.linalg.norm(orientation)), 1.0, rel_tol=0.0, abs_tol=1.0e-9
        )
    ):
        raise RoomEvaluationError("listener pose must be finite and unit normalized")
    return position, orientation


def _listener_pose_frames(
    *,
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    frame_count: int,
    frame_rate_hz: int,
    sensor_rig_trajectory: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one authoritative listener pose for every visual frame."""

    fixed_position, fixed_orientation = _listener_pose(
        listener_position_m, listener_orientation_wxyz
    )
    if sensor_rig_trajectory is None:
        return (
            np.repeat(fixed_position[None, :], frame_count, axis=0),
            np.repeat(fixed_orientation[None, :], frame_count, axis=0),
        )
    errors = validate_sensor_rig_trajectory(sensor_rig_trajectory)
    if errors:
        raise RoomEvaluationError(
            "sensor rig trajectory is invalid: " + "; ".join(errors)
        )
    if (
        sensor_rig_trajectory.get("frame_count") != frame_count
        or sensor_rig_trajectory.get("frame_rate_hz") != frame_rate_hz
    ):
        raise RoomEvaluationError(
            "sensor rig trajectory clock differs from the source trajectory bank"
        )
    frames = sensor_rig_trajectory["frames"]
    positions = np.asarray(
        [
            frame["world_from_rig"]["translation_m"]
            for frame in frames
        ],
        dtype=np.float64,
    )
    rotations_xyzw = np.asarray(
        [
            frame["world_from_rig"]["rotation_xyzw"]
            for frame in frames
        ],
        dtype=np.float64,
    )
    orientations_wxyz = rotations_xyzw[:, (3, 0, 1, 2)]
    if (
        positions.shape != (frame_count, 3)
        or orientations_wxyz.shape != (frame_count, 4)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(orientations_wxyz))
    ):
        raise RoomEvaluationError("sensor rig trajectory has invalid pose arrays")
    same_initial_orientation = math.isclose(
        abs(float(np.dot(orientations_wxyz[0], fixed_orientation))),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    if (
        not np.allclose(
            positions[0], fixed_position, rtol=0.0, atol=1.0e-9
        )
        or not same_initial_orientation
    ):
        raise RoomEvaluationError(
            "fixed listener pose must equal SensorRigTrajectory frame 0"
        )
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(orientations_wxyz),
    )


def _episode_listener_distance_xz(
    episode: TrajectoryEpisode, listener_positions: np.ndarray
) -> float:
    values = np.concatenate(
        [
            episode.source_center_paths_m[slot][:, (0, 2)]
            - listener_positions[:, (0, 2)]
            for slot in SOURCE_SLOTS
        ],
        axis=0,
    )
    return float(np.min(np.linalg.norm(values, axis=1)))


def _episode_azimuth_region_counts(
    episode: TrajectoryEpisode,
    *,
    listener_positions: np.ndarray,
    listener_orientations: np.ndarray,
) -> np.ndarray:
    directions_by_slot = [
        episode.source_center_paths_m[slot] - listener_positions
        for slot in SOURCE_SLOTS
    ]
    directions = np.concatenate(directions_by_slot, axis=0)
    if np.any(np.linalg.norm(directions, axis=1) <= 0.0):
        raise RoomEvaluationError(
            f"{episode.episode_id} contains a source at the listener position"
        )
    inverse_vectors = np.concatenate(
        [-listener_orientations[:, 1:]] * len(SOURCE_SLOTS), axis=0
    )
    inverse_scalars = np.concatenate(
        [listener_orientations[:, 0]] * len(SOURCE_SLOTS), axis=0
    )
    uv = np.cross(inverse_vectors, directions)
    uuv = np.cross(inverse_vectors, uv)
    local = directions + 2.0 * (inverse_scalars[:, None] * uv + uuv)
    azimuth = np.degrees(np.arctan2(local[:, 0], -local[:, 2]))
    counts = np.zeros(len(AZIMUTH_REGIONS), dtype=np.int64)
    counts[0] = np.count_nonzero((azimuth >= -45.0) & (azimuth < 45.0))
    counts[1] = np.count_nonzero((azimuth >= 45.0) & (azimuth < 135.0))
    counts[2] = np.count_nonzero((azimuth >= 135.0) | (azimuth < -135.0))
    counts[3] = np.count_nonzero((azimuth >= -135.0) & (azimuth < -45.0))
    if int(np.sum(counts)) != directions.shape[0]:
        raise RoomEvaluationError("azimuth-region accounting is incomplete")
    return counts


def _balanced_episodes(
    bank: TrajectoryBank,
    count: int,
    *,
    listener_positions: np.ndarray,
    listener_orientations: np.ndarray,
    minimum_listener_source_distance_m: float,
    balance_azimuth_regions: bool,
) -> tuple[TrajectoryEpisode, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RoomEvaluationError("episode_count must be positive")
    by_motion: dict[str, list[TrajectoryEpisode]] = defaultdict(list)
    for episode in bank.episodes:
        by_motion[episode.motion_case].append(episode)
    motion_cases = sorted(by_motion)
    if not motion_cases:
        raise RoomEvaluationError("trajectory bank has no motion cases")
    base, remainder = divmod(count, len(motion_cases))
    selected_by_motion: dict[str, list[TrajectoryEpisode]] = {}
    counts: dict[str, int] = {}
    selected: list[TrajectoryEpisode] = []
    selected_region_counts = np.zeros(len(AZIMUTH_REGIONS), dtype=np.int64)
    for ordinal, motion_case in enumerate(motion_cases):
        needed = base + (1 if ordinal < remainder else 0)
        available = [
            item
            for item in sorted(by_motion[motion_case], key=lambda item: item.episode_id)
            if _episode_listener_distance_xz(item, listener_positions)
            >= minimum_listener_source_distance_m
        ]
        if len(available) < needed:
            raise RoomEvaluationError(
                f"motion case {motion_case} has {len(available)} episodes, needs {needed}"
            )
        if balance_azimuth_regions:
            chosen: list[TrajectoryEpisode] = []
            pool = list(available)
            for _ in range(needed):
                target_per_region = (
                    (len(selected) + 1)
                    * bank.frame_count
                    * len(SOURCE_SLOTS)
                    / len(AZIMUTH_REGIONS)
                )

                def score(item: TrajectoryEpisode) -> tuple[float, float, str]:
                    trial = selected_region_counts + _episode_azimuth_region_counts(
                        item,
                        listener_positions=listener_positions,
                        listener_orientations=listener_orientations,
                    )
                    deviations = np.abs(trial - target_per_region)
                    return (
                        float(np.sum(deviations)),
                        float(np.max(deviations)),
                        item.episode_id,
                    )

                choice = min(pool, key=score)
                pool.remove(choice)
                chosen.append(choice)
                selected.append(choice)
                selected_region_counts += _episode_azimuth_region_counts(
                    choice,
                    listener_positions=listener_positions,
                    listener_orientations=listener_orientations,
                )
            selected_by_motion[motion_case] = chosen
        else:
            selected_by_motion[motion_case] = available[:needed]
            selected.extend(selected_by_motion[motion_case])
        counts[motion_case] = needed
    # Interleave the already-selected per-motion sets so a small prefix remains
    # diverse without changing which episodes satisfied the global balance.
    interleaved = [
        selected_by_motion[motion_case][index]
        for index in range(max(counts.values()))
        for motion_case in motion_cases
        if index < len(selected_by_motion[motion_case])
    ]
    if len(interleaved) != count or len({item.episode_id for item in interleaved}) != count:
        raise RoomEvaluationError("balanced episode selection is invalid")
    return tuple(interleaved)


def _sound_pairs(classes: Sequence[str]) -> tuple[tuple[str, str], ...]:
    normalized = tuple(str(value).strip() for value in classes)
    if len(normalized) < 2 or any(not value for value in normalized):
        raise RoomEvaluationError("at least two nonempty sound classes are required")
    if len(set(normalized)) != len(normalized):
        raise RoomEvaluationError("sound classes must be unique")
    return tuple(
        (first, second)
        for first in normalized
        for second in normalized
        if first != second
    )


def build_room_evaluation_plan(
    trajectory_bank: Mapping[str, Any],
    *,
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    stride_frames: int,
    episode_count: int = 100,
    sound_classes: Sequence[str] = DEFAULT_SOUND_CLASSES,
    minimum_listener_source_distance_m: float = 0.0,
    balance_azimuth_regions: bool = False,
    minimum_azimuth_region_fraction: float = 0.0,
    sensor_rig_trajectory: Mapping[str, Any] | None = None,
) -> RoomEvaluationPlan:
    """Build a balanced, dry-audio-independent room evaluation plan."""

    if (
        not np.isfinite(minimum_listener_source_distance_m)
        or minimum_listener_source_distance_m < 0.0
        or not np.isfinite(minimum_azimuth_region_fraction)
        or not 0.0 <= minimum_azimuth_region_fraction <= 0.25
    ):
        raise RoomEvaluationError("listener distance or azimuth fraction is invalid")
    source_bank = _load_bank(trajectory_bank)
    listener_positions, listener_orientations = _listener_pose_frames(
        listener_position_m=listener_position_m,
        listener_orientation_wxyz=listener_orientation_wxyz,
        frame_count=source_bank.frame_count,
        frame_rate_hz=source_bank.frame_rate_hz,
        sensor_rig_trajectory=sensor_rig_trajectory,
    )
    listener_position = listener_positions[0]
    listener_orientation = listener_orientations[0]
    selected = _balanced_episodes(
        source_bank,
        episode_count,
        listener_positions=listener_positions,
        listener_orientations=listener_orientations,
        minimum_listener_source_distance_m=minimum_listener_source_distance_m,
        balance_azimuth_regions=balance_azimuth_regions,
    )
    bank = TrajectoryBank(
        episodes=selected,
        frame_count=source_bank.frame_count,
        frame_rate_hz=source_bank.frame_rate_hz,
        seed=source_bank.seed,
    )
    rir_kwargs: dict[str, Any] = {}
    if sensor_rig_trajectory is not None:
        rir_kwargs.update(
            {
                "listener_positions_m_by_episode": {
                    episode.episode_id: listener_positions.tolist()
                    for episode in selected
                },
                "listener_orientations_wxyz_by_episode": {
                    episode.episode_id: listener_orientations.tolist()
                    for episode in selected
                },
            }
        )
    rir_plan = build_rir_job_plan(
        bank,
        listener_position_m=listener_position,
        listener_orientation_wxyz=listener_orientation,
        stride_frames=stride_frames,
        **rir_kwargs,
    )
    if rir_plan.get("schema") != RIR_JOB_PLAN_SCHEMA:
        raise RoomEvaluationError("RIR planner returned an unexpected schema")
    pairs = _sound_pairs(sound_classes)
    assignments = []
    pair_counts: dict[str, int] = defaultdict(int)
    motion_cases = sorted({episode.motion_case for episode in selected})
    motion_ordinals = {motion_case: index for index, motion_case in enumerate(motion_cases)}
    within_motion_counts: dict[str, int] = defaultdict(int)
    motion_pair_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for episode in selected:
        within_motion = within_motion_counts[episode.motion_case]
        pair = pairs[(within_motion + motion_ordinals[episode.motion_case]) % len(pairs)]
        within_motion_counts[episode.motion_case] += 1
        pair_key = f"{pair[0]}|{pair[1]}"
        pair_counts[pair_key] += 1
        motion_pair_counts[episode.motion_case][pair_key] += 1
        assignments.append(
            {
                "episode_id": episode.episode_id,
                "source_classes": {
                    "source1": pair[0],
                    "source2": pair[1],
                },
            }
        )
    sound_record = {
        "schema": ROOM_SOUND_ASSIGNMENTS_SCHEMA,
        "status": "pass",
        "semantics": (
            "source1/source2 are generic point-source trajectory slots; sound "
            "classes are dry-audio/query bindings and are not visible asset IDs"
        ),
        "both_sources_active": True,
        "sound_classes": list(sound_classes),
        "episode_count": len(assignments),
        "ordered_pair_counts": dict(sorted(pair_counts.items())),
        "assignments": assignments,
    }
    motion_counts: dict[str, int] = defaultdict(int)
    azimuth_counts = np.zeros(len(AZIMUTH_REGIONS), dtype=np.int64)
    listener_distances = []
    for episode in selected:
        motion_counts[episode.motion_case] += 1
        azimuth_counts += _episode_azimuth_region_counts(
            episode,
            listener_positions=listener_positions,
            listener_orientations=listener_orientations,
        )
        listener_distances.append(
            _episode_listener_distance_xz(episode, listener_positions)
        )
    azimuth_total = int(np.sum(azimuth_counts))
    azimuth_fractions = azimuth_counts.astype(np.float64) / azimuth_total
    if balance_azimuth_regions and np.any(
        azimuth_fractions < minimum_azimuth_region_fraction
    ):
        detail = dict(zip(AZIMUTH_REGIONS, azimuth_fractions.tolist(), strict=True))
        raise RoomEvaluationError(
            "selected episodes do not satisfy the azimuth fraction gate: " f"{detail}"
        )
    summary = {
        "schema": ROOM_EVALUATION_PLAN_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "episode_count": len(selected),
        "frame_count": bank.frame_count,
        "frame_rate_hz": bank.frame_rate_hz,
        "motion_case_counts": dict(sorted(motion_counts.items())),
        "sound_pair_counts": dict(sorted(pair_counts.items())),
        "motion_sound_pair_counts": {
            motion_case: dict(sorted(values.items()))
            for motion_case, values in sorted(motion_pair_counts.items())
        },
        "listener_position_m": listener_position.tolist(),
        "listener_orientation_wxyz": listener_orientation.tolist(),
        "listener_pose_mode": (
            "sensor_rig_trajectory_v1"
            if sensor_rig_trajectory is not None
            else "fixed"
        ),
        "minimum_listener_source_distance_m_requested": minimum_listener_source_distance_m,
        "minimum_listener_source_distance_m_observed": min(listener_distances),
        "azimuth_balance_requested": balance_azimuth_regions,
        "minimum_azimuth_region_fraction_requested": minimum_azimuth_region_fraction,
        "azimuth_region_frame_counts": dict(
            zip(AZIMUTH_REGIONS, azimuth_counts.astype(int).tolist(), strict=True)
        ),
        "azimuth_region_frame_fractions": dict(
            zip(AZIMUTH_REGIONS, azimuth_fractions.tolist(), strict=True)
        ),
        "unique_rir_job_count": rir_plan["unique_rir_job_count"],
        "requested_pair_state_count": rir_plan["requested_pair_state_count"],
        "cache_reuse_count": rir_plan["cache_reuse_count"],
        "dry_audio_independent": True,
        "visual_asset_independent": True,
    }
    if sensor_rig_trajectory is not None:
        summary["sensor_rig_trajectory"] = {
            "trajectory_id": sensor_rig_trajectory["trajectory_id"],
            "content_sha256": canonical_json_sha256(
                sensor_rig_trajectory
            ),
            "relative_path": "sensor_rig_trajectory.json",
            "first_pose_hash": sensor_rig_trajectory["frames"][0][
                "pose_hash"
            ],
            "last_pose_hash": sensor_rig_trajectory["frames"][-1][
                "pose_hash"
            ],
        }
    return RoomEvaluationPlan(
        trajectory_bank=bank.record(include_paths=True),
        rir_job_plan=rir_plan,
        sound_assignments=sound_record,
        summary=summary,
    )


__all__ = [
    "AZIMUTH_REGIONS",
    "DEFAULT_SOUND_CLASSES",
    "ROOM_EVALUATION_PLAN_SCHEMA",
    "ROOM_SOUND_ASSIGNMENTS_SCHEMA",
    "RoomEvaluationError",
    "RoomEvaluationPlan",
    "build_room_evaluation_plan",
    "build_static_source_trajectory_bank",
    "validate_episode_id",
]
