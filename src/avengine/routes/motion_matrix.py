"""Compact four-way human/animal motion matrix for Apartment pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


EPISODE_FRAME_COUNT = 75


class MotionMatrixError(ValueError):
    """The requested four-way route matrix cannot be constructed safely."""


@dataclass(frozen=True)
class MotionEpisode:
    episode_id: str
    start_frame: int
    end_frame_exclusive: int
    human_motion: str
    dog_motion: str


@dataclass(frozen=True)
class FourMotionMaster:
    actor_root_paths: Mapping[str, np.ndarray]
    episodes: tuple[MotionEpisode, ...]

    @property
    def frame_count(self) -> int:
        return sum(
            episode.end_frame_exclusive - episode.start_frame
            for episode in self.episodes
        )


def _anchor_positions(anchor_library: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for record in anchor_library.get("anchors", ()):
        if not isinstance(record, Mapping):
            continue
        anchor_id = record.get("anchor_id")
        try:
            position = np.asarray(record.get("position_m"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            isinstance(anchor_id, str)
            and anchor_id
            and anchor_id not in result
            and position.shape == (3,)
            and np.all(np.isfinite(position))
        ):
            result[anchor_id] = position
    return result


def _hold(point: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.repeat(point[None, :], EPISODE_FRAME_COUNT, axis=0)
    )


def _piecewise(
    points: tuple[np.ndarray, ...], frame_indices: tuple[int, ...]
) -> np.ndarray:
    if (
        len(points) != len(frame_indices)
        or len(points) < 2
        or frame_indices[0] != 0
        or frame_indices[-1] != EPISODE_FRAME_COUNT - 1
        or tuple(sorted(set(frame_indices))) != frame_indices
    ):
        raise MotionMatrixError("piecewise episode anchors are invalid")
    output = np.empty((EPISODE_FRAME_COUNT, 3), dtype=np.float64)
    for segment in range(len(points) - 1):
        start = frame_indices[segment]
        end = frame_indices[segment + 1]
        alpha = np.linspace(0.0, 1.0, end - start + 1, dtype=np.float64)
        output[start : end + 1] = (
            points[segment][None, :] * (1.0 - alpha[:, None])
            + points[segment + 1][None, :] * alpha[:, None]
        )
    return np.ascontiguousarray(output)


def build_four_motion_master(
    anchor_library: Mapping[str, Any],
) -> FourMotionMaster:
    """Build four position-continuous 5-second human/Beagle episodes.

    Execution order is static/static, human-only, both-moving, then dog-only.
    That order makes every 75-frame boundary position-continuous while still
    yielding the four requested motion labels as independent episodes.
    """

    anchors = _anchor_positions(anchor_library)
    required = {
        "human_front_left_start",
        "human_front_left_midpoint",
        "human_front_left_end",
        "dog_front_right_start",
        "dog_front_right_end",
    }
    missing = sorted(required.difference(anchors))
    if missing:
        raise MotionMatrixError(f"Apartment motion anchors are missing: {missing}")

    human_start = anchors["human_front_left_start"]
    human_midpoint = anchors["human_front_left_midpoint"]
    human_end = anchors["human_front_left_end"]
    dog_start = anchors["dog_front_right_start"]
    dog_end = anchors["dog_front_right_end"]

    human_forward = _piecewise(
        (human_start, human_midpoint, human_end), (0, 16, 74)
    )
    human_reverse = _piecewise(
        (human_end, human_midpoint, human_start), (0, 58, 74)
    )
    dog_forward = _piecewise((dog_start, dog_end), (0, 74))
    dog_reverse = _piecewise((dog_end, dog_start), (0, 74))

    human_episodes = (
        _hold(human_start),
        human_forward,
        human_reverse,
        _hold(human_start),
    )
    dog_episodes = (
        _hold(dog_start),
        _hold(dog_start),
        dog_forward,
        dog_reverse,
    )
    episode_specs = (
        ("static_static", "static", "static"),
        ("human_moving_dog_static", "moving", "static"),
        ("both_moving", "moving", "moving"),
        ("human_static_dog_moving", "static", "moving"),
    )
    episodes = tuple(
        MotionEpisode(
            episode_id=episode_id,
            start_frame=index * EPISODE_FRAME_COUNT,
            end_frame_exclusive=(index + 1) * EPISODE_FRAME_COUNT,
            human_motion=human_motion,
            dog_motion=dog_motion,
        )
        for index, (episode_id, human_motion, dog_motion) in enumerate(episode_specs)
    )
    human = np.ascontiguousarray(np.concatenate(human_episodes, axis=0))
    dog = np.ascontiguousarray(np.concatenate(dog_episodes, axis=0))
    for boundary in range(1, len(episodes)):
        index = boundary * EPISODE_FRAME_COUNT
        if not np.array_equal(human[index - 1], human[index]) or not np.array_equal(
            dog[index - 1], dog[index]
        ):
            raise MotionMatrixError("episode boundary introduced an actor teleport")
    return FourMotionMaster(
        actor_root_paths={"dog0": dog, "human0": human},
        episodes=episodes,
    )


def motion_matrix_record(master: FourMotionMaster) -> dict[str, Any]:
    """Return compact JSON metadata without duplicating dense trajectories."""

    return {
        "schema": "avengine_apartment_four_motion_matrix_v1",
        "frame_count": master.frame_count,
        "episode_frame_count": EPISODE_FRAME_COUNT,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "start_frame": episode.start_frame,
                "end_frame_exclusive": episode.end_frame_exclusive,
                "motion": {
                    "human0": episode.human_motion,
                    "dog0": episode.dog_motion,
                },
            }
            for episode in master.episodes
        ],
    }


__all__ = [
    "EPISODE_FRAME_COUNT",
    "FourMotionMaster",
    "MotionEpisode",
    "MotionMatrixError",
    "build_four_motion_master",
    "motion_matrix_record",
]
