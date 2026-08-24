"""Pure two-source motion geometry and explicit per-frame track descriptors.

Difficulty-profile helpers cover the historical side-margin and exact linear
continuous-separation measurements without deciding candidate legality. The
explicit-track entrypoint consumes caller-supplied frames without interpolating
them. Neither path claims actor visibility, native pixels, collision volumes,
or a successful Timeline/native run unless separate evidence establishes it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


MOTION_MODES = (
    "s1_moving_s2_static",
    "s1_static_s2_moving",
    "both_moving",
    "both_static",
)
DEFAULT_FIRST_ANCHOR_FRACTION = 4.0 / 75.0
DEFAULT_SECOND_ANCHOR_FRACTION = 22.5 / 75.0


class MotionFeasibilityError(ValueError):
    """Motion geometry inputs are invalid or name an unsupported mode."""


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MotionFeasibilityError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MotionFeasibilityError(f"{owner} must be a finite number")
    return result


def _fraction(value: Any, *, owner: str) -> float:
    result = _finite_number(value, owner=owner)
    if not 0.0 <= result <= 1.0:
        raise MotionFeasibilityError(f"{owner} must lie in [0, 1]")
    return result


def _point(value: Sequence[Real], *, owner: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MotionFeasibilityError(f"{owner} must contain at least two numbers")
    if len(value) < 2:
        raise MotionFeasibilityError(f"{owner} must contain at least two numbers")
    return tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )


def _path(
    value: Sequence[Sequence[Real]], *, owner: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise MotionFeasibilityError(f"{owner} must contain start/end points")
    start = _point(value[0], owner=f"{owner}.start")
    end = _point(value[1], owner=f"{owner}.end")
    if len(start) != len(end):
        raise MotionFeasibilityError(f"{owner} endpoints must have equal dimensions")
    return start, end


def _rotated_indices(count: int, offset: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    return tuple((offset + index) % count for index in range(count))


def interpolate_linear_path(
    path: Sequence[Sequence[Real]], fraction: Real
) -> tuple[float, ...]:
    """Interpolate a same-dimensional line segment at an explicit fraction."""

    start, end = _path(path, owner="path")
    interpolation_fraction = _fraction(fraction, owner="path fraction")
    return tuple(
        left + (right - left) * interpolation_fraction
        for left, right in zip(start, end, strict=True)
    )


def camera_side(
    camera_position_xy: Sequence[Real],
    camera_yaw_degrees: Real,
    point_xy: Sequence[Real],
) -> tuple[str, float]:
    """Return left/right and the normalized lateral cross-product margin."""

    camera = _point(camera_position_xy, owner="camera position")
    point = _point(point_xy, owner="side point")
    yaw = math.radians(_finite_number(camera_yaw_degrees, owner="camera yaw"))
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    delta_x = point[0] - camera[0]
    delta_y = point[1] - camera[1]
    cross = (forward_x * delta_y - forward_y * delta_x) / max(
        math.hypot(delta_x, delta_y), 1.0e-9
    )
    return ("right" if cross > 0.0 else "left"), abs(cross)


def minimum_linear_path_separation(
    path1: Sequence[Sequence[Real]], path2: Sequence[Sequence[Real]]
) -> float:
    """Return exact continuous XY separation for synchronized linear paths."""

    return linear_path_closest_approach(path1, path2)[0]


def linear_path_closest_approach(
    path1: Sequence[Sequence[Real]], path2: Sequence[Sequence[Real]]
) -> tuple[float, float]:
    """Return exact continuous XY separation and closest fraction in [0, 1]."""

    first = _path(path1, owner="source1 path")
    second = _path(path2, owner="source2 path")
    relative_start = (
        first[0][0] - second[0][0],
        first[0][1] - second[0][1],
    )
    relative_velocity = (
        (first[1][0] - first[0][0]) - (second[1][0] - second[0][0]),
        (first[1][1] - first[0][1]) - (second[1][1] - second[0][1]),
    )
    velocity_squared = sum(value * value for value in relative_velocity)
    if velocity_squared <= 1.0e-18:
        fraction = 0.0
    else:
        fraction = max(
            0.0,
            min(
                1.0,
                -sum(
                    start * velocity
                    for start, velocity in zip(
                        relative_start, relative_velocity, strict=True
                    )
                )
                / velocity_squared,
            ),
        )
    closest = tuple(
        start + fraction * velocity
        for start, velocity in zip(relative_start, relative_velocity, strict=True)
    )
    return math.hypot(*closest), fraction


@dataclass(frozen=True)
class MotionPairEvidence:
    source1_path: tuple[tuple[float, ...], tuple[float, ...]]
    source2_path: tuple[tuple[float, ...], tuple[float, ...]]
    first_anchor_fraction: float
    second_anchor_fraction: float
    first_anchor_sides: tuple[str, str]
    first_anchor_margins: tuple[float, float]
    second_anchor_margins: tuple[float, float]
    minimum_continuous_separation: float
    side_gate_passed: bool
    separation_gate_passed: bool
    passed: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "source1_path": [list(point) for point in self.source1_path],
            "source2_path": [list(point) for point in self.source2_path],
            "first_anchor_fraction": self.first_anchor_fraction,
            "second_anchor_fraction": self.second_anchor_fraction,
            "first_anchor_sides": list(self.first_anchor_sides),
            "first_anchor_margins": list(self.first_anchor_margins),
            "second_anchor_margins": list(self.second_anchor_margins),
            "minimum_continuous_separation": self.minimum_continuous_separation,
            "side_gate_passed": self.side_gate_passed,
            "separation_gate_passed": self.separation_gate_passed,
            "passed": self.passed,
            "difficulty_profile_match": self.passed,
            "candidate_validity_effect": "descriptor_only_never_filters_candidates",
        }


@dataclass(frozen=True)
class MotionModeEvidence:
    mode: str
    feasible: bool
    attempted_pair_count: int
    side_gate_pass_pair_count: int
    separation_gate_pass_pair_count: int
    feasible_pair_count: int
    representative_pair: MotionPairEvidence | None
    selected_pair: MotionPairEvidence | None
    best_failed_pair: MotionPairEvidence | None
    failure_reason: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feasible": self.feasible,
            "attempted_pair_count": self.attempted_pair_count,
            "side_gate_pass_pair_count": self.side_gate_pass_pair_count,
            "separation_gate_pass_pair_count": self.separation_gate_pass_pair_count,
            "feasible_pair_count": self.feasible_pair_count,
            "difficulty_profile_match": self.feasible,
            "candidate_validity_effect": "descriptor_only_never_filters_candidates",
            "candidate_pair_search_status": (
                "stopped_after_first_feasible_pair"
                if self.feasible
                else "complete_no_feasible_pair"
            ),
            "selected_pair": (
                None if self.selected_pair is None else self.selected_pair.to_record()
            ),
            "representative_pair": (
                None
                if self.representative_pair is None
                else self.representative_pair.to_record()
            ),
            "best_failed_pair": (
                None
                if self.best_failed_pair is None
                else self.best_failed_pair.to_record()
            ),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class MotionFeasibilityClassification:
    modes: tuple[MotionModeEvidence, ...]
    thresholds: Mapping[str, float]

    @property
    def feasible_modes(self) -> tuple[str, ...]:
        return tuple(item.mode for item in self.modes if item.feasible)

    @property
    def feasible_mode_count(self) -> int:
        return len(self.feasible_modes)

    @property
    def all_modes_feasible(self) -> bool:
        return self.feasible_mode_count == len(self.modes)

    @property
    def feasible_pair_count(self) -> int:
        return sum(item.feasible_pair_count for item in self.modes)

    @property
    def minimum_selected_separation(self) -> float:
        values = [
            item.selected_pair.minimum_continuous_separation
            for item in self.modes
            if item.selected_pair is not None
        ]
        return min(values) if values else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "coordinate_space": "caller_declared_xy",
            "claim_boundary": (
                "difficulty descriptors over source root/anchor points only; side "
                "margins and continuous XY separation never decide candidate legality; "
                "not native pixels, actor collision volumes, or Timeline execution"
            ),
            "candidate_filtering": "disabled",
            "legacy_feasible_field_semantics": "difficulty_profile_match_only",
            "thresholds": dict(self.thresholds),
            "feasible_modes": list(self.feasible_modes),
            "difficulty_profile_match_modes": list(self.feasible_modes),
            "feasible_mode_count": self.feasible_mode_count,
            "all_modes_feasible": self.all_modes_feasible,
            "feasible_pair_count": self.feasible_pair_count,
            "minimum_selected_separation": self.minimum_selected_separation,
            "modes": {item.mode: item.to_record() for item in self.modes},
        }


def evaluate_motion_pair(
    *,
    camera_position_xy: Sequence[Real],
    camera_yaw_degrees: Real,
    source1_path: Sequence[Sequence[Real]],
    source2_path: Sequence[Sequence[Real]],
    first_anchor_fraction: Real = DEFAULT_FIRST_ANCHOR_FRACTION,
    second_anchor_fraction: Real = DEFAULT_SECOND_ANCHOR_FRACTION,
    minimum_first_side_margin: Real = 0.12,
    minimum_second_side_margin: Real = 0.06,
    minimum_continuous_separation: Real = 60.0,
    allow_any_side: bool = False,
) -> MotionPairEvidence:
    """Evaluate one pair using explicit event fractions and unchanged gates."""

    first_path = _path(source1_path, owner="source1 path")
    second_path = _path(source2_path, owner="source2 path")
    first_fraction = _fraction(first_anchor_fraction, owner="first anchor fraction")
    second_fraction = _fraction(second_anchor_fraction, owner="second anchor fraction")
    first_margin_threshold = _finite_number(
        minimum_first_side_margin, owner="minimum first side margin"
    )
    second_margin_threshold = _finite_number(
        minimum_second_side_margin, owner="minimum second side margin"
    )
    separation_threshold = _finite_number(
        minimum_continuous_separation, owner="minimum continuous separation"
    )
    if min(first_margin_threshold, second_margin_threshold, separation_threshold) < 0:
        raise MotionFeasibilityError("motion thresholds must be nonnegative")
    first_anchors = (
        interpolate_linear_path(first_path, first_fraction),
        interpolate_linear_path(second_path, first_fraction),
    )
    second_anchors = (
        interpolate_linear_path(first_path, second_fraction),
        interpolate_linear_path(second_path, second_fraction),
    )
    first_sides_and_margins = tuple(
        camera_side(camera_position_xy, camera_yaw_degrees, point)
        for point in first_anchors
    )
    second_sides_and_margins = tuple(
        camera_side(camera_position_xy, camera_yaw_degrees, point)
        for point in second_anchors
    )
    first_sides = tuple(item[0] for item in first_sides_and_margins)
    first_margins = tuple(item[1] for item in first_sides_and_margins)
    second_margins = tuple(item[1] for item in second_sides_and_margins)
    side_passed = bool(
        allow_any_side
        or (
            first_sides[0] != first_sides[1]
            and min(first_margins) + 1.0e-12 >= first_margin_threshold
            and min(second_margins) + 1.0e-12 >= second_margin_threshold
        )
    )
    separation = minimum_linear_path_separation(first_path, second_path)
    separation_passed = separation + 1.0e-12 >= separation_threshold
    return MotionPairEvidence(
        source1_path=first_path,
        source2_path=second_path,
        first_anchor_fraction=first_fraction,
        second_anchor_fraction=second_fraction,
        first_anchor_sides=(first_sides[0], first_sides[1]),
        first_anchor_margins=(first_margins[0], first_margins[1]),
        second_anchor_margins=(second_margins[0], second_margins[1]),
        minimum_continuous_separation=separation,
        side_gate_passed=side_passed,
        separation_gate_passed=separation_passed,
        passed=side_passed and separation_passed,
    )


def _motion_sources(
    *,
    mode: str,
    directed_paths: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
    stand_points: tuple[tuple[float, ...], ...],
    seed: int,
) -> tuple[
    tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
    tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
    bool,
]:
    walk_order = _rotated_indices(len(directed_paths), seed)
    second_walk_order = _rotated_indices(len(directed_paths), seed * 5)
    stand_order = _rotated_indices(len(stand_points), seed * 3)
    second_stand_order = _rotated_indices(len(stand_points), seed * 5)
    first_walks = tuple(directed_paths[index] for index in walk_order)
    second_walks = tuple(directed_paths[index] for index in second_walk_order)
    first_stands = tuple(
        (stand_points[index], stand_points[index]) for index in stand_order
    )
    second_stands = tuple(
        (stand_points[index], stand_points[index]) for index in second_stand_order
    )
    if mode == "s1_moving_s2_static":
        return first_walks, first_stands, False
    if mode == "s1_static_s2_moving":
        return first_stands, first_walks, False
    if mode == "both_moving":
        return first_walks, second_walks, True
    if mode == "both_static":
        return first_stands, second_stands, True
    raise MotionFeasibilityError(f"unknown motion mode: {mode}")


def _path_side_item(
    path: tuple[tuple[float, ...], tuple[float, ...]],
    *,
    camera_position_xy: tuple[float, ...],
    camera_yaw_degrees: float,
    first_anchor_fraction: float,
    second_anchor_fraction: float,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], str, float, float]:
    first_anchor = interpolate_linear_path(path, first_anchor_fraction)
    second_anchor = interpolate_linear_path(path, second_anchor_fraction)
    first_side, first_margin = camera_side(
        camera_position_xy, camera_yaw_degrees, first_anchor
    )
    _second_side, second_margin = camera_side(
        camera_position_xy, camera_yaw_degrees, second_anchor
    )
    return path, first_side, first_margin, second_margin


def _pair_from_side_items(
    first: tuple[tuple[tuple[float, ...], tuple[float, ...]], str, float, float],
    second: tuple[tuple[tuple[float, ...], tuple[float, ...]], str, float, float],
    *,
    first_anchor_fraction: float,
    second_anchor_fraction: float,
    minimum_continuous_separation: float,
) -> MotionPairEvidence:
    separation = minimum_linear_path_separation(first[0], second[0])
    separation_passed = separation + 1.0e-12 >= minimum_continuous_separation
    return MotionPairEvidence(
        source1_path=first[0],
        source2_path=second[0],
        first_anchor_fraction=first_anchor_fraction,
        second_anchor_fraction=second_anchor_fraction,
        first_anchor_sides=(first[1], second[1]),
        first_anchor_margins=(first[2], second[2]),
        second_anchor_margins=(first[3], second[3]),
        minimum_continuous_separation=separation,
        side_gate_passed=True,
        separation_gate_passed=separation_passed,
        passed=separation_passed,
    )


def _failed_pair_key(evidence: MotionPairEvidence) -> tuple[float, ...]:
    return (
        float(evidence.side_gate_passed) + float(evidence.separation_gate_passed),
        min(evidence.first_anchor_margins),
        min(evidence.second_anchor_margins),
        evidence.minimum_continuous_separation,
    )


def _candidate_pair_count(
    first_paths: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
    second_paths: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
    *,
    require_distinct: bool,
) -> int:
    total = len(first_paths) * len(second_paths)
    if not require_distinct:
        return total
    second_counts: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
    for path in second_paths:
        second_counts[path] = second_counts.get(path, 0) + 1
    return total - sum(second_counts.get(path, 0) for path in first_paths)


def classify_motion_feasibility(
    *,
    camera_position_xy: Sequence[Real],
    camera_yaw_degrees: Real,
    directed_paths: Sequence[Sequence[Sequence[Real]]],
    stand_points: Sequence[Sequence[Real]],
    modes: Sequence[str] = MOTION_MODES,
    seed: int = 0,
    first_anchor_fraction: Real = DEFAULT_FIRST_ANCHOR_FRACTION,
    second_anchor_fraction: Real = DEFAULT_SECOND_ANCHOR_FRACTION,
    minimum_first_side_margin: Real = 0.12,
    minimum_second_side_margin: Real = 0.06,
    minimum_continuous_separation: Real = 60.0,
) -> MotionFeasibilityClassification:
    """Search each mode deterministically and retain compact failure evidence."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MotionFeasibilityError("seed must be an integer")
    if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence) or not modes:
        raise MotionFeasibilityError("modes must be a nonempty sequence")
    if len(set(modes)) != len(modes):
        raise MotionFeasibilityError("modes must not contain duplicates")
    normalized_camera = _point(camera_position_xy, owner="camera position")
    normalized_yaw = _finite_number(camera_yaw_degrees, owner="camera yaw")
    normalized_directed = tuple(
        _path(path, owner=f"directed_paths[{index}]")
        for index, path in enumerate(directed_paths)
    )
    normalized_stands = tuple(
        _point(point, owner=f"stand_points[{index}]")
        for index, point in enumerate(stand_points)
    )
    first_fraction = _fraction(first_anchor_fraction, owner="first anchor fraction")
    second_fraction = _fraction(second_anchor_fraction, owner="second anchor fraction")
    first_margin_threshold = _finite_number(
        minimum_first_side_margin, owner="minimum first side margin"
    )
    second_margin_threshold = _finite_number(
        minimum_second_side_margin, owner="minimum second side margin"
    )
    separation_threshold = _finite_number(
        minimum_continuous_separation, owner="minimum continuous separation"
    )
    if min(first_margin_threshold, second_margin_threshold, separation_threshold) < 0:
        raise MotionFeasibilityError("motion thresholds must be nonnegative")
    mode_results = []
    for mode in modes:
        first_paths, second_paths, require_distinct = _motion_sources(
            mode=mode,
            directed_paths=normalized_directed,
            stand_points=normalized_stands,
            seed=seed,
        )
        attempted = _candidate_pair_count(
            first_paths,
            second_paths,
            require_distinct=require_distinct,
        )
        representative = None
        for first_path in first_paths:
            for second_path in second_paths:
                if require_distinct and first_path == second_path:
                    continue
                representative = evaluate_motion_pair(
                    camera_position_xy=normalized_camera,
                    camera_yaw_degrees=normalized_yaw,
                    source1_path=first_path,
                    source2_path=second_path,
                    first_anchor_fraction=first_fraction,
                    second_anchor_fraction=second_fraction,
                    minimum_first_side_margin=first_margin_threshold,
                    minimum_second_side_margin=second_margin_threshold,
                    minimum_continuous_separation=separation_threshold,
                )
                break
            if representative is not None:
                break
        first_items = tuple(
            _path_side_item(
                path,
                camera_position_xy=normalized_camera,
                camera_yaw_degrees=normalized_yaw,
                first_anchor_fraction=first_fraction,
                second_anchor_fraction=second_fraction,
            )
            for path in first_paths
        )
        second_items = tuple(
            _path_side_item(
                path,
                camera_position_xy=normalized_camera,
                camera_yaw_degrees=normalized_yaw,
                first_anchor_fraction=first_fraction,
                second_anchor_fraction=second_fraction,
            )
            for path in second_paths
        )
        first_items = tuple(
            item
            for item in first_items
            if item[2] + 1.0e-12 >= first_margin_threshold
            and item[3] + 1.0e-12 >= second_margin_threshold
        )
        second_items = tuple(
            item
            for item in second_items
            if item[2] + 1.0e-12 >= first_margin_threshold
            and item[3] + 1.0e-12 >= second_margin_threshold
        )
        second_by_side = {
            side: tuple(item for item in second_items if item[1] == side)
            for side in ("left", "right")
        }
        side_passed = sum(
            len(second_by_side["left" if first[1] == "right" else "right"])
            for first in first_items
        )
        separation_passed = 0
        feasible_count = 0
        selected = None
        best_failed = None
        for first in first_items:
            opposite = "left" if first[1] == "right" else "right"
            for second in second_by_side[opposite]:
                if require_distinct and first[0] == second[0]:
                    continue
                evidence = _pair_from_side_items(
                    first,
                    second,
                    first_anchor_fraction=first_fraction,
                    second_anchor_fraction=second_fraction,
                    minimum_continuous_separation=separation_threshold,
                )
                separation_passed += int(evidence.separation_gate_passed)
                if evidence.passed:
                    selected = evidence
                    feasible_count = 1
                    break
                if best_failed is None or _failed_pair_key(evidence) > _failed_pair_key(
                    best_failed
                ):
                    best_failed = evidence
            if selected is not None:
                break
        if selected is not None:
            failure_reason = None
        elif attempted == 0:
            failure_reason = "no_candidate_pairs_for_motion_mode"
        elif side_passed == 0:
            failure_reason = "no_pair_satisfies_side_margins"
        else:
            failure_reason = "side_pass_pairs_fail_continuous_separation"
        mode_results.append(
            MotionModeEvidence(
                mode=mode,
                feasible=selected is not None,
                attempted_pair_count=attempted,
                side_gate_pass_pair_count=side_passed,
                separation_gate_pass_pair_count=separation_passed,
                feasible_pair_count=feasible_count,
                representative_pair=representative,
                selected_pair=selected,
                best_failed_pair=best_failed,
                failure_reason=failure_reason,
            )
        )
    return MotionFeasibilityClassification(
        modes=tuple(mode_results),
        thresholds={
            "first_anchor_fraction": first_fraction,
            "second_anchor_fraction": second_fraction,
            "minimum_first_side_margin": first_margin_threshold,
            "minimum_second_side_margin": second_margin_threshold,
            "minimum_continuous_separation": separation_threshold,
        },
    )


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray) and value.dtype.kind == "b":
        return True
    try:
        return any(_contains_boolean(item) for item in value)
    except TypeError:
        return False


def _track_xyz(value: Any, *, owner: str) -> np.ndarray:
    if _contains_boolean(value):
        raise MotionFeasibilityError(f"{owner} must not contain booleans")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionFeasibilityError(
            f"{owner} must be a finite [frame,3] track"
        ) from exc
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise MotionFeasibilityError(
            f"{owner} must contain real numeric values without booleans"
        )
    try:
        track = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionFeasibilityError(
            f"{owner} must be a finite [frame,3] track"
        ) from exc
    if track.ndim != 2 or track.shape[0] < 1 or track.shape[1] != 3:
        raise MotionFeasibilityError(
            f"{owner} must be a nonempty finite [frame,3] track"
        )
    if not np.all(np.isfinite(track)):
        raise MotionFeasibilityError(
            f"{owner} must be a nonempty finite [frame,3] track"
        )
    return np.ascontiguousarray(track)


def _frame_vector_track(value: Any, *, frame_count: int, owner: str) -> np.ndarray:
    if _contains_boolean(value):
        raise MotionFeasibilityError(f"{owner} must not contain booleans")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionFeasibilityError(
            f"{owner} must contain one finite XYZ vector or one per frame"
        ) from exc
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise MotionFeasibilityError(
            f"{owner} must contain real numeric values without booleans"
        )
    try:
        vectors = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionFeasibilityError(
            f"{owner} must contain one finite XYZ vector or one per frame"
        ) from exc
    if vectors.shape == (3,):
        vectors = np.broadcast_to(vectors, (frame_count, 3)).copy()
    if vectors.shape != (frame_count, 3) or not np.all(np.isfinite(vectors)):
        raise MotionFeasibilityError(
            f"{owner} must contain one finite XYZ vector or one per frame"
        )
    return np.ascontiguousarray(vectors)


def _axis_mapping(
    horizontal_axis_indices: Sequence[int], vertical_axis_index: int
) -> tuple[tuple[int, int], int]:
    try:
        raw_horizontal = tuple(horizontal_axis_indices)
    except TypeError as exc:
        raise MotionFeasibilityError(
            "horizontal_axis_indices must contain two integer axes"
        ) from exc
    if (
        isinstance(horizontal_axis_indices, (str, bytes))
        or len(raw_horizontal) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, Integral)
            for item in raw_horizontal
        )
    ):
        raise MotionFeasibilityError(
            "horizontal_axis_indices must contain two integer axes"
        )
    horizontal = tuple(int(item) for item in raw_horizontal)
    if len(set(horizontal)) != 2 or any(item not in (0, 1, 2) for item in horizontal):
        raise MotionFeasibilityError(
            "horizontal_axis_indices must contain two distinct axes in [0, 2]"
        )
    if (
        isinstance(vertical_axis_index, bool)
        or not isinstance(vertical_axis_index, Integral)
        or vertical_axis_index not in (0, 1, 2)
        or vertical_axis_index in horizontal
    ):
        raise MotionFeasibilityError(
            "vertical_axis_index must be the remaining axis in [0, 2]"
        )
    return (horizontal[0], horizontal[1]), int(vertical_axis_index)


def _frame_identifiers(
    value: Sequence[int] | None, *, frame_count: int, owner: str
) -> tuple[int, ...]:
    if value is None:
        return tuple(range(frame_count))
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise MotionFeasibilityError(
            f"{owner} must contain one integer per source-track frame"
        ) from exc
    if (
        isinstance(value, (str, bytes))
        or len(raw) != frame_count
        or any(isinstance(item, bool) or not isinstance(item, Integral) for item in raw)
    ):
        raise MotionFeasibilityError(
            f"{owner} must contain one integer per source-track frame"
        )
    result = tuple(int(item) for item in raw)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise MotionFeasibilityError(f"{owner} must be strictly increasing")
    return result


def _optional_ticks(
    value: Sequence[int] | None, *, frame_count: int
) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _frame_identifiers(value, frame_count=frame_count, owner="pts_ticks")


def _source_identifier(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MotionFeasibilityError(f"{owner} must be a nonempty string")
    return value


def _stable_norm_xyz_rows(value: np.ndarray) -> np.ndarray:
    result = np.hypot(np.hypot(value[:, 0], value[:, 1]), value[:, 2])
    if not np.all(np.isfinite(result)):
        raise MotionFeasibilityError("derived XYZ norm is not finite")
    return result


def _stable_norm_xyz(value: np.ndarray) -> float:
    result = math.hypot(float(value[0]), float(value[1]), float(value[2]))
    if not math.isfinite(result):
        raise MotionFeasibilityError("derived XYZ norm is not finite")
    return result


def _stable_distance(
    left: np.ndarray, right: np.ndarray, *, axes: Sequence[int]
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.subtract(left[list(axes)], right[list(axes)])
    if not np.all(np.isfinite(delta)):
        raise MotionFeasibilityError("derived point delta is not finite")
    if len(delta) == 2:
        result = math.hypot(float(delta[0]), float(delta[1]))
    elif len(delta) == 3:
        result = math.hypot(float(delta[0]), float(delta[1]), float(delta[2]))
    else:
        raise MotionFeasibilityError("distance axes must contain two or three axes")
    if not math.isfinite(result):
        raise MotionFeasibilityError("derived point distance is not finite")
    return result


def _listener_basis(
    *,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, str]:
    norms = np.stack(
        (
            _stable_norm_xyz_rows(forward),
            _stable_norm_xyz_rows(right),
            _stable_norm_xyz_rows(up),
        ),
        axis=1,
    )
    if not np.allclose(norms, 1.0, rtol=0.0, atol=tolerance):
        raise MotionFeasibilityError(
            "listener forward/right/up vectors must be an orthonormal basis per frame"
        )
    dot_fr = np.sum(forward * right, axis=1)
    dot_fu = np.sum(forward * up, axis=1)
    dot_ru = np.sum(right * up, axis=1)
    if (
        np.any(np.abs(dot_fr) > tolerance)
        or np.any(np.abs(dot_fu) > tolerance)
        or np.any(np.abs(dot_ru) > tolerance)
    ):
        raise MotionFeasibilityError(
            "listener forward/right/up vectors must be an orthonormal basis per frame"
        )
    handedness = np.sum(np.cross(forward, right) * up, axis=1)
    if np.all(handedness >= 1.0 - tolerance):
        label = "right_handed_forward_right_up"
    elif np.all(handedness <= -1.0 + tolerance):
        label = "left_handed_forward_right_up"
    else:
        raise MotionFeasibilityError(
            "listener basis handedness must be consistent across frames"
        )
    return handedness, label


def _local_angles_from_basis(
    delta: np.ndarray,
    *,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    side_center_tolerance: float,
) -> tuple[float | None, float | None, str, float, float, bool]:
    distance = _stable_norm_xyz(delta)
    if distance <= side_center_tolerance:
        return None, None, "coincident", 0.0, 0.0, False
    forward_component = float(np.dot(delta, forward))
    right_component = float(np.dot(delta, right))
    up_component = float(np.dot(delta, up))
    if not all(
        math.isfinite(value)
        for value in (forward_component, right_component, up_component)
    ):
        raise MotionFeasibilityError("listener-local derived components are not finite")
    azimuth = math.degrees(math.atan2(right_component, forward_component))
    elevation = math.degrees(
        math.atan2(
            up_component,
            math.hypot(forward_component, right_component),
        )
    )
    if right_component > side_center_tolerance:
        side = "right"
    elif right_component < -side_center_tolerance:
        side = "left"
    else:
        side = "center"
    side_margin = abs(right_component) / max(distance, 1.0e-12)
    return azimuth, elevation, side, side_margin, right_component, True


def _closest_frame_record(
    values: np.ndarray,
    *,
    frame_indices: tuple[int, ...],
) -> dict[str, Any]:
    minimum = float(np.min(values))
    tied_ordinals = tuple(int(item) for item in np.flatnonzero(values == minimum))
    ordinal = tied_ordinals[0]
    return {
        "frame_ordinal": ordinal,
        "frame_index": frame_indices[ordinal],
        "distance": minimum,
        "tie_policy": "earliest_frame_ordinal",
        "tied_frame_ordinals": list(tied_ordinals),
        "tied_frame_indices": [frame_indices[item] for item in tied_ordinals],
    }


def describe_source_pair_tracks(
    source1_positions_xyz: Any,
    source2_positions_xyz: Any,
    *,
    source1_id: str,
    source2_id: str,
    listener_positions_xyz: Any,
    listener_forward_xyz: Any,
    listener_right_xyz: Any,
    listener_up_xyz: Any,
    horizontal_axis_indices: Sequence[int],
    vertical_axis_index: int,
    coordinate_space: str,
    track_authority: str,
    frame_indices: Sequence[int] | None = None,
    pts_ticks: Sequence[int] | None = None,
    observed_coincidence_tolerance: Real = 1.0e-6,
    endpoint_exchange_tolerance: Real = 1.0e-6,
    side_center_tolerance: Real = 1.0e-9,
    basis_tolerance: Real = 1.0e-6,
) -> dict[str, Any]:
    """Describe two caller-supplied per-frame tracks without interpolation.

    ``horizontal_axis_indices`` and ``vertical_axis_index`` describe the input
    XYZ storage axes. Listener-local azimuth/elevation use the explicit
    per-frame forward/right/up basis, not an assumed backend convention.
    """

    first_id = _source_identifier(source1_id, owner="source1_id")
    second_id = _source_identifier(source2_id, owner="source2_id")
    if first_id == second_id:
        raise MotionFeasibilityError("source1_id and source2_id must differ")
    if not isinstance(coordinate_space, str) or not coordinate_space.strip():
        raise MotionFeasibilityError("coordinate_space must be a nonempty string")
    authority = _source_identifier(track_authority, owner="track_authority")
    source1 = _track_xyz(source1_positions_xyz, owner="source1_positions_xyz")
    source2 = _track_xyz(source2_positions_xyz, owner="source2_positions_xyz")
    if source1.shape != source2.shape:
        raise MotionFeasibilityError(
            "source1/source2 tracks must have equal [frame,3] shape"
        )
    frame_count = int(source1.shape[0])
    listener_positions = _frame_vector_track(
        listener_positions_xyz,
        frame_count=frame_count,
        owner="listener_positions_xyz",
    )
    listener_forward = _frame_vector_track(
        listener_forward_xyz,
        frame_count=frame_count,
        owner="listener_forward_xyz",
    )
    listener_right = _frame_vector_track(
        listener_right_xyz,
        frame_count=frame_count,
        owner="listener_right_xyz",
    )
    listener_up = _frame_vector_track(
        listener_up_xyz,
        frame_count=frame_count,
        owner="listener_up_xyz",
    )
    horizontal_axes, vertical_axis = _axis_mapping(
        horizontal_axis_indices, vertical_axis_index
    )
    indices = _frame_identifiers(
        frame_indices,
        frame_count=frame_count,
        owner="frame_indices",
    )
    ticks = _optional_ticks(pts_ticks, frame_count=frame_count)
    coincidence_tolerance = _finite_number(
        observed_coincidence_tolerance,
        owner="observed coincidence tolerance",
    )
    exchange_tolerance = _finite_number(
        endpoint_exchange_tolerance,
        owner="endpoint exchange tolerance",
    )
    side_tolerance = _finite_number(
        side_center_tolerance,
        owner="side center tolerance",
    )
    orthonormal_tolerance = _finite_number(
        basis_tolerance,
        owner="basis tolerance",
    )
    if (
        min(
            coincidence_tolerance,
            exchange_tolerance,
            side_tolerance,
            orthonormal_tolerance,
        )
        < 0.0
    ):
        raise MotionFeasibilityError("track descriptor tolerances must be nonnegative")
    if orthonormal_tolerance > 1.0e-3:
        raise MotionFeasibilityError("basis tolerance must not exceed 1e-3")
    _handedness, basis_handedness = _listener_basis(
        forward=listener_forward,
        right=listener_right,
        up=listener_up,
        tolerance=orthonormal_tolerance,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        pair_delta = np.subtract(source1, source2)
    if not np.all(np.isfinite(pair_delta)):
        raise MotionFeasibilityError("source-pair derived deltas are not finite")
    distance_xy = np.hypot(pair_delta[:, 0], pair_delta[:, 1])
    distance_horizontal = np.hypot(
        pair_delta[:, horizontal_axes[0]],
        pair_delta[:, horizontal_axes[1]],
    )
    distance_xyz = _stable_norm_xyz_rows(pair_delta)
    frames = []
    source1_right_coordinates = []
    source2_right_coordinates = []
    for ordinal in range(frame_count):
        with np.errstate(over="ignore", invalid="ignore"):
            first_delta = np.subtract(source1[ordinal], listener_positions[ordinal])
            second_delta = np.subtract(source2[ordinal], listener_positions[ordinal])
        if not np.all(np.isfinite(first_delta)) or not np.all(
            np.isfinite(second_delta)
        ):
            raise MotionFeasibilityError(
                "source/listener derived deltas are not finite"
            )
        (
            first_azimuth,
            first_elevation,
            first_side,
            first_margin,
            first_right,
            first_angles_defined,
        ) = _local_angles_from_basis(
            first_delta,
            forward=listener_forward[ordinal],
            right=listener_right[ordinal],
            up=listener_up[ordinal],
            side_center_tolerance=side_tolerance,
        )
        (
            second_azimuth,
            second_elevation,
            second_side,
            second_margin,
            second_right,
            second_angles_defined,
        ) = _local_angles_from_basis(
            second_delta,
            forward=listener_forward[ordinal],
            right=listener_right[ordinal],
            up=listener_up[ordinal],
            side_center_tolerance=side_tolerance,
        )
        source1_right_coordinates.append(first_right)
        source2_right_coordinates.append(second_right)
        pairwise_right_difference = first_right - second_right
        if not math.isfinite(pairwise_right_difference):
            raise MotionFeasibilityError(
                "pairwise listener-local-right difference is not finite"
            )
        if pairwise_right_difference > side_tolerance:
            pairwise_right_order = "source1_right_of_source2"
        elif pairwise_right_difference < -side_tolerance:
            pairwise_right_order = "source1_left_of_source2"
        else:
            pairwise_right_order = "tied"
        frame = {
            "frame_ordinal": ordinal,
            "frame_index": indices[ordinal],
            "listener": {
                "position_xyz": listener_positions[ordinal].tolist(),
                "forward_xyz": listener_forward[ordinal].tolist(),
                "right_xyz": listener_right[ordinal].tolist(),
                "up_xyz": listener_up[ordinal].tolist(),
            },
            "source1": {
                "source_id": first_id,
                "position_xyz": source1[ordinal].tolist(),
                "listener_local_azimuth_deg": first_azimuth,
                "listener_local_elevation_deg": first_elevation,
                "listener_angles_defined": first_angles_defined,
                "listener_side": first_side,
                "listener_side_margin": first_margin,
                "listener_local_right_coordinate": first_right,
                "projection": None,
                "in_frame": None,
                "occlusion": None,
                "visible_pixels": None,
            },
            "source2": {
                "source_id": second_id,
                "position_xyz": source2[ordinal].tolist(),
                "listener_local_azimuth_deg": second_azimuth,
                "listener_local_elevation_deg": second_elevation,
                "listener_angles_defined": second_angles_defined,
                "listener_side": second_side,
                "listener_side_margin": second_margin,
                "listener_local_right_coordinate": second_right,
                "projection": None,
                "in_frame": None,
                "occlusion": None,
                "visible_pixels": None,
            },
            "pair_distance_xy": float(distance_xy[ordinal]),
            "pair_distance_horizontal": float(distance_horizontal[ordinal]),
            "pair_distance_xyz": float(distance_xyz[ordinal]),
            "pairwise_listener_right_difference": pairwise_right_difference,
            "pairwise_listener_right_order": pairwise_right_order,
        }
        if ticks is not None:
            frame["pts_ticks"] = ticks[ordinal]
        frames.append(frame)

    observed_horizontal_coincidence_ordinals = tuple(
        int(index)
        for index in np.flatnonzero(distance_horizontal <= coincidence_tolerance)
    )
    pairwise_right_differences = np.asarray(source1_right_coordinates) - np.asarray(
        source2_right_coordinates
    )
    pairwise_right_order_swap_pairs = []
    previous_decisive = None
    for ordinal, difference in enumerate(pairwise_right_differences):
        if abs(float(difference)) <= side_tolerance:
            continue
        sign = 1 if difference > 0.0 else -1
        if previous_decisive is not None and sign != previous_decisive[1]:
            left = previous_decisive[0]
            pairwise_right_order_swap_pairs.append(
                {
                    "left_frame_ordinal": left,
                    "right_frame_ordinal": ordinal,
                    "left_frame_index": indices[left],
                    "right_frame_index": indices[ordinal],
                    "tied_frame_ordinals_between": list(range(left + 1, ordinal)),
                }
            )
        previous_decisive = (ordinal, sign)
    initial_horizontal_separation = float(distance_horizontal[0])
    final_horizontal_separation = float(distance_horizontal[-1])
    initial_xyz_separation = float(distance_xyz[0])
    final_xyz_separation = float(distance_xyz[-1])
    endpoint_exchange_horizontal = bool(
        frame_count >= 2
        and initial_horizontal_separation > exchange_tolerance
        and final_horizontal_separation > exchange_tolerance
        and _stable_distance(source1[0], source2[-1], axes=horizontal_axes)
        <= exchange_tolerance
        and _stable_distance(source2[0], source1[-1], axes=horizontal_axes)
        <= exchange_tolerance
    )
    endpoint_exchange_xyz = bool(
        frame_count >= 2
        and initial_xyz_separation > exchange_tolerance
        and final_xyz_separation > exchange_tolerance
        and _stable_distance(source1[0], source2[-1], axes=(0, 1, 2))
        <= exchange_tolerance
        and _stable_distance(source2[0], source1[-1], axes=(0, 1, 2))
        <= exchange_tolerance
    )
    endpoint_pairwise_listener_right_order_exchange = bool(
        frame_count >= 2
        and abs(float(pairwise_right_differences[0])) > side_tolerance
        and abs(float(pairwise_right_differences[-1])) > side_tolerance
        and (pairwise_right_differences[0] > 0.0)
        != (pairwise_right_differences[-1] > 0.0)
    )
    result = {
        "track_provenance": "caller_supplied_per_frame_positions_no_interpolation",
        "track_authority": authority,
        "runtime_track_claim": False,
        "runtime_readback_verification": "not_performed_by_descriptor",
        "frame_count": frame_count,
        "coordinate_space": coordinate_space,
        "axis_mapping": {
            "input_storage_axes": [0, 1, 2],
            "xy_distance_axes": [0, 1],
            "horizontal_axis_indices": list(horizontal_axes),
            "vertical_axis_index": vertical_axis,
            "listener_basis_handedness": basis_handedness,
        },
        "parameters": {
            "observed_coincidence_tolerance": coincidence_tolerance,
            "endpoint_exchange_tolerance": exchange_tolerance,
            "side_center_tolerance": side_tolerance,
            "basis_tolerance": orthonormal_tolerance,
        },
        "source_identity_binding": {
            "source1_id": first_id,
            "source2_id": second_id,
            "identity_preserved_through_crossing_or_swap": True,
        },
        "frames": frames,
        "distance_series": {
            "pair_distance_xy": distance_xy.tolist(),
            "pair_distance_horizontal": distance_horizontal.tolist(),
            "pair_distance_xyz": distance_xyz.tolist(),
        },
        "closest_frames": {
            "xy": _closest_frame_record(distance_xy, frame_indices=indices),
            "horizontal": _closest_frame_record(
                distance_horizontal, frame_indices=indices
            ),
            "xyz": _closest_frame_record(distance_xyz, frame_indices=indices),
        },
        "observed_crossing_and_swap": {
            "definition": (
                "observed-frame coincidence and pairwise listener-local-right "
                "order only; no "
                "between-frame interpolation or native mesh-intersection claim"
            ),
            "observed_horizontal_coincidence_frame_ordinals": list(
                observed_horizontal_coincidence_ordinals
            ),
            "observed_horizontal_coincidence_frame_indices": [
                indices[item] for item in observed_horizontal_coincidence_ordinals
            ],
            "observed_pairwise_listener_right_order_swap_frame_pairs": (
                pairwise_right_order_swap_pairs
            ),
            "observed_side_order_swap_frame_pairs": (pairwise_right_order_swap_pairs),
            "endpoint_exchange_horizontal": endpoint_exchange_horizontal,
            "endpoint_exchange_xyz": endpoint_exchange_xyz,
            "endpoint_pairwise_listener_right_order_exchange": (
                endpoint_pairwise_listener_right_order_exchange
            ),
            "between_frame_crossing_status": "not_inferred",
        },
        "native_projection_visibility": {
            "status": "pending_same_renderer_native_execution",
            "projection_by_frame": None,
            "in_frame_by_frame": None,
            "occlusion_by_frame": None,
            "visible_pixels_by_frame": None,
        },
        "legality": {
            "status": "not_evaluated_by_track_descriptor",
            "native_mesh_intersection": None,
            "native_collision": None,
            "runtime_pose_readback": None,
        },
        "difficulty": {
            "status": "descriptors_only",
            "candidate_filtering": "disabled",
            "minimum_pair_distance_horizontal": float(np.min(distance_horizontal)),
            "minimum_pair_distance_xyz": float(np.min(distance_xyz)),
        },
        "claim_boundary": (
            "exact caller-supplied frame positions and listener basis only; native "
            "collision, projection, in-frame, occlusion and visible pixels remain "
            "separate pending evidence"
        ),
    }
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MotionFeasibilityError(
            "track descriptor produced a non-JSON-safe derived value"
        ) from exc
    return result


def _xyz(value: Sequence[Real], *, owner: str) -> tuple[float, float, float]:
    point = _point(value, owner=owner)
    if len(point) == 2:
        return point[0], point[1], 0.0
    return point[0], point[1], point[2]


def _camera_angles(
    camera_position_xyz: tuple[float, float, float],
    camera_yaw_degrees: float,
    point_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    delta_x = point_xyz[0] - camera_position_xyz[0]
    delta_y = point_xyz[1] - camera_position_xyz[1]
    delta_z = point_xyz[2] - camera_position_xyz[2]
    world_azimuth = math.degrees(math.atan2(delta_y, delta_x))
    relative_azimuth = (world_azimuth - camera_yaw_degrees + 180.0) % 360.0 - 180.0
    elevation = math.degrees(math.atan2(delta_z, math.hypot(delta_x, delta_y)))
    return relative_azimuth, elevation


def _pair_frame_descriptor(
    pair: MotionPairEvidence,
    *,
    camera_position_xyz: tuple[float, float, float],
    camera_yaw_degrees: float,
    frame_count: int,
) -> dict[str, Any]:
    frames = []
    distance_xy = []
    distance_xyz = []
    source1_sides = []
    source2_sides = []
    for frame_index in range(frame_count):
        fraction = 0.0 if frame_count == 1 else frame_index / (frame_count - 1)
        source1 = _xyz(
            interpolate_linear_path(pair.source1_path, fraction),
            owner="source1 frame position",
        )
        source2 = _xyz(
            interpolate_linear_path(pair.source2_path, fraction),
            owner="source2 frame position",
        )
        xy_distance = math.dist(source1[:2], source2[:2])
        xyz_distance = math.dist(source1, source2)
        source1_side, source1_margin = camera_side(
            camera_position_xyz[:2], camera_yaw_degrees, source1[:2]
        )
        source2_side, source2_margin = camera_side(
            camera_position_xyz[:2], camera_yaw_degrees, source2[:2]
        )
        source1_azimuth, source1_elevation = _camera_angles(
            camera_position_xyz, camera_yaw_degrees, source1
        )
        source2_azimuth, source2_elevation = _camera_angles(
            camera_position_xyz, camera_yaw_degrees, source2
        )
        distance_xy.append(xy_distance)
        distance_xyz.append(xyz_distance)
        source1_sides.append(source1_side)
        source2_sides.append(source2_side)
        frames.append(
            {
                "frame_index": frame_index,
                "normalized_time": fraction,
                "source1": {
                    "source_slot_id": "source1",
                    "position": list(source1),
                    "camera_side": source1_side,
                    "side_margin": source1_margin,
                    "relative_azimuth_deg": source1_azimuth,
                    "relative_elevation_deg": source1_elevation,
                },
                "source2": {
                    "source_slot_id": "source2",
                    "position": list(source2),
                    "camera_side": source2_side,
                    "side_margin": source2_margin,
                    "relative_azimuth_deg": source2_azimuth,
                    "relative_elevation_deg": source2_elevation,
                },
                "pair_distance_xy": xy_distance,
                "pair_distance_xyz": xyz_distance,
            }
        )
    continuous_distance, continuous_fraction = linear_path_closest_approach(
        pair.source1_path, pair.source2_path
    )
    closest_frame = min(range(frame_count), key=lambda index: distance_xy[index])
    endpoint_exchange = bool(
        math.dist(pair.source1_path[0][:2], pair.source2_path[1][:2]) <= 1.0e-9
        and math.dist(pair.source1_path[1][:2], pair.source2_path[0][:2]) <= 1.0e-9
    )
    side_order_swapped = bool(
        source1_sides[0] != source1_sides[-1]
        and source2_sides[0] != source2_sides[-1]
        and source1_sides[0] == source2_sides[-1]
        and source2_sides[0] == source1_sides[-1]
    )
    return {
        "track_provenance": (
            "synthesized_linear_between_declared_endpoints_not_runtime_track"
        ),
        "runtime_track_claim": False,
        "source_identity_binding": {
            "source1": "source1",
            "source2": "source2",
            "identity_preserved_through_crossing_or_swap": True,
        },
        "frame_count": frame_count,
        "coordinate_space": "caller_declared_xy_or_xyz",
        "frames": frames,
        "summary": {
            "minimum_pair_distance_xy_by_frame": distance_xy[closest_frame],
            "minimum_pair_distance_xyz_by_frame": min(distance_xyz),
            "closest_frame_index_xy": closest_frame,
            "continuous_minimum_pair_distance_xy": continuous_distance,
            "continuous_closest_fraction_xy": continuous_fraction,
            "continuous_crossing_xy": continuous_distance <= 1.0e-9,
            "endpoint_exchange": endpoint_exchange,
            "camera_side_order_swapped": side_order_swapped,
            "difficulty_profile_match": pair.passed,
            "difficulty_thresholds_do_not_filter_candidate": True,
        },
        "native_projection_visibility": {
            "status": "pending_same_renderer_native_execution",
            "projection_by_frame": None,
            "in_frame_by_frame": None,
            "occlusion_by_frame": None,
            "visible_pixels_by_frame": None,
        },
        "claim_boundary": (
            "root/anchor linear geometry descriptors only; endpoint exchange and "
            "crossing are retained, while native mesh collision and same-renderer "
            "pixel visibility remain separate evidence"
        ),
    }


def describe_motion_geometry(
    classification: MotionFeasibilityClassification,
    *,
    camera_position_xyz: Sequence[Real],
    camera_yaw_degrees: Real,
    frame_count: int = 75,
) -> dict[str, Any]:
    """Materialize per-frame descriptors for each deterministic mode pair."""

    if not isinstance(classification, MotionFeasibilityClassification):
        raise MotionFeasibilityError(
            "classification must be MotionFeasibilityClassification"
        )
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise MotionFeasibilityError("frame_count must be a positive integer")
    camera = _xyz(camera_position_xyz, owner="camera position")
    yaw = _finite_number(camera_yaw_degrees, owner="camera yaw")
    modes = {}
    for mode in classification.modes:
        pair = mode.representative_pair
        modes[mode.mode] = {
            "representative_pair_status": (
                "present" if pair is not None else "no_legal_path_pair"
            ),
            "descriptor": (
                None
                if pair is None
                else _pair_frame_descriptor(
                    pair,
                    camera_position_xyz=camera,
                    camera_yaw_degrees=yaw,
                    frame_count=frame_count,
                )
            ),
        }
    return {
        "candidate_filtering": "disabled",
        "frame_count": frame_count,
        "modes": modes,
    }


__all__ = [
    "DEFAULT_FIRST_ANCHOR_FRACTION",
    "DEFAULT_SECOND_ANCHOR_FRACTION",
    "MOTION_MODES",
    "MotionFeasibilityClassification",
    "MotionFeasibilityError",
    "MotionModeEvidence",
    "MotionPairEvidence",
    "camera_side",
    "classify_motion_feasibility",
    "describe_motion_geometry",
    "describe_source_pair_tracks",
    "evaluate_motion_pair",
    "interpolate_linear_path",
    "linear_path_closest_approach",
    "minimum_linear_path_separation",
]
