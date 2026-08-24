"""Pure two-source motion geometry used by design and viewpoint selection.

The checks here intentionally cover only the existing side-margin and exact
continuous-separation requirements.  They do not claim actor visibility,
native pixels, collision volumes, or a successful Timeline/native run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence


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
    "evaluate_motion_pair",
    "interpolate_linear_path",
    "linear_path_closest_approach",
    "minimum_linear_path_separation",
]
