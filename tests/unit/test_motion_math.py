from __future__ import annotations

import math

import numpy as np
import pytest

from avengine.motion.math import (
    MotionMathError,
    canonical_quaternion_xyzw,
    quaternion_inverse_xyzw,
    quaternion_multiply_xyzw,
    retarget_world_rotation_xyzw,
    world_left_delta_xyzw,
)


def _axis_angle(axis: tuple[float, float, float], degrees: float) -> tuple[float, ...]:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half = math.radians(degrees) / 2.0
    xyz = vector * math.sin(half)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]), math.cos(half))


def _equivalent(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return min(float(np.max(np.abs(a - b))), float(np.max(np.abs(a + b)))) < 1e-12


def test_world_left_delta_uses_pose_times_inverse_rest() -> None:
    rest = _axis_angle((0.0, 0.0, 1.0), 90.0)
    delta = _axis_angle((1.0, 0.0, 0.0), 55.0)
    pose = quaternion_multiply_xyzw(delta, rest)

    assert _equivalent(world_left_delta_xyzw(pose, rest), delta)


def test_retarget_premultiplies_target_rest_for_different_bone_rolls() -> None:
    source_rest = _axis_angle((0.0, 0.0, 1.0), 90.0)
    source_delta = _axis_angle((1.0, 0.0, 0.0), 60.0)
    source_pose = quaternion_multiply_xyzw(source_delta, source_rest)
    target_rest = _axis_angle((0.0, 1.0, 0.0), 90.0)

    actual = retarget_world_rotation_xyzw(
        source_pose_world_xyzw=source_pose,
        source_rest_world_xyzw=source_rest,
        target_rest_world_xyzw=target_rest,
    )
    expected = quaternion_multiply_xyzw(source_delta, target_rest)
    assert _equivalent(actual, expected)

    # The historical right-delta order is non-commutative and differs here.
    legacy_delta = quaternion_multiply_xyzw(
        quaternion_inverse_xyzw(source_rest), source_pose
    )
    legacy = quaternion_multiply_xyzw(target_rest, legacy_delta)
    assert not _equivalent(actual, legacy)


def test_retarget_rest_pose_is_exactly_target_rest() -> None:
    source_rest = _axis_angle((1.0, 0.0, 0.0), 31.0)
    target_rest = _axis_angle((0.0, 1.0, 0.0), -47.0)
    actual = retarget_world_rotation_xyzw(
        source_pose_world_xyzw=source_rest,
        source_rest_world_xyzw=source_rest,
        target_rest_world_xyzw=target_rest,
    )
    assert _equivalent(actual, target_rest)


def test_motion_basis_conjugates_world_delta() -> None:
    source_delta = _axis_angle((1.0, 0.0, 0.0), 30.0)
    basis = _axis_angle((0.0, 0.0, 1.0), 90.0)
    actual = retarget_world_rotation_xyzw(
        source_pose_world_xyzw=source_delta,
        source_rest_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        target_rest_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        motion_basis_xyzw=basis,
    )
    assert _equivalent(actual, _axis_angle((0.0, 1.0, 0.0), 30.0))


def test_motion_amplitude_zero_keeps_target_rest() -> None:
    target_rest = _axis_angle((0.0, 1.0, 0.0), 12.0)
    actual = retarget_world_rotation_xyzw(
        source_pose_world_xyzw=_axis_angle((1.0, 0.0, 0.0), 80.0),
        source_rest_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        target_rest_world_xyzw=target_rest,
        motion_amplitude=0.0,
    )
    assert _equivalent(actual, target_rest)


def test_quaternion_canonicalization_and_invalid_amplitude() -> None:
    assert canonical_quaternion_xyzw((0.0, 0.0, 0.0, -1.0)) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    with pytest.raises(MotionMathError, match="amplitude"):
        retarget_world_rotation_xyzw(
            source_pose_world_xyzw=(0.0, 0.0, 0.0, 1.0),
            source_rest_world_xyzw=(0.0, 0.0, 0.0, 1.0),
            target_rest_world_xyzw=(0.0, 0.0, 0.0, 1.0),
            motion_amplitude=1.1,
        )
