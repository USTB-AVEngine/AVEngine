"""Pure NumPy quaternion math for rest-pose-aware motion retargeting."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np


QuaternionXYZW = tuple[float, float, float, float]
_ZERO_TOLERANCE = 1.0e-15


class MotionMathError(ValueError):
    """A motion transform is non-finite, degenerate, or out of contract."""


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _sign_component(quaternion: Sequence[float]) -> float:
    scalar = float(quaternion[3])
    if math.isclose(scalar, 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE):
        for component in quaternion[:3]:
            number = float(component)
            if not math.isclose(number, 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE):
                return number
    return scalar


def canonical_quaternion_xyzw(
    value: Sequence[float], *, owner: str = "quaternion"
) -> QuaternionXYZW:
    """Return one finite unit quaternion in a deterministic hemisphere."""

    try:
        quaternion = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MotionMathError(f"{owner} must contain four finite numbers") from exc
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise MotionMathError(f"{owner} must contain four finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise MotionMathError(f"{owner} cannot be a zero quaternion")
    quaternion /= norm
    if _sign_component(quaternion) < 0.0:
        quaternion = -quaternion
    return tuple(_canonical_float(component) for component in quaternion)  # type: ignore[return-value]


def quaternion_multiply_xyzw(
    left: Sequence[float], right: Sequence[float]
) -> QuaternionXYZW:
    """Compose two rotations using Hamilton multiplication in xyzw order."""

    lx, ly, lz, lw = canonical_quaternion_xyzw(left, owner="left quaternion")
    rx, ry, rz, rw = canonical_quaternion_xyzw(right, owner="right quaternion")
    return canonical_quaternion_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        owner="quaternion product",
    )


def quaternion_inverse_xyzw(value: Sequence[float]) -> QuaternionXYZW:
    """Invert one unit rotation."""

    x, y, z, w = canonical_quaternion_xyzw(value)
    return canonical_quaternion_xyzw((-x, -y, -z, w))


def _slerp_identity_xyzw(value: Sequence[float], amplitude: float) -> QuaternionXYZW:
    if (
        isinstance(amplitude, bool)
        or not isinstance(amplitude, Real)
        or not math.isfinite(float(amplitude))
        or not 0.0 <= float(amplitude) <= 1.0
    ):
        raise MotionMathError("motion amplitude must be a finite number in [0, 1]")
    target = np.asarray(canonical_quaternion_xyzw(value), dtype=np.float64)
    identity = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    dot = float(np.dot(identity, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    amount = float(amplitude)
    if dot > 0.9995:
        result = identity + amount * (target - identity)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        result = (
            math.sin((1.0 - amount) * theta) / sin_theta * identity
            + math.sin(amount * theta) / sin_theta * target
        )
    return canonical_quaternion_xyzw(result, owner="scaled motion delta")


def world_left_delta_xyzw(
    source_pose_world_xyzw: Sequence[float],
    source_rest_world_xyzw: Sequence[float],
) -> QuaternionXYZW:
    """Return the source world-space pose delta.

    A world-space delta premultiplies a rest orientation.  The order is
    intentionally ``pose * inverse(rest)``.  The historical AVEngine animal
    retarget used the opposite rest-local/right-delta order, which can rotate a
    sagittal gait into the target limb's lateral plane when bone rolls differ.
    """

    return quaternion_multiply_xyzw(
        source_pose_world_xyzw,
        quaternion_inverse_xyzw(source_rest_world_xyzw),
    )


def retarget_world_rotation_xyzw(
    *,
    source_pose_world_xyzw: Sequence[float],
    source_rest_world_xyzw: Sequence[float],
    target_rest_world_xyzw: Sequence[float],
    motion_basis_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    motion_amplitude: float = 1.0,
) -> QuaternionXYZW:
    """Retarget one source world rotation onto one target rest frame.

    ``motion_basis_xyzw`` maps the source motion coordinate frame into the
    target motion frame.  Its conjugation keeps the operation valid for any
    audited skeleton and avoids special-casing axis-aligned dogs.  Semantic
    chain selection and species/body-plan policy remain profile-owned.
    """

    delta = world_left_delta_xyzw(source_pose_world_xyzw, source_rest_world_xyzw)
    basis = canonical_quaternion_xyzw(
        motion_basis_xyzw, owner="motion basis quaternion"
    )
    aligned_delta = quaternion_multiply_xyzw(
        quaternion_multiply_xyzw(basis, delta),
        quaternion_inverse_xyzw(basis),
    )
    scaled_delta = _slerp_identity_xyzw(aligned_delta, motion_amplitude)
    return quaternion_multiply_xyzw(scaled_delta, target_rest_world_xyzw)
