"""Shared Habitat listener/camera orientation conventions for M5.1.

Habitat poses are ``world_from_local`` transforms with XYZW quaternions.  The
camera/listener looks along local ``-Z``; local ``+X`` points toward the right
ear and local ``+Y`` points up.  Keeping those axes here avoids duplicating
yaw sign formulas in review-only geometry and Topdown renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np


class M51OrientationError(ValueError):
    """A Habitat camera/listener orientation is not finite and usable."""


@dataclass(frozen=True)
class HabitatBasis:
    """World-space axes of a Habitat camera/listener local frame."""

    forward_xyz: tuple[float, float, float]
    right_xyz: tuple[float, float, float]
    up_xyz: tuple[float, float, float]

    @property
    def forward_xz(self) -> tuple[float, float]:
        return (self.forward_xyz[0], self.forward_xyz[2])

    @property
    def right_xz(self) -> tuple[float, float]:
        return (self.right_xyz[0], self.right_xyz[2])


def _unit_xyzw(value: Any) -> np.ndarray:
    try:
        quaternion = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M51OrientationError(
            "rotation_xyzw must contain four finite numbers"
        ) from exc
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise M51OrientationError("rotation_xyzw must contain four finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise M51OrientationError("rotation_xyzw quaternion has zero norm")
    return quaternion / norm


def _rotate_world_from_local(vector: Sequence[float], xyzw: np.ndarray) -> np.ndarray:
    local = np.asarray(vector, dtype=np.float64)
    q_vector = xyzw[:3]
    uv = np.cross(q_vector, local)
    uuv = np.cross(q_vector, uv)
    return local + 2.0 * (float(xyzw[3]) * uv + uuv)


def habitat_basis_from_xyzw(rotation_xyzw: Any) -> HabitatBasis:
    """Rotate local ``-Z/+X/+Y`` into world space using an XYZW quaternion."""

    quaternion = _unit_xyzw(rotation_xyzw)
    forward = _rotate_world_from_local((0.0, 0.0, -1.0), quaternion)
    right = _rotate_world_from_local((1.0, 0.0, 0.0), quaternion)
    up = _rotate_world_from_local((0.0, 1.0, 0.0), quaternion)

    def frozen(axis: np.ndarray) -> tuple[float, float, float]:
        axis[np.abs(axis) < 1.0e-15] = 0.0
        return tuple(float(component) for component in axis)

    return HabitatBasis(
        forward_xyz=frozen(forward),
        right_xyz=frozen(right),
        up_xyz=frozen(up),
    )


def habitat_basis_from_yaw_degrees(yaw_degrees: Real) -> HabitatBasis:
    """Return the shared basis for a positive rotation about Habitat ``+Y``."""

    if isinstance(yaw_degrees, bool) or not isinstance(yaw_degrees, Real):
        raise M51OrientationError("yaw_degrees must be a finite number")
    yaw = float(yaw_degrees)
    if not math.isfinite(yaw):
        raise M51OrientationError("yaw_degrees must be a finite number")
    half = math.radians(yaw) * 0.5
    return habitat_basis_from_xyzw((0.0, math.sin(half), 0.0, math.cos(half)))


def habitat_yaw_degrees_from_xyzw(rotation_xyzw: Any) -> float:
    """Return Habitat ``+Y`` yaw for a normalized or normalizable XYZW pose."""

    x, y, z, w = _unit_xyzw(rotation_xyzw)
    yaw = math.degrees(
        math.atan2(
            2.0 * (float(w) * float(y) + float(x) * float(z)),
            1.0 - 2.0 * (float(y) ** 2 + float(z) ** 2),
        )
    )
    return 0.0 if abs(yaw) < 1.0e-15 else float(yaw)


__all__ = [
    "HabitatBasis",
    "M51OrientationError",
    "habitat_basis_from_xyzw",
    "habitat_basis_from_yaw_degrees",
    "habitat_yaw_degrees_from_xyzw",
]
