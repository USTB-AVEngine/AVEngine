from __future__ import annotations

import math

import numpy as np

from avengine.m5_1.orientation import (
    habitat_basis_from_xyzw,
    habitat_basis_from_yaw_degrees,
    habitat_yaw_degrees_from_xyzw,
)


def test_positive_90_degree_habitat_yaw_rotates_local_forward_and_ears() -> None:
    half = math.sqrt(0.5)
    basis = habitat_basis_from_xyzw((0.0, half, 0.0, half))

    # Habitat camera/listener local axes: forward=-Z, right ear=+X.
    assert np.allclose(basis.forward_xyz, (-1.0, 0.0, 0.0), atol=1.0e-12)
    assert np.allclose(basis.right_xyz, (0.0, 0.0, -1.0), atol=1.0e-12)
    assert np.allclose(basis.up_xyz, (0.0, 1.0, 0.0), atol=1.0e-12)
    assert np.isclose(
        habitat_yaw_degrees_from_xyzw((0.0, half, 0.0, half)), 90.0
    )


def test_yaw_and_quaternion_entry_points_share_one_basis() -> None:
    yaw = 55.0
    half = math.radians(yaw) * 0.5
    from_yaw = habitat_basis_from_yaw_degrees(yaw)
    from_quaternion = habitat_basis_from_xyzw(
        (0.0, math.sin(half), 0.0, math.cos(half))
    )

    assert np.allclose(from_yaw.forward_xyz, from_quaternion.forward_xyz)
    assert np.allclose(from_yaw.right_xyz, from_quaternion.right_xyz)
    assert np.allclose(from_yaw.up_xyz, from_quaternion.up_xyz)
