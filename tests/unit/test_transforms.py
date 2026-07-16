from __future__ import annotations

import math

import numpy as np
import pytest

from avengine.contracts.transforms import (
    compose_transforms,
    invert_transform,
    normalized_quaternion_xyzw,
    quaternion_rotation_error,
    rotate_vector_xyzw,
    round_trip_via_parent,
    transform_error,
    validate_transform,
)


IDENTITY = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}


def test_compose_applies_parent_rotation_before_translation() -> None:
    half_angle = math.sqrt(0.5)
    world_from_parent = {
        "translation_m": [1.0, 2.0, 3.0],
        "rotation_xyzw": [0.0, 0.0, half_angle, half_angle],
    }
    parent_from_child = {
        "translation_m": [2.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    world_from_child = compose_transforms(world_from_parent, parent_from_child)

    assert world_from_child["translation_m"] == pytest.approx([1.0, 4.0, 3.0])
    assert (
        quaternion_rotation_error(
            world_from_child["rotation_xyzw"],
            world_from_parent["rotation_xyzw"],
        )
        < 1e-12
    )


def test_inverse_composes_to_identity_for_translation_and_rotation() -> None:
    transform = {
        "translation_m": [1.25, -0.5, 4.0],
        "rotation_xyzw": [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)],
    }

    inverse = invert_transform(transform)

    assert transform_error(compose_transforms(transform, inverse), IDENTITY) < 1e-12
    assert transform_error(compose_transforms(inverse, transform), IDENTITY) < 1e-12


def test_round_trip_via_parent_recovers_world_source_transform() -> None:
    world_from_rig = {
        "translation_m": [-2.5, 0.05, 0.0],
        "rotation_xyzw": [0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)],
    }
    world_from_source = {
        "translation_m": [2.0, 0.3, -1.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    recovered, error = round_trip_via_parent(world_from_rig, world_from_source)

    assert error < 1e-12
    assert transform_error(recovered, world_from_source) < 1e-12


def test_quaternion_sign_is_treated_as_the_same_rotation() -> None:
    assert quaternion_rotation_error(
        [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]
    ) == pytest.approx(0.0)


def test_rotate_vector_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="vector must contain 3 finite numbers"):
        rotate_vector_xyzw([0.0, 0.0, 0.0, 1.0], [1.0, np.nan, 0.0])


def test_normalized_quaternion_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="non-zero norm"):
        normalized_quaternion_xyzw([0.0, 0.0, 0.0, 0.0])


def test_transform_vectors_reject_booleans_as_numbers() -> None:
    errors = validate_transform(
        {
            "translation_m": [True, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    )

    assert any(
        "translation_m must contain 3 finite numbers" in error for error in errors
    )


@pytest.mark.parametrize(
    ("transform", "expected_error"),
    [
        (
            {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 2.0],
            },
            "must already be unit normalized",
        ),
        (
            {
                "translation_m": [0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "translation_m must contain 3 finite numbers",
        ),
        (
            {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "must contain exactly translation_m and rotation_xyzw",
        ),
    ],
)
def test_validate_transform_rejects_non_canonical_values(
    transform: dict[str, list[float]], expected_error: str
) -> None:
    assert any(expected_error in error for error in validate_transform(transform))
