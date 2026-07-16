from __future__ import annotations

from numbers import Real
from typing import Any, Iterable

import numpy as np


IDENTITY_TRANSFORM: dict[str, list[float]] = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}


def _vector(values: Iterable[float], length: int, name: str) -> np.ndarray:
    items = list(values)
    if len(items) != length or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in items
    ):
        raise ValueError(f"{name} must contain {length} finite numbers")
    result = np.asarray(items, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def normalized_quaternion_xyzw(values: Iterable[float]) -> np.ndarray:
    quat = _vector(values, 4, "rotation_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("rotation_xyzw must have non-zero norm")
    return quat / norm


def validate_transform(value: Any, *, name: str = "transform") -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{name} must be an object"]
    if set(value) != {"translation_m", "rotation_xyzw"}:
        errors.append(f"{name} must contain exactly translation_m and rotation_xyzw")
    try:
        _vector(value.get("translation_m", []), 3, f"{name}.translation_m")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    try:
        quat = normalized_quaternion_xyzw(value.get("rotation_xyzw", []))
        original = np.asarray(value.get("rotation_xyzw", []), dtype=np.float64)
        if original.shape == (4,) and not np.allclose(original, quat, atol=1e-9):
            errors.append(f"{name}.rotation_xyzw must already be unit normalized")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def quaternion_conjugate_xyzw(quat: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalized_quaternion_xyzw(quat)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quaternion_multiply_xyzw(
    left: Iterable[float], right: Iterable[float]
) -> np.ndarray:
    x1, y1, z1, w1 = normalized_quaternion_xyzw(left)
    x2, y2, z2, w2 = normalized_quaternion_xyzw(right)
    return normalized_quaternion_xyzw(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def rotate_vector_xyzw(quat: Iterable[float], vector: Iterable[float]) -> np.ndarray:
    q = normalized_quaternion_xyzw(quat)
    v = _vector(vector, 3, "vector")
    q_xyz = q[:3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q[3] * t + np.cross(q_xyz, t)


def invert_transform(value: dict[str, list[float]]) -> dict[str, list[float]]:
    translation = _vector(value["translation_m"], 3, "translation_m")
    inverse_rotation = quaternion_conjugate_xyzw(value["rotation_xyzw"])
    inverse_translation = -rotate_vector_xyzw(inverse_rotation, translation)
    return {
        "translation_m": inverse_translation.tolist(),
        "rotation_xyzw": inverse_rotation.tolist(),
    }


def compose_transforms(
    world_from_parent: dict[str, list[float]],
    parent_from_child: dict[str, list[float]],
) -> dict[str, list[float]]:
    parent_translation = _vector(
        world_from_parent["translation_m"], 3, "world_from_parent.translation_m"
    )
    child_translation = _vector(
        parent_from_child["translation_m"], 3, "parent_from_child.translation_m"
    )
    rotation = quaternion_multiply_xyzw(
        world_from_parent["rotation_xyzw"], parent_from_child["rotation_xyzw"]
    )
    translation = parent_translation + rotate_vector_xyzw(
        world_from_parent["rotation_xyzw"], child_translation
    )
    return {
        "translation_m": translation.tolist(),
        "rotation_xyzw": rotation.tolist(),
    }


def quaternion_rotation_error(left: Iterable[float], right: Iterable[float]) -> float:
    left_q = normalized_quaternion_xyzw(left)
    right_q = normalized_quaternion_xyzw(right)
    return float(
        min(np.linalg.norm(left_q - right_q), np.linalg.norm(left_q + right_q))
    )


def transform_error(
    left: dict[str, list[float]], right: dict[str, list[float]]
) -> float:
    translation_error = float(
        np.linalg.norm(
            _vector(left["translation_m"], 3, "left.translation_m")
            - _vector(right["translation_m"], 3, "right.translation_m")
        )
    )
    rotation_error = quaternion_rotation_error(
        left["rotation_xyzw"], right["rotation_xyzw"]
    )
    return max(translation_error, rotation_error)


def round_trip_via_parent(
    world_from_parent: dict[str, list[float]],
    world_from_child: dict[str, list[float]],
) -> tuple[dict[str, list[float]], float]:
    parent_from_world = invert_transform(world_from_parent)
    parent_from_child = compose_transforms(parent_from_world, world_from_child)
    recovered = compose_transforms(world_from_parent, parent_from_child)
    return recovered, transform_error(recovered, world_from_child)
