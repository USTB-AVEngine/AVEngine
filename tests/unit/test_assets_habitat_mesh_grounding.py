from __future__ import annotations

import numpy as np
import pytest

from tools.assets.audit_habitat_mesh_grounding import (
    GroundingAuditError,
    _pose_joint_matrices,
    _skin_actor_vertices,
)


def _mapping() -> dict[str, object]:
    return {
        "actor_from_skin_root": np.eye(4).tolist(),
        "root_joint_id": "bone_0",
        "joint_order": ["bone_0", "bone_1"],
        "runtime_joint_order": ["bone_1"],
        "joints": [
            {
                "joint_id": "bone_0",
                "parent_joint_id": None,
                "local_translation_m": [0.0, 0.0, 0.0],
            },
            {
                "joint_id": "bone_1",
                "parent_joint_id": "bone_0",
                "local_translation_m": [0.0, 1.0, 0.0],
            },
        ],
    }


def test_pose_joint_matrices_preserve_parent_translation() -> None:
    matrices = _pose_joint_matrices(
        _mapping(), np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    )

    assert np.allclose(matrices["bone_0"], np.eye(4))
    assert np.allclose(matrices["bone_1"][:3, 3], [0.0, 1.0, 0.0])


def test_skin_actor_vertices_applies_per_vertex_weights_and_actor_frame() -> None:
    actor_from_skin_root = np.eye(4)
    actor_from_skin_root[:3, 3] = [2.0, 3.0, 4.0]
    joint_translation = np.eye(4)
    joint_translation[0, 3] = 1.0

    result = _skin_actor_vertices(
        positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        joints=np.asarray([[0, 1, 0, 0]], dtype=np.int64),
        weights=np.asarray([[0.25, 0.75, 0.0, 0.0]], dtype=np.float64),
        inverse_bind=np.stack([np.eye(4), np.eye(4)]),
        mesh_node_global=np.eye(4),
        actor_from_skin_root=actor_from_skin_root,
        joint_matrices={"bone_0": np.eye(4), "bone_1": joint_translation},
        skin_joint_names=["bone_0", "bone_1"],
    )

    assert result.shape == (1, 3)
    assert np.allclose(result[0], [2.75, 3.0, 4.0])


def test_pose_joint_matrices_rejects_wrong_runtime_shape() -> None:
    with pytest.raises(GroundingAuditError, match="pose shape"):
        _pose_joint_matrices(_mapping(), np.zeros((2, 4), dtype=np.float64))
