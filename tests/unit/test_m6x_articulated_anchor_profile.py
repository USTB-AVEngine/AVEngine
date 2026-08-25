from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from avengine.capture.mixed_capture import trajectory_world_matrices
from avengine.m6x.articulated_anchor_profile import (
    AnchorProfileSpec,
    ArticulatedAnchorProfileError,
    compile_articulated_anchor_profile,
    materialize_articulated_anchor_paths,
)


def _fixture() -> tuple[np.ndarray, list[dict], AnchorProfileSpec]:
    roots = np.repeat(np.asarray([[1.0, 0.3, -2.0]]), 4, axis=0)
    matrices = trajectory_world_matrices(
        roots,
        local_forward_axis=(1.0, 0.0, 0.0),
        fallback_forward_xz=(0.0, -1.0),
    )[:, None, :, :]
    offsets = (np.asarray([0.4, 0.5, 0.0]), np.asarray([0.5, 0.6, 0.0]))
    records = []
    for frame_index in range(4):
        sample_index = frame_index % 2
        world = matrices[frame_index, 0]
        anchor = world[:3, :3] @ offsets[sample_index] + world[:3, 3]
        records.append(
            {
                "dog": {
                    "action_id": "idle",
                    "action_sample_index": sample_index,
                    "mouth_m": anchor.tolist(),
                }
            }
        )
    spec = AnchorProfileSpec(
        source_endpoint_id="dog_muzzle",
        actor_id="dog0",
        asset_id="dog_asset_v1",
        record_key="dog",
        anchor_id="muzzle",
        anchor_record_key="mouth_m",
        capture_matrix_index=0,
        local_anatomical_forward_axis=(1.0, 0.0, 0.0),
        action_sample_counts={"idle": 2, "walk": 2},
    )
    return matrices, records, spec


def test_profile_reconstructs_emitter_without_visual_render() -> None:
    matrices, records, spec = _fixture()
    profile = compile_articulated_anchor_profile(
        actor_world_matrices=matrices,
        frame_records=records,
        specs=(spec,),
    )
    roots = matrices[:, 0, :3, 3]
    actual = materialize_articulated_anchor_paths(
        profile,
        actor_root_paths={"dog0": roots},
        actor_fallback_forwards_xz={"dog0": (0.0, -1.0)},
    )["dog_muzzle"]
    expected = np.asarray([record["dog"]["mouth_m"] for record in records])
    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-12)
    assert profile["body_plan_policy"] == "profile_driven_no_species_branch"


def test_profile_rejects_inconsistent_repeated_action_sample() -> None:
    matrices, records, spec = _fixture()
    broken = deepcopy(records)
    broken[2]["dog"]["mouth_m"][0] += 0.01
    with pytest.raises(ArticulatedAnchorProfileError, match="repeat error"):
        compile_articulated_anchor_profile(
            actor_world_matrices=matrices,
            frame_records=broken,
            specs=(spec,),
        )
