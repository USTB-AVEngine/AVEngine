from __future__ import annotations

from pathlib import Path

import pytest

from avengine.assets.habitat_static_assets import HabitatAssetBinding
from avengine.assets.mp3d_region_actor_tracks import (
    MP3DRegionActorTrackError,
    _common_emitter,
)


def _binding(tmp_path: Path) -> HabitatAssetBinding:
    return HabitatAssetBinding(
        asset_id="human",
        entity_class="articulated_human",
        category="human",
        asset_kind="articulated_m2_package",
        glb_path=tmp_path / "visual.glb",
        glb_relative_path="visual.glb",
        semantic_template={"template_kind": "articulated_m2"},
        resting_pose={"attachment_surface": "floor", "base_plane_offset_m": 0.0},
        emitter={
            "anchor_id": "muzzle",
            "semantic_anchor_id": "mouth",
            "offset_m": [0.0, 0.0, 0.0],
            "offset_space": "joint_local",
            "root_offset_m": [0.0, 1.641311, 0.0],
            "joint_from_anchor": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        source="runtime_registry",
        revision="controlled_material_ue_v3",
        asset_manifest_path=tmp_path / "asset_manifest.json",
        base_m2_request_path=tmp_path / "base_m2_request.json",
    )


def test_common_emitter_maps_semantic_mouth_to_native_muzzle_and_root_offset(
    tmp_path: Path,
) -> None:
    actor = {
        "emitter_binding": {
            "semantic_anchor_id": "mouth",
            "emitter_offset_m": [99.0, 99.0, 99.0],
        }
    }

    normalized, route_offset = _common_emitter(
        actor,
        binding=_binding(tmp_path),
        owner="human_0",
    )

    assert normalized["semantic_anchor_id"] == "mouth"
    assert normalized["native_anchor_id"] == "muzzle"
    assert normalized["native_offset_m"] == [0.0, 0.0, 0.0]
    assert normalized["native_offset_space"] == "joint_local"
    assert normalized["emitter_offset_m"] == [0.0, 1.641311, 0.0]
    assert route_offset.tolist() == [0.0, 1.641311, 0.0]


def test_common_emitter_rejects_semantic_anchor_mismatch(tmp_path: Path) -> None:
    with pytest.raises(MP3DRegionActorTrackError, match="semantic anchor"):
        _common_emitter(
            {"emitter_binding": {"semantic_anchor_id": "muzzle"}},
            binding=_binding(tmp_path),
            owner="human_0",
        )
