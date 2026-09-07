from __future__ import annotations

from pathlib import Path

import pytest

from avengine.assets.habitat_static_assets import (
    HabitatStaticAssetError,
    load_habitat_asset_bindings,
    make_binding_delta,
    summarize_asset_inventory,
)
from avengine.assets.mp3d_region_actor_tracks import materialize_habitat_rigid_track


REPOSITORY = Path(__file__).resolve().parents[2]
RUNTIME_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
EXTERNAL_INDEX = Path("/data/avengine_external/assets/sound_source_assets_v1/index.json")
BEAGLE_MANIFEST = Path(
    "/data/avengine_external/datasets/m2/"
    "rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json"
)
BEAGLE_REQUEST = Path(
    "/data/avengine_external/review/"
    "current_mp3d_two_beagle_route_lateral_seed22_grounded/primary_m2_request.json"
)
SPEAKER_ID = "generated_bookshelf_speaker_compact_shelf_cabinet_black_ash_research_v1"


@pytest.mark.skipif(not EXTERNAL_INDEX.is_file(), reason="external sound-source index unavailable")
def test_inventory_preserves_external_animal_and_runtime_overlap() -> None:
    summary = summarize_asset_inventory(
        runtime_registry_path=RUNTIME_REGISTRY,
        external_index_path=EXTERNAL_INDEX,
    )
    assert summary["external_total"] == 44
    assert summary["external_by_entity_class"] == {
        "articulated_animal": 4,
        "rigid_object": 40,
    }
    assert summary["overlap_count"] == 44
    assert summary["union_total"] == 59
    assert summary["union_by_entity_class"] == {
        "articulated_animal": 12,
        "articulated_human": 7,
        "rigid_object": 40,
    }


@pytest.mark.skipif(not EXTERNAL_INDEX.is_file(), reason="external sound-source index unavailable")
def test_speaker_binding_resolves_glb_resting_pose_and_emitter() -> None:
    binding = load_habitat_asset_bindings(
        [SPEAKER_ID],
        runtime_registry_path=RUNTIME_REGISTRY,
        external_index_path=EXTERNAL_INDEX,
    )[SPEAKER_ID]
    assert binding.normalized_entity_class == "rigid_object"
    assert binding.category == "audio_playback"
    assert binding.glb_path.is_file()
    assert binding.resting_pose["attachment_surface"] == "floor"
    assert binding.emitter["anchor_id"] == "woofer_cone_front_baffle"
    assert len(binding.emitter["offset_m"]) == 3


@pytest.mark.skipif(not EXTERNAL_INDEX.is_file(), reason="external sound-source index unavailable")
def test_external_animal_uses_p12_binding_and_never_falls_back_to_rigid() -> None:
    animal_id = "generated_burmese_dark_sable_research_v1"
    binding = load_habitat_asset_bindings(
        [animal_id],
        runtime_registry_path=RUNTIME_REGISTRY,
        external_index_path=EXTERNAL_INDEX,
    )[animal_id]
    assert binding.normalized_entity_class == "articulated_animal"
    assert binding.asset_kind == "articulated_m2_package"
    assert binding.asset_manifest_path.is_file()
    assert binding.base_m2_request_path.is_file()
    # With no P12 registry, the external animal is still rejected as a static source.
    with pytest.raises(HabitatStaticAssetError, match="articulated assets require P12"):
        load_habitat_asset_bindings(
            [animal_id],
            external_index_path=EXTERNAL_INDEX,
        )


@pytest.mark.skipif(not EXTERNAL_INDEX.is_file(), reason="external sound-source index unavailable")
def test_binding_delta_has_forty_rigid_entries_and_four_exclusions() -> None:
    delta = make_binding_delta(
        runtime_registry_path=RUNTIME_REGISTRY,
        external_index_path=EXTERNAL_INDEX,
    )
    assert len(delta["bindings"]) == 40
    assert len(delta["excluded"]) == 4
    assert all(item["runtime_backend"]["entity_class"] == "rigid_static_object"
               for item in delta["bindings"])
    assert {item["entity_class"] for item in delta["excluded"]} == {"articulated_animal"}


@pytest.mark.skipif(not EXTERNAL_INDEX.is_file(), reason="external sound-source index unavailable")
def test_rigid_materialization_uses_emitter_and_resting_pose() -> None:
    binding = load_habitat_asset_bindings(
        [SPEAKER_ID],
        runtime_registry_path=RUNTIME_REGISTRY,
        external_index_path=EXTERNAL_INDEX,
    )[SPEAKER_ID].to_dict()
    request = {
        "sources": [
            {
                "source_id": "speaker_muzzle",
                "world_from_source": {
                    "translation_m": [-8.0, 0.173368, -3.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ]
    }
    clock = {
        "frame_count": 30,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "sample_count": 32000,
        "time_base_hz": 48000,
        "ticks_per_frame": 3200,
    }
    track = materialize_habitat_rigid_track(
        actor={
            "actor_id": "speaker_0",
            "source_slot_id": "source1",
            "source_endpoint_id": "speaker_muzzle",
            "semantic_id": 333,
            "asset_id": SPEAKER_ID,
            "asset_revision": "sound_source_assets_v1",
            "entity_class": "rigid_object",
        },
        m1_request=request,
        clock=clock,
        habitat_binding=binding,
        floor_height_m=0.072447,
    )
    assert track["entity_class"] == "rigid_object"
    assert len(track["frames"]) == 30
    assert track["frames"][0]["action_id"] == "static"
    assert track["frames"][0]["planned_world_from_object"]["translation_m"][1] == pytest.approx(0.072447)
    assert track["frames"][0]["joint_targets"] == []


@pytest.mark.skipif(
    not EXTERNAL_INDEX.is_file()
    or not BEAGLE_MANIFEST.is_file()
    or not BEAGLE_REQUEST.is_file(),
    reason="external sound-source index or beagle M2 package unavailable",
)
def test_binding_delta_can_register_beagle_m2_habitat_package() -> None:
    delta = make_binding_delta(
        runtime_registry_path=RUNTIME_REGISTRY,
        external_index_path=EXTERNAL_INDEX,
        beagle_asset_manifest_path=BEAGLE_MANIFEST,
        beagle_m2_request_path=BEAGLE_REQUEST,
    )
    beagle = next(
        item
        for item in delta["bindings"]
        if item["asset_id"] == "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
    )
    backend = beagle["runtime_backend"]
    assert backend["asset_kind"] == "articulated_m2_package"
    assert backend["semantic_template"]["template_kind"] == "articulated_m2"
    assert Path(backend["asset_manifest_path"]).is_file()
    assert Path(backend["base_m2_request_path"]).is_file()
