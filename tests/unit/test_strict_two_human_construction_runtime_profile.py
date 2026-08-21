from __future__ import annotations

import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
SOURCE_RECORD = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/human_roster_v2/source_assets/"
    "rocketbox_professions_construction_male_01.json"
)
BUILD_MANIFEST = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
    "rocketbox_batch_native_runtime_v1/"
    "rocketbox_professions_construction_male_01_original_v1/build_manifest.json"
)
NORMALIZATION_MANIFEST = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
    "rocketbox_batch_native_runtime_ue_v1/"
    "rocketbox_professions_construction_male_01_original_ue_v1/"
    "normalization_manifest.json"
)
UE_IMPORT_MANIFEST = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
    "rocketbox_batch_native_ue_import_v1/"
    "rocketbox_professions_construction_male_01_original_ue_v1/"
    "ue_import_manifest.json"
)
PACKAGE_MANIFEST = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
    "lead_b_siamese_post_approval_v1/packaged_runtime_v1/"
    "Standalone-Development/Linux/Manifest_UFSFiles_Linux.txt"
)
IDENTITY_ID = "rocketbox_professions_construction_male_01"
ASSET_ID = "lead_b_rocketbox_professions_construction_male_01_original_v1"
TAG = "gate_rocketbox_professions_construction_male_01_original_ue_v1"

_RETAINED_EVIDENCE = (
    SOURCE_RECORD,
    BUILD_MANIFEST,
    NORMALIZATION_MANIFEST,
    UE_IMPORT_MANIFEST,
    PACKAGE_MANIFEST,
)
if not all(path.is_file() for path in _RETAINED_EVIDENCE):
    pytest.skip(
        "retained lead-b construction evidence is available only on the A workspace",
        allow_module_level=True,
    )


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_construction_runtime_profile_closes_exact_lineage() -> None:
    registry = _load(REGISTRY)
    source = _load(SOURCE_RECORD)
    build = _load(BUILD_MANIFEST)
    normalization = _load(NORMALIZATION_MANIFEST)
    ue_import = _load(UE_IMPORT_MANIFEST)

    alias = registry["aliases"]["strict_two_human_construction_male"]
    assert alias == {"asset_id": ASSET_ID, "revision": "native_runtime_ue_v1"}
    matches = [asset for asset in registry["assets"] if asset["asset_id"] == ASSET_ID]
    assert len(matches) == 1
    profile = matches[0]

    assert source["asset_id"] == ASSET_ID
    assert source["base_avatar_id"] == IDENTITY_ID
    assert source["runtime"]["tag"] == f"{IDENTITY_ID}_original_ue_v1"
    assert source["runtime"]["animations"] == ["Walking", "Standing_Idle"]
    assert source["runtime"]["ue_import_readback"] == "passed"
    assert source["formal_dataset_registration_authorized"] is False

    assert build["base_avatar_id"] == IDENTITY_ID
    assert build["automatic_checks"]["overall"] == "passed"
    assert build["automatic_checks"]["actions_exactly_walk_idle"] == "passed"
    assert normalization["base_avatar_id"] == IDENTITY_ID
    assert normalization["automatic_checks"]["overall"] == "passed"
    assert normalization["expected_ue_qa"]["mouth_audio_height_cm"] == 166.4033031463623
    assert ue_import["base_avatar_id"] == IDENTITY_ID
    assert ue_import["reload_verification"]["status"] == "passed"

    content = ue_import["content"]
    unreal = profile["runtime_backends"]["spear_unreal"]
    blueprint_leaf = content["blueprint"].rsplit("/", 1)[-1]
    assert unreal["blueprint_class_path"] == (
        f"{content['blueprint']}.{blueprint_leaf}_C"
    )
    assert unreal["idle_animation"] == content["animations"]["Standing_Idle"]
    assert unreal["walking_animation"] == content["animations"]["Walking"]
    assert content["skeletal_mesh"].endswith("/runtime.runtime")
    assert content["skeleton"].endswith("/runtime_Skeleton.runtime_Skeleton")

    assert profile["revision"] == "native_runtime_ue_v1"
    assert profile["entity_class"] == "articulated_human"
    assert profile["identity"] == {"species_id": "human", "breed_id": None}
    assert profile["realized_attributes"] == {
        "life_stage": "adult",
        "sex_or_gender_label": "male",
    }
    assert profile["geometry"]["mesh_authority"] == "audited_library_asset"
    assert profile["geometry"]["source_mesh_uri"].endswith(
        "/rocketbox_professions_construction_male_01_original_ue_v1/runtime.glb"
    )
    assert profile["default_emitter_anchor_id"] == "mouth"
    assert profile["emitter_anchors"] == [
        {
            "anchor_id": "mouth",
            "anchor_type": "mouth",
            "offset_m": [0.0, 1.664033031463623, 0.0],
            "offset_space": "final_scaled_asset_root",
        }
    ]
    assert profile["admission_state"] == "research"


def test_construction_runtime_objects_are_present_in_current_package() -> None:
    package = PACKAGE_MANIFEST.read_text(encoding="utf-8")
    ue_import = _load(UE_IMPORT_MANIFEST)
    content = ue_import["content"]

    assert package.count(TAG) >= 5
    for required in (
        content["blueprint"],
        content["skeletal_mesh"],
        content["skeleton"],
        content["animations"]["Standing_Idle"],
        content["animations"]["Walking"],
    ):
        package_fragment = required.split(".", 1)[0].lower().replace(
            "/game/", "spearsim/content/"
        )
        assert package_fragment in package.lower()
