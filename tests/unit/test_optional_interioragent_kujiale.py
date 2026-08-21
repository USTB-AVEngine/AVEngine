from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.optional_backends.interioragent_kujiale import (
    BACKEND_ROLE,
    DATASET_ID,
    InteriorAgentPlanError,
    build_kujiale_review_plan,
    load_profile,
    load_room_metadata,
    preview_material_parameters,
    usd_meters_to_unreal_cm,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE = (
    REPOSITORY / "examples/m6z/interioragent_kujiale_0020_visual_profile.json"
)


def _profile() -> dict:
    return load_profile(PROFILE)


def test_compiles_external_usd_review_without_claiming_episode_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "kujiale_0020.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    rooms_path = tmp_path / "rooms.json"
    rooms_path.write_text(
        json.dumps(
            [
                {
                    "room_type": "living room",
                    "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                }
            ]
        ),
        encoding="utf-8",
    )

    plan = build_kujiale_review_plan(
        _profile(),
        source_stage=source,
        rooms=load_room_metadata(rooms_path),
    )

    assert plan["backend_role"] == BACKEND_ROLE
    assert BACKEND_ROLE == "production_visual"
    assert plan["dataset_id"] == DATASET_ID
    assert plan["source_stage"] == str(source.resolve())
    assert plan["room_polygons_xy_m"] == [
        [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    ]
    assert plan["camera_views"][0]["position_ue_cm"] == [-100.0, 0.0, 150.0]
    assert plan["review_lights"][1]["position_ue_cm"] == [
        -298.6,
        212.8,
        220.00000000000003,
    ]
    assert plan["external_asset_policy"]["redistribute_downloaded_data"] is False
    assert plan["external_asset_policy"]["repository_contains_downloaded_data"] is False
    assert (
        plan["authority"]["timeline_navigation_audio_topdown_flags_metadata"]
        == "habitat_native_avengine"
    )
    assert plan["authority"]["review_lights_are_acoustic_truth"] is False
    assert plan["authority"]["visual_pixels"] == "spear_ue_production_visual"


def test_preview_material_translation_retains_color_pbr_and_texture_flags() -> None:
    translated = preview_material_parameters(
        {
            "BaseColor_Color": (0.2, 0.4, 0.6),
            "Gloss_Color": (0.8, 0.8, 0.8),
            "Metallic_Color": (0.3, 0.3, 0.3),
            "Opacity": 0.7,
            "FresnelB": 1.45,
            "IsBaseColorTex": 1,
            "IsNormalTex": 1,
            "EmissiveIntensity": 0.5,
            "Emissive_Color": (0.5, 0.25, 0.0),
        }
    )

    assert translated["base_color"] == pytest.approx((0.2, 0.4, 0.6))
    assert translated["roughness"] == pytest.approx(0.2)
    assert translated["metallic"] == pytest.approx(0.3)
    assert translated["opacity"] == pytest.approx(0.7)
    assert translated["ior"] == pytest.approx(1.45)
    assert translated["use_base_texture"] is True
    assert translated["use_normal_texture"] is True
    assert translated["emissive_color"] == pytest.approx((0.25, 0.125, 0.0))


def test_glass_translation_is_explicit_and_bounded() -> None:
    translated = preview_material_parameters(
        {
            "BaseColor_Color": (0.1, 0.2, 0.3),
            "Gloss_Color": (0.1, 0.1, 0.1),
            "Metallic_Color": (1.0, 1.0, 1.0),
            "Opacity": 1.0,
        },
        mdl_source="OmniGlass.mdl",
    )

    assert translated["is_glass"] is True
    assert translated["base_color"] == pytest.approx((0.92, 0.97, 1.0))
    assert translated["roughness"] == pytest.approx(0.04)
    assert translated["metallic"] == 0.0
    assert translated["opacity"] == pytest.approx(0.12)


def test_profile_rejects_paths_as_scope_names_and_bad_light_geometry() -> None:
    profile = _profile()
    profile["selected_scopes"][0] = "../livingroom"
    with pytest.raises(InteriorAgentPlanError, match="selected_scopes"):
        build_kujiale_review_plan(profile)

    profile = _profile()
    profile["review_lights"][0]["soft_source_radius_m"] = 0.01
    with pytest.raises(InteriorAgentPlanError, match="photometric"):
        build_kujiale_review_plan(profile)


def test_profile_rejects_duplicate_camera_ids_and_absent_room() -> None:
    profile = _profile()
    profile["camera_views"][1]["view_id"] = profile["camera_views"][0]["view_id"]
    with pytest.raises(InteriorAgentPlanError, match="duplicate camera"):
        build_kujiale_review_plan(profile)

    with pytest.raises(InteriorAgentPlanError, match="absent"):
        build_kujiale_review_plan(
            _profile(),
            rooms=[
                {
                    "room_type": "bedroom",
                    "polygon_xy_m": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                }
            ],
        )


def test_profile_requires_exactly_four_review_views() -> None:
    profile = _profile()
    profile["camera_views"].pop()
    with pytest.raises(InteriorAgentPlanError, match="exactly four"):
        build_kujiale_review_plan(profile)


def test_coordinate_conversion_is_direct_usd_to_ue_centimeters() -> None:
    assert usd_meters_to_unreal_cm((1.25, -2.5, 1.6)) == (
        125.0,
        -250.0,
        160.0,
    )
