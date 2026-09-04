"""F2 generator consumers: domains, front/back shape, and Arc output."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import scene_sampler as SS  # noqa: E402
from design_qa_v3_scene_batch import (  # noqa: E402
    build_answer,
    materialize_emitter_paths,
    profile_query_geometry,
    recompute_azimuth,
    recompute_emitter_azimuth,
    validate_profiles,
)


PARAMS = {
    "THETA_FULL": 15.0,
    "THETA_HALF": 30.0,
    "MIN_AZIMUTH_SEP": 25.0,
    "MIN_CAMERA_DISTANCE_CM": 100.0,
    "VIDEO_FPS": 15,
}


def _timeline(azimuth_deg: float):
    radius = 400.0
    target = [radius * math.cos(math.radians(azimuth_deg)),
              radius * math.sin(math.radians(azimuth_deg)), 0.0]
    other = [radius * math.cos(math.radians(azimuth_deg + 90.0)),
             radius * math.sin(math.radians(azimuth_deg + 90.0)), 0.0]
    return {
        "room": {"map_path": "/Game/x", "room_profile_id": "x"},
        "render": {"frame_count": 75, "frame_rate_hz": 15,
                   "hfov_degrees": 105.0, "resolution_hw": [720, 1280]},
        "frames": [
            {"frame_index": frame,
             "camera": {"translation_ue_cm": [0.0, 0.0, 147.1],
                        "yaw_ue_deg": 0.0},
             "actor_states": [
                 {"source_slot_id": "source1",
                  "translation_ue_cm": target, "yaw_ue_deg": 0.0},
                 {"source_slot_id": "source2",
                  "translation_ue_cm": other, "yaw_ue_deg": 0.0},
             ]}
            for frame in range(75)
        ],
    }


def _front_back_profile():
    return {
        "id": "f2_front_back",
        "temporal": "instant",
        "answer_kind": "front_back",
        "binding_frames": [30],
        "idle_choices": [0],
        "anchor_binding": "query_caller",
        "answer_domain": "front_back",
        "answer_shape": {"binary_labels": ["front", "back"]},
        "vocab_key": "front_back",
        "query_geometry_by_answer": {
            "front": {"query_domain": "camera_cone",
                      "query_requires_visibility": True,
                      "secondary_anchor_bound_deg": 180.0,
                      "secondary_query_bound_deg": 180.0},
            "back": {"query_domain": "rear_cone",
                     "query_requires_visibility": False},
        },
        "convention": "dcase_foa_left_positive",
        "answer_bands_deg": [[-47.5, 47.5], [132.5, -132.5]],
    }


def test_emitter_truth_uses_registry_anchor_and_timeline_yaw():
    registry = {
        "assets": [
            {
                "asset_id": "dog_a",
                "revision": "r1",
                "default_emitter_anchor_id": "muzzle",
                "emitter_anchors": [
                    {"anchor_id": "muzzle", "offset_m": [0.4, 0.6, 0.0],
                     "offset_space": "final_scaled_asset_root"}
                ],
                "runtime_backends": {"spear_unreal": {}},
            },
            {
                "asset_id": "dog_b",
                "revision": "r2",
                "default_emitter_anchor_id": "muzzle",
                "emitter_anchors": [
                    {"anchor_id": "muzzle", "offset_m": [0.2, 0.5, 0.0],
                     "offset_space": "final_scaled_asset_root"}
                ],
                "runtime_backends": {"spear_unreal": {}},
            },
        ]
    }
    timeline = _timeline(20.0)
    timeline["actors"] = [
        {"source_slot_id": "source1"},
        {"source_slot_id": "source2"},
    ]
    paths, metadata = materialize_emitter_paths(
        timeline, registry, {"source1": "dog_a", "source2": "dog_b"})
    root_az = recompute_azimuth(timeline, "source1", 30)
    emitter_az = recompute_emitter_azimuth(
        timeline, "source1", 30, paths)
    assert abs(emitter_az - root_az) > 0.1
    assert metadata["source1"]["emitter_anchor_id"] == "muzzle"
    assert metadata["source1"]["offset_space"] == "final_scaled_asset_root"
    assert metadata["source1"]["frame_rate_hz"] == pytest.approx(15.0)


def test_front_back_domain_derives_two_disjoint_configured_arcs():
    profile = _front_back_profile()
    arcs = SS.derive_answer_arcs(
        profile, SimpleNamespace(hfov_deg=105.0),
        {"VISUAL_FOV_MARGIN_DEG": 5.0})
    assert len(arcs) == 2
    assert arcs[0].width_deg == pytest.approx(95.0)
    assert arcs[1].width_deg == pytest.approx(95.0)
    with pytest.raises(ValueError, match="regions overlap"):
        SS.derive_answer_arcs(
            dict(profile, answer_shape={
                "binary_labels": ["front", "back"],
                "front_arc": [0.0, 100.0],
                "back_arc": [90.0, 180.0],
            }),
            SimpleNamespace(hfov_deg=105.0),
            {"VISUAL_FOV_MARGIN_DEG": 5.0})


def test_front_back_shape_is_distinct_from_rear_cone_and_configurable():
    profile = _front_back_profile()
    validate_profiles([profile])
    front = profile_query_geometry(profile, {"answer_band": [-47.5, 47.5]})
    back = profile_query_geometry(profile, {"answer_band": [132.5, -132.5]})
    assert front["query_domain"] == "camera_cone"
    assert front["query_requires_visibility"] is True
    assert front["secondary_anchor_bound_deg"] == pytest.approx(180.0)
    assert front["secondary_query_bound_deg"] == pytest.approx(180.0)
    assert back["query_domain"] == "rear_cone"
    assert back["query_requires_visibility"] is False


def test_front_back_builds_two_label_gold_and_convention_metadata():
    profile = _front_back_profile()
    main = build_answer(
        "front_back", profile, {"answer_band": [-47.5, 47.5]},
        _timeline(20.0), None, [], "source1", "source2",
        {"source1": "black-and-white", "source2": "yellow"},
        20.0, 30, PARAMS,
    )
    assert main["truth"]["answer_label"] == "front"
    assert main["mcq"]["options_space"] == ["front", "back"]
    assert main["mcq"]["truth_option"] == "front"
    assert main["open"]["truth_value"] == "front"
    assert main["open"]["scoring"] == "closed_set"
    assert main["open"]["vocab_key"] == "front_back"
    assert main["truth"]["arc_engine_frame"]["schema"] == "avengine_qa_v3_arc_v1"
    assert main["mcq"]["convention"] == "dcase_foa_left_positive"
    assert main["open"]["truth_arc"]["schema"] == "avengine_qa_v3_arc_v1"


def test_full_circle_labels_keep_both_signed_180_edges():
    profile = {
        "id": "f2_full",
        "temporal": "instant",
        "answer_kind": "instant_azimuth_band",
        "binding_frames": [30],
        "answer_domain": "full_circle",
        "answer_shape": {"equal_bands": 4},
        "answer_bands_deg": [
            [-180.0, -90.0], [-90.0, 0.0],
            [0.0, 90.0], [90.0, 180.0],
        ],
        "anchor_binding": "query_caller",
        "idle_choices": [0],
        "convention": "dcase_foa_left_positive",
    }
    result = build_answer(
        "instant_azimuth_band", profile, {"answer_band": [90.0, 180.0]},
        _timeline(170.0), None, [], "source1", "source2",
        {"source1": "black-and-white", "source2": "yellow"},
        170.0, 30, PARAMS,
    )
    assert result["mcq"]["options_space"][0] == "[90, 180)"
    assert result["mcq"]["options_space"][-1] == "[-180, -90)"
    assert result["mcq"]["option_arcs"][0]["published"]["sweep_deg"] == pytest.approx(-90.0)
    assert result["mcq"]["option_arcs"][-1]["published"]["sweep_deg"] == pytest.approx(-90.0)


def test_numeric_wrap_answer_keeps_recoverable_published_arc():
    profile = {
        "id": "f2_rear",
        "temporal": "instant",
        "answer_kind": "instant_azimuth_band",
        "binding_frames": [30],
        "answer_domain": "rear_cone",
        "answer_shape": {"equal_bands": 1},
        "answer_bands_deg": [[170.0, -170.0]],
        "anchor_binding": "query_caller",
        "idle_choices": [0],
        "convention": "dcase_foa_left_positive",
    }
    result = build_answer(
        "instant_azimuth_band", profile, {"answer_band": [170.0, -170.0]},
        _timeline(175.0), None, [], "source1", "source2",
        {"source1": "black-and-white", "source2": "yellow"},
        175.0, 30, PARAMS,
    )
    assert result["mcq"]["options_space"][0].startswith("-170")
    option = result["mcq"]["option_arcs"][0]
    assert option["engine_frame"]["sweep_deg"] == pytest.approx(20.0)
    assert option["published"]["sweep_deg"] == pytest.approx(-20.0)
    assert result["open"]["truth_interval_arc"]["schema"] == "avengine_qa_v3_arc_v1"
    assert result["truth"]["answer_domain"] == "rear_cone"




def test_front_back_label_cannot_change_inside_query_window():
    from design_qa_v3_scene_batch import GenerationConstraintError
    timeline = _timeline(20)
    timeline["frames"][34]["actor_states"][0]["translation_ue_cm"] = [-400, 0, 0]
    with pytest.raises(GenerationConstraintError, match="crosses"):
        build_answer("front_back", _front_back_profile(),
                     {"answer_band": [-47.5, 47.5]}, timeline, None, [],
                     "source1", "source2", {"source1": "black", "source2": "yellow"},
                     20.0, 30, PARAMS)


def test_wide_band_cannot_accept_sweep_through_its_excluded_gap():
    from design_qa_v3_scene_batch import GenerationConstraintError
    timeline = _timeline(160)
    for frame in range(30, 38):
        angle = math.radians(160 + (frame - 30) * 40 / 7)
        timeline["frames"][frame]["actor_states"][0]["translation_ue_cm"] = [
            400 * math.cos(angle), 400 * math.sin(angle), 0]
    profile = {"id": "wide_arc", "temporal": "instant",
               "answer_kind": "instant_azimuth_band",
               "answer_bands_deg": [[-170, 170], [170, -170]]}
    with pytest.raises(GenerationConstraintError, match="crosses band"):
        build_answer("instant_azimuth_band", profile, {"answer_band": [-170, 170]},
                     timeline, None, [], "source1", "source2",
                     {"source1": "black", "source2": "yellow"}, 160, 30, PARAMS)
