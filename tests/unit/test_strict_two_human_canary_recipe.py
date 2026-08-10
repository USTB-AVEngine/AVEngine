from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_strict_two_human_canary_recipe.py"
SPEC = importlib.util.spec_from_file_location("strict_two_human_recipe", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_static_two_human_geometry_recomputes_both_human_yaws() -> None:
    plan = _load(REPOSITORY / "examples/qa/native_strict_two_human_canary_v1.json")
    registry = _load(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")

    bundle = TOOL.build_static_actor_bundle(plan, registry)

    assert [item["actor_id"] for item in bundle["declarations"]] == [
        "source1_actor",
        "source2_actor",
    ]
    assert [item["asset_id"] for item in bundle["declarations"]] == [
        "rocketbox_human_male_adult_01_m5_1_candidate",
        "lead_b_rocketbox_adults_female_adult_01_original_v1",
    ]
    for declaration in bundle["declarations"]:
        assert "exact_runtime_binding" not in declaration
        assert declaration["runtime_asset_expectation"]["source_slot_id"] == (
            declaration["actor_id"].removesuffix("_actor")
        )
        assert declaration["skeletal_mesh_path"].endswith("/runtime.runtime")
        assert declaration["skeleton_path"].endswith(
            "/runtime_Skeleton.runtime_Skeleton"
        )
        assert declaration["animation_paths_by_action_id"]["idle"] == declaration[
            "idle_animation"
        ]
        assert declaration["runtime_asset_expectation"]["schema"] == (
            "avengine_generic_human_runtime_asset_expectation_v1"
        )
    yaws = {
        item["actor_id"]: item["actor_yaw_ue_deg"]
        for item in bundle["state_templates"]
    }
    assert yaws["source1_actor"] == pytest.approx(-38.301, abs=0.01)
    assert yaws["source2_actor"] == pytest.approx(-83.564, abs=0.01)
    assert yaws["source1_actor"] != pytest.approx(-93.1818305917363)
    assert bundle["projection_offset_fraction"]["source1"] > 0.02
    assert bundle["projection_offset_fraction"]["source2"] < -0.02
    assert bundle["roots"]["source1"] == pytest.approx(
        [-1.9941582267673559, 0.4, -0.9886440992754871]
    )
    assert bundle["emitters"]["source1"][1] == pytest.approx(2.01)
    assert bundle["emitters"]["source2"][1] == pytest.approx(1.9690124571323395)
    assert bundle["projection_xy_fraction"]["source1"] == pytest.approx(
        [0.615099048196844, 0.3161825571019291]
    )
    assert bundle["projection_xy_fraction"]["source2"][0] < 0.5
    assert bundle["target_vertical_envelope_y_fraction"] == pytest.approx(
        {"top": 0.18317921251890928, "bottom": 0.8652476462779852}
    )
    assert bundle["composition_gate"]["status"] == "pass"


def test_static_two_human_geometry_fails_closed_on_identity_or_side_drift() -> None:
    plan = _load(REPOSITORY / "examples/qa/native_strict_two_human_canary_v1.json")
    registry = _load(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")

    invalid = deepcopy(plan)
    invalid["actors"][0]["expected_screen_side"] = "left"
    with pytest.raises(RuntimeError, match="screen-side dead zone"):
        TOOL.build_static_actor_bundle(invalid, registry)

    invalid = deepcopy(plan)
    invalid["actors"][1]["runtime_asset_id"] = "missing_second_identity"
    with pytest.raises(RuntimeError, match="runtime asset does not resolve once"):
        TOOL.build_static_actor_bundle(invalid, registry)

    invalid = deepcopy(plan)
    invalid["actors"][0]["root_translation_m"] = [
        -1.0850754976272583,
        0.4000000059604645,
        0.254158079624176,
    ]
    with pytest.raises(RuntimeError, match="deterministic camera-local placement"):
        TOOL.build_static_actor_bundle(invalid, registry)

    invalid = deepcopy(plan)
    invalid["deterministic_composition"]["target_vertical_envelope_from_root_m"] = [
        0.0,
        5.0,
    ]
    with pytest.raises(RuntimeError, match="vertical envelope touches"):
        TOOL.build_static_actor_bundle(invalid, registry)


def test_acoustic_binding_passes_planning_without_native_overclaim() -> None:
    plan = _load(REPOSITORY / "examples/qa/native_strict_two_human_canary_v1.json")
    registry = _load(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")
    bundle = TOOL.build_static_actor_bundle(plan, registry)

    report = TOOL.build_acoustic_binding_report(
        plan=plan,
        registry=registry,
        bundle=bundle,
    )

    assert report["status"] == "pass"
    assert report["qualification_claim"] is False
    assert report["profile_geometry_status"] == "pass"
    assert report["native_readback_status"] == "pending_required"
    scenario = report["scenarios"][0]
    assert scenario["binding_report"]["status"] == "pass"
    assert scenario["binding_report"]["qualification_claim"] is False
    assert scenario["binding_report"]["native_readback_status"] == "pending_required"
    assert {
        item["source_slot_id"]: item["native_readback"]
        for item in scenario["binding_report"]["bindings"]
    } == {
        "source1": "pending_required",
        "source2": "pending_required",
    }


def test_target_speech_window_covers_declared_frames_seven_through_thirty_one() -> None:
    start_sample, end_sample = TOOL.target_speech_sample_window(25626)

    assert (start_sample, end_sample) == (7467, 33093)
    assert (start_sample * 3, end_sample * 3) == (22401, 99279)
    assert start_sample * TOOL.FRAME_RATE_HZ // TOOL.SAMPLE_RATE_HZ == 7
    assert (end_sample - 1) * TOOL.FRAME_RATE_HZ // TOOL.SAMPLE_RATE_HZ == 31
