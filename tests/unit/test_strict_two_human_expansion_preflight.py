from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_strict_two_human_expansion_preflight.py"
TOOL_SPEC = importlib.util.spec_from_file_location(
    "build_strict_two_human_expansion_preflight", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _inputs() -> tuple[dict, dict]:
    plan = _load(REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json")
    registry = _load(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")
    return plan, registry


def test_strict_eight_plan_contract_and_balance_pass() -> None:
    plan, registry = _inputs()
    assert TOOL.validate_plan(plan, registry) == []

    rows = plan["rows"]
    assert len(rows) == 8
    assert [row["identity_pair"] for row in rows] == [
        "M/F",
        "M/F",
        "F/M",
        "F/M",
        "M/C",
        "C/M",
        "F/C",
        "C/F",
    ]
    assert [row["target_expected_screen_side"] for row in rows] == [
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
    ]
    assert {
        identity["original_identity_id"]
        for identity in plan["approved_identity_catalog"].values()
    } == {
        "rocketbox_adults_male_adult_01",
        "rocketbox_adults_female_adult_01",
        "rocketbox_professions_construction_male_01",
    }
    assert plan["formal_scene_count"] == 0
    assert plan["qualification_claim"] is False
    assert plan["execution_policy"]["gpu_or_rir_allowed_in_this_atom"] is False
    assert {
        key: identity["expected_speech_frame_window_inclusive"]
        for key, identity in plan["approved_identity_catalog"].items()
    } == {"M": [7, 31], "F": [7, 50], "C": [7, 50]}
    assert plan["timeline"]["target_speech_duration_policy"] == "full_dry_asset"
    thresholds = plan["projection_and_native_thresholds"]
    assert thresholds["target_visible_fraction_minimum"] == 0.8
    assert thresholds["distractor_visible_fraction_minimum"] == 0.5


def test_row7_v1_rejection_record_binds_retained_native_failure() -> None:
    plan, _ = _inputs()
    rejection_path = REPOSITORY / plan["evidence"]["rejected_row7_v1"]
    rejection = _load(rejection_path)
    row7 = plan["rows"][6]

    assert rejection["status"] == "rejected"
    assert rejection["decision"] == "fail"
    assert rejection["row_id"] == row7["row_id"]
    assert rejection["episode_id"] == row7["episode_id"]
    assert rejection["frame_index"] == plan["timeline"]["sparse_gate_frame_index"]
    assert rejection["formal_scene_count"] == 0
    assert rejection["qualification_claim"] is False
    assert rejection["original_output_mutated"] is False
    assert rejection["row8_executed"] is False

    pixels = _load(Path(rejection["evidence"]["pixel_visibility_truth"]))
    target = pixels["per_instance"]["source1"]["frames"][0]
    distractor = pixels["per_instance"]["source2"]["frames"][0]
    target_gate = rejection["target_gate"]
    distractor_gate = rejection["distractor_gate"]
    assert target["frame_index"] == rejection["frame_index"]
    assert target["target_pixels"] == target_gate["observed_target_pixels"]
    assert target["visible_pixels"] == target_gate["observed_visible_pixels"]
    assert target["visible_fraction"] == pytest.approx(
        target_gate["observed_visible_fraction"]
    )
    assert target_gate["observed_visible_fraction"] < 0.8
    assert distractor["visible_fraction"] == pytest.approx(
        distractor_gate["observed_visible_fraction"]
    )
    assert distractor_gate["observed_visible_fraction"] >= 0.5
    assert Path(rejection["evidence"]["normal_rgb"]).is_file()


def test_strict_eight_preflight_binds_native_floor_points(tmp_path: Path) -> None:
    plan_path = REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json"
    result_path = TOOL.build(plan_path, tmp_path / "preflight")
    result = _load(result_path)

    assert result["status"] == (
        "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates"
    )
    assert result["plan_id"] == _load(plan_path)["plan_id"]
    assert result["plan_record"]["path"] == str(plan_path.resolve())
    assert len(result["plan_record"]["sha256"]) == 64
    assert result["row_count"] == 8
    assert result["left_target_count"] == 4
    assert result["right_target_count"] == 4
    assert result["camera_translation_cluster_count"] == 8
    assert result["minimum_camera_translation_separation_m"] >= 0.75
    assert result["native_occupied_floor_point_count"] == 21
    assert all(
        record["status"] == "pass_native_occupied_floor_point"
        for record in result["occupied_floor_point_evidence"]
    )
    assert result["rows"][0]["status"] == "pass_existing_sparse_canary"
    assert all(
        row["status"] == "pass_cpu_geometry_pending_exact_rir_and_native_sparse"
        for row in result["rows"][1:]
    )
    assert result["formal_scene_count"] == 0
    assert result["qualification_claim"] is False
    assert result["gpu_or_rir_executed"] is False

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        TOOL.build(plan_path, tmp_path / "preflight")


def test_strict_eight_rejects_identity_side_voice_or_scope_drift() -> None:
    plan, registry = _inputs()

    invalid = deepcopy(plan)
    invalid["rows"][4]["actors"][1]["identity_key"] = "M"
    errors = TOOL.validate_plan(invalid, registry)
    assert "row 5 identities must differ" in errors
    assert "row 5 pair mismatch" in errors

    invalid = deepcopy(plan)
    invalid["rows"][2]["target_expected_screen_side"] = "left"
    errors = TOOL.validate_plan(invalid, registry)
    assert "target side sequence mismatch" in errors

    invalid = deepcopy(plan)
    invalid["rows"][6]["actors"][1]["voice_policy"] = "speaking"
    errors = TOOL.validate_plan(invalid, registry)
    assert "row 7 distractor must be silent" in errors

    invalid = deepcopy(plan)
    invalid["approved_identity_catalog"]["F"][
        "expected_speech_frame_window_inclusive"
    ] = [7, 31]
    errors = TOOL.validate_plan(invalid, registry)
    assert "identity F speech window mismatch" in errors

    invalid = deepcopy(plan)
    invalid["formal_scene_count"] = 8
    invalid["qualification_claim"] = True
    invalid["paper_catalog_mutation_allowed"] = True
    errors = TOOL.validate_plan(invalid, registry)
    assert "formal scene count must remain zero" in errors
    assert "qualification claim must remain false" in errors
    assert "paper catalog mutation forbidden" in errors

    invalid = deepcopy(plan)
    invalid["projection_and_native_thresholds"][
        "target_visible_fraction_minimum"
    ] = 0.5
    errors = TOOL.validate_plan(invalid, registry)
    assert "target visible-fraction minimum must remain 0.8" in errors

    invalid = deepcopy(plan)
    invalid["projection_and_native_thresholds"][
        "distractor_visible_fraction_minimum"
    ] = 0.8
    errors = TOOL.validate_plan(invalid, registry)
    assert "distractor visible-fraction minimum must remain 0.5" in errors


def test_strict_eight_rejects_camera_geometry_and_runtime_drift() -> None:
    plan, registry = _inputs()

    invalid = deepcopy(plan)
    invalid["rows"][1]["camera_pose"] = deepcopy(plan["rows"][0]["camera_pose"])
    errors = TOOL.validate_plan(invalid, registry)
    assert "eight distinct camera poses required" in errors
    assert "camera translation clusters are not separated enough" in errors

    invalid = deepcopy(plan)
    invalid["rows"][3]["camera_pose"]["rotation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    errors = TOOL.validate_plan(invalid, registry)
    assert any("camera quaternion/yaw mismatch" in error for error in errors)

    invalid_registry = deepcopy(registry)
    invalid_registry["aliases"].pop("strict_two_human_construction_male")
    errors = TOOL.validate_plan(plan, invalid_registry)
    assert "identity C alias mismatch" in errors

    invalid = deepcopy(plan)
    invalid["approved_identity_catalog"]["C"]["original_identity_id"] = (
        "rocketbox_professions_medical_female_01"
    )
    errors = TOOL.validate_plan(invalid, registry)
    assert "identity C is not approved" in errors
    assert "identity C is excluded" in errors


def test_strict_eight_rejects_floor_point_provenance_drift(tmp_path: Path) -> None:
    plan, _ = _inputs()
    invalid = deepcopy(plan)
    invalid["rows"][1]["actors"][0]["floor_point_provenance"][
        "frame_index"
    ] = 10
    plan_path = tmp_path / "invalid_plan.json"
    plan_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(RuntimeError, match="actor floor provenance mismatch"):
        TOOL.build(plan_path, tmp_path / "preflight")

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
# Guarding on tmp/ existing was wrong: running the engine in a
# checkout creates tmp/spear_instance_*, which made this look
# mounted and sent 49 tests into a run without their data.  The
# evidence mount signature is a lead_* workspace.
if not any(_RETAINED_TMP_WORKSPACE.glob("lead_*")):
    pytest.skip(
        "no lead_* evidence workspace under the repository tmp "
        "directory, so this checkout does not carry the retained "
        "strict-two-human evidence",
        allow_module_level=True,
    )
