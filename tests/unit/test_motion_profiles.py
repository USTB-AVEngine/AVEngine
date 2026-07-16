from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from avengine.motion.profiles import (
    MotionProfileError,
    load_motion_retarget_profile,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE = (
    REPOSITORY / "examples/m2/motion_profiles/quadruped_dog_to_rocketbox_beagle_v1.json"
)
SCHEMA = REPOSITORY / "schemas/motion_retarget_profile_v1.schema.json"


def test_beagle_profile_matches_json_schema_and_semantic_contract() -> None:
    raw = json.loads(PROFILE.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(raw)) == []

    profile = load_motion_retarget_profile(PROFILE)
    assert profile.adapter_id == "quadruped_mammal_locomotion_v1"
    assert profile.solver_id == "world_left_delta_v2"
    assert profile.capability.production_supported is True
    assert len(profile.joint_mappings) == 20
    assert tuple(action.output_action_name for action in profile.actions) == (
        "Idle",
        "Walking",
    )
    assert profile.attribute_domain.size == ("small", "medium", "large")
    assert profile.attribute_domain.body_build == (
        "slim",
        "standard",
        "stocky",
    )
    assert profile.attribute_domain.coat_values == (
        "light_tricolor",
        "standard_tricolor",
        "dark_tricolor",
    )
    assert profile.attribute_domain.life_stage == ("young", "adult", "senior")
    assert profile.qa_semantic_action_id == "walk"
    assert profile.qa_contract.sample_rate_hz == 15.0
    assert profile.qa_contract.required_chain_ids == (
        "hind_left",
        "hind_right",
        "fore_left",
        "fore_right",
    )
    assert profile.qa_coordinate_frame.forward_axis == "+X"
    assert profile.semantic_chains[2].target_end_effector_joint_id == ("beagle L Toe0")


def _mutated_profile(tmp_path: Path, mutate) -> Path:
    raw = json.loads(PROFILE.read_text(encoding="utf-8"))
    mutate(raw)
    output = tmp_path / "profile.json"
    output.write_text(json.dumps(raw), encoding="utf-8")
    return output


def test_unsupported_body_plan_adapter_fails_closed(tmp_path: Path) -> None:
    source = _mutated_profile(
        tmp_path,
        lambda value: value.update(
            adapter_id="avian_flight_v1",
            body_plan_id="avian_flight_raptor_v1",
        ),
    )
    with pytest.raises(MotionProfileError, match="unavailable"):
        load_motion_retarget_profile(source)


def test_profile_cannot_silently_reuse_cross_breed_coat_names(tmp_path: Path) -> None:
    source = _mutated_profile(
        tmp_path,
        lambda value: value["attribute_domain"].update(
            coat_values=["golden", "golden", "golden"]
        ),
    )
    with pytest.raises(MotionProfileError, match="unique"):
        load_motion_retarget_profile(source)


def test_profile_requires_complete_one_to_one_semantic_mapping(tmp_path: Path) -> None:
    source = _mutated_profile(
        tmp_path,
        lambda value: value["joint_mappings"].pop(),
    )
    with pytest.raises(MotionProfileError, match="every semantic joint"):
        load_motion_retarget_profile(source)


def test_profile_rejects_legacy_right_delta_solver(tmp_path: Path) -> None:
    source = _mutated_profile(
        tmp_path,
        lambda value: value["solver"].update(
            solver_id="legacy_rest_local_right_delta_v1"
        ),
    )
    with pytest.raises(MotionProfileError, match="world_left_delta_v2"):
        load_motion_retarget_profile(source)


def test_profile_rejects_non_orthogonal_qa_axes(tmp_path: Path) -> None:
    source = _mutated_profile(
        tmp_path,
        lambda value: value["qa_contract"]["coordinate_frame"].update(
            lateral_axis="-X"
        ),
    )
    with pytest.raises(MotionProfileError, match="orthogonal"):
        load_motion_retarget_profile(source)


def test_profile_requires_qa_for_every_locomotion_chain(tmp_path: Path) -> None:
    source = _mutated_profile(
        tmp_path,
        lambda value: value["qa_contract"]["required_chain_ids"].pop(),
    )
    with pytest.raises(MotionProfileError, match="exactly follow locomotion chains"):
        load_motion_retarget_profile(source)
