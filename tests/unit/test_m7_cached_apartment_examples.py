from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/m6x/build_cached_apartment_dataset_examples.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "build_cached_apartment_dataset_examples", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


def test_grounded_border_collie_profile_uses_a_new_asset_not_beagle_shape() -> None:
    bindings = _TOOL.SOURCE_BINDING_PROFILES["human_border_collie"]
    assert bindings["source1"]["asset_id"] == (
        "rocketbox_human_male_adult_01_m5_1_candidate"
    )
    assert bindings["source2"]["asset_id"] == (
        "generated_border_collie_black_white_medium_standard_adult_research_v1"
    )
    assert bindings["source2"]["asset_id"] != (
        _TOOL.HUMAN_BEAGLE_BINDINGS["source2"]["asset_id"]
    )
    assert _TOOL.SELECTION_SOURCE_BINDING_PROFILE[
        "human_border_collie_grounded"
    ] == "human_border_collie"
    selected = _TOOL.SELECTION_PROFILES["human_border_collie_grounded"]
    assert {value[0] for value in selected} == {"P0", "P1", "P2", "P3"}
    assert all("human_border_collie__" in value[2] for value in selected)


def test_timeline_rebinds_only_the_dog_actor_and_keeps_idle_heading_stable() -> None:
    bindings = _TOOL.SOURCE_BINDING_PROFILES["human_border_collie"]
    template = {
        "actors": [
            {
                "actor_id": "dog0",
                "asset_id": "old_beagle",
                "template_id": "old_beagle",
                "body_plan_id": "quadruped_canine",
            },
            {
                "actor_id": "human0",
                "asset_id": bindings["source1"]["asset_id"],
                "template_id": "human",
                "body_plan_id": "biped_human",
            },
        ],
        "source_manifest_events": [],
    }
    root_paths = {
        "source1": np.repeat([[0.0, 0.271, 0.0]], 75, axis=0),
        "source2": np.repeat([[1.0, 0.271, 1.0]], 75, axis=0),
    }
    timeline, headings = _TOOL._timeline(
        template=template,
        source_bindings=bindings,
        root_paths=root_paths,
        motion_by_slot={"source1": "static", "source2": "static"},
        listener_position_m=np.asarray([-0.7, 1.471, 0.65]),
    )
    actors = {value["actor_id"]: value for value in timeline["actors"]}
    assert actors["dog0"]["asset_id"] == bindings["source2"]["asset_id"]
    assert actors["dog0"]["template_id"] == (
        "generated_border_collie_target_native_v1"
    )
    assert actors["human0"]["asset_id"] == bindings["source1"]["asset_id"]
    assert {state["action_id"] for frame in timeline["frames"] for state in frame["actor_states"]} == {"idle"}
    np.testing.assert_allclose(
        headings["source2"],
        np.repeat(headings["source2"][:1], 75, axis=0),
        atol=0.0,
        rtol=0.0,
    )
