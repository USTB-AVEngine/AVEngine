from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPOSITORY / "tools/qa/build_strict_two_human_full_episode_batch.py"
AUDIT_PATH = REPOSITORY / "tools/qa/audit_strict_two_human_room_expansion.py"
REQUEST_PATH = REPOSITORY / "examples/qa/native_strict_two_human_full_episode_batch_v1.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module("strict2h_full75_builder", BUILDER_PATH)
AUDIT = _module("strict2h_room_audit", AUDIT_PATH)


def test_request_is_interim_single_room_and_gpu1_only() -> None:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    assert request["output_contract"]["episode_count"] == 100
    assert request["output_contract"]["first_phase_ready_pilot_target"] == 20
    assert request["output_contract"]["final_multi_room_minimum_ready_room_count"] == 3
    assert request["native_room_scope"]["ready_room_count"] == 1
    assert request["native_room_scope"]["interim_single_room_candidate_bank"] is True
    assert request["formal_episode_count"] == 0
    assert request["qualification_claim"] is False
    assert request["gpu_policy"] == {
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "forbidden_physical_gpu_indices": [0, 3],
        "cpu_builder_must_not_launch_gpu": True,
        "first_20_blocked_until_canaries_pass": True,
        "rows_21_to_100_blocked_until_three_real_rooms_are_ready": True,
    }


def test_projection_and_answer_order_balance() -> None:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    target = [[-0.8, 0.4, -3.0]] * 75
    distractor = [[0.8, 0.4, -3.0]] * 75
    metrics = BUILDER._geometry_metrics(
        request=request,
        camera=[0.0, 1.471, 0.0],
        yaw_deg=0.0,
        target_path=target,
        distractor_path=distractor,
        target_side="left",
        camera_pan=False,
    )
    assert metrics is not None
    assert metrics["minimum_projected_x_separation_fraction"] >= 0.15
    questions = [
        BUILDER._question(index, "left" if index % 2 == 0 else "right", f"e{index}")
        for index in range(100)
    ]
    assert Counter(item["correct_index"] for item in questions) == {0: 50, 1: 50}
    assert Counter(item["option_order_id"] for item in questions) == {
        "left_right": 50,
        "right_left": 50,
    }


def test_debug_maps_are_not_implicitly_promoted_to_rooms() -> None:
    assert (
        AUDIT._package_fragment(
            "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000.debug_0000"
        )
        == "spearsim/content/spear/scenes/debug_0000/maps/debug_0000"
    )
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    excluded = {
        item["room_id"] for item in request["native_room_scope"]["not_counted_as_ready_rooms"]
    }
    assert excluded == {"replicacad_apt_0", "habitat_mp3d_example_17DRP5sb8fy"}


@pytest.mark.skipif(
    not Path(
        "/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/"
        "apartment_asset_bound_ue_unique1000_full_20260723_01/"
        "suite_execution_plan.json"
    ).is_file(),
    reason="retained native Apartment 1000-Episode authority is not mounted",
)
def test_real_cpu_builder_emits_100_independent_rows(tmp_path: Path) -> None:
    paths = BUILDER.build(REQUEST_PATH, tmp_path / "plan")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    assert len(episodes) == 100
    assert len({item["camera_cluster_id"] for item in episodes}) == 100
    assert len({item["native_source_scenario_id"] for item in episodes}) == 100
    assert len({item["dedup_key_text"] for item in episodes}) == 100
    assert Counter(item["target"]["side"] for item in episodes) == {
        "left": 50,
        "right": 50,
    }
    assert Counter(item["mechanism"] for item in episodes) == {
        "both_static": 20,
        "target_moves": 20,
        "distractor_moves": 20,
        "both_move": 20,
        "camera_pan_both_static": 20,
    }
    assert all(item["formal"] is False for item in episodes)
    assert summary["single_room_mechanism_pilot_count"] == 20
    assert summary["final_multi_room_episode_count"] == 0
    assert summary["batch_launch_authorized"] is False
