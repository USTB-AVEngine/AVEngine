import json
from pathlib import Path

PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "qa"
    / "native_mixed_human_sparse_gate_v1.json"
)

EXPECTED_CASES = (
    ("human_border_collie__recombined_both_moving_0999", 44, "left"),
    ("border_collie_human__recombined_static_static_0919", 64, "left"),
    ("border_collie_human__recombined_both_moving_0125", 32, "left"),
    (
        "border_collie_human__recombined_source1_moving_source2_static_0342",
        8,
        "left",
    ),
    ("border_collie_human__recombined_both_moving_0425", 71, "right"),
    ("human_border_collie__recombined_both_moving_0523", 28, "right"),
    ("border_collie_human__recombined_static_static_0065", 8, "right"),
    (
        "border_collie_human__recombined_source1_static_source2_moving_0136",
        71,
        "right",
    ),
)


def test_native_mixed_human_sparse_gate_plan_is_frozen_and_fail_closed() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["schema"] == "avengine_native_mixed_human_sparse_gate_plan_v1"
    assert plan["source_suite_plan"] == (
        "/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/"
        "apartment_asset_bound_ue_unique1000_full_20260723_01/"
        "suite_execution_plan.json"
    )
    assert plan["gpu_policy"] == {
        "preferred_physical_gpu_index": 1,
        "required_idle_compute_process_count": 0,
        "forbidden_physical_gpu_indices": [0, 3],
        "graphics_adapter_argument": 1,
    }
    assert plan["gate_contract"]["formal_scene_count"] == 0
    assert plan["gate_contract"]["capture_frame_count_per_case"] == 1

    cases = plan["cases"]
    assert len(cases) == 8
    assert len({case["case_id"] for case in cases}) == 8
    assert len({case["episode_id"] for case in cases}) == 8
    assert tuple(
        (case["episode_id"], case["frame_index"], case["expected_side"])
        for case in cases
    ) == EXPECTED_CASES
    assert [case["expected_side"] for case in cases].count("left") == 4
    assert [case["expected_side"] for case in cases].count("right") == 4
    assert {
        case["episode_id"]
        for case in cases
        if case["legacy_scene_overlap"]
    } == {
        "border_collie_human__recombined_both_moving_0125",
        "human_border_collie__recombined_both_moving_0523",
    }
