from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[2]
QA = REPOSITORY / "examples/qa"
ASSETS = REPOSITORY / "docs/qa/assets"


def _load(name: str) -> dict[str, object]:
    return json.loads((QA / name).read_text(encoding="utf-8"))


def test_both_move_geometry_handoff_keeps_counterfactual_provenance_closed() -> None:
    receipt = _load("native_strict_two_human_both_move_v1_cpu_geometry_receipt_v1.json")
    preflight = _load("native_strict_two_human_both_move_v1_geometry_preflight_v1.json")
    exact = _load("native_strict_two_human_both_move_v1_exact_paths_v1.json")
    report = _load(
        "native_strict_two_human_both_move_v1_all75_hard_gate_report_v1.json"
    )
    row = preflight["canaries"][0]

    assert receipt["candidate_decision"] == "GO"
    assert receipt["gpu_launch_authorized"] is False
    assert receipt["rir_authorized"] is False
    assert row["mechanism"] == exact["mechanism"] == "both_move"
    assert row["native_same_scene_pair"] is False
    assert row["counterfactual_pairing_contract"]["is_native_same_scene_pair"] is False
    assert row["native_source_scenario_ids"] == [
        "human_border_collie__recombined_both_moving_0304",
        "border_collie_human__recombined_both_moving_0990",
    ]
    assert (
        report["global100_caveat"][
            "depth_safe_windows_from_frozen_global100_both_move_sources"
        ]
        == 0
    )


def test_both_move_exact_paths_pass_motion_projection_and_depth_gates() -> None:
    preflight = _load("native_strict_two_human_both_move_v1_geometry_preflight_v1.json")
    exact = _load("native_strict_two_human_both_move_v1_exact_paths_v1.json")
    row = preflight["canaries"][0]
    assert row["acoustic_state_expectation"] == {
        "source_frame_uses": 150,
        "target_unique_rir_states": 75,
        "distractor_unique_rir_states": 75,
        "total_unique_rir_states": 150,
        "exact_rir_required_before_gpu": True,
    }
    assert row["projection_and_separation_preflight"]["status"] == "pass"
    assert (
        row["projection_and_separation_preflight"][
            "minimum_synchronous_actor_separation_m"
        ]
        >= 1.3
    )
    for role, slot in (("target", "source1"), ("distractor", "source2")):
        binding = row[role]
        exact_role = exact["roles"][role]
        provenance = binding["path_provenance"]
        assert binding["source_slot_id"] == slot
        assert binding["root_path_m"] == exact_role["root_path_m"]
        assert len(binding["root_path_m"]) == 75
        assert len({tuple(point) for point in binding["root_path_m"]}) == 75
        assert provenance["endpoints_exact_native_readbacks"] is True
        assert provenance["interior_output_roots_exact_native_frame_readbacks"] is False
        assert provenance["counterfactual_cross_scenario_pair"] is True
        depth = row["depth_corridor_preflight"][role]
        assert depth["status"] == "pass"
        assert depth["minimum_depth_clearance_m"] >= 0.25
        assert depth["out_of_view_sample_count"] == 0


def test_both_move_projection_review_assets_are_retained() -> None:
    with Image.open(
        ASSETS / "both_move_v1_projection_depth_topdown_overlay.png"
    ) as image:
        assert image.width > 0 and image.height > 0
    with Image.open(
        ASSETS / "both_move_v1_f00_f37_f74_projection_contact_strip.png"
    ) as image:
        assert image.size == (1920, 360)
