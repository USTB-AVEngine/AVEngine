from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = (
    REPOSITORY / "tools/qa/build_strict_two_human_distractor_moves_v2_candidate.py"
)
PREFLIGHT_PATH = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_distractor_moves_v2_preflight_v1.json"
)
RECEIPT_PATH = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_distractor_moves_v2_cpu_geometry_receipt_v1.json"
)
OVERLAY_PATH = (
    REPOSITORY
    / "docs/qa/assets/distractor_moves_v2_projection_depth_topdown_overlay.png"
)
SPEC = importlib.util.spec_from_file_location("distractor_candidate", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_distractor_moves_v2_tracked_cpu_geometry_closure() -> None:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    row = preflight["canaries"][0]
    target = row["target"]["root_path_m"]
    distractor = row["distractor"]["root_path_m"]

    assert row["mechanism"] == "distractor_moves"
    assert row["target"]["identity_key"] == "F"
    assert row["distractor"]["identity_key"] == "M"
    assert len(target) == len(distractor) == 75
    assert len({tuple(point) for point in target}) == 1
    assert len({tuple(point) for point in distractor}) == 75
    assert row["camera"]["habitat_yaw_deg"] == 55.0
    assert row["camera"]["ue_yaw_deg"] == -145.0
    assert receipt["candidate_decision"] == "GO_TO_CPU_MATERIALIZATION_ONLY"
    assert receipt["gpu_launch_authorized"] is False
    assert receipt["camera_cluster_scope"]["independent_episode_claim"] is False
    assert receipt["strict_native_acceptance_gate"]["status"] == (
        "pending_fresh_native_target_only_capture"
    )
    assert OVERLAY_PATH.is_file()


def test_distractor_moves_v2_native_path_phase_and_depth_contract() -> None:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    row = preflight["canaries"][0]
    regenerated = TOOL.arc_length_resample(TOOL.NATIVE_HUMAN_ANCHORS, 75)

    assert regenerated == row["distractor"]["root_path_m"]
    assert TOOL.path_length(regenerated) == pytest.approx(
        0.850134492171555, abs=1.0e-12
    )
    assert row["distractor"]["path_provenance"][
        "native_source_frame_indices_inclusive"
    ] == [2, 17]
    assert row["distractor"]["path_provenance"]["native_anchor_count"] == 16
    assert (
        row["distractor"]["path_provenance"][
            "interior_output_roots_exact_native_frame_readbacks"
        ]
        is False
    )
    assert row["distractor"]["per_frame_action_phase"][0] == 0.125
    assert math.isclose(row["distractor"]["per_frame_action_phase"][-1], 0.0625)
    corridors = receipt["depth_corridor_preflight"]
    assert corridors["static_target"]["minimum_depth_clearance_m"] > 0.489
    assert corridors["moving_distractor"]["minimum_depth_clearance_m"] > 0.342
    assert (
        corridors["moving_distractor"][
            "minimum_environment_observation_count_per_sample"
        ]
        >= 67
    )


def test_wrong_yaw_0099_candidate_is_explicitly_rejected() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    correction = receipt["corrected_camera_contract"]

    assert correction["habitat_yaw_deg"] == 55.0
    assert correction["rejected_prior_scan_habitat_yaw_deg"] == 35.0
    assert correction["rejected_prior_candidate"].endswith("0099")
    assert "x=0.502" in correction["rejection_reason"]
