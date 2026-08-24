from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import write_json
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth
from avengine.qa.question_spec import evaluate_offscreen_entry_truth


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_mp3d_offscreen_minipilot.py"
SPEC = importlib.util.spec_from_file_location(
    "build_mp3d_offscreen_minipilot", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _truth() -> dict:
    height, width = 4, 8
    empty = np.zeros((height, width), dtype=np.int32)
    left = empty.copy()
    left[2:, 0:2] = 210
    right = empty.copy()
    right[2:, 6:8] = 211
    actor0 = [empty.copy() for _ in range(5)] + [left.copy() for _ in range(70)]
    actor1 = [empty.copy() for _ in range(10)] + [right.copy() for _ in range(65)]
    normal = [
        np.where(actor0[index] != 0, actor0[index], actor1[index])
        for index in range(75)
    ]
    common = {
        "renderer_backend": "habitat_sim_native_mp3d",
        "rgb_renderer_backend": "habitat_sim_native_mp3d",
        "camera_contract_id": "request:rig:view",
        "semantic_id_namespace": "m5_declared_actor_semantic_ids",
        "resolution_hw": [height, width],
        "frame_indices": list(range(75)),
        "camera_pose_ids": [f"pose-{index}" for index in range(75)],
    }
    return compile_pixel_visibility_truth(
        normal_semantic_masks=normal,
        target_only_semantic_masks_by_instance={
            "actor0": actor0,
            "actor1": actor1,
        },
        semantic_ids_by_instance={"actor0": 210, "actor1": 211},
        normal_context={"pass_kind": "modal_scene", **common},
        target_only_contexts_by_instance={
            actor_id: {
                "pass_kind": "target_only",
                "target_instance_id": actor_id,
                **common,
            }
            for actor_id in ("actor0", "actor1")
        },
    )


def _capture(tmp_path: Path) -> Path:
    capture = tmp_path / "capture"
    capture.mkdir()
    truth = _truth()
    write_json(capture / "pixel_visibility_truth.json", truth)
    write_json(
        capture / "research_receipt.json",
        {
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "capture": {"frame_count": 75, "actor_pair_collision_free": True},
            "case": {
                "case_id": "case-both-moving",
                "region_instance_id": "region:005",
                "motion_case": "both_moving",
            },
            "actors": {
                "actor_ids": ["actor0", "actor1"],
                "source_ids": ["source1", "source2"],
                "semantic_ids": [210, 211],
            },
            "visibility": {
                "candidate_filtering": "disabled",
                "authority": truth["authority"],
                "formal_modal_visibility": truth,
            },
            "artifacts": {"pixel_visibility_truth": "pixel_visibility_truth.json"},
        },
    )
    return capture


def test_bound_truth_helper_keeps_left_and_right_entries() -> None:
    truth = _truth()
    left = evaluate_offscreen_entry_truth(truth, target_instance_id="actor0")
    right = evaluate_offscreen_entry_truth(truth, target_instance_id="actor1")
    assert left["status"] == right["status"] == "pass"
    assert left["answer"]["value"] == "left"
    assert right["answer"]["value"] == "right"
    assert left["evidence"]["entry_frame_indices"] == [5]
    assert right["evidence"]["entry_frame_indices"] == [10]


def test_truth_helper_rejects_non_authoritative_pixel_truth() -> None:
    truth = _truth()
    truth["authority"] = "geometric_projection_only"
    result = evaluate_offscreen_entry_truth(truth, target_instance_id="actor0")
    assert result["status"] == "rejected"
    assert result["reason"]["code"] == "facts_not_observable"
    assert "unexpected authority" in result["reason"]["detail"]


def test_builds_balanced_no_clobber_native_minipilot(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    output = tmp_path / "output"
    result = TOOL.build_offscreen_minipilot(
        visual_capture_dir=capture,
        output_directory=output,
    )
    assert result == output.resolve()
    questions = json.loads((output / "questions.json").read_text())
    gate = json.loads((output / "gate_report.json").read_text())
    assert questions["status"] == "pass_research_offscreen_minipilot"
    assert [item["answer"] for item in questions["questions"]] == ["left", "right"]
    report = questions["answer_balance"]["by_type"]["QV2-TO-OFFSCREEN"]
    assert report["answer_key_counts"] == {"A": 1, "B": 1}
    assert report["status"] == "pass"
    assert gate["checks"]["same_renderer_target_only_authority"] is True
    assert gate["pending"]["formal_dataset_admission"] is True
    with pytest.raises(TOOL.MP3DOffscreenMiniPilotError, match="replace"):
        TOOL.build_offscreen_minipilot(
            visual_capture_dir=capture,
            output_directory=output,
        )
