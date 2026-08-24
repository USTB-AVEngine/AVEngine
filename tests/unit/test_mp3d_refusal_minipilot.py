from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import write_json
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_mp3d_refusal_minipilot.py"
SPEC = importlib.util.spec_from_file_location("build_mp3d_refusal_minipilot", TOOL_PATH)
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


def _base_question_documents(tmp_path: Path) -> list[Path]:
    first = tmp_path / "base_a.json"
    second = tmp_path / "base_b.json"
    questions = []
    for index in range(4):
        questions.append(
            {
                "point_id": f"point-{index}",
                "type_id": "QV2-TA-MOTION",
                "answer_arity": "binary",
                "options": ["moving", "staying still"],
                "answer": "moving" if index < 2 else "staying still",
                "semantic_answer_source_slot": "source1",
            }
        )
        questions.append(
            {
                "point_id": f"point-{index}",
                "type_id": "QV2-T7-DURING",
                "answer_arity": "binary",
                "options": ["moving", "staying still"],
                "answer": "moving" if index % 2 == 0 else "staying still",
                "semantic_answer_source_slot": "source2",
            }
        )
        questions.append(
            {
                "point_id": f"point-{index}",
                "type_id": "QV2-T5-DOA",
                "answer_arity": "numeric_banded",
                "options": None,
                "answer": float(index),
                "semantic_answer_source_slot": "source2",
            }
        )
    offscreen = [
        {
            "question_id": f"offscreen-{side}",
            "type_id": "QV2-TO-OFFSCREEN",
            "answer_arity": "binary",
            "options": ["left", "right"],
            "answer": side,
            "semantic_answer_source_slot": source,
        }
        for side, source in (("left", "source1"), ("right", "source2"))
    ]
    write_json(
        first,
        {
            "research_only": True,
            "episode_counted": False,
            "questions": questions,
        },
    )
    write_json(
        second,
        {
            "research_only": True,
            "episode_counted": False,
            "questions": offscreen,
        },
    )
    return [first, second]


def _visual_and_audio(tmp_path: Path) -> tuple[Path, Path]:
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
            "case": {"case_id": "case-both-moving"},
            "actors": {
                "actor_ids": ["actor0", "actor1"],
                "source_ids": ["source1", "source2"],
                "semantic_ids": [210, 211],
            },
            "visibility": {
                "candidate_filtering": "disabled",
                "formal_modal_visibility": truth,
            },
            "artifacts": {"pixel_visibility_truth": "pixel_visibility_truth.json"},
        },
    )
    audio_root = tmp_path / "audio"
    (audio_root / "audio/binaural").mkdir(parents=True)
    (audio_root / "audio/binaural/mixture.wav").write_bytes(b"wav")
    audio_receipt = audio_root / "research_receipt.json"
    write_json(
        audio_receipt,
        {
            "status": "pass",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "audio": {"sample_rate_hz": 16000, "sample_count": 80000},
            "audio_program": {"program_id": "program-v1"},
            "rir": {"prior_listener_rir_reused": False},
            "listener": {
                "view_identity": {"visual_capture_dir": str(capture.resolve())}
            },
            "outputs": {"audio/binaural/mixture.wav": "not-checked-by-this-loader"},
        },
    )
    return capture, audio_receipt


def test_real_audio_only_refusal_quota_appends_without_changing_base(
    tmp_path: Path,
) -> None:
    base_paths = _base_question_documents(tmp_path)
    capture, audio = _visual_and_audio(tmp_path)
    original = [
        row for path in base_paths for row in json.loads(path.read_text())["questions"]
    ]
    output = tmp_path / "output"
    result = TOOL.build_refusal_minipilot(
        base_question_documents=base_paths,
        visual_capture_dir=capture,
        audio_receipt_path=audio,
        output_directory=output,
    )
    assert result == output.resolve()
    document = json.loads((output / "questions.json").read_text())
    assert document["status"] == "pass_research_refusal_minipilot"
    assert document["questions"][:14] == original
    refusals = document["questions"][14:]
    assert len(refusals) == 2
    assert document["refusal_quota"]["final_fraction"] == 0.125
    assert all(row["answer"] == TOOL.REFUSAL_ANSWER for row in refusals)
    assert all(
        row["modality_mask"] == {"audio": True, "video": False} for row in refusals
    )
    assert [row["withheld_visual_gold_state"] for row in refusals] == [
        "visible_clear",
        "visible_clear",
    ]
    assert document["answer_balance"]["status"] == "pass"
    gate = json.loads((output / "gate_report.json").read_text())
    assert all(gate["checks"].values())
    with pytest.raises(TOOL.MP3DRefusalMiniPilotError, match="replace"):
        TOOL.build_refusal_minipilot(
            base_question_documents=base_paths,
            visual_capture_dir=capture,
            audio_receipt_path=audio,
            output_directory=output,
        )


def test_audio_receipt_must_bind_the_same_visual_capture(tmp_path: Path) -> None:
    base_paths = _base_question_documents(tmp_path)
    capture, audio = _visual_and_audio(tmp_path)
    receipt = json.loads(audio.read_text())
    receipt["listener"]["view_identity"]["visual_capture_dir"] = "/different"
    write_json(audio, receipt)
    with pytest.raises(TOOL.MP3DRefusalMiniPilotError, match="fresh RLR"):
        TOOL.build_refusal_minipilot(
            base_question_documents=base_paths,
            visual_capture_dir=capture,
            audio_receipt_path=audio,
            output_directory=tmp_path / "output",
        )
