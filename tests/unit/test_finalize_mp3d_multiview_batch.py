from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import write_json
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/finalize_mp3d_multiview_batch.py"
SPEC = importlib.util.spec_from_file_location(
    "finalize_mp3d_multiview_batch", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _truth(*, entry_actor: str, side: str) -> dict:
    height, width = 4, 8
    empty = np.zeros((height, width), dtype=np.int32)
    actor0 = [empty.copy() for _ in range(75)]
    actor1 = [empty.copy() for _ in range(75)]
    target = actor0 if entry_actor == "actor0" else actor1
    semantic_id = 210 if entry_actor == "actor0" else 211
    mask = empty.copy()
    if side == "left":
        mask[2:, :2] = semantic_id
    else:
        mask[2:, 6:] = semantic_id
    for index in range(10, 75):
        target[index] = mask.copy()
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


def _base_rows(point_id: str, motion: str, camera_slot: str) -> list[dict]:
    moving_answer = {
        "static_static": ("staying still", "staying still"),
        "s1_moving_s2_static": ("moving", "staying still"),
        "s1_static_s2_moving": ("staying still", "moving"),
        "both_moving": ("moving", "moving"),
    }[motion]
    shared = {
        "point_id": point_id,
        "pair_kind": "two_dog_same_beagle_asset",
        "motion_case": motion,
        "viewpoint_id": camera_slot,
        "native_visual_capture": f"/old/{point_id}",
        "randomized_audio_render_status": "pending",
    }
    return [
        {
            **shared,
            "type_id": "QV2-TA-MOTION",
            "answer_arity": "binary",
            "options": ["moving", "staying still"],
            "answer": moving_answer[0],
            "semantic_answer_source_slot": "source1",
        },
        {
            **shared,
            "type_id": "QV2-T7-DURING",
            "answer_arity": "binary",
            "options": ["moving", "staying still"],
            "answer": moving_answer[1],
            "semantic_answer_source_slot": "source2",
        },
        {
            **shared,
            "type_id": "QV2-T5-DOA",
            "answer_arity": "numeric_banded",
            "options": None,
            "answer": 1.0,
            "answer_numeric": 1.0,
            "event_frame": 20,
            "semantic_answer_source_slot": "source2",
        },
    ]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    design = tmp_path / "design"
    target = tmp_path / "target"
    audio = tmp_path / "audio"
    design.mkdir()
    target.mkdir()
    audio.mkdir()
    views = []
    plan = []
    motions = (
        "static_static",
        "s1_moving_s2_static",
        "s1_static_s2_moving",
        "both_moving",
    )
    for point_index, motion in enumerate(motions):
        camera_slot = "camera_slot_0"
        placement = "placement_0"
        point_id = f"mp3d_r000_{camera_slot}_p{point_index:03d}_{motion}"
        view_root = design / "view0"
        view_root.mkdir(exist_ok=True)
        point_root = view_root / point_id
        point_root.mkdir()
        program_id = f"program-{point_id}"
        write_json(
            point_root / "audio_program.json",
            {
                "program_id": program_id,
                "events": [
                    {
                        "event_id": "event0",
                        "sound_asset_id": "dog_beagle_v2_scheduled_dry",
                    }
                ],
            },
        )
        case_manifest = tmp_path / f"case-{point_index}.json"
        write_json(
            case_manifest,
            {
                "actors": [
                    {
                        "source_id": "source1",
                        "asset_id": "beagle-asset",
                    },
                    {
                        "source_id": "source2",
                        "asset_id": "beagle-asset",
                    },
                ]
            },
        )
        m1 = tmp_path / f"m1-{point_index}.json"
        write_json(m1, {"request_id": f"m1-{point_index}"})
        item = {
            "region_index": 0,
            "region_instance_id": "region:000",
            "camera_slot_id": camera_slot,
            "placement_id": placement,
            "point_index": point_index,
            "motion_case": motion,
            "route_family_id": "route-family-0",
            "m1_request_path": str(m1),
            "case_manifest_path": str(case_manifest),
        }
        plan.append(item)
        capture = target / "region_000" / f"point_{point_index:03d}_{motion}"
        capture.mkdir(parents=True)
        entry_actor = "actor0" if point_index < 2 else "actor1"
        side = "left" if point_index < 2 else "right"
        truth = _truth(entry_actor=entry_actor, side=side)
        write_json(capture / "pixel_visibility_truth.json", truth)
        write_json(
            capture / "research_receipt.json",
            {
                "status": "research_only",
                "research_only": True,
                "episode_counted": False,
                "capture": {
                    "frame_count": 75,
                    "actor_pair_collision_free": True,
                },
                "case": {
                    "region_instance_id": "region:000",
                    "motion_case": motion,
                    "route_family_id": "route-family-0",
                    "camera_m1_request": str(m1),
                },
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
        audio_root = audio / point_id
        (audio_root / "audio").mkdir(parents=True)
        (audio_root / "audio/mixture.wav").write_bytes(b"wav")
        write_json(
            audio_root / "research_receipt.json",
            {
                "status": "pass",
                "research_only": True,
                "episode_counted": False,
                "formal_dataset_count": 0,
                "audio": {"sample_rate_hz": 16000, "sample_count": 80000},
                "audio_program": {"program_id": program_id},
                "rir": {"prior_listener_rir_reused": False},
                "listener": {
                    "view_identity": {"visual_capture_dir": str(capture.resolve())}
                },
                "outputs": {"audio/mixture.wav": "existing-receipt-value"},
            },
        )
    view_root = design / "view0"
    rows = [
        row
        for point_index, motion in enumerate(motions)
        for row in _base_rows(
            f"mp3d_r000_camera_slot_0_p{point_index:03d}_{motion}",
            motion,
            "camera_slot_0",
        )
    ]
    write_json(
        view_root / "questions.json",
        {
            "status": "pending_randomized_audio_render",
            "questions": rows,
        },
    )
    point_ids = [
        f"mp3d_r000_camera_slot_0_p{index:03d}_{motion}"
        for index, motion in enumerate(motions)
    ]
    write_json(view_root / "manifest.json", {"points": point_ids})
    views.append(
        {
            "region_index": 0,
            "camera_slot_id": "camera_slot_0",
            "output": str(view_root),
        }
    )
    write_json(
        design / "manifest.json",
        {"status": "pending_randomized_audio_render", "views": views},
    )
    write_json(target / "plan.json", plan)
    return design, target, audio


def test_finalizes_specs_balanced_offscreen_and_refusal(tmp_path: Path) -> None:
    design, target, audio = _fixture(tmp_path)
    output = tmp_path / "output"
    result = TOOL.finalize_multiview_batch(
        design_root=design,
        target_only_root=target,
        audio_root=audio,
        output_directory=output,
    )
    assert result == output.resolve()
    document = json.loads((output / "questions.json").read_text())
    assert document["status"] == "pass_research_multiview_batch"
    assert document["counts"]["points"] == 4
    assert document["counts"]["base_questions"] == 12
    assert document["counts"]["selected_offscreen"] == 4
    assert document["counts"]["selected_refusals"] == 2
    assert document["counts"]["questions"] == 18
    assert document["refusal_quota"]["final_fraction"] == pytest.approx(2 / 18)
    assert document["answer_balance"]["status"] == "pass"
    offscreen = [
        row for row in document["questions"] if row["type_id"] == "QV2-TO-OFFSCREEN"
    ]
    assert Counter(row["answer"] for row in offscreen) == {"left": 2, "right": 2}
    refusals = [
        row for row in document["questions"] if row["type_id"] == "QV2-TR-REFUSAL"
    ]
    assert all(
        row["modality_mask"] == {"audio": True, "video": False} for row in refusals
    )
    specs = list((output / "point_specs").glob("*/spec.json"))
    assert len(specs) == 4
    assert all(
        json.loads(path.read_text())["viewpoint_id"] == "placement_0" for path in specs
    )
    with pytest.raises(TOOL.MP3DMultiviewFinalizeError, match="replace"):
        TOOL.finalize_multiview_batch(
            design_root=design,
            target_only_root=target,
            audio_root=audio,
            output_directory=output,
        )


def test_balanced_offscreen_requires_both_sides() -> None:
    rows = [
        {"question_id": f"q{index}", "answer": "left", "point_id": f"p{index}"}
        for index in range(3)
    ]
    with pytest.raises(TOOL.MP3DMultiviewFinalizeError, match="both entry sides"):
        TOOL.select_balanced_offscreen_candidates(rows)
