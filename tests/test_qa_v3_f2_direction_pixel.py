from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import join_f2_direction_pixel as binder  # noqa: E402


POINT_ID = "point"
SCENE_ID = "room"
PROFILE_ID = "f2_direction"
NATIVE_MAP = "/Game/TestMap"


def _frame(frame_index: int, state: str) -> dict:
    if state in binder.VISIBLE_STATES:
        return {
            "frame_index": frame_index,
            "state": state,
            "target_pixels": 100,
            "visible_pixels": 100,
        }
    if state == "out_of_view":
        return {
            "frame_index": frame_index,
            "state": state,
            "target_pixels": 0,
            "visible_pixels": 0,
        }
    return {
        "frame_index": frame_index,
        "state": state,
        "target_pixels": 100,
        "visible_pixels": 0,
    }


def _pixel_truth(states: dict[str, list[str]]) -> dict:
    return {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "frame_indices": [2, 3, 4],
        "per_instance": {
            slot: {
                "frames": [
                    _frame(frame, state)
                    for frame, state in zip((2, 3, 4), values, strict=True)
                ]
            }
            for slot, values in states.items()
        },
    }


def _write_fixture(tmp_path: Path, *, states=None):
    states = states or {
        "source1": ["out_of_view", "out_of_view", "out_of_view"],
        "source2": ["visible_clear", "visible_occluded", "visible_clear"],
    }
    truth = _pixel_truth(states)

    actor_selection = tmp_path / "actor_selection.json"
    actor_selection.write_text(json.dumps({
        "actors": [
            {"source_slot_id": "source1", "asset_id": "asset_a"},
            {"source_slot_id": "source2", "asset_id": "asset_b"},
        ]
    }))
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({"frame_count": 5}))
    m1_request = tmp_path / "m1_capture_request.json"
    m1_request.write_text(json.dumps({"room_id": SCENE_ID}))
    endpoint_registry = tmp_path / "source_endpoints.json"
    endpoint_registry.write_text(json.dumps({"source_endpoints": []}))

    programs = tmp_path / "programs"
    programs.mkdir()
    main_program = programs / "main_program.json"
    gatea_program = programs / "gatea_program.json"
    main_program.write_text(json.dumps({"program_id": "program_main"}))
    gatea_program.write_text(json.dumps({"program_id": "program_gatea"}))

    geometry = {
        "main": {
            "status": "pass",
            "policy": "out_of_view",
            "frame_bounds": [2, 4],
            "source_slot_id": "source1",
        },
        "gateA": {
            "status": "pass",
            "policy": "visible",
            "frame_bounds": [2, 4],
            "source_slot_id": "source2",
        },
    }
    common = {
        "schema": "qa_v3_fact_record_v2",
        "scene_id": SCENE_ID,
        "profile_id": PROFILE_ID,
        "point_id": POINT_ID,
        "query_window_frame_bounds": [2, 4],
        "room": {"native_map": NATIVE_MAP},
        "audio": {"source_endpoint_registry": "source_endpoints.json"},
        "generation_checks": {
            "query_visibility_window_geometry": geometry,
        },
    }
    main = dict(
        common,
        variant="main",
        target_slot="source1",
        query_visibility="out_of_view",
        audio={"program_id": "program_main", "source_endpoint_registry": "source_endpoints.json"},
    )
    gatea = dict(
        common,
        variant="gateA",
        gatea_of=POINT_ID,
        target_slot="source2",
        query_visibility="visible",
        audio={"program_id": "program_gatea", "source_endpoint_registry": "source_endpoints.json"},
    )
    main_path = tmp_path / "main_fact.json"
    gatea_path = tmp_path / "gatea_fact.json"
    main_path.write_text(json.dumps(main))
    gatea_path.write_text(json.dumps(gatea))

    visual_root = tmp_path / "visual_capture"
    visual_point = visual_root / POINT_ID
    visual_point.mkdir(parents=True)
    (visual_point / "research_receipt.json").write_text(json.dumps({
        "status": "research_only",
        "inputs": {
            "actor_selection": str(actor_selection),
            "timeline": str(timeline),
        },
    }))
    selection_manifest = tmp_path / "visual_selection.json"
    selection_manifest.write_text(json.dumps({"selected": [{"point_id": POINT_ID}]}))
    visual = tmp_path / "visual.json"
    visual.write_text(json.dumps({
        "schema": "qa_v3_visual_batch_verification_v1",
        "status": "pass",
        "inputs": {
            "selection_manifest": str(selection_manifest),
            "visual_root": str(visual_root),
        },
        "points": [{"point_id": POINT_ID, "status": "pass"}],
    }))

    audio_root = tmp_path / "audio"
    audio_program_ids = {"main": "program_main", "gateA": "program_gatea"}
    for variant, name in (("main", POINT_ID), ("gateA", f"{POINT_ID}_gateA")):
        render = audio_root / name / "audio" / "binaural"
        render.mkdir(parents=True)
        (render / "mixture.wav").write_bytes(b"fixture")
        receipt = {
            "status": "pass",
            "execution_variant": variant,
            "audio_program": {
                "program_id": audio_program_ids[variant],
                "variant_id": "A",
                "path": str(
                    main_program if variant == "main" else gatea_program
                ),
            },
            "inputs": {
                "m1_request": {"path": str(m1_request)},
                "source_endpoint_registry": {"path": str(endpoint_registry)},
            },
        }
        (audio_root / name / "research_receipt.json").write_text(
            json.dumps(receipt)
        )
    audio = tmp_path / "audio_verification.json"
    audio.write_text(json.dumps({
        "schema": "qa_v3_audio_batch_verification_v1",
        "audio_root": str(audio_root),
        "checked_renders": 2,
        "audio_variant_waveform_nonidentity_pairs": 1,
        "gatea_semantic_flip_pairs": 1,
        "execution_variant_verification": {
            "field": "execution_variant",
            "status": "verified",
            "verified_renders": [POINT_ID, f"{POINT_ID}_gateA"],
            "unverified_renders": [],
            "failed_renders": [],
        },
        "failures": [],
    }))

    pixel = tmp_path / "pixel_evidence.json"
    pixel.write_text(json.dumps({
        "schema": "qa_v3_current_timeline_native_pixel_probe_v1",
        "status": "pass",
        "native_map": NATIVE_MAP,
        "inputs": {
            "actor_selection": str(actor_selection),
            "timeline": str(timeline),
        },
        "pixel_visibility": truth,
    }))
    return main_path, gatea_path, visual, audio, pixel


def test_direction_join_checks_pair_bindings_and_declared_windows(tmp_path):
    paths = _write_fixture(tmp_path)
    result = binder.join(*paths)
    assert result["status"] == "research_candidate"
    assert result["pixel_join_status"] == "pass"
    assert result["checks"]["fact_variants"]["main_gateA_target_slots_exchanged"]
    assert result["checks"]["query_visibility_window_geometry"]["both_pass"]
    assert result["checks"]["visual_binding"]["selected_point"]
    assert result["checks"]["audio_binding"]["execution_variant_verified"]
    assert result["checks"]["pixel_binding"]["timeline_path_equal"]
    assert result["checks"]["main_window"]["passed"]
    assert result["checks"]["gateA_window"]["passed"]


def test_direction_join_rejects_out_of_view_leakage(tmp_path):
    paths = _write_fixture(
        tmp_path,
        states={
            "source1": ["out_of_view", "visible_clear", "out_of_view"],
            "source2": ["visible_clear", "visible_clear", "visible_clear"],
        },
    )
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert (
        "main.source1.frame_3.expected_out_of_view_got_visible_clear"
        in result["rejection_reasons"]
    )


def test_direction_join_accepts_any_without_state_restriction(tmp_path):
    paths = list(_write_fixture(tmp_path))
    main = json.loads(paths[0].read_text())
    main["query_visibility"] = "any"
    main["generation_checks"]["query_visibility_window_geometry"]["main"][
        "policy"
    ] = "any"
    paths[0].write_text(json.dumps(main))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pass"


def test_direction_join_requires_geometry_and_audio_verification(tmp_path):
    paths = list(_write_fixture(tmp_path))
    for index in (0, 1):
        fact = json.loads(paths[index].read_text())
        del fact["generation_checks"]["query_visibility_window_geometry"]
        paths[index].write_text(json.dumps(fact))
    audio = json.loads(paths[3].read_text())
    audio["execution_variant_verification"] = {
        "status": "unverified",
        "verified_renders": [],
    }
    paths[3].write_text(json.dumps(audio))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert "main_query_visibility_window_geometry_not_pass" in result["rejection_reasons"]
    assert any(
        reason.startswith("execution_variant_verification")
        for reason in result["rejection_reasons"]
    )


def test_direction_join_rejects_missing_gatea_pair(tmp_path):
    paths = list(_write_fixture(tmp_path))
    audio = json.loads(paths[3].read_text())
    audio["checked_renders"] = 1
    audio["audio_variant_waveform_nonidentity_pairs"] = 0
    audio["gatea_semantic_flip_pairs"] = 0
    audio["execution_variant_verification"]["verified_renders"] = [POINT_ID]
    paths[3].write_text(json.dumps(audio))
    (tmp_path / "audio" / f"{POINT_ID}_gateA").rename(
        tmp_path / "audio" / "missing_gateA"
    )
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert "audio_gateA_render_missing" in result["rejection_reasons"]


def test_direction_join_rejects_variant_and_target_slot_mismatch(tmp_path):
    paths = list(_write_fixture(tmp_path))
    gatea = json.loads(paths[1].read_text())
    gatea["variant"] = "main"
    gatea["target_slot"] = "source1"
    paths[1].write_text(json.dumps(gatea))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert "gateA_fact_variant_must_be_gateA" in result["rejection_reasons"]
    assert "main_gateA_target_slot_not_exchanged" in result["rejection_reasons"]




def test_direction_join_rejects_visual_capture_timeline_mismatch(tmp_path):
    paths = list(_write_fixture(tmp_path))
    wrong_timeline = tmp_path / "wrong_visual_timeline.json"
    wrong_timeline.write_text(json.dumps({"frame_count": 5}))
    receipt_path = tmp_path / "visual_capture" / POINT_ID / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"]["timeline"] = str(wrong_timeline)
    receipt_path.write_text(json.dumps(receipt))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert (
        "visual_capture_timeline_path_mismatch"
        in result["rejection_reasons"]
    )


def test_direction_join_rejects_pixel_input_path_mismatch(tmp_path):
    paths = list(_write_fixture(tmp_path))
    wrong_timeline = tmp_path / "wrong_timeline.json"
    wrong_timeline.write_text(json.dumps({"frame_count": 5}))
    pixel = json.loads(paths[4].read_text())
    pixel["inputs"]["timeline"] = str(wrong_timeline)
    paths[4].write_text(json.dumps(pixel))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert "pixel_timeline_path_mismatch" in result["rejection_reasons"]


def test_direction_join_cli_writes_research_candidate_and_no_clobber(tmp_path):
    paths = _write_fixture(tmp_path)
    output = tmp_path / "joined.json"
    command = [
        sys.executable,
        str(TOOLS / "join_f2_direction_pixel.py"),
        "--main-fact",
        str(paths[0]),
        "--gatea-fact",
        str(paths[1]),
        "--visual-verification",
        str(paths[2]),
        "--audio-verification",
        str(paths[3]),
        "--pixel-evidence",
        str(paths[4]),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["status"] == "pass"
    report = json.loads(output.read_text())
    assert report["status"] == "research_candidate"
    assert report["qualification_claim"] is False

    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 2
    assert "refusing to overwrite output" in second.stderr


def test_direction_join_rejects_invalid_visibility_state(tmp_path):
    paths = list(_write_fixture(tmp_path))
    main = json.loads(paths[0].read_text())
    main["query_visibility"] = "sometimes"
    paths[0].write_text(json.dumps(main))
    with pytest.raises(binder.F2DirectionPixelJoinError, match="query_visibility"):
        binder.join(*paths)
