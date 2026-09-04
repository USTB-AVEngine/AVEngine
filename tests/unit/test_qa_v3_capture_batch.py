"""Actual capture files, variable clocks and resume correctness."""
from pathlib import Path
import copy
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qa_capture_batch", ROOT / "tools/qa/run_qa_v3_capture_batch.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def capture(root, count=3):
    root.mkdir()
    (root / "arrays").mkdir()
    np.save(root / "arrays/rgb.npy", np.zeros((count, 2, 4, 3), dtype=np.uint8))
    pose = {"location_cm": [1, 2, 3], "rotation_deg": [0, 0, 0]}
    frames = [{"frame_index": i, "pts_ticks": i * 4800,
        "camera": {"translation_ue_cm": [1, 2, 3], "yaw_ue_deg": 0},
        "camera_pose": copy.deepcopy(pose),
        "actor_states": [{"source_slot_id": "sound_slot", "actor_id": "custom_body",
                          "translation_ue_cm": [1, 2, 3], "yaw_ue_deg": 0,
                          "action_id": None, "action_phase": 0}],
        "actor_anchor_poses": {"sound_slot": copy.deepcopy(pose)},
        "animation_readbacks": []} for i in range(count)]
    receipt = {"status": "research_only", "capture": {
        "frame_count": count, "completed_frame_count": count,
        "frame_rate_hz": 10, "ticks_per_frame": 4800,
        "root_readback_summary": {"camera": {"status": "pass"}, "custom_body": {"status": "pass"}},
        "animation_readback_summary": {"status": "not_applicable", "actors": {}}},
        "artifacts": {"rgb": "arrays/rgb.npy", "frame_records": "frame_records.json"}}
    (root / "frame_records.json").write_text(json.dumps({"frames": frames}))
    (root / "research_receipt.json").write_text(json.dumps(receipt))
    return frames, receipt


def test_static_capture_uses_actual_clock_and_requires_the_rgb_file(tmp_path):
    out = tmp_path / "capture"
    capture(out)
    assert TOOL.point_state(out) == "complete"
    (out / "arrays/rgb.npy").unlink()
    assert TOOL.point_state(out) == "partial"


def test_failed_receipt_cannot_pass_using_a_75_frame_animation_marker(tmp_path):
    out = tmp_path / "capture"
    _, receipt = capture(out)
    receipt["status"] = "fail"
    receipt["capture"]["animation_readback_summary"]["frame_count"] = 75
    (out / "research_receipt.json").write_text(json.dumps(receipt))
    assert TOOL.point_state(out) == "partial"


def test_readback_position_error_is_not_reported_as_success(tmp_path):
    out = tmp_path / "capture"
    frames, _ = capture(out)
    frames[1]["actor_anchor_poses"]["sound_slot"]["location_cm"][0] += 100
    (out / "frame_records.json").write_text(json.dumps({"frames": frames}))
    assert TOOL.point_state(out) == "partial"


def test_resume_compares_the_requested_timeline_without_a_content_hash(tmp_path):
    out = tmp_path / "capture"
    frames, _ = capture(out)
    timeline = {"render": {"frame_count": 3, "frame_rate_hz": 10,
                           "ticks_per_frame": 4800, "resolution_hw": [2, 4]}, "frames": frames}
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(timeline))
    assert TOOL.point_state(out, timeline_path=path) == "complete"
    timeline["frames"][1]["camera"]["yaw_ue_deg"] = 30
    path.write_text(json.dumps(timeline))
    assert TOOL.point_state(out, timeline_path=path) == "partial"



def test_intervention_description_resolves_resources_from_its_own_directory(tmp_path):
    point = tmp_path / "point"
    description_dir = point / "interventions"
    gateb_dir = point / "gateb"
    description_dir.mkdir(parents=True)
    gateb_dir.mkdir()
    selection = gateb_dir / "actor_selection_gateB.json"
    timeline = gateb_dir / "timeline_gateB.json"
    selection.write_text(json.dumps({"actors": []}))
    timeline.write_text(json.dumps({"frames": []}))
    description = description_dir / "visual_swap.json"
    description.write_text(json.dumps({
        "actor_selection": "../gateb/actor_selection_gateB.json",
        "timeline": "../gateb/timeline_gateB.json",
    }))
    got_selection, got_timeline, got_description = TOOL.resolve_capture_inputs(
        point, "interventions/visual_swap.json")
    assert got_selection == selection.resolve()
    assert got_timeline == timeline.resolve()
    assert got_description == description.resolve()


def test_capture_input_resolution_keeps_main_default(tmp_path):
    point = tmp_path / "point"
    point.mkdir()
    selection = point / "actor_selection.json"
    timeline = point / "timeline.json"
    selection.write_text("{}")
    timeline.write_text("{}")
    assert TOOL.resolve_capture_inputs(point) == (
        selection.resolve(), timeline.resolve(), None)


def test_resume_rejects_capture_from_a_different_selection(tmp_path):
    out = tmp_path / "capture"
    capture(out)
    frames = json.loads((out / "frame_records.json").read_text())["frames"]
    timeline = {"render": {"frame_count": 3, "frame_rate_hz": 10,
                           "ticks_per_frame": 4800, "resolution_hw": [2, 4]},
                "frames": frames}
    timeline_path = tmp_path / "timeline_gateB.json"
    timeline_path.write_text(json.dumps(timeline))
    main_selection = tmp_path / "actor_selection.json"
    gateb_selection = tmp_path / "actor_selection_gateB.json"
    main_selection.write_text(json.dumps({"actors": []}))
    gateb_selection.write_text(json.dumps({"actors": []}))
    receipt_path = out / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"] = {
        "actor_selection": str(gateb_selection.resolve()),
        "timeline": str(timeline_path.resolve()),
    }
    receipt_path.write_text(json.dumps(receipt))
    assert TOOL.point_state(
        out, timeline_path=timeline_path, selection_path=gateb_selection
    ) == "complete"
    assert TOOL.point_state(
        out, timeline_path=timeline_path, selection_path=main_selection
    ) == "partial"
