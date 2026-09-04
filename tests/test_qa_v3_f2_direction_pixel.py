from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import join_f2_direction_pixel as binder  # noqa: E402


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
    pixel = tmp_path / "pixel_evidence.json"
    pixel.write_text(
        json.dumps({
            "schema": "qa_v3_current_timeline_native_pixel_probe_v1",
            "status": "pass",
            "pixel_visibility": truth,
        })
    )

    geometry = {
        "main": {"passes": True},
        "gateA": {"passes": True},
    }
    common = {
        "scene_id": "room",
        "profile_id": "f2_direction",
        "point_id": "point",
        "query_window_frame_bounds": [2, 4],
        "generation_checks": {
            "query_visibility_window_geometry": geometry,
        },
    }
    main = dict(
        common,
        variant="main",
        target_slot="source1",
        query_visibility="out_of_view",
    )
    gatea = dict(
        common,
        variant="gateA",
        target_slot="source2",
        query_visibility="visible",
    )
    main_path = tmp_path / "main_fact.json"
    gatea_path = tmp_path / "gatea_fact.json"
    main_path.write_text(json.dumps(main))
    gatea_path.write_text(json.dumps(gatea))

    visual = tmp_path / "visual.json"
    visual.write_text(json.dumps({"status": "pass"}))
    audio = tmp_path / "audio.json"
    audio.write_text(json.dumps({
        "failures": [],
        "execution_variant_verification": {"status": "verified"},
    }))
    return main_path, gatea_path, visual, audio, pixel


def test_direction_join_checks_both_declared_windows(tmp_path):
    paths = _write_fixture(tmp_path)
    result = binder.join(*paths)
    assert result["status"] == "research_candidate"
    assert result["pixel_join_status"] == "pass"
    assert result["checks"]["query_visibility_window_geometry"]["both_pass"] is True
    assert result["checks"]["main_window"]["passed"] is True
    assert result["checks"]["gateA_window"]["passed"] is True


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
    paths[0].write_text(json.dumps(main))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pass"
    assert all(row["passed"] for row in result["checks"]["main_window"]["frames"])


def test_direction_join_requires_geometry_and_audio_verification(tmp_path):
    paths = list(_write_fixture(tmp_path))
    main = json.loads(paths[0].read_text())
    del main["generation_checks"]["query_visibility_window_geometry"]
    paths[0].write_text(json.dumps(main))
    gatea = json.loads(paths[1].read_text())
    del gatea["generation_checks"]["query_visibility_window_geometry"]
    paths[1].write_text(json.dumps(gatea))
    audio = json.loads(paths[3].read_text())
    del audio["execution_variant_verification"]
    paths[3].write_text(json.dumps(audio))
    result = binder.join(*paths)
    assert result["pixel_join_status"] == "pixel_rejected"
    assert "main_query_visibility_window_geometry_not_pass" in result["rejection_reasons"]
    assert any(
        reason.startswith("execution_variant_verification_status_")
        for reason in result["rejection_reasons"]
    )


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
