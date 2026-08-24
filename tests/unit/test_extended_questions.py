from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.qa.extended_questions import (
    ExtendedQuestionEvidenceError,
    choose_refusal_quota,
    event_frame_index,
    event_frame_window,
    listener_local_azimuth_deg_ue,
    load_native_pixel_visibility,
    load_native_point_evidence,
    require_native_out_of_view,
    score_t5_doa_prediction,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_capture(root: Path, *, point_id: str = "p001", yaw: float = 0.0) -> None:
    frames = []
    for index in range(75):
        anchors = {
            "source1": {"location_cm": [10.0, -10.0, 27.1]},
            "source2": {"location_cm": [40.0, 40.0, 27.1]},
        }
        states = [
            {"source_slot_id": slot, "translation_ue_cm": record["location_cm"]}
            for slot, record in anchors.items()
        ]
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "camera": {
                    "translation_ue_cm": [0.0, 0.0, 100.0],
                    "yaw_ue_deg": yaw,
                },
                "camera_pose": {
                    "location_cm": [0.0, 0.0, 100.0],
                    "rotation_deg": [0.0, 0.0, yaw],
                },
                "actor_anchor_poses": anchors,
                "actor_states": states,
            }
        )
    point = root / point_id
    _write_json(point / "frame_records.json", {"frames": frames})
    _write_json(
        point / "research_receipt.json",
        {
            "research_only": True,
            "episode_counted": False,
            "capture": {
                "completed_frame_count": 75,
                "frame_count": 75,
                "frame_rate_hz": 15,
                "ticks_per_frame": 3200,
                "root_readback_summary": {
                    "camera": {"status": "pass"},
                    "source1_actor": {"status": "pass"},
                    "source2_actor": {"status": "pass"},
                },
            },
        },
    )


def _write_visibility(root: Path, *, point_id: str = "p001") -> None:
    def frames(slot: str):
        result = []
        for index in range(75):
            out = slot == "source1" and 4 <= index <= 17
            result.append(
                {
                    "frame_index": index,
                    "state": "out_of_view" if out else "visible_clear",
                    "visible_pixels": 0 if out else 100,
                    "target_pixels": 0 if out else 100,
                }
            )
        return result

    _write_json(
        root / point_id / "pixel_visibility_truth.json",
        {
            "schema": "avengine_qa_pixel_visibility_truth_v1",
            "status": "computed_modal_target_only_v1",
            "frame_indices": list(range(75)),
            "resolution_hw": [720, 1280],
            "per_instance": {
                "source1": {"frames": frames("source1")},
                "source2": {"frames": frames("source2")},
            },
        },
    )


def test_native_readback_drives_positive_right_azimuth(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    evidence = load_native_point_evidence(
        tmp_path,
        point_id="p001",
        expected_camera_position_ue_cm=[0.0, 0.0, 100.0],
        expected_camera_yaw_deg=0.0,
    )

    assert evidence.azimuth_deg("source1", 4) == pytest.approx(-45.0)
    assert evidence.azimuth_deg("source2", 22) == pytest.approx(45.0)
    assert event_frame_index(4000) == 4
    assert event_frame_index(24000) == 22
    assert event_frame_window(4000, 18000) == tuple(range(4, 18))
    assert listener_local_azimuth_deg_ue([10, 10, 0], [0, 0, 0], 0) == pytest.approx(45)


def test_native_readback_rejects_viewpoint_mismatch(tmp_path: Path) -> None:
    _write_capture(tmp_path)
    with pytest.raises(ExtendedQuestionEvidenceError, match="camera yaw disagrees"):
        load_native_point_evidence(
            tmp_path,
            point_id="p001",
            expected_camera_position_ue_cm=[0.0, 0.0, 100.0],
            expected_camera_yaw_deg=180.0,
        )


def test_offscreen_requires_native_target_only_out_of_view(tmp_path: Path) -> None:
    _write_visibility(tmp_path)
    truth = load_native_pixel_visibility(tmp_path, point_id="p001")

    row = require_native_out_of_view(truth, source_slot="source1", frame_index=4)
    assert row["target_pixels"] == 0
    with pytest.raises(ExtendedQuestionEvidenceError, match="not native-pixel"):
        require_native_out_of_view(truth, source_slot="source2", frame_index=4)

    path = tmp_path / "p001/pixel_visibility_truth.json"
    shuffled = json.loads(path.read_text(encoding="utf-8"))
    shuffled["per_instance"]["source1"]["frames"][4]["frame_index"] = 5
    _write_json(path, shuffled)
    with pytest.raises(ExtendedQuestionEvidenceError, match="frame indices"):
        load_native_pixel_visibility(tmp_path, point_id="p001")


def test_t5_circular_scorer_executes_bands_and_wraparound() -> None:
    assert score_t5_doa_prediction("-179 degrees", 179)["score"] == 1.0
    assert score_t5_doa_prediction(15, 0)["band"] == "full_credit"
    assert score_t5_doa_prediction("30°", 0)["score"] == 0.5
    assert score_t5_doa_prediction(30.1, 0)["score"] == 0.0
    with pytest.raises(ExtendedQuestionEvidenceError, match="exactly one number"):
        score_t5_doa_prediction("between 10 and 20 degrees", 0)


def test_refusal_quota_is_deterministic_and_in_range() -> None:
    count, indices, fraction = choose_refusal_quota(15, 3)
    assert count == 2
    assert indices == (0, 2)
    assert 0.10 <= fraction <= 0.15

    with pytest.raises(ExtendedQuestionEvidenceError, match="cannot satisfy"):
        choose_refusal_quota(5, 1)
