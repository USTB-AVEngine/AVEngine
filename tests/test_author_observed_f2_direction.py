from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import author_observed_f2_direction as author  # noqa: E402


def _profile(domain: str) -> dict:
    value = {
        "schema": author.PROFILE_SCHEMA,
        "id": f"test_{domain}",
        "answer_domain": domain,
        "query_event_index": 0,
        "query_window": {
            "kind": "audio_event",
            "start_padding_samples": 0,
            "end_padding_samples": 0,
        },
    }
    if domain == "full_circle":
        value["answer_shape"] = {"equal_bands": 4}
    else:
        value["front_back_split_deg"] = 90.0
    return value


def _write_inputs(tmp_path: Path, *, crossing: bool = False):
    m1 = {
        "primary_camera_rig": {
            "world_from_rig": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "listener": {
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    }
    m1_path = tmp_path / "m1.json"
    m1_path.write_text(json.dumps(m1), encoding="utf-8")
    frame_positions = [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0] if not crossing else [1.0, 0.0, -1.0],
        [2.0, 0.0, 0.0] if not crossing else [1.0, 0.0, 1.0],
        [3.0, 0.0, 0.0],
    ]
    frames = []
    for index, source in enumerate(frame_positions):
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 4,
                "camera_readback": {
                    "agent": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                },
                "source_positions_m": [source],
                "actor_readbacks": [
                    {
                        "source_endpoint_id": "native_source1",
                        "emitter_world_position_m": source,
                    }
                ],
            }
        )
    records = {
        "render": {
            "frame_count": 4,
            "sample_count": 400,
            "sample_rate_hz": 100,
            "ticks_per_frame": 4,
            "time_base_hz": 400,
            "ticks_per_sample": 4,
            "video_fps": 100,
            "frame_rate_hz": 100,
        },
        "frames": frames,
    }
    records_path = tmp_path / "frame_records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    program = {
        "program_id": "program_test",
        "candidate_source_endpoint_ids": ["native_source1"],
        "timeline": {
            "frame_count": 4,
            "sample_count": 400,
            "sample_rate_hz": 100,
            "ticks_per_frame": 4,
            "time_base_hz": 400,
            "ticks_per_sample": 4,
            "video_fps": 100,
        },
        "events": [
            {
                "event_id": "source1_event_0",
                "source_endpoint_id": "native_source1",
                "sound_asset_id": "sound_test",
                "start_sample": 100,
                "end_sample_exclusive": 300,
                "source_start_sample": 0,
                "source_end_sample_exclusive": 200,
            }
        ],
    }
    program_path = tmp_path / "audio_program.json"
    program_path.write_text(json.dumps(program), encoding="utf-8")
    return records_path, m1_path, program_path


def test_author_uses_observed_emitter_and_listener_orientation(tmp_path):
    records, m1, program = _write_inputs(tmp_path)
    output = tmp_path / "out"
    result = author.author_observed_f2_direction(
        frame_records_path=records,
        m1_path=m1,
        audio_program_path=program,
        profile=_profile("full_circle"),
        output_directory=output,
    )
    fact = json.loads((output / "fact_record.json").read_text())
    assert result["event_id"] == "source1_event_0"
    assert fact["truth"]["band_index"] == 3
    assert fact["question"]["mcq"]["truth_option"] == "[90, 180)"
    assert fact["observed_geometry"]["planned_routes_used"] is False
    assert fact["observed_geometry"]["actor_root_used"] is False
    assert fact["audio_validation"]["status"] == "pending"
    assert fact["counterfactual"]["status"] == "pending"
    questions = json.loads((output / "questions.json").read_text())
    assert questions[0]["point_id"] == "program_test:source1_event_0"


def test_front_back_domain_is_derived_from_listener_basis(tmp_path):
    records, m1, program = _write_inputs(tmp_path)
    output = tmp_path / "front_back"
    author.author_observed_f2_direction(
        frame_records_path=records,
        m1_path=m1,
        audio_program_path=program,
        profile=_profile("front_back"),
        output_directory=output,
    )
    fact = json.loads((output / "fact_record.json").read_text())
    assert fact["question"]["mcq"]["options_space"] == ["front", "back"]
    assert fact["question"]["mcq"]["truth_option"] == "front"
    assert fact["truth"]["convention"] == "listener_forward_minus_z_right_plus_x"


def test_event_spanning_answer_bands_is_rejected(tmp_path):
    records, m1, program = _write_inputs(tmp_path, crossing=True)
    with pytest.raises(author.ObservedF2DirectionError, match="more than one full-circle"):
        author.author_observed_f2_direction(
            frame_records_path=records,
            m1_path=m1,
            audio_program_path=program,
            profile=_profile("full_circle"),
            output_directory=tmp_path / "crossing",
        )


def test_listener_rotation_changes_relative_azimuth():
    identity = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    yaw90 = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)],
    }
    assert author.relative_azimuth_from_listener([1.0, 0.0, 0.0], identity) == pytest.approx(90.0)
    assert author.relative_azimuth_from_listener([-1.0, 0.0, 0.0], yaw90) == pytest.approx(0.0)


def test_matching_audio_receipt_connects_main_audio_evidence(tmp_path):
    records, m1, program = _write_inputs(tmp_path)
    program_doc = json.loads(program.read_text())
    receipt = {
        "status": "pass",
        "audio_program": {
            "path": str(program.resolve()),
            "program_id": program_doc["program_id"],
            "timeline": program_doc["timeline"],
        },
        "inputs": {
            "m1_request": {"path": str(m1.resolve())},
            "visual_capture_frame_records": {"path": str(records.resolve())},
        },
        "sources": {
            "source_ids": ["native_source1"],
            "frame_count": 4,
            "frame_rate_hz": 100,
            "ticks_per_frame": 4,
            "time_base_hz": 400,
        },
    }
    receipt_path = tmp_path / "audio_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "with_audio_receipt"
    author.author_observed_f2_direction(
        frame_records_path=records,
        m1_path=m1,
        audio_program_path=program,
        profile=_profile("full_circle"),
        output_directory=output,
        audio_receipt_path=receipt_path,
    )
    fact = json.loads((output / "fact_record.json").read_text())
    assert fact["audio_validation"]["status"] == "pass"
    assert all(fact["audio_validation"]["checks"].values())
    assert fact["truth_status"] == "observed_geometry_and_audio_receipt_pending_media"


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("time_base_hz", 480, "time_base_hz"),
        ("ticks_per_sample", 5, "ticks_per_sample"),
        ("video_fps", 99, "video_fps"),
    ],
)
def test_clock_relationship_mismatch_is_rejected(tmp_path, field, value, pattern):
    records, m1, program = _write_inputs(tmp_path)
    program_doc = json.loads(program.read_text())
    program_doc["timeline"][field] = value
    program.write_text(json.dumps(program_doc), encoding="utf-8")
    with pytest.raises(author.ObservedF2DirectionError, match=pattern):
        author._load_program_and_frames(
            frame_records_path=records,
            m1_path=m1,
            audio_program_path=program,
        )


def test_non_15fps_non_integer_samples_per_frame_clock_is_supported(tmp_path):
    records, m1, program = _write_inputs(tmp_path)
    records_doc = json.loads(records.read_text())
    program_doc = json.loads(program.read_text())
    clock = {
        "frame_count": 3,
        "sample_count": 1_600,
        "sample_rate_hz": 16_000,
        "ticks_per_frame": 1_600,
        "time_base_hz": 48_000,
        "ticks_per_sample": 3,
        "video_fps": 30,
    }
    records_doc["render"].update(clock, frame_rate_hz=30)
    records_doc["frames"] = records_doc["frames"][:3]
    for index, frame in enumerate(records_doc["frames"]):
        frame["frame_index"] = index
        frame["pts_ticks"] = index * 1_600
    program_doc["timeline"] = clock
    records.write_text(json.dumps(records_doc), encoding="utf-8")
    program.write_text(json.dumps(program_doc), encoding="utf-8")
    loaded, _m1, _program, _records_path, _m1_path, _program_path = author._load_program_and_frames(
        frame_records_path=records,
        m1_path=m1,
        audio_program_path=program,
    )
    assert loaded["render"]["frame_count"] == 3
    assert loaded["render"]["frame_rate_hz"] == 30
