"""Unit tests for room-centric QA-v3 pilot assembly/finalization helpers."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from assemble_qa_v3_room_pilot import (  # noqa: E402
    STRATA,
    _balanced_choice,
    _interleave_by_height,
    _point_signature,
    assemble,
)
from build_qa_v3_released_probe_items import build as build_released_items  # noqa: E402
from finalize_qa_v3_room_pilot import _card17_distinct  # noqa: E402
from materialize_qa_v3_dual_gateb import _swap_dynamic_states  # noqa: E402


def test_balanced_choice_round_robins_answer_strata():
    pool = []
    for label, count in [("left", 7), ("center", 2), ("right", 8)]:
        for index in range(count):
            pool.append({
                "mcq_truth_option": label,
                "geometry_signature": [label, index],
            })
    chosen = _balanced_choice(pool, 6)
    assert Counter(item["mcq_truth_option"] for item in chosen) == {
        "left": 2, "center": 2, "right": 2}


def test_camera_height_is_a_declared_secondary_stratum():
    """The 1.8 m fallback concentrates in cluttered corners, so a selection that
    only balances answers can hand every tall-camera clip to one answer."""
    assert STRATA == ("mcq_truth_option", "camera_height_m")
    pool = []
    # every 'left' candidate that comes first is a fallback pose; a naive
    # head-of-list pick would take only those
    for index in range(6):
        pool.append({"mcq_truth_option": "left", "camera_height_m": 1.8,
                     "camera_height_fallback_used": True,
                     "geometry_signature": ["left-high", index]})
    for index in range(6):
        pool.append({"mcq_truth_option": "left", "camera_height_m": 1.471,
                     "camera_height_fallback_used": False,
                     "geometry_signature": ["left-low", index]})
    for index in range(6):
        pool.append({"mcq_truth_option": "right", "camera_height_m": 1.471,
                     "camera_height_fallback_used": False,
                     "geometry_signature": ["right-low", index]})
    chosen = _balanced_choice(pool, 6)
    # answers stay balanced, and the tall camera no longer owns one answer
    assert Counter(item["mcq_truth_option"] for item in chosen) == {
        "left": 3, "right": 3}
    left_heights = Counter(item["camera_height_m"] for item in chosen
                           if item["mcq_truth_option"] == "left")
    # both heights appear and neither owns the answer group
    assert set(left_heights) == {1.8, 1.471}, left_heights
    assert max(left_heights.values()) <= 2, left_heights

    # a group with one height is passed through unchanged
    single = [{"camera_height_m": 1.471, "geometry_signature": ["a", i]}
              for i in range(3)]
    assert _interleave_by_height(single) == single
    # a missing height is a stratum of its own rather than a crash
    mixed = [{"camera_height_m": None, "geometry_signature": ["n", 0]},
             {"camera_height_m": 1.8, "geometry_signature": ["h", 0]}]
    assert len(_interleave_by_height(mixed)) == 2


def test_route_swap_changes_dynamic_tracks_without_swapping_identity():
    timeline = {
        "frames": [{
            "actor_states": [
                {
                    "source_slot_id": "source1",
                    "asset_id": "asset_a",
                    "translation_ue_cm": [1.0, 0.0, 0.0],
                    "yaw_ue_deg": 10.0,
                    "action_id": "idle",
                    "action_phase": 0.0,
                },
                {
                    "source_slot_id": "source2",
                    "asset_id": "asset_b",
                    "translation_ue_cm": [2.0, 0.0, 0.0],
                    "yaw_ue_deg": 20.0,
                    "action_id": "walk",
                    "action_phase": 0.5,
                },
            ]
        }]
    }
    twin = _swap_dynamic_states(timeline)
    left, right = twin["frames"][0]["actor_states"]
    assert left["source_slot_id"] == "source1"
    assert left["asset_id"] == "asset_a"
    assert left["translation_ue_cm"] == [2.0, 0.0, 0.0]
    assert right["source_slot_id"] == "source2"
    assert right["asset_id"] == "asset_b"
    assert right["translation_ue_cm"] == [1.0, 0.0, 0.0]


def test_card17_runtime_distinct_check_uses_actual_readbacks(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    base = {
        "frames": [{
            "camera_pose": {"location_cm": [0, 0, 1], "rotation_deg": [0, 0, 0]},
            "actor_anchor_poses": {
                "source1": {"location_cm": [1, 0, 0]},
                "source2": {"location_cm": [2, 0, 0]},
            },
        }]
    }
    changed = json.loads(json.dumps(base))
    changed["frames"][0]["camera_pose"]["location_cm"][0] = 10
    (first / "frame_records.json").write_text(json.dumps(base))
    (second / "frame_records.json").write_text(json.dumps(changed))
    result = _card17_distinct(first, second)
    assert result["runtime_readbacks_differ"] is True



def _write_room_candidate(root, *, scene_id="room_a", profile_id="card12",
                          point_id="card12_001", forms=("open",)):
    batch = root / "batch"
    point = batch / point_id
    (point / "programs").mkdir(parents=True)
    fact = {
        "schema": "qa_v3_fact_record_v1",
        "scene_id": scene_id,
        "point_id": point_id,
        "profile_id": profile_id,
        "answer_forms": list(forms),
        "mcq": {"stem": "which?", "options_space": ["a", "b"],
                "truth_option": "a"},
        "open": {"stem": "which?", "truth_value": "a",
                 "scoring": "closed_set"},
        "camera": {"height_m": 1.471,
                   "translation_ue_cm": [0.0, 0.0, 147.1],
                   "ue_yaw_deg": 0.0},
        "audio": {"program_id": "main_program"},
    }
    (point / "fact_record.json").write_text(json.dumps(fact))
    timeline = {
        "frames": [
            {"camera": {"translation_ue_cm": [0.0, 0.0, 147.1],
                         "yaw_ue_deg": 0.0},
             "actor_states": [
                 {"source_slot_id": "source1",
                  "translation_ue_cm": [100.0, 0.0, 0.0]},
                 {"source_slot_id": "source2",
                  "translation_ue_cm": [200.0, 0.0, 0.0]},
             ]}
            for _ in range(75)
        ]
    }
    (point / "timeline.json").write_text(json.dumps(timeline))
    (point / "actor_selection.json").write_text("{}")
    (point / "m1_capture_request.json").write_text("{}")
    (point / "audio_program.json").write_text("{}")
    (point / "audio_program_gateA.json").write_text("{}")
    (batch / "questions.jsonl").write_text(
        json.dumps({"point_id": point_id, "profile_id": profile_id,
                    "variant": "main", "form": forms[0],
                    "question": "which?"}) + "\n")
    (batch / "questions_gateA.jsonl").write_text(
        json.dumps({"point_id": point_id, "profile_id": profile_id,
                    "variant": "gateA", "form": forms[0],
                    "question": "which?"}) + "\n")
    batch_manifest = batch / "batch_manifest.json"
    batch_manifest.write_text(json.dumps({
        "question_request": {"answer_forms": list(forms)},
    }))
    matrix = root / "scene_profile_matrix.json"
    matrix.write_text(json.dumps({
        "schema": "qa_v3_scene_profile_matrix_v1",
        "scenes": [{"scene_id": scene_id}],
        "question_request": {"answer_forms": list(forms)},
        "matrix": [{"scene_id": scene_id, "profile_id": profile_id,
                     "attempt_status": "generated", "requested_cells": 1,
                     "batch_manifest": str(batch_manifest)}],
    }))
    return matrix, point


def test_assemble_uses_scheduler_quota_and_reports_requested_question_count(tmp_path):
    matrix, _ = _write_room_candidate(tmp_path, forms=("open",))
    manifest = assemble(matrix_roots=[matrix.parent], profiles=[{"id": "card12"}])
    room = manifest["rooms"]["room_a"]
    entry = room["profiles"]["card12"]
    assert entry["status"] == "selected"
    assert entry["requested_cells"] == 1
    assert entry["quota_source"] == "scheduler_requested_cells"
    assert entry["answer_forms"] == ["open"]
    assert entry["question_count"] == 1
    assert entry["counterfactual_question_count"] == 1
    assert room["question_count"] == manifest["question_count"] == 1
    assert manifest["answer_forms"] == ["open"]
    assert entry["candidates"][0]["pilot_id"].startswith("pilot:")


def test_assemble_reports_observed_resource_status_without_profile_whitelist(tmp_path):
    matrix = tmp_path / "scene_profile_matrix.json"
    matrix.write_text(json.dumps({
        "scenes": [{"scene_id": "room_a"}],
        "question_request": {"answer_forms": ["open"]},
        "matrix": [{"scene_id": "room_a", "profile_id": "card13",
                     "attempt_status": "resource_unavailable",
                     "requested_cells": 1}],
    }))
    manifest = assemble(matrix_roots=[tmp_path], profiles=[{"id": "card13"}])
    entry = manifest["rooms"]["room_a"]["profiles"]["card13"]
    assert entry["status"] == "resource_unavailable"
    assert entry["selected_count"] == 0
    assert manifest["resource_profile_count"] == 1
    assert manifest["question_count"] == 0


def test_explicit_per_profile_remains_a_pilot_subset(tmp_path):
    matrix, _ = _write_room_candidate(tmp_path, forms=("open",))
    manifest = assemble(
        matrix_roots=[matrix.parent], profiles=[{"id": "card12"}],
        per_profile=1,
    )
    assert manifest["rooms"]["room_a"]["profiles"]["card12"][
        "quota_source"] == "explicit_per_profile"



def test_assembled_manifest_flows_to_released_adapter_by_pilot_id(tmp_path):
    matrix, _ = _write_room_candidate(tmp_path, forms=("open",))
    manifest = assemble(matrix_roots=[matrix.parent], profiles=[{"id": "card12"}])
    candidate = manifest["rooms"]["room_a"]["profiles"]["card12"]["candidates"][0]
    pilot_id = candidate["pilot_id"]
    audio = tmp_path / "audio" / pilot_id / "audio/binaural"
    media = tmp_path / "media" / pilot_id
    audio.mkdir(parents=True)
    media.mkdir(parents=True)
    (audio / "mixture.wav").write_bytes(b"wav")
    (media / "video_only.mp4").write_bytes(b"mp4")
    rows = build_released_items(
        manifest, audio_root=tmp_path / "audio", media_root=tmp_path / "media")
    assert len(rows) == manifest["question_count"] == 1
    assert rows[0]["form"] == "open"
    assert rows[0]["pilot_id"] == pilot_id


def test_offscreen_manifest_declared_fact_enters_assembly_without_prefix_glob(tmp_path):
    matrix, point = _write_room_candidate(
        tmp_path,
        scene_id="room_a",
        profile_id="offscreen_profile",
        point_id="room_a_f2_offscreen_identity_001",
        forms=("mcq",),
    )
    batch_manifest = point.parent / "batch_manifest.json"
    batch_manifest.write_text(json.dumps({
        "schema": "avengine_qa_v3_offscreen_identity_batch_v1",
        "status": "research_candidate",
        "records": [{
            "point_id": point.name,
            "artifacts": {"fact": str((point / "fact_record.json").resolve())},
        }],
        "counts": {
            "cells_requested": 1,
            "candidates": 1,
            "rejected": 0,
        },
    }))
    manifest = assemble(
        matrix_roots=[matrix.parent],
        profiles=[{
            "id": "offscreen_profile",
            "execution_backend": "offscreen_identity",
        }],
    )
    entry = manifest["rooms"]["room_a"]["profiles"]["offscreen_profile"]
    assert entry["status"] == "selected"
    assert entry["selected_count"] == 1
    assert entry["question_count"] == 1
    assert entry["candidates"][0]["source_point_id"] == point.name


def _geometry_timeline(frame_count):
    return {
        "render": {"frame_count": frame_count},
        "frames": [
            {
                "frame_index": index,
                "pts_ticks": index * 7,
                "camera": {
                    "translation_ue_cm": [0.0, 0.0, 150.0],
                    "yaw_ue_deg": 0.0,
                },
                "actor_states": [
                    {
                        "source_slot_id": "source1",
                        "translation_ue_cm": [float(index), 0.0, 0.0],
                        "yaw_ue_deg": 10.0,
                    },
                    {
                        "source_slot_id": "source2",
                        "translation_ue_cm": [0.0, float(index), 0.0],
                        "yaw_ue_deg": 20.0,
                    },
                ],
            }
            for index in range(frame_count)
        ],
    }


@pytest.mark.parametrize("frame_count", [1, 60, 90, 150])
def test_point_signature_supports_declared_configurable_frame_counts(frame_count):
    timeline = _geometry_timeline(frame_count)
    signature = _point_signature(timeline)
    assert len(signature) == frame_count


def test_point_signature_observes_a_single_changed_middle_frame():
    original = _geometry_timeline(150)
    changed = json.loads(json.dumps(original))
    changed["frames"][100]["actor_states"][0]["translation_ue_cm"][1] = 42.0
    assert _point_signature(original) != _point_signature(changed)


def test_empty_assembly_reports_partial_instead_of_research_candidate(tmp_path):
    matrix = tmp_path / "scene_profile_matrix.json"
    matrix.write_text(json.dumps({
        "scenes": [{"scene_id": "room_a"}],
        "question_request": {"answer_forms": ["open"]},
        "matrix": [{
            "scene_id": "room_a",
            "profile_id": "card13",
            "attempt_status": "not_found_within_budget",
            "requested_cells": 1,
        }],
    }))
    manifest = assemble(matrix_roots=[tmp_path], profiles=[{"id": "card13"}])
    assert manifest["status"] == "partial"
    assert manifest["selected_candidate_count"] == 0
    assert manifest["rooms"]["room_a"]["status"] == "partial"


def test_declared_candidate_must_remain_inside_its_batch_root(tmp_path):
    matrix, point = _write_room_candidate(tmp_path / "matrix")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_fact = outside / "fact_record.json"
    outside_fact.write_text((point / "fact_record.json").read_text())
    batch_manifest = point.parent / "batch_manifest.json"
    batch_manifest.write_text(json.dumps({
        "records": [{
            "point_id": point.name,
            "artifacts": {"fact": str(outside_fact.resolve())},
        }],
        "question_request": {"answer_forms": ["open"]},
    }))
    with pytest.raises(RuntimeError, match="outside its batch root"):
        assemble(matrix_roots=[matrix.parent], profiles=[{"id": "card12"}])
