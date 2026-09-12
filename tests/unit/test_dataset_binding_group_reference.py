from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct
import wave

import pytest

from avengine.dataset.binding_group_reference import (
    REFERENCE_ACTOR_IDS,
    ReferenceNativeError,
    _clone_reference_variant,
    build_reference_audio_assignment_plan,
    compare_reference_visual_plans,
    compare_reference_audio_pcm,
    plan_reference_group,
    _pool_sound_ids_for_actor,
    schedule_reference_audio_plan,
    select_two_nearest_to_leftmost,
    select_two_nearest_to_named_reference,
    validate_candidate_scope,
)


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)


def _write_float_wav(path: Path, payload: bytes) -> None:
    fmt = struct.pack("<HHIIHH", 3, 2, 16000, 16000 * 2 * 4, 2 * 4, 32)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def test_compare_reference_audio_pcm_accepts_native_float_wav(tmp_path):
    payload = struct.pack("<ffff", 0.1, -0.2, 0.3, -0.4)
    left = tmp_path / "left_float.wav"
    right = tmp_path / "right_float.wav"
    _write_float_wav(left, payload)
    _write_float_wav(right, payload)
    result = compare_reference_audio_pcm(left, right)
    assert result["status"] == "pass"
    assert result["same"] is True
    assert result["channels"] == 2
    assert result["sample_width"] == 4
    assert result["sample_rate_hz"] == 16000
    assert result["frame_count"] == 2


def _request(tmp_path: Path) -> dict:
    source_registry = tmp_path / "source_registry.json"
    assets = []
    for index, actor_id in enumerate(REFERENCE_ACTOR_IDS, start=1):
        if actor_id == "source4":
            assets.append({
                "asset_id": f"asset_{index}",
                "revision": "v1",
                "display_label": "Blue-shirt person",
                "entity_class": "articulated_human",
                "identity": {"species_id": "human", "breed_id": None},
                "realized_attributes": {"top_color": "blue"},
            })
        else:
            assets.append({
                "asset_id": f"asset_{index}",
                "revision": "v1",
                "display_label": f"Dog {index}",
                "entity_class": "articulated_animal",
                "identity": {"species_id": "dog", "breed_id": f"dog_{index}"},
                "realized_attributes": {"coat_profile": {"value": "reviewed"}},
            })
    source_registry.write_text(json.dumps({"assets": assets}))
    pool = tmp_path / "sound_pool.json"
    pool.write_text("{}")
    prepared = tmp_path / "prepared_manifest.json"
    prepared.write_text("{}")
    paths = []
    for index in range(1, 4):
        clip = tmp_path / f"clip_{index}.wav"
        _write_wav(clip)
        paths.append(clip)
    return {
        "schema": "avengine_native_qa_room_request_v1",
        "qa_ids": ["QA-05"],
        "source_asset_ids": [f"asset_{i}" for i in range(1, 5)],
        "entities": {"total_count": 4, "silent_count": 1},
        "camera": {
            "motion": "static",
            "fov_deg": 85,
            "resolution_hw": [720, 1280],
        },
        "frame_count": 150,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "post_assembly_convolution_gain": 0.5,
        "profile": {"reserve_tail_s": 3.0},
        "room_id": "room",
        "source_registry": str(source_registry),
        "sound_pool": str(pool),
        "prepared_manifest": str(prepared),
        "visual_selector": {
            "kind": "two_nearest_to_leftmost",
            "minimum_margin_px": 50,
            "candidate_scope_en": "three dogs and one person wearing a blue shirt",
            "candidate_scope_zh": "三只狗和一位穿蓝上衣的人",
        },
        "reference_time_s": 0,
        "window_s": [4, 6],
        "reference_positions_m": {
            "v0": [4.0, 0.0, 0.0],
            "v1": [5.0, 0.0, 0.0],
        },
        "speaking_actor_ids": ["source1", "source2", "source3"],
        "reference_actor_id": "source4",
        "candidate_scope_bindings": {
            "source1": {"kind": "animal", "species_id": "dog"},
            "source2": {"kind": "animal", "species_id": "dog"},
            "source3": {"kind": "animal", "species_id": "dog"},
            "source4": {"kind": "human", "species_id": "human", "top_color": "blue"},
        },
        "expected_pairs_by_variant": {
            "v0": ["source1", "source2"],
            "v1": ["source1", "source3"],
        },
        "audio_start_times_s": {
            "a0": {"source1": 4.0, "source2": 4.0, "source3": 6.0},
            "a1": {"source1": 4.0, "source2": 6.0, "source3": 4.0},
        },
    }


def _plan(tmp_path: Path, request: dict) -> dict:
    clips = [tmp_path / f"clip_{i}.wav" for i in range(1, 4)]
    actors = []
    for index, actor_id in enumerate(REFERENCE_ACTOR_IDS, start=1):
        asset_id = f"asset_{index}"
        anchor = "mouth" if actor_id == "source4" else "muzzle"
        actors.append({
            "actor_id": actor_id,
            "asset_id": asset_id,
            "entity_class": "articulated_human" if actor_id == "source4" else "articulated_animal",
            "identity": {"species_id": "human" if actor_id == "source4" else "dog"},
            "realized_attributes": {"top_color": "blue"} if actor_id == "source4" else {},
            "emitter_binding": {
                "source_slot_id": actor_id,
                "semantic_anchor_id": anchor,
                "emitter_offset_m": [0.0, 1.0, 0.0],
            },
        })
    states = {
        actor_id: {
            "actor_id": actor_id,
            "root_transform": {
                "translation_m": [float(index), 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "planned_emitter_m": [float(index), 1.0, 0.0],
            "moving": False,
            "action_id": "static",
            "action_phase": 0.0,
            "action_time_ticks": 0,
        }
        for index, actor_id in enumerate(REFERENCE_ACTOR_IDS, start=1)
    }
    camera = {
        "candidate_id": "camera_0",
        "motion": "static",
        "position_m": [0.0, 2.0, -5.0],
        "basis": {
            "forward": [0.0, 0.0, 1.0],
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        },
        "horizontal_fov_deg": 85.0,
        "resolution_hw": [720, 1280],
    }
    frames = []
    for frame_index in range(150):
        frames.append({
            "frame_index": frame_index,
            "pts_ticks": frame_index * 3200,
            "camera_state": {**camera, "frame_index": frame_index},
            "actor_states": deepcopy(list(states.values())),
        })
    events = []
    bindings = []
    for index, (actor_id, clip) in enumerate(zip(
        ("source1", "source2", "source3"), clips, strict=True
    ), start=1):
        sound_id = f"sound_{index}"
        event = {
            "event_id": f"evt_src{index}",
            "actor_id": actor_id,
            "source_endpoint_id": f"{actor_id}_muzzle",
            "sound_asset_id": sound_id,
            "path": str(clip),
            "sample_count": 16000,
            "sample_rate_hz": 16000,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 16000,
            "source_activity_intervals_samples": [[1600, 12800]],
        }
        events.append(event)
        bindings.append({
            "sound_asset_id": sound_id,
            "path": str(clip),
            "sample_count": 16000,
            "sample_rate_hz": 16000,
            "sound_class": "dog_bark",
            "species_id": "dog",
            "compatible_asset_ids": ["asset_1", "asset_2", "asset_3"],
            "actor_id": actor_id,
            "source_endpoint_id": f"{actor_id}_muzzle",
        })
    return {
        "kind": "avengine_question_driven_episode",
        "episode_id": "reference_v0",
        "seed": 9,
        "clock": {
            "frame_count": 150,
            "frame_rate_hz": 15.0,
            "sample_rate_hz": 16000,
            "sample_count": 160000,
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
        },
        "scene": {"room_id": "room"},
        "resources": {
            "room_package": {"family": "hm3d", "room_id": "room"},
        },
        "visual_plan": {
            "camera": camera,
            "actors": actors,
            "frames": frames,
        },
        "audio_events": events,
        "voice_bindings": bindings,
        "condition_profile": {"speech_motion": "all_still"},
    }


def _centers():
    return {
        "v0": {
            "source1": [200, 100],
            "source2": [300, 100],
            "source3": [700, 100],
            "source4": [100, 100],
        },
        "v1": {
            "source1": [200, 100],
            "source2": [700, 100],
            "source3": [300, 100],
            "source4": [100, 100],
        },
    }


def test_selector_requires_unique_leftmost_and_nearest_margin():
    proof = select_two_nearest_to_leftmost(
        _centers()["v0"], minimum_margin_px=50, expected_pair=("source1", "source2")
    )
    assert proof["reference_actor_id"] == "source4"
    assert proof["selected_pair"] == ["source1", "source2"]
    assert proof["leftmost_unique_margin_px"] == pytest.approx(100)
    assert proof["nearest_selection_margin_px"] == pytest.approx(400)
    with pytest.raises(ReferenceNativeError, match="leftmost uniqueness"):
        select_two_nearest_to_leftmost(
            {**_centers()["v0"], "source1": [120, 100]},
            minimum_margin_px=50,
        )
    with pytest.raises(ReferenceNativeError, match="nearest-selection"):
        select_two_nearest_to_leftmost(
            {
                "source1": [200, 100],
                "source2": [300, 100],
                "source3": [301, 100],
                "source4": [100, 100],
            },
            minimum_margin_px=50,
        )


def test_scope_and_visual_relation_preserve_silent_reference_only(tmp_path: Path):
    request = _request(tmp_path)
    plan = _plan(tmp_path, request)
    scope = validate_candidate_scope(request, plan=plan)
    assert scope["status"] == "pass"
    assert scope["candidate_scope_en"].startswith("three dogs")
    variant = _clone_reference_variant(
        plan, reference_actor_id="source4", position=[5.0, 0.0, 0.0], variant="v1"
    )
    relation = compare_reference_visual_plans(plan, variant)
    assert relation["status"] == "pass"
    assert relation["changed_actor"] == "source4"
    broken = deepcopy(variant)
    broken["visual_plan"]["frames"][0]["actor_states"][0]["root_transform"]["translation_m"][0] += 1
    with pytest.raises(ReferenceNativeError, match="speaking actor"):
        compare_reference_visual_plans(plan, broken)


def test_audio_schedule_switches_overlap_truth_without_moving_clip_identity(tmp_path: Path):
    request = _request(tmp_path)
    plan = _plan(tmp_path, request)
    original = deepcopy(plan)
    a0, req0, sched0 = schedule_reference_audio_plan(
        plan,
        request,
        audio_variant="a0",
        start_times_s_by_actor=request["audio_start_times_s"]["a0"],
        target_pair=("source1", "source2"),
    )
    a1, req1, sched1 = schedule_reference_audio_plan(
        plan,
        request,
        audio_variant="a1",
        start_times_s_by_actor=request["audio_start_times_s"]["a1"],
        target_pair=("source1", "source3"),
    )
    assert plan == original
    assert sched0["pair_overlap_truth"]["source1+source2"] is True
    assert sched0["pair_overlap_truth"]["source1+source3"] is False
    assert sched1["pair_overlap_truth"]["source1+source3"] is True
    assert sched1["pair_overlap_truth"]["source1+source2"] is False
    for before, after in zip(plan["audio_events"], a0["audio_events"], strict=True):
        assert (after["sound_asset_id"], after["source_endpoint_id"]) == (
            before["sound_asset_id"], before["source_endpoint_id"]
        )
        assert after["source_start_sample"] == 0
        assert after["source_end_sample_exclusive"] == 16000
    assert req0["sound_pool"] == request["sound_pool"]
    assert req1["prepared_manifest"] == request["prepared_manifest"]
    assert sched0["available_tail_s"] == pytest.approx(3.0)


def test_assignment_wrapper_keeps_each_event_on_its_native_actor(tmp_path: Path):
    request = _request(tmp_path)
    plan = _plan(tmp_path, request)
    scheduled, scheduled_request, _ = schedule_reference_audio_plan(
        plan,
        request,
        audio_variant="a0",
        start_times_s_by_actor=request["audio_start_times_s"]["a0"],
        target_pair=("source1", "source2"),
    )
    assigned, rebound = build_reference_audio_assignment_plan(
        scheduled,
        scheduled_request,
        "a1",
        endpoint_by_actor={
            "source1": "source1_muzzle",
            "source2": "source2_muzzle",
            "source3": "source3_muzzle",
        },
    )
    assert [
        (row["event_id"], row["actor_id"], row["source_endpoint_id"])
        for row in assigned["audio_events"]
    ] == [
        ("evt_src1", "source1", "source1_muzzle"),
        ("evt_src2", "source2", "source2_muzzle"),
        ("evt_src3", "source3", "source3_muzzle"),
    ]
    assert rebound["audio_assignment_targets"] == {
        "evt_src1": "source1",
        "evt_src2": "source2",
        "evt_src3": "source3",
    }




def test_pcm_helper_uses_exact_payload(tmp_path: Path):
    left, right, changed = (
        tmp_path / "left.wav",
        tmp_path / "right.wav",
        tmp_path / "changed.wav",
    )
    _write_wav(left)
    _write_wav(right)
    _write_wav(changed)
    with wave.open(str(changed), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\1\0" * 16000)
    assert compare_reference_audio_pcm(left, right)["same"] is True
    assert compare_reference_audio_pcm(left, changed)["same"] is False

def test_plan_reference_group_writes_four_members_and_private_selector_proof(tmp_path: Path):
    request = _request(tmp_path)
    plan = _plan(tmp_path, request)
    plan_path = tmp_path / "plan.json"
    request_path = tmp_path / "request.json"
    plan_path.write_text(json.dumps(plan))
    request_path.write_text(json.dumps(request))
    output = tmp_path / "out"
    summary = plan_reference_group(
        base_plan_path=plan_path,
        request_path=request_path,
        output_root=output,
        pixel_centers_by_variant=_centers(),
        group_id="reference_cpu_v1",
        world_id="world_reference_cpu_v1",
    )
    assert summary["status"] == "research_candidate"
    assert summary["native_execution"] == "not_run"
    assert summary["rlr_execution"] == "not_run"
    assert summary["visual_plan_relation"]["reference_position_only"] is True
    assert summary["audio_columns"]["a0"]["same_timing_clip_endpoint_plan"] is True
    assert summary["audio_schedules"]["v0_a0"]["pair_overlap_truth"]["source1+source2"] is True
    assert summary["audio_schedules"]["v1_a1"]["pair_overlap_truth"]["source1+source3"] is True
    assert len(list((output / "plans").glob("*.json"))) == 4
    assert {p.name for p in (output / "selector").glob("*.json")} == {
        "v0_proof.json",
        "v1_proof.json",
    }
    group = json.loads((output / "group_spec.json").read_text())
    query = group["groups"][0]["query"]
    assert query["visual_selector"]["kind"] == "two_nearest_to_leftmost"
    assert query["visual_selector"]["candidate_scope_en"].startswith("three dogs")
    assert "expected_pairs_by_variant" not in query
    assert "centers_px" not in query

def test_sound_allowlist_uses_registry_compatibility(tmp_path: Path):
    request = _request(tmp_path)
    pool = Path(request["sound_pool"])
    pool.write_text(json.dumps({
        "sounds": [
            {
                "sound_asset_id": "dog_event",
                "sound_class": "dog_bark",
                "species_id": "dog",
                "sample_count": 16000,
                "sample_rate_hz": 16000,
            },
            {
                "sound_asset_id": "speech_event",
                "sound_class": "speech_playback",
                "gender": "M",
                "sample_count": 16000,
                "sample_rate_hz": 16000,
            },
        ]
    }))
    registry = {
        row["asset_id"]: row
        for row in json.loads(Path(request["source_registry"]).read_text())["assets"]
    }
    assert _pool_sound_ids_for_actor(request, registry, "source1") == ["dog_event"]


def test_named_reference_selector_allows_reference_away_from_left_edge():
    centers = {
        "v0": {
            "source1": [400, 100],
            "source2": [600, 100],
            "source3": [800, 100],
            "source4": [500, 100],
        },
        "v1": {
            "source1": [400, 100],
            "source2": [800, 100],
            "source3": [550, 100],
            "source4": [500, 100],
        },
    }
    appearance = {"field": "top_color", "value": "blue"}
    proof0 = select_two_nearest_to_named_reference(
        centers["v0"],
        reference_actor_id="source4",
        reference_appearance=appearance,
        minimum_margin_px=20,
        expected_pair=("source1", "source2"),
    )
    proof1 = select_two_nearest_to_named_reference(
        centers["v1"],
        reference_actor_id="source4",
        reference_appearance=appearance,
        minimum_margin_px=20,
        expected_pair=("source1", "source3"),
    )
    assert proof0["selected_pair"] == ["source1", "source2"]
    assert proof1["selected_pair"] == ["source1", "source3"]
    assert proof0["reference_actor_id"] == proof1["reference_actor_id"] == "source4"
    assert proof0["reference_appearance"] == appearance
    with pytest.raises(ReferenceNativeError, match="nearest-selection"):
        select_two_nearest_to_named_reference(
            {
                "source1": [400, 100],
                "source2": [600, 100],
                "source3": [700, 100],
                "source4": [500, 100],
            },
            reference_actor_id="source4",
            reference_appearance=appearance,
            minimum_margin_px=200,
        )


def test_named_reference_scope_requires_unique_blue_human(tmp_path: Path):
    request = _request(tmp_path)
    request["visual_selector"] = {
        "kind": "two_nearest_to_named_reference",
        "minimum_margin_px": 50,
        "candidate_scope_en": "three dogs and one person wearing a blue shirt",
        "candidate_scope_zh": "三只狗和一位穿蓝上衣的人",
        "reference_appearance": {"field": "top_color", "value": "blue"},
    }
    scope = validate_candidate_scope(request)
    assert scope["status"] == "pass"
    assert scope["named_reference"]["reference_actor_id"] == "source4"
    assert scope["named_reference"]["matching_actor_ids"] == ["source4"]
    request["candidate_scope_bindings"]["source4"]["top_color"] = "green"
    with pytest.raises(ReferenceNativeError, match="candidate scope top_color"):
        validate_candidate_scope(request)


def test_plan_reference_group_named_selector_keeps_reference_id_private(tmp_path: Path):
    request = _request(tmp_path)
    request["visual_selector"] = {
        "kind": "two_nearest_to_named_reference",
        "minimum_margin_px": 20,
        "candidate_scope_en": "three dogs and one person wearing a blue shirt",
        "candidate_scope_zh": "三只狗和一位穿蓝上衣的人",
        "reference_appearance": {"field": "top_color", "value": "blue"},
    }
    plan = _plan(tmp_path, request)
    plan_path = tmp_path / "plan_named.json"
    request_path = tmp_path / "request_named.json"
    plan_path.write_text(json.dumps(plan))
    request_path.write_text(json.dumps(request))
    named_centers = {
        "v0": {
            "source1": [400, 100],
            "source2": [600, 100],
            "source3": [800, 100],
            "source4": [500, 100],
        },
        "v1": {
            "source1": [400, 100],
            "source2": [800, 100],
            "source3": [550, 100],
            "source4": [500, 100],
        },
    }
    summary = plan_reference_group(
        base_plan_path=plan_path,
        request_path=request_path,
        output_root=tmp_path / "named_out",
        pixel_centers_by_variant=named_centers,
        group_id="reference_named_cpu_v1",
        world_id="world_reference_named_cpu_v1",
    )
    assert Path(summary["selector_proofs"]["v0"]).is_file()
    assert Path(summary["selector_proofs"]["v1"]).is_file()
    proof0 = json.loads(Path(summary["selector_proofs"]["v0"]).read_text())
    assert proof0["reference_actor_id"] == "source4"
    group = json.loads((tmp_path / "named_out" / "group_spec.json").read_text())
    query = group["groups"][0]["query"]
    assert query["visual_selector"]["kind"] == "two_nearest_to_named_reference"
    assert query["visual_selector"]["reference_appearance"] == {
        "field": "top_color",
        "value": "blue",
    }
    assert "reference_actor_id" not in query["visual_selector"]


def test_reference_audio_pairs_are_configurable_with_shared_source2(tmp_path: Path):
    request = _request(tmp_path)
    request["expected_pairs_by_variant"] = {
        "v0": ["source1", "source2"],
        "v1": ["source2", "source3"],
    }
    request["audio_start_times_s"] = {
        "a0": {"source1": 4, "source2": 4, "source3": 6},
        "a1": {"source1": 4, "source2": 5, "source3": 5},
    }
    plan = _plan(tmp_path, request)
    original = deepcopy(plan)
    scheduled0, _, schedule0 = schedule_reference_audio_plan(
        plan,
        request,
        audio_variant="a0",
        start_times_s_by_actor=request["audio_start_times_s"]["a0"],
        target_pair=("source1", "source2"),
    )
    scheduled1, _, schedule1 = schedule_reference_audio_plan(
        plan,
        request,
        audio_variant="a1",
        start_times_s_by_actor=request["audio_start_times_s"]["a1"],
        target_pair=("source2", "source3"),
    )
    assert plan == original
    assert schedule0["target_pair"] == ["source1", "source2"]
    assert schedule0["opposite_pair"] == ["source2", "source3"]
    assert schedule0["pair_overlap_truth"]["source1+source2"] is True
    assert schedule0["pair_overlap_truth"]["source2+source3"] is False
    assert schedule1["target_pair"] == ["source2", "source3"]
    assert schedule1["opposite_pair"] == ["source1", "source2"]
    assert schedule1["pair_overlap_truth"]["source2+source3"] is True
    assert schedule1["pair_overlap_truth"]["source1+source2"] is False
    for before, after in zip(plan["audio_events"], scheduled0["audio_events"], strict=True):
        assert (before["event_id"], before["sound_asset_id"], before["source_endpoint_id"]) == (
            after["event_id"], after["sound_asset_id"], after["source_endpoint_id"]
        )
