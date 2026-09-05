from __future__ import annotations

import json

import pytest

from avengine.qa.unified_catalog import (
    CATALOG,
    generate_unified_questions,
    get_requirements,
    normalize_episode_bundle,
)
from avengine.qa.unified_scoring import score_unified_item


def _fixture() -> dict:
    frame_count = 40
    colors = ("blue", "pink", "green", "white")
    positions: dict[str, list[list[float]]] = {}
    for index, actor_id in enumerate(("a0", "a1", "a2", "a3")):
        rows: list[list[float]] = []
        for frame in range(frame_count):
            if actor_id == "a0":
                if frame <= 2:
                    point = [2.0, 0.0, -4.0]
                elif frame <= 5:
                    point = [1.0, 0.0, -3.0]
                elif frame == 6:
                    point = [1.0, 0.0, -3.0]
                else:
                    point = [-5.0, 0.0, -2.0]
            elif actor_id == "a1":
                point = [5.0, 0.0, -2.0] if frame < 12 else [2.0, 0.0, -1.0]
            elif actor_id == "a2":
                point = [0.0, 0.0, 5.0]
            else:
                point = [0.0, 0.0, -5.0]
            rows.append(point)
        positions[actor_id] = rows

    actors = {
        actor_id: {
            "asset_id": f"asset_{actor_id}",
            "species_id": "human",
            "display_label": f"{color} actor",
            "realized_attributes": {
                "coat_profile": {"value": color},
            },
        }
        for actor_id, color in zip(("a0", "a1", "a2", "a3"), colors)
    }
    frame_readbacks = {
        "clock": {
            "frame_count": frame_count,
            "frame_rate_hz": 10,
            "sample_rate_hz": 16000,
            "sample_count": 64000,
        },
        "actors": {
            actor_id: [
                {"frame_index": frame, "position_m": position}
                for frame, position in enumerate(path)
            ]
            for actor_id, path in positions.items()
        },
        "emitters": {
            actor_id: [
                {"frame_index": frame, "position_m": position}
                for frame, position in enumerate(path)
            ]
            for actor_id, path in positions.items()
        },
    }
    visibility: dict[str, dict] = {}
    for actor_id in actors:
        frames = []
        for frame in range(frame_count):
            state = "visible_clear"
            if actor_id == "a0":
                if frame == 8:
                    state = "fully_occluded"
                elif frame == 10:
                    state = "visible_occluded"
                elif frame == frame_count - 1:
                    state = "fully_occluded"
            if actor_id == "a3" and frame < 3:
                state = "out_of_view"
            record = {"frame_index": frame, "state": state}
            if actor_id == "a3" and frame == 3:
                record["target_centroid_xy_px"] = [90.0, 50.0]
            if actor_id == "a0" and frame == 10:
                record["occluder_instance_ids"] = ["table"]
            if actor_id == "a1" and frame == 12:
                record["state"] = "visible_occluded"
                record["occluder_instance_ids"] = ["chair"]
            frames.append(record)
        visibility[actor_id] = {
            "frames": frames,
            "semantic_id": 100 + int(actor_id[1:]),
            "state_counts": {},
        }
    pixel_truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "resolution_hw": [100, 100],
        "per_instance": visibility,
    }
    events = [
        {
            "event_id": "e0",
            "actor_id": "a0",
            "sound_asset_id": "s0",
            "start_sample": 3200,
            "end_sample_exclusive": 9600,
            "sound_class": "bark",
            "event_segmentation_status": "reviewed",
        },
        {
            "event_id": "e1",
            "actor_id": "a1",
            "sound_asset_id": "s1",
            "start_sample": 16000,
            "end_sample_exclusive": 22400,
            "sound_class": "speech",
            "transcript": "hello world",
        },
        {
            "event_id": "e2",
            "actor_id": "a2",
            "sound_asset_id": "s2",
            "start_sample": 19200,
            "end_sample_exclusive": 25600,
            "sound_class": "laugh",
            "transcript": "ha ha",
            "event_segmentation_status": "reviewed",
        },
        {
            "event_id": "e3",
            "actor_id": "a3",
            "sound_asset_id": "s3",
            "start_sample": 32000,
            "end_sample_exclusive": 36800,
            "sound_class": "whistle",
            "transcript": "tweet",
            "event_segmentation_status": "reviewed",
        },
    ]
    return {
        "episode_id": "unified_fixture",
        "actors": actors,
        "frame_readbacks": {
            **frame_readbacks,
            "camera": [
                {"frame_index": frame, "position_m": [0.0, 0.0, 0.0], "yaw_deg": 0.0}
                for frame in range(frame_count)
            ],
        },
        "pixel_visibility_truth": pixel_truth,
        "appearance_review": {
            "status": "reviewed",
            "actors": {
                actor_id: {
                    "status": "reviewed",
                    "value": color,
                    "frame_refs": [0, 10, 20],
                }
                for actor_id, color in zip(
                    ("a0", "a1", "a2", "a3"),
                    colors,
                )
            },
        },
        "audio_program": {
            "timeline": {
                "time_base_hz": 48000,
                "sample_rate_hz": 16000,
                "frame_count": frame_count,
                "sample_count": 64000,
            },
            "events": events,
        },
        "audio_readback": {
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 64000,
            "proof": "fixture_readback",
            "channel_order": ["left", "right"],
            "hrtf_id": "fixture_hrtf",
            "wet_tail_intervals": [
                {"event_id": "e0", "start_s": 0.2, "end_s": 0.61},
                {"event_id": "e1", "start_s": 1.0, "end_s": 1.41},
                {"event_id": "e2", "start_s": 1.2, "end_s": 1.61},
                {"event_id": "e3", "start_s": 2.0, "end_s": 2.31},
            ],
        },
        "occluder_registry": {
            "table": "table",
            "chair": "chair",
        },
    }


def test_catalog_is_complete_and_requirements_are_pre_capture() -> None:
    assert [item["qa_id"] for item in CATALOG] == [
        f"QA-{index:02d}" for index in range(1, 25)
    ]
    requirements = get_requirements("qa_13")
    assert requirements["qa_id"] == "QA-13"
    assert requirements["min_entities"] == 2
    assert requirements["events"]["post_sound_query"] is True
    for key in (
        "min_entities",
        "events",
        "appearance",
        "speech_content",
        "motion",
        "after_sound",
        "pixel_visibility",
        "occlusion",
        "entry",
        "distinct_sound_classes",
    ):
        assert key in requirements["potential_requirements"]
        assert key in requirements["evidence_requirements"]


def test_normalize_accepts_native_shapes_without_old_fact_wrapper() -> None:
    facts = normalize_episode_bundle(_fixture())
    assert facts["schema"] == "avengine_qa_unified_episode_facts_v1"
    assert facts["time"]["frame_count"] == 40
    assert set(facts["actors"]) == {"a0", "a1", "a2", "a3"}
    assert facts["audio"]["status"] == "pass"
    assert facts["listener"]["source"] == "frame_readbacks.camera"
    assert facts["events"][0]["start_frame"] == 2


def test_generation_restores_visibility_frame_keys_after_json_round_trip() -> None:
    normalized = normalize_episode_bundle(_fixture())
    round_tripped = json.loads(json.dumps(normalized))
    result = generate_unified_questions(round_tripped, qa_ids=["QA-13"])
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    assert result["items"][0]["evidence"]["query_frame"] == 7


def test_qa13_generated_open_form_uses_scorer_angle_convention() -> None:
    normalized = normalize_episode_bundle(_fixture())
    round_tripped = json.loads(json.dumps(normalized))
    result = generate_unified_questions(round_tripped, qa_ids=["QA-13"])
    item = result["items"][0]
    form = item["forms"]["open"]
    assert form["convention"] == "right_positive"
    assert form["convention_description"] == (
        "azimuth_deg; front=0°, right_positive, range=[-180°,180°)"
    )
    scored = score_unified_item(item, str(form["truth"]), form="open")
    assert scored["status"] == "scored"
    assert scored["score"] == 1.0
    assert scored["angle_convention"] == "right_positive"


def test_ue_camera_basis_maps_forward_and_right_without_a_90_degree_bias() -> None:
    raw = _fixture()
    camera = raw["frame_readbacks"]["camera"]
    for record in camera:
        record["rotation_deg"] = [0.0, 0.0, -90.0]
    facts = normalize_episode_bundle(raw)
    basis = facts["listener"]["basis_m3"][0]
    assert basis["forward"] == pytest.approx([0.0, 0.0, -1.0])
    assert basis["right"] == pytest.approx([1.0, 0.0, 0.0])


def test_generation_reports_valid_rows_and_exact_missing_conditions() -> None:
    result = generate_unified_questions(_fixture(), seed="fixture")
    assert result["counts"] == {"requested": 24, "valid": 24, "deferred": 0}
    assert {item["qa_id"] for item in result["items"]} == {
        f"QA-{index:02d}" for index in range(1, 25)
    }
    by_id = {item["qa_id"]: item for item in result["items"]}
    assert by_id["QA-13"]["truth"]["value"] < -60.0
    assert "numeric azimuth" in by_id["QA-13"]["forms"]["open"]["question_en"]
    assert "front is 0°" in by_id["QA-13"]["forms"]["open"]["question_en"]
    assert "front [-45°, 45°)" in by_id["QA-13"]["forms"]["mcq"]["question_en"]
    assert by_id["QA-13"]["forms"]["open"]["question_en"] != by_id["QA-13"]["forms"]["mcq"]["question_en"]
    assert by_id["QA-16"]["truth"]["value"] == "farther"
    assert by_id["QA-17"]["truth"]["value"] == "yes"
    assert by_id["QA-22"]["truth"]["value"] == [4, 4]
    assert by_id["QA-23"]["truth"]["value"] == [4]
    assert by_id["QA-24"]["truth"]["value"] == "fully_occluded"
    for option in by_id["QA-03"]["model_input"]["mcq"]["options"]:
        assert "value" not in option
        assert "source" not in option["label_en"].casefold()


def test_missing_pixel_truth_is_deferred_without_fabricating_states() -> None:
    raw = _fixture()
    raw.pop("pixel_visibility_truth")
    result = generate_unified_questions(raw, qa_ids=["QA-07", "QA-08", "QA-24"])
    assert result["counts"] == {"requested": 3, "valid": 0, "deferred": 3}
    assert {item["code"] for item in result["deferred"]} == {
        "missing_visibility_resolution",
        "no_event_visibility",
        "missing_pixel_visibility",
    }


def test_unresolved_audio_binding_cannot_become_a_false_negative() -> None:
    raw = _fixture()
    raw["audio_program"]["events"][0].pop("actor_id")
    result = generate_unified_questions(raw, qa_ids=["QA-03", "QA-23"])
    assert [item["qa_id"] for item in result["items"]] == ["QA-23"]
    assert result["deferred"][0]["code"] == "unresolved_event_attribution"


def test_non_speech_event_count_requires_explicit_segmentation_review() -> None:
    raw = _fixture()
    raw["audio_program"]["events"][0].pop("event_segmentation_status")
    result = generate_unified_questions(raw, qa_ids=["QA-23"])
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "event_segmentation_not_reviewed"


def test_post_sound_query_skips_wet_tail_and_programmed_events() -> None:
    raw = _fixture()
    raw["audio_readback"]["wet_tail_intervals"][0]["end_s"] = 0.9
    result = generate_unified_questions(raw, qa_ids=["QA-13", "QA-17"])
    assert result["counts"]["valid"] == 2
    for item in result["items"]:
        assert item["evidence"]["post_sound"]["query_frame"] >= 17


def test_qa13_defers_only_mcq_when_open_angles_share_a_sector() -> None:
    raw = _fixture()
    for actor_id, point in {
        "a0": [-2.0, 0.0, -2.4],
        "a1": [1.4, 0.0, -2.4],
        "a2": [0.0, 0.0, 5.0],
        "a3": [0.0, 0.0, 5.0],
    }.items():
        for stream in ("actors", "emitters"):
            for frame in raw["frame_readbacks"][stream][actor_id]:
                frame["position_m"] = list(point)
    result = generate_unified_questions(raw, qa_ids=["QA-13"])
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    item = result["items"][0]
    assert item["form_status"]["open"]["status"] == "pass"
    assert item["form_status"]["mcq"] == {
        "status": "deferred",
        "code": "mcq_same_sector",
        "detail": (
            "MCQ requires every distractor to occupy a different "
            "equal-width half-open sector"
        ),
        "target_sector": "front",
        "conflicting_distractors": {"a1": "front"},
    }
    assert "mcq" not in item["forms"]
    assert "0.700 seconds" in item["question"]["en"]
    assert "第0.700秒" in item["question"]["zh"]


def test_qa13_searches_a_later_frame_for_mcq_sector_separation() -> None:
    raw = _fixture()
    for actor_id, point in {
        "a1": [1.4, 0.0, -2.4],
        "a2": [0.0, 0.0, 5.0],
        "a3": [0.0, 0.0, 5.0],
    }.items():
        for stream in ("actors", "emitters"):
            for frame in raw["frame_readbacks"][stream][actor_id]:
                frame["position_m"] = list(point)
    for stream in ("actors", "emitters"):
        for frame in raw["frame_readbacks"][stream]["a0"]:
            frame["position_m"] = (
                [-2.0, 0.0, -2.4]
                if frame["frame_index"] == 7
                else [-5.0, 0.0, -2.0]
            )
    result = generate_unified_questions(raw, qa_ids=["QA-13"])
    item = result["items"][0]
    assert item["form_status"]["mcq"]["status"] == "pass"
    assert item["evidence"]["query_frame"] == 9
    assert item["evidence"]["target_sector"] == "left"
    assert item["evidence"]["distractor_sectors"]["a1"] == "front"


def test_missing_wet_tail_evidence_defers_temporal_queries() -> None:
    raw = _fixture()
    raw["audio_readback"].pop("wet_tail_intervals")
    result = generate_unified_questions(raw, qa_ids=["QA-13", "QA-16", "QA-17"])
    assert result["counts"] == {"requested": 3, "valid": 0, "deferred": 3}
    assert {item["code"] for item in result["deferred"]} == {
        "no_valid_post_sound_window"
    }


def test_occlusion_questions_emit_no_only_when_the_trigger_exists() -> None:
    raw = _fixture()
    full_frames = raw["pixel_visibility_truth"]["per_instance"]["a0"]["frames"]
    for frame in full_frames[9:]:
        frame["state"] = "fully_occluded"
    partial_frames = raw["pixel_visibility_truth"]["per_instance"]["a1"]["frames"]
    partial_frames[10]["state"] = "visible_occluded"
    partial_frames[11]["state"] = "visible_occluded"
    for frame in partial_frames[12:]:
        frame["state"] = "fully_occluded"
    result = generate_unified_questions(raw, qa_ids=["QA-09", "QA-11"])
    assert result["counts"] == {"requested": 2, "valid": 2, "deferred": 0}
    by_id = {item["qa_id"]: item for item in result["items"]}
    assert by_id["QA-09"]["truth"]["value"] == "no"
    assert by_id["QA-11"]["truth"]["value"] == "no"


def test_qa10_keeps_open_form_when_only_one_real_occluder_is_registered() -> None:
    raw = _fixture()
    raw["pixel_visibility_truth"]["per_instance"]["a1"]["frames"][12].pop(
        "occluder_instance_ids"
    )
    raw["occluder_registry"].pop("chair")
    result = generate_unified_questions(raw, qa_ids=["QA-10"])
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    item = result["items"][0]
    assert item["form_status"]["open"]["status"] == "pass"
    assert item["form_status"]["mcq"]["status"] == "deferred"
    assert "mcq" not in item["forms"]
