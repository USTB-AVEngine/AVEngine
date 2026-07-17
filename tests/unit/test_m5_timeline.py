from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m5.timeline import (
    ALLOWED_COUNTERFACTUAL_FIELDS,
    AUDIO_SAMPLE_COUNT,
    DERIVED_COUNTERFACTUAL_FIELDS,
    DURATION_TICKS,
    FOA_AUTHORITY,
    FRAME_COUNT,
    FROZEN_COUNTERFACTUAL_FIELDS,
    M5TimelineError,
    TIME_BASE_HZ,
    VISUAL_VOCAL_ARTICULATION,
    build_counterfactual_pair,
    build_timeline,
    compare_counterfactual_pair,
    frame_sample_interval,
    json_schema_errors,
    sample_boundary,
    validate_dynamic_audio_render_manifest,
    validate_episode_request,
    validate_timeline_semantics,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _request() -> dict:
    value = {
        "schema": "avengine_m5_episode_request_v1",
        "request_id": "m5_two_actor_canary",
        "counterfactual_pair_id": "m5_two_actor_pair",
        "qualification_claim": False,
        "seed": 17,
        "timeline_profile": {
            "time_base_hz": 48_000,
            "duration_ticks": 240_000,
            "video": {
                "fps_num": 15,
                "fps_den": 1,
                "frame_count": 75,
                "ticks_per_frame": 3_200,
                "view_ids": ["view0"],
            },
            "audio": {
                "sample_rate_hz": 16_000,
                "sample_count": 80_000,
                "ticks_per_sample": 3,
                "authority": deepcopy(FOA_AUTHORITY),
            },
        },
        "visual_vocal_articulation": deepcopy(VISUAL_VOCAL_ARTICULATION),
        "listener": {
            "listener_id": "listener0",
            "camera_rig_id": "camera_rig_0",
            "view_id": "view0",
        },
        "actors": [
            {
                "actor_id": "actor0",
                "asset_id": "dog_asset",
                "template_id": "dog_template",
                "body_plan_id": "quadruped",
                "instance_offset_m": [0.0, 0.0, 0.7],
                "semantic_id": 210,
                "skeleton_revision": "skeleton-v1",
                "mesh_sha256": _sha("mesh"),
            },
            {
                "actor_id": "actor1",
                "asset_id": "dog_asset",
                "template_id": "dog_template",
                "body_plan_id": "quadruped",
                "instance_offset_m": [0.25, 0.0, -0.7],
                "semantic_id": 211,
                "skeleton_revision": "skeleton-v1",
                "mesh_sha256": _sha("mesh"),
            },
        ],
        "sources": [
            {
                "source_id": "source0",
                "actor_id": "actor0",
                "semantic_anchor_id": "muzzle",
                "emitter_link": "dog_muzzle",
                "emitter_path_sha256": _sha("emitter-path-0"),
            },
            {
                "source_id": "source1",
                "actor_id": "actor1",
                "semantic_anchor_id": "muzzle",
                "emitter_link": "dog_muzzle",
                "emitter_path_sha256": _sha("emitter-path-1"),
            },
        ],
        "audio_program": {
            "program_id": "six_simultaneous_calls_v1",
            "clip_source_interval": {"start_sample": 3_200, "end_sample": 8_000},
            "fade_samples": 80,
            "linear_gain": 0.18,
            "simultaneous_windows": [
                {
                    "window_id": f"simultaneous{index}",
                    "start_sample": start,
                    "end_sample": start + 4_800,
                }
                for index, start in enumerate(
                    (6_400, 19_200, 32_000, 44_800, 57_600, 70_400)
                )
            ],
        },
        "events": [
            {
                "event_id": "event0",
                "actor_id": "actor0",
                "source_id": "source0",
                "event_type": "vocalization",
                "audio_program_id": "six_simultaneous_calls_v1",
                "emitter_link": "dog_muzzle",
                "emitter_path_sha256": _sha("emitter-path-0"),
                "dry_audio_asset_sha256": _sha("dry-bark-0"),
                "semantic_sync_required": True,
            },
            {
                "event_id": "event1",
                "actor_id": "actor1",
                "source_id": "source1",
                "event_type": "vocalization",
                "audio_program_id": "six_simultaneous_calls_v1",
                "emitter_link": "dog_muzzle",
                "emitter_path_sha256": _sha("emitter-path-1"),
                "dry_audio_asset_sha256": _sha("dry-bark-1"),
                "semantic_sync_required": True,
            },
        ],
        "counterfactual": {
            "operation": "swap_dry_audio_source_routing",
            "variants": ["A", "B"],
            "frozen_fields": list(FROZEN_COUNTERFACTUAL_FIELDS),
            "allowed_changed_fields": list(ALLOWED_COUNTERFACTUAL_FIELDS),
            "derived_changed_fields": list(DERIVED_COUNTERFACTUAL_FIELDS),
        },
    }
    value["request_content_sha256"] = canonical_json_sha256(value)
    return value


def _actor_state(actor_id: str, frame_index: int) -> dict:
    return {
        "actor_id": actor_id,
        "root_transform": {
            "translation_m": [float(actor_id[-1]), 0.0, frame_index / 100.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "action_id": "idle",
        "action_time_ticks": frame_index * 3_200,
        "action_phase": frame_index / 75.0,
        "pose_hash": _sha(f"{actor_id}-pose-{frame_index}"),
        "contacts": {
            "front_left": True,
            "front_right": True,
            "rear_left": True,
            "rear_right": True,
        },
    }


def _visual_frames() -> list[dict]:
    return [
        {
            "actor_states": [
                _actor_state("actor0", frame_index),
                _actor_state("actor1", frame_index),
            ],
            "view_pose_hashes": {"view0": _sha(f"view0-{frame_index}")},
        }
        for frame_index in range(75)
    ]


def _timeline() -> dict:
    return build_timeline(_request(), _visual_frames())


def test_exact_integer_clock_and_boundary_formula() -> None:
    assert TIME_BASE_HZ == 48_000
    assert DURATION_TICKS == 240_000
    assert FRAME_COUNT == 75
    assert AUDIO_SAMPLE_COUNT == 80_000
    assert [sample_boundary(index) for index in range(4)] == [0, 1067, 2133, 3200]
    assert [
        frame_sample_interval(index)[1] - frame_sample_interval(index)[0]
        for index in range(6)
    ] == [
        1067,
        1066,
        1067,
        1067,
        1066,
        1067,
    ]
    assert sample_boundary(75) == 80_000
    assert (
        sum(
            frame_sample_interval(index)[1] - frame_sample_interval(index)[0]
            for index in range(75)
        )
        == 80_000
    )
    with pytest.raises(ValueError, match="0..75"):
        sample_boundary(True)
    with pytest.raises(ValueError, match="0..74"):
        frame_sample_interval(75)


def test_episode_request_schema_and_cross_references_are_closed() -> None:
    request = _request()
    assert not json_schema_errors(request, "avengine_m5_episode_request_v1")
    assert not validate_episode_request(request)
    assert request["timeline_profile"]["audio"]["authority"]["channel_count"] == 4
    assert request["timeline_profile"]["audio"]["authority"]["normalization"] == "N3D"
    assert request["timeline_profile"]["video"]["view_ids"] == ["view0"]
    assert request["visual_vocal_articulation"] == {
        "mode": "disabled_for_shortcut_control",
        "mouth_motion_present": False,
    }
    assert [actor["instance_offset_m"] for actor in request["actors"]] == [
        [0.0, 0.0, 0.7],
        [0.25, 0.0, -0.7],
    ]
    assert [actor["semantic_id"] for actor in request["actors"]] == [210, 211]
    assert len(request["audio_program"]["simultaneous_windows"]) == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("second_view", "view0"),
        ("binaural_authority", "FOA"),
        ("source_reference", "one-to-one"),
        ("long_interval", "exactly six simultaneous windows"),
        ("wrong_window_duration", "source clip duration"),
        ("duplicate_semantic", "distinct semantic IDs"),
        ("same_dry", "distinct SHA-256"),
        ("mouth_mode", "disabled_for_shortcut_control"),
    ],
)
def test_episode_request_semantics_fail_closed(mutation: str, message: str) -> None:
    request = _request()
    if mutation == "second_view":
        request["timeline_profile"]["video"]["view_ids"] = ["view0", "view1"]
    elif mutation == "binaural_authority":
        request["timeline_profile"]["audio"]["authority"]["channel_count"] = 2
    elif mutation == "source_reference":
        request["events"][1]["source_id"] = "source0"
    elif mutation == "long_interval":
        request["audio_program"]["simultaneous_windows"] = [
            {
                "window_id": "continuous",
                "start_sample": 6_400,
                "end_sample": 75_200,
            }
        ]
    elif mutation == "wrong_window_duration":
        request["audio_program"]["simultaneous_windows"][2]["end_sample"] += 1
    elif mutation == "duplicate_semantic":
        request["actors"][1]["semantic_id"] = 210
    elif mutation == "same_dry":
        request["events"][1]["dry_audio_asset_sha256"] = request["events"][0][
            "dry_audio_asset_sha256"
        ]
    elif mutation == "mouth_mode":
        request["visual_vocal_articulation"]["mode"] = "animated"
    request["request_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "request_content_sha256"
        }
    )
    assert any(message in error for error in validate_episode_request(request))


def test_timeline_builder_emits_exact_two_actor_simultaneous_schedule() -> None:
    timeline = _timeline()
    assert not validate_timeline_semantics(timeline, episode_request=_request())
    assert timeline["schema"] == "avengine_authoritative_timeline_v2"
    assert timeline["video"] == {
        "fps_num": 15,
        "fps_den": 1,
        "frame_count": 75,
        "ticks_per_frame": 3200,
        "view_ids": ["view0"],
    }
    assert timeline["audio"] == {
        "sample_rate_hz": 16_000,
        "sample_count": 80_000,
        "ticks_per_sample": 3,
        "channel_count": 4,
    }
    assert len(timeline["frames"]) == 75
    assert timeline["frames"][-1]["sample_end"] == 80_000
    starts = [6_400, 19_200, 32_000, 44_800, 57_600, 70_400]
    assert [event["start_sample"] for event in timeline["audio_events"]] == starts * 2
    assert [event["end_sample"] for event in timeline["audio_events"]] == [
        start + 4_800 for start in starts
    ] * 2

    for frame in timeline["frames"]:
        assert list(frame["view_pose_hashes"]) == ["view0"]
        assert [state["actor_id"] for state in frame["actor_states"]] == [
            "actor0",
            "actor1",
        ]
        assert all(
            state["mouth_state"]["open_ratio"] == 0.0 for state in frame["actor_states"]
        )
        expected = any(
            start <= frame["sample_start"] < start + 4_800 for start in starts
        )
        assert [
            state["mouth_state"]["vocalizing"] for state in frame["actor_states"]
        ] == [expected, expected]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("pts", "pts_ticks"),
        ("boundary", "sample_end"),
        ("view", "view_ids"),
        ("mouth", "open_ratio"),
        ("audio", "four FOA channels"),
        ("event", "exactly six distinct"),
        ("actor", "actor_states"),
    ],
)
def test_timeline_semantic_validator_rejects_cross_field_drift(
    mutation: str, message: str
) -> None:
    timeline = _timeline()
    if mutation == "pts":
        timeline["frames"][4]["pts_ticks"] += 1
    elif mutation == "boundary":
        timeline["frames"][4]["sample_end"] += 1
    elif mutation == "view":
        timeline["video"]["view_ids"] = ["view0", "debug_topdown"]
    elif mutation == "mouth":
        timeline["frames"][16]["actor_states"][0]["mouth_state"]["open_ratio"] = 0.5
    elif mutation == "audio":
        timeline["audio"]["channel_count"] = 2
    elif mutation == "event":
        timeline["audio_events"][6]["start_sample"] += 1
    elif mutation == "actor":
        timeline["frames"][3]["actor_states"].reverse()
    assert any(message in error for error in validate_timeline_semantics(timeline))


def test_semantic_validators_report_malformed_cross_fields_without_throwing() -> None:
    timeline = _timeline()
    timeline["frames"].append(deepcopy(timeline["frames"][-1]))
    timeline["audio_events"][0]["actor_id"] = ["not", "hashable"]
    errors = validate_timeline_semantics(timeline)
    assert any("75 frames" in error for error in errors)
    assert any("does not resolve" in error for error in errors)

    request = _request()
    request["events"][0]["source_id"] = ["not", "hashable"]
    request["request_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "request_content_sha256"
        }
    )
    assert any(
        "does not resolve" in error for error in validate_episode_request(request)
    )


def test_builder_refuses_visual_mouth_motion_and_noncanonical_actor_order() -> None:
    frames = _visual_frames()
    frames[0]["actor_states"][0]["mouth_state"] = {
        "open_ratio": 0.1,
        "vocalizing": False,
    }
    with pytest.raises(M5TimelineError, match="mouth motion"):
        build_timeline(_request(), frames)
    frames = _visual_frames()
    frames[0]["actor_states"].reverse()
    with pytest.raises(M5TimelineError, match="canonical actor order"):
        build_timeline(_request(), frames)


def test_counterfactual_pair_swaps_only_dry_audio_source_routing() -> None:
    pair = build_counterfactual_pair(_request(), _visual_frames())
    assert pair["comparison"]["status"] == "pass"
    assert pair["comparison"]["visual_invariant"] is True
    assert pair["comparison"]["dry_audio_source_routing_swap"] is True
    assert not pair["comparison"]["unexpected_differences"]
    assert pair["allowed_changed_fields"] == list(ALLOWED_COUNTERFACTUAL_FIELDS)

    a = pair["episodes"]["A"]
    b = pair["episodes"]["B"]
    assert a["timeline"]["video"] == b["timeline"]["video"]
    assert a["timeline"]["actors"] == b["timeline"]["actors"]
    assert a["timeline"]["frames"] == b["timeline"]["frames"]
    assert a["visual_state_sha256"] == b["visual_state_sha256"]
    a_hashes = [
        route["dry_audio_asset_sha256"]
        for route in a["dynamic_audio_render_manifest"]["source_routes"]
    ]
    b_hashes = [
        route["dry_audio_asset_sha256"]
        for route in b["dynamic_audio_render_manifest"]["source_routes"]
    ]
    assert b_hashes == list(reversed(a_hashes))
    for episode in (a, b):
        manifest = episode["dynamic_audio_render_manifest"]
        assert not validate_dynamic_audio_render_manifest(
            manifest,
            request=episode["request"],
            timeline=episode["timeline"],
        )
        assert manifest["authority"] == FOA_AUTHORITY
        assert manifest["audio_program"] == episode["request"]["audio_program"]
        assert manifest["actor_instances"] == [
            {
                "actor_id": actor["actor_id"],
                "instance_offset_m": actor["instance_offset_m"],
                "semantic_id": actor["semantic_id"],
            }
            for actor in episode["request"]["actors"]
        ]
        assert [route["emitter_link"] for route in manifest["source_routes"]] == [
            "dog_muzzle",
            "dog_muzzle",
        ]
        assert manifest["frame_sample_boundary"]["formula"] == "B(f)=(3200*f+1)//3"
        assert (
            manifest["render_policy"]["source_pose_evaluation"]
            == "timeline_frame_fixed_state"
        )
        assert (
            manifest["render_policy"]["rir_application"]
            == "raised_cosine_source_time_partition_v1"
        )
        assert (
            manifest["render_policy"]["tail_policy"]
            == "retain_full_tail_then_crop_half_open_0_80000"
        )


def test_counterfactual_comparator_rejects_visual_or_identity_changes() -> None:
    pair = build_counterfactual_pair(_request(), _visual_frames())
    tampered = deepcopy(pair)
    tampered["episodes"]["B"]["timeline"]["frames"][0]["actor_states"][0][
        "pose_hash"
    ] = _sha("tampered-pose")
    proof = compare_counterfactual_pair(tampered)
    assert proof["status"] == "fail"
    assert proof["visual_invariant"] is False
    assert any("pose_hash" in path for path in proof["unexpected_differences"])

    tampered = deepcopy(pair)
    tampered["episodes"]["B"]["dynamic_audio_render_manifest"]["source_routes"][0][
        "source_id"
    ] = "source1"
    proof = compare_counterfactual_pair(tampered)
    assert proof["status"] == "fail"
    assert any("source_id" in path for path in proof["unexpected_differences"])


def test_builders_are_deterministic_detached_and_do_not_write(tmp_path) -> None:
    request = _request()
    frames = _visual_frames()
    before = list(tmp_path.iterdir())
    first = build_counterfactual_pair(request, frames)
    second = build_counterfactual_pair(request, frames)
    assert first == second
    assert list(tmp_path.iterdir()) == before
    first["episodes"]["A"]["timeline"]["frames"][0]["actor_states"][0]["contacts"][
        "front_left"
    ] = False
    assert frames[0]["actor_states"][0]["contacts"]["front_left"] is True
    assert (
        second["episodes"]["A"]["timeline"]["frames"][0]["actor_states"][0]["contacts"][
            "front_left"
        ]
        is True
    )
