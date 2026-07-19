from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import load_json
from avengine.m5_1.delivery import (
    M51DeliveryError,
    actual_emitter_trajectory_record,
    binaural_frame_diagnostics,
    build_legacy_overlay_tracks,
    event_overlay_state,
    executable_event_mappings,
    semantic_centroid_track,
    source_actor_binding_record,
    validate_retained_acoustic_binding,
    verify_source_event_audio_activity,
)
from avengine.m5_1.dry_audio import DryAudioClipSpec
from avengine.m5_1.source_contracts import load_source_manifest


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPOSITORY / "examples/m5_1/legacy_apartment/source_manifest.json"
ROUTE_MANIFEST = REPOSITORY / "examples/m5_1/legacy_apartment/route_manifest.json"


def test_semantic_centroid_uses_nan_only_when_actor_is_absent() -> None:
    semantic = np.zeros((2, 4, 5), dtype=np.int32)
    semantic[0, 1:3, 2:4] = 220
    result = semantic_centroid_track(semantic, 220)
    assert np.array_equal(result[0], [2.5, 1.5])
    assert np.all(np.isnan(result[1]))


def test_source_events_expand_half_open_and_reject_overlap() -> None:
    source = {
        "event_windows": [
            {"event_id": "a", "start_frame": 1, "end_frame_exclusive": 3},
            {"event_id": "b", "start_frame": 4, "end_frame_exclusive": 5},
        ]
    }
    events, active = event_overlay_state(source, 5)
    assert events == (None, "a", "a", None, "b")
    assert active == (False, True, True, False, True)

    source["event_windows"].append(
        {"event_id": "c", "start_frame": 2, "end_frame_exclusive": 4}
    )
    with pytest.raises(M51DeliveryError, match="overlapping"):
        event_overlay_state(source, 5)


def test_overlay_tracks_bind_actual_links_semantics_events_and_point_gates() -> None:
    source = load_source_manifest(SOURCE_MANIFEST)
    route = load_json(ROUTE_MANIFEST)
    anchors = np.zeros((270, 3, 3), dtype=np.float64)
    anchors[:, 1, :] = (1.0, 1.5, -2.0)
    anchors[:, 2, :] = (-1.0, 0.5, -2.0)
    semantic = np.zeros((270, 4, 5), dtype=np.int32)
    semantic[:, 1, 1] = 220
    semantic[:, 2, 3] = 221

    tracks = build_legacy_overlay_tracks(
        source,
        route,
        anchor_positions_m=anchors,
        semantic_frames=semantic,
    )
    assert tuple(track.source_id for track in tracks) == ("source0", "source1")
    assert tracks[0].current_event_by_frame[75] == "event_human_speech_001"
    assert tracks[1].current_event_by_frame[90] == "event_beagle_bark_001"
    assert "steady_walk" in tracks[0].true_flags
    assert np.array_equal(tracks[0].main_marker_xy[0], [1.0, 1.0])
    assert np.array_equal(tracks[1].main_marker_xy[0], [3.0, 2.0])
    assert float(np.min(tracks[0].center_clearance_m)) == pytest.approx(0.2)


def test_executable_events_preserve_manifest_slice_and_add_gain_fade() -> None:
    source = load_source_manifest(SOURCE_MANIFEST)
    events = executable_event_mappings(
        source,
        gain_by_source={"source0": 0.18, "source1": 0.12},
        fade_samples=80,
    )
    human = next(event for event in events if event["source_id"] == "source0")
    dog = next(event for event in events if event["source_id"] == "source1")
    assert human["dry_clip_start_sample"] == 0
    assert human["dry_clip_end_sample_exclusive"] == 152156
    assert human["linear_gain"] == 0.18
    assert dog["dry_clip_start_sample"] == 3200
    assert dog["dry_clip_end_sample_exclusive"] == 8000
    assert dog["fade_samples"] == 80

    with pytest.raises(M51DeliveryError, match="finite and positive"):
        executable_event_mappings(
            source,
            gain_by_source={"source0": 0.0, "source1": 0.12},
            fade_samples=80,
        )


def test_replica_binding_requires_explicit_anchor_and_uses_route_semantics() -> None:
    source = load_source_manifest(SOURCE_MANIFEST)
    route = {
        "route_id": "route",
        "room_id": "room",
        "routes": {"human0": {}, "dog0": {}},
        "semantic_ids": {"human0": 810, "dog0": 811},
    }
    actors = [
        {
            "actor_id": "human0",
            "actor_class": "human",
            "emitter_link": "Bip01 MJaw",
            "emitter_anchor_id": "human0.mouth_emitter",
            "emitter_anchor_index": 1,
            "semantic_id": 810,
        },
        {
            "actor_id": "dog0",
            "actor_class": "dog",
            "emitter_link": "beagle Xtra Mouth",
            "emitter_anchor_id": "dog0.mouth_emitter",
            "emitter_anchor_index": 2,
            "semantic_id": 811,
        },
    ]
    record = source_actor_binding_record(
        source, route, {"actors": actors}, room_family="replicacad"
    )
    assert record["bindings"]["source0"]["actor_id"] == "human0"
    assert record["bindings"]["source1"]["semantic_id"] == 811

    missing_anchor = [dict(actor) for actor in actors]
    missing_anchor[1].pop("emitter_anchor_index")
    with pytest.raises(M51DeliveryError, match="lacks emitter_anchor_index"):
        source_actor_binding_record(
            source,
            route,
            {"actors": missing_anchor},
            room_family="replicacad",
        )


def test_each_source_event_requires_nonzero_dry_bus_and_binaural_stem() -> None:
    source = {
        "sources": [
            {
                "source_id": "source0",
                "event_windows": [
                    {"event_id": "speech", "start_frame": 1, "end_frame_exclusive": 3}
                ],
            },
            {
                "source_id": "source1",
                "event_windows": [
                    {"event_id": "bark", "start_frame": 1, "end_frame_exclusive": 3}
                ],
            },
        ]
    }
    clip = DryAudioClipSpec.from_values(
        frame_count=4,
        fps_numerator=2,
        sample_rate_hz=4,
    )
    dry = {source_id: np.ones(8) for source_id in ("source0", "source1")}
    stems = {
        source_id: np.ones((2, 8)) for source_id in ("source0", "source1")
    }
    result = verify_source_event_audio_activity(
        source, clip, dry_buses=dry, binaural_stems=stems
    )
    assert result["source0"]["events"][0]["dry_peak_absolute"] == 1.0

    stems["source1"][:, 2:6] = 0.0
    with pytest.raises(M51DeliveryError, match="binaural stem is silent"):
        verify_source_event_audio_activity(
            source, clip, dry_buses=dry, binaural_stems=stems
        )


def test_acoustic_binding_rejects_wrong_source_room_or_capture() -> None:
    expected = {
        "expected_room_id": "replicacad_apt_0",
        "expected_route_id": "route-v2",
        "expected_request_id": "request-v1",
        "expected_capture_evidence_sha256": "a" * 64,
        "expected_capture_content_sha256": "b" * 64,
        "expected_source_manifest_sha256": "c" * 64,
        "expected_source_binding_sha256": "d" * 64,
    }
    evidence = {
        "source_room": {"room_id": "replicacad_apt_0"},
        "capture_binding": {
            "room_id": expected["expected_room_id"],
            "route_id": expected["expected_route_id"],
            "request_id": expected["expected_request_id"],
            "capture_evidence_sha256": expected[
                "expected_capture_evidence_sha256"
            ],
            "capture_content_sha256": expected[
                "expected_capture_content_sha256"
            ],
            "source_manifest_sha256": expected[
                "expected_source_manifest_sha256"
            ],
            "source_actor_binding_content_sha256": expected[
                "expected_source_binding_sha256"
            ],
        },
    }
    validate_retained_acoustic_binding(evidence, **expected)

    wrong_room = dict(evidence)
    wrong_room["source_room"] = {"room_id": "habitat_mp3d_example_17DRP5sb8fy"}
    with pytest.raises(M51DeliveryError, match="source_room"):
        validate_retained_acoustic_binding(wrong_room, **expected)

    wrong_capture = {
        **evidence,
        "capture_binding": {
            **evidence["capture_binding"],
            "capture_evidence_sha256": "e" * 64,
        },
    }
    with pytest.raises(M51DeliveryError, match="capture/route/source"):
        validate_retained_acoustic_binding(wrong_capture, **expected)


def test_binaural_frame_diagnostics_and_actual_trajectory_are_hash_bound() -> None:
    clip = DryAudioClipSpec.from_values(
        frame_count=2,
        fps_numerator=2,
        sample_rate_hz=4,
    )
    mixture = np.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    labels, records = binaural_frame_diagnostics(mixture, clip, maximum_itd_samples=1)
    assert labels[0].startswith("ILD=+0.00dB ITD_xcorr=+0.0us")
    assert labels[1] == "silent"
    assert records[0]["itd_xcorr_samples"] == 0
    assert records[1]["active"] is False

    anchors = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    first = actual_emitter_trajectory_record(
        anchors,
        capture_evidence_sha256="a" * 64,
    )
    second = actual_emitter_trajectory_record(
        anchors.copy(),
        capture_evidence_sha256="a" * 64,
    )
    assert first == second
    assert first["sources"]["source0"]["positions_m"] == anchors[:, 1, :].tolist()
    assert len(first["record_content_sha256"]) == 64
