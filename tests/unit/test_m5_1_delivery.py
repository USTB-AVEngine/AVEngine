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
