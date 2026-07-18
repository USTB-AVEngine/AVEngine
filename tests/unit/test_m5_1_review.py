from __future__ import annotations

import numpy as np
import pytest

from avengine.m5_1.review import (
    M51ReviewError,
    SourceOverlayTrack,
    _source_geometry,
    compose_annotated_frames,
    encode_annotated_review,
)


def _track(source_id: str, color: tuple[int, int, int]) -> SourceOverlayTrack:
    return SourceOverlayTrack(
        source_id=source_id,
        label=source_id.upper(),
        asset_class="human" if source_id == "human0" else "animal",
        sound_class="speech" if source_id == "human0" else "bark",
        color_rgb=color,
        positions_m=np.asarray([[0.0, 1.0, -2.0], [0.1, 1.0, -2.0]]),
        current_event_by_frame=(f"{source_id}_event", None),
        active_by_frame=(True, False),
        true_flags=("steady_walk",),
        center_clearance_m=np.asarray([0.2, 0.15]),
        main_marker_xy=np.asarray([[12.0, 8.0], [13.0, 8.0]]),
    )


def test_compose_annotated_frames_is_deterministic_and_exact_shape() -> None:
    main = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    topdown = np.full((2, 48, 64, 3), 128, dtype=np.uint8)
    tracks = (_track("human0", (42, 210, 220)), _track("dog0", (250, 120, 70)))
    kwargs = dict(
        main_rgb=main,
        topdown_rgb=topdown,
        tracks=tracks,
        clip_id="legacy_apartment_compare_v1",
        room_id="legacy_apartment",
        listener_position_m=(0.0, 1.2, 0.0),
        listener_yaw_deg=55.0,
        aggregate_true_flags=("steady_walk", "sources_pass_each_other"),
        audio_diagnostic_by_frame=(
            "ILD=+1.00dB ITD_xcorr=-62.5us",
            "silent",
        ),
        center_gate_pass=True,
    )
    first = compose_annotated_frames(**kwargs)
    second = compose_annotated_frames(**kwargs)
    assert first.shape == (2, 480, 1280, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.count_nonzero(first) > 0


def test_source_geometry_uses_habitat_forward_and_positive_right_azimuth() -> None:
    listener = (0.0, 0.0, 0.0)

    # At +90-degree Habitat yaw, head-forward is world -X and right ear is -Z.
    assert _source_geometry((-2.0, 0.0, 0.0), listener, 90.0) == (2.0, 0.0)
    assert _source_geometry((0.0, 0.0, -2.0), listener, 90.0) == (2.0, 90.0)
    assert _source_geometry((0.0, 0.0, 2.0), listener, 90.0) == (2.0, -90.0)


def test_compose_rejects_event_length_mismatch() -> None:
    track = _track("human0", (42, 210, 220))
    bad = SourceOverlayTrack(
        **{
            **track.__dict__,
            "current_event_by_frame": ("only_one",),
        }
    )
    with pytest.raises(M51ReviewError, match="current_event_by_frame"):
        compose_annotated_frames(
            main_rgb=np.zeros((2, 24, 32, 3), dtype=np.uint8),
            topdown_rgb=np.zeros((2, 24, 32, 3), dtype=np.uint8),
            tracks=(bad,),
            clip_id="clip",
            room_id="room",
            listener_position_m=(0.0, 1.0, 0.0),
            listener_yaw_deg=0.0,
            center_gate_pass=True,
        )


def test_compose_rejects_audio_diagnostic_length_mismatch() -> None:
    with pytest.raises(M51ReviewError, match="audio_diagnostic_by_frame"):
        compose_annotated_frames(
            main_rgb=np.zeros((2, 24, 32, 3), dtype=np.uint8),
            topdown_rgb=np.zeros((2, 24, 32, 3), dtype=np.uint8),
            tracks=(_track("human0", (42, 210, 220)),),
            clip_id="clip",
            room_id="room",
            listener_position_m=(0.0, 1.0, 0.0),
            listener_yaw_deg=0.0,
            audio_diagnostic_by_frame=("only-one",),
            center_gate_pass=True,
        )


def test_encode_review_round_trips_frame_count(tmp_path) -> None:
    frames = np.zeros((3, 480, 1280, 3), dtype=np.uint8)
    frames[1, 100:140, 100:140] = 255
    output = tmp_path / "review.mp4"
    report = encode_annotated_review(frames, output, fps=15)
    assert report["frame_count"] == 3
    assert report["duration_seconds"] == pytest.approx(0.2)
    assert report["topdown_is_qa_only"] is True
    assert output.is_file()
