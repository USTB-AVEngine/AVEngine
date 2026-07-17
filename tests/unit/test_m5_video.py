from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

import avengine.m5.video as video
from avengine.m4.audio import write_float32_wav
from avengine.m5.video import (
    AUDIO_SAMPLE_COUNT,
    FRAME_COUNT,
    M5VideoError,
    aac_decode_diagnostics,
    compose_main_topdown_frames,
    compose_main_topdown_panel,
    encode_h264_base_video,
    encode_h264_qa_base_video,
    mux_binaural_wav,
    mux_qa_binaural_wav,
    probe_episode_video,
    probe_qa_review_video,
    video_packet_sha256,
)


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
MEDIA_TOOLS_AVAILABLE = FFMPEG_AVAILABLE and shutil.which("ffprobe") is not None


def _frames() -> np.ndarray:
    y, x = np.indices((video.FRAME_HEIGHT, video.FRAME_WIDTH))
    result = np.empty(
        (FRAME_COUNT, video.FRAME_HEIGHT, video.FRAME_WIDTH, 3),
        dtype=np.uint8,
    )
    for index in range(FRAME_COUNT):
        result[index, :, :, 0] = (x + index * 3) % 256
        result[index, :, :, 1] = (2 * y + index * 5) % 256
        result[index, :, :, 2] = (x // 2 + y // 2 + index * 7) % 256
        left = 4 + (index * 4) % 280
        result[index, 88:120, left : left + 32] = (245, 230, 40)
    return result


def _binaural_samples() -> np.ndarray:
    rng = np.random.default_rng(20260718)
    time = np.arange(AUDIO_SAMPLE_COUNT, dtype=np.float64) / 16_000.0
    left = (
        0.18 * np.sin(2.0 * np.pi * 437.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1193.0 * time)
        + 0.018 * rng.standard_normal(AUDIO_SAMPLE_COUNT)
    )
    right = (
        0.16 * np.sin(2.0 * np.pi * 683.0 * time + 0.31)
        + 0.045 * np.sin(2.0 * np.pi * 1549.0 * time)
        + 0.018 * rng.standard_normal(AUDIO_SAMPLE_COUNT)
    )
    fade = np.ones(AUDIO_SAMPLE_COUNT, dtype=np.float64)
    fade[:160] = np.linspace(0.0, 1.0, 160, endpoint=False)
    fade[-160:] = np.linspace(1.0, 0.0, 160, endpoint=False)
    return np.vstack((left * fade, right * fade))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mux_command_is_explicit_metadata_free_and_never_uses_shortest(
    tmp_path: Path,
) -> None:
    command = video._mux_command(
        ffmpeg="ffmpeg",
        base_video=tmp_path / "base.mp4",
        audio_wav=tmp_path / "audio.wav",
        destination=tmp_path / "muxed.mp4",
    )

    assert "-shortest" not in command
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-map_metadata") + 1] == "-1"
    assert command[command.index("-map_chapters") + 1] == "-1"
    map_indices = [index for index, item in enumerate(command) if item == "-map"]
    assert [command[index + 1] for index in map_indices] == ["0:v:0", "1:a:0"]


def test_main_topdown_compositor_is_exact_560x240_and_supports_overlays() -> None:
    main = np.full((240, 320, 3), (10, 20, 30), dtype=np.uint8)
    topdown = np.full((240, 240, 3), (40, 50, 60), dtype=np.uint8)
    main_before = main.copy()
    topdown_before = topdown.copy()

    plain = compose_main_topdown_panel(main, topdown)
    annotated = compose_main_topdown_panel(
        main,
        topdown,
        text="frame=17 actor=dog0",
        trajectory=[(8, 220), (80, 140), (180, 32)],
    )

    assert plain.shape == annotated.shape == (240, 560, 3)
    assert plain.dtype == np.uint8
    np.testing.assert_array_equal(plain[:, :320], main)
    np.testing.assert_array_equal(plain[:, 320:], topdown)
    assert not np.array_equal(annotated, plain)
    np.testing.assert_array_equal(main, main_before)
    np.testing.assert_array_equal(topdown, topdown_before)


def test_frame_panel_compositor_binds_per_frame_metadata() -> None:
    mains = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(2)]
    topdowns = [np.zeros((240, 240, 3), dtype=np.uint8) for _ in range(2)]

    panels = compose_main_topdown_frames(
        mains,
        topdowns,
        text_by_frame=["frame 0", "frame 1"],
        trajectory_by_frame=[[(4, 4)], [(4, 4), (20, 20)]],
    )

    assert len(panels) == 2
    assert all(panel.shape == (240, 560, 3) for panel in panels)
    assert not np.array_equal(panels[0], panels[1])
    with pytest.raises(M5VideoError, match="frame counts differ"):
        compose_main_topdown_frames(mains, topdowns[:1])


@pytest.mark.skipif(not MEDIA_TOOLS_AVAILABLE, reason="FFmpeg tools are unavailable")
def test_encoder_rejects_noncanonical_frame_count_and_cleans_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "short.mp4"
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(M5VideoError, match="exactly 75 frames"):
        encode_h264_base_video([frame], output)

    assert not output.exists()
    assert not list(tmp_path.glob(".short.*.mp4"))


@pytest.fixture(scope="module")
def encoded_media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if not MEDIA_TOOLS_AVAILABLE:
        pytest.skip("FFmpeg tools are unavailable")
    root = tmp_path_factory.mktemp("m5_video")
    frames = _frames()
    audio_a = _binaural_samples()
    audio_b = np.vstack((audio_a[1] * 0.75, -audio_a[0] * 0.75))
    wav_a = root / "episode_a.wav"
    wav_b = root / "episode_b.wav"
    write_float32_wav(wav_a, audio_a, 16_000)
    write_float32_wav(wav_b, audio_b, 16_000)

    base_a = root / "base_a.mp4"
    base_b = root / "base_b.mp4"
    base_a_report = encode_h264_base_video(frames, base_a)
    base_b_report = encode_h264_base_video(frames, base_b)

    episode_a = root / "episode_a.mp4"
    episode_b = root / "episode_b.mp4"
    episode_a_report = mux_binaural_wav(base_a, wav_a, episode_a)
    episode_b_report = mux_binaural_wav(base_a, wav_b, episode_b)

    topdown_frames = np.ascontiguousarray(frames[:, :, :240])
    qa_frames = compose_main_topdown_frames(
        frames,
        topdown_frames,
        text_by_frame=[f"frame={index:02d}" for index in range(FRAME_COUNT)],
        trajectory_by_frame=[
            [(8, 220), (16 + index * 2, 120)] for index in range(FRAME_COUNT)
        ],
    )
    qa_base = root / "qa_base.mp4"
    qa_episode = root / "qa_episode.mp4"
    qa_base_report = encode_h264_qa_base_video(qa_frames, qa_base)
    qa_episode_report = mux_qa_binaural_wav(qa_base, wav_a, qa_episode)
    return {
        "root": root,
        "wav_a": wav_a,
        "base_a": base_a,
        "base_b": base_b,
        "base_a_report": base_a_report,
        "base_b_report": base_b_report,
        "episode_a": episode_a,
        "episode_b": episode_b,
        "episode_a_report": episode_a_report,
        "episode_b_report": episode_b_report,
        "qa_base": qa_base,
        "qa_episode": qa_episode,
        "qa_base_report": qa_base_report,
        "qa_episode_report": qa_episode_report,
    }


def test_h264_base_encode_is_deterministic_and_exact(
    encoded_media: dict[str, Any],
) -> None:
    first = encoded_media["base_a_report"]
    second = encoded_media["base_b_report"]

    assert first["video"] == second["video"] == {
        "codec_name": "h264",
        "width": 320,
        "height": 240,
        "pixel_format": "yuv420p",
        "frame_count": 75,
        "frame_rate": "15/1",
        "first_pts": 0,
        "duration_ticks": 240_000,
        "duration_seconds": 5,
    }
    assert first["audio"] is None
    assert first["video_packet_hash"] == second["video_packet_hash"]
    assert _sha256(encoded_media["base_a"]) == _sha256(encoded_media["base_b"])


def test_mux_readback_is_exact_and_video_packets_are_copied_for_ab_pair(
    encoded_media: dict[str, Any],
) -> None:
    episode_a = encoded_media["episode_a"]
    episode_b = encoded_media["episode_b"]
    report = probe_episode_video(episode_a)
    base_hash = video_packet_sha256(encoded_media["base_a"])
    a_hash = video_packet_sha256(episode_a)
    b_hash = video_packet_sha256(episode_b)

    assert report["video"]["frame_count"] == 75
    assert report["video"]["frame_rate"] == "15/1"
    assert report["video"]["duration_seconds"] == 5
    assert report["audio"] == {
        "codec_name": "aac",
        "profile": "LC",
        "sample_rate_hz": 16_000,
        "channel_count": 2,
        "channel_layout": "stereo",
        "duration_seconds": 5,
        "start_seconds": 0,
    }
    assert base_hash == a_hash == b_hash
    assert encoded_media["episode_a_report"]["video_stream_copy_verified"] is True
    assert encoded_media["episode_b_report"]["video_stream_copy_verified"] is True
    assert _sha256(episode_a) != _sha256(episode_b)


def test_aac_full_decode_reports_count_alignment_quality_and_lr_diagnostic(
    encoded_media: dict[str, Any],
) -> None:
    report = aac_decode_diagnostics(
        encoded_media["episode_a"],
        encoded_media["wav_a"],
    )

    assert report["reference_sample_count"] == AUDIO_SAMPLE_COUNT
    assert report["decoded_sample_count"] >= AUDIO_SAMPLE_COUNT
    assert report["presentation_sample_count"] == AUDIO_SAMPLE_COUNT
    assert report["presentation_sample_count_matches"] is True
    assert report["decoded_padding_samples"] == (
        report["decoded_sample_count"] - AUDIO_SAMPLE_COUNT
    )
    assert report["decoded_shortfall_samples"] == 0
    assert report["lag_samples"] == 0
    assert report["aligned_sample_count"] == AUDIO_SAMPLE_COUNT
    assert report["minimum_correlation"] > 0.95
    assert report["minimum_snr_db"] > 15.0
    assert report["lr_normal_correlation"] > report["lr_swapped_correlation"] + 0.5
    assert report["lr_swap_suspected"] is False
    assert report["diagnostic_only"] is True


def test_composited_qa_review_encodes_muxes_and_reads_back_exactly(
    encoded_media: dict[str, Any],
) -> None:
    qa_base = encoded_media["qa_base"]
    qa_episode = encoded_media["qa_episode"]
    base_report = probe_qa_review_video(qa_base, require_audio=False)
    muxed_report = probe_qa_review_video(qa_episode)

    assert base_report["video"] == {
        "codec_name": "h264",
        "width": 560,
        "height": 240,
        "pixel_format": "yuv420p",
        "frame_count": 75,
        "frame_rate": "15/1",
        "first_pts": 0,
        "duration_ticks": 240_000,
        "duration_seconds": 5,
    }
    assert base_report["audio"] is None
    assert muxed_report["audio"]["sample_rate_hz"] == 16_000
    assert muxed_report["audio"]["channel_count"] == 2
    assert muxed_report["audio"]["duration_seconds"] == 5
    assert encoded_media["qa_episode_report"]["video_stream_copy_verified"] is True
    assert video_packet_sha256(qa_base) == video_packet_sha256(qa_episode)
    with pytest.raises(M5VideoError, match="320x240"):
        probe_episode_video(qa_episode)


def test_mux_rejects_non_stereo_authoritative_wav_before_publication(
    encoded_media: dict[str, Any],
) -> None:
    root = encoded_media["root"]
    mono = root / "mono.wav"
    write_float32_wav(mono, _binaural_samples()[:1], 16_000)
    output = root / "bad_audio.mp4"

    with pytest.raises(M5VideoError, match="16 kHz stereo"):
        mux_binaural_wav(encoded_media["base_a"], mono, output)

    assert not output.exists()
