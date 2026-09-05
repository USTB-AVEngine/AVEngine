"""Machine audition: the checks catch what they claim to catch.

Every negative test here plants exactly one defect in an otherwise sound
episode, because a gate is only trusted once it has been watched firing.
The channel-swap test is the one that earned its place: the per-frame ILD
gate this module almost shipped could not tell a swap from noise on the
real calibration episode, so the whole-clip gate below has to.

The PNG decoder is additionally exercised against real rendered frames in
calibration (a frame diffed against itself measured exactly zero); here the
fixtures use filter-0 images the test writes itself, keeping the suite free
of image dependencies.
"""

from __future__ import annotations

import json
import struct
import wave
import zlib
from pathlib import Path

import pytest

from avengine.review.episode_audition import (
    AuditionError,
    audit_episode,
    changed_fraction,
    read_png_first_channel,
    read_wav_levels,
    write_audition,
)


def _write_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    def chunk(chunk_id: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + chunk_id
            + body
            + struct.pack(">I", zlib.crc32(chunk_id + body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _frame_rows(width: int, height: int, block_x: int) -> list[bytes]:
    """Mid-grey frame with a bright 4x4 block at block_x (the moving source)."""

    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            bright = 4 <= y < 8 and block_x <= x < block_x + 4
            row += bytes((240, 240, 240) if bright else (100, 100, 100))
        rows.append(bytes(row))
    return rows


def _write_wav(path: Path, channels: list[list[float]], rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(len(channels))
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for index in range(len(channels[0])):
            for channel in channels:
                value = max(-1.0, min(1.0, channel[index]))
                frames += struct.pack("<h", int(value * 32767))
        handle.writeframes(bytes(frames))


FRAMES = 8
FRAME_RATE = 4.0
RATE = 1000
POSITION = [1.0, 1.5, 2.0]
AIM = [0.0, 0.0, -1.0]


def _episode(
    tmp_path: Path,
    *,
    left_amp: float = 0.3,
    right_amp: float = 0.15,
    report_median_ild: float = 6.0,
    within: int = FRAMES,
    foa_candidate: int = 0,
    black_frames: bool = False,
) -> Path:
    root = tmp_path / "render"
    (root / "video").mkdir(parents=True)
    (root / "audio_foa").mkdir()
    (root / "audio_binaural").mkdir()
    (root / "listener_pose").mkdir()

    for index in range(FRAMES):
        rows = (
            [bytes(24 * 3) for _ in range(16)]
            if black_frames
            else _frame_rows(24, 16, block_x=2 * index)
        )
        _write_png(root / "video" / f"frame_{index:04d}.png", 24, 16, rows)

    samples = int(RATE * (FRAMES / FRAME_RATE + 0.2))
    tone = [1.0 if (i // 8) % 2 == 0 else -1.0 for i in range(samples)]
    _write_wav(
        root / "audio_binaural" / "moving_source.binaural.wav",
        [[left_amp * v for v in tone], [right_amp * v for v in tone]],
        RATE,
    )
    _write_wav(
        root / "audio_foa" / "moving_source.ambisonic.wav",
        [[0.2 * v for v in tone] for _ in range(4)],
        RATE,
    )

    (root / "listener_pose" / "pose.json").write_text(
        json.dumps(
            {
                "accepted_index": 0,
                "candidates": [{"position_m": POSITION, "aim_world": AIM}],
            }
        )
    )
    (root / "audio_foa" / "render_report.json").write_text(
        json.dumps(
            {
                "frames_rendered": FRAMES,
                "frames_within_tolerance": within,
                "tolerance_deg": 5.0,
                "direction_error_deg": {"median": 0.1, "maximum": 2.0},
                "listener_pose_candidate": foa_candidate,
                "listener_m": POSITION,
                "bank": "bank.json",
            }
        )
    )
    (root / "audio_binaural" / "render_report.json").write_text(
        json.dumps(
            {
                "cardinal_probes": {
                    "left": {"difference_db": 6.1},
                    "right": {"difference_db": -4.4},
                },
                "cardinal_margin_db": 3.0,
                "interaural_level_difference_db": {"median": report_median_ild},
                "listener_m": POSITION,
                "head_aim_world": AIM,
                "bank": "bank.json",
            }
        )
    )
    (root / "video" / "video_manifest.json").write_text(
        json.dumps(
            {
                "listener_m": POSITION,
                "camera_aim_world": AIM,
                "frames": FRAMES,
                "frame_rate_hz": FRAME_RATE,
            }
        )
    )
    (root / "episode_binaural.mp4").write_bytes(b"not probed in unit tests")
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "bank": "bank.json",
                "frame_rate_hz": FRAME_RATE,
                "listener_pose": str(root / "listener_pose" / "pose.json"),
                "foa_report": str(root / "audio_foa" / "render_report.json"),
                "foa_wav": str(root / "audio_foa" / "moving_source.ambisonic.wav"),
                "binaural_wav": str(
                    root / "audio_binaural" / "moving_source.binaural.wav"
                ),
                "video_manifest": str(root / "video" / "video_manifest.json"),
                "deliverable_mp4": str(root / "episode_binaural.mp4"),
            }
        )
    )
    return root


def _by_name(document: dict) -> dict[str, dict]:
    return {check["name"]: check for check in document["checks"]}


def test_sound_episode_passes_with_reasons_in_words(tmp_path: Path) -> None:
    document = audit_episode(_episode(tmp_path), ffprobe=None)
    checks = _by_name(document)
    assert document["verdict"] == "pass"
    assert checks["chain_identity"]["status"] == "pass"
    assert checks["foa_direction"]["status"] == "pass"
    assert checks["binaural_cardinal"]["status"] == "pass"
    assert checks["binaural_wav_levels"]["status"] == "pass"
    assert checks["binaural_wav_matches_report"]["status"] == "pass"
    assert checks["video_frames"]["status"] == "pass"
    # without ffprobe the mux check must say it was skipped, not pass silently
    assert checks["deliverable_mux"]["status"] == "info"
    assert "达标" in document["summary_zh"]
    for check in document["checks"]:
        assert check["reason_zh"]


def test_swapped_channels_fail_the_wav_report_agreement(tmp_path: Path) -> None:
    """The gate that survived calibration: a swap flips the whole-clip sign."""

    root = _episode(tmp_path, left_amp=0.15, right_amp=0.3, report_median_ild=6.0)
    document = audit_episode(root, ffprobe=None)
    assert document["verdict"] == "fail"
    check = _by_name(document)["binaural_wav_matches_report"]
    assert check["status"] == "fail"
    assert "调换" in check["reason_zh"]


def test_small_reported_ild_is_recorded_not_judged(tmp_path: Path) -> None:
    root = _episode(tmp_path, left_amp=0.2, right_amp=0.2, report_median_ild=0.3)
    document = audit_episode(root, ffprobe=None)
    check = _by_name(document)["binaural_wav_matches_report"]
    assert check["status"] == "info"
    assert document["verdict"] == "pass"


def test_direction_below_the_floor_fails(tmp_path: Path) -> None:
    document = audit_episode(_episode(tmp_path, within=3), ffprobe=None)
    assert document["verdict"] == "fail"
    assert _by_name(document)["foa_direction"]["status"] == "fail"


def test_black_frames_fail(tmp_path: Path) -> None:
    document = audit_episode(_episode(tmp_path, black_frames=True), ffprobe=None)
    assert document["verdict"] == "fail"
    assert _by_name(document)["video_frames"]["status"] == "fail"


def test_identity_mismatch_fails(tmp_path: Path) -> None:
    """FOA rendered candidate 1 while the pose accepted 0: two different ears."""

    document = audit_episode(_episode(tmp_path, foa_candidate=1), ffprobe=None)
    assert document["verdict"] == "fail"
    assert _by_name(document)["chain_identity"]["status"] == "fail"


def test_missing_artifact_is_a_failed_check_not_a_crash(tmp_path: Path) -> None:
    root = _episode(tmp_path)
    (root / "audio_binaural" / "moving_source.binaural.wav").unlink()
    document = audit_episode(root, ffprobe=None)
    assert document["verdict"] == "fail"
    artifacts = [c for c in document["checks"] if c["name"] == "artifacts"]
    assert artifacts and "binaural_wav" in artifacts[0]["reason_zh"]


def test_write_audition_refuses_to_clobber(tmp_path: Path) -> None:
    root = _episode(tmp_path)
    written = write_audition(root, ffprobe=None)
    assert (root / "machine_audition.json").is_file()
    assert written["verdict"] == "pass"
    with pytest.raises(AuditionError, match="already exists"):
        write_audition(root, ffprobe=None)
    assert write_audition(root, ffprobe=None, refresh=True)["verdict"] == "pass"


def test_wav_and_png_readers_measure_what_was_written(tmp_path: Path) -> None:
    wav = tmp_path / "probe.wav"
    _write_wav(wav, [[0.5, -0.5, 0.5, -0.5], [0.25, -0.25, 0.25, -0.25]], 4)
    levels = read_wav_levels(wav)
    assert levels["channel_count"] == 2
    assert levels["duration_s"] == 1.0
    assert abs(levels["peak"] - 0.5) < 1e-3
    assert abs((levels["rms_dbfs"][0] - levels["rms_dbfs"][1]) - 6.02) < 0.05

    png = tmp_path / "probe.png"
    _write_png(png, 24, 16, _frame_rows(24, 16, block_x=0))
    width, height, plane = read_png_first_channel(png)
    assert (width, height) == (24, 16)
    assert plane[0] == 100 and plane[4 * 24 + 1] == 240
    _write_png(tmp_path / "probe2.png", 24, 16, _frame_rows(24, 16, block_x=8))
    _, _, moved = read_png_first_channel(tmp_path / "probe2.png")
    assert changed_fraction(plane, plane) == 0.0
    assert changed_fraction(plane, moved) > 0.01


def _add_explicit_clock(root: Path, *, alias_full_tail: bool = False) -> None:
    aligned_samples = int(RATE * (FRAMES / FRAME_RATE))
    tone = [1.0 if (i // 8) % 2 == 0 else -1.0 for i in range(aligned_samples)]
    foa_aligned = root / "audio_foa/moving_source.ambisonic.aligned.wav"
    binaural_aligned = root / "audio_binaural/moving_source.binaural.aligned.wav"
    _write_wav(foa_aligned, [[0.2 * value for value in tone] for _ in range(4)], RATE)
    _write_wav(
        binaural_aligned,
        [[0.3 * value for value in tone], [0.15 * value for value in tone]],
        RATE,
    )
    clock = {
        "frame_count": FRAMES,
        "frame_rate_hz": FRAME_RATE,
        "sample_rate_hz": RATE,
        "clip_seconds": FRAMES / FRAME_RATE,
        "sample_count": aligned_samples,
        "compatibility": "configured",
    }
    manifest_path = root / "video/video_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "frame_count": FRAMES,
        "sample_rate_hz": RATE,
        "sample_count": aligned_samples,
        "clock": clock,
    })
    manifest_path.write_text(json.dumps(manifest))
    for report_path, full_path in (
        (root / "audio_foa/render_report.json", root / "audio_foa/moving_source.ambisonic.wav"),
        (root / "audio_binaural/render_report.json", root / "audio_binaural/moving_source.binaural.wav"),
    ):
        report = json.loads(report_path.read_text())
        report["full_tail_sample_count"] = read_wav_levels(full_path)["sample_count"]
        report_path.write_text(json.dumps(report))
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({
        "clock": clock,
        "frame_count": FRAMES,
        "frame_rate_hz": FRAME_RATE,
        "sample_rate_hz": RATE,
        "sample_count": aligned_samples,
        "clip_seconds": FRAMES / FRAME_RATE,
        "foa_wav_full_tail": str(
            foa_aligned if alias_full_tail else root / "audio_foa/moving_source.ambisonic.wav"
        ),
        "binaural_wav_full_tail": str(
            binaural_aligned if alias_full_tail else root / "audio_binaural/moving_source.binaural.wav"
        ),
        "foa_wav_aligned": str(foa_aligned),
        "binaural_wav_aligned": str(binaural_aligned),
    })
    receipt_path.write_text(json.dumps(receipt))


def test_explicit_clock_proves_separate_full_tail_and_aligned_files(tmp_path: Path) -> None:
    root = _episode(tmp_path)
    _add_explicit_clock(root)
    document = audit_episode(root, ffprobe=None)
    checks = _by_name(document)
    assert checks["episode_clock"]["status"] == "pass"
    assert checks["aligned_audio_window"]["status"] == "pass"
    assert checks["full_tail_audio"]["status"] == "pass"
    assert document["verdict"] == "pass"


def test_full_tail_aliasing_aligned_file_is_rejected(tmp_path: Path) -> None:
    root = _episode(tmp_path)
    _add_explicit_clock(root, alias_full_tail=True)
    document = audit_episode(root, ffprobe=None)
    assert _by_name(document)["full_tail_audio"]["status"] == "fail"
    assert document["verdict"] == "fail"
