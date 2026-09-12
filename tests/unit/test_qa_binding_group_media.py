from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from avengine.qa import binding_groups  # noqa: E402
from avengine.qa.binding_groups import BindingGroupError  # noqa: E402


HAVE_MEDIA_TOOLS = all(
    shutil.which(tool) is not None for tool in ("ffmpeg", "ffprobe")
)
pytestmark = [
    pytest.mark.media_readback,
    pytest.mark.skipif(
        not HAVE_MEDIA_TOOLS,
        reason="ffmpeg and ffprobe are needed for media clock fixtures",
    ),
]


def _make_video(
    path: Path,
    *,
    frames: int = 6,
    fps: int = 15,
    metadata: bool = False,
) -> Path:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s=8x8:r={fps}",
        "-frames:v",
        str(frames),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
    ]
    if metadata:
        command.extend(["-metadata", "comment=synthetic fixture metadata"])
    command.extend(["-map_metadata", "0" if metadata else "-1", "-map_chapters", "-1", str(path)])
    subprocess.run(command, check=True, capture_output=True)
    return path


def _make_video_with_audio(path: Path) -> Path:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=8x8:r=15",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=0.4",
        "-frames:v",
        "6",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        "0.4",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return path


def _make_audio(path: Path, *, frames: int = 6400, sample_rate: int = 16000) -> Path:
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    samples = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 440.0 * time),
            0.08 * np.sin(2.0 * np.pi * 660.0 * time),
        )
    ).astype(np.float32)
    sf.write(path, samples, sample_rate, subtype="FLOAT")
    return path


def _facts(
    *,
    frame_count: int = 6,
    frame_rate: float = 15.0,
    sample_rate: int = 16000,
    sample_count: int = 6400,
    channel_count: int = 2,
) -> dict:
    return {
        "time": {
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate,
            "sample_rate_hz": sample_rate,
            "sample_count": sample_count,
            "duration_seconds": sample_count / sample_rate,
        },
        "audio": {"channel_count": channel_count},
        "visibility_meta": {"resolution_hw": [8, 8]},
    }


def test_validate_clock_accepts_exact_synthetic_fixture_clock(tmp_path: Path):
    video = _make_video(tmp_path / "native.mp4")
    audio = _make_audio(tmp_path / "native.wav")
    observed = binding_groups._validate_clock(
        _facts(), video, audio, {}
    )
    assert observed["status"] == "pass"
    assert observed["video"]["frame_count"] == 6
    assert observed["video"]["frame_rate_hz"] == pytest.approx(15.0)
    assert observed["audio"] == {
        "sample_rate_hz": 16000,
        "sample_count": 6400,
        "channel_count": 2,
    }


def test_export_prefix_includes_query_frame_and_audio_stops_at_query(tmp_path: Path):
    source_video = _make_video(tmp_path / "native.mp4")
    source_audio = _make_audio(tmp_path / "native.wav")
    output = tmp_path / "export"
    (output / "media").mkdir(parents=True)
    cache: dict = {}
    media = binding_groups._export_media(
        source_video,
        source_audio,
        output,
        "prefix",
        cutoff=0.2,
        frame_rate=15.0,
        cache=cache,
    )
    observed = binding_groups._validate_clock(
        _facts(),
        output / media["video_path"],
        output / media["audio_path"],
        cache,
        cutoff=0.2,
    )
    # round(0.2 * 15) + 1 includes frame index 3, the query frame.
    assert observed["video"]["frame_count"] == 4
    assert observed["audio"]["sample_count"] == 3200
    assert observed["observation_cutoff_s"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("audio_frames", "audio_rate"),
    ((6400, 8000), (6300, 16000)),
)
def test_validate_clock_rejects_wrong_audio_rate_or_sample_count(
    tmp_path: Path,
    audio_frames: int,
    audio_rate: int,
):
    video = _make_video(tmp_path / "native.mp4")
    audio = _make_audio(
        tmp_path / f"wrong_{audio_frames}_{audio_rate}.wav",
        frames=audio_frames,
        sample_rate=audio_rate,
    )
    with pytest.raises(BindingGroupError, match="audio sample clock"):
        binding_groups._validate_clock(_facts(), video, audio, {})


def test_validate_clock_rejects_truncated_video(tmp_path: Path):
    video = _make_video(tmp_path / "truncated.mp4", frames=5)
    audio = _make_audio(tmp_path / "native.wav")
    with pytest.raises(BindingGroupError, match="decoded video clock"):
        binding_groups._validate_clock(_facts(), video, audio, {})


def test_video_rejects_hidden_audio_and_export_strips_metadata(tmp_path: Path):
    with_audio = _make_video_with_audio(tmp_path / "with_audio.mp4")
    with pytest.raises(BindingGroupError, match="hidden audio"):
        binding_groups._video_info(with_audio)

    source_video = _make_video(tmp_path / "tagged.mp4", metadata=True)
    source_audio = _make_audio(tmp_path / "native.wav")
    output = tmp_path / "export"
    (output / "media").mkdir(parents=True)
    media = binding_groups._export_media(
        source_video,
        source_audio,
        output,
        "metadata_free",
        cutoff=None,
        frame_rate=15.0,
        cache={},
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-of",
                "json",
                str(output / media["video_path"]),
            ],
            text=True,
        )
    )
    tags = probe.get("format", {}).get("tags", {})
    assert "comment" not in {str(key).casefold() for key in tags}


def _lossless_video(path: Path, *, preset: str, frames: int = 6) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=8x8:r=15",
         "-frames:v", str(frames), "-map", "0:v:0", "-c:v", "libx264",
         "-preset", preset, "-crf", "0", "-pix_fmt", "yuv420p", "-an",
         "-map_metadata", "-1", "-map_chapters", "-1", str(path)],
        check=True, capture_output=True)
    return path


def test_media_equality_reads_decoded_content_not_container_bytes(tmp_path: Path):
    left = _lossless_video(tmp_path / "left.mp4", preset="veryfast")
    right = _lossless_video(tmp_path / "right.mp4", preset="veryslow")
    assert left.read_bytes() != right.read_bytes()
    assert binding_groups._same_video(left, right) is True
    shorter = _lossless_video(tmp_path / "shorter.mp4", preset="veryfast", frames=5)
    assert binding_groups._same_video(left, shorter) is False

    audio = _make_audio(tmp_path / "a.wav")
    copy = _make_audio(tmp_path / "b.wav")
    assert binding_groups._same_audio(audio, copy) is True
    samples, rate = sf.read(copy, dtype="float32", always_2d=True)
    samples[1234, 0] += 1e-3
    sf.write(copy, samples, rate, subtype="FLOAT")
    assert binding_groups._same_audio(audio, copy) is False


def test_two_crops_of_one_recording_do_not_deliver_the_same_audio(tmp_path: Path):
    """Why the crop identity of a shared-audio pair has to be checked.

    Cropping a long recording is allowed, but two members that each pick their
    own window deliver different samples, so calling that pair shared audio
    would be a claim about an input nobody held fixed.
    """
    rate = 16000
    time = np.arange(4 * rate, dtype=np.float32) / float(rate)
    recording = np.column_stack(
        (0.2 * np.sin(2.0 * np.pi * 220.0 * time),
         0.2 * np.sin(2.0 * np.pi * 330.0 * time))).astype(np.float32)
    first, second = tmp_path / "crop_a.wav", tmp_path / "crop_b.wav"
    sf.write(first, recording[: 2 * rate], rate, subtype="FLOAT")
    sf.write(second, recording[rate: 3 * rate], rate, subtype="FLOAT")
    assert binding_groups._same_audio(first, second) is False
    same_window = tmp_path / "crop_a_again.wav"
    sf.write(same_window, recording[: 2 * rate], rate, subtype="FLOAT")
    assert binding_groups._same_audio(first, same_window) is True
