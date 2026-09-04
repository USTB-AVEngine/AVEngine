"""Deterministic M5 episode-video assembly and independent readback.

The four-channel FOA WAVE remains the dataset-authority spatial signal.  This
module owns the ordinary review-video boundary: a requested-clock H.264
base MP4 and one explicitly mapped AAC presentation track. Formal wrappers retain the historical 75-frame, 5-second, 16 kHz defaults.
It never uses FFmpeg's ``-shortest`` policy and it verifies the published bytes
before returning them to a caller.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


FRAME_COUNT = 75
FRAME_RATE = 15
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
VIDEO_TIME_BASE_HZ = 48_000
VIDEO_DURATION_TICKS = 240_000

AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_SAMPLE_COUNT = 80_000
AUDIO_CHANNEL_COUNT = 2
AAC_CODEC_FRAME_SAMPLES = 1024

QA_PANEL_WIDTH = 560
QA_PANEL_HEIGHT = 240
TOPDOWN_WIDTH = 240


class M5VideoError(ValueError):
    """An input, media tool, or decoded output violates the M5 contract."""


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M5VideoError(f"{owner} must be a positive integer")
    return int(value)


def _positive_rate(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, Fraction))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise M5VideoError(f"{owner} must be a finite positive number")
    return float(value)


def _rate_fraction(value: Any, *, owner: str) -> Fraction:
    rate = _positive_rate(value, owner=owner)
    try:
        result = Fraction(str(rate))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise M5VideoError(f"{owner} is not a valid frame rate") from exc
    if result <= 0:
        raise M5VideoError(f"{owner} must be positive")
    return result


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _video_duration_tolerance(frame_rate_hz: float) -> float:
    del frame_rate_hz
    # Video PTS are written on the 48 kHz MP4 time base. Half a tick is
    # sufficient for quantization while frame count is checked separately.
    return 0.5 / VIDEO_TIME_BASE_HZ


def _audio_duration_tolerance(sample_rate_hz: int) -> float:
    # AAC packetization can expose one codec frame of padding in a container.
    # Decoded shortfall is checked independently.
    return AAC_CODEC_FRAME_SAMPLES / float(sample_rate_hz) + 0.5 / VIDEO_TIME_BASE_HZ


def _tool(value: str | Path, *, owner: str) -> str:
    executable = shutil.which(os.fspath(value))
    if executable is None:
        raise M5VideoError(f"{owner} executable is unavailable: {value}")
    return executable


def _media_path(value: str | Path, *, owner: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise M5VideoError(f"{owner} is missing or not a file: {path}")
    return path


def _new_mp4_path(value: str | Path, *, owner: str) -> Path:
    path = Path(value).resolve()
    if path.suffix.casefold() != ".mp4":
        raise M5VideoError(f"{owner} must use the .mp4 suffix")
    if os.path.lexists(path):
        raise M5VideoError(f"refusing to overwrite {owner}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _temporary_mp4(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _publish(temporary: Path, destination: Path, *, owner: str) -> None:
    try:
        # The files share a directory/filesystem.  Linking publishes atomically
        # and refuses a concurrent writer instead of replacing its output.
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise M5VideoError(f"{owner} appeared during encoding: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _run(
    command: Sequence[str],
    *,
    owner: str,
    timeout: float = 120.0,
    binary_stdout: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=not binary_stdout,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise M5VideoError(f"could not execute {owner}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise M5VideoError(
            f"{owner} returned {completed.returncode}: {str(stderr).strip()}"
        )
    return completed


def _ffprobe_json(
    path: Path,
    arguments: Sequence[str],
    *,
    ffprobe: str | Path,
    owner: str,
) -> Mapping[str, Any]:
    command = [
        _tool(ffprobe, owner="ffprobe"),
        "-v",
        "error",
        *arguments,
        "-of",
        "json",
        str(path),
    ]
    completed = _run(command, owner=owner, timeout=60.0)
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise M5VideoError(f"{owner} returned malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise M5VideoError(f"{owner} JSON root is not an object")
    return value


def _rgb_frame(value: Any, *, shape: tuple[int, int, int], owner: str) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.shape != shape:
        raise M5VideoError(
            f"{owner} must be uint8 RGB with shape {shape}, got "
            f"dtype={frame.dtype} shape={frame.shape}"
        )
    return np.ascontiguousarray(frame)


def _base_encode_command(
    *,
    ffmpeg: str,
    width: int,
    height: int,
    destination: Path,
    frame_rate_hz: float = FRAME_RATE,
) -> list[str]:
    rate = _rate_fraction(frame_rate_hz, owner="video frame rate")
    # x264 requires an integer GOP length. Input PTS still carry the rational rate.
    gop = max(1, int(round(float(rate))))
    rate_text = _fraction_text(rate)
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+bitexact",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        rate_text,
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-threads",
        "1",
        "-flags:v",
        "+bitexact",
        "-video_track_timescale",
        str(VIDEO_TIME_BASE_HZ),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _encode_h264_video_profile(
    frames: Iterable[Any],
    output_path: str | Path,
    *,
    width: int,
    height: int,
    profile_name: str,
    frame_count: int | None = FRAME_COUNT,
    frame_rate_hz: float = FRAME_RATE,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    owner = f"{profile_name} base video output"
    expected_count = (
        None if frame_count is None
        else _positive_int(frame_count, owner="expected video frame count")
    )
    rate = _positive_rate(frame_rate_hz, owner="expected video frame rate")
    destination = _new_mp4_path(output_path, owner=owner)
    executable = _tool(ffmpeg, owner="ffmpeg")
    temporary = _temporary_mp4(destination)
    command = _base_encode_command(
        ffmpeg=executable,
        width=width,
        height=height,
        destination=temporary,
        frame_rate_hz=rate,
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise M5VideoError(f"could not start FFmpeg H.264 encoder: {exc}") from exc

    count = 0
    try:
        assert process.stdin is not None
        for count, value in enumerate(frames, start=1):
            frame = _rgb_frame(
                value,
                shape=(height, width, 3),
                owner=f"frame[{count - 1}]",
            )
            process.stdin.write(frame.tobytes(order="C"))
        process.stdin.close()
        return_code = process.wait()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise M5VideoError(
                f"FFmpeg H.264 encoder returned {return_code}: {stderr}"
            )
        if expected_count is not None and count != expected_count:
            raise M5VideoError(
                f"{profile_name} base video requires exactly {expected_count} frames, "
                f"received {count}"
            )
        report = _probe_video_profile(
            temporary,
            expected_width=width,
            expected_height=height,
            profile_name=profile_name,
            require_audio=False,
            expected_frame_count=expected_count,
            expected_frame_rate_hz=rate,
            ffprobe=ffprobe,
        )
        packet_hash = video_packet_sha256(
            temporary, expected_frame_count=count, ffprobe=ffprobe)
        _publish(temporary, destination, owner=owner)
        report["path"] = str(destination)
        report["video_packet_hash"] = packet_hash
        return report
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)
        raise


def encode_h264_base_video(
    frames: Iterable[Any],
    output_path: str | Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Encode exactly 75 ``320x240`` RGB frames into a video-only base MP4.

    The input is streamed as raw RGB, avoiding PNG metadata and filesystem
    ordering.  x264 uses one thread, a fixed GOP and no B frames; metadata is
    stripped and the MP4 video track uses the authoritative 48 kHz time base.
    Existing outputs are never replaced.
    """

    return _encode_h264_video_profile(
        frames,
        output_path,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        profile_name="formal episode",
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )


def encode_h264_qa_base_video(
    frames: Iterable[Any],
    output_path: str | Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Encode exactly 75 ``560x240`` main+topdown QA frames as H.264 MP4."""

    return _encode_h264_video_profile(
        frames,
        output_path,
        width=QA_PANEL_WIDTH,
        height=QA_PANEL_HEIGHT,
        profile_name="QA review",
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )


def _mux_command(
    *,
    ffmpeg: str,
    base_video: Path,
    audio_wav: Path,
    destination: Path,
    sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ,
    channel_count: int = AUDIO_CHANNEL_COUNT,
) -> list[str]:
    """Build an explicit stream-mapped mux command without shortest truncation."""
    sample_rate = _positive_int(sample_rate_hz, owner="audio sample rate")
    channels = _positive_int(channel_count, owner="audio channel count")

    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+bitexact",
        "-i",
        str(base_video),
        "-i",
        str(audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        "96k",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-threads:a",
        "1",
        "-flags:a",
        "+bitexact",
        "-video_track_timescale",
        str(VIDEO_TIME_BASE_HZ),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _stream_duration(stream: Mapping[str, Any], *, owner: str) -> Fraction:
    time_base = stream.get("time_base")
    duration_ticks = stream.get("duration_ts")
    try:
        if isinstance(time_base, str) and duration_ticks is not None:
            return int(duration_ticks) * Fraction(time_base)
        return Fraction(str(stream["duration"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise M5VideoError(f"{owner} does not expose an exact duration") from exc


def _integer_field(value: Mapping[str, Any], field: str, *, owner: str) -> int:
    try:
        result = int(value[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise M5VideoError(f"{owner}.{field} is absent or not an integer") from exc
    return result


def _probe_video_profile(
    path: str | Path,
    *,
    expected_width: int,
    expected_height: int,
    profile_name: str,
    require_audio: bool = True,
    expected_frame_count: int | None = FRAME_COUNT,
    expected_frame_rate_hz: float | None = FRAME_RATE,
    expected_audio_sample_rate_hz: int | None = AUDIO_SAMPLE_RATE_HZ,
    expected_audio_sample_count: int | None = AUDIO_SAMPLE_COUNT,
    expected_audio_channel_count: int | None = AUDIO_CHANNEL_COUNT,
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    if expected_frame_count is not None:
        expected_frame_count = _positive_int(
            expected_frame_count, owner="expected video frame count")
    expected_rate_fraction = (
        None if expected_frame_rate_hz is None
        else _rate_fraction(expected_frame_rate_hz, owner="expected video frame rate")
    )
    expected_audio_rate = (
        None if expected_audio_sample_rate_hz is None
        else _positive_int(expected_audio_sample_rate_hz, owner="expected audio sample rate")
    )
    expected_audio_count = (
        None if expected_audio_sample_count is None
        else _positive_int(expected_audio_sample_count, owner="expected audio sample count")
    )
    expected_audio_channels = (
        None if expected_audio_channel_count is None
        else _positive_int(expected_audio_channel_count, owner="expected audio channel count")
    )

    media = _media_path(path, owner=f"{profile_name} video")
    value = _ffprobe_json(
        media,
        [
            "-count_frames",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,"
                "avg_frame_rate,r_frame_rate,time_base,start_pts,start_time,"
                "duration_ts,duration,nb_frames,nb_read_frames,sample_rate,"
                "channels,channel_layout:format=format_name,start_time,duration"
            ),
            "-show_streams",
            "-show_format",
        ],
        ffprobe=ffprobe,
        owner=f"FFprobe {profile_name} readback",
    )
    streams = value.get("streams")
    format_value = value.get("format")
    if not isinstance(streams, list) or not all(
        isinstance(stream, Mapping) for stream in streams
    ):
        raise M5VideoError(f"FFprobe {profile_name} readback has no valid stream list")
    if not isinstance(format_value, Mapping) or "mp4" not in str(
        format_value.get("format_name", "")
    ).split(","):
        raise M5VideoError(f"{profile_name} video container is not MP4")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise M5VideoError(f"{profile_name} must contain exactly one video stream")
    expected_audio_count_streams = 1 if require_audio else 0
    if (len(audios) != expected_audio_count_streams
            or len(streams) != 1 + expected_audio_count_streams):
        raise M5VideoError(
            f"{profile_name} must contain one video and "
            f"{expected_audio_count_streams} audio streams"
        )

    video = videos[0]
    if video.get("codec_name") != "h264":
        raise M5VideoError(f"{profile_name} video codec must be H.264")
    if (
        _integer_field(video, "width", owner="video") != expected_width
        or _integer_field(video, "height", owner="video") != expected_height
        or video.get("pix_fmt") != "yuv420p"
    ):
        raise M5VideoError(
            f"{profile_name} video must read back as "
            f"{expected_width}x{expected_height} yuv420p"
        )
    try:
        average_rate = Fraction(str(video["avg_frame_rate"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise M5VideoError("video average frame rate is malformed") from exc
    if average_rate <= 0:
        raise M5VideoError("video average frame rate must be positive")
    if expected_rate_fraction is not None and average_rate != expected_rate_fraction:
        raise M5VideoError(
            f"{profile_name} video average frame rate is not "
            f"{_fraction_text(expected_rate_fraction)}"
        )
    frame_count = _integer_field(video, "nb_read_frames", owner="video")
    if expected_frame_count is not None and frame_count != expected_frame_count:
        raise M5VideoError(
            f"{profile_name} video has {frame_count} decoded frames, "
            f"expected {expected_frame_count}"
        )
    if _integer_field(video, "start_pts", owner="video") != 0:
        raise M5VideoError(
            f"{profile_name} video first presentation timestamp is not zero"
        )
    video_duration = _stream_duration(video, owner="video")
    expected_video_duration = (
        None if expected_frame_count is None or expected_rate_fraction is None
        else Fraction(expected_frame_count, 1) / expected_rate_fraction
    )
    if (
        expected_video_duration is not None
        and abs(float(video_duration - expected_video_duration))
        > _video_duration_tolerance(float(expected_rate_fraction))
    ):
        raise M5VideoError(
            f"{profile_name} video duration {float(video_duration):.9f}s "
            f"differs from expected {float(expected_video_duration):.9f}s"
        )

    audio_report: dict[str, Any] | None = None
    if require_audio:
        audio = audios[0]
        if audio.get("codec_name") != "aac":
            raise M5VideoError("episode presentation audio codec must be AAC")
        actual_audio_rate = _integer_field(audio, "sample_rate", owner="audio")
        actual_audio_channels = _integer_field(audio, "channels", owner="audio")
        if (
            expected_audio_rate is not None
            and actual_audio_rate != expected_audio_rate
        ) or (
            expected_audio_channels is not None
            and actual_audio_channels != expected_audio_channels
        ):
            expected_rate_text = (
                str(expected_audio_rate) if expected_audio_rate is not None else "any"
            )
            expected_channels_text = (
                str(expected_audio_channels)
                if expected_audio_channels is not None else "any"
            )
            raise M5VideoError(
                f"episode presentation audio must be "
                f"{expected_rate_text} Hz/{expected_channels_text} channels"
            )
        audio_duration = _stream_duration(audio, owner="audio")
        expected_audio_duration = (
            None if expected_audio_count is None or expected_audio_rate is None
            else Fraction(expected_audio_count, 1) / expected_audio_rate
        )
        if (
            expected_audio_duration is not None
            and abs(float(audio_duration - expected_audio_duration))
            > _audio_duration_tolerance(actual_audio_rate)
        ):
            raise M5VideoError(
                f"episode presentation audio duration {float(audio_duration):.9f}s "
                f"differs from expected {float(expected_audio_duration):.9f}s"
            )
        try:
            audio_start = Fraction(str(audio.get("start_time", "0")))
        except ValueError as exc:
            raise M5VideoError("audio start time is malformed") from exc
        if audio_start != 0:
            raise M5VideoError("episode presentation audio does not start at zero")
        audio_report = {
            "codec_name": "aac",
            "profile": audio.get("profile"),
            "sample_rate_hz": actual_audio_rate,
            "channel_count": actual_audio_channels,
            "channel_layout": audio.get("channel_layout"),
            "duration_seconds": float(audio_duration),
            "start_seconds": float(audio_start),
        }

    video_duration_ticks = video.get("duration_ts")
    try:
        video_duration_ticks = int(video_duration_ticks)
    except (TypeError, ValueError):
        video_duration_ticks = int(round(float(video_duration) * VIDEO_TIME_BASE_HZ))
    return {
        "path": str(media),
        "format_name": format_value.get("format_name"),
        "video": {
            "codec_name": "h264",
            "width": expected_width,
            "height": expected_height,
            "pixel_format": "yuv420p",
            "frame_count": frame_count,
            "frame_rate": _fraction_text(average_rate),
            "first_pts": 0,
            "duration_ticks": video_duration_ticks,
            "duration_seconds": float(video_duration),
        },
        "audio": audio_report,
    }


def probe_episode_video(
    path: str | Path,
    *,
    require_audio: bool = True,
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Read back and strictly validate the formal ``320x240`` M5 MP4."""

    return _probe_video_profile(
        path,
        expected_width=FRAME_WIDTH,
        expected_height=FRAME_HEIGHT,
        profile_name="formal episode",
        require_audio=require_audio,
        ffprobe=ffprobe,
    )


def probe_qa_review_video(
    path: str | Path,
    *,
    require_audio: bool = True,
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Read back and strictly validate the QA ``560x240`` review MP4."""

    return _probe_video_profile(
        path,
        expected_width=QA_PANEL_WIDTH,
        expected_height=QA_PANEL_HEIGHT,
        profile_name="QA review",
        require_audio=require_audio,
        ffprobe=ffprobe,
    )


def video_packet_sha256(
    path: str | Path,
    *,
    expected_frame_count: int | None = FRAME_COUNT,
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Hash ordered H.264 packet payloads independently of MP4 metadata.

    FFprobe hashes each encoded packet.  This function then hashes the ordered,
    length-bound digest sequence and separately binds its PTS/DTS timeline.  A
    counterfactual A/B pair must have equal payload and timeline hashes.
    """

    if expected_frame_count is not None:
        expected_frame_count = _positive_int(
            expected_frame_count, owner="expected packet frame count")
    media = _media_path(path, owner="video packet input")
    value = _ffprobe_json(
        media,
        [
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-show_entries",
            "packet=pts,dts,duration,size,flags,data_hash",
        ],
        ffprobe=ffprobe,
        owner="FFprobe video packet hashing",
    )
    packets = value.get("packets")
    if not isinstance(packets, list):
        raise M5VideoError("video packet hashing returned an invalid packet list")
    if expected_frame_count is not None and len(packets) != expected_frame_count:
        raise M5VideoError(
            f"video packet hashing expected {expected_frame_count} packets, got "
            f"{len(packets)}"
        )
    payload_digest = hashlib.sha256()
    payload_digest.update(b"avengine_m5_h264_packet_payloads_v1\0")
    timeline: list[dict[str, Any]] = []
    packet_hashes: list[str] = []
    for index, packet in enumerate(packets):
        if not isinstance(packet, Mapping):
            raise M5VideoError(f"video packet {index} is malformed")
        declaration = packet.get("data_hash")
        if not isinstance(declaration, str) or ":" not in declaration:
            raise M5VideoError(f"video packet {index} has no payload hash")
        algorithm, hexadecimal = declaration.split(":", 1)
        try:
            digest_bytes = bytes.fromhex(hexadecimal)
        except ValueError as exc:
            raise M5VideoError(
                f"video packet {index} payload hash is malformed"
            ) from exc
        if algorithm.casefold() != "sha256" or len(digest_bytes) != 32:
            raise M5VideoError(f"video packet {index} payload hash is not SHA-256")
        size = _integer_field(packet, "size", owner=f"packet[{index}]")
        payload_digest.update(struct.pack(">Q", size))
        payload_digest.update(digest_bytes)
        packet_hashes.append(hexadecimal.casefold())
        timeline.append(
            {
                "pts": packet.get("pts"),
                "dts": packet.get("dts"),
                "duration": packet.get("duration"),
                "flags": packet.get("flags"),
                "size": size,
                "data_sha256": hexadecimal.casefold(),
            }
        )
    timeline_bytes = json.dumps(
        timeline,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "algorithm": "sha256_length_bound_ordered_packet_payloads_v1",
        "packet_count": len(packets),
        "payload_sha256": payload_digest.hexdigest(),
        "timeline_sha256": hashlib.sha256(timeline_bytes).hexdigest(),
        "packet_sha256": packet_hashes,
    }


def _probe_authoritative_wav(
    path: Path,
    *,
    expected_sample_rate_hz: int | None = AUDIO_SAMPLE_RATE_HZ,
    expected_sample_count: int | None = AUDIO_SAMPLE_COUNT,
    expected_channel_count: int | None = AUDIO_CHANNEL_COUNT,
    ffprobe: str | Path,
) -> dict[str, Any]:
    expected_rate = (
        None if expected_sample_rate_hz is None
        else _positive_int(expected_sample_rate_hz, owner="expected audio sample rate")
    )
    expected_count = (
        None if expected_sample_count is None
        else _positive_int(expected_sample_count, owner="expected audio sample count")
    )
    expected_channels = (
        None if expected_channel_count is None
        else _positive_int(expected_channel_count, owner="expected audio channel count")
    )
    value = _ffprobe_json(
        path,
        [
            "-show_entries",
            (
                "stream=codec_type,codec_name,sample_rate,channels,channel_layout,"
                "time_base,start_time,duration_ts,duration"
            ),
            "-show_streams",
        ],
        ffprobe=ffprobe,
        owner="FFprobe authoritative WAVE readback",
    )
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(
        streams[0], Mapping
    ):
        raise M5VideoError("authoritative WAVE must contain exactly one stream")
    stream = streams[0]
    if stream.get("codec_type") != "audio" or not str(
        stream.get("codec_name", "")
    ).startswith("pcm_"):
        raise M5VideoError("authoritative WAVE must contain PCM audio")
    actual_rate = _integer_field(stream, "sample_rate", owner="authoritative WAVE")
    actual_channels = _integer_field(stream, "channels", owner="authoritative WAVE")
    if (
        (expected_rate is not None and actual_rate != expected_rate)
        or (expected_channels is not None and actual_channels != expected_channels)
    ):
        if expected_rate == AUDIO_SAMPLE_RATE_HZ and expected_channels == AUDIO_CHANNEL_COUNT:
            raise M5VideoError("authoritative WAVE must be 16 kHz stereo")
        raise M5VideoError(
            f"authoritative WAVE must be {expected_rate or 'any'} Hz/"
            f"{expected_channels or 'any'} channels"
        )
    duration = _stream_duration(stream, owner="authoritative WAVE")
    sample_count_fraction = duration * actual_rate
    if sample_count_fraction.denominator != 1:
        raise M5VideoError("authoritative WAVE duration is not aligned to samples")
    actual_count = int(sample_count_fraction)
    if expected_count is not None and actual_count != expected_count:
        raise M5VideoError(
            f"authoritative WAVE has {actual_count} samples, "
            f"expected {expected_count}"
        )
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate_hz": actual_rate,
        "channel_count": actual_channels,
        "sample_count": actual_count,
        "duration_seconds": float(duration),
    }


def _mux_binaural_wav_profile(
    base_video_path: str | Path,
    authoritative_wav_path: str | Path,
    output_path: str | Path,
    *,
    expected_width: int,
    expected_height: int,
    profile_name: str,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    sample_rate_hz: int | None = None,
    sample_count: int | None = None,
    audio_channel_count: int | None = AUDIO_CHANNEL_COUNT,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    base_video = _media_path(base_video_path, owner=f"{profile_name} base video")
    audio_wav = _media_path(authoritative_wav_path, owner="authoritative WAVE")
    output_owner = f"muxed {profile_name} output"
    destination = _new_mp4_path(output_path, owner=output_owner)
    executable = _tool(ffmpeg, owner="ffmpeg")

    requested_frame_count = (
        None if frame_count is None
        else _positive_int(frame_count, owner="expected video frame count")
    )
    requested_frame_rate = (
        None if frame_rate_hz is None
        else _positive_rate(frame_rate_hz, owner="expected video frame rate")
    )
    requested_sample_rate = (
        None if sample_rate_hz is None
        else _positive_int(sample_rate_hz, owner="expected audio sample rate")
    )
    requested_sample_count = (
        None if sample_count is None
        else _positive_int(sample_count, owner="expected audio sample count")
    )
    requested_channels = (
        None if audio_channel_count is None
        else _positive_int(audio_channel_count, owner="expected audio channel count")
    )

    base_report = _probe_video_profile(
        base_video,
        expected_width=expected_width,
        expected_height=expected_height,
        profile_name=profile_name,
        require_audio=False,
        expected_frame_count=requested_frame_count,
        expected_frame_rate_hz=requested_frame_rate,
        ffprobe=ffprobe,
    )
    actual_frame_count = int(base_report["video"]["frame_count"])
    actual_frame_rate = float(Fraction(base_report["video"]["frame_rate"]))
    wav_report = _probe_authoritative_wav(
        audio_wav,
        expected_sample_rate_hz=requested_sample_rate,
        expected_sample_count=requested_sample_count,
        expected_channel_count=requested_channels,
        ffprobe=ffprobe,
    )
    actual_sample_rate = int(wav_report["sample_rate_hz"])
    actual_sample_count = int(wav_report["sample_count"])
    actual_channels = int(wav_report["channel_count"])

    video_duration = actual_frame_count / actual_frame_rate
    audio_duration = actual_sample_count / actual_sample_rate
    if abs(video_duration - audio_duration) > max(
        _video_duration_tolerance(actual_frame_rate),
        1.0 / actual_sample_rate,
    ):
        raise M5VideoError(
            f"{profile_name} video/audio durations differ: "
            f"{video_duration:.9f}s vs {audio_duration:.9f}s"
        )

    decoded_reference = _decode_audio_f32(
        audio_wav, channel_count=actual_channels, ffmpeg=executable)
    if decoded_reference.shape != (actual_sample_count, actual_channels):
        raise M5VideoError(
            f"authoritative WAVE decode is not exactly "
            f"{actual_sample_count} samples/{actual_channels} channels"
        )

    temporary = _temporary_mp4(destination)
    command = _mux_command(
        ffmpeg=executable,
        base_video=base_video,
        audio_wav=audio_wav,
        destination=temporary,
        sample_rate_hz=actual_sample_rate,
        channel_count=actual_channels,
    )
    try:
        _run(command, owner="FFmpeg AAC mux")
        report = _probe_video_profile(
            temporary,
            expected_width=expected_width,
            expected_height=expected_height,
            profile_name=profile_name,
            require_audio=True,
            expected_frame_count=actual_frame_count,
            expected_frame_rate_hz=actual_frame_rate,
            expected_audio_sample_rate_hz=actual_sample_rate,
            expected_audio_sample_count=actual_sample_count,
            expected_audio_channel_count=actual_channels,
            ffprobe=ffprobe,
        )
        decoded_presentation = _decode_audio_f32(
            temporary, channel_count=actual_channels, ffmpeg=executable)
        if len(decoded_presentation) < actual_sample_count:
            raise M5VideoError(
                f"encoded AAC presentation is short by "
                f"{actual_sample_count - len(decoded_presentation)} samples"
            )
        report["audio"]["decoded_sample_count"] = int(len(decoded_presentation))
        report["audio"]["decoded_padding_samples"] = max(
            0, int(len(decoded_presentation) - actual_sample_count))
        base_hash = video_packet_sha256(
            base_video, expected_frame_count=actual_frame_count, ffprobe=ffprobe)
        muxed_hash = video_packet_sha256(
            temporary, expected_frame_count=actual_frame_count, ffprobe=ffprobe)
        if (
            base_hash["payload_sha256"] != muxed_hash["payload_sha256"]
            or base_hash["timeline_sha256"] != muxed_hash["timeline_sha256"]
        ):
            raise M5VideoError("mux changed the copied H.264 packet stream")
        _publish(temporary, destination, owner=output_owner)
        report["path"] = str(destination)
        report["video_packet_hash"] = muxed_hash
        report["video_stream_copy_verified"] = True
        report["authoritative_wav"] = {
            "path": str(audio_wav),
            "sample_rate_hz": actual_sample_rate,
            "channel_count": actual_channels,
            "sample_count": actual_sample_count,
            "duration_seconds": audio_duration,
        }
        return report
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def mux_binaural_wav(
    base_video_path: str | Path,
    authoritative_wav_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Mux an exact formal base video and authoritative binaural WAVE as AAC.

    The video stream is copied, not decoded or re-encoded.  Input metadata and
    chapters are discarded.  Duration is established by the two exact inputs;
    ``-shortest`` is intentionally absent.
    """

    return _mux_binaural_wav_profile(
        base_video_path,
        authoritative_wav_path,
        output_path,
        expected_width=FRAME_WIDTH,
        expected_height=FRAME_HEIGHT,
        profile_name="formal episode",
        frame_count=FRAME_COUNT,
        frame_rate_hz=FRAME_RATE,
        sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
        sample_count=AUDIO_SAMPLE_COUNT,
        audio_channel_count=AUDIO_CHANNEL_COUNT,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )


def mux_qa_binaural_wav(
    qa_base_video_path: str | Path,
    authoritative_wav_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Mux the exact ``560x240`` QA base video with the binaural AAC track."""

    return _mux_binaural_wav_profile(
        qa_base_video_path,
        authoritative_wav_path,
        output_path,
        expected_width=QA_PANEL_WIDTH,
        expected_height=QA_PANEL_HEIGHT,
        profile_name="QA review",
        frame_count=FRAME_COUNT,
        frame_rate_hz=FRAME_RATE,
        sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
        sample_count=AUDIO_SAMPLE_COUNT,
        audio_channel_count=AUDIO_CHANNEL_COUNT,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )


def _decode_audio_f32(
    path: Path,
    *,
    channel_count: int = AUDIO_CHANNEL_COUNT,
    ffmpeg: str | Path,
) -> np.ndarray:
    channel_count = _positive_int(channel_count, owner="audio channel count")
    command = [
        _tool(ffmpeg, owner="ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = _run(
        command,
        owner="FFmpeg AAC/WAVE float32 decode",
        timeout=120.0,
        binary_stdout=True,
    )
    payload = completed.stdout
    if not isinstance(payload, bytes) or len(payload) % (4 * channel_count):
        raise M5VideoError(
            "decoded audio byte count is not whole float32 frames"
        )
    values = np.frombuffer(payload, dtype="<f4")
    result = values.reshape(-1, channel_count).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise M5VideoError("decoded audio contains non-finite samples")
    return result


def _best_lag_samples(
    reference: np.ndarray,
    decoded: np.ndarray,
    *,
    maximum_lag: int,
) -> int:
    if maximum_lag < 0:
        raise M5VideoError("maximum AAC diagnostic lag must be non-negative")
    reference_centered = reference - np.mean(reference, axis=0, keepdims=True)
    decoded_centered = decoded - np.mean(decoded, axis=0, keepdims=True)
    if not np.any(reference_centered) or not np.any(decoded_centered):
        raise M5VideoError(
            "AAC lag diagnostic requires a non-silent reference and decode"
        )
    linear_size = len(reference) + len(decoded) - 1
    fft_size = 1 << (linear_size - 1).bit_length()
    correlation = np.zeros(linear_size, dtype=np.float64)
    for channel in range(AUDIO_CHANNEL_COUNT):
        convolution = np.fft.irfft(
            np.fft.rfft(decoded_centered[:, channel], fft_size)
            * np.fft.rfft(reference_centered[::-1, channel], fft_size),
            fft_size,
        )[:linear_size]
        correlation += convolution
    lags = np.arange(-(len(reference) - 1), len(decoded), dtype=np.int64)
    mask = np.abs(lags) <= maximum_lag
    if not np.any(mask):
        raise M5VideoError("AAC lag search has no candidate samples")
    candidates = np.flatnonzero(mask)
    peak = candidates[int(np.argmax(correlation[mask]))]
    return int(lags[peak])


def _aligned(
    reference: np.ndarray,
    decoded: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        count = min(len(reference), len(decoded) - lag)
        reference_start = 0
        decoded_start = lag
    else:
        count = min(len(reference) + lag, len(decoded))
        reference_start = -lag
        decoded_start = 0
    if count <= 0:
        raise M5VideoError("AAC lag leaves no overlapping samples")
    return (
        reference[reference_start : reference_start + count],
        decoded[decoded_start : decoded_start + count],
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _snr_db(reference: np.ndarray, decoded: np.ndarray) -> float:
    signal = float(np.sum(np.square(reference, dtype=np.float64)))
    error = float(np.sum(np.square(decoded - reference, dtype=np.float64)))
    if signal <= 0.0:
        raise M5VideoError("AAC SNR reference channel is silent")
    if error == 0.0:
        return 300.0
    return float(10.0 * math.log10(signal / error))


def aac_decode_diagnostics(
    muxed_video_path: str | Path,
    authoritative_wav_path: str | Path,
    *,
    maximum_lag_samples: int = 4096,
    expected_width: int = FRAME_WIDTH,
    expected_height: int = FRAME_HEIGHT,
    expected_frame_count: int | None = FRAME_COUNT,
    expected_frame_rate_hz: float | None = FRAME_RATE,
    expected_sample_rate_hz: int | None = AUDIO_SAMPLE_RATE_HZ,
    expected_sample_count: int | None = AUDIO_SAMPLE_COUNT,
    expected_channel_count: int | None = AUDIO_CHANNEL_COUNT,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Decode AAC and report count, lag, correlation, SNR and LR cues.

    These are codec/readback diagnostics, not a replacement for the
    authoritative WAVE. The function reports padding and shortfall without
    silently trimming either stream.
    """

    muxed = _media_path(muxed_video_path, owner="muxed episode video")
    reference_path = _media_path(
        authoritative_wav_path, owner="authoritative binaural WAVE"
    )
    wav_report = _probe_authoritative_wav(
        reference_path,
        expected_sample_rate_hz=expected_sample_rate_hz,
        expected_sample_count=expected_sample_count,
        expected_channel_count=expected_channel_count,
        ffprobe=ffprobe,
    )
    actual_sample_rate = int(wav_report["sample_rate_hz"])
    actual_sample_count = int(wav_report["sample_count"])
    actual_channel_count = int(wav_report["channel_count"])
    if actual_channel_count != AUDIO_CHANNEL_COUNT:
        raise M5VideoError("AAC diagnostics require two-channel binaural audio")
    _probe_video_profile(
        muxed,
        expected_width=expected_width,
        expected_height=expected_height,
        profile_name="AAC diagnostics",
        require_audio=True,
        expected_frame_count=expected_frame_count,
        expected_frame_rate_hz=expected_frame_rate_hz,
        expected_audio_sample_rate_hz=actual_sample_rate,
        expected_audio_sample_count=actual_sample_count,
        expected_audio_channel_count=actual_channel_count,
        ffprobe=ffprobe,
    )
    reference = _decode_audio_f32(
        reference_path, channel_count=actual_channel_count, ffmpeg=ffmpeg)
    decoded = _decode_audio_f32(
        muxed, channel_count=actual_channel_count, ffmpeg=ffmpeg)
    if reference.shape != (actual_sample_count, actual_channel_count):
        raise M5VideoError(
            f"reference decode is not exactly "
            f"{actual_sample_count} samples/{actual_channel_count} channels"
        )
    lag = _best_lag_samples(
        reference,
        decoded,
        maximum_lag=maximum_lag_samples,
    )
    aligned_reference, aligned_decoded = _aligned(reference, decoded, lag)
    correlation_by_channel = [
        _correlation(aligned_reference[:, channel], aligned_decoded[:, channel])
        for channel in range(actual_channel_count)
    ]
    snr_by_channel = [
        _snr_db(aligned_reference[:, channel], aligned_decoded[:, channel])
        for channel in range(actual_channel_count)
    ]
    normal_score = float(np.mean(correlation_by_channel))
    swapped_correlations = [
        _correlation(aligned_reference[:, 0], aligned_decoded[:, 1]),
        _correlation(aligned_reference[:, 1], aligned_decoded[:, 0]),
    ]
    swapped_score = float(np.mean(swapped_correlations))
    return {
        "reference_sample_count": int(len(reference)),
        "decoded_sample_count": int(len(decoded)),
        "sample_count_matches": len(decoded) == actual_sample_count,
        "presentation_sample_count": min(int(len(decoded)), actual_sample_count),
        "presentation_sample_count_matches": len(decoded) >= actual_sample_count,
        "decoded_padding_samples": max(0, int(len(decoded) - actual_sample_count)),
        "decoded_shortfall_samples": max(0, actual_sample_count - int(len(decoded))),
        "sample_rate_hz": actual_sample_rate,
        "channel_count": actual_channel_count,
        "lag_samples": lag,
        "aligned_sample_count": int(len(aligned_reference)),
        "correlation_by_channel": correlation_by_channel,
        "minimum_correlation": min(correlation_by_channel),
        "snr_db_by_channel": snr_by_channel,
        "minimum_snr_db": min(snr_by_channel),
        "lr_normal_correlation": normal_score,
        "lr_swapped_correlation": swapped_score,
        "lr_swap_suspected": swapped_score > normal_score + 0.02,
        "diagnostic_only": True,
    }


def _trajectory_points(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M5VideoError("QA trajectory must be a sequence of top-down pixel pairs")
    result: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
            raise M5VideoError(f"QA trajectory point {index} is not a pair")
        if len(point) != 2:
            raise M5VideoError(f"QA trajectory point {index} is not a pair")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise M5VideoError(
                f"QA trajectory point {index} is not numeric"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise M5VideoError(f"QA trajectory point {index} is not finite")
        if not (0.0 <= x < TOPDOWN_WIDTH and 0.0 <= y < QA_PANEL_HEIGHT):
            raise M5VideoError(f"QA trajectory point {index} is outside the panel")
        result.append((x, y))
    return result


def compose_main_topdown_panel(
    main_rgb: Any,
    topdown_rgb: Any,
    *,
    text: str | None = None,
    trajectory: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Compose one deterministic ``560x240`` main + QA-topdown RGB panel.

    Trajectory coordinates are expressed in the 240x240 top-down image.  Text
    and trajectory overlays are QA-only; they never become formal ``view_ids``.
    """

    main = _rgb_frame(
        main_rgb,
        shape=(FRAME_HEIGHT, FRAME_WIDTH, 3),
        owner="main RGB frame",
    )
    topdown = _rgb_frame(
        topdown_rgb,
        shape=(QA_PANEL_HEIGHT, TOPDOWN_WIDTH, 3),
        owner="top-down RGB frame",
    )
    if text is not None and not isinstance(text, str):
        raise M5VideoError("QA panel text must be a string or None")
    points = _trajectory_points(trajectory)
    panel = np.concatenate((main, topdown), axis=1)
    if not points and not text:
        return np.ascontiguousarray(panel)

    image = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(image)
    shifted = [(FRAME_WIDTH + x, y) for x, y in points]
    if len(shifted) >= 2:
        draw.line(shifted, fill=(255, 215, 0), width=2)
    if shifted:
        x, y = shifted[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 64, 64))
    if text:
        box = draw.textbbox((4, 4), text)
        draw.rectangle(
            (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2),
            fill=(0, 0, 0),
        )
        draw.text((4, 4), text, fill=(255, 255, 255))
    result = np.asarray(image, dtype=np.uint8)
    if result.shape != (QA_PANEL_HEIGHT, QA_PANEL_WIDTH, 3):
        raise AssertionError("QA panel compositor produced the wrong shape")
    return np.ascontiguousarray(result)


def compose_main_topdown_frames(
    main_frames: Iterable[Any],
    topdown_frames: Iterable[Any],
    *,
    text_by_frame: Sequence[str | None] | None = None,
    trajectory_by_frame: Sequence[Sequence[Sequence[float]] | None] | None = None,
) -> list[np.ndarray]:
    """Compose matching frame sequences with optional per-frame QA overlays."""

    mains = list(main_frames)
    topdowns = list(topdown_frames)
    if len(mains) != len(topdowns):
        raise M5VideoError("main and top-down QA frame counts differ")
    if text_by_frame is not None and len(text_by_frame) != len(mains):
        raise M5VideoError("per-frame QA text count differs from frame count")
    if trajectory_by_frame is not None and len(trajectory_by_frame) != len(mains):
        raise M5VideoError("per-frame QA trajectory count differs from frame count")
    return [
        compose_main_topdown_panel(
            main,
            topdown,
            text=None if text_by_frame is None else text_by_frame[index],
            trajectory=(
                None
                if trajectory_by_frame is None
                else trajectory_by_frame[index]
            ),
        )
        for index, (main, topdown) in enumerate(zip(mains, topdowns, strict=True))
    ]


__all__ = [
    "AUDIO_CHANNEL_COUNT",
    "AUDIO_SAMPLE_COUNT",
    "AUDIO_SAMPLE_RATE_HZ",
    "FRAME_COUNT",
    "FRAME_HEIGHT",
    "FRAME_RATE",
    "FRAME_WIDTH",
    "M5VideoError",
    "QA_PANEL_HEIGHT",
    "QA_PANEL_WIDTH",
    "aac_decode_diagnostics",
    "compose_main_topdown_frames",
    "compose_main_topdown_panel",
    "encode_h264_base_video",
    "encode_h264_qa_base_video",
    "mux_binaural_wav",
    "mux_qa_binaural_wav",
    "probe_episode_video",
    "probe_qa_review_video",
    "video_packet_sha256",
]
