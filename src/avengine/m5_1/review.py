"""Annotated main-view + Topdown review media for the M5.1 comparison.

M5.1 review media is deliberately separate from immutable Timeline v2 and its
five-second/75-frame formal video contract.  This module accepts an explicit
frame count and never promotes the right-side Topdown panel to a dataset view.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.m5_1.orientation import habitat_basis_from_yaw_degrees


REVIEW_SCHEMA = "avengine_m5_1_annotated_review_v1"
REVIEW_WIDTH = 1280
REVIEW_HEIGHT = 480
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
DEFAULT_FPS = 15


class M51ReviewError(ValueError):
    """An annotated M5.1 review input or encoded output is invalid."""


@dataclass(frozen=True)
class SourceOverlayTrack:
    """Per-frame source state used only to explain a review video."""

    source_id: str
    label: str
    asset_class: str
    sound_class: str
    color_rgb: tuple[int, int, int]
    positions_m: np.ndarray
    current_event_by_frame: tuple[str | None, ...]
    active_by_frame: tuple[bool, ...]
    true_flags: tuple[str, ...] = ()
    center_clearance_m: np.ndarray | None = None
    main_marker_xy: np.ndarray | None = None


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _rgb_frames(value: Any, *, owner: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 4
        or array.shape[-1] != 3
        or array.dtype != np.uint8
        or array.shape[0] < 1
    ):
        raise M51ReviewError(
            f"{owner} must be nonempty uint8 [frame,height,width,3] RGB"
        )
    return np.ascontiguousarray(array)


def _validated_track(track: SourceOverlayTrack, frame_count: int) -> SourceOverlayTrack:
    if not track.source_id or not track.label:
        raise M51ReviewError("source overlay identity and label must be nonempty")
    if not track.asset_class or not track.sound_class:
        raise M51ReviewError("source asset_class and sound_class must be nonempty")
    if len(track.color_rgb) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
        for value in track.color_rgb
    ):
        raise M51ReviewError("source color_rgb must contain three uint8 integers")
    positions = np.asarray(track.positions_m, dtype=np.float64)
    if positions.shape != (frame_count, 3) or not np.all(np.isfinite(positions)):
        raise M51ReviewError("source positions_m must be finite [frame,3]")
    if len(track.current_event_by_frame) != frame_count:
        raise M51ReviewError("current_event_by_frame length differs from video")
    if len(track.active_by_frame) != frame_count:
        raise M51ReviewError("active_by_frame length differs from video")
    if any(
        event is not None and (not isinstance(event, str) or not event)
        for event in track.current_event_by_frame
    ):
        raise M51ReviewError("event IDs must be nonempty strings or null")
    if len(set(track.true_flags)) != len(track.true_flags) or any(
        not isinstance(flag, str) or not flag for flag in track.true_flags
    ):
        raise M51ReviewError("true_flags must contain unique nonempty strings")
    if track.center_clearance_m is not None:
        clearance = np.asarray(track.center_clearance_m, dtype=np.float64)
        if clearance.shape != (frame_count,) or not np.all(np.isfinite(clearance)):
            raise M51ReviewError("center_clearance_m must be finite [frame]")
    if track.main_marker_xy is not None:
        markers = np.asarray(track.main_marker_xy, dtype=np.float64)
        if markers.shape != (frame_count, 2):
            raise M51ReviewError("main_marker_xy must have shape [frame,2]")
        finite_or_nan_pair = np.logical_or(
            np.all(np.isfinite(markers), axis=1), np.all(np.isnan(markers), axis=1)
        )
        if not np.all(finite_or_nan_pair):
            raise M51ReviewError("main markers must be finite pairs or NaN pairs")
    return track


def _source_geometry(
    position_m: Sequence[float],
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
) -> tuple[float, float]:
    source = np.asarray(position_m, dtype=np.float64)
    listener = np.asarray(listener_position_m, dtype=np.float64)
    delta = source - listener
    distance = float(np.linalg.norm(delta))
    # Match RLR/Habitat exactly: world_from_listener rotates local forward -Z
    # and local/right-ear +X.  M4/M5 diagnostics define positive azimuth right.
    basis = habitat_basis_from_yaw_degrees(listener_yaw_deg)
    right = np.asarray(basis.right_xyz, dtype=np.float64)
    forward = np.asarray(basis.forward_xyz, dtype=np.float64)
    local_right = float(np.dot(delta, right))
    local_forward = float(np.dot(delta, forward))
    azimuth = math.degrees(math.atan2(local_right, local_forward))
    return distance, azimuth


def _alpha_box(
    image: Image.Image, bounds: tuple[int, int, int, int], alpha: int = 176
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(bounds, fill=(0, 0, 0, alpha))
    image.alpha_composite(overlay)


def _draw_main_marker(
    draw: ImageDraw.ImageDraw,
    marker_xy: Sequence[float],
    *,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = (float(marker_xy[0]), float(marker_xy[1]))
    radius = 9
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=3,
    )
    draw.line((x - 13, y, x + 13, y), fill=color, width=2)
    draw.line((x, y - 13, x, y + 13), fill=color, width=2)
    draw.text(
        (x + 13, y - 15),
        label,
        font=_font(13),
        fill=color,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def compose_annotated_frames(
    *,
    main_rgb: Any,
    topdown_rgb: Any,
    tracks: Sequence[SourceOverlayTrack],
    clip_id: str,
    room_id: str,
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
    aggregate_true_flags: Sequence[str] = (),
    audio_diagnostic_by_frame: Sequence[str] | None = None,
    center_gate_pass: bool,
    fps: int = DEFAULT_FPS,
) -> np.ndarray:
    """Compose deterministic 1280x480 annotated comparison frames."""

    main = _rgb_frames(main_rgb, owner="main_rgb")
    topdown = _rgb_frames(topdown_rgb, owner="topdown_rgb")
    if main.shape[0] != topdown.shape[0]:
        raise M51ReviewError("main and Topdown frame counts differ")
    if not clip_id or not room_id:
        raise M51ReviewError("clip_id and room_id must be nonempty")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise M51ReviewError("fps must be a positive integer")
    listener = np.asarray(listener_position_m, dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise M51ReviewError("listener_position_m must be finite [3]")
    if not math.isfinite(float(listener_yaw_deg)):
        raise M51ReviewError("listener_yaw_deg must be finite")
    if len(tracks) < 1 or len({track.source_id for track in tracks}) != len(tracks):
        raise M51ReviewError("tracks must contain unique source IDs")
    checked = tuple(_validated_track(track, main.shape[0]) for track in tracks)
    flags = tuple(aggregate_true_flags)
    if len(flags) != len(set(flags)) or any(
        not isinstance(flag, str) or not flag for flag in flags
    ):
        raise M51ReviewError("aggregate_true_flags must be unique nonempty strings")
    diagnostics = (
        None
        if audio_diagnostic_by_frame is None
        else tuple(audio_diagnostic_by_frame)
    )
    if diagnostics is not None and (
        len(diagnostics) != main.shape[0]
        or any(not isinstance(value, str) or not value for value in diagnostics)
    ):
        raise M51ReviewError(
            "audio_diagnostic_by_frame must contain one nonempty string per frame"
        )

    font = _font(15)
    small = _font(13)
    composed: list[np.ndarray] = []
    resampling = Image.Resampling.BILINEAR
    for frame_index in range(main.shape[0]):
        left = (
            Image.fromarray(main[frame_index], mode="RGB")
            .resize((PANEL_WIDTH, PANEL_HEIGHT), resampling)
            .convert("RGBA")
        )
        right = (
            Image.fromarray(topdown[frame_index], mode="RGB")
            .resize((PANEL_WIDTH, PANEL_HEIGHT), resampling)
            .convert("RGBA")
        )
        canvas = Image.new("RGBA", (REVIEW_WIDTH, REVIEW_HEIGHT), (0, 0, 0, 255))
        canvas.alpha_composite(left, (0, 0))
        canvas.alpha_composite(right, (PANEL_WIDTH, 0))
        draw = ImageDraw.Draw(canvas)

        for track in checked:
            if track.main_marker_xy is None:
                continue
            marker = np.asarray(track.main_marker_xy[frame_index], dtype=np.float64)
            if np.all(np.isfinite(marker)):
                source_height, source_width = main.shape[1:3]
                scaled = (
                    float(marker[0]) * PANEL_WIDTH / source_width,
                    float(marker[1]) * PANEL_HEIGHT / source_height,
                )
                _draw_main_marker(
                    draw,
                    scaled,
                    label=track.label,
                    color=track.color_rgb,
                )

        line_height = 19
        box_height = line_height * (
            3 + len(checked) + int(diagnostics is not None)
        ) + 8
        _alpha_box(canvas, (0, 0, REVIEW_WIDTH - 1, box_height))
        draw = ImageDraw.Draw(canvas)
        gate = "PASS" if center_gate_pass else "FAIL"
        minimum_values = [
            float(np.min(np.asarray(track.center_clearance_m)))
            for track in checked
            if track.center_clearance_m is not None
        ]
        minimum = min(minimum_values) if minimum_values else float("nan")
        clearance_text = "n/a" if not math.isfinite(minimum) else f"{minimum:.3f}m"
        draw.text(
            (8, 4),
            (
                f"M5.1 | {clip_id} | room={room_id} | "
                f"frame={frame_index:03d}/{main.shape[0] - 1:03d} "
                f"t={frame_index / fps:05.2f}s | center-point={gate} min={clearance_text}"
            ),
            font=font,
            fill=(255, 255, 255),
        )
        for source_index, track in enumerate(checked):
            event = track.current_event_by_frame[frame_index] or "none"
            active = "ACTIVE" if track.active_by_frame[frame_index] else "silent"
            distance, azimuth = _source_geometry(
                track.positions_m[frame_index], listener, listener_yaw_deg
            )
            source_flags = ",".join(track.true_flags) if track.true_flags else "none"
            line = (
                f"{track.label} [{track.source_id}] {track.asset_class}/{track.sound_class} "
                f"event={event}:{active} dist={distance:.2f}m az={azimuth:+.1f}deg "
                f"flags={source_flags}"
            )
            draw.text(
                (8, 4 + line_height * (source_index + 1)),
                line[:190],
                font=small,
                fill=track.color_rgb,
            )
        aggregate = ",".join(flags) if flags else "none"
        draw.text(
            (8, 4 + line_height * (len(checked) + 1)),
            f"clip flags={aggregate} | right panel=QA Topdown (not a dataset view)",
            font=small,
            fill=(235, 235, 235),
        )
        if diagnostics is not None:
            draw.text(
                (8, 4 + line_height * (len(checked) + 2)),
                f"binaural mix: {diagnostics[frame_index]}"[:190],
                font=small,
                fill=(255, 225, 120),
            )
        draw.text(
            (PANEL_WIDTH + 8, REVIEW_HEIGHT - 20),
            "Habitat Topdown / complete paths / current source centers",
            font=small,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        composed.append(np.asarray(canvas.convert("RGB"), dtype=np.uint8))
    return np.ascontiguousarray(np.stack(composed, axis=0))


def encode_annotated_review(
    frames: Iterable[Any],
    output_path: str | Path,
    *,
    fps: int = DEFAULT_FPS,
    audio_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Encode a review-only H.264 MP4 and optionally mux a stereo audio track."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise M51ReviewError("ffmpeg and ffprobe are required")
    destination = Path(output_path).resolve()
    if destination.suffix.casefold() != ".mp4":
        raise M51ReviewError("review output must use .mp4")
    if os.path.lexists(destination):
        raise M51ReviewError(f"refusing to overwrite review output: {destination}")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise M51ReviewError("fps must be a positive integer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = destination.with_name(f".{destination.stem}.video_only.mp4")
    if os.path.lexists(base):
        raise M51ReviewError(f"temporary review path already exists: {base}")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{REVIEW_WIDTH}x{REVIEW_HEIGHT}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(base),
    ]
    count = 0
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for count, value in enumerate(frames, start=1):
            frame = np.asarray(value)
            if frame.dtype != np.uint8 or frame.shape != (
                REVIEW_HEIGHT,
                REVIEW_WIDTH,
                3,
            ):
                raise M51ReviewError("encoded review frames must be uint8 [480,1280,3]")
            process.stdin.write(np.ascontiguousarray(frame).tobytes(order="C"))
        process.stdin.close()
        returncode = process.wait()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        if returncode != 0:
            raise M51ReviewError(f"ffmpeg video encoding failed: {stderr}")
        if count < 1:
            raise M51ReviewError("review video cannot contain zero frames")

        if audio_path is None:
            os.link(base, destination)
        else:
            audio = Path(audio_path).resolve()
            if not audio.is_file():
                raise M51ReviewError(f"review audio is missing: {audio}")
            duration = count / fps
            mux = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(base),
                    "-i",
                    str(audio),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-ar",
                    "16000",
                    "-ac",
                    "2",
                    "-t",
                    f"{duration:.12f}",
                    "-map_metadata",
                    "-1",
                    "-movflags",
                    "+faststart",
                    str(destination),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if mux.returncode != 0:
                destination.unlink(missing_ok=True)
                raise M51ReviewError(f"ffmpeg audio mux failed: {mux.stderr.strip()}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        destination.unlink(missing_ok=True)
        raise
    finally:
        base.unlink(missing_ok=True)

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise M51ReviewError(f"ffprobe review readback failed: {probe.stderr.strip()}")
    payload = json.loads(probe.stdout)
    video = next(
        (
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if (
        not isinstance(video, Mapping)
        or video.get("codec_name") != "h264"
        or int(video.get("width", 0)) != REVIEW_WIDTH
        or int(video.get("height", 0)) != REVIEW_HEIGHT
        or int(video.get("nb_read_frames", 0)) != count
    ):
        raise M51ReviewError("encoded review video readback differs")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "schema": REVIEW_SCHEMA,
        "path": str(destination),
        "sha256": digest,
        "byte_size": destination.stat().st_size,
        "frame_count": count,
        "frame_rate_hz": fps,
        "duration_seconds": count / fps,
        "width": REVIEW_WIDTH,
        "height": REVIEW_HEIGHT,
        "topdown_is_qa_only": True,
        "audio_muxed": audio_path is not None,
        "ffprobe": payload,
    }


__all__ = [
    "DEFAULT_FPS",
    "M51ReviewError",
    "REVIEW_HEIGHT",
    "REVIEW_SCHEMA",
    "REVIEW_WIDTH",
    "SourceOverlayTrack",
    "compose_annotated_frames",
    "encode_annotated_review",
]
