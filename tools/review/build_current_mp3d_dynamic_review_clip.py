#!/usr/bin/env python3
"""Build the current MP3D dynamic-audio review clip from engine artifacts.

Encodes the captured RGB array at its native resolution and muxes the
authoritative dynamic-audio binaural mixture through the review audio
contract. Review assembly only; no dataset admission is claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.rooms.visual_profile import (  # noqa: E402
    ReviewVisualProfile,
    encode_profiled_h264_base_video,
    mux_profiled_binaural_wav,
    resolve_review_capture_channel_order,
)


def _declared_clock(capture_dir: Path, rgb: np.ndarray, mixture: Path,
                    frame_rate_override: float | None) -> dict[str, float | int]:
    receipt_path = capture_dir / "research_receipt.json"
    receipt = {}
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read capture receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError(f"capture receipt must be an object: {receipt_path}")
    capture = receipt.get("capture") or {}
    if not isinstance(capture, dict):
        raise ValueError("capture receipt capture field must be an object")
    declared_count = capture.get("frame_count")
    completed_count = capture.get("completed_frame_count")
    for name, value in (("frame_count", declared_count),
                        ("completed_frame_count", completed_count)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"capture {name} must be a positive integer")
    actual_count = int(rgb.shape[0])
    for name, value in (("frame_count", declared_count),
                        ("completed_frame_count", completed_count)):
        if value is not None and value != actual_count:
            raise ValueError(
                f"RGB has {actual_count} frames but capture {name} declares {value}"
            )
    receipt_rate = capture.get("frame_rate_hz")
    if receipt_rate is not None:
        if (
            isinstance(receipt_rate, bool)
            or not isinstance(receipt_rate, (int, float))
            or not math.isfinite(float(receipt_rate))
            or float(receipt_rate) <= 0.0
        ):
            raise ValueError("capture receipt frame rate must be finite and positive")
    frame_rate = frame_rate_override if frame_rate_override is not None else receipt_rate
    if frame_rate is None:
        frame_rate = 15.0
    if (
        isinstance(frame_rate, bool)
        or not isinstance(frame_rate, (int, float))
        or not math.isfinite(float(frame_rate))
        or float(frame_rate) <= 0.0
    ):
        raise ValueError("review frame rate must be finite and positive")
    if receipt_rate is not None and float(receipt_rate) != float(frame_rate):
        raise ValueError(
            f"requested frame rate {frame_rate} differs from capture receipt {receipt_rate}"
        )
    try:
        audio_info = sf.info(mixture)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot inspect mixture WAV: {mixture}") from exc
    if audio_info.channels != 2:
        raise ValueError(
            f"review mixture must be stereo, got {audio_info.channels} channels"
        )
    receipt_audio = receipt.get("audio") or {}
    for name, actual in (
        ("sample_rate_hz", audio_info.samplerate),
        ("sample_count", audio_info.frames),
    ):
        declared = receipt_audio.get(name)
        if declared is not None and declared != actual:
            raise ValueError(
                f"mixture {name} {actual} differs from capture receipt {declared}"
            )
    return {
        "frame_count": actual_count,
        "frame_rate_hz": float(frame_rate),
        "sample_rate_hz": int(audio_info.samplerate),
        "sample_count": int(audio_info.frames),
        "channel_count": int(audio_info.channels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-capture-dir", required=True, type=Path)
    parser.add_argument("--mixture-wav", required=True, type=Path)
    parser.add_argument(
        "--channel-order",
        choices=("rgb", "bgr"),
        default=None,
        help="read channel order from the capture receipt; legacy UE arrays "
        "without metadata need --channel-order bgr (other historical inputs default to rgb)",
    )
    parser.add_argument(
        "--frame-rate-hz", type=float,
        help="override the capture receipt frame rate when no receipt exists",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to replace output: {output}")
    if output.suffix != ".mp4":
        parser.error("output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    rgb_path = args.visual_capture_dir.resolve() / "arrays/rgb.npy"
    rgb = np.load(rgb_path, allow_pickle=False)
    if rgb.ndim != 4 or rgb.shape[3] != 3 or rgb.shape[0] < 1:
        parser.error(
            f"rgb array must be [positive frames, height, width, 3]: {rgb.shape}")
    mixture = args.mixture_wav.resolve()
    try:
        clock = _declared_clock(
            args.visual_capture_dir.resolve(), rgb, mixture, args.frame_rate_hz)
        channel_order = resolve_review_capture_channel_order(
            args.visual_capture_dir.resolve(), args.channel_order)
    except ValueError as exc:
        parser.error(str(exc))
    if channel_order == "bgr":
        rgb = np.ascontiguousarray(rgb[..., ::-1])
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    profile = ReviewVisualProfile(
        path=rgb_path,
        raw={},
        profile_id=f"current_mp3d_dynamic_review_{width}x{height}",
        capture_resolution_hw=(height, width),
        diagnostic_panel_resolution_hw=(height, width),
        capture_frame_rate_hz=float(clock["frame_rate_hz"]),
    )
    base = output.with_name(output.stem + ".base.mp4")
    encode_profiled_h264_base_video(
        rgb, base, profile=profile,
        frame_count=clock["frame_count"],
        frame_rate_hz=clock["frame_rate_hz"],
    )
    mux_report = mux_profiled_binaural_wav(
        base, mixture, output, profile=profile,
        frame_count=clock["frame_count"],
        frame_rate_hz=clock["frame_rate_hz"],
        sample_rate_hz=clock["sample_rate_hz"],
        sample_count=clock["sample_count"],
        audio_channel_count=clock["channel_count"],
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "base_video": str(base),
                "clock": clock,
                "stored_channel_order": channel_order,
                "audio": mux_report.get("audio"),
            },
            ensure_ascii=False,
            default=str,
        )[:600]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
