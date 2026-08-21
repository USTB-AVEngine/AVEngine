#!/usr/bin/env python3
"""Build the current MP3D dynamic-audio review clip from engine artifacts.

Encodes the captured 75-frame RGB array at its native resolution and muxes
the authoritative dynamic-audio binaural mixture through the frozen M5 audio
contract. Review assembly only; no dataset admission is claimed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.m6x.visual_profile import (  # noqa: E402
    ReviewVisualProfile,
    encode_profiled_h264_base_video,
    mux_profiled_binaural_wav,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-capture-dir", required=True, type=Path)
    parser.add_argument("--mixture-wav", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to replace output: {output}")
    if output.suffix != ".mp4":
        parser.error("output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    rgb_path = args.visual_capture_dir.resolve() / "arrays/rgb.npy"
    rgb = np.load(rgb_path)
    if rgb.ndim != 4 or rgb.shape[3] != 3:
        parser.error(f"rgb array must be [frames, height, width, 3]: {rgb.shape}")
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    profile = ReviewVisualProfile(
        path=rgb_path,
        raw={},
        profile_id=f"current_mp3d_dynamic_review_{width}x{height}",
        capture_resolution_hw=(height, width),
        diagnostic_panel_resolution_hw=(height, width),
    )
    base = output.with_name(output.stem + ".base.mp4")
    encode_profiled_h264_base_video(rgb, base, profile=profile)
    mux_report = mux_profiled_binaural_wav(
        base, args.mixture_wav.resolve(), output, profile=profile
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "base_video": str(base),
                "audio": mux_report.get("audio"),
            },
            ensure_ascii=False,
            default=str,
        )[:600]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
