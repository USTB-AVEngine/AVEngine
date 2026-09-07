"""Exposure gate rejects blown-out frames and accepts a normal interior."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from avengine.qa.exposure_gate import (
    PLACEHOLDER_EXPOSURE_GATE_CONFIG,
    apply_exposure_gate,
    compute_exposure_stats,
)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"cv2.imwrite failed: {path}")
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb).save(path)


def test_placeholder_thresholds_are_labeled():
    assert PLACEHOLDER_EXPOSURE_GATE_CONFIG["threshold_kind"] == "placeholder"
    assert PLACEHOLDER_EXPOSURE_GATE_CONFIG["mean_gray_fail_above"] == 235.0
    assert PLACEHOLDER_EXPOSURE_GATE_CONFIG["sat_share_fail_above"] == 0.20


def test_all_white_frame_fails_and_normal_frame_passes(tmp_path):
    white = np.full((32, 48, 3), 255, dtype=np.uint8)
    normal = np.full((32, 48, 3), 120, dtype=np.uint8)
    white_stats = compute_exposure_stats(white)
    normal_stats = compute_exposure_stats(normal)
    assert white_stats["mean_gray"] > 235.0
    assert white_stats["sat_share"] > 0.20
    assert abs(normal_stats["mean_gray"] - 120.0) < 1e-6
    assert normal_stats["sat_share"] == 0.0

    white_root = tmp_path / "white"
    normal_root = tmp_path / "normal"
    for index in range(3):
        _write_png(white_root / "capture" / "frames" / f"frame_{index:04d}.png", white)
        _write_png(normal_root / "capture" / "frames" / f"frame_{index:04d}.png", normal)

    failed = apply_exposure_gate({"status": "delivered", "episode_id": "white"}, white_root)
    assert failed["status"] == "review_failed"
    assert "mean_gray" in failed["reason"]
    assert failed["exposure_gate"]["threshold_kind"] == "placeholder"
    assert failed["exposure_gate"]["status"] == "failed"

    passed = apply_exposure_gate({"status": "delivered", "episode_id": "normal"}, normal_root)
    assert passed["status"] == "delivered"
    assert passed["exposure_gate"]["status"] == "pass"
    assert "reason" not in passed
