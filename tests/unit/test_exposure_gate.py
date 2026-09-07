"""Exposure gate rejects blown-out frames and records the frame source."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from avengine.qa.exposure_gate import (
    CAPTURE_FRAMES_KIND,
    PLACEHOLDER_EXPOSURE_GATE_CONFIG,
    REVIEW_FRAMES_KIND,
    RGB_NPY_KIND,
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
    assert failed["frame_source"]["kind"] == CAPTURE_FRAMES_KIND
    assert failed["exposure_gate"]["frame_source"]["kind"] == CAPTURE_FRAMES_KIND

    passed = apply_exposure_gate({"status": "delivered", "episode_id": "normal"}, normal_root)
    assert passed["status"] == "delivered"
    assert passed["exposure_gate"]["status"] == "pass"
    assert "reason" not in passed
    assert passed["frame_source"]["kind"] == CAPTURE_FRAMES_KIND


def test_habitat_rgb_npy_first_mid_last_when_no_png_dir(tmp_path):
    first = np.full((8, 12, 3), 80, dtype=np.uint8)
    mid = np.full((8, 12, 3), 90, dtype=np.uint8)
    last = np.full((8, 12, 3), 100, dtype=np.uint8)
    stack = np.stack([first, np.full((8, 12, 3), 85, dtype=np.uint8), mid, last], axis=0)
    npy = tmp_path / "capture" / "rgb.npy"
    npy.parent.mkdir(parents=True)
    np.save(npy, stack)
    review_dir = tmp_path / "batch_review" / "frames"
    for index in range(3):
        _write_png(review_dir / f"frame_{index:03d}.png", np.full((8, 12, 3), 255, dtype=np.uint8))

    result = apply_exposure_gate({"status": "delivered", "episode_id": "habitat"}, tmp_path)
    assert result["status"] == "delivered"
    assert result["frame_source"]["kind"] == RGB_NPY_KIND
    assert result["exposure_gate"]["frame_source"]["kind"] == RGB_NPY_KIND
    indices = [row["index"] for row in result["exposure_gate"]["frames"]]
    assert indices == [0, 2, 3]
    means = [row["mean_gray"] for row in result["exposure_gate"]["frames"]]
    assert abs(means[0] - 80.0) < 1e-6
    assert abs(means[1] - 90.0) < 1e-6
    assert abs(means[2] - 100.0) < 1e-6


def test_capture_png_is_preferred_over_rgb_npy_and_review_extracts(tmp_path):
    capture = np.full((8, 12, 3), 110, dtype=np.uint8)
    _write_png(tmp_path / "capture" / "frames" / "frame_0000.png", capture)
    np.save(tmp_path / "capture" / "rgb.npy", np.full((4, 8, 12, 3), 255, dtype=np.uint8))
    _write_png(tmp_path / "batch_review" / "frames" / "frame_000.png", np.full((8, 12, 3), 255, dtype=np.uint8))
    result = apply_exposure_gate({"status": "delivered", "episode_id": "ue"}, tmp_path)
    assert result["frame_source"]["kind"] == CAPTURE_FRAMES_KIND
    assert result["exposure_gate"]["status"] == "pass"


def test_review_extracts_are_used_only_without_capture_or_rgb_npy(tmp_path):
    normal = np.full((8, 12, 3), 130, dtype=np.uint8)
    for index in range(3):
        _write_png(tmp_path / "batch_review" / "frames" / f"frame_{index:03d}.png", normal)
    result = apply_exposure_gate({"status": "delivered", "episode_id": "review_only"}, tmp_path)
    assert result["frame_source"]["kind"] == REVIEW_FRAMES_KIND
    assert result["exposure_gate"]["status"] == "pass"
    assert "review extract" in result["frame_source"]["note"]
