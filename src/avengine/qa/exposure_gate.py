"""Placeholder RGB exposure gate used by batch review.

The numeric thresholds are placeholders. They still fail an episode when the
mean gray exceeds 235 or the share of pixels at or above 250 exceeds 20%.
Do not loosen them to make a batch pass.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLACEHOLDER_EXPOSURE_GATE_CONFIG: dict[str, Any] = {
    "schema": "avengine_qa_exposure_gate_config_v1",
    "threshold_kind": "placeholder",
    "mean_gray_fail_above": 235.0,
    "sat_share_fail_above": 0.20,
    "sat_value": 250,
    "luma_weights_rgb": (0.299, 0.587, 0.114),
    "frames": "first_mid_last",
}


def compute_exposure_stats(
    rgb: np.ndarray,
    *,
    sat_value: int = 250,
    luma_weights_rgb: Sequence[float] = (0.299, 0.587, 0.114),
) -> dict[str, float]:
    """Return mean Rec.601 gray and the fraction of pixels with gray >= sat_value."""
    frame = np.asarray(rgb)
    if frame.ndim == 2:
        gray = frame.astype(np.float64)
    elif frame.ndim == 3 and frame.shape[-1] >= 3:
        weights = np.asarray(luma_weights_rgb, dtype=np.float64).reshape(1, 1, 3)
        gray = np.sum(frame[..., :3].astype(np.float64) * weights, axis=-1)
    else:
        raise ValueError(f"RGB frame must be HxW or HxWx3, got shape {frame.shape}")
    if gray.size == 0:
        raise ValueError("RGB frame is empty")
    sat = float(np.count_nonzero(gray >= float(sat_value)) / gray.size)
    return {
        "mean_gray": float(np.mean(gray)),
        "sat_share": sat,
        "sat_value": float(sat_value),
        "pixel_count": int(gray.size),
        "min_gray": float(np.min(gray)),
        "max_gray": float(np.max(gray)),
    }


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read RGB frame: {path}")
    return np.asarray(image[..., ::-1])


def _frame_directories(episode_root: Path) -> list[Path]:
    root = Path(episode_root)
    return [
        root / "capture" / "frames",
        root / "frames",
        root / "batch_review" / "frames",
        root / "episode" / "capture" / "frames",
    ]


def _list_frame_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _select_gate_frames(episode_root: Path) -> list[dict[str, Any]]:
    frames: list[Path] = []
    source = None
    for directory in _frame_directories(episode_root):
        found = _list_frame_paths(directory)
        if found:
            frames = found
            source = directory
            break
    if not frames:
        return []
    if len(frames) == 1:
        chosen = [frames[0], frames[0], frames[0]]
    else:
        chosen = [frames[0], frames[len(frames) // 2], frames[-1]]
    roles = ["first", "mid", "last"]
    return [
        {"role": role, "path": path, "source_dir": str(source)}
        for role, path in zip(roles, chosen)
    ]


def apply_exposure_gate(
    review: Mapping,
    episode_root: Path,
    *,
    config: Mapping | None = None,
) -> dict:
    """If exposure fails, return a copy of review with status review_failed and a numeric reason. Otherwise return review unchanged (or with an exposure_gate pass block)."""
    merged = dict(PLACEHOLDER_EXPOSURE_GATE_CONFIG)
    if config:
        merged.update(dict(config))
    merged["threshold_kind"] = "placeholder"
    mean_limit = float(merged["mean_gray_fail_above"])
    sat_limit = float(merged["sat_share_fail_above"])
    sat_value = int(merged["sat_value"])
    selected = _select_gate_frames(Path(episode_root))
    if not selected:
        result = deepcopy(dict(review))
        result["status"] = "review_failed"
        result["reason"] = (
            "exposure_gate placeholder: no RGB frames under capture/frames, "
            "frames, or batch_review/frames"
        )
        result["exposure_gate"] = {
            "status": "failed",
            "threshold_kind": "placeholder",
            "config": merged,
            "frames": [],
        }
        return result
    measurements = []
    failures = []
    for item in selected:
        stats = compute_exposure_stats(_read_rgb(item["path"]), sat_value=sat_value)
        row = {
            "role": item["role"],
            "path": str(item["path"]),
            "mean_gray": stats["mean_gray"],
            "sat_share": stats["sat_share"],
            "sat_value": sat_value,
        }
        measurements.append(row)
        if stats["sat_share"] > sat_limit or stats["mean_gray"] > mean_limit:
            failures.append(row)
    block = {
        "status": "failed" if failures else "pass",
        "threshold_kind": "placeholder",
        "config": {
            "mean_gray_fail_above": mean_limit,
            "sat_share_fail_above": sat_limit,
            "sat_value": sat_value,
            "threshold_kind": "placeholder",
        },
        "frames": measurements,
    }
    result = deepcopy(dict(review))
    result["exposure_gate"] = block
    if not failures:
        return result
    parts = [
        f"{row['role']} mean_gray={row['mean_gray']:.3f} sat_share={row['sat_share']:.4f}"
        for row in failures
    ]
    result["status"] = "review_failed"
    result["reason"] = (
        "exposure_gate placeholder: sat_share>"
        f"{sat_limit} or mean_gray>{mean_limit}: " + "; ".join(parts)
    )
    return result
