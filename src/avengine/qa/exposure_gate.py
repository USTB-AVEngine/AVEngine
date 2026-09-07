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

CAPTURE_FRAMES_KIND = "capture/frames"
REVIEW_FRAMES_KIND = "batch_review/frames"
RGB_NPY_KIND = "rgb.npy"


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


def _list_frame_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _choose_roles(paths: Sequence[Path]) -> list[tuple[str, Path]]:
    if not paths:
        return []
    if len(paths) == 1:
        chosen = [paths[0], paths[0], paths[0]]
    else:
        chosen = [paths[0], paths[len(paths) // 2], paths[-1]]
    return list(zip(("first", "mid", "last"), chosen))


def _classify_png_directory(directory: Path, episode_root: Path) -> str:
    try:
        relative = directory.resolve().relative_to(Path(episode_root).resolve())
    except ValueError:
        relative = Path(directory.name)
    parts = [part.lower() for part in relative.parts]
    if "batch_review" in parts:
        return REVIEW_FRAMES_KIND
    return CAPTURE_FRAMES_KIND


def _rgb_npy_candidates(episode_root: Path) -> list[Path]:
    root = Path(episode_root)
    return [
        root / "capture" / "rgb.npy",
        root / "episode" / "capture" / "rgb.npy",
        root / "rgb.npy",
    ]


def _load_rgb_npy_gate_frames(path: Path) -> list[dict[str, Any]]:
    array = np.load(path, mmap_mode="r")
    try:
        frames = array
        if frames.ndim == 3:
            frames = frames[None, ...]
        if frames.ndim != 4 or int(frames.shape[0]) < 1:
            raise ValueError(f"rgb.npy must be NxHxWxC, got shape {getattr(array, 'shape', None)}")
        count = int(frames.shape[0])
        indices = [0, count // 2, count - 1] if count > 1 else [0, 0, 0]
        selected = []
        for role, index in zip(("first", "mid", "last"), indices):
            selected.append({
                "role": role,
                "index": int(index),
                "path": f"{path}[{index}]",
                "rgb": np.array(frames[int(index)], copy=True),
                "source_dir": str(path),
            })
        return selected
    finally:
        del array


def _png_selection(directory: Path, *, kind: str) -> dict[str, Any]:
    found = _list_frame_paths(directory)
    if not found:
        return {"kind": kind, "source": str(directory), "frames": [], "note": ""}
    rows = []
    for role, path in _choose_roles(found):
        rows.append({
            "role": role,
            "path": path,
            "source_dir": str(directory),
        })
    if kind == CAPTURE_FRAMES_KIND:
        note = "real capture PNG frames"
    else:
        note = "batch_review extract PNG frames"
    return {"kind": kind, "source": str(directory), "frames": rows, "note": note}


def _select_gate_frames(episode_root: Path) -> dict[str, Any]:
    root = Path(episode_root)
    capture_dirs = [
        root / "capture" / "frames",
        root / "episode" / "capture" / "frames",
        root / "frames",
    ]
    for directory in capture_dirs:
        found = _list_frame_paths(directory)
        if not found:
            continue
        kind = _classify_png_directory(directory, root)
        if kind != CAPTURE_FRAMES_KIND:
            continue
        return _png_selection(directory, kind=CAPTURE_FRAMES_KIND)

    for npy in _rgb_npy_candidates(root):
        if not npy.is_file():
            continue
        try:
            frames = _load_rgb_npy_gate_frames(npy)
        except (OSError, ValueError) as exc:
            return {
                "kind": RGB_NPY_KIND,
                "source": str(npy),
                "frames": [],
                "note": f"rgb.npy unreadable: {exc}",
                "error": str(exc),
            }
        return {
            "kind": RGB_NPY_KIND,
            "source": str(npy),
            "frames": frames,
            "note": "habitat rgb.npy frames 0/mid/last",
        }

    review_dirs = [
        root / "batch_review" / "frames",
        root / "episode" / "batch_review" / "frames",
    ]
    for directory in review_dirs:
        if _list_frame_paths(directory):
            return _png_selection(directory, kind=REVIEW_FRAMES_KIND)

    return {
        "kind": None,
        "source": None,
        "frames": [],
        "note": "no RGB frames under capture/frames, rgb.npy, or batch_review/frames",
    }


def _frame_source_block(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": selected.get("kind"),
        "path": selected.get("source"),
        "note": selected.get("note"),
    }


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
    frame_source = _frame_source_block(selected)
    if not selected.get("frames"):
        result = deepcopy(dict(review))
        result["status"] = "review_failed"
        result["reason"] = (
            "exposure_gate placeholder: no RGB frames under capture/frames, "
            "rgb.npy, or batch_review/frames"
            + (f" ({selected.get('note')})" if selected.get("error") else "")
        )
        result["exposure_gate"] = {
            "status": "failed",
            "threshold_kind": "placeholder",
            "config": merged,
            "frames": [],
            "frame_source": frame_source,
        }
        result["frame_source"] = frame_source
        return result
    measurements = []
    failures = []
    for item in selected["frames"]:
        rgb = item.get("rgb")
        if rgb is None:
            rgb = _read_rgb(Path(item["path"]))
        stats = compute_exposure_stats(rgb, sat_value=sat_value)
        row = {
            "role": item["role"],
            "path": str(item["path"]),
            "mean_gray": stats["mean_gray"],
            "sat_share": stats["sat_share"],
            "sat_value": sat_value,
        }
        if "index" in item:
            row["index"] = int(item["index"])
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
        "frame_source": frame_source,
    }
    result = deepcopy(dict(review))
    result["exposure_gate"] = block
    result["frame_source"] = frame_source
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
