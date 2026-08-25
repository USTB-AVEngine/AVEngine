"""Rendered-pose probe for static-camera captures.

The capture frame records can claim advancing walk phases while the render
plays no montage (the actor slides as a rigid template). The existing
animation readback validates the scheduled montage time, so it cannot catch
that failure class. This probe checks the pixels instead: for frame pairs
whose declared walk phases differ by a large cyclic distance, a sliding
actor's foreground is a similarity transform of itself (small residual
after the best translation-plus-scale fit), while a playing gait changes
the silhouette (large residual). Localization uses frame-difference
components (robust slot assignment by horizontal order); the residual is
restricted to pixels that are also foreground against the temporal-median
background plate, which drops the revealed-background band behind fast
movers. Needs only a static camera and the per-frame RGB array; no
semantic masks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

PROBE_SCHEMA = "avengine_capture_animation_playback_probe_v1"
MIN_COMPONENT_PIXELS = 250
DIFF_THRESHOLD = 14
SEARCH_RADIUS_PX = 8
SEARCH_SCALES = (0.90, 0.94, 0.98, 1.0, 1.02, 1.06, 1.10)
DEFAULT_MIN_PHASE_DISTANCE = 0.25
DEFAULT_MAX_FRAME_GAP = 12
DEFAULT_MAX_PAIRS = 8
SLIDING_RESIDUAL_MAX = 7.5
ANIMATED_RESIDUAL_MIN = 9.5


class AnimationProbeError(RuntimeError):
    """Raised when the probe inputs violate its contract."""


def cyclic_phase_distance(a: float, b: float) -> float:
    distance = abs(float(a) - float(b)) % 1.0
    return min(distance, 1.0 - distance)


def select_walk_phase_pairs(
    frames: Sequence[Mapping[str, Any]],
    slot_id: str,
    *,
    min_phase_distance: float = DEFAULT_MIN_PHASE_DISTANCE,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> list[tuple[int, int]]:
    """Frame index pairs with nearby indices but distant declared walk phases."""

    walk_phases: dict[int, float] = {}
    for index, frame in enumerate(frames):
        for state in frame.get("actor_states") or ():
            if (
                state.get("source_slot_id") == slot_id
                and state.get("action_id") == "walk"
            ):
                walk_phases[index] = float(state.get("action_phase", 0.0))
    indices = sorted(walk_phases)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in indices:
        if i in used:
            continue
        for j in indices:
            if j <= i or j - i > max_frame_gap or j in used:
                continue
            if (
                cyclic_phase_distance(walk_phases[i], walk_phases[j])
                >= min_phase_distance
            ):
                pairs.append((i, j))
                used.update((i, j))
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


def static_camera_background(rgb: Any, *, frame_stride: int = 3) -> np.ndarray:
    """Per-pixel temporal median plate for a static-camera capture."""

    sample = np.asarray(rgb[::frame_stride]).astype(np.float32)
    return np.median(sample, axis=0)


def foreground_mask(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Pixels that differ from the temporal-median background plate."""

    diff = np.abs(frame.astype(np.float32) - background).max(axis=-1)
    return diff > DIFF_THRESHOLD


def moving_component_masks(
    earlier: np.ndarray, later: np.ndarray, *, expected_count: int
) -> list[tuple[np.ndarray, tuple[float, float]]]:
    """Connected moving regions between two consecutive frames, largest first."""

    diff = np.abs(earlier.astype(np.int16) - later.astype(np.int16)).max(axis=-1)
    mask = ndimage.binary_dilation(diff > DIFF_THRESHOLD, iterations=2)
    labels, count = ndimage.label(mask)
    components: list[tuple[np.ndarray, tuple[float, float]]] = []
    if not count:
        return components
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, count + 1))
    order = np.argsort(sizes)[::-1]
    for rank in order[: expected_count * 2]:
        if sizes[rank] < MIN_COMPONENT_PIXELS:
            continue
        component = labels == (int(rank) + 1)
        rows, cols = np.nonzero(component)
        components.append((component, (float(rows.mean()), float(cols.mean()))))
        if len(components) == expected_count:
            break
    return components


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float64).mean(axis=-1)


def aligned_pose_residual(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask_a: np.ndarray,
    centroid_a: tuple[float, float],
    centroid_b: tuple[float, float],
) -> float | None:
    """Median gray residual over the mask after the best similarity fit.

    Approach routes change the on-screen size between the paired frames, so
    the exhaustive search covers translation and scale. A rigid slide
    collapses to sensor-level residual under some similarity transform; a
    played gait is non-rigid and does not.
    """

    gray_a = _gray(frame_a)
    gray_b = _gray(frame_b)
    rows, cols = np.nonzero(mask_a)
    if rows.size < MIN_COMPONENT_PIXELS:
        return None
    values_a = gray_a[rows, cols]
    height, width = gray_b.shape
    best = None
    for scale in SEARCH_SCALES:
        scaled_rows = np.round(
            centroid_b[0] + scale * (rows - centroid_a[0])
        ).astype(int)
        scaled_cols = np.round(
            centroid_b[1] + scale * (cols - centroid_a[1])
        ).astype(int)
        for dr in range(-SEARCH_RADIUS_PX, SEARCH_RADIUS_PX + 1):
            for dc in range(-SEARCH_RADIUS_PX, SEARCH_RADIUS_PX + 1):
                shifted_rows = scaled_rows + dr
                shifted_cols = scaled_cols + dc
                keep = (
                    (shifted_rows >= 0)
                    & (shifted_rows < height)
                    & (shifted_cols >= 0)
                    & (shifted_cols < width)
                )
                if keep.sum() < MIN_COMPONENT_PIXELS:
                    continue
                residual = np.median(
                    np.abs(
                        values_a[keep]
                        - gray_b[shifted_rows[keep], shifted_cols[keep]]
                    )
                )
                if best is None or residual < best:
                    best = float(residual)
    return best


def probe_capture_animation_playback(
    visual_capture_dir: str | Path,
    *,
    slot_order_left_to_right: Sequence[str],
    min_phase_distance: float = DEFAULT_MIN_PHASE_DISTANCE,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    sliding_max: float = SLIDING_RESIDUAL_MAX,
    animated_min: float = ANIMATED_RESIDUAL_MIN,
) -> dict[str, Any]:
    """Probe one static-camera capture for scheduled-but-unrendered walks.

    ``slot_order_left_to_right`` assigns the moving components to slots by
    their horizontal image order (derive it from the capture geometry).
    """

    capture = Path(visual_capture_dir).resolve()
    payload = json.loads((capture / "frame_records.json").read_text("utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise AnimationProbeError("frame_records must carry frames")
    first_camera = frames[0].get("camera_pose")
    for frame in frames:
        if frame.get("camera_pose") != first_camera:
            raise AnimationProbeError(
                "the probe requires a static camera; this capture moves it"
            )
    rgb = np.load(capture / "arrays" / "rgb.npy", mmap_mode="r")
    if rgb.ndim != 4 or rgb.shape[0] != len(frames):
        raise AnimationProbeError("rgb array must align with the frame records")
    background = static_camera_background(rgb)

    slots: dict[str, Any] = {}
    for slot_position, slot_id in enumerate(slot_order_left_to_right):
        pairs = select_walk_phase_pairs(
            frames,
            slot_id,
            min_phase_distance=min_phase_distance,
            max_frame_gap=max_frame_gap,
            max_pairs=max_pairs,
        )
        residuals: list[float] = []
        for i, j in pairs:
            if i + 1 >= len(frames) or j + 1 >= len(frames):
                continue
            frame_i = np.asarray(rgb[i])
            frame_j = np.asarray(rgb[j])
            components_i = moving_component_masks(
                frame_i,
                np.asarray(rgb[i + 1]),
                expected_count=len(slot_order_left_to_right),
            )
            components_j = moving_component_masks(
                frame_j,
                np.asarray(rgb[j + 1]),
                expected_count=len(slot_order_left_to_right),
            )
            if len(components_i) < len(slot_order_left_to_right) or len(
                components_j
            ) < len(slot_order_left_to_right):
                continue
            by_column_i = sorted(components_i, key=lambda item: item[1][1])
            by_column_j = sorted(components_j, key=lambda item: item[1][1])
            mask_a, centroid_a = by_column_i[slot_position]
            _, centroid_b = by_column_j[slot_position]
            pure_mask = mask_a & foreground_mask(frame_i, background)
            residual = aligned_pose_residual(
                frame_i, frame_j, pure_mask, centroid_a, centroid_b
            )
            if residual is not None:
                residuals.append(residual)
        if not residuals:
            verdict = "insufficient_evidence"
            median_residual = None
        else:
            median_residual = float(np.median(residuals))
            if median_residual <= sliding_max:
                verdict = "sliding_without_animation"
            elif median_residual >= animated_min:
                verdict = "animated"
            else:
                verdict = "inconclusive"
        slots[slot_id] = {
            "pair_count": len(residuals),
            "pair_residuals": [round(value, 2) for value in residuals],
            "median_residual": median_residual,
            "verdict": verdict,
        }
    return {
        "schema": PROBE_SCHEMA,
        "capture": str(capture),
        "sliding_max": sliding_max,
        "animated_min": animated_min,
        "slots": slots,
        "status": (
            "fail"
            if any(
                slot["verdict"] == "sliding_without_animation"
                for slot in slots.values()
            )
            else "inconclusive"
            if any(
                slot["verdict"] in ("inconclusive", "insufficient_evidence")
                for slot in slots.values()
            )
            else "pass"
        ),
    }
