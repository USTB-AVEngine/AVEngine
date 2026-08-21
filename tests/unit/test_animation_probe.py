from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.m7.animation_probe import (
    AnimationProbeError,
    cyclic_phase_distance,
    probe_capture_animation_playback,
    select_walk_phase_pairs,
)

RNG = np.random.default_rng(20260821)
TEXTURE_A = RNG.integers(40, 220, size=(30, 18, 3), dtype=np.uint8)
TEXTURE_B = TEXTURE_A[::-1].copy()


def _synthetic_capture(
    tmp_path: Path, *, animate: bool, frame_count: int = 24
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    rgb = np.zeros((frame_count, 120, 220, 3), dtype=np.uint8)
    rgb[:] = 90
    for index in range(frame_count):
        phase = (index * 0.1) % 1.0
        for slot, base_col in (("left_actor", 20), ("right_actor", 130)):
            col = base_col + index * 3
            texture = TEXTURE_A
            if animate and phase >= 0.5:
                texture = TEXTURE_B
            rgb[index, 40:70, col : col + 18] = texture
        frames.append(
            {
                "frame_index": index,
                "camera_pose": {"location_cm": [0, 0, 0], "rotation_deg": [0, 0, 0]},
                "actor_states": [
                    {
                        "source_slot_id": slot,
                        "action_id": "walk",
                        "action_phase": phase,
                    }
                    for slot in ("left_actor", "right_actor")
                ],
            }
        )
    (tmp_path / "arrays").mkdir()
    np.save(tmp_path / "arrays" / "rgb.npy", rgb)
    (tmp_path / "frame_records.json").write_text(json.dumps({"frames": frames}))
    return tmp_path


def test_cyclic_phase_distance_wraps() -> None:
    assert cyclic_phase_distance(0.9, 0.1) == pytest.approx(0.2)
    assert cyclic_phase_distance(0.25, 0.75) == pytest.approx(0.5)


def test_pair_selection_requires_phase_distance() -> None:
    frames = [
        {
            "actor_states": [
                {
                    "source_slot_id": "a",
                    "action_id": "walk",
                    "action_phase": (i * 0.1) % 1.0,
                }
            ]
        }
        for i in range(20)
    ]
    pairs = select_walk_phase_pairs(frames, "a")
    assert pairs
    assert all(j - i <= 12 for i, j in pairs)


def test_probe_flags_sliding_and_accepts_animation(tmp_path: Path) -> None:
    sliding = _synthetic_capture(tmp_path / "sliding", animate=False)
    animated = _synthetic_capture(tmp_path / "animated", animate=True)

    sliding_report = probe_capture_animation_playback(
        sliding, slot_order_left_to_right=("left_actor", "right_actor")
    )
    animated_report = probe_capture_animation_playback(
        animated, slot_order_left_to_right=("left_actor", "right_actor")
    )
    assert sliding_report["status"] == "fail"
    assert all(
        slot["verdict"] == "sliding_without_animation"
        for slot in sliding_report["slots"].values()
    )
    assert animated_report["status"] == "pass"
    assert all(
        slot["verdict"] == "animated"
        for slot in animated_report["slots"].values()
    )


def test_probe_rejects_moving_cameras(tmp_path: Path) -> None:
    capture = _synthetic_capture(tmp_path, animate=True)
    payload = json.loads((capture / "frame_records.json").read_text())
    payload["frames"][5]["camera_pose"] = {
        "location_cm": [1, 0, 0],
        "rotation_deg": [0, 0, 0],
    }
    (capture / "frame_records.json").write_text(json.dumps(payload))
    with pytest.raises(AnimationProbeError):
        probe_capture_animation_playback(
            capture, slot_order_left_to_right=("left_actor", "right_actor")
        )
