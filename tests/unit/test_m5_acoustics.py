from __future__ import annotations

import pytest

from avengine.m3.runtime import RuntimeContractError
from avengine.m5.acoustics import (
    AcousticKeyframe,
    trajectory_record,
    validate_acoustic_keyframes,
)


def _frames() -> list[AcousticKeyframe]:
    return [
        AcousticKeyframe(
            tick=3200 * index,
            sample_index=(3200 * index + 1) // 3,
            source_positions_m={
                "source0": (0.0, 1.0, index / 75.0),
                "source1": (1.0, 1.0, -index / 75.0),
            },
            listener_position_m=(-2.5, 1.55, 0.0),
            listener_orientation_wxyz=(2**-0.5, 0.0, -(2**-0.5), 0.0),
        )
        for index in range(75)
    ]


def test_exact_visual_frame_acoustic_grid() -> None:
    frames = validate_acoustic_keyframes(
        _frames(), expected_source_ids=("source0", "source1")
    )
    assert len(frames) == 75
    assert frames[0].sample_index == 0
    assert frames[-1].sample_index == 78_933
    record = trajectory_record(frames, ("source0", "source1"))
    assert record["source_ids"] == ["source0", "source1"]
    assert record["keyframes"][-1]["tick"] == 236_800


def test_rejects_rounded_fixed_1067_sample_grid() -> None:
    frames = _frames()
    frames[2] = AcousticKeyframe(
        tick=6400,
        sample_index=2134,
        source_positions_m=frames[2].source_positions_m,
        listener_position_m=frames[2].listener_position_m,
        listener_orientation_wxyz=frames[2].listener_orientation_wxyz,
    )
    with pytest.raises(RuntimeContractError, match="non-rational"):
        validate_acoustic_keyframes(
            frames, expected_source_ids=("source0", "source1")
        )


def test_rejects_source_identity_drift() -> None:
    frames = _frames()
    frames[30] = AcousticKeyframe(
        tick=frames[30].tick,
        sample_index=frames[30].sample_index,
        source_positions_m={"source0": (0.0, 0.0, 0.0), "renamed": (1.0, 0.0, 0.0)},
        listener_position_m=frames[30].listener_position_m,
        listener_orientation_wxyz=frames[30].listener_orientation_wxyz,
    )
    with pytest.raises(RuntimeContractError, match="identity set"):
        validate_acoustic_keyframes(
            frames, expected_source_ids=("source0", "source1")
        )
