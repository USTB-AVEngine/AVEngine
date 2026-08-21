from __future__ import annotations

import numpy as np
import pytest

from avengine.m2.glb import extract_actions, parse_glb
from avengine.m2.glb_write import build_glb
from avengine.m5_1.human_runtime import (
    HumanRuntimeError,
    _retime_walking_loop_to_profile,
    prepare_rocketbox_habitat_runtime,
)


def _append(binary: bytearray, values: np.ndarray) -> tuple[int, int]:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    payload = np.ascontiguousarray(values).tobytes(order="C")
    binary.extend(payload)
    return offset, len(payload)


def _two_action_glb(
    *,
    walking_times: np.ndarray | None = None,
    second_walking_times: np.ndarray | None = None,
) -> bytes:
    first_times = np.asarray(
        np.arange(1, 39, dtype=np.float32) / np.float32(30.0)
        if walking_times is None
        else walking_times,
        dtype=np.dtype("<f4"),
    )
    second_times = np.asarray(
        first_times[[0, -1]] if second_walking_times is None else second_walking_times,
        dtype=np.dtype("<f4"),
    )
    idle_times = np.asarray([0.0, 1.0], dtype=np.dtype("<f4"))
    first_values = np.tile(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.dtype("<f4")),
        (len(first_times), 1),
    )
    second_values = np.tile(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.dtype("<f4")),
        (len(second_times), 1),
    )
    second_values[:, 1] = np.linspace(0.0, 0.2, len(second_times), dtype=np.float32)
    second_values /= np.linalg.norm(second_values, axis=1, keepdims=True)
    idle_values = np.tile(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.dtype("<f4")),
        (len(idle_times), 1),
    )
    arrays = (
        first_times,
        second_times,
        idle_times,
        first_values,
        second_values,
        idle_values,
    )
    binary = bytearray()
    views = []
    accessors = []
    for index, array in enumerate(arrays):
        offset, length = _append(binary, array)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        accessors.append(
            {
                "bufferView": index,
                "componentType": 5126,
                "count": len(array),
                "type": "SCALAR" if array.ndim == 1 else "VEC4",
            }
        )
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "Bip01 Pelvis"}, {"name": "Bip01 Head"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "animations": [
            {
                "name": "Walking",
                "samplers": [
                    {"input": 0, "output": 3, "interpolation": "LINEAR"},
                    {"input": 1, "output": 4, "interpolation": "LINEAR"},
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
                    {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
                ],
            },
            {
                "name": "Idle",
                "samplers": [{"input": 2, "output": 5, "interpolation": "LINEAR"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
            },
        ],
    }
    return build_glb(document, binary)


def test_profile_retime_is_explicit_and_preserves_values_and_idle() -> None:
    source = _two_action_glb()
    before = extract_actions(parse_glb(source))

    untouched, record = _retime_walking_loop_to_profile(
        source, walking_profile_sample_count=None
    )
    assert untouched == source
    assert record is None

    payload, record = _retime_walking_loop_to_profile(
        source, walking_profile_sample_count=19
    )
    after = extract_actions(parse_glb(payload))
    walking_before, idle_before = before
    walking_after, idle_after = after

    assert idle_after == idle_before
    assert record == {
        "strategy": "affine_retime_to_profile_sample_count",
        "source_duration_ticks": 59_200,
        "target_duration_ticks": 60_800,
        "profile_sample_count": 19,
        "source_key_counts": [2, 38],
        "time_scale": 38 / 37,
    }
    assert {
        round(
            (max(channel.timestamps_seconds) - min(channel.timestamps_seconds)) * 48_000
        )
        for channel in walking_after.channels
    } == {60_800}
    assert all(
        after_channel.values == before_channel.values
        and len(after_channel.timestamps_seconds)
        == len(before_channel.timestamps_seconds)
        for before_channel, after_channel in zip(
            walking_before.channels, walking_after.channels, strict=True
        )
    )


@pytest.mark.parametrize("value", [True, 0, -1, 18, 20, 19.0])
def test_profile_retime_rejects_invalid_or_unsupported_sample_count(
    value: object,
) -> None:
    with pytest.raises(HumanRuntimeError, match="positive integer|audited 19-sample"):
        _retime_walking_loop_to_profile(
            _two_action_glb(),
            walking_profile_sample_count=value,  # type: ignore[arg-type]
        )


def test_profile_retime_rejects_wrong_duration_key_count_or_timeline() -> None:
    wrong_count = np.arange(1, 38, dtype=np.float32) / np.float32(30.0)
    with pytest.raises(HumanRuntimeError, match="2/38-key.*59,200-tick"):
        _retime_walking_loop_to_profile(
            _two_action_glb(walking_times=wrong_count),
            walking_profile_sample_count=19,
        )

    drifted = np.arange(1, 39, dtype=np.float32) / np.float32(30.0)
    drifted[-1] += np.float32(1.0 / 30.0)
    with pytest.raises(HumanRuntimeError, match="2/38-key.*59,200-tick"):
        _retime_walking_loop_to_profile(
            _two_action_glb(walking_times=drifted),
            walking_profile_sample_count=19,
        )

    second = np.asarray([1.0 / 30.0, 39.0 / 30.0], dtype=np.float32)
    with pytest.raises(HumanRuntimeError, match="2/38-key.*59,200-tick"):
        _retime_walking_loop_to_profile(
            _two_action_glb(second_walking_times=second),
            walking_profile_sample_count=19,
        )


@pytest.mark.parametrize("stem", ["", "human/1", "human-1", 1, True])
def test_runtime_package_stem_must_be_a_safe_identifier(stem: object) -> None:
    with pytest.raises(HumanRuntimeError, match="package_stem"):
        prepare_rocketbox_habitat_runtime(
            "missing.glb",
            "output",
            package_stem=stem,  # type: ignore[arg-type]
        )
