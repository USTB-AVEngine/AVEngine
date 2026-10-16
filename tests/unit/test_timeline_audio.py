from __future__ import annotations

import numpy as np
import pytest

from avengine.spatial_audio.audio import AudioContractError
from avengine.timeline.audio import (
    extract_faded_clip,
    place_simultaneous_events,
    raised_cosine_partition,
    render_dynamic_stems_and_mix,
    time_varying_convolve,
)


def test_raised_cosine_partition_is_exact_and_continuous() -> None:
    weights = raised_cosine_partition((0, 7, 13, 19), 25)
    assert weights.shape == (4, 25)
    assert np.max(np.abs(weights.sum(axis=0) - 1.0)) < 2.0e-15
    assert weights[0, 0] == 1.0
    assert weights[-1, -1] == 1.0
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


def test_identical_rirs_reduce_to_one_static_full_convolution() -> None:
    rng = np.random.default_rng(7)
    dry = rng.normal(size=40)
    ir = rng.normal(size=(2, 9))
    rirs = np.repeat(ir[None, :, :], 4, axis=0)
    result = time_varying_convolve(
        dry,
        rirs,
        (0, 11, 23, 34),
        output_sample_count=40,
    )
    expected = np.stack([np.convolve(dry, channel) for channel in ir])
    assert result.full_tail.shape == expected.shape
    assert np.allclose(result.full_tail, expected, rtol=0.0, atol=2.0e-12)
    assert np.array_equal(result.episode, result.full_tail[:, :40])


def test_simultaneous_events_and_counterfactual_routing() -> None:
    source = np.linspace(-1.0, 1.0, 100)
    left = extract_faded_clip(source, start_sample=10, end_sample=30, fade_samples=2)
    right = extract_faded_clip(-source, start_sample=10, end_sample=30, fade_samples=2)
    buses_a, events_a = place_simultaneous_events(
        {"beagle": left, "golden": right},
        {"source0": "beagle", "source1": "golden"},
        start_samples=(5, 35, 65),
        output_sample_count=100,
        linear_gain=0.25,
    )
    buses_b, events_b = place_simultaneous_events(
        {"beagle": left, "golden": right},
        {"source0": "golden", "source1": "beagle"},
        start_samples=(5, 35, 65),
        output_sample_count=100,
        linear_gain=0.25,
    )
    assert [item["start_sample"] for item in events_a] == [5, 5, 35, 35, 65, 65]
    assert [item["simultaneous_group_id"] for item in events_a] == [
        "simultaneous0",
        "simultaneous0",
        "simultaneous1",
        "simultaneous1",
        "simultaneous2",
        "simultaneous2",
    ]
    assert np.array_equal(buses_a["source0"], buses_b["source1"])
    assert np.array_equal(buses_a["source1"], buses_b["source0"])
    assert {item["source_id"] for item in events_b} == {"source0", "source1"}


def test_dynamic_named_stems_preserve_exact_episode_length() -> None:
    dry = {
        "source0": np.pad(np.ones(8), (0, 24)),
        "source1": np.pad(-np.ones(8), (8, 16)),
    }
    # [K,S,C,L]
    rirs = np.zeros((2, 2, 2, 3), dtype=np.float64)
    rirs[:, 0, 0, 0] = 1.0
    rirs[:, 0, 1, 1] = 0.5
    rirs[:, 1, 0, 1] = 0.5
    rirs[:, 1, 1, 0] = 1.0
    stems, mix = render_dynamic_stems_and_mix(
        dry,
        rirs,
        np.full((2, 2), 3, dtype=np.uint32),
        source_ids=("source0", "source1"),
        keyframe_samples=(0, 16),
        output_sample_count=32,
    )
    assert mix.shape == (2, 32)
    assert all(value.episode.shape == (2, 32) for value in stems.values())


def test_event_windows_must_fit() -> None:
    with pytest.raises(AudioContractError, match="escapes"):
        place_simultaneous_events(
            {"a": np.ones(10)},
            {"source0": "a", "source1": "a"},
            start_samples=(95,),
            output_sample_count=100,
        )
