from __future__ import annotations

from pathlib import Path

import pytest

from avengine.episode_clock import EpisodeClock, EpisodeClockError


def test_clock_derives_exact_sample_count_and_boundaries() -> None:
    clock = EpisodeClock.from_values(
        frame_count=75, frame_rate_hz=15, sample_rate_hz=16_000
    )
    assert clock.to_dict() == {
        "frame_count": 75,
        "frame_rate_hz": 15.0,
        "sample_rate_hz": 16_000,
        "clip_seconds": 5.0,
        "sample_count": 80_000,
        "compatibility": "configured",
    }
    boundaries = clock.frame_boundaries()
    assert len(boundaries) == 76
    assert boundaries[0] == 0
    assert boundaries[-1] == 80_000
    # Per-frame rounding varies, but the total never drifts to 75*round(16000/15).
    assert sum(end - start for start, end in zip(boundaries, boundaries[1:])) == 80_000


def test_clock_supports_a_ten_second_150_frame_episode() -> None:
    clock = EpisodeClock.from_values(
        frame_count=150, frame_rate_hz=15, sample_rate_hz=16_000
    )
    assert clock.clip_seconds_float == 10.0
    assert clock.sample_count == 160_000
    assert clock.sample_boundary(149) == 158_933
    assert clock.sample_boundary(150) == 160_000


def test_clock_rejects_independent_duration_or_sample_count() -> None:
    with pytest.raises(EpisodeClockError, match="requires clip_seconds"):
        EpisodeClock.from_values(
            frame_count=150,
            frame_rate_hz=15,
            sample_rate_hz=16_000,
            clip_seconds=5,
        )
    with pytest.raises(EpisodeClockError, match="sample_count"):
        EpisodeClock.from_mapping(
            {
                "frame_count": 75,
                "frame_rate_hz": 15,
                "sample_rate_hz": 16_000,
                "clip_seconds": 5,
                "sample_count": 80_001,
            }
        )


def test_legacy_compatibility_is_explicit_in_serialized_clock() -> None:
    clock = EpisodeClock.from_values(
        frame_count=75,
        frame_rate_hz=15,
        sample_rate_hz=16_000,
        compatibility="legacy_inferred",
    )
    assert clock.to_dict()["compatibility"] == "legacy_inferred"



def test_clock_convolver_uses_the_full_aligned_window_without_hop_drift(
    monkeypatch, tmp_path
) -> None:
    # The renderer module imports optional Habitat bindings at module import.
    import importlib.util
    import sys
    import types
    import numpy as np

    quaternion = types.ModuleType("quaternion")
    habitat = types.ModuleType("habitat_sim")
    sensor = types.ModuleType("habitat_sim.sensor")
    sensor.AudioSensorSpec = object
    sensor.RLRAudioPropagationChannelLayoutType = types.SimpleNamespace()
    habitat.sensor = sensor
    for name, module in (
        ("quaternion", quaternion),
        ("habitat_sim", habitat),
        ("habitat_sim.sensor", sensor),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/audio/render_moving_source.py"
    )
    spec = importlib.util.spec_from_file_location("hm3d_audio_clock_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    clock = EpisodeClock.from_values(
        frame_count=75, frame_rate_hz=15, sample_rate_hz=16_000
    )
    responses = [
        np.ones((2, 4), dtype=float),
        np.ones((2, 4), dtype=float),
    ]
    wet = module.convolve_route_with_clock(
        responses, [0, 74], clock, seed=3, channels=2
    )
    assert wet.shape == (2, clock.sample_count + 4)
    assert wet[:, : clock.sample_count].shape[1] == 80_000


def test_clock_rejects_boolean_declared_sample_count() -> None:
    with pytest.raises(EpisodeClockError, match="positive integer"):
        EpisodeClock.from_mapping({
            "frame_count": 1,
            "frame_rate_hz": 1,
            "sample_rate_hz": 1,
            "sample_count": True,
        })
