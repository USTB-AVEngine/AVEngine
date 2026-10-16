from __future__ import annotations

import numpy as np
import pytest

from avengine.acoustics.metrics import (
    AcousticMetricError,
    MetricConfig,
    analyze_ir,
)


def _synthetic_ir(*, decay_seconds: float, sample_rate_hz: int = 16_000) -> np.ndarray:
    """Return a deterministic two-channel IR with a known exponential tail."""

    sample_count = sample_rate_hz
    arrival = 160
    result = np.zeros((2, sample_count), dtype=np.float64)
    result[:, arrival] = (1.0, 0.92)
    tail_sample = np.arange(sample_count - arrival - 1, dtype=np.float64)
    envelope = 0.08 * np.exp(-tail_sample / (decay_seconds * sample_rate_hz))
    rng = np.random.default_rng(917)
    tail = envelope * rng.normal(size=tail_sample.size)
    result[0, arrival + 1 :] = tail
    result[1, arrival + 1 :] = 0.85 * tail
    return result


def test_expected_absorption_directions_are_measured_from_raw_ir() -> None:
    low_absorption = analyze_ir(_synthetic_ir(decay_seconds=0.24), 16_000)
    high_absorption = analyze_ir(_synthetic_ir(decay_seconds=0.045), 16_000)

    assert low_absorption.direct_arrival_sample == 160
    assert high_absorption.direct_arrival_sample == 160
    assert low_absorption.edt_seconds > high_absorption.edt_seconds
    assert low_absorption.drr_db < high_absorption.drr_db
    assert low_absorption.late_energy_ratio > high_absorption.late_energy_ratio
    assert low_absorption.edt_fit_r2 > 0.98
    assert high_absorption.edt_fit_r2 > 0.98
    assert low_absorption.edt_decay_span_db >= 10.0
    assert high_absorption.edt_decay_span_db >= 10.0


def test_channel_axis_is_explicit_and_layout_invariant() -> None:
    channel_major = _synthetic_ir(decay_seconds=0.12)

    first = analyze_ir(channel_major, 16_000, channel_axis=0)
    second = analyze_ir(channel_major.T, 16_000, channel_axis=1)

    assert first.to_dict() == second.to_dict()


def test_mono_ir_is_supported_without_layout_guessing() -> None:
    result = analyze_ir(_synthetic_ir(decay_seconds=0.12)[0], 16_000)

    assert result.channel_count == 1
    assert result.sample_count == 16_000
    assert result.direct_arrival_sample == 160


def test_edt_does_not_remove_a_dominant_direct_impulse_to_improve_fit() -> None:
    """A poor 0 to -10 dB EDT fit remains visible instead of becoming T10."""

    direct_dominated = _synthetic_ir(decay_seconds=0.045)
    direct_dominated[:, 161:] *= 0.3

    result = analyze_ir(direct_dominated, 16_000)

    assert result.edt_decay_span_db >= 10.0
    assert result.edt_fit_r2 < 0.9


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(16_000),
        np.full(16_000, np.nan),
        np.full(16_000, np.inf),
    ],
)
def test_silent_or_nonfinite_ir_fails_closed(value: np.ndarray) -> None:
    with pytest.raises(AcousticMetricError):
        analyze_ir(value, 16_000)


def test_ir_without_reverberant_tail_fails_closed() -> None:
    ir = np.zeros(16_000)
    ir[160] = 1.0

    with pytest.raises(AcousticMetricError, match="reverberant energy"):
        analyze_ir(ir, 16_000)


def test_short_ir_cannot_claim_a_late_energy_measurement() -> None:
    ir = np.zeros(1_000)
    ir[900] = 1.0
    ir[901:] = 0.01

    with pytest.raises(AcousticMetricError, match="too short"):
        analyze_ir(ir, 16_000)


@pytest.mark.parametrize("sample_rate", [0, -1, np.nan, np.inf, True, "bad"])
def test_invalid_sample_rate_fails_closed(sample_rate: object) -> None:
    with pytest.raises(AcousticMetricError, match="sample_rate_hz"):
        analyze_ir(_synthetic_ir(decay_seconds=0.12), sample_rate)  # type: ignore[arg-type]


def test_metric_configuration_rejects_ambiguous_boundaries() -> None:
    with pytest.raises(AcousticMetricError, match="later than"):
        analyze_ir(
            _synthetic_ir(decay_seconds=0.12),
            16_000,
            configuration=MetricConfig(direct_window_ms=80.0, late_start_ms=80.0),
        )


def test_shape_and_channel_axis_are_not_inferred() -> None:
    with pytest.raises(AcousticMetricError, match="shape"):
        analyze_ir(np.zeros((2, 3, 4)), 16_000)
    with pytest.raises(AcousticMetricError, match="channel_axis"):
        analyze_ir(np.zeros((2, 100)), 16_000, channel_axis=2)
