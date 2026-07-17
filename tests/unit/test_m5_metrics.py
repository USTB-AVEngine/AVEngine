from __future__ import annotations

import math

import numpy as np
import pytest

from avengine.m5.metrics import (
    M5MetricsError,
    estimate_itd_gcc_phat,
    listener_local_azimuth_deg,
    measure_binaural_mixture_diagnostic,
    measure_binaural_rir_frame_cues,
    measure_binaural_rir_sequence_cues,
    measure_binaural_wet_stem_cues,
)


SAMPLE_RATE_HZ = 16_000


def _phase_error(actual: float, expected: float) -> float:
    return abs(math.atan2(math.sin(actual - expected), math.cos(actual - expected)))


def _fractional_impulse(
    delay_samples: float,
    *,
    sample_count: int = 512,
    center_sample: int = 200,
    half_width: int = 20,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Finite Lanczos-windowed sinc used as a deterministic delay fixture."""

    indices = np.arange(sample_count, dtype=np.float64)
    offset = indices - (center_sample + delay_samples)
    result = np.sinc(offset) * np.sinc(offset / half_width)
    result[np.abs(offset) >= half_width] = 0.0
    result /= np.linalg.norm(result)
    return np.ascontiguousarray(amplitude * result)


def _wet_fractional_delay_fixture(
    delay_samples: float,
    *,
    left_amplitude: float = 2.0,
) -> np.ndarray:
    rng = np.random.default_rng(20260718)
    dry = np.zeros(4096, dtype=np.float64)
    dry[512:-512] = rng.standard_normal(4096 - 1024)
    right_ir = _fractional_impulse(
        0.0,
        sample_count=81,
        center_sample=40,
        half_width=20,
    )
    left_ir = _fractional_impulse(
        delay_samples,
        sample_count=81,
        center_sample=40,
        half_width=20,
        amplitude=left_amplitude,
    )
    left = np.convolve(dry, left_ir, mode="full")
    right = np.convolve(dry, right_ir, mode="full")
    return np.stack((left, right), axis=0)


def test_listener_local_azimuth_uses_negative_z_forward_and_positive_right() -> None:
    listener = [0.0, 0.0, 0.0]
    identity_wxyz = [1.0, 0.0, 0.0, 0.0]
    assert listener_local_azimuth_deg(
        [0.0, 0.0, -2.0], listener, identity_wxyz
    ) == pytest.approx(0.0)
    assert listener_local_azimuth_deg(
        [2.0, 0.0, 0.0], listener, identity_wxyz
    ) == pytest.approx(90.0)
    assert listener_local_azimuth_deg(
        [-2.0, 0.0, 0.0], listener, identity_wxyz
    ) == pytest.approx(-90.0)
    assert abs(
        listener_local_azimuth_deg(
            [0.0, 0.0, 2.0], listener, identity_wxyz
        )
    ) == pytest.approx(180.0)

    # +90 degree world_from_listener yaw turns local -Z toward world -X.
    half = math.sqrt(0.5)
    assert listener_local_azimuth_deg(
        [-2.0, 0.0, 0.0], listener, [half, 0.0, half, 0.0]
    ) == pytest.approx(0.0, abs=1.0e-12)


def test_listener_local_azimuth_fails_closed_on_invalid_pose() -> None:
    with pytest.raises(M5MetricsError, match="positions must differ"):
        listener_local_azimuth_deg(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        )
    with pytest.raises(M5MetricsError, match="unit normalized"):
        listener_local_azimuth_deg(
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        )


@pytest.mark.parametrize("delay_samples", [-7.4, -3.25, -0.4, 0.4, 3.25, 7.4])
def test_gcc_phat_fractional_delay_sign_and_subsample_accuracy(
    delay_samples: float,
) -> None:
    right = _fractional_impulse(0.0)
    left = _fractional_impulse(delay_samples)
    report = estimate_itd_gcc_phat(
        left,
        right,
        SAMPLE_RATE_HZ,
        max_itd_seconds=0.001,
        interpolation_factor=64,
    )
    assert report["sign_convention"] == "t_left_minus_t_right"
    assert report["itd_samples"] == pytest.approx(delay_samples, abs=0.035)
    assert report["itd_seconds"] == pytest.approx(
        delay_samples / SAMPLE_RATE_HZ,
        abs=0.035 / SAMPLE_RATE_HZ,
    )
    assert report["at_search_boundary"] is False


def test_binaural_rir_direct_window_reports_ild_ipd_and_positive_left_delay() -> None:
    delay_samples = 3.25
    right = _fractional_impulse(0.0)
    left = _fractional_impulse(delay_samples, amplitude=2.0)
    report = measure_binaural_rir_frame_cues(
        np.stack((left, right), axis=0),
        SAMPLE_RATE_HZ,
        direct_arrival_sample=200,
        pre_direct_ms=2.0,
        direct_window_ms=8.0,
        gcc_interpolation_factor=64,
    )

    assert report["analysis_scope"] == "per_source_rir_direct_window"
    assert report["sign_convention"]["itd"] == "t_left_minus_t_right"
    assert report["ild_db"] == pytest.approx(20.0 * math.log10(2.0), abs=0.03)
    assert report["itd"]["itd_samples"] == pytest.approx(delay_samples, abs=0.035)
    for frequency in (250, 500, 1000, 2000):
        expected = -2.0 * math.pi * frequency * delay_samples / SAMPLE_RATE_HZ
        actual = report["ipd_radians_by_frequency_hz"][str(frequency)]
        assert _phase_error(actual, expected) < 0.035


def test_binaural_rir_sequence_keeps_explicit_frame_indices() -> None:
    right = _fractional_impulse(0.0)
    first = np.stack((_fractional_impulse(-2.5), right), axis=0)
    second = np.stack((_fractional_impulse(2.5), right), axis=0)
    report = measure_binaural_rir_sequence_cues(
        np.stack((first, second), axis=0),
        SAMPLE_RATE_HZ,
        direct_arrival_samples=[200, 200],
        pre_direct_ms=2.0,
        direct_window_ms=8.0,
        gcc_interpolation_factor=64,
    )
    assert report["frame_count"] == 2
    assert [frame["frame_index"] for frame in report["frames"]] == [0, 1]
    assert report["frames"][0]["itd"]["itd_samples"] < 0.0
    assert report["frames"][1]["itd"]["itd_samples"] > 0.0


def test_wet_stem_stft_reports_source_specific_fractional_delay_cues() -> None:
    delay_samples = 3.5
    wet = _wet_fractional_delay_fixture(delay_samples)
    report = measure_binaural_wet_stem_cues(
        wet,
        SAMPLE_RATE_HZ,
        768,
        3328,
        source_id="source_moving",
        n_fft=256,
        hop_length=128,
        gcc_interpolation_factor=32,
        relative_energy_floor_db=-40.0,
    )

    assert report["source_id"] == "source_moving"
    assert report["diagnostic_only"] is False
    assert report["source_specific_acceptance_allowed"] is True
    assert report["summary"]["broadband_ild_db_median"] == pytest.approx(
        20.0 * math.log10(2.0), abs=0.08
    )
    assert report["summary"]["itd_samples_median"] == pytest.approx(
        delay_samples, abs=0.12
    )
    assert report["stft"]["measured_frame_count"] > 10
    for frequency in (250, 500, 1000, 2000):
        frequency_report = report["summary"]["by_frequency_hz"][str(frequency)]
        assert frequency_report["valid_frame_count"] > 0
        assert frequency_report["ild_db_median"] == pytest.approx(
            20.0 * math.log10(2.0), abs=0.35
        )
        expected = -2.0 * math.pi * frequency * delay_samples / SAMPLE_RATE_HZ
        assert _phase_error(
            frequency_report["ipd_circular_mean_radians"], expected
        ) < 0.2


def test_mixture_metrics_are_permanently_diagnostic_only() -> None:
    mixture = _wet_fractional_delay_fixture(-2.75, left_amplitude=0.75)
    report = measure_binaural_mixture_diagnostic(
        mixture,
        SAMPLE_RATE_HZ,
        768,
        3328,
        n_fft=256,
        hop_length=128,
        gcc_interpolation_factor=16,
    )
    assert report["analysis_scope"] == (
        "binaural_mixture_active_window_stft_diagnostic"
    )
    assert report["diagnostic_only"] is True
    assert report["source_specific_acceptance_allowed"] is False
    assert "not identifiable" in report["diagnostic_limitation"]
    assert "source_id" not in report


def test_wet_metrics_reject_implicit_or_too_short_active_windows() -> None:
    wet = _wet_fractional_delay_fixture(1.5)
    with pytest.raises(M5MetricsError, match="requires source_id"):
        measure_binaural_wet_stem_cues(
            wet,
            SAMPLE_RATE_HZ,
            768,
            3328,
            source_id="",
            n_fft=256,
        )
    with pytest.raises(M5MetricsError, match="at least one STFT frame"):
        measure_binaural_wet_stem_cues(
            wet,
            SAMPLE_RATE_HZ,
            100,
            200,
            source_id="source0",
            n_fft=256,
        )
