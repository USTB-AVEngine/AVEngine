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
    summarize_lateral_cue_consistency,
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
        listener_local_azimuth_deg([0.0, 0.0, 2.0], listener, identity_wxyz)
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
        assert (
            _phase_error(frequency_report["ipd_circular_mean_radians"], expected) < 0.2
        )


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


def _semantic_frame(
    frame_index: int,
    azimuth_deg: float,
    ild_db: float,
    itd_seconds: float,
    *,
    at_search_boundary: bool = False,
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "listener_local_azimuth_deg": azimuth_deg,
        "ild_db": ild_db,
        "ipd_radians_by_frequency_hz": {"500": math.pi},
        "itd": {
            "itd_seconds": itd_seconds,
            "at_search_boundary": at_search_boundary,
        },
    }


def test_lateral_cue_summary_excludes_gcc_search_boundary_from_itd_gate() -> None:
    right = _fractional_impulse(0.0)
    left = _fractional_impulse(16.0, amplitude=0.5)
    cue = measure_binaural_rir_frame_cues(
        np.stack((left, right), axis=0),
        SAMPLE_RATE_HZ,
        direct_arrival_sample=200,
        pre_direct_ms=2.0,
        direct_window_ms=8.0,
        gcc_interpolation_factor=64,
    )
    assert cue["itd"]["at_search_boundary"] is True
    assert cue["itd"]["formal_acceptance_allowed"] is False

    cue["frame_index"] = 0
    cue["listener_local_azimuth_deg"] = 30.0
    summary = summarize_lateral_cue_consistency(
        [
            cue,
            _semantic_frame(1, 30.0, -4.0, 0.0002),
            _semantic_frame(2, -30.0, 4.0, -0.0002),
        ]
    )
    assert summary["status"] == "pass"
    assert summary["formal_acceptance_allowed"] is True
    assert summary["counts"]["gcc_boundary_rejected_frames"] == 1
    assert summary["frames"][0]["classification"] == "gcc_boundary_rejected"
    assert summary["frames"][0]["itd_vote"] == "rejected_search_boundary"
    assert summary["frames"][0]["accepted_cue_count"] == 1
    assert (
        summary["aggregate_by_side"]["right"]["gcc_boundary_rejected_frame_count"] == 1
    )
    assert summary["aggregate_by_side"]["right"]["rates"][
        "cue_coverage_rate"
    ] == pytest.approx(0.75)


def test_lateral_cue_summary_detects_left_right_reversal() -> None:
    consistent = [
        _semantic_frame(0, 30.0, -4.0, 0.0002),
        _semantic_frame(1, -30.0, 4.0, -0.0002),
    ]
    passed = summarize_lateral_cue_consistency(consistent)
    assert passed["status"] == "pass"
    assert passed["rates"]["ild_sign_consistency_rate"] == 1.0
    assert passed["rates"]["itd_sign_consistency_rate"] == 1.0
    assert passed["lateral_separation"]["ild_left_minus_right_db"] == 8.0
    assert passed["lateral_separation"]["itd_right_minus_left_seconds"] == (
        pytest.approx(0.0004)
    )
    assert "sign_gate" in passed["ipd_role"]

    reversed_cues = [
        _semantic_frame(0, 30.0, 4.0, -0.0002),
        _semantic_frame(1, -30.0, -4.0, 0.0002),
    ]
    failed = summarize_lateral_cue_consistency(reversed_cues)
    assert failed["status"] == "fail"
    assert failed["rates"]["ild_sign_consistency_rate"] == 0.0
    assert failed["rates"]["itd_sign_consistency_rate"] == 0.0
    assert all(frame["classification"] == "inconsistent" for frame in failed["frames"])


def test_lateral_cue_summary_exempts_near_median_but_not_all_zero() -> None:
    report = summarize_lateral_cue_consistency(
        [
            _semantic_frame(0, 0.5, 0.0, 0.0),
            _semantic_frame(1, 25.0, -3.0, 0.00015),
            _semantic_frame(2, -25.0, 3.0, -0.00015),
        ]
    )
    assert report["status"] == "pass"
    assert report["counts"]["median_plane_exempt_frames"] == 1
    assert report["frames"][0]["classification"] == "median_plane_exempt"

    all_zero = summarize_lateral_cue_consistency(
        [
            _semantic_frame(0, 25.0, 0.0, 0.0),
            _semantic_frame(1, -25.0, 0.0, 0.0),
        ]
    )
    assert all_zero["status"] == "fail"
    assert all_zero["rates"]["combined_sign_consistency_rate"] is None
    assert "right_has_no_non_ambiguous_ild_vote" in all_zero["rejection_reasons"]
    assert "left_has_no_non_ambiguous_itd_vote" in all_zero["rejection_reasons"]
    assert "right_median_ild_not_negative_nonzero" in all_zero["rejection_reasons"]
