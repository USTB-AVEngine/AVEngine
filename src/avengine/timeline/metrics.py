"""Pure-NumPy spatial metrics for M5 dynamic binaural evidence.

The public sign convention is deliberately explicit and shared by every
function in this module:

* channels are ``[left, right]``;
* ILD is ``10*log10(E_left/E_right)``; and
* ITD is ``t_left - t_right``.

Consequently, a source on the listener's right normally has negative ILD and
positive ITD.  Per-source RIRs and wet stems can support source-specific
checks.  A mixture cannot separate overlapping sources, so its entry point is
permanently labelled diagnostic-only.

No SciPy operation or implicit resampling is used.  GCC-PHAT first finds the
integer peak with an FFT correlation and then evaluates the same PHAT spectrum
on a dense one-sample neighbourhood to obtain a deterministic sub-sample
estimate.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_IPD_FREQUENCIES_HZ: tuple[float, ...] = (
    250.0,
    500.0,
    1000.0,
    2000.0,
)
DEFAULT_MAX_ITD_SECONDS = 0.001
_ENERGY_EPSILON = np.finfo(np.float64).tiny
_PHAT_RELATIVE_FLOOR = 1.0e-12

DEFAULT_MINIMUM_LATERAL_ANGLE_DEG = 3.0
DEFAULT_MINIMUM_ABSOLUTE_ILD_DB = 0.5
DEFAULT_MINIMUM_ABSOLUTE_ITD_SECONDS = 5.0e-6
DEFAULT_MINIMUM_CUE_CONSISTENCY_RATE = 0.51
DEFAULT_MINIMUM_CUE_COVERAGE_RATE = 0.5


class M5MetricsError(ValueError):
    """A spatial-metric input violates the explicit M5 contract."""


def _finite_vector3(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M5MetricsError(f"{owner} must contain three finite numbers") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise M5MetricsError(f"{owner} must contain three finite numbers")
    return result


def _unit_quaternion_wxyz(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M5MetricsError(
            f"{owner} must contain four finite wxyz components"
        ) from exc
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise M5MetricsError(f"{owner} must contain four finite wxyz components")
    norm = float(np.linalg.norm(result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise M5MetricsError(f"{owner} must already be unit normalized")
    return result


def listener_local_source_geometry(
    source_position_m: Any,
    listener_position_m: Any,
    listener_orientation_wxyz: Any,
) -> dict[str, Any]:
    """Return exact listener-local direction, distance, azimuth and elevation.

    ``listener_orientation_wxyz`` is ``world_from_listener``.  AVEngine's
    listener-local frame is right handed: right is ``+X``, up is ``+Y`` and
    forward is ``-Z``.  Azimuth is positive toward the right in
    ``[-180, 180]`` and elevation is positive upward in ``[-90, 90]``.
    """

    source = _finite_vector3(source_position_m, owner="source_position_m")
    listener = _finite_vector3(listener_position_m, owner="listener_position_m")
    quaternion = _unit_quaternion_wxyz(
        listener_orientation_wxyz,
        owner="listener_orientation_wxyz",
    )
    world_direction = source - listener
    distance = float(np.linalg.norm(world_direction))
    if distance <= 0.0:
        raise M5MetricsError("source and listener positions must differ")

    # Rotate by the inverse world_from_listener quaternion.  For a unit
    # quaternion this is its conjugate.  The compact cross-product expression
    # avoids constructing a matrix and keeps the convention visible.
    w = float(quaternion[0])
    inverse_vector = -quaternion[1:]
    uv = np.cross(inverse_vector, world_direction)
    uuv = np.cross(inverse_vector, uv)
    local = world_direction + 2.0 * (w * uv + uuv)
    azimuth = math.degrees(math.atan2(float(local[0]), -float(local[2])))
    if math.isclose(azimuth, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        azimuth = 0.0
    if azimuth > 180.0:
        azimuth -= 360.0
    horizontal_distance = math.hypot(float(local[0]), float(local[2]))
    elevation = math.degrees(math.atan2(float(local[1]), horizontal_distance))
    if math.isclose(elevation, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        elevation = 0.0
    return {
        "coordinate_frame": "listener_x_right_y_up_negative_z_forward",
        "unit_direction_xyz": (local / distance).tolist(),
        "distance_m": distance,
        "azimuth_deg": float(azimuth),
        "elevation_deg": float(elevation),
    }


def listener_local_azimuth_deg(
    source_position_m: Any,
    listener_position_m: Any,
    listener_orientation_wxyz: Any,
) -> float:
    """Return listener-local azimuth in degrees, positive toward the right."""

    return float(
        listener_local_source_geometry(
            source_position_m,
            listener_position_m,
            listener_orientation_wxyz,
        )["azimuth_deg"]
    )


def _positive_sample_rate(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise M5MetricsError("sample_rate_hz must be a positive integer")
    return int(value)


def _finite_positive(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise M5MetricsError(f"{owner} must be finite and positive")
    return float(value)


def _positive_integer(value: Any, *, owner: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise M5MetricsError(f"{owner} must be a positive integer")
    return int(value)


def _binaural_channel_major(
    value: Any,
    *,
    owner: str,
    channel_axis: int,
    minimum_samples: int = 2,
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise M5MetricsError(f"{owner} must contain real numeric samples")
    if source.ndim != 2:
        raise M5MetricsError(f"{owner} must have two dimensions")
    if channel_axis in (0, -2):
        oriented = source
    elif channel_axis in (1, -1):
        oriented = source.T
    else:
        raise M5MetricsError("channel_axis must explicitly identify axis 0 or 1")
    if oriented.shape[0] != 2 or oriented.shape[1] < minimum_samples:
        raise M5MetricsError(
            f"{owner} must have shape [2, samples] in [left, right] order"
        )
    try:
        result = np.ascontiguousarray(oriented, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M5MetricsError(f"{owner} cannot be represented as float64") from exc
    if not np.all(np.isfinite(result)):
        raise M5MetricsError(f"{owner} must contain only finite samples")
    energies = np.sum(np.square(result), axis=1, dtype=np.float64)
    if np.any(energies <= 0.0) or not np.all(np.isfinite(energies)):
        raise M5MetricsError(f"{owner} must contain positive energy in both ears")
    return result


def _ipd_frequencies(
    values: Iterable[Any], *, sample_rate_hz: int
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise M5MetricsError("ipd_frequencies_hz must be a sequence")
    frequencies: list[float] = []
    for index, value in enumerate(values):
        frequency = _finite_positive(value, owner=f"ipd_frequencies_hz[{index}]")
        if frequency >= sample_rate_hz / 2.0:
            raise M5MetricsError("every IPD frequency must be strictly below Nyquist")
        frequencies.append(frequency)
    if not frequencies or len(set(frequencies)) != len(frequencies):
        raise M5MetricsError("IPD frequencies must be non-empty and unique")
    return tuple(frequencies)


def _frequency_key(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, ".12g")


def _wrap_phase(value: float) -> float:
    wrapped = math.atan2(math.sin(value), math.cos(value))
    return 0.0 if wrapped == 0.0 else float(wrapped)


def _ild_db(channels: np.ndarray) -> float:
    energies = np.sum(np.square(channels), axis=1, dtype=np.float64)
    if np.any(energies <= 0.0) or not np.all(np.isfinite(energies)):
        raise M5MetricsError("ILD requires finite positive energy in both ears")
    return float(10.0 * math.log10(float(energies[0] / energies[1])))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def estimate_itd_gcc_phat(
    left_samples: Any,
    right_samples: Any,
    sample_rate_hz: int,
    *,
    max_itd_seconds: float = DEFAULT_MAX_ITD_SECONDS,
    interpolation_factor: int = 32,
) -> dict[str, Any]:
    """Estimate ``t_left - t_right`` with bounded, sub-sample GCC-PHAT.

    Positive output means that the left channel arrives later.  The search is
    clamped to ``+/-max_itd_seconds`` (one millisecond by default).  The
    returned boundary flag lets a formal caller reject an estimate whose true
    peak may lie outside that physical search range.
    """

    rate = _positive_sample_rate(sample_rate_hz)
    maximum_seconds = _finite_positive(max_itd_seconds, owner="max_itd_seconds")
    interpolation = _positive_integer(
        interpolation_factor, owner="interpolation_factor"
    )
    if interpolation > 256:
        raise M5MetricsError("interpolation_factor must not exceed 256")

    try:
        left = np.ascontiguousarray(left_samples, dtype=np.float64)
        right = np.ascontiguousarray(right_samples, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M5MetricsError("GCC-PHAT inputs must be real numeric samples") from exc
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or left.size < 2
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise M5MetricsError(
            "GCC-PHAT inputs must be equal-length finite one-dimensional arrays"
        )
    if float(np.dot(left, left)) <= 0.0 or float(np.dot(right, right)) <= 0.0:
        raise M5MetricsError("GCC-PHAT inputs must contain positive energy")

    maximum_lag = int(math.ceil(maximum_seconds * rate))
    if maximum_lag < 1:
        raise M5MetricsError("max_itd_seconds must span at least one sample")
    maximum_lag = min(maximum_lag, left.size - 1)
    n_fft = _next_power_of_two(2 * left.size - 1)
    left_spectrum = np.fft.fft(left, n=n_fft)
    right_spectrum = np.fft.fft(right, n=n_fft)
    cross_spectrum = left_spectrum * np.conjugate(right_spectrum)
    magnitude = np.abs(cross_spectrum)
    maximum_magnitude = float(np.max(magnitude))
    if not math.isfinite(maximum_magnitude) or maximum_magnitude <= 0.0:
        raise M5MetricsError("GCC-PHAT cross spectrum has no usable energy")
    valid = magnitude > maximum_magnitude * _PHAT_RELATIVE_FLOOR
    phat = np.zeros_like(cross_spectrum)
    phat[valid] = cross_spectrum[valid] / magnitude[valid]

    correlation = np.fft.ifft(phat).real
    bounded = np.concatenate(
        (correlation[-maximum_lag:], correlation[: maximum_lag + 1])
    )
    integer_offset = int(np.argmax(np.abs(bounded)))
    integer_lag = integer_offset - maximum_lag

    # Evaluate the periodic PHAT correlation at fractional lags within one
    # sample of the integer maximum.  Signed FFT frequencies are essential at
    # non-integer lags; raw DFT indices would add an erroneous phase turn to
    # the negative-frequency half.
    low = max(-float(maximum_lag), float(integer_lag) - 1.0)
    high = min(float(maximum_lag), float(integer_lag) + 1.0)
    candidate_count = max(2, int(round((high - low) * interpolation)) + 1)
    candidate_lags = np.linspace(low, high, candidate_count, dtype=np.float64)
    frequencies = np.fft.fftfreq(n_fft)
    candidate_values = np.empty(candidate_count, dtype=np.float64)
    for index, lag in enumerate(candidate_lags):
        phase = np.exp(2.0j * math.pi * frequencies * float(lag))
        candidate_values[index] = float(np.real(np.sum(phat * phase)) / n_fft)
    peak_index = int(np.argmax(np.abs(candidate_values)))
    lag_samples = float(candidate_lags[peak_index])

    # A three-point parabolic refinement removes most of the dense-grid
    # quantization while remaining deterministic and dependency-free.
    if 0 < peak_index < candidate_count - 1:
        magnitudes = np.abs(candidate_values)
        before = float(magnitudes[peak_index - 1])
        peak = float(magnitudes[peak_index])
        after = float(magnitudes[peak_index + 1])
        denominator = before - 2.0 * peak + after
        if abs(denominator) > np.finfo(np.float64).eps:
            step = float(candidate_lags[1] - candidate_lags[0])
            lag_samples += 0.5 * (before - after) / denominator * step
    lag_samples = float(np.clip(lag_samples, -maximum_lag, maximum_lag))

    peak_magnitude = float(abs(candidate_values[peak_index]))
    comparison = np.abs(candidate_values).copy()
    comparison[max(0, peak_index - 1) : min(candidate_count, peak_index + 2)] = 0.0
    second_magnitude = float(np.max(comparison)) if comparison.size > 3 else 0.0
    peak_ratio = peak_magnitude / second_magnitude if second_magnitude > 0.0 else None
    boundary_tolerance = 1.0 / interpolation
    at_search_boundary = bool(abs(lag_samples) >= maximum_lag - boundary_tolerance)
    return {
        "method": "gcc_phat_dense_subsample_v1",
        "sign_convention": "t_left_minus_t_right",
        "itd_seconds": lag_samples / rate,
        "itd_samples": lag_samples,
        "integer_peak_lag_samples": integer_lag,
        "max_itd_seconds": maximum_seconds,
        "max_itd_samples": maximum_lag,
        "interpolation_factor": interpolation,
        "absolute_peak": peak_magnitude,
        "peak_to_second_ratio": peak_ratio,
        "at_search_boundary": at_search_boundary,
        "formal_acceptance_allowed": not at_search_boundary,
    }


def _finite_nonnegative(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise M5MetricsError(f"{owner} must be finite and non-negative")
    return float(value)


def _unit_interval(value: Any, *, owner: str) -> float:
    result = _finite_nonnegative(value, owner=owner)
    if result > 1.0:
        raise M5MetricsError(f"{owner} must not exceed one")
    return result


def _signed_cue_vote(
    value: float,
    *,
    expected_sign: int,
    ambiguity_threshold: float,
) -> str:
    if abs(value) < ambiguity_threshold:
        return "ambiguous_exempt"
    actual_sign = 1 if value > 0.0 else -1
    return "consistent" if actual_sign == expected_sign else "inconsistent"


def _consistency_rate(consistent: int, inconsistent: int) -> float | None:
    measured = consistent + inconsistent
    return float(consistent / measured) if measured else None


def summarize_lateral_cue_consistency(
    frame_reports: Sequence[Mapping[str, Any]],
    *,
    listener_local_azimuths_deg: Sequence[float] | None = None,
    minimum_lateral_angle_deg: float = DEFAULT_MINIMUM_LATERAL_ANGLE_DEG,
    minimum_absolute_ild_db: float = DEFAULT_MINIMUM_ABSOLUTE_ILD_DB,
    minimum_absolute_itd_seconds: float = DEFAULT_MINIMUM_ABSOLUTE_ITD_SECONDS,
    minimum_consistency_rate: float = DEFAULT_MINIMUM_CUE_CONSISTENCY_RATE,
    minimum_cue_coverage_rate: float = DEFAULT_MINIMUM_CUE_COVERAGE_RATE,
) -> dict[str, Any]:
    """Summarize whether a two-sided ILD/ITD bundle agrees with azimuth.

    Listener-local azimuth is positive to the right.  A right-side source
    therefore expects negative ILD and positive ITD; a left-side source
    expects the inverse signs.  Formal acceptance requires both sides, signed
    non-zero ILD side medians, present raw ITD side medians without an explicit
    wrong sign at or beyond the ambiguity threshold, left/right median
    separation, per-side cue coverage, and per-side sign agreement above
    chance.  Per-frame rates remain visible so a barely-above-chance side
    cannot be mistaken for strong evidence.

    Frames close to the front/back median plane and cue magnitudes below the
    explicit ambiguity thresholds do not cast sign votes.  GCC-PHAT estimates
    at the bounded search edge are likewise excluded from accepted ITD cues;
    their lost coverage remains visible and can fail the coverage threshold.
    A raw side ITD median inside the same ambiguity band is not independently
    rejected or recomputed after filtering; the frame votes, coverage, and
    left/right separation remain authoritative evidence around that median.
    IPD is deliberately excluded from the sign gate because it is circular and
    frequency-dependent; retained IPD values are diagnostic only.
    """

    if not isinstance(frame_reports, Sequence) or isinstance(
        frame_reports, (str, bytes)
    ):
        raise M5MetricsError("frame_reports must be a non-empty sequence")
    if not frame_reports:
        raise M5MetricsError("frame_reports must be a non-empty sequence")
    if listener_local_azimuths_deg is not None:
        if isinstance(listener_local_azimuths_deg, (str, bytes)) or len(
            listener_local_azimuths_deg
        ) != len(frame_reports):
            raise M5MetricsError(
                "listener_local_azimuths_deg must match the frame count"
            )

    lateral_threshold = _finite_positive(
        minimum_lateral_angle_deg, owner="minimum_lateral_angle_deg"
    )
    if lateral_threshold >= 90.0:
        raise M5MetricsError("minimum_lateral_angle_deg must be below 90")
    ild_threshold = _finite_positive(
        minimum_absolute_ild_db, owner="minimum_absolute_ild_db"
    )
    itd_threshold = _finite_positive(
        minimum_absolute_itd_seconds,
        owner="minimum_absolute_itd_seconds",
    )
    consistency_threshold = _unit_interval(
        minimum_consistency_rate, owner="minimum_consistency_rate"
    )
    if consistency_threshold <= 0.5:
        raise M5MetricsError("minimum_consistency_rate must exceed one half")
    coverage_threshold = _unit_interval(
        minimum_cue_coverage_rate, owner="minimum_cue_coverage_rate"
    )

    counts: dict[str, int] = {
        "total_frames": len(frame_reports),
        "median_plane_exempt_frames": 0,
        "lateral_frames": 0,
        "gcc_boundary_rejected_frames": 0,
        "ild_consistent_votes": 0,
        "ild_inconsistent_votes": 0,
        "ild_ambiguous_exempt_votes": 0,
        "itd_consistent_votes": 0,
        "itd_inconsistent_votes": 0,
        "itd_ambiguous_exempt_votes": 0,
    }
    side_counts: dict[str, dict[str, int]] = {
        side: {
            "lateral_frames": 0,
            "gcc_boundary_rejected_frames": 0,
            "ild_consistent_votes": 0,
            "ild_inconsistent_votes": 0,
            "ild_ambiguous_exempt_votes": 0,
            "itd_consistent_votes": 0,
            "itd_inconsistent_votes": 0,
            "itd_ambiguous_exempt_votes": 0,
        }
        for side in ("left", "right")
    }
    side_values: dict[str, dict[str, list[float]]] = {
        side: {"ild_db": [], "itd_seconds": []} for side in ("left", "right")
    }
    observed_sides: set[str] = set()
    frames: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(frame_reports):
        if not isinstance(frame, Mapping):
            raise M5MetricsError(f"frame_reports[{ordinal}] must be a mapping")
        if listener_local_azimuths_deg is None:
            azimuth_value = frame.get("listener_local_azimuth_deg")
        else:
            azimuth_value = listener_local_azimuths_deg[ordinal]
        if (
            isinstance(azimuth_value, bool)
            or not isinstance(azimuth_value, Real)
            or not math.isfinite(float(azimuth_value))
            or not -180.0 <= float(azimuth_value) <= 180.0
        ):
            raise M5MetricsError(
                f"frame_reports[{ordinal}] listener-local azimuth is invalid"
            )
        azimuth = float(azimuth_value)
        lateral_angle = min(abs(azimuth), 180.0 - abs(azimuth))
        frame_index = frame.get("frame_index", frame.get("stft_frame_index", ordinal))
        if isinstance(frame_index, bool) or not isinstance(
            frame_index, (int, np.integer)
        ):
            raise M5MetricsError(f"frame_reports[{ordinal}] frame index is invalid")

        if lateral_angle < lateral_threshold:
            boundary = bool(
                isinstance(frame.get("itd"), Mapping)
                and frame["itd"].get("at_search_boundary") is True
            )
            counts["median_plane_exempt_frames"] += 1
            if boundary:
                counts["gcc_boundary_rejected_frames"] += 1
            frames.append(
                {
                    "frame_index": int(frame_index),
                    "listener_local_azimuth_deg": azimuth,
                    "lateral_angle_deg": lateral_angle,
                    "side": "median_plane",
                    "classification": "median_plane_exempt",
                    "accepted_cue_count": 0,
                    "ild_vote": "not_evaluated",
                    "itd_vote": (
                        "rejected_search_boundary" if boundary else "not_evaluated"
                    ),
                    "gcc_at_search_boundary": boundary,
                }
            )
            continue

        side = "right" if azimuth > 0.0 else "left"
        observed_sides.add(side)
        counts["lateral_frames"] += 1
        side_counts[side]["lateral_frames"] += 1
        expected_ild_sign = -1 if side == "right" else 1
        expected_itd_sign = 1 if side == "right" else -1

        ild_value = frame.get("ild_db", frame.get("broadband_ild_db"))
        itd_report = frame.get("itd")
        if (
            isinstance(ild_value, bool)
            or not isinstance(ild_value, Real)
            or not math.isfinite(float(ild_value))
        ):
            raise M5MetricsError(f"frame_reports[{ordinal}] ILD is invalid")
        if not isinstance(itd_report, Mapping):
            raise M5MetricsError(f"frame_reports[{ordinal}] ITD report is invalid")
        itd_value = itd_report.get("itd_seconds")
        if (
            isinstance(itd_value, bool)
            or not isinstance(itd_value, Real)
            or not math.isfinite(float(itd_value))
        ):
            raise M5MetricsError(f"frame_reports[{ordinal}] ITD value is invalid")

        ild_vote = _signed_cue_vote(
            float(ild_value),
            expected_sign=expected_ild_sign,
            ambiguity_threshold=ild_threshold,
        )
        boundary = itd_report.get("at_search_boundary") is True
        if boundary:
            itd_vote = "rejected_search_boundary"
            counts["gcc_boundary_rejected_frames"] += 1
            side_counts[side]["gcc_boundary_rejected_frames"] += 1
        else:
            itd_vote = _signed_cue_vote(
                float(itd_value),
                expected_sign=expected_itd_sign,
                ambiguity_threshold=itd_threshold,
            )

        counts[f"ild_{ild_vote}_votes"] += 1
        side_counts[side][f"ild_{ild_vote}_votes"] += 1
        if itd_vote != "rejected_search_boundary":
            counts[f"itd_{itd_vote}_votes"] += 1
            side_counts[side][f"itd_{itd_vote}_votes"] += 1
        side_values[side]["ild_db"].append(float(ild_value))
        if not boundary:
            side_values[side]["itd_seconds"].append(float(itd_value))
        cue_votes = (ild_vote, itd_vote)
        if boundary:
            classification = "gcc_boundary_rejected"
        elif "inconsistent" in cue_votes:
            classification = "inconsistent"
        elif cue_votes == ("ambiguous_exempt", "ambiguous_exempt"):
            classification = "ambiguous_exempt"
        else:
            classification = "consistent"
        frames.append(
            {
                "frame_index": int(frame_index),
                "listener_local_azimuth_deg": azimuth,
                "lateral_angle_deg": lateral_angle,
                "side": side,
                "classification": classification,
                "accepted_cue_count": sum(
                    vote == "consistent" or vote == "inconsistent" for vote in cue_votes
                ),
                "expected_ild_sign": expected_ild_sign,
                "expected_itd_sign": expected_itd_sign,
                "ild_db": float(ild_value),
                "itd_seconds": float(itd_value),
                "ild_vote": ild_vote,
                "itd_vote": itd_vote,
                "gcc_at_search_boundary": boundary,
            }
        )

    def rates_for(cue_counts: Mapping[str, int]) -> dict[str, float | None]:
        ild_consistent = cue_counts["ild_consistent_votes"]
        ild_inconsistent = cue_counts["ild_inconsistent_votes"]
        itd_consistent = cue_counts["itd_consistent_votes"]
        itd_inconsistent = cue_counts["itd_inconsistent_votes"]
        all_consistent = ild_consistent + itd_consistent
        all_inconsistent = ild_inconsistent + itd_inconsistent
        possible_cues = 2 * cue_counts["lateral_frames"]
        measured_cues = all_consistent + all_inconsistent
        return {
            "ild_sign_consistency_rate": _consistency_rate(
                ild_consistent, ild_inconsistent
            ),
            "itd_sign_consistency_rate": _consistency_rate(
                itd_consistent, itd_inconsistent
            ),
            "combined_sign_consistency_rate": _consistency_rate(
                all_consistent, all_inconsistent
            ),
            "cue_coverage_rate": (
                float(measured_cues / possible_cues) if possible_cues else 0.0
            ),
        }

    rates = rates_for(counts)
    rates_by_side = {side: rates_for(side_counts[side]) for side in ("left", "right")}
    aggregate_by_side: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        ild_values = side_values[side]["ild_db"]
        itd_values = side_values[side]["itd_seconds"]
        aggregate_by_side[side] = {
            "lateral_frame_count": side_counts[side]["lateral_frames"],
            "ild_db_median": (float(np.median(ild_values)) if ild_values else None),
            "itd_seconds_median": (
                float(np.median(itd_values)) if itd_values else None
            ),
            "gcc_boundary_rejected_frame_count": side_counts[side][
                "gcc_boundary_rejected_frames"
            ],
            "rates": rates_by_side[side],
        }

    left_ild = aggregate_by_side["left"]["ild_db_median"]
    right_ild = aggregate_by_side["right"]["ild_db_median"]
    left_itd = aggregate_by_side["left"]["itd_seconds_median"]
    right_itd = aggregate_by_side["right"]["itd_seconds_median"]
    lateral_separation = {
        "ild_left_minus_right_db": (
            float(left_ild - right_ild)
            if left_ild is not None and right_ild is not None
            else None
        ),
        "itd_right_minus_left_seconds": (
            float(right_itd - left_itd)
            if left_itd is not None and right_itd is not None
            else None
        ),
    }

    rejection_reasons: list[str] = []
    if not counts["lateral_frames"]:
        rejection_reasons.append("no_lateral_frame_outside_median_plane_exemption")
    if observed_sides != {"left", "right"}:
        rejection_reasons.append("both_left_and_right_lateral_sides_required")
    for side in ("left", "right"):
        side_rates = rates_by_side[side]
        for cue in ("ild", "itd"):
            rate = side_rates[f"{cue}_sign_consistency_rate"]
            if rate is None:
                rejection_reasons.append(f"{side}_has_no_non_ambiguous_{cue}_vote")
            elif rate < consistency_threshold:
                rejection_reasons.append(
                    f"{side}_{cue}_sign_consistency_below_threshold"
                )
        if side_rates["cue_coverage_rate"] < coverage_threshold:
            rejection_reasons.append(f"{side}_cue_coverage_below_threshold")
    if rates["cue_coverage_rate"] < coverage_threshold:
        rejection_reasons.append("cue_coverage_below_threshold")

    if right_ild is None or right_ild > -ild_threshold:
        rejection_reasons.append("right_median_ild_not_negative_nonzero")
    if left_ild is None or left_ild < ild_threshold:
        rejection_reasons.append("left_median_ild_not_positive_nonzero")
    # Keep the reported raw medians unchanged.  A near-zero median is
    # ambiguous, not evidence of a channel reversal; only a missing median or
    # an explicit wrong sign at/beyond the configured ambiguity threshold is
    # a standalone side-median rejection.  Frame voting, coverage, and
    # left/right separation below still fail closed independently.
    if right_itd is None or right_itd <= -itd_threshold:
        rejection_reasons.append("right_median_itd_not_positive_nonzero")
    if left_itd is None or left_itd >= itd_threshold:
        rejection_reasons.append("left_median_itd_not_negative_nonzero")
    ild_separation = lateral_separation["ild_left_minus_right_db"]
    if ild_separation is None or ild_separation < 2.0 * ild_threshold:
        rejection_reasons.append("left_right_ild_median_separation_below_threshold")
    itd_separation = lateral_separation["itd_right_minus_left_seconds"]
    if itd_separation is None or itd_separation < 2.0 * itd_threshold:
        rejection_reasons.append("left_right_itd_median_separation_below_threshold")

    passed = not rejection_reasons
    return {
        "schema": "avengine_m5_binaural_lateral_cue_consistency_v1",
        "analysis_scope": "per_source_frame_cue_azimuth_semantics",
        "status": "pass" if passed else "fail",
        "formal_acceptance_allowed": passed,
        "sign_convention": {
            "listener_local_azimuth": "positive_right_degrees",
            "right_expected_ild": "negative",
            "right_expected_itd": "positive_t_left_minus_t_right",
            "left_expected_ild": "positive",
            "left_expected_itd": "negative_t_left_minus_t_right",
        },
        "ipd_role": ("frequency_dependent_circular_diagnostic_only_not_a_sign_gate"),
        "thresholds": {
            "minimum_lateral_angle_deg": lateral_threshold,
            "minimum_absolute_ild_db": ild_threshold,
            "minimum_absolute_itd_seconds": itd_threshold,
            "minimum_consistency_rate": consistency_threshold,
            "minimum_cue_coverage_rate": coverage_threshold,
        },
        "observed_lateral_sides": sorted(observed_sides),
        "counts": counts,
        "rates": rates,
        "aggregate_by_side": aggregate_by_side,
        "lateral_separation": lateral_separation,
        "rejection_reasons": rejection_reasons,
        "frames": frames,
    }


def _exact_frequency_ipd(
    channels: np.ndarray,
    sample_rate_hz: int,
    frequencies_hz: Sequence[float],
) -> dict[str, float]:
    sample_indices = np.arange(channels.shape[1], dtype=np.float64)
    result: dict[str, float] = {}
    for frequency in frequencies_hz:
        kernel = np.exp(-2.0j * math.pi * frequency * sample_indices / sample_rate_hz)
        left = np.sum(channels[0] * kernel)
        right = np.sum(channels[1] * kernel)
        cross = left * np.conjugate(right)
        if not (
            math.isfinite(float(abs(left)))
            and math.isfinite(float(abs(right)))
            and abs(left) > _ENERGY_EPSILON
            and abs(right) > _ENERGY_EPSILON
        ):
            raise M5MetricsError(
                f"direct window has no usable energy at {frequency:g} Hz"
            )
        result[_frequency_key(frequency)] = _wrap_phase(float(np.angle(cross)))
    return result


def measure_binaural_rir_frame_cues(
    impulse_response: Any,
    sample_rate_hz: int,
    *,
    direct_arrival_sample: int | None = None,
    pre_direct_ms: float = 1.0,
    direct_window_ms: float = 5.0,
    ipd_frequencies_hz: Sequence[float] = DEFAULT_IPD_FREQUENCIES_HZ,
    max_itd_seconds: float = DEFAULT_MAX_ITD_SECONDS,
    gcc_interpolation_factor: int = 32,
    channel_axis: int = 0,
) -> dict[str, Any]:
    """Measure one binaural RIR inside an explicit direct-arrival window."""

    rate = _positive_sample_rate(sample_rate_hz)
    channels = _binaural_channel_major(
        impulse_response,
        owner="impulse_response",
        channel_axis=channel_axis,
    )
    frequencies = _ipd_frequencies(ipd_frequencies_hz, sample_rate_hz=rate)
    pre_seconds = _finite_positive(pre_direct_ms, owner="pre_direct_ms") / 1000.0
    window_seconds = (
        _finite_positive(direct_window_ms, owner="direct_window_ms") / 1000.0
    )
    pre_samples = max(1, int(round(pre_seconds * rate)))
    window_samples = max(2, int(round(window_seconds * rate)))

    if direct_arrival_sample is None:
        energy = np.sum(np.square(channels), axis=0, dtype=np.float64)
        arrival = int(np.argmax(energy))
        arrival_method = "maximum_binaural_sample_energy"
    else:
        if (
            isinstance(direct_arrival_sample, bool)
            or not isinstance(direct_arrival_sample, (int, np.integer))
            or not 0 <= int(direct_arrival_sample) < channels.shape[1]
        ):
            raise M5MetricsError(
                "direct_arrival_sample must index the impulse response"
            )
        arrival = int(direct_arrival_sample)
        arrival_method = "caller_declared"

    start = max(0, arrival - pre_samples)
    end = min(channels.shape[1], start + window_samples)
    if end - start < 2:
        raise M5MetricsError("direct window contains fewer than two samples")
    direct = channels[:, start:end]
    direct = _binaural_channel_major(
        direct,
        owner="direct_window",
        channel_axis=0,
    )
    itd = estimate_itd_gcc_phat(
        direct[0],
        direct[1],
        rate,
        max_itd_seconds=max_itd_seconds,
        interpolation_factor=gcc_interpolation_factor,
    )
    return {
        "schema": "avengine_m5_binaural_rir_frame_cues_v1",
        "analysis_scope": "per_source_rir_direct_window",
        "channel_order": ["left", "right"],
        "sign_convention": {
            "ild": "10_log10_energy_left_over_right",
            "ipd": "angle_left_times_conjugate_right",
            "itd": "t_left_minus_t_right",
            "listener_local_azimuth": "positive_right_degrees",
        },
        "sample_rate_hz": rate,
        "direct_arrival_sample": arrival,
        "direct_arrival_method": arrival_method,
        "direct_window": {
            "start_sample": start,
            "end_sample_exclusive": end,
            "sample_count": end - start,
            "pre_direct_samples_requested": pre_samples,
        },
        "ild_db": _ild_db(direct),
        "ipd_radians_by_frequency_hz": _exact_frequency_ipd(direct, rate, frequencies),
        "itd": itd,
    }


def measure_binaural_rir_sequence_cues(
    impulse_responses: Sequence[Any] | np.ndarray,
    sample_rate_hz: int,
    *,
    direct_arrival_samples: Sequence[int | None] | None = None,
    **frame_options: Any,
) -> dict[str, Any]:
    """Measure every keyframe of a channel-major binaural RIR sequence."""

    if isinstance(impulse_responses, np.ndarray):
        if impulse_responses.ndim != 3:
            raise M5MetricsError(
                "impulse_responses array must have shape [frames, 2, samples]"
            )
        frames: Sequence[Any] = tuple(
            impulse_responses[index] for index in range(impulse_responses.shape[0])
        )
    elif isinstance(impulse_responses, Sequence) and not isinstance(
        impulse_responses, (str, bytes)
    ):
        frames = impulse_responses
    else:
        raise M5MetricsError("impulse_responses must be a non-empty sequence")
    if not frames:
        raise M5MetricsError("impulse_responses must be a non-empty sequence")

    if direct_arrival_samples is None:
        arrivals: Sequence[int | None] = (None,) * len(frames)
    else:
        if len(direct_arrival_samples) != len(frames):
            raise M5MetricsError(
                "direct_arrival_samples must match the RIR frame count"
            )
        arrivals = direct_arrival_samples
    reports: list[dict[str, Any]] = []
    for frame_index, (frame, arrival) in enumerate(zip(frames, arrivals, strict=True)):
        report = measure_binaural_rir_frame_cues(
            frame,
            sample_rate_hz,
            direct_arrival_sample=arrival,
            **frame_options,
        )
        reports.append({"frame_index": frame_index, **report})
    return {
        "schema": "avengine_m5_binaural_rir_sequence_cues_v1",
        "analysis_scope": "per_source_rir_direct_window_sequence",
        "frame_count": len(reports),
        "frames": reports,
    }


def _active_interval(
    start: Any,
    end: Any,
    *,
    sample_count: int,
) -> tuple[int, int]:
    for value, owner in (
        (start, "active_start_sample"),
        (end, "active_end_sample"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise M5MetricsError(f"{owner} must be an integer")
    first = int(start)
    last = int(end)
    if not 0 <= first < last <= sample_count:
        raise M5MetricsError(
            "active sample interval must be non-empty and inside the wet audio"
        )
    return first, last


def _circular_mean(
    phases: Sequence[float], weights: Sequence[float]
) -> tuple[float | None, float]:
    phase_array = np.asarray(phases, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    vector = np.sum(weight_array * np.exp(1.0j * phase_array))
    total_weight = float(np.sum(weight_array))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise M5MetricsError("IPD circular weights must sum to a positive value")
    resultant_length = float(abs(vector) / total_weight)
    if abs(vector) <= _ENERGY_EPSILON:
        return None, resultant_length
    return _wrap_phase(float(np.angle(vector))), resultant_length


def _measure_wet_window(
    samples: Any,
    sample_rate_hz: int,
    active_start_sample: int,
    active_end_sample: int,
    *,
    analysis_scope: str,
    source_id: str | None,
    diagnostic_only: bool,
    n_fft: int,
    hop_length: int,
    ipd_frequencies_hz: Sequence[float],
    max_itd_seconds: float,
    gcc_interpolation_factor: int,
    relative_energy_floor_db: float,
    channel_axis: int,
) -> dict[str, Any]:
    rate = _positive_sample_rate(sample_rate_hz)
    channels = _binaural_channel_major(
        samples,
        owner="binaural wet samples",
        channel_axis=channel_axis,
    )
    start, end = _active_interval(
        active_start_sample,
        active_end_sample,
        sample_count=channels.shape[1],
    )
    fft_size = _positive_integer(n_fft, owner="n_fft")
    hop = _positive_integer(hop_length, owner="hop_length")
    if fft_size < 8 or fft_size % 2:
        raise M5MetricsError("n_fft must be an even integer >= 8")
    if hop > fft_size:
        raise M5MetricsError("hop_length must not exceed n_fft")
    if end - start < fft_size:
        raise M5MetricsError("active interval must contain at least one STFT frame")
    if (
        isinstance(relative_energy_floor_db, bool)
        or not isinstance(relative_energy_floor_db, Real)
        or not math.isfinite(float(relative_energy_floor_db))
        or float(relative_energy_floor_db) > 0.0
    ):
        raise M5MetricsError("relative_energy_floor_db must be finite and <= 0")
    frequencies = _ipd_frequencies(ipd_frequencies_hz, sample_rate_hz=rate)
    if diagnostic_only:
        if source_id is not None:
            raise M5MetricsError("a mixture diagnostic cannot claim a source_id")
    elif not isinstance(source_id, str) or not source_id:
        raise M5MetricsError("a per-source wet-stem report requires source_id")

    window = np.hanning(fft_size).astype(np.float64)
    frame_starts = list(range(start, end - fft_size + 1, hop))
    frame_energies = np.asarray(
        [
            float(
                np.sum(
                    np.square(channels[:, offset : offset + fft_size] * window),
                    dtype=np.float64,
                )
            )
            for offset in frame_starts
        ],
        dtype=np.float64,
    )
    peak_energy = float(np.max(frame_energies))
    if not math.isfinite(peak_energy) or peak_energy <= 0.0:
        raise M5MetricsError("active interval contains no measurable frame energy")
    threshold = peak_energy * 10.0 ** (float(relative_energy_floor_db) / 10.0)

    fft_frequencies = np.fft.rfftfreq(fft_size, d=1.0 / rate)
    requested_bins = {
        frequency: int(np.argmin(np.abs(fft_frequencies - frequency)))
        for frequency in frequencies
    }
    frame_reports: list[dict[str, Any]] = []
    for ordinal, (offset, frame_energy) in enumerate(
        zip(frame_starts, frame_energies, strict=True)
    ):
        if float(frame_energy) < threshold:
            continue
        frame = channels[:, offset : offset + fft_size] * window
        spectrum = np.fft.rfft(frame, n=fft_size, axis=1)
        per_frequency: dict[str, Any] = {}
        for requested, bin_index in requested_bins.items():
            left = spectrum[0, bin_index]
            right = spectrum[1, bin_index]
            maximum_bin_magnitude = max(
                float(np.max(np.abs(spectrum[0]))),
                float(np.max(np.abs(spectrum[1]))),
            )
            floor = maximum_bin_magnitude * _PHAT_RELATIVE_FLOOR
            valid = abs(left) > floor and abs(right) > floor
            key = _frequency_key(requested)
            if valid:
                ild = float(
                    20.0
                    * math.log10(
                        max(float(abs(left)), _ENERGY_EPSILON)
                        / max(float(abs(right)), _ENERGY_EPSILON)
                    )
                )
                ipd = _wrap_phase(float(np.angle(left * np.conjugate(right))))
            else:
                ild = None
                ipd = None
            per_frequency[key] = {
                "requested_frequency_hz": requested,
                "analysis_frequency_hz": float(fft_frequencies[bin_index]),
                "valid": bool(valid),
                "ild_db": ild,
                "ipd_radians": ipd,
                "cross_magnitude": float(abs(left * np.conjugate(right))),
            }
        itd = estimate_itd_gcc_phat(
            frame[0],
            frame[1],
            rate,
            max_itd_seconds=max_itd_seconds,
            interpolation_factor=gcc_interpolation_factor,
        )
        frame_reports.append(
            {
                "stft_frame_index": ordinal,
                "start_sample": offset,
                "center_sample": offset + fft_size // 2,
                "relative_energy_db": float(
                    10.0 * math.log10(float(frame_energy) / peak_energy)
                ),
                "broadband_ild_db": _ild_db(frame),
                "itd": itd,
                "by_frequency_hz": per_frequency,
            }
        )
    if not frame_reports:
        raise M5MetricsError("no STFT frame passed the relative energy floor")

    summary_by_frequency: dict[str, Any] = {}
    for frequency in frequencies:
        key = _frequency_key(frequency)
        valid_records = [
            frame["by_frequency_hz"][key]
            for frame in frame_reports
            if frame["by_frequency_hz"][key]["valid"]
        ]
        if valid_records:
            ild_values = [float(record["ild_db"]) for record in valid_records]
            ipd_values = [float(record["ipd_radians"]) for record in valid_records]
            weights = [float(record["cross_magnitude"]) for record in valid_records]
            ipd_mean, ipd_resultant_length = _circular_mean(ipd_values, weights)
            summary_by_frequency[key] = {
                "requested_frequency_hz": frequency,
                "analysis_frequency_hz": valid_records[0]["analysis_frequency_hz"],
                "valid_frame_count": len(valid_records),
                "ild_db_median": float(np.median(ild_values)),
                "ipd_circular_mean_radians": ipd_mean,
                "ipd_resultant_length": ipd_resultant_length,
            }
        else:
            summary_by_frequency[key] = {
                "requested_frequency_hz": frequency,
                "analysis_frequency_hz": float(
                    fft_frequencies[requested_bins[frequency]]
                ),
                "valid_frame_count": 0,
                "ild_db_median": None,
                "ipd_circular_mean_radians": None,
                "ipd_resultant_length": None,
            }

    report: dict[str, Any] = {
        "schema": "avengine_m5_binaural_wet_stft_cues_v1",
        "analysis_scope": analysis_scope,
        "diagnostic_only": diagnostic_only,
        "source_specific_acceptance_allowed": not diagnostic_only,
        "channel_order": ["left", "right"],
        "sign_convention": {
            "ild": "10_log10_energy_left_over_right",
            "ipd": "angle_left_times_conjugate_right",
            "itd": "t_left_minus_t_right",
        },
        "sample_rate_hz": rate,
        "active_interval": {
            "start_sample": start,
            "end_sample_exclusive": end,
        },
        "stft": {
            "window": "periodic_false_hann_numpy_hanning",
            "n_fft": fft_size,
            "hop_length": hop,
            "candidate_frame_count": len(frame_starts),
            "measured_frame_count": len(frame_reports),
            "relative_energy_floor_db": float(relative_energy_floor_db),
        },
        "summary": {
            "broadband_ild_db_median": float(
                np.median([frame["broadband_ild_db"] for frame in frame_reports])
            ),
            "itd_seconds_median": float(
                np.median([frame["itd"]["itd_seconds"] for frame in frame_reports])
            ),
            "itd_samples_median": float(
                np.median([frame["itd"]["itd_samples"] for frame in frame_reports])
            ),
            "by_frequency_hz": summary_by_frequency,
        },
        "frames": frame_reports,
    }
    if source_id is not None:
        report["source_id"] = source_id
    if diagnostic_only:
        report["diagnostic_limitation"] = (
            "overlapping sources are not identifiable from mixture cues; "
            "use retained per-source stems for source-specific acceptance"
        )
    return report


def measure_binaural_wet_stem_cues(
    samples: Any,
    sample_rate_hz: int,
    active_start_sample: int,
    active_end_sample: int,
    *,
    source_id: str,
    n_fft: int = 512,
    hop_length: int = 128,
    ipd_frequencies_hz: Sequence[float] = DEFAULT_IPD_FREQUENCIES_HZ,
    max_itd_seconds: float = DEFAULT_MAX_ITD_SECONDS,
    gcc_interpolation_factor: int = 16,
    relative_energy_floor_db: float = -60.0,
    channel_axis: int = 0,
) -> dict[str, Any]:
    """Measure source-specific cues on the active interval of one wet stem."""

    return _measure_wet_window(
        samples,
        sample_rate_hz,
        active_start_sample,
        active_end_sample,
        analysis_scope="per_source_wet_stem_active_window_stft",
        source_id=source_id,
        diagnostic_only=False,
        n_fft=n_fft,
        hop_length=hop_length,
        ipd_frequencies_hz=ipd_frequencies_hz,
        max_itd_seconds=max_itd_seconds,
        gcc_interpolation_factor=gcc_interpolation_factor,
        relative_energy_floor_db=relative_energy_floor_db,
        channel_axis=channel_axis,
    )


def measure_binaural_mixture_diagnostic(
    samples: Any,
    sample_rate_hz: int,
    active_start_sample: int,
    active_end_sample: int,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    ipd_frequencies_hz: Sequence[float] = DEFAULT_IPD_FREQUENCIES_HZ,
    max_itd_seconds: float = DEFAULT_MAX_ITD_SECONDS,
    gcc_interpolation_factor: int = 16,
    relative_energy_floor_db: float = -60.0,
    channel_axis: int = 0,
) -> dict[str, Any]:
    """Measure a mixture for listening diagnostics, never source acceptance."""

    return _measure_wet_window(
        samples,
        sample_rate_hz,
        active_start_sample,
        active_end_sample,
        analysis_scope="binaural_mixture_active_window_stft_diagnostic",
        source_id=None,
        diagnostic_only=True,
        n_fft=n_fft,
        hop_length=hop_length,
        ipd_frequencies_hz=ipd_frequencies_hz,
        max_itd_seconds=max_itd_seconds,
        gcc_interpolation_factor=gcc_interpolation_factor,
        relative_energy_floor_db=relative_energy_floor_db,
        channel_axis=channel_axis,
    )


__all__ = [
    "DEFAULT_MINIMUM_ABSOLUTE_ILD_DB",
    "DEFAULT_MINIMUM_ABSOLUTE_ITD_SECONDS",
    "DEFAULT_MINIMUM_CUE_CONSISTENCY_RATE",
    "DEFAULT_MINIMUM_CUE_COVERAGE_RATE",
    "DEFAULT_MINIMUM_LATERAL_ANGLE_DEG",
    "DEFAULT_IPD_FREQUENCIES_HZ",
    "DEFAULT_MAX_ITD_SECONDS",
    "M5MetricsError",
    "estimate_itd_gcc_phat",
    "listener_local_azimuth_deg",
    "listener_local_source_geometry",
    "measure_binaural_mixture_diagnostic",
    "measure_binaural_rir_frame_cues",
    "measure_binaural_rir_sequence_cues",
    "measure_binaural_wet_stem_cues",
    "summarize_lateral_cue_consistency",
]
