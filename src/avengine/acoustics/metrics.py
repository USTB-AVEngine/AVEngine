"""Independent, fail-closed room-impulse-response metrics for M3.

The RLR runtime can write its own metric report, but the M3 material canary
also evaluates a copied raw IR with this module.  Keeping the calculation in
AVEngine makes the admission gate inspectable and prevents a missing or stale
runtime-side report from being treated as evidence.

All calculations are broadband.  Channels are combined in the energy domain;
the input layout is explicit and is never inferred from its dimensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


class AcousticMetricError(ValueError):
    """The IR cannot support the requested acoustic measurement."""


@dataclass(frozen=True)
class MetricConfig:
    """Parameters for one deterministic broadband IR measurement.

    ``direct_arrival_threshold_db`` is an amplitude threshold relative to the
    global peak.  DRR uses the energy from the detected arrival through the
    half-open direct window and treats all following energy as reverberant.
    ``late_start_ms`` defaults to the conventional 80 ms clarity boundary.
    EDT is the -0 dB to -10 dB least-squares slope of the Schroeder energy
    decay curve, extrapolated to -60 dB.
    """

    direct_arrival_threshold_db: float = -20.0
    direct_window_ms: float = 5.0
    late_start_ms: float = 80.0
    minimum_edt_fit_samples: int = 8

    def validate(self) -> None:
        values = (
            self.direct_arrival_threshold_db,
            self.direct_window_ms,
            self.late_start_ms,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise AcousticMetricError("metric configuration must be finite")
        if not -120.0 < self.direct_arrival_threshold_db < 0.0:
            raise AcousticMetricError(
                "direct_arrival_threshold_db must be between -120 and 0 dB"
            )
        if self.direct_window_ms <= 0.0:
            raise AcousticMetricError("direct_window_ms must be positive")
        if self.late_start_ms <= self.direct_window_ms:
            raise AcousticMetricError(
                "late_start_ms must be later than the direct window"
            )
        if (
            isinstance(self.minimum_edt_fit_samples, bool)
            or not isinstance(self.minimum_edt_fit_samples, int)
            or self.minimum_edt_fit_samples < 2
        ):
            raise AcousticMetricError(
                "minimum_edt_fit_samples must be an integer of at least two"
            )


@dataclass(frozen=True)
class IRMetrics:
    """Broadband metrics and the exact measurement boundaries used."""

    sample_rate_hz: float
    channel_count: int
    sample_count: int
    direct_arrival_sample: int
    direct_arrival_seconds: float
    direct_window_end_sample: int
    late_start_sample: int
    peak_absolute_amplitude: float
    total_energy: float
    direct_energy: float
    reverberant_energy: float
    late_energy: float
    drr_db: float
    late_energy_ratio: float
    late_energy_db: float
    edt_seconds: float
    edt_fit_r2: float
    edt_fit_sample_count: int
    edt_decay_span_db: float
    configuration: MetricConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with no NaN/Infinity values."""

        value = asdict(self)
        if not _all_finite(value):
            raise AcousticMetricError("metric result contains a non-finite value")
        return value


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _channel_major_ir(ir: np.ndarray, *, channel_axis: int) -> np.ndarray:
    source = np.asarray(ir)
    if np.iscomplexobj(source):
        raise AcousticMetricError("IR samples must be real-valued")
    if source.ndim == 1:
        if channel_axis not in {0, -1}:
            raise AcousticMetricError("a one-dimensional IR has no channel axis")
        source = source[np.newaxis, :]
    elif source.ndim == 2:
        if channel_axis in {-2, 0}:
            pass
        elif channel_axis in {-1, 1}:
            source = source.T
        else:
            raise AcousticMetricError("channel_axis must identify axis 0 or axis 1")
    else:
        raise AcousticMetricError("IR must have shape [samples] or [channels, samples]")

    result = np.ascontiguousarray(source, dtype=np.float64)
    if result.shape[0] < 1 or result.shape[1] < 2:
        raise AcousticMetricError("IR must contain at least one channel and two samples")
    if not np.all(np.isfinite(result)):
        raise AcousticMetricError("IR samples must all be finite")
    return result


def _positive_sample_count(milliseconds: float, sample_rate_hz: float) -> int:
    return max(1, int(round(milliseconds * sample_rate_hz / 1000.0)))


def _detect_direct_arrival(
    channels: np.ndarray, *, threshold_db: float
) -> tuple[int, float]:
    amplitude = np.max(np.abs(channels), axis=0)
    peak = float(np.max(amplitude))
    if not math.isfinite(peak) or peak <= 0.0:
        raise AcousticMetricError("IR is silent; a direct arrival cannot be detected")
    threshold = peak * 10.0 ** (threshold_db / 20.0)
    candidates = np.flatnonzero(amplitude >= threshold)
    if candidates.size == 0:
        raise AcousticMetricError("IR has no sample above the direct-arrival threshold")
    return int(candidates[0]), peak


def _edt_from_energy(
    energy: np.ndarray,
    *,
    arrival: int,
    sample_rate_hz: float,
    minimum_fit_samples: int,
) -> tuple[float, float, int, float]:
    decay_energy = np.cumsum(energy[arrival:][::-1], dtype=np.float64)[::-1]
    initial_energy = float(decay_energy[0])
    if not math.isfinite(initial_energy) or initial_energy <= 0.0:
        raise AcousticMetricError("IR contains no energy at or after direct arrival")

    positive = decay_energy > 0.0
    decay_db = np.full(decay_energy.shape, -np.inf, dtype=np.float64)
    decay_db[positive] = 10.0 * np.log10(decay_energy[positive] / initial_energy)
    crossings = np.flatnonzero(decay_db <= -10.0)
    if crossings.size == 0:
        raise AcousticMetricError("Schroeder curve does not span the -10 dB EDT range")

    end = int(crossings[0])
    fit_count = end + 1
    if fit_count < minimum_fit_samples:
        raise AcousticMetricError(
            "Schroeder -0 dB to -10 dB range has too few samples for EDT"
        )
    y = decay_db[:fit_count]
    if not np.all(np.isfinite(y)):
        raise AcousticMetricError("Schroeder EDT fit contains non-finite values")
    x = np.arange(fit_count, dtype=np.float64) / sample_rate_hz
    slope, intercept = np.polyfit(x, y, 1)
    if not math.isfinite(float(slope)) or slope >= 0.0:
        raise AcousticMetricError("Schroeder EDT fit does not have a negative slope")

    predicted = slope * x + intercept
    residual_sum = float(np.sum(np.square(y - predicted), dtype=np.float64))
    centered_sum = float(np.sum(np.square(y - float(np.mean(y))), dtype=np.float64))
    if centered_sum <= 0.0:
        raise AcousticMetricError("Schroeder EDT fit has zero decay variance")
    fit_r2 = 1.0 - residual_sum / centered_sum
    # Round-off can move a mathematically bounded R2 a few ulps outside [0, 1].
    fit_r2 = min(1.0, max(0.0, float(fit_r2)))
    edt_seconds = -60.0 / float(slope)
    decay_span_db = -float(y[-1])
    if not all(math.isfinite(value) and value > 0.0 for value in (edt_seconds, decay_span_db)):
        raise AcousticMetricError("EDT result is not finite and positive")
    return edt_seconds, fit_r2, fit_count, decay_span_db


def analyze_ir(
    ir: np.ndarray,
    sample_rate_hz: float,
    *,
    channel_axis: int = 0,
    configuration: MetricConfig | None = None,
) -> IRMetrics:
    """Measure EDT, DRR and late energy from a copied raw impulse response.

    Args:
        ir: One-dimensional mono samples or a two-dimensional array.
        sample_rate_hz: Sampling rate attached to this exact IR buffer.
        channel_axis: Explicit channel axis for two-dimensional input.  The
            modern AVEngine RLR adapter returns channel-major arrays, so zero is
            the default.
        configuration: Measurement boundary configuration.

    Raises:
        AcousticMetricError: if the input is malformed, silent, truncated, or
            does not contain enough decay evidence for a finite measurement.
    """

    if isinstance(sample_rate_hz, bool):
        raise AcousticMetricError("sample_rate_hz must be a finite positive number")
    try:
        rate = float(sample_rate_hz)
    except (TypeError, ValueError) as exc:
        raise AcousticMetricError(
            "sample_rate_hz must be a finite positive number"
        ) from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise AcousticMetricError("sample_rate_hz must be a finite positive number")

    config = MetricConfig() if configuration is None else configuration
    if not isinstance(config, MetricConfig):
        raise AcousticMetricError("configuration must be a MetricConfig")
    config.validate()
    channels = _channel_major_ir(ir, channel_axis=channel_axis)
    arrival, peak = _detect_direct_arrival(
        channels, threshold_db=config.direct_arrival_threshold_db
    )

    energy = np.mean(np.square(channels), axis=0, dtype=np.float64)
    direct_end = min(
        channels.shape[1],
        arrival + _positive_sample_count(config.direct_window_ms, rate),
    )
    late_start = arrival + _positive_sample_count(config.late_start_ms, rate)
    if direct_end >= channels.shape[1]:
        raise AcousticMetricError("IR ends inside the configured direct window")
    if late_start >= channels.shape[1]:
        raise AcousticMetricError("IR is too short for the configured late-energy window")

    total_energy = float(np.sum(energy[arrival:], dtype=np.float64))
    direct_energy = float(np.sum(energy[arrival:direct_end], dtype=np.float64))
    reverberant_energy = float(np.sum(energy[direct_end:], dtype=np.float64))
    late_energy = float(np.sum(energy[late_start:], dtype=np.float64))
    named_energies = {
        "total": total_energy,
        "direct": direct_energy,
        "reverberant": reverberant_energy,
        "late": late_energy,
    }
    for name, value in named_energies.items():
        if not math.isfinite(value) or value <= 0.0:
            raise AcousticMetricError(f"IR {name} energy must be finite and positive")

    drr_db = 10.0 * math.log10(direct_energy / reverberant_energy)
    late_energy_ratio = late_energy / total_energy
    late_energy_db = 10.0 * math.log10(late_energy_ratio)
    edt_seconds, edt_fit_r2, edt_fit_count, decay_span_db = _edt_from_energy(
        energy,
        arrival=arrival,
        sample_rate_hz=rate,
        minimum_fit_samples=config.minimum_edt_fit_samples,
    )

    result = IRMetrics(
        sample_rate_hz=rate,
        channel_count=int(channels.shape[0]),
        sample_count=int(channels.shape[1]),
        direct_arrival_sample=arrival,
        direct_arrival_seconds=arrival / rate,
        direct_window_end_sample=direct_end,
        late_start_sample=late_start,
        peak_absolute_amplitude=peak,
        total_energy=total_energy,
        direct_energy=direct_energy,
        reverberant_energy=reverberant_energy,
        late_energy=late_energy,
        drr_db=drr_db,
        late_energy_ratio=late_energy_ratio,
        late_energy_db=late_energy_db,
        edt_seconds=edt_seconds,
        edt_fit_r2=edt_fit_r2,
        edt_fit_sample_count=edt_fit_count,
        edt_decay_span_db=decay_span_db,
        configuration=config,
    )
    result.to_dict()
    return result


__all__ = [
    "AcousticMetricError",
    "IRMetrics",
    "MetricConfig",
    "analyze_ir",
]
