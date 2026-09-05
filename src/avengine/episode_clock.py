"""One explicit frame/audio clock for audiovisual episode producers.

The HM3D review tools historically inferred a five-second, 75-frame clip in
several places.  Keeping the arithmetic here makes a configured clock a single
input: the sample count is derived from the frame count, frame rate and sample
rate, and every producer can use the same half-open sample boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Mapping


# These are compatibility defaults for old banks that predate an explicit
# clock object.  New callers should pass the values from configuration.
LEGACY_FRAME_COUNT = 75
LEGACY_FRAME_RATE_HZ = Fraction(15, 1)
LEGACY_SAMPLE_RATE_HZ = 16_000


class EpisodeClockError(ValueError):
    """Raised when one audiovisual clock cannot be represented exactly."""


def _positive_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EpisodeClockError(f"{owner} must be a positive integer")
    result = int(value)
    if result < 1:
        raise EpisodeClockError(f"{owner} must be a positive integer")
    return result


def _positive_rate(value: Any, *, owner: str) -> Fraction:
    if isinstance(value, bool):
        raise EpisodeClockError(f"{owner} must be a positive finite rate")
    try:
        if isinstance(value, Fraction):
            rate = value
        elif isinstance(value, Integral):
            rate = Fraction(int(value), 1)
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("non-finite")
            rate = Fraction(str(numeric))
        else:
            rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise EpisodeClockError(f"{owner} must be a positive finite rate") from exc
    if rate <= 0:
        raise EpisodeClockError(f"{owner} must be a positive finite rate")
    return rate


def _round_half_up(value: Fraction) -> int:
    if value < 0:
        raise EpisodeClockError("clock boundaries cannot be negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


@dataclass(frozen=True)
class EpisodeClock:
    """A validated visual/audio clock with derived integer sample count."""

    frame_count: int
    frame_rate_hz: Fraction
    sample_rate_hz: int
    clip_seconds: Fraction
    sample_count: int
    compatibility: str = "configured"

    @classmethod
    def from_values(
        cls,
        *,
        frame_count: Any,
        frame_rate_hz: Any,
        sample_rate_hz: Any,
        clip_seconds: Any | None = None,
        compatibility: str = "configured",
    ) -> "EpisodeClock":
        frames = _positive_integer(frame_count, owner="frame_count")
        rate = _positive_rate(frame_rate_hz, owner="frame_rate_hz")
        samples = _positive_integer(sample_rate_hz, owner="sample_rate_hz")
        expected_clip = Fraction(frames, 1) / rate
        if clip_seconds is None:
            clip = expected_clip
        else:
            clip = _positive_rate(clip_seconds, owner="clip_seconds")
            if clip != expected_clip:
                raise EpisodeClockError(
                    f"frame_count/frame_rate_hz requires clip_seconds="
                    f"{float(expected_clip):g}, got {float(clip):g}"
                )
        exact_samples = clip * samples
        if exact_samples.denominator != 1:
            raise EpisodeClockError(
                "clip_seconds*sample_rate_hz must be an integer sample count"
            )
        sample_count = int(exact_samples)
        if sample_count < 1:
            raise EpisodeClockError("derived sample_count must be positive")
        if not isinstance(compatibility, str) or not compatibility:
            raise EpisodeClockError("compatibility must be a non-empty string")
        return cls(
            frame_count=frames,
            frame_rate_hz=rate,
            sample_rate_hz=samples,
            clip_seconds=clip,
            sample_count=sample_count,
            compatibility=compatibility,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        compatibility: str | None = None,
    ) -> "EpisodeClock":
        if not isinstance(value, Mapping):
            raise EpisodeClockError("clock must be an object")
        required = {"frame_count", "frame_rate_hz", "sample_rate_hz"}
        missing = sorted(required - set(value))
        if missing:
            raise EpisodeClockError("clock is missing " + ", ".join(missing))
        result = cls.from_values(
            frame_count=value["frame_count"],
            frame_rate_hz=value["frame_rate_hz"],
            sample_rate_hz=value["sample_rate_hz"],
            clip_seconds=value.get("clip_seconds"),
            compatibility=(
                compatibility
                if compatibility is not None
                else str(value.get("compatibility", "configured"))
            ),
        )
        declared_count = value.get("sample_count")
        if declared_count is not None:
            declared_count = _positive_integer(
                declared_count, owner="clock.sample_count"
            )
            if declared_count != result.sample_count:
                raise EpisodeClockError(
                    f"clock.sample_count must equal derived {result.sample_count}"
                )
        return result

    @property
    def frame_rate_float(self) -> float:
        return float(self.frame_rate_hz)

    @property
    def clip_seconds_float(self) -> float:
        return float(self.clip_seconds)

    def sample_boundary(self, frame_index: int) -> int:
        """Return the rounded half-open sample boundary for a frame."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
            raise EpisodeClockError("frame_index must be an integer")
        if not 0 <= frame_index <= self.frame_count:
            raise EpisodeClockError(
                f"frame_index must be in [0,{self.frame_count}]"
            )
        return _round_half_up(
            Fraction(int(frame_index) * self.sample_rate_hz, 1) / self.frame_rate_hz
        )

    def frame_boundaries(self) -> tuple[int, ...]:
        boundaries = tuple(
            self.sample_boundary(index) for index in range(self.frame_count + 1)
        )
        if boundaries[-1] != self.sample_count:
            raise EpisodeClockError("last frame boundary differs from sample_count")
        if any(a > b for a, b in zip(boundaries, boundaries[1:])):
            raise EpisodeClockError("frame sample boundaries are not monotonic")
        return boundaries

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "frame_rate_hz": self.frame_rate_float,
            "sample_rate_hz": self.sample_rate_hz,
            "clip_seconds": self.clip_seconds_float,
            "sample_count": self.sample_count,
            "compatibility": self.compatibility,
        }


__all__ = [
    "EpisodeClock",
    "EpisodeClockError",
    "LEGACY_FRAME_COUNT",
    "LEGACY_FRAME_RATE_HZ",
    "LEGACY_SAMPLE_RATE_HZ",
]
