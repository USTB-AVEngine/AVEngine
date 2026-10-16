"""Pinned first-order Ambisonics contract for the M4 RLR authority path.

The pinned RLR binary documents N3D normalization and world-coordinate
alignment.  Its four channel indices were frozen by a six-cardinal,
direct-only executable canary.  AVEngine therefore stores the raw result as
``[W, Y, Z, X]`` in Habitat/AVEngine world axes; it is not silently relabeled
as AmbiX (whose common interchange convention is ACN/SN3D).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


RLR_FOA_FORMAT_ID = "rlr_foa_acn_n3d_world_v1"
RLR_FOA_CHANNEL_ORDER = ("W", "Y", "Z", "X")
RLR_FOA_NORMALIZATION = "N3D"
RLR_FOA_COORDINATE_FRAME = "avengine_world"
RLR_FOA_CHANNEL_COUNT = 4

_CARDINAL_COMPONENTS: dict[str, tuple[int, float, str]] = {
    "+X": (3, 1.0, "right"),
    "-X": (3, -1.0, "left"),
    "+Y": (1, 1.0, "up"),
    "-Y": (1, -1.0, "down"),
    "+Z": (2, 1.0, "back"),
    "-Z": (2, -1.0, "front"),
}


class SpatialContractError(ValueError):
    """FOA samples or their measured direction violate the pinned contract."""


@dataclass(frozen=True)
class CardinalFOAMeasurement:
    direction: str
    semantic_direction: str
    direct_arrival_sample: int
    w_amplitude: float
    directional_channel_index: int
    directional_to_w_ratio: float
    maximum_off_axis_to_w_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "semantic_direction": self.semantic_direction,
            "direct_arrival_sample": self.direct_arrival_sample,
            "w_amplitude": self.w_amplitude,
            "directional_channel_index": self.directional_channel_index,
            "directional_to_w_ratio": self.directional_to_w_ratio,
            "maximum_off_axis_to_w_ratio": self.maximum_off_axis_to_w_ratio,
        }


def rlr_foa_contract() -> dict[str, Any]:
    """Return a JSON-safe copy of the frozen raw-FOA spatial contract."""

    return {
        "format_id": RLR_FOA_FORMAT_ID,
        "ambisonic_order": 1,
        "channel_count": RLR_FOA_CHANNEL_COUNT,
        "raw_channel_order": list(RLR_FOA_CHANNEL_ORDER),
        "acn_indices": [0, 1, 2, 3],
        "normalization": RLR_FOA_NORMALIZATION,
        "coordinate_frame": RLR_FOA_COORDINATE_FRAME,
        "handedness": "right",
        "axes": {
            "right": "+X",
            "up": "+Y",
            "back": "+Z",
            "forward": "-Z",
        },
        "raw_array_layout": "channel_major_[channels,samples]",
        "dtype": "float32_le",
    }


def rlr_foa_wav_metadata() -> dict[str, Any]:
    """Metadata suitable for :func:`avengine.spatial_audio.audio.write_float32_wav`."""

    return {"spatial_format": rlr_foa_contract()}


def _foa_channel_major(
    value: Any,
    *,
    owner: str,
    channel_axis: int,
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise SpatialContractError(f"{owner} must contain real numeric samples")
    if source.ndim != 2:
        raise SpatialContractError(f"{owner} must have two dimensions")
    if channel_axis in (0, -2):
        oriented = source
    elif channel_axis in (1, -1):
        oriented = source.T
    else:
        raise SpatialContractError("channel_axis must explicitly identify axis 0 or 1")
    if oriented.shape[0] != RLR_FOA_CHANNEL_COUNT or oriented.shape[1] < 1:
        raise SpatialContractError(
            f"{owner} must have shape [4, samples] in the declared channel axis"
        )
    try:
        result = np.ascontiguousarray(oriented, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpatialContractError(f"{owner} cannot be represented as float64") from exc
    if not np.all(np.isfinite(result)):
        raise SpatialContractError(f"{owner} must contain only finite samples")
    return result


def validate_foa_samples(
    samples: Any,
    *,
    channel_axis: int = 0,
) -> np.ndarray:
    """Return a validated, owned ``float64`` raw-FOA array."""

    return _foa_channel_major(
        samples,
        owner="FOA samples",
        channel_axis=channel_axis,
    ).copy()


def validate_cardinal_foa(
    impulse_responses: Mapping[str, Any],
    *,
    channel_axis: int = 0,
    ratio_rtol: float = 2.0e-4,
    off_axis_ratio_atol: float = 2.0e-5,
    equal_distance_rtol: float = 2.0e-4,
) -> dict[str, Any]:
    """Validate the direct-only six-cardinal RLR FOA canary.

    Every source must be at the same distance from an identity-orientation
    listener.  At the W-channel peak, first-order N3D requires a directional
    coefficient of ``+/-sqrt(3) * W`` and zero off-axis coefficients.
    Reflections, diffraction, transmission, source/listener radius, and
    temporal coherence belong outside this cardinal-format canary.
    """

    if not isinstance(impulse_responses, Mapping):
        raise SpatialContractError("impulse_responses must be a direction mapping")
    if set(impulse_responses) != set(_CARDINAL_COMPONENTS):
        raise SpatialContractError(
            "cardinal FOA canary requires exactly +X, -X, +Y, -Y, +Z, -Z"
        )
    tolerances = (ratio_rtol, off_axis_ratio_atol, equal_distance_rtol)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in tolerances
    ):
        raise SpatialContractError("cardinal validation tolerances must be finite and non-negative")

    expected_magnitude = math.sqrt(3.0)
    measurements: list[CardinalFOAMeasurement] = []
    arrival_samples: list[int] = []
    w_magnitudes: list[float] = []
    for direction in _CARDINAL_COMPONENTS:
        expected_channel, expected_sign, semantic = _CARDINAL_COMPONENTS[direction]
        channels = _foa_channel_major(
            impulse_responses[direction],
            owner=f"impulse_responses[{direction!r}]",
            channel_axis=channel_axis,
        )
        w_absolute = np.abs(channels[0])
        arrival = int(np.argmax(w_absolute))
        w = float(channels[0, arrival])
        if not math.isfinite(w) or w == 0.0:
            raise SpatialContractError(f"{direction} W channel has no direct peak")
        direct = channels[:, arrival]
        ratio = float(direct[expected_channel] / w)
        expected_ratio = expected_sign * expected_magnitude
        if not math.isclose(
            ratio,
            expected_ratio,
            rel_tol=float(ratio_rtol),
            abs_tol=float(off_axis_ratio_atol),
        ):
            raise SpatialContractError(
                f"{direction} channel {expected_channel}/W ratio {ratio:.17g} "
                f"does not match N3D {expected_ratio:.17g}"
            )
        off_axis_indices = tuple(
            index for index in (1, 2, 3) if index != expected_channel
        )
        maximum_off_axis_ratio = max(
            abs(float(direct[index] / w)) for index in off_axis_indices
        )
        if maximum_off_axis_ratio > float(off_axis_ratio_atol):
            raise SpatialContractError(
                f"{direction} has off-axis/W ratio {maximum_off_axis_ratio:.17g}"
            )
        arrival_samples.append(arrival)
        w_magnitudes.append(abs(w))
        measurements.append(
            CardinalFOAMeasurement(
                direction=direction,
                semantic_direction=semantic,
                direct_arrival_sample=arrival,
                w_amplitude=w,
                directional_channel_index=expected_channel,
                directional_to_w_ratio=ratio,
                maximum_off_axis_to_w_ratio=maximum_off_axis_ratio,
            )
        )

    if len(set(arrival_samples)) != 1:
        raise SpatialContractError(
            "equal-distance cardinal sources must have one direct-arrival sample"
        )
    reference_w = w_magnitudes[0]
    if any(
        not math.isclose(
            value,
            reference_w,
            rel_tol=float(equal_distance_rtol),
            abs_tol=0.0,
        )
        for value in w_magnitudes[1:]
    ):
        raise SpatialContractError(
            "equal-distance cardinal sources must have equal W magnitude"
        )

    return {
        "status": "pass",
        "spatial_format": rlr_foa_contract(),
        "expected_directional_to_w_magnitude": expected_magnitude,
        "direct_arrival_sample": arrival_samples[0],
        "measurements": [item.to_dict() for item in measurements],
    }


def validate_world_aligned_foa(
    identity_orientation: Any,
    rotated_listener_orientation: Any,
    *,
    channel_axis: int = 0,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Prove that raw RLR FOA did not rotate into listener-local axes."""

    first = _foa_channel_major(
        identity_orientation,
        owner="identity_orientation",
        channel_axis=channel_axis,
    )
    second = _foa_channel_major(
        rotated_listener_orientation,
        owner="rotated_listener_orientation",
        channel_axis=channel_axis,
    )
    if first.shape != second.shape:
        raise SpatialContractError("world-alignment comparison shapes differ")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise SpatialContractError(f"{name} must be finite and non-negative")
    maximum_difference = float(np.max(np.abs(first - second)))
    if not np.allclose(first, second, rtol=float(rtol), atol=float(atol)):
        raise SpatialContractError(
            "raw FOA changed with listener orientation; it is not world-aligned"
        )
    return {
        "status": "pass",
        "format_id": RLR_FOA_FORMAT_ID,
        "sample_count": int(first.shape[1]),
        "maximum_absolute_difference": maximum_difference,
        "rtol": float(rtol),
        "atol": float(atol),
    }
