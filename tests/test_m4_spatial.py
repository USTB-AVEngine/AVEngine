from __future__ import annotations

import math

import numpy as np
import pytest

from avengine.m4.spatial import (
    RLR_FOA_CHANNEL_ORDER,
    RLR_FOA_FORMAT_ID,
    SpatialContractError,
    rlr_foa_contract,
    validate_cardinal_foa,
    validate_foa_samples,
    validate_world_aligned_foa,
)


_DIRECTION_CHANNEL = {
    "+X": (3, 1.0),
    "-X": (3, -1.0),
    "+Y": (1, 1.0),
    "-Y": (1, -1.0),
    "+Z": (2, 1.0),
    "-Z": (2, -1.0),
}


def _cardinal_ir() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for direction, (channel, sign) in _DIRECTION_CHANNEL.items():
        ir = np.zeros((4, 24), dtype=np.float64)
        ir[0, 7] = 0.5
        ir[channel, 7] = sign * math.sqrt(3.0) * 0.5
        result[direction] = ir
    return result


def test_raw_rlr_foa_contract_is_explicit_and_not_ambix() -> None:
    contract = rlr_foa_contract()

    assert contract["format_id"] == RLR_FOA_FORMAT_ID
    assert contract["raw_channel_order"] == list(RLR_FOA_CHANNEL_ORDER) == [
        "W",
        "Y",
        "Z",
        "X",
    ]
    assert contract["normalization"] == "N3D"
    assert contract["coordinate_frame"] == "avengine_world"
    assert contract["acn_indices"] == [0, 1, 2, 3]
    assert contract["handedness"] == "right"
    assert contract["axes"] == {
        "right": "+X",
        "up": "+Y",
        "back": "+Z",
        "forward": "-Z",
    }
    assert contract["raw_array_layout"] == "channel_major_[channels,samples]"
    assert contract["dtype"] == "float32_le"


def test_six_cardinal_canary_freezes_acn_n3d_world_mapping() -> None:
    evidence = validate_cardinal_foa(_cardinal_ir())

    assert evidence["status"] == "pass"
    assert evidence["direct_arrival_sample"] == 7
    assert evidence["expected_directional_to_w_magnitude"] == pytest.approx(
        math.sqrt(3.0)
    )
    by_direction = {
        item["direction"]: item for item in evidence["measurements"]
    }
    assert by_direction["+X"]["directional_channel_index"] == 3
    assert by_direction["+Y"]["directional_channel_index"] == 1
    assert by_direction["+Z"]["directional_channel_index"] == 2
    assert by_direction["-Z"]["semantic_direction"] == "front"


def test_cardinal_canary_accepts_explicit_frame_major_input() -> None:
    frame_major = {key: value.T for key, value in _cardinal_ir().items()}

    evidence = validate_cardinal_foa(frame_major, channel_axis=1)

    assert evidence["status"] == "pass"


def test_cardinal_canary_rejects_wrong_normalization_order_and_shape() -> None:
    sn3d = _cardinal_ir()
    sn3d["+X"][3, 7] = 0.5
    with pytest.raises(SpatialContractError, match="does not match N3D"):
        validate_cardinal_foa(sn3d)

    wrong_order = _cardinal_ir()
    wrong_order["+Y"][[1, 3]] = wrong_order["+Y"][[3, 1]]
    with pytest.raises(SpatialContractError, match="channel 1/W ratio"):
        validate_cardinal_foa(wrong_order)

    wrong_shape = _cardinal_ir()
    wrong_shape["-Z"] = np.zeros((3, 24))
    with pytest.raises(SpatialContractError, match=r"shape \[4, samples\]"):
        validate_cardinal_foa(wrong_shape)


def test_cardinal_canary_rejects_tampered_direction_set_arrival_and_energy() -> None:
    missing = _cardinal_ir()
    missing.pop("-Y")
    with pytest.raises(SpatialContractError, match="exactly"):
        validate_cardinal_foa(missing)

    shifted = _cardinal_ir()
    shifted["-X"] = np.roll(shifted["-X"], 1, axis=1)
    with pytest.raises(SpatialContractError, match="direct-arrival"):
        validate_cardinal_foa(shifted)

    unequal = _cardinal_ir()
    unequal["+Z"] *= 0.8
    with pytest.raises(SpatialContractError, match="equal W magnitude"):
        validate_cardinal_foa(unequal)


def test_raw_foa_is_world_aligned_across_listener_rotation() -> None:
    raw = _cardinal_ir()["+X"]
    evidence = validate_world_aligned_foa(raw, raw.copy())

    assert evidence["status"] == "pass"
    assert evidence["maximum_absolute_difference"] == 0.0

    rotated_channels = raw.copy()
    rotated_channels[[1, 3]] = rotated_channels[[3, 1]]
    with pytest.raises(SpatialContractError, match="changed with listener orientation"):
        validate_world_aligned_foa(raw, rotated_channels)


def test_foa_sample_validator_rejects_nonfinite_and_ambiguous_axis() -> None:
    valid = validate_foa_samples(np.zeros((4, 3)))
    assert valid.shape == (4, 3)
    assert valid.dtype == np.float64

    with pytest.raises(SpatialContractError, match="finite"):
        validate_foa_samples(np.full((4, 3), np.nan))
    with pytest.raises(SpatialContractError, match="channel_axis"):
        validate_foa_samples(np.zeros((4, 3)), channel_axis=2)
