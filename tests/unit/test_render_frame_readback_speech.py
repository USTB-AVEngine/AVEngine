from __future__ import annotations

import numpy as np
import pytest

from tools.acoustics.render_frame_readback_sequential_speech import (
    _animation_readback_qa,
    _cm_to_m,
    _place_wet_event,
    _ue_rotator_to_m3_basis,
    _ue_rotator_to_m3_orientation_wxyz,
    _validate_readback_lengths,
)


def test_ue_location_exchanges_height_into_canonical_y():
    assert _cm_to_m([100.0, 200.0, 300.0]) == (1.0, 3.0, 2.0)


def test_identity_ue_optical_frame_maps_to_m3_listener_basis_quaternion():
    quaternion = _ue_rotator_to_m3_orientation_wxyz([0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert np.isclose(abs(quaternion[0]), 2.0 ** -0.5)
    assert np.isclose(abs(quaternion[2]), 2.0 ** -0.5)


def test_wet_event_preserves_event_start_and_stereo_shape():
    mixture = np.zeros((2, 32), dtype=np.float64)
    dry = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)
    ir = np.asarray([[1.0, 0.5], [0.25, 0.0]], dtype=np.float64)
    start, end, placed = _place_wet_event(
        mixture, dry, ir, start_sample=11, gain=1.0
    )
    assert (start, end) == (11, 15)
    assert placed.shape == (2, 4)
    assert np.flatnonzero(np.any(mixture != 0.0, axis=0)).tolist() == [11, 12, 13, 14]
    assert np.allclose(mixture[:, :11], 0.0)


def test_ue_optical_forward_and_up_axes_map_to_m3_listener_basis():
    assert _ue_rotator_to_m3_basis([0.0, 0.0, 0.0])[0] == (1.0, 0.0, 0.0)
    assert _ue_rotator_to_m3_basis([0.0, 0.0, 0.0])[2] == (0.0, 1.0, 0.0)
    assert np.allclose(_ue_rotator_to_m3_basis([0.0, 0.0, 90.0])[0], (0.0, 0.0, 1.0))
    assert np.allclose(_ue_rotator_to_m3_basis([0.0, 0.0, 180.0])[0], (-1.0, 0.0, 0.0))
    assert np.allclose(_ue_rotator_to_m3_basis([0.0, 90.0, 0.0])[0], (0.0, 1.0, 0.0))
    assert np.allclose(_ue_rotator_to_m3_basis([0.0, -90.0, 0.0])[0], (0.0, -1.0, 0.0))
    assert np.isclose(np.linalg.norm(_ue_rotator_to_m3_orientation_wxyz([0.0, 0.0, 90.0])), 1.0)


def test_readback_animation_status_requires_complete_values():
    assert _animation_readback_qa({}, ["actor"], 1)["status"] == "not_run"
    result = _animation_readback_qa({"animations": {"actor": []}}, ["actor"], 1)
    assert result["status"] == "fail"


def test_readback_lengths_reject_missing_frames_instead_of_clamping():
    with pytest.raises(ValueError, match="exactly 2"):
        _validate_readback_lengths(
            {"camera": [{}], "emitters": {"actor": [{}]}}, 2, ["actor"]
        )
