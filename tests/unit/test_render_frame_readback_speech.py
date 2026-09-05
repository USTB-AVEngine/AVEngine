from __future__ import annotations

import numpy as np

from tools.acoustics.render_frame_readback_sequential_speech import (
    _cm_to_m,
    _place_wet_event,
    _ue_rotator_to_m3_orientation_wxyz,
)


def test_ue_location_exchanges_height_into_canonical_y():
    assert _cm_to_m([100.0, 200.0, 300.0]) == (1.0, 3.0, 2.0)


def test_identity_listener_rotator_maps_to_wxyz_identity():
    assert _ue_rotator_to_m3_orientation_wxyz([0.0, 0.0, 0.0]) == (
        1.0,
        0.0,
        0.0,
        0.0,
    )


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
