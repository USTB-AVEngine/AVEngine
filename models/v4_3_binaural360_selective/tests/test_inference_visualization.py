from pathlib import Path

import numpy as np
from PIL import Image

from avengine_v43.inference_visualization import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    circular_error_deg,
    compass_point,
    continuous_segments,
    render_panel_frame,
    signed_degrees,
)


def test_compass_uses_native_front_right_rear_left_convention():
    center = (100.0, 100.0)
    assert compass_point(0.0, center=center, radius=10.0) == (100.0, 90.0)
    assert compass_point(90.0, center=center, radius=10.0) == (110.0, 100.0)
    assert np.allclose(
        compass_point(180.0, center=center, radius=10.0),
        (100.0, 110.0),
    )
    assert np.allclose(
        compass_point(270.0, center=center, radius=10.0),
        (90.0, 100.0),
    )


def test_circular_error_and_signed_axis_wrap_are_correct():
    assert circular_error_deg([359.0, 1.0, 180.0], [1.0, 359.0, 0.0]).tolist() == [
        2.0,
        2.0,
        180.0,
    ]
    assert signed_degrees([0.0, 90.0, 180.0, 270.0]).tolist() == [
        0.0,
        90.0,
        -180.0,
        -90.0,
    ]


def test_track_splits_at_display_seam_only():
    assert continuous_segments([170.0, 179.0, 181.0, 190.0]) == [(0, 2), (2, 4)]
    assert continuous_segments([10.0, 20.0, 30.0]) == [(0, 3)]


def test_panel_renderer_writes_expected_static_surface(tmp_path: Path):
    target = np.vstack(
        [
            np.linspace(350.0, 20.0, 75),
            np.linspace(90.0, 180.0, 75),
        ]
    )
    prediction = (target + np.asarray([[5.0], [-8.0]])) % 360.0
    output = tmp_path / "frame.png"
    render_panel_frame(
        frame_index=37,
        targets_deg=target,
        predictions_deg=prediction,
        captions=["dog barking", "human speech"],
        sample_id="sample__v00",
        output_path=output,
    )
    with Image.open(output) as image:
        assert image.size == (PANEL_WIDTH, PANEL_HEIGHT)
        assert image.mode == "RGB"
