"""The resting-pose check has to catch the lean the old metric could not see."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = REPOSITORY_ROOT / "tools/assets"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

import measure_static_resting_pose as resting  # noqa: E402
import publish_static_source_assets as publisher  # noqa: E402
from test_assets_static_upright import _box, _write_glb  # noqa: E402


def _rotation(degrees: float, axis: str) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    if axis == "y":
        return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def _write(path: Path, size, degrees: float = 0.0, axis: str = "x") -> Path:
    points, faces = _box(*size)
    _write_glb(path, (_rotation(degrees, axis) @ points.T).T, faces)
    return path


CABINET = (0.2, 0.33, 0.25)
# Thin, tall, wide: the long axis is the horizontal width, which is exactly the
# axis a backward lean turns about.
PANEL = (0.05, 0.6, 1.1)


def test_a_box_sitting_flat_is_level(tmp_path):
    pose = resting.measure(_write(tmp_path / "flat.glb", CABINET))
    assert pose["verdict"] == "level"
    assert pose["base_normal_tilt_deg"] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    "degrees,verdict",
    [(1.5, "level"), (2.9, "level"), (5.0, "acceptable"), (7.9, "acceptable"),
     (12.0, "leaning"), (26.0, "leaning")],
)
def test_the_bands_are_loose_but_not_blind(tmp_path, degrees, verdict):
    pose = resting.measure(_write(tmp_path / f"t{degrees}.glb", CABINET, degrees))
    assert pose["base_normal_tilt_deg"] == pytest.approx(degrees, abs=0.1)
    assert pose["verdict"] == verdict


def test_yaw_is_not_a_lean(tmp_path):
    pose = resting.measure(_write(tmp_path / "yaw.glb", CABINET, 37.0, "y"))
    assert pose["base_normal_tilt_deg"] == pytest.approx(0.0, abs=0.05)
    assert pose["verdict"] == "level"


def test_the_offset_a_room_needs_is_recorded(tmp_path):
    pose = resting.measure(_write(tmp_path / "flat.glb", CABINET))
    # A room places the asset by putting its origin here and nothing else. The
    # finalizer grounds assets so this is zero; the field exists so a future
    # asset that is not grounded still needs no per-room derivation.
    assert pose["base_plane_offset_m"] == pytest.approx(0.0, abs=1e-3)
    assert pose["footprint_extent_m"] == pytest.approx(
        [CABINET[0], CABINET[2]], abs=1e-3
    )


def test_it_catches_the_lean_the_long_axis_metric_reports_as_upright(tmp_path):
    """The regression this measure exists for.

    A flat panel tipped onto its back keeps a level width axis, so the
    area-weighted principal axis still reads horizontal and the old number
    passes it. This is not hypothetical: a published television scored 1.0
    degrees that way while its base was 26 degrees off.
    """

    path = _write(tmp_path / "panel.glb", PANEL, 26.0, "z")
    old = publisher.mesh_extent_and_tilt(path)

    assert old["long_axis_elevation_deg"] == pytest.approx(0.0, abs=0.2)
    assert old["resting_pose"]["base_normal_tilt_deg"] == pytest.approx(26.0, abs=0.2)
    assert old["resting_pose"]["verdict"] == "leaning"
