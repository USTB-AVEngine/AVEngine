"""The upright measurement has to recover a tilt it was never told about."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = REPOSITORY_ROOT / "tools/assets"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

import measure_static_upright_correction as upright  # noqa: E402


def _box(width: float, height: float, depth: float) -> tuple[np.ndarray, np.ndarray]:
    half_x, half_z = width / 2.0, depth / 2.0
    points = np.array(
        [
            [-half_x, 0.0, -half_z],
            [half_x, 0.0, -half_z],
            [half_x, 0.0, half_z],
            [-half_x, 0.0, half_z],
            [-half_x, height, -half_z],
            [half_x, height, -half_z],
            [half_x, height, half_z],
            [-half_x, height, half_z],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],      # base, winding down
            [4, 5, 6], [4, 6, 7],      # top
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return points, faces


def _write_glb(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    position = points.astype("<f4").tobytes()
    index = faces.astype("<u4").reshape(-1).tobytes()
    buffer = position + index
    padding = (-len(buffer)) % 4
    buffer += b"\0" * padding
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(points),
                "type": "VEC3",
                "min": points.min(0).tolist(),
                "max": points.max(0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": faces.size,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position)},
            {"buffer": 0, "byteOffset": len(position), "byteLength": len(index)},
        ],
        "buffers": [{"byteLength": len(buffer)}],
    }
    encoded = json.dumps(gltf).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(buffer)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(buffer), 0x004E4942)
        + buffer
    )


def _tilted(path: Path, degrees: float, axis: str = "x") -> None:
    points, faces = _box(0.2, 0.33, 0.25)
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        rotation = np.array(
            [[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=float
        )
    else:
        # glTF is y-up, so yaw turns about y and must not read as tilt at all.
        rotation = np.array(
            [[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=float
        )
    _write_glb(path, (rotation @ points.T).T, faces)


def _measure(path: Path, **overrides):
    arguments = {
        "slab": 0.005,
        "bucket_deg": 5.0,
        "tolerance_deg": 1.5,
        "maximum_tilt_deg": 30.0,
    }
    arguments.update(overrides)
    return upright.measure(path, **arguments)


@pytest.mark.parametrize("degrees", [0.0, 3.0, 11.7, 18.7])
def test_recovers_the_tilt_it_was_given(tmp_path, degrees):
    path = tmp_path / "box.glb"
    _tilted(path, degrees)
    report = _measure(path)
    assert report["agreed"], report.get("refusal")
    assert report["tilt_from_upright_deg"] == pytest.approx(degrees, abs=0.5)
    assert report["authority_disagreement_deg"] <= 1.5


def test_the_correction_actually_stands_the_object_up(tmp_path):
    path = tmp_path / "box.glb"
    _tilted(path, 14.0)
    report = _measure(path)
    rotation = np.array(report["correction"]["matrix_gltf"], dtype=float)
    points, _faces = upright.load_glb(path)
    upright_points = (rotation @ points.T).T
    # The base of a standing box is one horizontal plane.
    lowest = upright_points[:, 1].min()
    base = upright_points[upright_points[:, 1] <= lowest + 1.0e-6]
    assert len(base) == 4
    assert base[:, 1].max() - base[:, 1].min() < 1.0e-6


def test_a_tilt_past_the_plausible_bound_is_refused(tmp_path):
    path = tmp_path / "box.glb"
    _tilted(path, 45.0)
    report = _measure(path)
    assert not report["agreed"]
    assert "plausibly" in report["refusal"]


def test_yaw_alone_is_not_reported_as_tilt(tmp_path):
    path = tmp_path / "box.glb"
    _tilted(path, 30.0, axis="y")
    report = _measure(path)
    assert report["agreed"], report.get("refusal")
    assert report["tilt_from_upright_deg"] == pytest.approx(0.0, abs=0.5)
