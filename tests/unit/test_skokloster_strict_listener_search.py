from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

THIS_FILE = Path(__file__).resolve()
STAGING_LAYOUT = (THIS_FILE.parents[1] / "tools").is_dir()
ROOT = THIS_FILE.parents[1] if STAGING_LAYOUT else THIS_FILE.parents[2]
SCRIPT = (
    ROOT / "tools/search_skokloster_strict_listener.py"
    if STAGING_LAYOUT
    else ROOT / "tools/qa/search_skokloster_strict_listener.py"
)
SPEC = importlib.util.spec_from_file_location("skok_listener_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_old_listener_reproduces_mouths_but_rejects_full_body() -> None:
    camera = [3.93178129196167, 1.6583894044160843, 9.969188690185547]
    yaw = 79.91754325521669
    mouths = [
        MODULE.project_point(camera, yaw, point) for point in MODULE.SOURCE_MOUTHS
    ]
    assert mouths[0]["x_px"] == pytest.approx(402.1932585754115)
    assert mouths[1]["x_px"] == pytest.approx(933.1269193982973)
    assert MODULE.screen_projection(camera, yaw) is None


def test_coupled_farther_listener_passes_conservative_envelopes() -> None:
    camera = [5.4, 1.63, 9.65]
    yaw = MODULE.yaw_toward(camera, (2.55, 9.65))
    report = MODULE.screen_projection(camera, yaw)
    assert report is not None
    assert report["minimum_envelope_edge_margin_px"] >= 48.0
    assert report["mouth_projections"][0]["x_px"] <= 0.42 * 1280
    assert report["mouth_projections"][1]["x_px"] >= 0.58 * 1280


def test_listener_orientation_quaternion_convention() -> None:
    yaw = 90.0
    wxyz = [
        math.cos(math.radians(yaw) / 2.0),
        0.0,
        math.sin(math.radians(yaw) / 2.0),
        0.0,
    ]
    assert wxyz == pytest.approx([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
