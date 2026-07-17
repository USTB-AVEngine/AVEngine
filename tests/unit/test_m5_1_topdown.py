from __future__ import annotations

from pathlib import Path

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.m5_1.topdown import render_legacy_topdown_frame


ROOT = Path(__file__).resolve().parents[2]
ROUTE = ROOT / "examples" / "m5_1" / "legacy_apartment" / "route_manifest.json"


def test_legacy_topdown_frame_is_deterministic_and_draws_routes() -> None:
    manifest = load_json(ROUTE)
    first = render_legacy_topdown_frame(manifest, 134)
    second = render_legacy_topdown_frame(manifest, 134)
    assert first.shape == (240, 320, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 20
