"""Ground-domain filtering for multi-level UE navigation banks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "routes"))

from filter_route_bank_by_ground import filter_routes  # noqa: E402


def route(route_id, z_values):
    return {"route_id": route_id,
            "waypoints_ue_cm": [[float(i), 0.0, z]
                                 for i, z in enumerate(z_values)]}


def test_filter_keeps_only_the_declared_ground_domain():
    bank = {"schema": "avengine_apartment_route_bank_v1", "routes": [
        route("ground", [0.0, 0.4, 1.0]),
        route("upper", [156.0, 156.0]),
        {"route_id": "legacy-no-z", "waypoints_ue_cm": [[0.0, 0.0]]},
    ]}
    kept, counts = filter_routes(
        bank, ground_z_ue_cm=0.0, tolerance_ue_cm=1.0)
    assert [item["route_id"] for item in kept] == ["ground"]
    assert counts == {
        "input_routes": 3, "kept_routes": 1,
        "rejected_missing_z": 1, "rejected_other_ground": 1,
    }
