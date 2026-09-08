from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.acoustics.diagnose_region_ray_leakage import (
    _annotate_cpu_checks,
    _ray_declarations,
)


def test_region_probe_flatten_preserves_route_provenance_and_normalizes_direction():
    declarations = _ray_declarations(
        {
            "regions": [
                {
                    "region_id": "living",
                    "path_id": "walk0",
                    "rays": [
                        {
                            "ray_id": "r0",
                            "origin_m": [1.0, 1.0, 1.0],
                            "direction": [0.0, 0.0, 2.0],
                            "distance_m": 3.0,
                            "expectation": "hit_within_m",
                            "frame_index": 7,
                            "source_id": "source1",
                            "expected_object_id": "Wall_0",
                        }
                    ],
                }
            ]
        }
    )
    assert declarations[0]["region_id"] == "living"
    assert declarations[0]["path_id"] == "walk0"
    assert declarations[0]["frame_index"] == 7
    assert declarations[0]["expected_object_id"] == "Wall_0"
    assert declarations[0]["direction"] == [0.0, 0.0, 1.0]


def test_region_probe_rejects_zero_direction():
    try:
        _ray_declarations(
            [
                {
                    "check_id": "bad",
                    "origin_m": [0.0, 0.0, 0.0],
                    "direction": [0.0, 0.0, 0.0],
                    "distance_m": 1.0,
                    "expectation": "clear_until_m",
                }
            ]
        )
    except ValueError as error:
        assert "nonzero" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("zero ray direction was accepted")


def test_cpu_annotation_uses_reviewed_sidecar_to_package_mapping():
    scene = SimpleNamespace(
        objects=[
            {
                "object_id": "node39_mesh39_primitive0",
                "triangles": np.zeros((1, 3), dtype=np.uint32),
                "triangle_material_ids": np.zeros((1,), dtype=np.uint32),
            }
        ],
        material_categories=("wall",),
    )
    checks, status = _annotate_cpu_checks(
        [
            {
                "check_id": "r0",
                "status": "pass",
                "expectation": "hit_within_m",
                "expected_object_id": "Wall_010_pier_end",
                "measured_triangle_index": 0,
            }
        ],
        scene=scene,
        semantics={
            "Wall_010_pier_end": {
                "category": "wall",
                "kind": "Wall_010_pier_end",
            }
        },
        semantic_mapping={
            "Wall_010_pier_end": {
                "package_object_id": "node39_mesh39_primitive0",
                "source_node_index": 39,
                "triangle_offset": 8012,
                "triangle_count": 188,
                "triangle_end_exclusive": 8200,
                "sidecar_category": "wall",
                "sidecar_kind": "Wall_010_pier_end",
            }
        },
        declarations={},
    )
    assert status == "pass"
    assert checks[0]["semantic_match"] is True
    assert checks[0]["expected_package_object_id"] == "node39_mesh39_primitive0"
    assert checks[0]["expected_triangle_offset"] == 8012
    assert checks[0]["mapping_status"] == "resolved"
