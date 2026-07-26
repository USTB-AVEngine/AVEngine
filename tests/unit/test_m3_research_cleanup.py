from __future__ import annotations

import numpy as np
import pytest

from avengine.m3.research_cleanup import (
    CLEANUP_POLICY,
    DERIVED_PACKAGE_SUFFIX,
    ResearchCleanupError,
    _derived_package_id,
    filter_research_geometry,
)


def _objects() -> list[dict[str, object]]:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return [
        {
            "object_id": "surface",
            "source_material_name": "wall",
            "source_mesh_index": 0,
            "source_node_index": 0,
            "source_primitive_index": 0,
            "source_world_matrix": identity,
            "transform_baked": True,
            "vertex_offset": 0,
            "vertex_count": 3,
            "triangle_offset": 0,
            "triangle_count": 1,
            "world_from_object": identity,
        },
        {
            "object_id": "empty_scan_fragment",
            "source_material_name": "wall",
            "source_mesh_index": 1,
            "source_node_index": 1,
            "source_primitive_index": 0,
            "source_world_matrix": identity,
            "transform_baked": True,
            "vertex_offset": 3,
            "vertex_count": 3,
            "triangle_offset": 1,
            "triangle_count": 1,
            "world_from_object": identity,
        },
    ]


def test_cleanup_derived_package_identity_tracks_v2_policy() -> None:
    assert CLEANUP_POLICY.endswith("_v2")
    assert DERIVED_PACKAGE_SUFFIX.endswith("_v2")
    assert (
        _derived_package_id("kujiale_0020")
        == "kujiale_0020_rlr_incompatible_filter_v2"
    )
    assert "_v1" not in _derived_package_id("kujiale_0020")


def test_filter_removes_only_fully_degenerate_object_and_reindexes() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype="<f4",
    )
    triangles = np.asarray([[0, 1, 2], [3, 3, 4]], dtype="<u4")
    materials = np.asarray([0, 0], dtype="<u4")
    result = filter_research_geometry(vertices, triangles, materials, _objects())
    assert np.array_equal(result.vertices, vertices[:3])
    assert np.array_equal(result.triangles, [[0, 1, 2]])
    assert np.array_equal(result.material_ids, [0])
    assert [item["object_id"] for item in result.objects] == ["surface"]
    assert result.record["removed_triangle_count"] == 1
    assert result.record["removed_vertex_count"] == 3
    assert result.record["removed_objects"][0]["object_id"] == "empty_scan_fragment"
    assert len(result.record["record_content_sha256"]) == 64


def test_filter_rejects_package_without_degenerate_faces() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype="<f4",
    )
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    with pytest.raises(ResearchCleanupError, match="no RLR-incompatible"):
        filter_research_geometry(
            vertices, triangles, np.asarray([0], dtype="<u4"), _objects()[:1]
        )


def test_filter_rejects_removing_the_only_use_of_a_material() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype="<f4",
    )
    triangles = np.asarray([[0, 1, 2], [3, 3, 4]], dtype="<u4")
    with pytest.raises(ResearchCleanupError, match="material category"):
        filter_research_geometry(
            vertices, triangles, np.asarray([0, 1], dtype="<u4"), _objects()
        )


def test_filter_matches_native_rlr_near_zero_cross_product_rule() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.000001, 0.0, 0.0],
            [2.0, 0.000001, 0.0],
        ],
        dtype="<f4",
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="<u4")
    materials = np.asarray([0, 0], dtype="<u4")
    result = filter_research_geometry(vertices, triangles, materials, _objects())
    assert result.record["removed_triangle_count"] == 1
    assert result.record["rlr_cross_norm_squared_threshold_inclusive"] == 1.0e-20
    assert result.record["rlr_equivalent_area_threshold_m2_inclusive"] == 5.0e-11
    assert result.record["minimum_retained_cross_norm_squared"] > 1.0e-20
    assert result.record["removed_objects"] == [
        {
            "object_id": "empty_scan_fragment",
            "vertex_offset": 3,
            "vertex_count": 3,
            "triangle_offset": 1,
            "triangle_count": 1,
            "reason": "all_triangles_geometry_qa_or_native_rlr_incompatible",
        }
    ]
