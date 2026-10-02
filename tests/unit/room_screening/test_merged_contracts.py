"""Regression checks for shared multi-floor measurement and independent review."""

import json
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import box
from tools.rooms.room_screening.geometry import (
    parse_semantic_annotations,
    parse_semantic_labels,
)
from tools.rooms.room_screening.review_server import ReviewApp, read_manifest
from tools.rooms.room_screening.build_review_manifest import build_manifest
from tools.rooms.room_screening import audit_furniture_obstacles as audit
from tools.rooms.room_selection.measurements import navigation_geometry
from tools.rooms.room_selection.run import load_parameters

pytestmark = pytest.mark.fast_unit


def test_duplicate_palette_keeps_both_regions_without_assigning_conflicting_faces(
    tmp_path,
):
    path = tmp_path / "semantic.txt"
    path.write_text('HM3D\n1,ABCDEF,"floor",2\n2,ABCDEF,"sofa",7\n3,FF0000,"bed",7\n')
    instances, colours, regions = parse_semantic_annotations(path)
    assert set(instances) == {1, 2, 3}
    assert set(regions) == {2, 7}
    assert colours[int("ABCDEF", 16)] is None
    assert int("ABCDEF", 16) not in parse_semantic_labels(path)
    assert regions[2]["ambiguous_colours"] == ["ABCDEF"]
    assert regions[7]["ambiguous_colours"] == ["ABCDEF"]


def test_video_root_and_split_overlay_are_explicit_and_stay_within_declared_roots(
    tmp_path,
):
    assets = tmp_path / "assets"
    assets.mkdir()
    media = tmp_path / "tours"
    media.mkdir()
    (assets / "rgb.png").write_bytes(b"unit fixture")
    (assets / "split.png").write_bytes(b"unit fixture")
    (media / "tour.mp4").write_bytes(b"unit fixture")
    manifest = build_manifest(
        dict(
            items=[
                dict(
                    id="case",
                    image_path="rgb.png",
                    video_path=str(media / "tour.mp4"),
                    split_overlay_paths=["split.png"],
                    automatic=dict(status="pass", reason_codes=[]),
                    second_review=True,
                )
            ]
        ),
        assets,
        media,
    )
    assert manifest["items"][0]["video_asset_root"] == "media"
    assert manifest["items"][0]["split_overlay_files"] == ["split.png"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert (
        read_manifest(path, assets, media)["items"][0]["automatic"]["status"] == "pass"
    )
    with pytest.raises(ValueError):
        read_manifest(path, assets)


def test_blind_second_mode_removes_hints_at_api_boundary(tmp_path):
    (tmp_path / "rgb.png").write_bytes(b"unit fixture")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version="room_screening_review_manifest_v1",
                items=[
                    dict(
                        id="second",
                        image_file="rgb.png",
                        second_review=True,
                        automatic={"status": "pass"},
                        gate={"status": "fail"},
                        metrics=[{"value": 20}],
                        split_overlay_files=[],
                    ),
                    dict(id="first", image_file="rgb.png", second_review=False),
                ],
            )
        )
    )
    app = ReviewApp(path, tmp_path, tmp_path / "feedback.json", blind_second=True)
    assert len(app.manifest["items"]) == 1
    item = app.manifest["items"][0]
    assert item["id"] == "second"
    assert all(
        k not in item for k in ["automatic", "gate", "metrics", "split_overlay_files"]
    )
    assert app.manifest["blind_second_reviewer"] is True
    assert not (tmp_path / "feedback.json").exists()


def test_missing_overhead_is_explicit_not_fake_scan_image(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = build_manifest(
        dict(items=[dict(id="missing", image_missing_reason="capture unavailable")]),
        tmp_path,
    )
    path.write_text(json.dumps(manifest))
    assert (
        read_manifest(path, tmp_path)["items"][0]["image_missing_reason"]
        == "capture unavailable"
    )
    assert "image_file" not in manifest["items"][0]


def test_furniture_audit_does_not_overwrite_first_floor_of_same_region(
    tmp_path, monkeypatch
):
    house = "hm3d_test_00001_synthetic"
    scene = tmp_path / "test/00001-synthetic"
    scene.mkdir(parents=True)
    (scene / "synthetic.basis.navmesh").write_bytes(b"unit fixture only")
    ground = {
        0: [
            dict(polygon=box(0, 0, 2, 2), ys=0, category="floor"),
            dict(polygon=box(0, 0, 2, 2), ys=3, category="floor"),
        ]
    }
    monkeypatch.setattr(
        audit,
        "load_semantic_ground_and_instances",
        lambda *a: (ground, {0}, [], np.empty((0, 3, 3)), 0, 0, {}, 2, 2),
    )
    monkeypatch.setattr(
        audit.base,
        "navmesh_triangles",
        lambda *a: (None, [box(0, 0, 2, 2), box(0, 0, 2, 2)], np.array([0.1, 3.1])),
    )
    settings = dict(agent_height=1.5, agent_radius=0.1, cell_size=0.05, cell_height=0.1)
    raw = dict(
        regions=[
            dict(
                region_id=0,
                floor_id_candidate=0,
                navmesh_floor_cluster_index=0,
                candidate_semantic_region_nav_projected_area_m2=4,
            ),
            dict(
                region_id=0,
                floor_id_candidate=1,
                navmesh_floor_cluster_index=1,
                candidate_semantic_region_nav_projected_area_m2=4,
            ),
        ]
    )
    result = audit.process_house(
        None,
        dict(house=house, navmesh=dict(navmesh_settings=settings)),
        raw,
        tmp_path,
        footprint_method="shape_preserving",
    )
    assert [
        (r["region_id"], r["floor_id_candidate"]) for r in result["region_floor_rows"]
    ] == [(0, 0), (0, 1)]
    assert [r["raw_candidate_area_m2"] for r in result["region_floor_rows"]] == [4, 4]


def test_navmesh_area_excludes_another_storey_and_clips_scope():
    p = load_parameters(
        Path(__file__).parents[3] / "tools/rooms/room_selection/thresholds.yaml"
    )[1]
    result = navigation_geometry(
        [box(0, 0, 3, 3), box(0, 0, 5, 5)],
        np.array([0.1, 3.1]),
        0,
        box(0, 0, 2, 2),
        p,
        box(0, 0, 2, 2),
    )
    assert result["nav_triangle_intersection_area_m2"] == 4
    assert result["nav_main_area_m2"] == 4


def test_native_connected_support_retains_scan_fragments_instead_of_largest_polygon():
    p = load_parameters(
        Path(__file__).parents[3] / "tools/rooms/room_selection/thresholds.yaml"
    )[1]
    # A sub-grid scan crack splits the projected floor, but native adjacency
    # still supports both sides of that same connected component.
    scope = box(0, 0, 1, 2).union(box(1.01, 0, 2, 2))
    result = navigation_geometry(
        [box(0, 0, 2, 2)], np.array([0.1]), 0, scope, p, box(0, 0, 2, 2)
    )
    assert result["nav_main_area_m2"] == pytest.approx(3.98)
    assert result["nav_largest_projected_polygon_area_m2"] == 2


def test_inventory_retains_annotation_region_without_mapped_mesh(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.rooms.room_screening import build_inventory as inventory

    scene = tmp_path / "test/00001_synthetic"
    # HM3D filesystem names use a dash, not the public house separator.
    scene = tmp_path / "test/00001-synthetic"
    scene.mkdir(parents=True)
    for suffix in ["semantic.glb", "basis.navmesh"]:
        (scene / f"synthetic.{suffix}").write_bytes(b"unit fixture")
    (scene / "synthetic.semantic.txt").write_text(
        'HM3D\n1,FF0000,"floor",0\n2,00FF00,"wall",7\n'
    )
    monkeypatch.setattr(
        inventory,
        "inventory_rooms",
        lambda *a: dict(rooms=[dict(region_id=0, floor_area_m2=4)]),
    )

    class Pathfinder:
        nav_mesh_settings = SimpleNamespace(agent_height=1.5, cell_height=0.1)

        def load_nav_mesh(self, path):
            return True

    result = inventory.build_inventory(
        ["hm3d_test_00001_synthetic"], tmp_path, SimpleNamespace(PathFinder=Pathfinder)
    )
    assert result["region_count"] == 2
    missing = result["regions"][1]
    assert missing["region_id"] == 7
    assert missing["source_floor_area_m2"] is None
    assert missing["geometry_inventory_status"] == "unavailable"
