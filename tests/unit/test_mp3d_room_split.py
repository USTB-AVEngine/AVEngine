"""MP3D adaptation tests: wall axes, placement boundaries and native pose feet."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from shapely.geometry import box, Point
from shapely.affinity import rotate
from tools.rooms.room_split_auto import mp3d_adapter as a
from tools.rooms.room_split_auto.mp3d_render import projection


class ClearMesh:
    def __init__(self):
        self.ray = SimpleNamespace(
            receipt={"backend": "fixture"},
            intersects_location=lambda origins, directions, multiple_hits=False:
                (np.empty((0, 3)), np.array([], int), np.array([], int)))


class NativeFeet:
    def snap_point(self, point):
        return point
    def is_navigable(self, point):
        return True


PARAMETERS = {"camera_height_m": 1.5, "source_height_m": 1.2, "ray_endpoint_tolerance_m": .03}


def witness(camera=(1, 1.5, 1), source1=(2, 1.2, 1), source2=(3, 1.2, 2)):
    return {"found": True, "camera_m": list(camera), "source_1_m": list(source1), "source_2_m": list(source2)}


def test_witness_requires_real_outline_margin():
    g = box(0, 0, 4, 4)
    audit = a.validate_witness(g, witness(), ClearMesh(), NativeFeet(), PARAMETERS)
    assert audit["passed"]
    audit = a.validate_witness(g, witness(camera=(.1, 1.5, 1)), ClearMesh(), NativeFeet(), PARAMETERS)
    assert not audit["passed"]
    assert audit["poses"][0]["inside_real_filled_outline"]
    assert not audit["poses"][0]["margin_ok"]


def test_navigation_cannot_create_a_placement_bridge():
    g = box(0, 0, 2, 4).union(box(2.2, 0, 4.2, 4))
    w = witness(camera=(2.1, 1.5, 2), source1=(1, 1.2, 1), source2=(3, 1.2, 2))
    audit = a.validate_witness(g, w, ClearMesh(), NativeFeet(), PARAMETERS)
    assert not audit["passed"]
    assert not audit["poses"][0]["inside_real_filled_outline"]
    assert audit["poses"][0]["navmesh_foot_valid"]


def test_scan_holes_are_filled_for_placement():
    g = box(0, 0, 4, 4).difference(box(.8, .8, 1.2, 1.2))
    audit = a.validate_witness(g, witness(), ClearMesh(), NativeFeet(), PARAMETERS)
    assert audit["passed"]
    assert g.contains(Point(1, 1)) is False


def test_native_support_is_checked_at_pose_feet():
    class OffsetNativeFeet(NativeFeet):
        def snap_point(self, point):
            return np.asarray(point) + [0, .01, 0]
    audit = a.validate_witness(box(0, 0, 4, 4), witness(), ClearMesh(), OffsetNativeFeet(), PARAMETERS)
    assert not audit["passed"]
    assert all(not q["navmesh_foot_valid"] for q in audit["poses"])


def test_all_three_raw_mesh_rays_are_required():
    mesh = ClearMesh()
    def blocked(origins, directions, multiple_hits=False):
        return origins + .5 * directions, np.arange(len(origins)), np.zeros(len(origins), int)
    mesh.ray.intersects_location = blocked
    audit = a.validate_witness(box(0, 0, 4, 4), witness(), mesh, NativeFeet(), PARAMETERS)
    assert not audit["passed"]
    assert audit["three_scan_rays_clear"] == [False, False, False]


def test_wall_axis_uses_vertical_wall_triangles():
    angle = math.radians(17)
    v = np.array([math.cos(angle), 0, math.sin(angle)]) * 4
    start = np.array([2., 0, 2])
    tri = np.array([[start, start + v, start + v + [0, 2.4, 0]],
                    [start, start + v + [0, 2.4, 0], start + [0, 2.4, 0]]])
    axis = a.wall_axis([{"category": "wall", "triangles": tri}], box(0, 0, 8, 8), 0, 45)
    assert axis["verified_wall_mesh"]
    assert axis["primary_deg"] == pytest.approx(17)
    assert a.wall_axis([], box(0, 0, 8, 8), 0, 45)["primary_deg"] == 45


def test_owner_projection_matches_cpu_raster_axes():
    frame = projection(10, 20, 3, 20, 1000)
    clip = np.array(frame["projection"]) @ np.array(frame["view"])
    c = clip @ np.array([15., 3., 25., 1.])
    pixel = np.array([(c[0] + 1) * 500, (1 - c[1]) * 500])
    assert pixel == pytest.approx([750, 750])


def test_other_semantic_rooms_do_not_supply_nav_bridges():
    objects = SimpleNamespace(ground={
        1: [{"polygon": box(0, 0, 2, 2), "ys": 0.}],
        2: [{"polygon": box(2, 0, 3, 2), "ys": 0.}],
        3: [{"polygon": box(1, 0, 4, 2), "ys": 4.}],
    })
    foreign = a.foreign_floor_geometry(objects, 1, 0, box(0, 0, 4, 2),
                                      {"floor_height_separation_m": .3})
    assert foreign.equals(box(2, 0, 3, 2))
    assert foreign.intersection(box(0, 0, 2, 2)).area == 0


def test_area_bins_cover_exact_cap_and_ten_twenty_boundaries():
    from tools.rooms.room_split_auto.mp3d_delivery import area_bins
    bins = area_bins([{"floor_area_m2": q} for q in [6, 9.9, 10, 19.9, 20, 29.9, 30, 35]])
    assert bins == {"6_10": 2, "10_20": 2, "20_30": 2, "30_35": 2}


def test_exact_pair_join_is_local_and_uses_navmesh_only_between_parts():
    g = box(0, 0, 2, 3).union(box(2.2, 0, 4.2, 3))
    ctx = a.PairConnectivity(box(-1, -1, 5, 4))
    assert len(ctx.groups(g)) == 1
    link = ctx.certificate(g)["links"][0]
    from shapely.geometry import shape
    support = shape(link["nav_support_geometry_xz_m"])
    expected = box(0, 0, 2, 3).buffer(.3, join_style=2).intersection(
        box(2.2, 0, 4.2, 3).buffer(.3, join_style=2))
    assert support.difference(expected).area < 1e-8


def test_other_room_floor_cannot_complete_a_pair_bridge():
    g = box(0, 0, 2, 3).union(box(2.2, 0, 4.2, 3))
    ctx = a.PairConnectivity(box(-1, -1, 5, 4), box(2, -1, 2.2, 4))
    assert len(ctx.groups(g)) == 2


def test_navmesh_never_widens_an_internal_neck_in_certificate():
    g = box(0, 0, 3, 3).union(box(4, 0, 7, 3)).union(box(3, 1.3, 4, 1.7))
    ctx = a.PairConnectivity(box(-1, -1, 8, 4))
    assert len(ctx.groups(g)) == 1
    cert = ctx.certificate(g)
    assert cert["opened_components"] == 2
    assert cert["links"] == []


def test_delivery_ledger_and_draft_serialize_originals_and_cut_sources_once(tmp_path):
    import json
    from shapely.geometry import mapping
    from tools.rooms.room_split_auto import mp3d_delivery as delivery
    from tools.rooms.room_split_auto.mp3d_run import dump
    root = tmp_path
    original_dir = root / "mp3d_existing_rooms_connectivity_v1"
    cut_dir = root / "mp3d_delivery_v1/final_v1"
    rows = []
    for i in range(235):
        large = i < 22
        house, label = "mp3d_fixture", "R" + str(i)
        original = box(0, 0, 10, 4) if large else box(0, 0, 4, 4)
        room = dict(house=house, room_label=label, room_id=house + "/" + label, region_id=i,
                    selected_floor_id="F0", floor_y_m=0., height_range_m=[0, 0], ground_face_count=2,
                    floor_area_m2=original.area, short_side_m=4., floor_polygon_xz_m=mapping(original),
                    source_list_name="strict.csv", floor_area_method="unit fixture",
                    semantic_source="unit fixture", scene_directory="unit fixture", annotation_source="unit fixture",
                    navmesh_source="unit fixture")
        rows.append(room)
        base = a.source_region(room)
        base.update(requires_split=False, status="delegated_to_cap35_split" if large else "unchanged",
                    metrics={"affected": False})
        if not large:
            block = dict(id=house + "__" + label + "__F0__E000", house=house, source_region=label,
                         source_region_id=i, floor_id="F0", floor_y_m=0., floor_area_m2=16.,
                         floor_polygon_xz_m=mapping(original), short_side_m=4., decision="retain", native_main=True,
                         placement_witness={"found": True, "validation": {"passed": True}},
                         discard_reasons=[], unresolved_reasons=[], source="original", leakage=None)
            base["blocks"] = [block]
            dump(original_dir / "rooms" / (block["id"] + ".json"), block)
        dump(original_dir / "regions" / (house + "__" + label + ".json"), base)
        if large:
            cut = a.source_region(room)
            for j, geom in enumerate((box(0, 0, 5, 4), box(5, 0, 10, 4))):
                b = dict(id=house + "__" + label + "__F0__M" + str(j), house=house, source_region=label,
                         source_region_id=i, floor_id="F0", floor_y_m=0., floor_area_m2=20.,
                         floor_polygon_xz_m=mapping(geom), short_side_m=4., decision="retain", new_room=True,
                         placement_witness={"found": True, "validation": {"passed": True}},
                         discard_reasons=[], unresolved_reasons=[], source="cap_cut_new", leakage=None)
                cut["blocks"].append(b)
                dump(cut_dir / "rooms" / (b["id"] + ".json"), b)
            dump(cut_dir / "regions" / (house + "__" + label + ".json"), cut)
    dump(root / "plan.json", {"jobs": [{"rooms": rows}], "prep": "unit fixture"})
    dump(cut_dir / "completed.json", {"source_commit": "unit fixture"})
    delivery.summarize(root)
    summary = json.loads((root / "mp3d_room_list_draft_v1/summary.json").read_text())
    assert summary["total_rooms"] == 257
    assert summary["area_bins"] == {"6_10": 0, "10_20": 213, "20_30": 44, "30_35": 0}
    assert summary["pending"] == 0
    check = json.loads((root / "geometry_and_witness_validation.json").read_text())
    assert check["passed"]
    assert check["original_rooms"] == 235
    assert check["max_area_error_m2"] == 0
    assert len(json.loads((root / "area_ledger.json").read_text())) == 235
