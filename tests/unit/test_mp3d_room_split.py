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
