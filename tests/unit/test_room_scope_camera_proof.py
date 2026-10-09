"""Geometric checks for truthful full-house native camera proof annotations."""
import numpy as np
import pytest
from shapely.geometry import Polygon,Point,MultiPolygon
from tools.rooms.room_split_auto.camera_view_proof import camera_basis,filled_footprint,project,unproject

pytestmark = pytest.mark.fast_unit

def test_depth_is_camera_plane_distance_with_nonsquare_pixels_and_translation():
    projection=np.array([[1.,0.,0.,0.],[0.,2.,0.,0.],[0.,0.,-1.001,-.1],[0.,0.,-1.,0.]])
    view=np.eye(4);view[:3,3]=[-7.,2.,-3.]
    depth=np.full((2,4),2.)
    world=unproject(depth,projection,view)
    np.testing.assert_allclose(world[0,0],[5.5,-1.5,1.])
    np.testing.assert_allclose(world[1,3],[8.5,-2.5,1.])
    pixels,_=project(world.reshape(-1,3),projection,view,4,2)
    np.testing.assert_allclose(pixels,[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]])

@pytest.mark.parametrize("forward",[[0,0,-1],[1,0,0],[0,0,1],[1,-.1,.25],[-1,-.1,-.25]])
def test_camera_rotation_points_negative_local_z_toward_requested_direction(forward):
    rotation=camera_basis(forward);unit=np.array(forward,dtype=float);unit/=np.linalg.norm(unit)
    np.testing.assert_allclose(rotation@[0.,0.,-1.],unit)
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
    assert np.linalg.det(rotation)==pytest.approx(1.)
    assert rotation[1,1]>0

def test_semantic_furniture_holes_do_not_count_as_adjacent_rooms():
    room=Polygon([(0,0),(4,0),(4,4),(0,4)],holes=[[(1,1),(3,1),(3,3),(1,3)]])
    conservative=filled_footprint(room).buffer(.20)
    assert conservative.covers(Point(2,2))
    assert conservative.covers(Point(4.1,2))
    assert not conservative.covers(Point(4.25,2))
    compound=MultiPolygon([room,Polygon([(6,0),(8,0),(8,2),(6,2)])])
    assert filled_footprint(compound).covers(Point(7,1))
    assert not filled_footprint(compound).covers(Point(5,1))
