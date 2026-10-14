import numpy as np
import pytest
from avengine.qa.kujiale_asset_geometry import triangle_box_overlap
pytestmark=pytest.mark.fast_unit


def test_wall_triangle_crossing_an_asset_is_detected_even_when_its_vertices_are_outside():
    triangles=np.array([[[0,-10,-10],[0,10,-10],[0,0,10]],[[3,-10,-10],[3,10,-10],[3,0,10]]])
    assert triangle_box_overlap(triangles,[-1,-1,-1],[1,1,1]).tolist()==[True,False]


def test_bbox_overlap_does_not_replace_triangle_geometry():
    triangles=np.array([[[0,3,0],[3,0,0],[3,3,0]],[[0,0,0],[3,0,0],[0,3,0]]])
    assert triangle_box_overlap(triangles,[-.1,-.1,-.1],[.1,.1,.1]).tolist()==[False,True]


def test_floor_contact_only_intersects_when_the_body_enters_the_floor():
    triangles=np.array([[[-2,0,-2],[2,0,-2],[0,0,2]]])
    assert not triangle_box_overlap(triangles,[-.5,.01,-.5],[.5,1,.5])[0]
    assert triangle_box_overlap(triangles,[-.5,-.01,-.5],[.5,1,.5])[0]
