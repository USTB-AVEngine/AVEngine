import math
import pytest
from shapely.geometry import box
from avengine.timeline.v7_offscreen_construction import joint_offscreen_floor
pytestmark=pytest.mark.fast_unit


def test_quartet_constructs_a_real_depth_witness_instead_of_waiving_vision_invariance():
    ctx={'room_floor_polygon':box(-6,-6,6,6).__geo_interface__,'listener_m':[0.,1.2,0.]}
    actual,yaw,inside,outside=joint_offscreen_floor(ctx,1,42,visible_people=3,minimum_visible_radial_difference_m=.95)
    assert inside[1][2]-inside[0][2]>=.95
    assert all(math.dist(a[:2],b[:2])>=1.05 for i,a in enumerate([*inside,*outside])for b in [*inside,*outside][i+1:])
    assert len(inside)==3 and len(outside)==1


def test_requested_depth_follows_physical_role_through_the_colour_swap():
    from avengine.timeline.v7_registered_sampling import registered_distance_band
    scene={'native_endpoint_distance_bands_m':{'voice1':[2.6,2.9],'voice2':[3.7,4.0]}}
    assert registered_distance_band(scene,'voice1','green')==registered_distance_band(scene,'voice1','yellow')==[2.6,2.9]
    assert registered_distance_band(scene,'voice2','yellow')==registered_distance_band(scene,'voice2','green')==[3.7,4.0]
