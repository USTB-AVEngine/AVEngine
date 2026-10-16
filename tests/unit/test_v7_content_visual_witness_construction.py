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


def test_geometry_repair_preserves_people_voices_and_exact_emission_schedule():
    from avengine.timeline.v7_content_offscreen_construction import repair_registered_content_geometry
    old={'scene_id':'failed','scene_seed':42,'asset_ids':['a','b','c','d'],'semantic_sound_ids':['whole_a','whole_b','whole_c'],'semantic_start_offsets_s':[3.,6.5,0.],'production_scenario':{'semantic_epoch_sound_ids':[['a1','b1','c1']]},'automatic_construction':{}}
    ctx={'room_floor_polygon':box(-6,-6,6,6).__geo_interface__,'listener_m':[0.,1.2,0.],'source_proposals':[[1.,1.5,2.],[-1.,1.5,2.]]}
    actual=repair_registered_content_geometry(old,ctx,scene_id='fresh_attempt')
    for key in ['asset_ids','semantic_sound_ids','semantic_start_offsets_s']:
        assert actual[key]==old[key]
    assert actual['production_scenario']['semantic_epoch_sound_ids']==old['production_scenario']['semantic_epoch_sound_ids']
    assert actual['scene_seed']==42 and actual['scene_id']=='fresh_attempt'and old['scene_id']=='failed'
