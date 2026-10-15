import math
import numpy as np
import pytest
from avengine.timeline.v7_spatial_layout import separated_layout,spacing_rejection,noncollinear_rejection,body_yaw
pytestmark=pytest.mark.fast_unit

def actor(name,kind,root,route=None):
    value={'endpoint_id':name,'declaration':{'kind':kind},'resolved_root_m':root}
    if route is not None:value['motion_episode']={'world_roots_m':route}
    return value

def test_person_spacing_checks_the_complete_walking_track():
    person={'kind':'human'};route=[[3.,0.,0.],[1.2,0.,0.],[.5,0.,0.]]
    other=actor('walker','human',route[0],route)
    rejected=spacing_rejection(person,[0.,0.,0.],[other])
    assert rejected['minimum_centre_distance_m']==pytest.approx(.5)
    assert rejected['required_distance_m']==1.
    assert spacing_rejection(person,[0.,0.,2.],[other]) is None

def test_animal_person_spacing_and_unplaced_entries():
    other=actor('person','human',[0.,0.,0.])
    assert spacing_rejection({'kind':'animal'},[.7,0.,0.],[other])
    assert spacing_rejection({'kind':'animal'},[.8,0.,0.],[other]) is None
    other['phase2_layout_qualified']=False
    assert spacing_rejection({'kind':'animal'},[0.,0.,0.],[other]) is None

def test_third_person_does_not_extend_a_single_row():
    others=[actor('a','human',[0.,0.,0.]),actor('b','human',[2.,0.,0.])]
    assert noncollinear_rejection({'kind':'human'},[4.,0.,0.],others)
    assert not noncollinear_rejection({'kind':'human'},[2.,0.,2.],others)

def test_human_headings_are_diverse_and_do_not_all_face_camera():
    values=[body_yaw(0.,{'kind':'human'},i) for i in range(5)]
    assert len({round(v,3) for v in values})==5
    assert values[0]==pytest.approx(0.)
    assert max(abs(v)for v in values)>math.radians(120)
    assert sum(abs(v)<math.radians(40)for v in values)<len(values)

def test_automatic_layout_keeps_requested_people_and_uses_room_depth():
    layout={'listener_m':[0.,1.3,0.],'room_floor_polygon':{'type':'Polygon','coordinates':[[[-3.,-3.],[3.,-3.],[3.,3.],[-3.,3.],[-3.,-3.]]]},'valid':[{'position_m':[0.,1.3,-2.]}]}
    result=separated_layout(layout,5);points=np.asarray([p['position_m'] for p in result])[:,[0,2]]
    assert len(result)==5
    assert min(np.linalg.norm(a-b)for i,a in enumerate(points)for b in points[i+1:])>=1.
    assert np.linalg.svd(points-points.mean(0),compute_uv=False)[1]>=.25

def test_too_small_room_fails_without_silently_removing_people():
    layout={'listener_m':[0.,1.3,0.],'room_floor_polygon':{'type':'Polygon','coordinates':[[[0,0],[1,0],[1,1],[0,1],[0,0]]]},'valid':[{'position_m':[.5,1.3,.5]}]}
    with pytest.raises(ValueError,match='requested count'):separated_layout(layout,5)
