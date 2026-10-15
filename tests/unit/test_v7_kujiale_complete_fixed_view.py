import numpy as np
import pytest
from avengine.capture.kujiale_pilot_environment import construct_dialogue_camera
pytestmark=pytest.mark.fast_unit


def real_failed_group():
    return ([-2.125,1.5099949836730957,9.125],
            [[-.2251027226448059,1.3,9.117323875427246],[-.7325805425643921,1.3,8.20705795288086]],
            [[-.23258057236671448,1.3,7.207057952880859]])


def offset(listener,point,yaw):
    ray=np.asarray(point)-listener
    return (np.degrees(np.arctan2(ray[0],-ray[2]))-yaw+180)%360-180


def test_actual_0031_group_keeps_third_body_inside_original_85_degree_view():
    eye,pair,extra=real_failed_group();old=construct_dialogue_camera(eye,pair)
    assert abs(offset(eye,extra[0],old['yaw_deg']))>42.5
    new=construct_dialogue_camera(eye,pair,additional_positions=extra)
    assert all(abs(offset(eye,p,new['yaw_deg']))<=42.5 for p in pair+extra)
    assert len(set(new['source_sectors']))==2 and not new['qa_sector_gate_changed']
    assert new['native_hfov_deg']==85.


def test_mirrored_arbitrary_room_geometry_keeps_all_required_bodies():
    eye,pair,extra=real_failed_group();eye[0]*=-1
    for p in pair+extra:p[0]*=-1
    result=construct_dialogue_camera(eye,pair,additional_positions=extra)
    assert all(abs(offset(eye,p,result['yaw_deg']))<=42.5 for p in pair+extra)


def test_incompatible_extra_body_requires_new_geometry_without_dropping_it():
    with pytest.raises(ValueError,match='required-body layout'):
        construct_dialogue_camera([0,1.5,0],[[-1,1.3,-3],[1,1.3,-3]],additional_positions=[[0,1.3,3]])
