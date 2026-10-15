import numpy as np
import pytest
from avengine.timeline.v7_spatial_layout import person_camera_rejection,native_person_camera_clearance
pytestmark=pytest.mark.fast_unit

def test_entire_human_route_keeps_the_construction_margin():
    assert person_camera_rejection({'kind':'human'},[[3,0,0],[1.49,0,0]],[0,1.3,0])
    assert person_camera_rejection({'kind':'human'},[[3,0,0],[1.5,0,0]],[0,1.3,0])is None
    assert person_camera_rejection({'kind':'animal'},[[.9,0,0]],[0,1.3,0])is None

def visual(distances):
    frames=[];cameras=[]
    for i,distance in enumerate(distances):
        matrix=np.eye(4);matrix[0,3]=distance
        frames.append({'frame_index':i,'actors':[{'asset_id':'person','body_to_world_m':matrix.tolist(),'native_emitter_present':False}]})
        cameras.append({'frame_index':i,'position_m':[0,1.3,0]})
    return {'frames':frames,'camera_frame_readbacks':cameras,'frame_count':len(frames)}

def test_one_close_frame_fails_even_when_start_and_end_are_far():
    assert native_person_camera_clearance(visual([2.,1.19,2.]),[{'asset_id':'person','kind':'human'}])['status']=='fail'
    assert native_person_camera_clearance(visual([2.,1.2,2.]),[{'asset_id':'person','kind':'human'}])['status']=='pass'

def test_missing_readback_cannot_pass_distance_gate():
    value=visual([2.,2.]);value['camera_frame_readbacks'].pop()
    assert native_person_camera_clearance(value,[{'asset_id':'person','kind':'human'}])['status']=='fail'
