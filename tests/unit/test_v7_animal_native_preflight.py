from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.animal_motion.native_preflight import native_animal_preflight

pytestmark=pytest.mark.fast_unit


def fixture(monkeypatch, pixels=150, error=False):
    from avengine.capture import continuous_native_visibility as visibility
    state={'applied':[],'closed':False}
    route={'animal_native_preflight_required':True,'require_rest_poses_in_frame':True,
           'emission_slot':{'start_s':1.3,'end_s':3.6,'post_until_s':5.9}}
    entry={'endpoint_id':'voice1','declaration':{'asset_id':'animal','native_instance_id':1},
           'motion_episode':{'world_roots_m':np.zeros((901,3)).tolist()}}
    sim=SimpleNamespace(get_sensor_observations=lambda:{'semantic':np.zeros((2,2),int)})
    video=SimpleNamespace(actors=[entry],sim=sim,_apply_motion_entry=lambda e,index:state['applied'].append(index),
                          _native_pair_pixels=lambda scene,entries:None)
    class Stream:
        def __init__(self,*args):pass
        def capture(self,index,modal,identities):
            if error:raise RuntimeError('native mask control failed')
            return {'per_instance':{'1':{'frames':[{'visible_pixels':pixels}]}}}
        def close(self):state['closed']=True
        def receipt(self):return {'status':'hermetic_control_fixture'}
    monkeypatch.setattr(visibility,'NativeVisibilityStream',Stream)
    scene={'production_scenario':{'motion_requests':{'voice1':{'motion_quartet_route':route}}}}
    return video,scene,[entry],state


@pytest.mark.parametrize('pixels,status',[(100,'pass'),(99,'fail')])
def test_pixel_witness_uses_final_context_and_restores_frame_zero(monkeypatch,pixels,status):
    video,scene,entries,state=fixture(monkeypatch,pixels)
    result=native_animal_preflight(video,scene,entries,rest_framing_fn=lambda *args:{'state':'inside'})
    assert result['status']==status
    assert result['final_complete_stream_still_required']
    assert state['applied'][-1]==0 and state['closed']
    assert len(result['measured_frames'])>=4


def test_native_control_failure_restores_actual_gait_before_propagation(monkeypatch):
    video,scene,entries,state=fixture(monkeypatch,error=True)
    with pytest.raises(RuntimeError,match='native mask control'):
        native_animal_preflight(video,scene,entries,rest_framing_fn=lambda *args:{'state':'inside'})
    assert state['applied'][-1]==0 and state['closed']


def test_actual_stationary_edge_fails_before_expensive_full_media(monkeypatch):
    video,scene,entries,state=fixture(monkeypatch)
    result=native_animal_preflight(video,scene,entries,rest_framing_fn=lambda *args:{'state':'edge'})
    assert result['status']=='fail'
    assert any('stationary_body' in reason for reason in result['reasons'])
