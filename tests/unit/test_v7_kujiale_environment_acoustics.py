from pathlib import Path
import numpy as np
import pytest
from avengine.timeline.v7_kujiale_environment_acoustics import EnvironmentPositionAdmission
pytestmark=pytest.mark.fast_unit


def test_physical_construction_uses_the_original_doa_gate_and_exact_fixed_listener(tmp_path,monkeypatch):
    from avengine.timeline import v7_position_admission
    calls=[]
    class Gate:
        def __init__(self,scene,house,inputs,cache,receipt):
            assert scene['listener_m']==[1.,1.,2.] and house['room']['placement']['camera_m']==[1.,1.,2.]
            assert scene['camera_forward']==[0.,0.,-1.]
            assert cache==tmp_path/'native/parents/new_seed/rir_cache'
            assert inputs['simulation']['max_diffraction_order']==10
        def check(self,endpoint,point):
            calls.append((endpoint,point.tolist()))
            return {'status':'blocked'if point[0]==0 else'pass','unchanged_gate_deg':5.}
        def close(self):calls.append('closed')
    monkeypatch.setattr(v7_position_admission,'PositionAdmission',Gate)
    row={'scene_id':'new_seed','room_id':'R0','construction_request':{'required_sound_classes':['dog_bark']},
         'measured_static_layout':{'listener_m':[1.,1.,2.],'fixed_camera_yaw_deg':0.,'valid':[{'position_m':[2.,1.3,0.]}]}}
    house={'acoustic_package_manifest':'original_acoustic_scene','input':{'ue_config':{'output_root':str(tmp_path)}},
           'checked_rooms':[{'room_label':'R0','placement':{'camera_m':[9,9,9]}}]}
    checker=EnvironmentPositionAdmission(row,house,{'simulation':{'max_diffraction_order':10}})
    assert checker.check('environment1',np.array([0.,.2,1.]))['status']=='blocked'
    assert checker.check('environment1',np.array([1.,.2,1.]))['status']=='pass'
    checker.close();assert calls[-1]=='closed'
    assert house['checked_rooms'][0]['placement']['camera_m']==[9,9,9]


def test_geometry_only_caller_never_claims_a_measured_acoustic_pass():
    checker=EnvironmentPositionAdmission({}, {}, {})
    assert checker.check('environment1',[0,.2,1])['status']=='deferred_to_current_native_scene'
    checker.close()
