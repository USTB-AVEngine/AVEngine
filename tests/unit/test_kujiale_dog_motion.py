import numpy as np
from avengine.qa.kujiale_dog_motion import calibrate
from avengine.qa.kujiale_asset_geometry import rotation

def fixture():
    names=['FL','FR','HL','HR','mouth']
    points=np.array([[-.15,.08,-.3],[.15,.08,-.3],[-.12,.11,.3],[.12,.11,.3],[0,.5,-.5]])
    authored={'frames':[],'yaw_rad':[.3,.5]}
    native={'frames':[]}
    Q=rotation(.7);T=np.array([8.,-3.,2.])
    for index,yaw in enumerate(authored['yaw_rad']):
        moving=points.copy();moving[0,1]+=index*.02
        root=np.array([2.,0.,4.]);world=moving@rotation(yaw).T+root
        authored['frames'].append({'world_root_m':root.tolist(),'mouth_world_m':world[4].tolist(),'feet':{k:{'ankle_world_m':world[i].tolist()} for i,k in enumerate(names[:4])}})
        target=moving[:,[0,2,1]]*100.
        imported=(target-T)@Q/1.2
        native['frames'].append({'native_component_bones_cm':dict(zip(names,imported.tolist()))})
    return native,authored

def test_one_rig_calibration_preserves_later_independent_animation_key():
    native,authored=fixture();rec=calibrate(native,authored)
    assert rec['status']=='pass'
    assert rec['maximum_all_key_bone_error_m']<1e-10
    assert abs(rec['uniform_scale']-1.2)<1e-10

def test_an_animation_key_with_a_moved_paw_cannot_be_refitted_into_a_pass():
    native,authored=fixture();native['frames'][1]['native_component_bones_cm']['FL'][2]+=.8
    rec=calibrate(native,authored)
    assert rec['status']=='fail'
    assert rec['maximum_all_key_bone_error_m']>.003

def test_fixed_floor_calibration_cannot_hide_incompatible_contact_keys():
    from avengine.qa.kujiale_dog_motion import fixed_floor_calibration
    import pytest
    def frame(gaps):return {"sole_depth":{"four_paws":[{"floor_gap_m":v,"expected_in_contact":False} for v in gaps]}}
    rec=fixed_floor_calibration([frame([.003,.02,.04,.05]),frame([.0165,.02,.04,.05])])
    assert -.012<=.003+rec["fixed_root_y_offset_m"]<=.012
    assert abs(.0165+rec["fixed_root_y_offset_m"])<=.012
    with pytest.raises(ValueError,match="no single"):
        fixed_floor_calibration([frame([-.015,.02,.04,.05]),frame([.020,.04,.05,.06])])
