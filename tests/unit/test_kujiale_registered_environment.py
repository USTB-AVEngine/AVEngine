import json
import pytest
from avengine.capture.kujiale_registered_environment import environment_bindings, require_binding_project

def registry(tmp_path):
    native={'status':'pass_static_native','ue_asset_paths':['/Game/Owned/Dog.Dog'],'emitter':{'actual_m':[1.,.4,2.]}}
    path=tmp_path/'admission.json'
    path.write_text(json.dumps({'assets':[{'asset_id':'dog','ue_native_physical_status':'pass','ue_project':str(tmp_path/'Owned.uproject'),'silent_decor':False,'native_house_admissions':{'room':native}}]}))
    return path

def scene(asset='dog',position=(1.,.4,2.)):
    return {'sources':{'voice1':{},'pet':{'actor_id':'pet1','native_asset_id':asset,'trajectory':[{'position_m':position}]}}}

def test_physical_source_requires_an_admitted_visual_and_actual_sound_point(tmp_path):
    path=registry(tmp_path)
    assert environment_bindings(scene(),'room',path,{'voice1'})['pet']['asset']['asset_id']=='dog'
    with pytest.raises(ValueError,match='physical native_asset_id'):
        environment_bindings(scene(asset=None),'room',path,{'voice1'})
    with pytest.raises(ValueError,match='physical emitter'):
        environment_bindings(scene(position=(1.,1.4,2.)),'room',path,{'voice1'})

def test_wrong_room_and_silent_decoration_cannot_become_a_sound_source(tmp_path):
    path=registry(tmp_path)
    with pytest.raises(ValueError,match='passed native visual'):
        environment_bindings(scene(),'other_room',path,{'voice1'})
    data=json.loads(path.read_text());data['assets'][0]['silent_decor']=True;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='silent decoration'):
        environment_bindings(scene(),'room',path,{'voice1'})


def test_incomplete_motion_admission_cannot_be_used_as_a_physical_source(tmp_path):
    path=registry(tmp_path)
    data=json.loads(path.read_text());data['assets'][0]['ue_native_physical_status']='fail_or_pending';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='physical admission is incomplete'):
        environment_bindings(scene(),'room',path,{'voice1'})

def test_rendering_admitted_package_names_in_another_project_is_rejected(tmp_path):
    path=registry(tmp_path);bindings=environment_bindings(scene(),'room',path,{'voice1'})
    require_binding_project(bindings,{'uproject':str(tmp_path/'Owned.uproject')})
    with pytest.raises(ValueError,match='UE project differs'):
        require_binding_project(bindings,{'uproject':str(tmp_path/'Other.uproject')})


def test_nonfinite_and_diagonal_sound_point_drift_cannot_bypass_readback_gate(tmp_path):
    path=registry(tmp_path)
    for point in [(float('nan'),.4,2.),(1.0004,.4004,2.)]:
        with pytest.raises(ValueError,match='physical emitter'):
            environment_bindings(scene(position=point),'room',path,{'voice1'})
