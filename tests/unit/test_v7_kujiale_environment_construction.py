import json
from pathlib import Path
import pytest
from avengine.timeline.v7_kujiale_environment_construction import native_class_candidates,bind_inventory_project,separate_environment_schedule,_clear_segment
import numpy as np
pytestmark=pytest.mark.fast_unit

def test_required_native_class_cannot_be_replaced_by_unimported_or_silent_asset(tmp_path):
    project=tmp_path/'P.uproject';package=tmp_path/'Content/Dog/D.uasset';package.parent.mkdir(parents=True);package.write_bytes(b'UE original dog')
    valid={'asset_id':'dog','ue_native_physical_status':'pass','ue_import_status':'pass','silent_decor':False,'allowed_v7_classes_candidate':['dog_bark'],'ue_asset_paths':['/Game/Dog/D.D'],'dimensions_m':[.3,.4,.5]}
    result=native_class_candidates({'assets':[valid]},['dog_bark'],project,0);assert result[0][1][0]['asset_id']=='dog'
    valid['silent_decor']=True
    with pytest.raises(ValueError,match='no admitted'):native_class_candidates({'assets':[valid]},['dog_bark'],project,0)
    valid['silent_decor']=False;package.unlink()
    with pytest.raises(ValueError,match='no admitted'):native_class_candidates({'assets':[valid]},['dog_bark'],project,0)

def test_runtime_project_view_preserves_source_registration(tmp_path):
    source=tmp_path/'source.json';source.write_text(json.dumps({'assets':[{'asset_id':'dog','ue_project':'original.uproject'}]}));before=source.read_bytes();inputs={'kujiale_ue_asset_admission':str(source)};house={'input':{'ue_config':{'uproject':'runtime.uproject'}}}
    result=bind_inventory_project(inputs,house,tmp_path);bound=json.loads(Path(result['kujiale_ue_asset_admission']).read_text());assert source.read_bytes()==before and bound['assets'][0]['original_registered_ue_project']=='original.uproject'and bound['assets'][0]['ue_project']=='runtime.uproject'

def test_required_animal_starts_after_whole_human_speech_and_stays_in_each_cycle():
    sounds=[{'scenario_id':'s','sound_asset_id':str(i),'role_id':i,'voice_reference':{'gender':'female','speaker_id':'v'+str(i)},'synthesis_readback':{'duration_s':d}}for i,d in enumerate([1.72,1.76])];row={'scene_seed':0,'production_scenario':{'gender':'F'}}
    end,effect=separate_environment_schedule(row,{'sounds':sounds});assert end<=4.8 and effect>=end+.499 and effect+2<8.35
    assert row['production_scenario']['environment_start_offsets_s']==[effect]

def test_camera_to_animal_segment_checks_registered_human_envelope():
    assert not _clear_segment(np.array([0,1.,0]),np.array([0,.3,4]),[(np.array([-.4,0,1.5]),np.array([.4,1.9,2.3]))])
    assert _clear_segment(np.array([0,1.,0]),np.array([2,.3,4]),[(np.array([-.4,0,1.5]),np.array([.4,1.9,2.3]))])


def test_cached_support_proposals_match_original_native_placement_anchor():
    from avengine.timeline.v7_kujiale_environment_construction import NativeSupportProposalCache
    from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,rotation
    vertices=np.array([[x,y,z]for x in [-.6,.4]for y in [0.,.5]for z in [-.2,.2]]+[[.45,.8,.05]])
    cache=NativeSupportProposalCache(vertices)
    for normal in [[0,1,0],[0,-1,0],[1,0,0],[-1,0,0],[0,0,1]]:
        for yaw in [0.,.4,1.57]:
            support={'normal':normal,'point_m':[2.,1.,3.]};n=np.asarray(normal,float);point=np.asarray(support['point_m'])
            class Plane:
                def ray(self,start,direction,length):
                    start=np.asarray(start);direction=np.asarray(direction);t=np.dot(point-start,n)/np.dot(direction,n)
                    return {'point_m':(start+t*direction).tolist()}if 0<=t<=length else None
                def collision(self,body,root,angle,normal,tolerance):
                    distance=(body@rotation(angle).T+root-point)@n
                    return {'status':'pass'if distance.min()>=-tolerance else'fail'}
            original=NativeSupportGeometry.placement(Plane(),vertices,support,yaw,.012)
            assert original is not None
            matrix,rotated,root,low,high=cache.pose(support,yaw)
            assert np.allclose(root,original['root_m'],atol=1e-12)
            world=vertices@matrix.T+original['root_m'];assert np.allclose(low,world.min(axis=0),atol=1e-12)and np.allclose(high,world.max(axis=0),atol=1e-12)
    assert len(cache.rotations)==15


def test_required_real_dog_retries_legal_floor_behind_camera_without_removing_it(tmp_path,monkeypatch):
    from avengine.timeline import v7_kujiale_environment_construction as construction
    from avengine.qa import kujiale_asset_geometry as geometry
    from avengine.capture import kujiale_source_floor_paws as paws
    vertices=np.array([[x,y,z]for x in [-.1,.1]for y in [0.,.2]for z in [-.1,.1]])
    manifest=tmp_path/'inputs/manifest.json';manifest.parent.mkdir();manifest.write_text(json.dumps({'revision':'r','assets':[{'asset_id':'tiny_dog'}]}))
    receipt=tmp_path/'evidence/import/r/tiny_dog.json';receipt.parent.mkdir(parents=True);receipt.write_text('{}')
    primary=tmp_path/'primary.json';primary.write_text(json.dumps({'input_manifest':str(manifest)}))
    asset={'asset_id':'tiny_dog','primary_native_evidence':str(primary),'dimensions_m':[.2,.2,.2],
           'attachment_surface':'floor','category':'dog','emitter_local_m':[0.,.1,0.],
           'ue_asset_paths':['/Game/ActualDog/Body.Body']}
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'assets':[asset]}))
    monkeypatch.setattr(construction,'native_class_candidates',lambda *a:[('dog_bark',[asset])])
    monkeypatch.setattr(geometry,'import_vertices',lambda r:vertices)
    contact_calls=[];paw_calls=[]
    class Support:
        def support_catalog(self,*args):return {'floor':[{'point_m':[0.,0.,3.],'normal':[0,1,0]}]},object()
        def ray(self,*args):return None
        def placement(self,mesh,support,yaw,tolerance):
            contact_calls.append(tolerance)
            return {'root_m':support['point_m'],'yaw_rad':yaw,'normal':support['normal']}
    monkeypatch.setattr(geometry,'NativeSupportGeometry',lambda *a:Support())
    monkeypatch.setattr(geometry,'camera_candidates',lambda *a:[{'position_m':[2.,1.,3.],'target_m':[0.,.1,3.]}]*3)
    class Paws:
        def __init__(self,*args):pass
        def measure(self,*args):paw_calls.append(True);return {'status':'pass'}
    monkeypatch.setattr(paws,'PreparedPawContacts',Paws)
    humans=[{'root_m':[x,0.,z],'position_m':[x,1.3,z]}for x,z in [(-2.,-3.),(2.,-3.),(0.,-5.)]]
    row={'house':'any_house','room_id':'any_room','scene_seed':1,'production_scenario':{},
         'measured_static_layout':{'listener_m':[0.,1.,0.],'valid':humans,'fixed_camera_yaw_deg':0.}}
    house={'input':{'ue_config':{'native_floor_surface':{},'uproject':'registered_project'}},'checked_rooms':[{'room_label':'any_room'}]}
    plans=construction.construct_environment(row,house,{'kujiale_ue_asset_admission':str(inventory)},['dog_bark'])
    assert len(plans)==1 and plans[0]['asset_id']=='tiny_dog' and plans[0]['sound_class']=='dog_bark'
    assert plans[0]['intentional_offscreen_role']['real_registered_drawable_required']
    assert contact_calls==[.012] and paw_calls==[True]
    assert plans[0]['placement']['root_m']==[0.,0.,3.] and len(humans)==3
