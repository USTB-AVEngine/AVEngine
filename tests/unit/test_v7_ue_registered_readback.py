"""UE load identity stays distinct from the source GLB; basis is native."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.capture.v7_ue_registered_readback import measured_body_matrix,GLB_TO_UE_CM,digest
from avengine.timeline.v7_native_asset_readback import require_loaded_asset,_ue_receipt
pytestmark=pytest.mark.fast_unit

def test_native_basis_keeps_rotation_and_translation(monkeypatch):
    from avengine.backends.spear_ue import research_runtime
    root=np.array([2.,.1,-3.]);rotation=np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
    points=[root+rotation[:,i] for i in range(3)]
    def pose(i):
        p=points[i];return {'location_cm':[100*p[0],100*p[2],100*p[1]]}
    monkeypatch.setattr(research_runtime,'read_scene_component_pose',pose)
    matrix,_=measured_body_matrix({'body_basis_probes':[0,1,2]},root)
    assert np.allclose(np.asarray(matrix)[:3,:3],rotation)
    assert np.allclose(np.asarray(matrix)[:3,3],root)
    points[0]+=np.array([.1,0,0])
    with pytest.raises(ValueError,match='rigid metre'):measured_body_matrix({'body_basis_probes':[0,1,2]},root)

def test_native_load_does_not_relabel_glb_hash(tmp_path):
    package=tmp_path/'body.uasset';original=tmp_path/'original.uasset';glb=tmp_path/'body.glb'
    package.write_bytes(b'UE actual package');original.write_bytes(package.read_bytes());glb.write_bytes(b'original GLB')
    proof={'status':'pass','backend':'ue_spear','asset_id':'human','source_registry_sha256':'registry',
        'actual_class_path':'/Game/Human.Human_C','files':[{'path':str(package),'source':str(original),'sha256':digest(package)}],
        'original_glb':str(glb),'original_glb_sha256':digest(glb),'loaded_artifact_sha256':digest(package)}
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(proof))
    actor={'asset_id':'human','native_load_backend':'ue_spear','native_load_receipt':str(receipt),
        'source_registry_sha256':'registry','loaded_artifact_sha256':digest(package),'original_glb_sha256':digest(glb)}
    asset={'asset_id':'human','ue_generated_class':'/Game/Human.Human_C','habitat_glb':str(glb),'actual_artifact_sha256':{str(glb):digest(glb)}}
    require_loaded_asset(actor,asset)
    actor['loaded_artifact_sha256']=digest(glb)
    with pytest.raises(ValueError,match='loaded UE actor'):require_loaded_asset(actor,asset)
    actor['loaded_artifact_sha256']=digest(package);package.write_bytes(b'changed UE package');_ue_receipt.cache_clear()
    with pytest.raises(ValueError,match='package changed'):require_loaded_asset(actor,asset)

def test_registered_ue_receipt_compares_native_handles(tmp_path,monkeypatch):
    from avengine.backends.spear_ue import research_runtime
    from avengine.capture.v7_ue_registered_readback import prepare_readback
    original=tmp_path/'source';project=tmp_path/'output/project';project.mkdir(parents=True)
    paths=['/Game/Actors/Human.Human_C','/Game/Mesh/Body.Body','/Game/Mesh/Material.Material']
    for content in [original/'ue_project/Content',project/'Content']:
        for obj in paths:
            path=content/(obj.partition('.')[0][6:]+'.uasset');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(obj.encode())
    registry=original/'registry.json';registry.write_text(json.dumps({'artifact_root':str(original),'ue_project':'ue_project/Humans.uproject'}))
    glb=original/'body.glb';glb.write_bytes(b'original imported body')
    service=SimpleNamespace(get_class=lambda **kw:17,load_class=lambda **kw:17,load_object=lambda **kw:21)
    def objects(**kw):
        # A raw int in GetPathName would serialize as a number, not a UE Ptr.
        return SimpleNamespace(GetPathName=lambda **kw:pytest.fail('raw handle was sent to reflection'))
    game=SimpleNamespace(unreal_service=service,get_unreal_object=objects)
    video=SimpleNamespace(game=game,ue={'uproject':str(project/'SpearSim.uproject'),'output_root':str(tmp_path/'output')},instance=SimpleNamespace(_process=SimpleNamespace(pid=1234)))
    point=np.array([.01,1.6,-.02]);asset={'asset_id':'human','mouth_local_glb_m':point.tolist(),
        'registered_native_ue_mouth_local_cm':(GLB_TO_UE_CM@point).tolist(),'source_registry':str(registry),
        'source_registry_sha256':digest(registry),'ue_generated_class':paths[0],
        'registered_native_ue':{'mesh_paths':[paths[1]],'materials':[paths[2]]},'habitat_glb':str(glb),'actual_artifact_sha256':{str(glb):digest(glb)}}
    video.native_measurement_registration={'assets':[{'asset_id':'human','actual_class_path':paths[0],'actual_mesh_paths':[paths[1]],'mouth_local_glb_m':point.tolist(),'glb_to_ue_local_cm':GLB_TO_UE_CM.tolist(),'packages':[{'object_path':obj,'sha256':digest(project/'Content'/(obj.partition('.')[0][6:]+'.uasset'))}for obj in paths]}]}
    video.native_measurement_registration_receipt={'path':'formal.json','sha256':'fixture','revision':'test'}
    monkeypatch.setattr(research_runtime,'attach_emitter_component',lambda *a,**kw:kw['emitter_local_ue_cm'])
    component=SimpleNamespace(get_property_value=lambda **kw:21)
    result=prepare_readback(video,asset,{'actor_id':'source1'},object(),object(),[component])
    proof=json.loads(Path(result['native_load_receipt']).read_text())
    assert proof['native_class_handle']==proof['registered_class_handle']==17
    assert proof['actual_mesh_handles']==[21]
    assert proof['original_glb_sha256']!=proof['loaded_artifact_sha256']
    service.get_class=lambda **kw:18
    with pytest.raises(ValueError,match='class handle differs'):
        prepare_readback(video,asset,{'actor_id':'source1'},object(),object(),[component])

def test_native_dialogue_camera_constructs_distinct_observable_sectors():
    from avengine.capture.kujiale_pilot_environment import construct_dialogue_camera
    from avengine.qa.unified_catalog import _azimuth_sector
    # A centered 28-degree pair previously produced two front-sector answers.
    points=[[np.sin(np.radians(a))*3,1.6,-np.cos(np.radians(a))*3]for a in [-14.,14.]]
    plan=construct_dialogue_camera([0.,1.5,0.],points)
    assert plan['qa_sector_gate_changed'] is False
    assert len(set(plan['source_sectors']))==2
    assert max(abs(a)for a in plan['source_azimuths_deg'])<42.5
    assert all(_azimuth_sector(a)==s for a,s in zip(plan['source_azimuths_deg'],plan['source_sectors']))

def test_namespace_reuse_rechecks_changed_frames_and_original_files(tmp_path,monkeypatch):
    from copy import deepcopy
    from avengine.qa import v7_roster_intake
    from avengine.timeline.v7_registry_snapshot import qualify_namespace,_NAMESPACE_CACHE
    glb=tmp_path/'body.glb';glb.write_bytes(b'original body');sha=digest(glb)
    registry=tmp_path/'registry.json';registry.write_text('{}');visual_path=tmp_path/'native.json';visual_path.write_text('{}')
    calls=[]
    def intake(path):
        calls.append(path)
        return {'source_registry_sha256':'registry','records':[{'asset_id':'human','standing_registered_qualified':True,
            'mouth_local_glb_m':[0.,1.,0.],'habitat_glb':str(glb),'actual_artifact_sha256':{str(glb):digest(glb)}}]}
    monkeypatch.setattr(v7_roster_intake,'intake_roster',intake)
    actor={'actor_id':1,'asset_id':'human','source_registry_sha256':'registry','loaded_artifact_sha256':sha,
        'moving':False,'action_id':'stand','body_to_world_m':np.eye(4).tolist(),'emitter_position_m':[0.,1.,0.]}
    visual={'frames':[{'actors':[deepcopy(actor)]}for _ in range(450)]};target=tmp_path/'receipt.json';_NAMESPACE_CACHE.clear()
    qualify_namespace(visual,registry,'registry',target,visual_path)
    qualify_namespace(visual,registry,'registry',target,visual_path)
    assert len(calls)==1
    visual['frames'][0]['actors'][0]['emitter_position_m']=[0.,1.1,0.]
    with pytest.raises(ValueError,match='2mm'):qualify_namespace(visual,registry,'registry',target,visual_path)
    visual['frames'][0]['actors'][0]['emitter_position_m']=[0.,1.,0.];glb.write_bytes(b'changed body')
    with pytest.raises(ValueError,match='native GLB differs'):qualify_namespace(visual,registry,'registry',target,visual_path)
    assert len(calls)==3
