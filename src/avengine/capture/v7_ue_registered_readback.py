"""Actual UE basis probes and original registered package loading receipts.

The GLB import basis is checked against the registered mouth, then three
attached one-metre probes measure the world matrix. No planned Euler angle or
planned position is substituted for a native transform. UE package hashes and
source GLB hashes remain separate identities.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from avengine.capture.kujiale_spear_runtime import from_ue
from avengine.timeline.continuous_pilot import write_json

REVISION='registered-UE-native-basis-and-load-v1'
GLB_TO_UE_CM=np.array([[0.,0.,-100.],[100.,0.,0.],[0.,100.,0.]])

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def _package(content,object_path):
    if not object_path.startswith('/Game/'):raise ValueError('registered UE resource outside Game mount')
    path=(Path(content)/(object_path.partition('.')[0][6:]+'.uasset')).resolve()
    if not path.is_relative_to(Path(content).resolve()) or not path.is_file():
        raise ValueError('actual registered UE package missing or outside Content')
    return path

def prepare_readback(video,asset,declaration,actor,root,components):
    from avengine.backends.spear_ue.research_runtime import attach_emitter_component
    mouth=np.asarray(asset['mouth_local_glb_m'],float)
    local=np.asarray(asset['registered_native_ue_mouth_local_cm'],float)
    if np.max(np.abs(GLB_TO_UE_CM@mouth-local))>1e-6:
        raise ValueError('registered GLB/native local mouth bases disagree')
    game=video.game;objects=game.get_unreal_object(uclass='UKismetSystemLibrary')
    native_class=int(game.unreal_service.get_class(uobject=actor,as_handle=True))
    expected_class=int(game.unreal_service.load_class(uclass='AActor',name=asset['ue_generated_class'],as_handle=True))
    if not native_class or native_class!=expected_class:
        raise ValueError('actual spawned class handle differs from registered Blueprint: '+str((native_class,expected_class)))
    class_path=asset['ue_generated_class']
    expected_meshes={int(game.unreal_service.load_object(uclass='UStaticMesh',name=p,as_handle=True)):p for p in asset['registered_native_ue']['mesh_paths']}
    meshes=[];mesh_handles=[]
    for c in components:
        handle=int(c.get_property_value(property_name='StaticMesh',as_handle=True))
        if not handle or handle not in expected_meshes:raise ValueError('actual UE mesh handle is not an original registered part')
        meshes.append(expected_meshes[handle]);mesh_handles.append(handle)
    if sorted(meshes)!=sorted(asset['registered_native_ue']['mesh_paths']):
        raise ValueError('actual UE mesh components differ from original registration')
    registry=Path(asset['source_registry']);manifest=json.loads(registry.read_text())
    if digest(registry)!=asset['source_registry_sha256']:raise ValueError('original registry changed before native load')
    content=Path(video.ue['uproject']).parent/'Content'
    source_project=Path(manifest['ue_project'])
    if not source_project.is_absolute():source_project=Path(manifest['artifact_root'])/source_project
    original=(source_project.parent if source_project.suffix=='.uproject' else source_project)/'Content'
    paths=sorted(set([class_path,*meshes,*asset['registered_native_ue']['materials']]))
    files=[]
    for obj in paths:
        actual=_package(content,obj);source=_package(original,obj)
        observed=digest(actual);expected=digest(source)
        if observed!=expected:raise ValueError('loaded UE package differs from original registered package: '+obj)
        files.append({'object_path':obj,'path':str(actual),'source':str(source),'sha256':observed})
    glb=Path(asset['habitat_glb']);glb_sha=digest(glb)
    if glb_sha!=asset['actual_artifact_sha256'][str(glb)]:raise ValueError('original import GLB changed')
    bp=next(f for f in files if f['object_path']==class_path)
    receipt={'status':'pass','revision':REVISION,'backend':'ue_spear','asset_id':asset['asset_id'],
        'source_registry':str(registry),'source_registry_sha256':asset['source_registry_sha256'],
        'actual_class_path':class_path,'actual_mesh_paths':meshes,'files':files,
        'native_class_handle':native_class,'registered_class_handle':expected_class,'actual_mesh_handles':mesh_handles,
        'native_resource_identity_method':'exact live get_class/StaticMesh handles equal resolved registered class/mesh handles',
        'loaded_artifact_sha256':bp['sha256'],'loaded_artifact_kind':'UE Blueprint uasset',
        'original_glb':str(glb),'original_glb_sha256':glb_sha,
        'glb_to_ue_local_cm':GLB_TO_UE_CM.tolist(),
        'mouth_local_glb_m':mouth.tolist(),'mouth_local_ue_cm':local.tolist()}
    target=Path(video.ue['output_root'])/'inputs/native_loads'/(asset['asset_id']+'_'+bp['sha256'][:16]+'_pid'+str(video.instance._process.pid)+'.json')
    if target.exists():
        if json.loads(target.read_text())!=receipt:raise ValueError('native original load receipt changed')
    else:target.parent.mkdir(parents=True,exist_ok=True);write_json(target,receipt)
    probes=[attach_emitter_component(game,actor_id=declaration['actor_id']+'_basis_'+str(i),anchor_root=root,
            emitter_local_ue_cm=GLB_TO_UE_CM[:,i].tolist())for i in range(3)]
    return {'body_basis_probes':probes,'native_load_receipt':str(target),
            'loaded_artifact_sha256':bp['sha256'],'original_glb_sha256':glb_sha}

def measured_body_matrix(native,root_position):
    from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
    points=[from_ue(read_scene_component_pose(c)['location_cm'])for c in native['body_basis_probes']]
    root=np.asarray(root_position,float);matrix=np.eye(4);matrix[:3,3]=root
    matrix[:3,:3]=(np.asarray(points)-root).T
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,atol=1e-5)):
        raise ValueError('actual native body probes are not a rigid metre transform')
    return matrix.tolist(),points
