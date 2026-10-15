"""UE load identity stays distinct from the source GLB; basis is native."""
import json
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
