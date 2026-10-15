"""Production reads measured bases/package identity; loaded map comes from UE."""
import json
import numpy as np
import pytest
from avengine.registry.kujiale_ue_measurements import native_asset,loaded_map_package,read_registration,sha256
pytestmark=pytest.mark.fast_unit

def test_formal_basis_is_not_a_world_pose_or_fabricated_mouth():
    asset={'asset_id':'h','ue_generated_class':'/Game/H.H_C','registered_native_ue':{'mesh_paths':['/Game/M.M']},'mouth_local_glb_m':[0,1.6,0]}
    row={'asset_id':'h','actual_class_path':'/Game/H.H_C','actual_mesh_paths':['/Game/M.M'],'glb_to_ue_local_cm':[[0,0,-100],[100,0,0],[0,100,0]],'mouth_local_glb_m':[0,1.6,0]}
    actual,matrix=native_asset(asset,{'assets':[row]});assert np.allclose(matrix@asset['mouth_local_glb_m'],[0,0,160])
    row['actual_mesh_paths']=['/Game/Other.Other']
    with pytest.raises(ValueError,match='meshes differ'):native_asset(asset,{'assets':[row]})
    with pytest.raises(ValueError,match='no measured'):native_asset(asset,{'assets':[]})

def test_current_loaded_map_must_match_ue_log(tmp_path):
    project=tmp_path/'Spear.uproject';content=tmp_path/'Content/Room';content.mkdir(parents=True);package=content/'R.umap';package.write_bytes(b'original umap');launch=tmp_path/'launch';launch.mkdir();log=launch/'UnrealEditor.log';log.write_text('LogLoad: LoadMap: /Game/Room/R?Name=Player\n')
    proof=loaded_map_package(project,'/Game/Room/R',launch);assert proof['sha256']==sha256(package)
    log.write_text('LogLoad: LoadMap: /Game/Wrong/Room\n')
    with pytest.raises(ValueError,match='actual UE loaded map'):loaded_map_package(project,'/Game/Room/R',launch)
