from pathlib import Path
import json,hashlib
import pytest
from avengine.timeline.v7_selection_resume import prepare_completed_parent,NATIVE_STAGES
pytestmark=pytest.mark.fast_unit

def make_source(root):
    source=root/'attempt';(source/'inputs/planned').mkdir(parents=True)
    row={'scene_id':'same_physical_parent','trial_only':True,'members':['v0_a0'],'asset_ids':['body1','body2','body3'],
        'motion_requests':{},'listener_m':[1,1.5,2],'camera_yaw_knots':[{'time_s':0,'yaw_deg':30},{'time_s':30,'yaw_deg':30}]}
    manifest={'storage_budget_root':str(root),'parents':[{'row':row}],'storage_admission_prechecked':True}
    (source/'inputs/generated_manifest.json').write_text(json.dumps(manifest));(source/'inputs/planned/000.json').write_text(json.dumps(row));(source/'inputs/demand_state.json').write_text('{}')
    parent=source/'native/parents/same_physical_parent';parent.mkdir(parents=True);(parent/'input_recipe.json').write_text('{"original":true}')
    for name in NATIVE_STAGES:(parent/(name+'_completed.json')).write_text(json.dumps({'attempt':'native_attempt','recipe':{'same':True},'required_files':[]}))
    return source,parent,row

def test_selection_resume_changes_only_sampling_and_never_source_identity_or_checkpoints(tmp_path):
    source,parent,row=make_source(tmp_path);before={str(p):p.read_bytes()for p in parent.iterdir()}
    manifest,original,hashes=prepare_completed_parent(source,tmp_path/'new_selection',tmp_path)
    selected=manifest['parents'][0]['row'];assert original==row
    assert {k:v for k,v in selected.items()if k!='acceptance_allow_partial_families'}==row
    assert selected['acceptance_allow_partial_families']is True
    assert before=={str(p):p.read_bytes()for p in parent.iterdir()}
    assert set(hashes)==set(before)
    marker=json.loads((tmp_path/'new_selection/native/parents/same_physical_parent/visible_native_completed.json').read_text())
    assert Path(marker['attempt'])==parent/'native_attempt'

def test_selection_resume_refuses_invalid_old_media_and_external_roots(tmp_path):
    source,parent,row=make_source(tmp_path)
    with pytest.raises(ValueError,match='declared own output root'):
        prepare_completed_parent(source,tmp_path.parent/'external_selection',tmp_path)
    (source/'INVALID_20261015.md').write_text('old media invalid')
    with pytest.raises(ValueError,match='invalidated old media'):
        prepare_completed_parent(source,tmp_path/'new_selection',tmp_path)
