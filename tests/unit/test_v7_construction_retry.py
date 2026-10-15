import json
import pytest
from avengine.timeline.v7_construction_retry import queue_construction_retry
pytestmark=pytest.mark.fast_unit


def test_retry_keeps_required_request_and_seed_and_uses_a_fresh_attempt(tmp_path):
    task={'slot':0,'constructor':'automatic_family','target_family':'QA-28','row':{'house':'new','room_id':'r1','scene_id':'new_scene','scene_seed':928},'alternative_room_ids':['r2'],'construction_attempt':3}
    manifest={'worktree':'frozen','code_commit':'a'*40,'parents':[task]};path=tmp_path/'plan.json';path.write_text(json.dumps(manifest))
    receipt={'slot':0,'status':'failed','error_type':'ValueError','reason':'no strict native floor placement for body'}
    assert queue_construction_retry(path,receipt)==1
    result=json.loads(path.read_text())['parents'][1]
    assert result['target_family']=='QA-28'and result['row']['scene_seed']==928
    assert result['row']['room_id']=='r2'and result['row']['scene_id']=='new_scene__attempt_04'
    assert queue_construction_retry(path,receipt)is None
    assert queue_construction_retry(path,{**receipt,'error_type':'ImportError'})is None
