import json
import pytest
from tools.timeline.run_v7_motion_quartet_batch import infrastructure_retries
pytestmark=pytest.mark.fast_unit

def test_fixed_infrastructure_failure_retries_the_same_automatic_room_and_source_type(tmp_path):
    house={'house':'unseen_house','checked_rooms':[{'room_label':'unseen_room'}]}
    task={'slot':0,'code_commit':'older','row':{'house':'unseen_house','room_id':'unseen_room','scene_seed':19,'motion_source_type':'human_speech'}}
    path=tmp_path/'evidence/jobs';path.mkdir(parents=True)
    (path/'000.json').write_text(json.dumps({'status':'failed','reason':'[Errno 17] File exists: trial progress'}))
    manifest={'code_commit':'older','parents':[task]}
    result=infrastructure_retries(manifest,tmp_path,'fixed',[house])
    assert len(result)==1 and result[0][3]is task
    assert not infrastructure_retries(manifest,tmp_path,'older',[house])
    manifest['parents'].append({'slot':1,'code_commit':'fixed','retry_of_slot':0,'row':task['row']})
    assert not infrastructure_retries(manifest,tmp_path,'fixed',[house])

def test_physical_rejection_uses_room_planner_rather_than_infrastructure_replay(tmp_path):
    task={'slot':0,'row':{'house':'unknown','room_id':'room'},'code_commit':'old'}
    path=tmp_path/'evidence/jobs';path.mkdir(parents=True)
    (path/'000.json').write_text(json.dumps({'status':'failed','reason':'unchanged collision gate'}))
    assert not infrastructure_retries({'code_commit':'old','parents':[task]},tmp_path,'new',[])
