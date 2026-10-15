from pathlib import Path
import json
import pytest
from avengine.timeline.v7_construction_retry import queue_unmet_family_retry
pytestmark=pytest.mark.fast_unit


def fixture(tmp_path,has_candidate=False):
    root=tmp_path;parent=root/'native/parents/p';parent.mkdir(parents=True);wid='p__v0_a0__w00'
    pool=root/'pool'/wid;pool.mkdir(parents=True);(pool/'questions.json').write_text(json.dumps([{'qa_id':'QA-21'}]if has_candidate else[]))
    seal=parent/'seal.json';seal.write_text(json.dumps({'result':{'windows':[{'window_id':wid,'member':'v0_a0'}]}}))
    (parent/'selected_windows_v0_a0_completed.json').write_text(json.dumps({'recipe':{'pool_directory':str(pool.parent)}}))
    manifest=root/'inputs/ACCEPTANCE_PLAN.json';manifest.parent.mkdir();manifest.write_text(json.dumps({'worktree':'new-frozen','code_commit':'new',
      'parents':[{'slot':0,'constructor':'automatic_family','target_family':'QA-21','row':{'scene_id':'p','scene_seed':42,'house':'legal','room_id':'R'}}]}))
    receipt={'slot':0,'status':'passed','export':{'source_seal':str(seal),'question_family_counts':{'QA-14':1}}}
    return manifest,receipt


def test_absent_family_reconstructs_without_invalidating_a_valid_parent_or_changing_seed(tmp_path):
    manifest,receipt=fixture(tmp_path);assert queue_unmet_family_retry(manifest,receipt)==1
    task=json.loads(manifest.read_text())['parents'][1]
    assert receipt['status']=='passed'and task['row']['scene_seed']==42
    assert task['condition_retry_of_slot']==0 and task['snapshot_commit']=='new'
    assert queue_unmet_family_retry(manifest,receipt)is None


def test_unselected_existing_legal_candidate_is_left_for_reselection_without_rendering(tmp_path):
    manifest,receipt=fixture(tmp_path,True)
    assert queue_unmet_family_retry(manifest,receipt)is None
    assert len(json.loads(manifest.read_text())['parents'])==1
