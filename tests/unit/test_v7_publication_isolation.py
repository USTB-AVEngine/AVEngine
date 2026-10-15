import json
import pytest
from avengine.timeline.v7_acceptance_publication import publish_admitted_parent
pytestmark=pytest.mark.fast_unit

def test_unrelated_batch_exception_does_not_destroy_a_passed_parent(tmp_path,monkeypatch):
    p=tmp_path/'evidence/jobs/000.json';p.parent.mkdir(parents=True)
    original={'status':'passed','scene_id':'native_parent','reader':{'status':'pass'},'export':{'scene_delivery_policy':{'status':'pass'}}}
    p.write_text(json.dumps(original))
    def failure(root):raise KeyError('truth')
    monkeypatch.setattr('avengine.timeline.v7_acceptance_publication.publish_current_parents',failure)
    assert publish_admitted_parent(tmp_path,p)['status']=='pending_retry'
    actual=json.loads(p.read_text())
    assert actual['status']=='passed'and actual['reader']==original['reader']and actual['export']==original['export']
    assert actual['batch_publication']['error_type']=='KeyError'
    assert not(tmp_path/'export/train.jsonl').exists()

def test_current_native_policy_rejection_is_never_restored(tmp_path,monkeypatch):
    p=tmp_path/'evidence/jobs/000.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'passed'}))
    def failure(root):
        p.write_text(json.dumps({'status':'failed','reason':'current native gate rejected'}));raise ValueError('gate evidence')
    monkeypatch.setattr('avengine.timeline.v7_acceptance_publication.publish_current_parents',failure)
    publish_admitted_parent(tmp_path,p)
    assert json.loads(p.read_text())=={'status':'failed','reason':'current native gate rejected'}
