import json
import pytest
from avengine.timeline.v7_phase2_rir_prefetch import run_prefetch_plan

pytestmark=pytest.mark.fast_unit

def test_no_queue_is_a_noop_for_existing_serial_requests(tmp_path):
    p=tmp_path/'parent/audio/fresh/token/request.json';p.parent.mkdir(parents=True)
    assert run_prefetch_plan(p,{}, {})is None


def test_queue_cannot_write_outside_its_own_parent(tmp_path):
    root=tmp_path/'parent/audio/fresh';p=root/'token/request.json';p.parent.mkdir(parents=True)
    plan={'schema':'v7-fresh-rir-prefetch-plan-v1','plan_id':'unit',
          'parent_directory':str(tmp_path/'parent'),'cache_directory':str(tmp_path/'someone_else')}
    (root/'prefetch_plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='leaves the active parent'):run_prefetch_plan(p,{}, {})


def test_queue_cannot_relabel_a_different_native_configuration(tmp_path):
    root=tmp_path/'parent/audio/fresh';p=root/'token/request.json';p.parent.mkdir(parents=True)
    plan={'schema':'v7-fresh-rir-prefetch-plan-v1','plan_id':'unit',
          'parent_directory':str(tmp_path/'parent'),'cache_directory':str(tmp_path/'parent/cache'),
          'cache_identity':{'wrong':'scene'}}
    (root/'prefetch_plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='differs from the active native request'):
        run_prefetch_plan(p,{'expected_native_identity':{}}, {})


def test_measuring_worker_never_spawns_a_coordinator_without_a_queue(tmp_path,monkeypatch):
    from avengine.timeline.v7_phase2_rir_prefetch import maybe_schedule_prefetch
    def unexpected(*args,**kwargs):raise AssertionError('unexpected child process')
    monkeypatch.setattr('avengine.timeline.v7_phase2_rir_prefetch.subprocess.run',unexpected)
    assert maybe_schedule_prefetch(tmp_path/'root/token/request.json',{}, {})is None


def test_native_parallel_caller_does_not_activate_a_serial_queue(tmp_path,monkeypatch):
    from avengine.timeline.v7_phase2_rir_prefetch import maybe_schedule_prefetch
    root=tmp_path/'parent/audio/fresh';p=root/'token/request.json';p.parent.mkdir(parents=True)
    (root/'prefetch_plan.json').write_text('{}')
    def unexpected(*args,**kwargs):raise AssertionError('unexpected child process')
    monkeypatch.setattr('avengine.timeline.v7_phase2_rir_prefetch.subprocess.run',unexpected)
    assert maybe_schedule_prefetch(p,{'inputs':{'fresh_rir_workers':8}}, {})is None


def test_queued_cache_cannot_run_inside_an_independent_measuring_worker(tmp_path,monkeypatch):
    root=tmp_path/'parent/audio/fresh';p=root/'token/request.json';p.parent.mkdir(parents=True)
    (root/'prefetch_plan.json').write_text(json.dumps({'schema':'v7-fresh-rir-prefetch-plan-v1',
        'separate_coordinator_required':True}))
    monkeypatch.setattr('avengine.timeline.v7_phase2_rir_prefetch.sys.argv',['v7_phase2_fresh_rir.py'])
    assert run_prefetch_plan(p,{}, {})is None


def test_next_queue_revision_preserves_the_previous_plan(tmp_path):
    from avengine.timeline.v7_phase2_rir_prefetch import _plan_path
    (tmp_path/'prefetch_plan.json').write_text('old')
    (tmp_path/'prefetch_plan_002.json').write_text('new')
    (tmp_path/'prefetch_finished_unit.json').write_text('receipt')
    assert _plan_path(tmp_path).name=='prefetch_plan_002.json'
    assert (tmp_path/'prefetch_plan.json').read_text()=='old'
