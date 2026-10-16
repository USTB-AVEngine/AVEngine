import json
import pytest
from tools.timeline.monitor_v7_acceptance import next_parent_log,retained_parent_log
pytestmark=pytest.mark.fast_unit


def test_requeued_slot_gets_a_new_log_without_overwriting_prior_evidence(tmp_path):
    (tmp_path/'logs').mkdir();old=tmp_path/'logs/parent_008.log';old.write_text('original failed affinity attempt\n')
    actual=next_parent_log(tmp_path,8)
    assert actual!=old and not actual.exists()
    with actual.open('x')as stream:stream.write('new dispatch\n')
    assert old.read_text()=='original failed affinity attempt\n'
    (tmp_path/'evidence/launches').mkdir(parents=True)
    (tmp_path/'evidence/launches/008.json').write_text(json.dumps({'log':str(actual)}))
    assert retained_parent_log(tmp_path,8)==actual


def test_optional_pool_preserves_unconfigured_command():
    from tools.timeline.monitor_v7_acceptance import registered_parent_command
    command=['python','worker.py','--worker','5']
    assert registered_parent_command(command,pool_root=None,cpus=[0,1],owner='trial')==command


def test_pool_registers_exact_producer_cores_before_worker():
    from tools.timeline.monitor_v7_acceptance import registered_parent_command
    command=['python','worker.py','--worker','5']
    wrapped=registered_parent_command(command,pool_root='/explicit/pool',cpus=[96,97,104],owner='trial5',python_executable='/python')
    assert wrapped[:5]==['/python','-B','-m','avengine.timeline.v7_rir_work_pool','--root']
    assert wrapped[wrapped.index('--cpus')+1]=='96,97,104'
    assert wrapped[wrapped.index('--owner')+1]=='trial5'
    assert wrapped[-len(command):]==command
    assert wrapped.index('run')<wrapped.index('--')
