import datetime
import json
from pathlib import Path
import pytest
from tools.timeline.monitor_v7_acceptance import launch_pause_reason
from avengine.timeline.v7_storage_budget import ENV_KEY, StorageBudgetExceeded, check_write

pytestmark = pytest.mark.fast_unit

@pytest.mark.parametrize("override,reason", [
    ({"pss": 141_000_000_000}, "PSS_limit_no_new_jobs"),
    ({"free": 99_000_000_000}, "disk_headroom_no_new_jobs"),
    ({"available_memory": 29_000_000_000}, "memory_headroom_no_new_jobs"),
    ({"allocated_size": 61_000_000_000}, "task_disk_limit_no_new_jobs"),
    ({"now": datetime.datetime(2026,10,15,4,tzinfo=datetime.timezone.utc)}, "delivery_deadline_no_new_jobs"),
])
def test_resource_pressure_only_changes_new_job_admission(override,reason):
    fields=dict(now=datetime.datetime(2026,10,15,0,tzinfo=datetime.timezone.utc),
                deadline=datetime.datetime(2026,10,15,4,tzinfo=datetime.timezone.utc),
                pss=1, limit=140_000_000_000, free=120_000_000_000,
                allocated_size=1, available_memory=50_000_000_000)
    fields.update(override)
    assert launch_pause_reason(**fields)==reason


def test_active_job_resource_policy_keeps_own_output_boundary(tmp_path,monkeypatch):
    root=tmp_path/'own';root.mkdir()
    monkeypatch.setenv(ENV_KEY,json.dumps(dict(root=str(root),resource_admission_managed=True)))
    # No resource read or allocation can interrupt the admitted producer.
    assert check_write(root/'media',expected_bytes=10**15)['resource_admission_managed']
    with pytest.raises(StorageBudgetExceeded,match='escaped'):
        check_write(tmp_path/'other')


def test_registered_required_condition_is_idempotent_for_recipe_resume():
    from copy import deepcopy
    from tools.timeline.run_v7_acceptance import add_required_condition
    row={'scene_demand':{'requested_conditions':['two_speakers']}}
    add_required_condition(row,'authored_native_quartet');expected=deepcopy(row)
    add_required_condition(row,'authored_native_quartet')
    assert row==expected


def test_historical_recipe_conditions_are_preserved_during_resume():
    from tools.timeline.run_v7_acceptance import add_required_condition
    row={'scene_demand':{'requested_conditions':['authored_native_quartet']*3}}
    add_required_condition(row,'authored_native_quartet')
    assert len(row['scene_demand']['requested_conditions'])==3


def test_forecast_includes_peer_workers_before_admitting_another_parent():
    from tools.timeline.monitor_v7_acceptance import combined_peak_budget
    gb=1_000_000_000
    assert combined_peak_budget(90*gb,80*gb,7*gb,36*gb)<=135*gb
    assert combined_peak_budget(97*gb,83*gb,7*gb,36*gb)>135*gb


def test_actual_cpu_counter_keeps_time_of_waited_native_children():
    import os,subprocess,sys
    from tools.timeline.monitor_v7_acceptance import proc
    before=proc(os.getpid())['cpu_s']
    subprocess.run([sys.executable,'-c','sum(range(5_000_000))'],check=True)
    assert proc(os.getpid())['cpu_s']-before>.02


def test_independent_helper_sdk_prefetch_is_included_without_worker_receipt(tmp_path):
    import subprocess,sys
    from tools.timeline.monitor_v7_acceptance import peer_processes
    root=tmp_path/'approved_helper';root.mkdir()
    child=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()',str(root)],stdin=subprocess.PIPE)
    try:
        records=peer_processes({'peer_output_roots':[str(root)]})
        assert any(r['pid']==child.pid for r in records)
    finally:
        child.stdin.close();child.wait(timeout=5)
