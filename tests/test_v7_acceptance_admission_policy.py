import datetime
import json
from pathlib import Path
import pytest
from tools.timeline.monitor_v7_acceptance import launch_pause_reason
from avengine.timeline.v7_storage_budget import ENV_KEY, StorageBudgetExceeded, check_write

pytestmark = pytest.mark.fast_unit

@pytest.mark.parametrize("override,reason", [
    ({"pss": 141_000_000_000}, "PSS_limit_no_new_jobs"),
    ({"free": 59_000_000_000}, "disk_headroom_no_new_jobs"),
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


def test_owner_temporary_disk_floor_expires_at_1230_singapore():
    from avengine.timeline.v7_storage_budget import acceptance_minimum_free_bytes,policy_minimum_free_bytes
    before=datetime.datetime(2026,10,15,4,29,tzinfo=datetime.timezone.utc)
    after=datetime.datetime(2026,10,15,10,30,tzinfo=datetime.timezone.utc)
    assert acceptance_minimum_free_bytes(before)==60_000_000_000
    assert acceptance_minimum_free_bytes(after)==100_000_000_000
    assert policy_minimum_free_bytes({'root':'/own/v7_acceptance_20261015','minimum_free_bytes':100_000_000_000},before)==60_000_000_000
    assert policy_minimum_free_bytes({'root':'/other/review','minimum_free_bytes':100_000_000_000},before)==100_000_000_000


def test_temporary_floor_never_reintroduces_a_write_guard_for_admitted_job():
    from avengine.timeline.v7_storage_budget import policy_minimum_free_bytes
    policy={'root':'/own/v7_gap_motion_20261015','minimum_free_bytes':0}
    assert policy_minimum_free_bytes(policy,datetime.datetime(2026,10,15,2,tzinfo=datetime.timezone.utc))==0
    assert policy_minimum_free_bytes(policy,datetime.datetime(2026,10,15,5,tzinfo=datetime.timezone.utc))==0


def test_owner_explicit_parallel_increase_retains_cpu_and_memory_guards():
    from tools.timeline.monitor_v7_acceptance import admission_limits
    now=datetime.datetime(2026,10,15,2,tzinfo=datetime.timezone.utc)
    limits=admission_limits({'maximum_pss_bytes':180_000_000_000,'maximum_parents':18},now)
    assert limits=={'maximum_pss_bytes':180_000_000_000,'maximum_parents':18,'maximum_cpu_processes':80}
    fields=dict(now=now,deadline=now+datetime.timedelta(hours=2),pss=150_000_000_000,
                limit=limits['maximum_pss_bytes'],free=80_000_000_000,allocated_size=10_000_000_000,
                available_memory=31_000_000_000)
    assert launch_pause_reason(**fields)==''
    fields['available_memory']=29_000_000_000
    assert launch_pause_reason(**fields)=='memory_headroom_no_new_jobs'
    fields.update(available_memory=31_000_000_000,pss=180_000_000_000)
    assert launch_pause_reason(**fields)=='PSS_limit_no_new_jobs'


def test_independent_own_auxiliary_sdk_is_in_the_resource_budget(tmp_path):
    import subprocess,sys
    from tools.timeline.monitor_v7_acceptance import output_processes
    root=tmp_path/'own_acceptance';root.mkdir()
    child=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()',str(root)],stdin=subprocess.PIPE)
    try:
        assert any(p['pid']==child.pid for p in output_processes([root]))
    finally:
        child.stdin.close();child.wait(timeout=5)


def test_owner_workload_budget_and_host_memory_have_separate_scopes():
    from tools.timeline.monitor_v7_acceptance import parent_launch_headroom
    gb=1_000_000_000
    assert parent_launch_headroom(100*gb,110*gb,7*gb,180*gb,40*gb)
    assert not parent_launch_headroom(100*gb,110*gb,7*gb,180*gb,36*gb)
    assert not parent_launch_headroom(120*gb,110*gb,7*gb,180*gb,40*gb)
    assert not parent_launch_headroom(170*gb,174*gb,7*gb,180*gb,80*gb)


def test_combined_workloads_and_critical_memory_pause_new_launches():
    fields=dict(now=datetime.datetime(2026,10,15,3,tzinfo=datetime.timezone.utc),
        deadline=datetime.datetime(2026,10,15,4,tzinfo=datetime.timezone.utc),pss=100_000_000_000,
        limit=150_000_000_000,combined_pss=191_000_000_000,combined_limit=190_000_000_000,
        free=90_000_000_000,allocated_size=40_000_000_000,available_memory=40_000_000_000)
    assert launch_pause_reason(**fields)=='combined_PSS_limit_no_new_jobs'
    fields.update(combined_pss=130_000_000_000,available_memory=19_000_000_000)
    assert launch_pause_reason(**fields)=='critical_memory_ranked_alert_no_new_jobs'
