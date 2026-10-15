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


def test_explicit_temporary_disk_floor_expires_at_the_configured_deadline():
    from avengine.timeline.v7_storage_budget import acceptance_minimum_free_bytes,policy_minimum_free_bytes
    before=datetime.datetime(2026,10,15,4,29,tzinfo=datetime.timezone.utc)
    after=datetime.datetime(2026,10,15,10,30,tzinfo=datetime.timezone.utc)
    configured={'minimum_free_bytes':60_000_000_000,'expires_utc':'2026-10-15T10:30:00+00:00','after_expiry_minimum_free_bytes':100_000_000_000}
    assert acceptance_minimum_free_bytes(before,configured)==60_000_000_000
    assert acceptance_minimum_free_bytes(after,configured)==100_000_000_000
    assert policy_minimum_free_bytes({'root':'/own/arbitrary-task','minimum_free_bytes':100_000_000_000,'launch_disk_policy':configured},before)==60_000_000_000
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
                available_memory=31_000_000_000,disk_policy={'minimum_free_bytes':60_000_000_000,'expires_utc':'2026-10-15T10:30:00+00:00','after_expiry_minimum_free_bytes':100_000_000_000})
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


def test_no_explicit_disk_exception_uses_normal_behavior_for_any_directory(tmp_path):
    from avengine.timeline.v7_storage_budget import resolve_launch_disk_policy,acceptance_minimum_free_bytes,policy_minimum_free_bytes
    assert resolve_launch_disk_policy()=={}
    assert resolve_launch_disk_policy(root=tmp_path/'never-configured')=={}
    assert acceptance_minimum_free_bytes()==100_000_000_000
    assert policy_minimum_free_bytes({'root':str(tmp_path/'arbitrary-task'),'minimum_free_bytes':100_000_000_000})==100_000_000_000


def test_missing_family_gets_a_slot_before_repeating_an_inflight_visual_family():
    from tools.timeline.monitor_v7_acceptance import construction_priority
    visual={'slot':1,'target_family':'QA-07'};motion={'slot':2,'target_family':'QA-15'}
    observed={};gaps={'QA-07':100.,'QA-15':50.};active={'QA-07':2}
    assert construction_priority(motion,observed,gaps,100.,active,{})<construction_priority(visual,observed,gaps,100.,active,{})
    observed['QA-15']=3
    assert construction_priority(visual,observed,gaps,100.,active,{})<construction_priority(motion,observed,gaps,100.,active,{})


def test_worker_keeps_its_frozen_snapshot_after_global_manifest_advances(tmp_path,monkeypatch):
    from tools.timeline.run_v7_acceptance import bind_worker_snapshot
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    monkeypatch.setattr('avengine.timeline.v7_frozen_code.source_commit',lambda root:'old_sha')
    monkeypatch.setattr('avengine.timeline.v7_frozen_code.verify_source',lambda root:True)
    manifest={'worktree':str(new),'code_commit':'new_sha','parents':[{'snapshot_worktree':str(old),'snapshot_commit':'old_sha'}]}
    actual=bind_worker_snapshot(manifest,0,old)
    assert actual['code_commit']=='old_sha'and actual['worktree']==str(old)
    assert manifest['code_commit']=='new_sha'
    with pytest.raises(ValueError,match='differs from its frozen task'):bind_worker_snapshot(manifest,0,new)


def test_fresh_rirs_cannot_escape_the_parent_allocation():
    from tools.timeline.run_v7_acceptance import worker_cpu_policy
    manifest={'cpu_affinity_ids':list(range(32)),'fresh_rir_workers':8}
    task={'cpu_ids':list(range(8,16))}
    parent,rir,workers=worker_cpu_policy(manifest,task,4,{'fresh_rir_cpu_ids':list(range(96,104))})
    assert parent==8 and rir==list(range(8,16)) and workers==8
    assert worker_cpu_policy(manifest,task,0,{})[1]==list(range(8,16))
    assert worker_cpu_policy(manifest,task,0,{'fresh_rir_cpu_ids':[9,96]})[1]==[9]


def test_authorized_task_size_is_configurable_without_stopping_active_jobs():
    now=datetime.datetime(2026,10,16,tzinfo=datetime.timezone.utc)
    values=dict(now=now,deadline=now+datetime.timedelta(hours=6),pss=1,limit=150_000_000_000,
        free=500_000_000_000,allocated_size=100_000_000_000,available_memory=100_000_000_000,
        disk_policy={'minimum_free_bytes':400_000_000_000},maximum_task_bytes=150_000_000_000)
    assert launch_pause_reason(**values)==''
    values['allocated_size']=150_000_000_000
    assert launch_pause_reason(**values)=='task_disk_limit_no_new_jobs'
    values.update(allocated_size=1,free=399_000_000_000)
    assert launch_pause_reason(**values)=='disk_headroom_no_new_jobs'


def test_configured_shared_memory_floor_reserves_headroom_for_running_jobs():
    from tools.timeline.monitor_v7_acceptance import parent_launch_headroom
    assert not parent_launch_headroom(0,0,7_000_000_000,150_000_000_000,46_000_000_000,40_000_000_000)
    assert parent_launch_headroom(0,0,7_000_000_000,150_000_000_000,47_000_000_000,40_000_000_000)
