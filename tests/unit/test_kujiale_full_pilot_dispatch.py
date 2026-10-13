"""Hermetic dispatch fixtures; these do not certify UE rendering or native gates."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from avengine.timeline import kujiale_asset_pilot as pilot


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def root(tmp_path):
    root=tmp_path/'ue_round21_pilot';(root/'inputs').mkdir(parents=True)
    house={'house':'kujiale_fixture','family':'kujiale','split':'train','acoustic_status':'pass',
           'checked_room_ids':['R0'],'checked_rooms':[{'room_label':'R0','native_qa_status':'pass',
           'floor_reference':{'status':'pass'}}],'input':{'ue_config':{}}}
    runtime={'render_backend':'ue_spear','houses':[house]}
    for directory in (root/'inputs',root/'inputs/full_coverage_v1'):
        for name,data in [('houses',{'houses':[house]}),('runtime',runtime),
                          ('assets',{'assets':[{'asset_id':'original_person'}]}),
                          ('source_admission',{'records':[{'input_version':'frozen'}]}),
                          ('human_admission',{'humans':['original']})]:
            put(directory/(name+'.json'),data)
    seed=root/'inputs/full_coverage_v1/seeds.jsonl'
    seed.write_text('\n'.join(json.dumps({'scene_id':f'new_{i}','house':house['house'],'room_id':'R0',
                'split':'train'}) for i in range(3))+'\n')
    put(root/'inputs/task_scope.json',{'physical_gpus':[1,2,3],'seed_path':str(seed),
        'deadline_epoch':9999999999,'required_formal_canary_scene_id':'required_canary',
        'require_formal_canary_before_rollout':True,'reviewed_parallel_batch_limit':2,
        'shared_parent_peak_non_ue_processes':2,'per_parent_peak_non_ue_processes':4,
        'per_parent_peak_pss_bytes':8_000_000_000,'worker_cpu_ids':[1,2,3,4]})
    put(root/'evidence/input_seal.json',{'status':'sealed_inputs'})
    put(root/'JOBS.json',[])
    (root/'logs').mkdir()
    (root/'scripts').mkdir()
    (root/'PROGRESS.md').write_text('')
    return root


def seal(root,scene,status='pass'):
    put(root/'evidence/parent_results'/f'{scene}.json',{'scene_id':scene,'status':'formal_alpha_admitted'})
    put(root/'production/parents'/scene/'alpha_seal_completed.json',{})
    put(root/'production/parents'/scene/'independent_truth_audit.json',
        {'status':status,'mismatches':[] if status=='pass' else ['truth differs']})


def test_default_input_keeps_frozen_source_and_all_full_inputs_can_change(root):
    directory,bundle=pilot.load_pilot_inputs(root)
    assert directory==root/'inputs'
    assert bundle['source_admission']['records'][0]['input_version']=='frozen'
    full=root/'inputs/full_coverage_v1'
    for name in ('houses','runtime','assets','source_admission','human_admission'):
        value=json.loads((full/(name+'.json')).read_text());value['derived_marker']=name
        put(full/(name+'.json'),value)
    directory,bundle=pilot.load_pilot_inputs(root,full)
    assert directory==full
    assert {name:value['derived_marker'] for name,value in bundle.items()}=={name:name for name in bundle}
    assert 'derived_marker' not in pilot.load_pilot_inputs(root)[1]['source_admission']


@pytest.mark.parametrize('kind',['directory','json','seed'])
def test_inputs_and_seed_reject_escaped_or_symlinked_paths(root,kind):
    external=root.parent/'outside';external.mkdir();put(external/'source.json',{})
    if kind=='directory':
        with pytest.raises(ValueError,match='own output root'):pilot.own_input_directory(root,external)
    elif kind=='json':
        (root/'inputs/full_coverage_v1/source_admission.json').unlink()
        (root/'inputs/full_coverage_v1/source_admission.json').symlink_to(external/'source.json')
        with pytest.raises(ValueError,match='escaped'):pilot.load_pilot_inputs(root,root/'inputs/full_coverage_v1')
    else:
        (root/'inputs/escaped.jsonl').symlink_to(external/'source.json')
        with pytest.raises(ValueError,match='own output root'):pilot.own_seed_path(root,root/'inputs/escaped.jsonl')


@pytest.mark.parametrize('change',[{'split':'dev'},{'room_id':'R9'}])
def test_only_original_admitted_training_rooms_can_be_used(root,change):
    houses=pilot.load_pilot_inputs(root)[1]['houses']['houses']
    row={'house':'kujiale_fixture','room_id':'R0','split':'train'}
    assert pilot.validate_training_room(root,row,houses)['house']==row['house']
    with pytest.raises(ValueError,match='training'):pilot.validate_training_room(root,dict(row,**change),houses)
    invented=deepcopy(houses);invented[0]['checked_room_ids']=['R9']
    invented[0]['checked_rooms'][0]['room_label']='R9'
    with pytest.raises(ValueError,match='original native UE admission'):
        pilot.validate_training_room(root,dict(row,room_id='R9'),invented)


def test_original_ue_room_and_floor_admissions_are_required(root):
    bundle=pilot.load_pilot_inputs(root)[1];houses=deepcopy(bundle['houses']['houses'])
    row={'house':'kujiale_fixture','room_id':'R0','split':'train'}
    houses[0]['checked_rooms'][0]['floor_reference']['status']='fail'
    with pytest.raises(ValueError,match='floor/QA'):pilot.validate_training_room(root,row,houses)
    runtime=bundle['runtime'];runtime['render_backend']='habitat'
    put(root/'inputs/full_coverage_v1/runtime.json',runtime)
    with pytest.raises(ValueError,match='native UE/SPEAR'):pilot.load_pilot_inputs(root,root/'inputs/full_coverage_v1')


def test_required_canary_cannot_be_replaced_by_an_old_formal_parent(root):
    scope=pilot.read_scope(root);rows=[{'scene_id':'old_good'}]
    seal(root,'old_good')
    assert not pilot.formal_canary_passed(root,scope,rows)
    seal(root,'required_canary',status='fail')
    assert not pilot.formal_canary_passed(root,scope,rows)
    seal(root,'required_canary')
    assert pilot.formal_canary_passed(root,scope,rows)
    (root/'production/parents/required_canary/alpha_seal_completed.json').unlink()
    assert not pilot.formal_canary_passed(root,scope,rows)


def test_full_inputs_cannot_launch_even_index_zero_before_required_canary(root,monkeypatch):
    monkeypatch.setattr(pilot,'live_parent_workers',lambda root:[])
    with pytest.raises(RuntimeError,match='required formal native canary'):
        pilot.require_launch_authorization(root,0,1,None,root/'inputs/full_coverage_v1',False)


@pytest.mark.parametrize('gpus',[[0],[1,0,2],[1,1],[4],[True],[]])
def test_gpu_zero_and_undeclared_or_invalid_gpu_lists_rejected(gpus):
    with pytest.raises(ValueError,match='GPU0'):pilot.authorized_gpus({'physical_gpus':gpus})


def test_shared_gpu_visibility_hides_zero_and_matches_dynamic_scope(tmp_path):
    for gpu in (1,2,3):(tmp_path/f'nvidia{gpu}').touch()
    pilot.require_shared_gpu_isolation({'physical_gpus':[1,2,3]},tmp_path)
    (tmp_path/'nvidia0').touch()
    with pytest.raises(ValueError,match='hide physical GPU0'):
        pilot.require_shared_gpu_isolation({'physical_gpus':[1,2,3]},tmp_path)
    (tmp_path/'nvidia0').unlink()
    with pytest.raises(ValueError,match='visibility'):
        pilot.require_shared_gpu_isolation({'physical_gpus':[2,3]},tmp_path)


def test_explicit_review_changes_batch_size_without_changing_first_canary_rule():
    assert pilot.reviewed_batch_limit({},True)==1
    gate=pilot.NativePilotBatchGate(False,4)
    assert gate.batch_limit==1
    gate.observe({'status':'formal_alpha_admitted','independent_truth':{'status':'pass'}})
    assert gate.batch_limit==4
    assert pilot.reviewed_batch_limit({'reviewed_parallel_batch_limit':4},True)==4
    with pytest.raises(ValueError):pilot.reviewed_batch_limit({'reviewed_parallel_batch_limit':13},True)


def test_peak_reserves_only_active_actual_deficits_and_initial_shared_tracker(root):
    scope=pilot.read_scope(root)
    active=[SimpleNamespace(name='a',peak_non_ue=2,peak_pss=8_000_000_000),
            SimpleNamespace(name='b',peak_non_ue=2,peak_pss=8_000_000_000)]
    measured={'processes':[{'jobs':['a'],'role':'python','pss_bytes':1_000_000_000},
              {'jobs':['a'],'role':'ue','pss_bytes':5_000_000_000},
              {'jobs':['b'],'role':'python','pss_bytes':1_000_000_000},
              {'jobs':['b'],'role':'ffmpeg','pss_bytes':100_000_000},
              {'jobs':['b'],'role':'ue','pss_bytes':5_000_000_000}]}
    processes,pss=pilot.launch_reservations(scope,measured,active,True,False)
    assert processes==3  # new Python+aux, plus a's missing aux; b already has its aux.
    assert pss==11_900_000_000  # new8GB + a2GB deficit + b1.9GB deficit.
    first=pilot.launch_reservations(scope,{'processes':[]},[],True,True)
    assert first==(3,8_000_000_000)
    assert pilot.launch_reservations(scope,{'processes':[]},[],False)==(4,8_000_000_000)
    with pytest.raises(ValueError,match='conservative'):
        pilot.launch_reservations({},measured,[],True)


def test_atomic_jobs_merge_preserves_other_jobs_and_child_completed_status(root):
    put(root/'JOBS.json',[{'name':'other_own_old','status':'completed','pid':42}])
    pilot.update_job(root,'new',{'status':'completed','pid':43,'returncode':0})
    pilot.update_job(root,'new',{'status':'running','command':['python','spawn'],'start_ticks':'100'},register=True)
    jobs=json.loads((root/'JOBS.json').read_text())
    assert jobs[0]=={'name':'other_own_old','status':'completed','pid':42}
    assert jobs[1]['status']=='completed' and jobs[1]['command']==['python','spawn']
    assert not list(root.glob('JOBS.json.*.pending'))


def test_exact_pid_start_time_cleanup_refuses_reused_pid(root,monkeypatch):
    calls=[];helper=SimpleNamespace(snapshot=lambda pid:{'pid':pid,'command':['different']},
        still_same=lambda record:False,stop_owned_tree=lambda *args:calls.append(args))
    monkeypatch.setattr(pilot,'load_local_helper',lambda *args:helper)
    handle=SimpleNamespace(pid=321,identity={'pid':321,'start_ticks':'old','command':['own']})
    with pytest.raises(RuntimeError,match='refusing PID'):pilot.cleanup_worker(root,handle)
    assert calls==[]


def test_worker_reads_full_bundle_and_shared_gpu_adapter_without_real_native_run(root,monkeypatch):
    full=root/'inputs/full_coverage_v1';captured={}
    for name in ('runtime','assets','source_admission','human_admission'):
        value=json.loads((full/(name+'.json')).read_text());value['marker']=name
        if name=='assets':value['assets']=[{'asset_id':'derived_person'}]
        put(full/(name+'.json'),value)
    (root/'scripts/ue_shared').mkdir();(root/'scripts/ue_shared/ue_gpu1').touch()
    monkeypatch.setattr(pilot,'prepare_worker_runtime',lambda *args:None)
    def generate(row,house,runtime,assets,source,human,out):
        captured.update(row=row,house=house,runtime=runtime,assets=assets,source=source,human=human)
        return {'scene_id':row['scene_id'],'status':'formal_alpha_admitted','delivery_directory':'fixture_only'}
    monkeypatch.setitem(sys.modules,'avengine.timeline.v7_registered_generation',SimpleNamespace(generate_registered_parent=generate))
    monkeypatch.setitem(sys.modules,'avengine.timeline.v7_storage_budget',SimpleNamespace(configure_kujiale_production=lambda *a,**k:None,check_write=lambda *a,**k:None))
    monkeypatch.setitem(sys.modules,'avengine.qa.kujiale_pilot_truth',SimpleNamespace(audit_sealed_parent=lambda *a:{'status':'pass','checked_forms':1,'mismatches':[]}))
    pilot.worker(root,0,1,None,full,True,1)
    assert captured['runtime']['marker']=='runtime'
    assert captured['source']['marker']=='source_admission'
    assert captured['human']['marker']=='human_admission'
    assert captured['assets']==[{'asset_id':'derived_person'}]
    ue=captured['house']['input']['ue_config']
    assert ue['physical_gpu_index']==1 and ue['graphics_adapter']==0
    assert ue['unreal_editor']==str(root/'scripts/ue_shared/ue_gpu1')
    assert not pilot.formal_native_parent_passed(root,'new_0')  # CPU fixture cannot certify UE.


def test_resource_preflight_uses_uuid_even_when_visible_index_differs(root,monkeypatch):
    put(root/'kujiale_native_gpu_policy.json',{'allowed_gpu_indices':[1,2,3],
        'gpu_uuids':{'1':'physical-one'},'measured_peak_mib':6144,'headroom_mib':4096})
    measured={'processes':[],'launch_allowed':True,'launch_hold_reasons':[],
        'hard_stop_reasons':[],'gpus':[{'physical_index':0,'uuid':'physical-one','free_bytes':20*1024**3}]}
    monkeypatch.setattr(pilot,'resource_snapshot',lambda *a,**k:deepcopy(measured))
    calls=[]
    def limits(measurement,scope,processes,pss):
        calls.append((processes,pss))
        return {'launch_allowed':True,'launch_hold_reasons':[],'hard_stop_reasons':[]}
    monkeypatch.setattr(pilot,'load_local_helper',lambda *a:SimpleNamespace(evaluate_limits=limits))
    _,proof=pilot.preflight_launch(root,1,[],False)
    assert proof['requested_physical_gpu_uuid']=='physical-one'
    assert calls==[(4,8_000_000_000)]
    measured['gpus'][0]['uuid']='wrong'
    with pytest.raises(pilot.ResourceLaunchHold,match='UUID_not_visible'):pilot.preflight_launch(root,1,[],False)


def coordinator_fakes(root,monkeypatch,bad=False):
    seal(root,'required_canary');spawned=[]
    monkeypatch.setattr(sys,'argv',['pilot','--pilot-root',str(root),'--input-directory',str(root/'inputs/full_coverage_v1'),
        '--limit','3','--workers-per-gpu','1'])
    monkeypatch.setattr(pilot,'live_parent_workers',lambda *a:[])
    monkeypatch.setattr(pilot.os,'nice',lambda value:10)
    monkeypatch.setattr(pilot.os,'sched_getaffinity',lambda value:{1,2,3,4})
    monkeypatch.setattr(pilot.os,'sched_setaffinity',lambda *a:None)
    monkeypatch.setattr(pilot.signal,'signal',lambda *a:None)
    monkeypatch.setattr(pilot.time,'sleep',lambda *a:None)
    monkeypatch.setattr(pilot,'load_local_helper',lambda *a:SimpleNamespace(still_same=lambda row:False))
    def launch(root,index,gpu,*args,**kwargs):
        spawned.append((index,gpu))
        if bad:
            put(root/'evidence/parent_results'/f'new_{index}.json',{'status':'failed','error_type':'ValueError',
                'reason':f'only {4 if index==0 else 0} measured binding questions; need 5'})
        else:seal(root,f'new_{index}')
        return SimpleNamespace(index=index,gpu=gpu,pid=100+index,name=f'parent_fake_{index}',
            identity={'start_ticks':'1'},kind='legacy',started=pilot.time.time(),
            poll=lambda:1 if bad else 0,wait=lambda **kwargs:None,observed_descendants={})
    monkeypatch.setattr(pilot,'launch_worker',launch)
    return spawned


def test_coordinator_runs_one_reviewed_batch_and_leaves_remaining_unqueued(root,monkeypatch):
    spawned=coordinator_fakes(root,monkeypatch)
    pilot.main()
    assert spawned==[(0,1),(1,2)]
    state=json.loads((root/'evidence/coordinator_state.json').read_text())
    assert state['status']=='batch_review_required' and state['pending_not_queued']==[2]
    assert state['parents']==state['passed']==2
    assert state['training_exported'] is False and state['CCI_transferred'] is False


def test_second_identical_failure_prevents_next_invocation_from_dispatching(root,monkeypatch):
    spawned=coordinator_fakes(root,monkeypatch,bad=True)
    pilot.main();assert len(spawned)==2
    state=json.loads((root/'evidence/coordinator_state.json').read_text())
    assert state['gate']['consecutive_failures']==2
    with pytest.raises(RuntimeError,match='two consecutive'):pilot.main()
    assert len(spawned)==2



def test_actual_worker_initialization_writes_single_cpu_and_pool_evidence(root):
    helper_source=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round21_pilot/scripts/pilot_resource_snapshot_v3.py')
    if not helper_source.is_file():pytest.skip('task-local resource writer unavailable outside this pilot machine')
    import os
    from shutil import copyfile
    copyfile(helper_source,root/'scripts/pilot_resource_snapshot_v3.py')
    cpus=os.sched_getaffinity(0)
    try:pilot.prepare_worker_runtime(min(cpus),root)
    finally:os.sched_setaffinity(0,cpus)
    helper=pilot.load_local_helper(root,'pilot_resource_snapshot_v3.py')
    actual=helper.proc_identity(os.getpid())
    proof=json.loads(helper.thread_configuration_path(root,actual).read_text())
    assert proof['pid']==os.getpid() and proof['start_ticks']==actual['start_ticks']
    assert proof['cpu_affinity']==[min(cpus)] and proof['opencv_threads']==1
    assert helper.configuration_verifies_single_pool(proof)



@pytest.mark.parametrize('shared',[True,False])
def test_launch_uses_shared_spawn_or_direct_bwrap_with_no_per_parent_tracker(root,monkeypatch,shared):
    import multiprocessing
    scope=pilot.read_scope(root);row={'scene_id':'new_0'};full=root/'inputs/full_coverage_v1'
    put(root/'ue_env.json',{'PYTHONPATH':'native_sdk'})
    monkeypatch.setattr(pilot,'require_launch_authorization',lambda *a:(scope,row,full))
    monkeypatch.setattr(pilot,'preflight_launch',lambda *a:(scope,{'launch_allowed':True}))
    monkeypatch.setattr(pilot.os,'environ',dict(pilot.os.environ,CUDA_VISIBLE_DEVICES='2'))
    process=SimpleNamespace(pid=6543,exitcode=None,start=lambda:None,poll=lambda:None)
    recorded=[]
    def popen(command,**kwargs):
        recorded.append(command);return process
    def make_process(**kwargs):
        recorded.append(kwargs);return process
    monkeypatch.setattr(multiprocessing,'get_context',lambda mode:SimpleNamespace(Process=make_process))
    monkeypatch.setattr(pilot.subprocess,'Popen',popen)
    command=['python','-c','from multiprocessing.spawn import spawn_main; spawn_main(...)','--multiprocessing-fork'] if shared else ['/usr/bin/bwrap','--dev','/dev']
    monkeypatch.setattr(pilot,'stable_process_identity',lambda *a:{'pid':6543,'ppid':123,'start_ticks':'456','command':command})
    handle=pilot.launch_worker(root,0,1,0,None,full,shared,49)
    jobs=json.loads((root/'JOBS.json').read_text());job=jobs[-1]
    assert job['pid']==6543 and job['command']==command and job['start_ticks']=='456'
    assert 'supervisor_pid' not in job and job['input_directory']==str(full)
    assert job['uid']==pilot.os.getuid() and job['job_type']=='native_parent_worker'
    assert job['task_root']==str(root.resolve()) and job['gpu_physical']==1
    assert handle.kind==('shared' if shared else 'legacy')
    if shared:
        assert recorded[0]['target'] is pilot.shared_worker_entry
        assert recorded[0]['args'][4]==full and recorded[0]['args'][5]==49
        assert 'CUDA_VISIBLE_DEVICES' not in pilot.os.environ
    else:
        assert recorded[0][:4]==['nice','-n','10','/usr/bin/bwrap']
        assert 'tracked_job.py' not in ' '.join(recorded[0])
        assert recorded[0][recorded[0].index('--input-directory')+1]==str(full)
        assert '/dev/nvidia1' in recorded[0] and '/dev/nvidia0' not in recorded[0]



def test_failed_canonical_and_every_attempt_are_immutable_and_latest_is_used(root):
    scene='required_canary';old={'scene_id':scene,'status':'failed','reason':'old native failure'}
    original=pilot.publish_parent_result(root,scene,old)
    before=original.read_bytes()
    newer={'scene_id':scene,'status':'failed','reason':'later measured failure'}
    attempt=pilot.publish_parent_result(root,scene,newer)
    assert original.name==scene+'.json' and '.attempt_' in attempt.name
    assert original.read_bytes()==before
    assert pilot.read_parent_result(root,scene)==newer
    good={'scene_id':scene,'status':'formal_alpha_admitted'}
    final=pilot.publish_parent_result(root,scene,good)
    put(root/'production/parents'/scene/'alpha_seal_completed.json',{})
    put(root/'production/parents'/scene/'independent_truth_audit.json',{'status':'pass','mismatches':[]})
    assert pilot.formal_canary_passed(root,pilot.read_scope(root),[])
    assert pilot.result_for_worker(root,{'scene_id':scene},0)['status']=='formal_alpha_admitted'
    assert original.read_bytes()==before and json.loads(attempt.read_text())==newer
    pointer=json.loads((original.parent/(scene+'.latest.json')).read_text())
    assert pointer['result_path']==str(final)


def test_latest_result_supports_old_timestamp_records_and_exact_sid_only(root):
    directory=pilot.parent_result_directory(root)
    put(directory/'old_scene.json',{'status':'failed','reason':'canonical'})
    put(directory/'old_scene_1791910000000000000.json',{'status':'failed','reason':'old immutable record'})
    put(directory/'old_scene_repair1.json',{'status':'formal_alpha_admitted'})
    assert pilot.read_parent_result(root,'old_scene')['reason']=='old immutable record'
    with pytest.raises(ValueError):pilot.publish_parent_result(root,'../escape',{})


def test_same_failed_canary_sid_can_resume_default_inputs_before_formal_pass(root,monkeypatch):
    seed=pilot.own_seed_path(root)
    rows=[json.loads(line) for line in seed.read_text().splitlines()]
    rows[0]['scene_id']='required_canary';seed.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    pilot.publish_parent_result(root,'required_canary',{'status':'failed','reason':'wrapper parser failed in v1'})
    monkeypatch.setattr(pilot,'live_parent_workers',lambda *a:[])
    _,row,directory=pilot.require_launch_authorization(root,0,2,seed,None,False)
    assert row['scene_id']=='required_canary' and directory==root/'inputs'
    assert not pilot.formal_canary_passed(root,pilot.read_scope(root),rows)
    with pytest.raises(RuntimeError,match='not authorized'):
        pilot.require_launch_authorization(root,0,2,seed,root/'inputs/full_coverage_v1',False)
