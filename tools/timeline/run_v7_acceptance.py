#!/usr/bin/env python3
"""Run isolated native acceptance parents under a committed resource policy."""
from __future__ import annotations
import argparse,datetime,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path


def load(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def add_required_condition(row,condition):
    conditions=row['scene_demand'].setdefault('requested_conditions',[])
    if condition not in conditions:conditions.append(condition)


def run_worker(manifest,slot):
    task=manifest['parents'][slot];row=task['row'];root=Path(manifest['output_root']).resolve()
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice >=10 required')
    cpus=manifest.get('cpu_ids') or manifest.get('cpu_affinity_ids') or sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpus[slot%len(cpus)]})
    import avengine
    if not Path(avengine.__file__).resolve().is_relative_to(Path(task.get('snapshot_worktree',manifest['worktree'])).resolve()/'src'):
        raise ValueError('import escaped the acceptance worktree')
    from avengine.timeline.v7_storage_budget import ENV_KEY,check_write,launch_minimum_free_bytes
    budget_root=Path(manifest.get('resource_budget_root',str(root))).resolve()
    if not root.is_relative_to(budget_root):raise ValueError('parent output escaped the explicit task resource root')
    os.environ[ENV_KEY]=json.dumps({'root':str(budget_root),'filesystem':'/data','maximum_task_bytes':60_000_000_000,
        'minimum_free_bytes':100_000_000_000,'concurrent_write_margin_bytes':manifest.get('concurrent_write_margin_bytes',128_000_000),
        'policy':'owner-20261015-acceptance-only-no-training-or-CCI',
        'resource_admission_managed':True})
    check_write(root,expected_bytes=0)
    # Owner 2026-10-15: resource thresholds pause launches. A running parent
    # must be allowed to finish its writes after passing the launch check.
    active_storage=json.loads(os.environ[ENV_KEY]);active_storage['minimum_free_bytes']=0
    os.environ[ENV_KEY]=json.dumps(active_storage)
    inputs=load(task.get('runtime_input') or manifest['runtime_'+task['device']]);inputs['production_input_contract']=manifest['contract']
    inputs['launch_disk_policy']=manifest.get('launch_disk_policy',manifest.get('disk_launch_policy',{}))
    inputs.update(fresh_rir_workers=min(16,max(1,int(task.get('fresh_rir_workers',manifest.get('fresh_rir_workers',inputs.get('fresh_rir_workers',16)))))),fresh_rir_cpu_ids=list(inputs.get('fresh_rir_cpu_ids') or cpus))
    catalog=load(manifest['assets_'+task['device']]);catalog=catalog['assets']if isinstance(catalog,dict)else catalog
    houses=load(manifest['houses_'+task['device']])['houses'];house=next(h for h in houses if h['house']==row['house'])
    contract=load(manifest['contract']);source=load(contract['inputs']['source_admission']['path']);human=load(manifest['human_registered_cpu'] if task['profile']=='registered'and task['device']=='cpu'else manifest['human_'+task['device']])
    from avengine.timeline.v7_demand import read_demand_state,plan_phase2_demand,add_scene_demand,record_selected_parent,SELECTION_FEEDBACK_ENV
    os.environ[SELECTION_FEEDBACK_ENV]=str(root/'inputs/demand_state.json')
    state=read_demand_state(root/'inputs/demand_state.json')
    state['minimum_motion_independent_questions']=int(manifest.get('motion_quartet_final_targets',{}).get('minimum_independent_questions_per_family',30))
    if task['profile']=='phase2':
        previous=root/'inputs/planned'/f'{slot:03d}.json'
        if previous.exists():row=load(previous)
        elif task.get('focus')=='motion_quartet_template' or task.get('constructor')=='motion_quartet_template':
            from avengine.timeline.v7_motion_quartet_planning import plan_native_room_motion_quartet
            row=plan_native_room_motion_quartet(row,state,house,catalog,load(inputs['semantic_manifest']),source['records'],inputs,attempt=int(task.get('construction_attempt',0)))
        else:row=plan_phase2_demand(row,state,catalog,source['records'],contract,focus=task.get('focus'))
        if manifest.get('require_authored_tts_per_new_parent') and not previous.exists() and row.get('scene_constructor')!='motion_quartet_template':
            from avengine.timeline.v7_demand import assign_authored_tts
            row=assign_authored_tts(row,catalog,load(inputs['semantic_manifest']),rotation_index=task.get('voice_rotation_index',slot))
        inputs.update(phase2=True,integration_pilot=True,render_backend=inputs.get('render_backend','cpu_llvmpipe'))
        from avengine.timeline.v7_phase2_generation import generate_phase2_parent as generate
        if row.get('automatic_construction'):
            from avengine.timeline.v7_family_clock import generate_constructed_phase2_parent as generate
    else:
        previous=root/'inputs/planned'/f'{slot:03d}.json'
        if previous.exists():
            row=load(previous)
        elif task.get('focus','content_binding')=='content_binding':
            from avengine.timeline.v7_demand import plan_registered_content_demand
            row=plan_registered_content_demand(row,state,catalog,load(inputs['semantic_manifest']))
        else:
            row=add_scene_demand(row,state,focus=task.get('focus'))
        row['window_count']=5;row['window_start_ticks']=[0,50,100,150,200]
        add_required_condition(row,'authored_native_quartet')
        inputs['render_backend']='ue_spear' if task['device']=='ue' else 'cpu_llvmpipe'
        from avengine.timeline.v7_registered_generation import generate_registered_parent as generate
    if task['profile']=='phase2'and task.get('focus')=='human_motion'and not previous.exists():
        from avengine.timeline.v7_acceptance_motion_planner import prepare_conditioned_motion_gap
        row=prepare_conditioned_motion_gap(row,state,catalog,source['records'],inputs,house,load(inputs['semantic_manifest']))
    if task['profile']=='phase2':
        from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion,REVISION as EXECUTION_REVISION
        if not previous.exists():row['motion_execution_revision']=EXECUTION_REVISION
        if row.get('scene_constructor')!='motion_quartet_template' and not row.get('motion_quartet_template_revision'):row=prepare_scene_motion(row,catalog)
        from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
        row=prepare_motion_during_row(row,catalog,source['records'])
    write(root/'inputs/planned'/f'{slot:03d}.json',row)
    start=time.monotonic();cpu=time.process_time()
    receipt={'slot':slot,'device':task['device'],'house':row['house'],'room_id':row['room_id'],
        'scene_id':row['scene_id'],'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'code_commit':task.get('snapshot_commit',manifest['code_commit']),
        'code_snapshot':task.get('snapshot_worktree',manifest['worktree']),
        'selection_feedback_state':os.environ[SELECTION_FEEDBACK_ENV],'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    try:
        from avengine.timeline.v7_native_resource_slots import acquire_native_resource
        parent_execution_lease=acquire_native_resource('parent')
        native_root=Path(task.get('native_output_root') or root/'native').resolve()
        if not native_root.is_relative_to(root):raise ValueError('native output must remain inside this authorized task root')
        result=generate(row,house,inputs,catalog,source,human,str(native_root))
        admitted=result['status']in{'integration_pilot_admitted','formal_alpha_admitted'}
        if not admitted:raise ValueError('native producer did not admit the parent')
        parent=Path(result['delivery_directory']).parent
        if row.get('scene_constructor')=='motion_quartet_template':
            from avengine.timeline.v7_motion_quartet_planning import native_template_candidate_counts
            receipt['motion_quartet_native_realization']=native_template_candidate_counts(row,parent)
        # Derived deduplicated delivery is sparse. Export revalidates the original
        # raw seal with the unchanged 14-question gate, then deduplicates itself.
        pointer=next((p for p in [parent/'integration_seal_completed.json',parent/'alpha_seal_completed.json']if p.is_file()),None)
        if pointer is None:raise ValueError('original producer seal is absent')
        info={'slot':slot,'family':task['family'],'status':'passed','house':row['house'],
              'scene_id':result['scene_id'],'parent_directory':str(parent),'seal_path':str(pointer)}
        from tools.timeline.export_v7_trial import parent_payload
        rows,exported=parent_payload(info,root/'export',root)
        for item in rows:item['dataset_role']='acceptance_only'
        # Per-parent tables are never appended concurrently to a shared table.
        dest=root/'export'/'parent_rows'/f'{slot:03d}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows))
        from avengine.timeline.v7_s_batch_checks import open_delivered_media
        from avengine.timeline.v7_english_export import validate_english_model_text
        for item in rows:validate_english_model_text(item)
        # Native reader on this complete delivered parent, including all streams.
        readroot=root/'reader_checks_v2'/f'{slot:03d}';readroot.mkdir(parents=True,exist_ok=True)
        media_rows=[]
        for win in exported['windows']:
            wid=win['window_id'];folder=Path('parents')/result['scene_id']/'windows'/wid
            for name in ['video.mp4','audio_foa.wav','audio_mono.wav']:
                source=root/'export'/folder/name;target=readroot/folder/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():os.link(source,target)
            media_rows.append({'window_id':wid,'video':str(folder/'video.mp4'),'audio_foa':str(folder/'audio_foa.wav'),'audio_mono':str(folder/'audio_mono.wav')})
        (readroot/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in media_rows))
        reader=open_delivered_media(readroot)
        if reader['status']!='pass' or reader['actual_decoded_windows']!=len(exported['windows']):
            raise ValueError('delivered media reader failed: '+str(reader))
        feedback={**result,'windows':[{'family_counts':exported['question_family_counts'],'logical_questions':exported['independent_questions']}],'raw_logical_questions':exported['raw_export_questions'],'dedup_dropped':exported['dedup_dropped']}
        record_selected_parent(root/'inputs/demand_state.json',feedback)
        receipt.update(status='passed',producer_result=result,export=exported,reader=reader,
                       independent_questions=exported['independent_questions'],raw_questions=exported['raw_export_questions'])
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    receipt.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items()if k not in ['producer_result','export','reader','traceback']},ensure_ascii=False),flush=True)
    return 0 if receipt['status']=='passed'else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--worker',type=int,required=True)
    a=p.parse_args();m=load(a.manifest)
    if m.get('dataset_role')!='acceptance_only' or m.get('training_dataset_merge')is not False or m.get('CCI_transfer')is not False:
        raise ValueError('explicit isolated acceptance scope required')
    try:code=run_worker(m,a.worker)
    except Exception as exc:
        write(Path(m['output_root'])/'evidence/jobs'/f'{a.worker:03d}.json',{'slot':a.worker,'status':'failed','scene_id':m['parents'][a.worker]['row']['scene_id'],'house':m['parents'][a.worker]['row']['house'],'room_id':m['parents'][a.worker]['row']['room_id'],'pid':os.getpid(),'stage':'construction_preflight','reason':str(exc),'error_type':type(exc).__name__,'traceback':traceback.format_exc()})
        traceback.print_exc();code=2
    raise SystemExit(code)

if __name__=='__main__':main()
