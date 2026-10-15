#!/usr/bin/env python3
"""Run isolated native acceptance parents under a committed resource policy."""
from __future__ import annotations
from avengine.timeline.v7_acceptance_tasks import task_by_slot
import argparse,datetime,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path


def load(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def add_required_condition(row,condition):
    conditions=row['scene_demand'].setdefault('requested_conditions',[])
    if condition not in conditions:conditions.append(condition)


def configure_walkable_maps(manifest):
    """One map cache is shared by construction and native admission workers."""
    value=manifest.get('walkable_map_root')
    if value is None:return
    root=Path(value).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError('configured walkable map cache root is missing: '+str(root))
    os.environ['AVENGINE_WALKABLE_MAP_ROOT']=str(root)
    os.environ['AVENGINE_REQUIRE_WALKABLE_MAP']='1'


def worker_cpu_policy(manifest,task,slot,runtime,*,inherited_cpu_ids=None):
    """Keep fresh SDK processes inside this parent's scheduler allocation."""
    allocated=list(task.get('cpu_ids') or manifest.get('cpu_ids') or manifest.get('cpu_affinity_ids') or sorted(os.sched_getaffinity(0)))
    if inherited_cpu_ids is not None:allocated=[cpu for cpu in allocated if cpu in inherited_cpu_ids]
    if not allocated:raise ValueError('native parent has no allocated CPU inside the inherited taskset')
    requested=runtime.get('fresh_rir_cpu_ids')
    rir=[int(cpu) for cpu in requested if int(cpu) in allocated] if requested else list(allocated)
    if not rir:rir=list(allocated)
    workers=min(16,max(1,int(task.get('fresh_rir_workers',manifest.get('fresh_rir_workers',runtime.get('fresh_rir_workers',8))))))
    return allocated[0] if task.get('cpu_ids') else allocated[slot%len(allocated)],rir,workers


def question_worker_cpu_policy(manifest,task,allocated_cpu_ids):
    cpus=sorted(set(allocated_cpu_ids))
    if not cpus:raise ValueError('question workers need an admitted parent CPU group')
    requested=int(task.get('registered_question_workers',manifest.get('registered_question_workers',1)))
    if requested<1:raise ValueError('at least one registered question worker required')
    count=min(requested,5,len(cpus))
    return {'registered_question_workers':count,'registered_question_cpu_ids':cpus[:count]}


def run_worker(manifest,slot):
    configure_walkable_maps(manifest)
    task=dict(task_by_slot(manifest,slot));row=task['row'];root=Path(manifest['output_root']).resolve();prepared_native=None
    dispatch_profile=root/'inputs/dispatch_profiles'/f'{slot:03d}.json'
    if dispatch_profile.exists():task['profile']=load(dispatch_profile)['profile']
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice >=10 required')
    cpus=manifest.get('cpu_ids') or manifest.get('cpu_affinity_ids') or sorted(os.sched_getaffinity(0))
    runtime_path=task.get('runtime_input') or manifest['runtime_'+task['device']]
    runtime=load(runtime_path)
    inherited_cpus=set(os.sched_getaffinity(0))
    parent_cpu,rir_cpus,rir_workers=worker_cpu_policy(manifest,task,slot,runtime,inherited_cpu_ids=inherited_cpus)
    os.sched_setaffinity(0,{parent_cpu})
    import avengine
    if not Path(avengine.__file__).resolve().is_relative_to(Path(task.get('snapshot_worktree',manifest['worktree'])).resolve()/'src'):
        raise ValueError('import escaped the acceptance worktree')
    from avengine.timeline.v7_unattended_resources import configure_parent_storage
    configure_parent_storage(manifest,root)
    inputs=runtime;inputs['production_input_contract']=manifest['contract']
    inputs['launch_disk_policy']=manifest.get('launch_disk_policy',manifest.get('disk_launch_policy',{}))
    inputs.update(fresh_rir_workers=rir_workers,fresh_rir_cpu_ids=rir_cpus)
    inputs['cpu_visual_threads']=int(task.get('cpu_visual_threads',manifest.get('cpu_visual_threads',inputs.get('cpu_visual_threads',1))))
    from avengine.timeline.v7_cpu_visual_runtime import configure_cpu_visual
    visual_cpu_policy=configure_cpu_visual(inputs,cpu_ids=[cpu for cpu in(task.get('cpu_ids')or cpus)if cpu in inherited_cpus])
    inputs.update(question_worker_cpu_policy(manifest,task,[cpu for cpu in(task.get('cpu_ids')or cpus)if cpu in inherited_cpus]))
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
        elif task.get('constructor')=='automatic_family':
            from avengine.timeline.v7_family_dispatch import construct_requested_scene
            room=next(r for r in contract['rooms']if r['house']==row['house']and r['room_id']==row['room_id'])
            proposal=construct_requested_scene(room,house,state,catalog,source['records'],load(inputs['semantic_manifest']),contract,inputs,
                plugin_specs=task.get('family_plugins',manifest.get('family_plugins',{})),
                scene_id=row['scene_id'],seed=row['scene_seed'],family=task['target_family'],attempt=int(task.get('construction_attempt',0)))
            if proposal['profile']not in {'phase2','registered'}:raise ValueError('unsupported automatic constructor profile: '+str(proposal['profile']))
            task['profile']=proposal['profile'];write(dispatch_profile,{'profile':task['profile'],'constructor':task.get('constructor'),'target_family':task.get('target_family')})
            if task['profile']=='registered'and task['device']=='cpu':human=load(manifest['human_registered_cpu'])
            row=proposal['row']
            prepared_native=proposal.get('prepared_native',proposal.get('prepared'))
            if prepared_native is not None:row=prepared_native.row
        elif task.get('focus')=='motion_quartet_template' or task.get('constructor')=='motion_quartet_template':
            from avengine.timeline.v7_motion_quartet_planning import plan_native_room_motion_quartet
            if task.get('target_family')in {'QA-06','QA-15','QA-16','QA-17'}:row['motion_requested_families']=[task['target_family']]
            row=plan_native_room_motion_quartet(row,state,house,catalog,load(inputs['semantic_manifest']),source['records'],inputs,attempt=int(task.get('construction_attempt',0)))
        else:
            def construction_context():
                from avengine.acoustics.runtime import load_habitat_runtime
                hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime']);pf=hs.PathFinder()
                if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('registered native NavMesh missing')
                return {'room':next(r for r in house['checked_rooms']if r['room_label']==row['room_id']),'semantic_manifest':load(inputs['semantic_manifest']),'pathfinder':pf}
            row=plan_phase2_demand(row,state,catalog,source['records'],contract,focus=task.get('focus'),construction_context=construction_context)
        if manifest.get('require_authored_tts_per_new_parent') and not previous.exists() and row.get('scene_constructor')!='motion_quartet_template' and not row.get('automatic_construction'):
            from avengine.timeline.v7_demand import assign_authored_tts
            row=assign_authored_tts(row,catalog,load(inputs['semantic_manifest']),rotation_index=task.get('voice_rotation_index',slot))
        if task['profile']=='phase2':
            inputs.update(phase2=True,integration_pilot=True,render_backend=inputs.get('render_backend','cpu_llvmpipe'))
            from avengine.timeline.v7_phase2_generation import generate_phase2_parent as generate
            if row.get('automatic_construction'):
                from avengine.timeline.v7_family_clock import generate_constructed_phase2_parent as generate
        else:
            row['window_count']=5;row['window_start_ticks']=[0,50,100,150,200]
            add_required_condition(row,'authored_native_quartet')
            inputs.update(phase2=False,integration_pilot=True,render_backend='cpu_llvmpipe')
            from avengine.timeline.v7_registered_generation import generate_registered_parent as generate
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
    if task['profile']=='phase2'and prepared_native is None and task.get('focus')=='human_motion'and not previous.exists():
        from avengine.timeline.v7_acceptance_motion_planner import prepare_conditioned_motion_gap
        row=prepare_conditioned_motion_gap(row,state,catalog,source['records'],inputs,house,load(inputs['semantic_manifest']))
    if task['profile']=='phase2'and prepared_native is None:
        from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion,REVISION as EXECUTION_REVISION
        if not previous.exists():row['motion_execution_revision']=EXECUTION_REVISION
        if row.get('scene_constructor')!='motion_quartet_template' and not row.get('motion_quartet_template_revision'):row=prepare_scene_motion(row,catalog)
        from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
        row=prepare_motion_during_row(row,catalog,source['records'])
    write(root/'inputs/planned'/f'{slot:03d}.json',row)
    start=time.monotonic();cpu=time.process_time()
    receipt={'slot':slot,'device':task['device'],'house':row['house'],'room_id':row['room_id'],
        'scene_id':row['scene_id'],'execution_profile':task['profile'],'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'visual_cpu_policy':visual_cpu_policy,'allocated_cpu_ids':task.get('cpu_ids',cpus),'fresh_rir_cpu_ids':rir_cpus,'fresh_rir_workers':rir_workers,
        'code_commit':task.get('snapshot_commit',manifest['code_commit']),
        'code_snapshot':task.get('snapshot_worktree',manifest['worktree']),
        'selection_feedback_state':os.environ[SELECTION_FEEDBACK_ENV],'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    try:
        from avengine.timeline.v7_native_resource_slots import acquire_native_resource
        parent_execution_lease=acquire_native_resource('parent')
        native_root=Path(task.get('native_output_root') or root/'native').resolve()
        if not native_root.is_relative_to(root):raise ValueError('native output must remain inside this authorized task root')
        if task.get('automatic_registered_room'):
            from avengine.timeline.v7_kujiale_category_qualification import qualify_missing_humans
            from avengine.timeline.v7_kujiale_cpu_phases import cpu_phase
            from avengine.registry.kujiale_room_categories import bind_formal_room_category,required_category_witnesses
            row,inputs,formal_category=bind_formal_room_category(row,inputs,native_root)
            if formal_category:receipt['formal_native_room_category_binding']=formal_category
            current_contract=load(inputs['production_input_contract']);cell=next(r for r in current_contract['rooms']if r['key']==row['house']+'::'+row['room_id']);missing=set(required_category_witnesses(row,inputs))-set(cell['categories'])
            if missing:
                with cpu_phase(root,'new_native_category',estimate_bytes=16000000000 if missing-{'human.standing'}else 12000000000):
                    inputs,category=qualify_missing_humans(row,house,inputs,catalog,source,human,current_contract,root)
            else:category=None
            if category:receipt['new_current_native_category_qualification']=category
        if task.get('automatic_registered_room'):
            from avengine.timeline.v7_registered_native_workers import prepare_native
            from avengine.timeline.v7_kujiale_cpu_phases import cpu_phase
            prepare_native(row,house,inputs,catalog,source,human,str(native_root),root,source_path=contract['inputs']['source_admission']['path'],human_path=manifest['human_ue'],catalog_path=manifest['assets_ue'])
            phases.enter_context(cpu_phase(root,'audio_questions_reader',estimate_bytes=2000000000))
            result=generate(row,house,inputs,catalog,source,human,str(native_root))
        elif prepared_native is not None:
            try:result=generate(row,house,inputs,catalog,source,human,str(native_root),prepared_native=prepared_native)
            finally:prepared_native.close()
        else:result=generate(row,house,inputs,catalog,source,human,str(native_root))
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
              'scene_id':result['scene_id'],'parent_directory':str(parent),'seal_path':str(pointer),
              'drop_invalid_speed_windows':manifest.get('drop_invalid_speed_windows',True)}
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
        feedback={**result,'motion_source_type':row.get('motion_source_type'),'windows':[{'family_counts':exported['question_family_counts'],'logical_questions':exported['independent_questions']}],'raw_logical_questions':exported['raw_export_questions'],'dedup_dropped':exported['dedup_dropped']}
        record_selected_parent(root/'inputs/demand_state.json',feedback)
        receipt.update(status='passed',producer_result=result,export=exported,reader=reader,
                       independent_questions=exported['independent_questions'],raw_questions=exported['raw_export_questions'])
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    receipt.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    if receipt['status']=='passed':
        from avengine.timeline.v7_acceptance_publication import publish_admitted_parent
        publish_admitted_parent(root,root/'evidence/jobs'/f'{slot:03d}.json')
        receipt=load(root/'evidence/jobs'/f'{slot:03d}.json')
        from avengine.timeline.v7_construction_retry import queue_unmet_family_retry
        reconstruction=queue_unmet_family_retry(root/'inputs/ACCEPTANCE_PLAN.json',receipt)
        if reconstruction is not None:
            receipt['unmet_target_family_reconstruction_slot']=reconstruction;write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items()if k not in ['producer_result','export','reader','traceback']},ensure_ascii=False),flush=True)
    return 0 if receipt['status']=='passed'else 2



def bind_worker_snapshot(manifest,slot,entry_root):
    from copy import deepcopy
    from avengine.timeline.v7_frozen_code import source_commit,verify_source
    source=Path(entry_root).resolve();task=task_by_slot(manifest,slot)
    expected=Path(task.get('snapshot_worktree',manifest['worktree'])).resolve()
    if source!=expected:raise ValueError('executing worker differs from its frozen task snapshot')
    actual=source_commit(source)
    if task.get('snapshot_commit',manifest['code_commit'])!=actual or not verify_source(source):
        raise ValueError('executing worker source does not match the admitted committed snapshot')
    effective=deepcopy(manifest);effective.update(worktree=str(source),code_commit=actual)
    return effective


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--worker',type=int,required=True)
    a=p.parse_args();m=bind_worker_snapshot(load(a.manifest),a.worker,Path(__file__).resolve().parents[2])
    if m.get('dataset_role')!='acceptance_only' or m.get('training_dataset_merge')is not False or m.get('CCI_transfer')is not False:
        raise ValueError('explicit isolated acceptance scope required')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic();cpu=time.process_time()
    try:code=run_worker(m,a.worker)
    except Exception as exc:
        write(Path(m['output_root'])/'evidence/jobs'/f'{a.worker:03d}.json',{'slot':a.worker,'status':'failed','scene_id':task_by_slot(m,a.worker)['row']['scene_id'],'house':task_by_slot(m,a.worker)['row']['house'],'room_id':task_by_slot(m,a.worker)['row']['room_id'],'pid':os.getpid(),'stage':'construction_preflight','reason':str(exc),'error_type':type(exc).__name__,'traceback':traceback.format_exc()})
        traceback.print_exc();code=2
    raise SystemExit(code)

if __name__=='__main__':main()
