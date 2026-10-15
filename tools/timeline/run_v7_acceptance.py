#!/usr/bin/env python3
"""Run isolated native acceptance parents under a committed resource policy."""
from __future__ import annotations
import argparse,datetime,json,os,resource,sys,time,traceback
from pathlib import Path


def load(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def configure_walkable_maps(manifest):
    """One map cache is shared by construction and native admission workers."""
    value=manifest.get('walkable_map_root')
    if value is None:return
    root=Path(value).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError('configured walkable map cache root is missing: '+str(root))
    os.environ['AVENGINE_WALKABLE_MAP_ROOT']=str(root)
    os.environ['AVENGINE_REQUIRE_WALKABLE_MAP']='1'


def run_worker(manifest,slot):
    configure_walkable_maps(manifest)
    task=manifest['parents'][slot];row=task['row'];root=Path(manifest['output_root']).resolve()
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice >=10 required')
    cpus=manifest.get('cpu_ids') or sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpus[slot%len(cpus)]})
    import avengine
    if not Path(avengine.__file__).resolve().is_relative_to(Path(manifest['worktree']).resolve()/'src'):
        raise ValueError('import escaped the acceptance worktree')
    from avengine.timeline.v7_storage_budget import ENV_KEY,check_write,launch_minimum_free_bytes
    budget_root=Path(manifest.get('resource_budget_root',str(root))).resolve()
    os.environ[ENV_KEY]=json.dumps({'root':str(budget_root),'filesystem':'/data','maximum_task_bytes':60_000_000_000,
        'minimum_free_bytes':launch_minimum_free_bytes(manifest.get('disk_launch_policy')),'concurrent_write_margin_bytes':128_000_000,
        'policy':'owner-20261015-acceptance-only-no-training-or-CCI'})
    check_write(root,expected_bytes=0)
    # Owner 2026-10-15: resource thresholds pause launches. A running parent
    # must be allowed to finish its writes after passing the launch check.
    active_storage=json.loads(os.environ[ENV_KEY]);active_storage['minimum_free_bytes']=0
    os.environ[ENV_KEY]=json.dumps(active_storage)
    inputs=load(task.get('runtime_input') or manifest['runtime_'+task['device']]);inputs['production_input_contract']=manifest['contract']
    catalog=load(manifest['assets_'+task['device']]);catalog=catalog['assets']if isinstance(catalog,dict)else catalog
    houses=load(manifest['houses_'+task['device']])['houses'];house=next(h for h in houses if h['house']==row['house'])
    contract=load(manifest['contract']);source=load(contract['inputs']['source_admission']['path']);human=load(manifest['human_registered_cpu'] if task['profile']=='registered'and task['device']=='cpu'else manifest['human_'+task['device']])
    from avengine.timeline.v7_demand import read_demand_state,plan_phase2_demand,add_scene_demand,record_selected_parent
    state=read_demand_state(root/'inputs/demand_state.json')
    if task['profile']=='phase2':
        previous=root/'inputs/planned'/f'{slot:03d}.json'
        if previous.exists():row=load(previous)
        elif task.get('constructor')=='motion_quartet_template':
            from avengine.timeline.v7_motion_quartet_template import motion_quartet_template
            from avengine.acoustics.runtime import load_habitat_runtime
            hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
            pf=hs.PathFinder()
            if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('registered native NavMesh is missing')
            room=next(r for r in house['checked_rooms'] if r['room_label']==row['room_id'])
            row=motion_quartet_template(row,room,catalog,load(inputs['semantic_manifest']),source['records'],pf,
                attempt=int(task.get('construction_attempt',0)),contract=contract)
        else:
            def construction_context():
                from avengine.acoustics.runtime import load_habitat_runtime
                hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime']);pf=hs.PathFinder()
                if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('registered native NavMesh missing')
                return {'room':next(r for r in house['checked_rooms']if r['room_label']==row['room_id']),'semantic_manifest':load(inputs['semantic_manifest']),'pathfinder':pf}
            row=plan_phase2_demand(row,state,catalog,source['records'],contract,focus=task.get('focus'),construction_context=construction_context)
        inputs.update(phase2=True,integration_pilot=True,render_backend=inputs.get('render_backend','cpu_llvmpipe'))
        from avengine.timeline.v7_phase2_generation import generate_phase2_parent as generate
    else:
        previous=root/'inputs/planned'/f'{slot:03d}.json'
        row=load(previous) if previous.exists() else add_scene_demand(row,state,focus=task.get('focus','content_binding'))
        row['window_count']=5;row['window_start_ticks']=[0,50,100,150,200]
        row['scene_demand']['requested_conditions'].append('authored_native_quartet')
        inputs['render_backend']='ue_spear' if task['device']=='ue' else 'cpu_llvmpipe'
        from avengine.timeline.v7_registered_generation import generate_registered_parent as generate
    if task['profile']=='phase2':
        from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion,REVISION as EXECUTION_REVISION
        if not previous.exists():row['motion_execution_revision']=EXECUTION_REVISION
        row=prepare_scene_motion(row,catalog)
        from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
        row=prepare_motion_during_row(row,catalog,source['records'])
    write(root/'inputs/planned'/f'{slot:03d}.json',row)
    start=time.monotonic();cpu=time.process_time()
    receipt={'slot':slot,'device':task['device'],'house':row['house'],'room_id':row['room_id'],
        'scene_id':row['scene_id'],'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'code_commit':task.get('code_commit',manifest['code_commit']),'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    try:
        from avengine.timeline.v7_native_resource_slots import acquire_native_resource
        parent_execution_lease=acquire_native_resource('parent')
        result=generate(row,house,inputs,catalog,source,human,str(root/'native'))
        admitted=result['status']in{'integration_pilot_admitted','formal_alpha_admitted'}
        if not admitted:raise ValueError('native producer did not admit the parent')
        parent=Path(result['delivery_directory']).parent
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
        feedback={**result,'motion_source_type':row.get('motion_source_type'),'windows':[{'family_counts':exported['question_family_counts'],'logical_questions':exported['independent_questions']}],'raw_logical_questions':exported['raw_export_questions'],'dedup_dropped':exported['dedup_dropped']}
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
        write(Path(m['output_root'])/'evidence/jobs'/f'{a.worker:03d}.json',{'slot':a.worker,'status':'failed','scene_id':m['parents'][a.worker]['row']['scene_id'],'reason':str(exc),'error_type':type(exc).__name__,'traceback':traceback.format_exc()})
        traceback.print_exc();code=2
    raise SystemExit(code)

if __name__=='__main__':main()
