#!/usr/bin/env python3
"""Run isolated native acceptance parents under a committed resource policy."""
from __future__ import annotations
import argparse,datetime,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path


def load(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def run_worker(manifest,slot):
    task=manifest['parents'][slot];row=task['row'];root=Path(manifest['output_root']).resolve()
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice >=10 required')
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[slot%len(cpus)]})
    import avengine
    if not Path(avengine.__file__).resolve().is_relative_to(Path(manifest['worktree']).resolve()/'src'):
        raise ValueError('import escaped the acceptance worktree')
    from avengine.timeline.v7_storage_budget import ENV_KEY,check_write,task_disk_policy
    os.environ[ENV_KEY]=json.dumps({'root':str(root),'filesystem':'/data','maximum_task_bytes':60_000_000_000,
        'minimum_free_bytes':task_disk_policy(root,100_000_000_000)['minimum_free_bytes'],'concurrent_write_margin_bytes':128_000_000,
        'policy':'owner-20261015-acceptance-only-no-training-or-CCI'})
    check_write(root,expected_bytes=0)
    start=time.monotonic();cpu=time.process_time()
    initial={'slot':slot,'device':task['device'],'house':row['house'],'room_id':row['room_id'],'scene_id':row['scene_id'],'pid':os.getpid(),'status':'running','phase':'automatic construction / native input preflight','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=manifest['worktree'],text=True).strip()}
    write(root/'evidence/jobs'/f'{slot:03d}.json',initial)
    inputs=load(task.get('runtime_inputs') or manifest['runtime_'+task['device']]);inputs['production_input_contract']=manifest['contract']
    catalog=load(manifest['assets_'+task['device']]);catalog=catalog['assets']if isinstance(catalog,dict)else catalog
    houses=load(manifest['houses_'+task['device']])['houses'];house=next(h for h in houses if h['house']==row['house'])
    contract=load(manifest['contract']);source=load(contract['inputs']['source_admission']['path']);human=load(manifest['human_registered_cpu'] if task['profile']=='registered'and task['device']=='cpu'else manifest['human_'+task['device']])
    if 'ue_physical_gpu'in task:
        from copy import deepcopy
        gpu=int(task['ue_physical_gpu']);house=deepcopy(house)
        house['input']['ue_config'].update(physical_gpu_index=gpu,graphics_adapter=0,unreal_editor=str(root/'scripts'/f'ue_gpu{gpu}'))
        os.sched_setaffinity(0,{12+slot%4})
    if task.get('automatic_registered_room'):
        from avengine.capture.kujiale_registered_room_floor import bind_room_floor
        house=bind_room_floor(house,row['room_id'],contract=contract)
    from avengine.timeline.v7_demand import read_demand_state,plan_phase2_demand,add_scene_demand,record_selected_parent
    if task.get('automatic_registered_room') and row.get('required_sound_classes'):
        from avengine.timeline.v7_kujiale_environment_construction import bind_inventory_project
        inputs=bind_inventory_project(inputs,house,root)
    if task.get('automatic_registered_room') and not (root/'inputs/planned'/f'{slot:03d}.json').exists():
        from avengine.timeline.v7_kujiale_construction import construct_request
        from avengine.registry.kujiale_ue_measurements import read_registration
        registration,_=read_registration(inputs.get('kujiale_native_measurement_registration'))
        from avengine.timeline.v7_kujiale_cpu_phases import cpu_phase
        with cpu_phase(root,'construct_room',estimate_bytes=4000000000):
            row=construct_request(row,house,inputs,catalog,load(inputs['semantic_manifest']),registration)
    state=read_demand_state(root/'inputs/demand_state.json')
    if task['profile']=='phase2':
        previous=root/'inputs/planned'/f'{slot:03d}.json'
        row=load(previous) if previous.exists() else plan_phase2_demand(row,state,catalog,source['records'],contract,focus=task.get('focus'))
        inputs.update(phase2=True,integration_pilot=True,render_backend='cpu_llvmpipe')
        from avengine.timeline.v7_phase2_fallback import generate_parent_with_element_fallback as generate
    else:
        previous=root/'inputs/planned'/f'{slot:03d}.json'
        row=load(previous) if previous.exists() else add_scene_demand(row,state,focus=task.get('focus','content_binding'))
        row['window_count']=5;row['window_start_ticks']=[0,50,100,150,200]
        conditions=row['scene_demand']['requested_conditions']
        if 'authored_native_quartet' not in conditions:conditions.append('authored_native_quartet')
        inputs['render_backend']='ue_spear' if task['device']=='ue' else 'cpu_llvmpipe'
        from avengine.timeline.v7_registered_generation import generate_registered_parent as generate
    write(root/'inputs/planned'/f'{slot:03d}.json',row)
    receipt={'slot':slot,'device':task['device'],'house':row['house'],'room_id':row['room_id'],
        'scene_id':row['scene_id'],'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'code_commit':initial['code_commit'],'planned_code_commit':manifest['code_commit'],'status':'running','started_utc':initial['started_utc']}
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    from contextlib import ExitStack
    phases=ExitStack()
    try:
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
        else:result=generate(row,house,inputs,catalog,source,human,str(native_root))
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
        feedback={**result,'windows':[{'family_counts':exported['question_family_counts'],'logical_questions':exported['independent_questions']}],'raw_logical_questions':exported['raw_export_questions'],'dedup_dropped':exported['dedup_dropped']}
        record_selected_parent(root/'inputs/demand_state.json',feedback)
        receipt.update(status='passed',producer_result=result,export=exported,reader=reader,
                       independent_questions=exported['independent_questions'],raw_questions=exported['raw_export_questions'])
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    finally:phases.close()
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
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic();cpu=time.process_time()
    try:code=run_worker(m,a.worker)
    except Exception as exc:
        path=Path(m['output_root'])/'evidence/jobs'/f'{a.worker:03d}.json'
        receipt=load(path)if path.exists()else {'slot':a.worker,'scene_id':m['parents'][a.worker]['row']['scene_id'],'pid':os.getpid(),'started_utc':started}
        # Construction can fail before the normal generation try/except.
        # Retain its original PID/code/start provenance instead of erasing it.
        receipt.update(status='failed',reason=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(path,receipt)
        traceback.print_exc();code=2
    raise SystemExit(code)

if __name__=='__main__':main()
