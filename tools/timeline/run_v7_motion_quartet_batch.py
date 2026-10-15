#!/usr/bin/env python3
"""Construct new motion requests from standard inventories; no hand-authored scenes."""
from __future__ import annotations
import argparse,datetime,json,os,shutil,subprocess,sys,time,hashlib
from pathlib import Path
from collections import Counter
import psutil
from avengine.timeline.v7_storage_budget import launch_minimum_free_bytes

FAMILIES=('QA-06','QA-15','QA-16','QA-17')
def load(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def candidate_rooms(houses):
    eligible=[h for h in houses if h['split']=='train' and h['acoustic_status']=='pass']
    groups={family:[] for family in ('hm3d','mp3d')}
    for h in eligible:
        if h['family'] not in groups:continue
        rooms=sorted(h['checked_rooms'],key=lambda r:-r.get('floor_area_m2',0))
        if rooms:groups[h['family']].append((h,rooms))
    for group in groups.values():group.sort(key=lambda pair:(-max(r.get('floor_area_m2',0)for r in pair[1]),pair[0]['house']))
    ordered=[]
    while any(groups.values()):
        for family in ('hm3d','mp3d'):
            if groups[family]:ordered.append(groups[family].pop(0))
    return ordered

def asset_inventory_for_parent(catalog,visual):
    actual={a['asset_id']for frame in visual.get('frames',[])for a in frame['actors']}
    return [asset for asset in catalog if asset['asset_id']in actual]

def infrastructure_retries(manifest,out,commit,houses):
    """Replay automatic requests affected by an older implementation failure."""
    by_house={house['house']:house for house in houses}
    already={task['retry_of_slot'] for task in manifest['parents']
        if task.get('code_commit',manifest['code_commit'])==commit and 'retry_of_slot'in task}
    result=[]
    for task in manifest['parents']:
        if task['slot']in already or task.get('code_commit',manifest['code_commit'])==commit:continue
        receipt=Path(out)/'evidence/jobs'/f'{task["slot"]:03d}.json'
        if not receipt.exists():continue
        failed=load(receipt);reason=failed.get('reason','')
        if failed.get('status')!='failed' or not ('[Errno 17] File exists'in reason or 'drawable has no accepted native support'in reason):continue
        house=by_house.get(task['row']['house'])
        if house is None:continue
        room=next((room for room in house['checked_rooms']if room['room_label']==task['row']['room_id']),None)
        if room is not None:result.append((house,room,task.get('construction_attempt',0),task))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('contract','houses','assets','runtime','human-registry','output-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--walkable-map-root',type=Path,help='Shared precomputed static room maps; configured missing maps fail closed')
    p.add_argument('--resource-budget-root',type=Path)
    p.add_argument('--related-task-resource-root',type=Path,action='append',default=[],help='Own related output roots included in the same PSS ceiling; empty by default')
    p.add_argument('--worktree',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--minimum-houses',type=int,default=10)
    p.add_argument('--minimum-per-family',type=int,default=30)
    p.add_argument('--maximum-attempts',type=int,default=80)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--first-only',action='store_true')
    p.add_argument('--launch-deadline-utc')
    p.add_argument('--expand-after-first',action='store_true')
    p.add_argument('--house-families',nargs='+',choices=['hm3d','mp3d'],default=['hm3d','mp3d'])
    p.add_argument('--template-focuses',nargs='+',choices=['auto','M1','M2'],default=['auto'])
    p.add_argument('--source-types',nargs='+',default=['human_speech','human_other','dog_walk','cat_walk','animal_slide','human_footsteps'])
    a=p.parse_args()
    if not 1<=a.workers<=6:raise ValueError('parent workers must be <=6; shared native SDK queue reserves <=10')
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True);budget=(a.resource_budget_root or out).resolve()
    if not out.is_relative_to(budget):raise ValueError('output escaped the explicit own resource root')
    for folder in ('inputs/planned','evidence/jobs','logs','export/parent_rows'):(out/folder).mkdir(parents=True,exist_ok=True)
    policy_path=budget/'inputs/native_resource_policy.json'
    if not policy_path.exists():
        save(policy_path,{'rir':{'slots':10,'minimum_available_bytes':30_000_000_000},
            'parent':{'slots':6,'minimum_available_bytes':30_000_000_000},
            'motion_capture':{'slots':4,'minimum_available_bytes':30_000_000_000,'maximum_system_pss_bytes':190_000_000_000}})
    policy=load(policy_path)
    policy['template_visual']={'slots':a.workers,'minimum_available_bytes':30_000_000_000,
        'expected_incremental_bytes':0,'maximum_task_pss_bytes':40_000_000_000,'maximum_system_pss_bytes':190_000_000_000,
        'related_task_budget_roots':[str(root.resolve()) for root in a.related_task_resource_root]}
    save(policy_path,policy)
    manifest_path=out/'inputs/motion_quartet_batch.json'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.worktree,text=True).strip()
    if manifest_path.exists():manifest=load(manifest_path)
    else:
        manifest={'schema':'v7-automatic-motion-quartet-batch-v1','output_root':str(out),
            'resource_budget_root':str(budget),'worktree':str(a.worktree.resolve()),'code_commit':commit,
            'dataset_role':'acceptance_only','training_dataset_merge':False,'CCI_transfer':False,
            'contract':str(a.contract.resolve()),'runtime_cpu':str(a.runtime.resolve()),
            'houses_cpu':str(a.houses.resolve()),'assets_cpu':str(a.assets.resolve()),
            'human_cpu':str(a.human_registry.resolve()),'human_registered_cpu':str(a.human_registry.resolve()),
            'cpu_ids':sorted(os.sched_getaffinity(0))[:16],
            'disk_launch_policy':{'minimum_free_bytes':40_000_000_000,
                'expires_utc':'2026-10-15T10:30:00+00:00','after_expiry_minimum_free_bytes':100_000_000_000},
            'parents':[]}
        save(manifest_path,manifest)
    if a.walkable_map_root is not None:
        map_root=a.walkable_map_root.expanduser().resolve()
        if not map_root.is_dir():raise FileNotFoundError('walkable map cache root is missing: '+str(map_root))
        manifest['walkable_map_root']=str(map_root)
    manifest['default_new_parent_code_commit']=commit;save(manifest_path,manifest)
    houses=load(a.houses)['houses'];candidates=[item for item in candidate_rooms(houses) if item[0]['family'] in a.house_families]
    asset_inventory=load(a.assets);asset_inventory=asset_inventory.get('assets',asset_inventory) if isinstance(asset_inventory,dict)else asset_inventory
    pending=[]
    existing={(t['row']['house'],t['row']['room_id'],t.get('construction_attempt',0))for t in manifest['parents'] if (t.get('code_commit',manifest['code_commit'])==commit or (out/'evidence/jobs'/f'{t["slot"]:03d}.json').exists()and load(out/'evidence/jobs'/f'{t["slot"]:03d}.json')['status']=='passed')}
    for attempt in range(3):
        for house,rooms in candidates:
            room=rooms[min(attempt,len(rooms)-1)]
            if (house['house'],room['room_label'],attempt)in existing:continue
            pending.append((house,room,attempt,None))
    pending=infrastructure_retries(manifest,out,commit,houses)+pending
    active={};last_progress=0.;completed_at_start=sum((out/'evidence/jobs'/f'{t["slot"]:03d}.json').exists()for t in manifest['parents'])
    class AdoptedProcess:
        def __init__(self,pid,created):self.pid=pid;self.created=created;self.returncode=None
        def poll(self):
            try:
                p=psutil.Process(self.pid)
                if abs(p.create_time()-self.created)<.01 and p.status()!=psutil.STATUS_ZOMBIE:return None
            except psutil.NoSuchProcess:pass
            self.returncode=0;return 0
    prior_active=load(out/'ACTIVE_JOBS.json')if(out/'ACTIVE_JOBS.json').exists()else[]
    for entry in prior_active:
        adopted=AdoptedProcess(entry['pid'],entry['create_time'])
        if adopted.poll()is None:active[entry['slot']]={**{k:v for k,v in entry.items()if k!='slot'},'process':adopted,'adopted':True}
    def report():
        totals=Counter();passed=[];failures=Counter();rows=[];attempted=0;native_attempted=0;typed={}
        for task in manifest['parents']:
            path=out/'evidence/jobs'/f'{task["slot"]:03d}.json'
            receipt=load(path)if path.exists()else{'status':'not_started'}
            row={'slot':task['slot'],'house':task['row']['house'],'room_id':task['row']['room_id'],
                 'constructed_source_type':task['row'].get('motion_source_type','human_speech'),
                 'status':receipt['status'],'receipt':str(path),'reason':receipt.get('reason')}
            if receipt['status']in {'passed','failed'}:
                attempted+=1
                if receipt.get('started_utc'):native_attempted+=1
            if receipt['status']=='passed':
                from avengine.timeline.v7_spatial_layout import native_person_camera_clearance
                parent=out/'native/parents'/task['row']['scene_id'];pointer=parent/'visible_native_completed.json'
                camera_report={'status':'fail','reason':'completed native camera evidence missing'}
                if pointer.exists():
                    native_path=parent/load(pointer)['attempt']/'native_visible_visual_evidence_phase2.json'
                    if native_path.exists():camera_report=native_person_camera_clearance(load(native_path),asset_inventory_for_parent(asset_inventory,load(native_path)))
                row['person_camera_clearance']=camera_report
                if camera_report['status']!='pass':
                    receipt={**receipt,'status':'invalidated','reason':'native person/camera clearance below 1.2m or missing readback'}
                    row.update(status=receipt['status'],reason=receipt['reason'])
            if receipt['status']=='passed':
                exported=receipt.get('export',{});counts=exported.get('question_family_counts',{})
                totals.update({f:counts.get(f,0)for f in FAMILIES});row['family_counts']=counts
                row['independent_questions']=receipt.get('independent_questions',0);passed.append(task)
                from avengine.timeline.v7_motion_type_counts import question_type_counts
                type_receipt=out/'evidence'/f'motion_source_type_counts_{task["slot"]:03d}.json'
                if not type_receipt.exists():save(type_receipt,question_type_counts(out/'export',out/'export/parent_rows'/f'{task["slot"]:03d}.jsonl'))
                counted=load(type_receipt);row['question_source_type_counts']=counted
                for kind,family_counts in counted['counts_by_actual_question_target_source_type'].items():typed.setdefault(kind,Counter()).update(family_counts)
            elif receipt['status']in {'failed','invalidated'}:failures[(receipt.get('reason')or'unknown').split(':')[0]]+=1
            row['code_commit']=receipt.get('code_commit',task.get('code_commit',manifest['code_commit']))
            motion_parent=out/'native/parents'/task['row']['scene_id']
            row['native_route_trials']=[{'source':str(p),**load(p)}for p in motion_parent.glob('**/motion_route_attempt_summary.json')]
            rows.append(row)
        eligible=out/'export/eligible_train.jsonl';temporary=eligible.with_suffix('.tmp')
        with temporary.open('w')as stream:
            for task in passed:
                source=out/'export/parent_rows'/f'{task["slot"]:03d}.jsonl'
                if source.exists():stream.write(source.read_text())
        temporary.replace(eligible)
        distinct={t['row']['house']for t in passed};families={t['family']for t in passed}
        meets=len(distinct)>=a.minimum_houses and {'hm3d','mp3d'}<=families and all(totals[f]>=a.minimum_per_family for f in FAMILIES)
        data={'time_utc':now(),'status':'complete'if meets else'running','code_commit':commit,'execution_commits':sorted({r['code_commit']for r in rows}),
              'eligible_export_path':str(eligible),'independent_motion_family_counts':dict(totals),'distinct_passed_houses':len(distinct),
              'house_families':sorted(families),'construction_attempts_completed':attempted,'native_parent_attempts_completed':native_attempted,
              'counts_by_actual_question_target_source_type':{kind:dict(c)for kind,c in typed.items()},
              'native_parent_passes':len(passed),'native_pass_rate':len(passed)/native_attempted if native_attempted else None,
              'failure_categories':dict(failures),'parents':rows}
        save(out/'MOTION_QUARTET_RESULTS.json',data);return data
    while True:
        result=report()
        for slot,job in list(active.items()):
            if job['process'].poll()is not None:
                del active[slot]
                receipt_path=out/'evidence/jobs'/f'{slot:03d}.json'
                if not receipt_path.exists():
                    log_text=Path(job['log']).read_text(errors='replace')
                    receipt={'status':'failed','reason':'construction_preflight: '+log_text.splitlines()[-1] if log_text.splitlines() else 'construction_preflight: no receipt','stage':'construction','log':job['log'],'exit_code':job['process'].returncode}
                    save(receipt_path,receipt)
                receipt=load(receipt_path)
                print(now(),'FINISHED',slot,receipt['status'],receipt.get('reason'),flush=True)
                for audit in (out/'native/parents'/manifest['parents'][slot]['row']['scene_id']).glob('**/candidate_audit.json'):
                    data=load(audit);qa=data.get('v6_generation',{}).get('unmet_quota_by_qa',{})
                    save(out/'evidence'/f'candidate_audit_{slot:03d}.json',{'source':str(audit),'v6_generation.unmet_quota_by_qa':qa})
                    missing={f:qa.get(f)for f in FAMILIES if not receipt.get('export',{}).get('question_family_counts',{}).get(f,0)}
                    if missing:
                        with(budget/'PROGRESS.md').open('a')as stream:stream.write(now()+' 原始运动候选拒因 scene='+manifest['parents'][slot]['row']['scene_id']+' source='+str(audit)+' '+json.dumps(missing,ensure_ascii=False)+'\n')
                if a.first_only and receipt['status']=='passed' and receipt.get('export',{}).get('question_family_counts',{}).get('QA-15',0)>0:
                    result=report();print('FIRST_HUMAN_QA15_EXPORT_PASS',json.dumps(result,ensure_ascii=False),flush=True)
                    if a.expand_after_first:a.first_only=False
                    else:return
        if result['status']=='complete' and not active:return
        completed=sum(r['status']in {'passed','failed'}for r in result['parents'])
        while pending and len(active)<(1+sum(bool(j.get('adopted'))for j in active.values())if a.first_only else a.workers) and len(manifest['parents'])<a.maximum_attempts and result['status']!='complete':
            if a.launch_deadline_utc and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(a.launch_deadline_utc.replace('Z','+00:00')):pending.clear();break
            if psutil.virtual_memory().available<30_000_000_000 or shutil.disk_usage('/data').free<launch_minimum_free_bytes(manifest['disk_launch_policy']):break
            house,room,attempt,retry_task=pending.pop(0);slot=len(manifest['parents'])
            launch_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.worktree,text=True).strip()
            seed=retry_task['row']['scene_seed'] if retry_task else a.seed+slot*101+attempt
            identity=hashlib.sha256((house['house']+room['room_label']+str(seed)+(str(retry_task['slot'])+launch_commit if retry_task else '')).encode()).hexdigest()[:14]
            row={'schema':'v7-visible-registered-pilot-phase2-v1','split':{'development':'dev'}.get(house['split'],house['split']),
                 'house':house['house'],'room_id':room['room_label'],'scene_seed':seed,
                 'motion_source_type':(retry_task['row'].get('motion_source_type','human_speech') if retry_task else 'human_speech'if a.first_only else a.source_types[slot%len(a.source_types)]),
                 'scene_id':'motion_quartet_'+identity,'question_quota_base':10,'questions_per_window':14,
                 'window_count':5,'members':['v0_a0'],'trial_only':True}
            if a.template_focuses[slot%len(a.template_focuses)]!='auto':row['motion_template_focus']=a.template_focuses[slot%len(a.template_focuses)]
            task={'slot':slot,'row':row,'device':'cpu','profile':'phase2','family':house['family'],
                  'constructor':'motion_quartet_template','construction_attempt':attempt,'code_commit':launch_commit}
            if retry_task is not None:task['retry_of_slot']=retry_task['slot']
            manifest['parents'].append(task);save(manifest_path,manifest)
            logfile=out/'logs'/f'parent_{slot:03d}.log';stream=logfile.open('a')
            command=[sys.executable,str(a.worktree/'tools/timeline/run_v7_frozen_acceptance.py'),
                     '--manifest',str(manifest_path),'--worker',str(slot)]
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);stream.close()
            active[slot]={'process':process,'pid':process.pid,'create_time':psutil.Process(process.pid).create_time(),
                          'command':command,'log':str(logfile)}
            print(now(),'STARTED',slot,process.pid,house['house'],room['room_label'],flush=True)
        save(out/'ACTIVE_JOBS.json',[{'slot':k,**{x:y for x,y in j.items()if x!='process'}}for k,j in active.items()])
        if time.monotonic()-last_progress>=900:
            line=now()+' motion_quartet_template 自动批次：'+json.dumps({k:v for k,v in result.items()if k!='parents'},ensure_ascii=False)
            print(line,flush=True)
            with(budget/'PROGRESS.md').open('a')as stream:stream.write(line+'\n')
            last_progress=time.monotonic()
        if not active and (not pending or len(manifest['parents'])>=a.maximum_attempts):return
        time.sleep(10)

if __name__=='__main__':main()
