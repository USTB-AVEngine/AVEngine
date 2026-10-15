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

def candidate_rooms(houses,contract=None,excluded_houses=()):
    eligible=[h for h in houses if h['split']=='train' and h['acoustic_status']=='pass' and h['house']not in excluded_houses and h['house']not in (contract or {}).get('test_houses',[])]
    legal=None if contract is None else {(r['house'],r['room_id'])for r in contract['rooms']if 'standing_human'in r['categories']}
    groups={family:[] for family in ('hm3d','mp3d')}
    for h in eligible:
        if h['family'] not in groups:continue
        rooms=sorted((r for r in h['checked_rooms']if legal is None or (h['house'],r['room_label'])in legal),key=lambda r:-r.get('floor_area_m2',0))
        if rooms:groups[h['family']].append((h,rooms))
    for group in groups.values():group.sort(key=lambda pair:(-max(r.get('floor_area_m2',0)for r in pair[1]),pair[0]['house']))
    ordered=[]
    while any(groups.values()):
        for family in ('hm3d','mp3d'):
            if groups[family]:ordered.append(groups[family].pop(0))
    return ordered

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('contract','houses','assets','runtime','human-registry','output-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resource-budget-root',type=Path)
    p.add_argument('--worktree',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--minimum-houses',type=int,default=10)
    p.add_argument('--minimum-per-family',type=int,default=30)
    p.add_argument('--maximum-attempts',type=int,default=80)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--first-only',action='store_true')
    p.add_argument('--admission-manifest',type=Path)
    p.add_argument('--exclude-evidence-roots',type=Path,nargs='*',default=[])
    p.add_argument('--launch-until-utc')
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
    policy['template_visual']={'slots':min(2,a.workers),'minimum_available_bytes':30_000_000_000,
        'expected_incremental_bytes':16_000_000_000,'maximum_system_pss_bytes':190_000_000_000}
    save(policy_path,policy)
    manifest_path=out/'inputs/motion_quartet_batch.json'
    from avengine.timeline.v7_frozen_code import source_commit,verify_source
    if not verify_source(a.worktree):raise ValueError('clean committed source snapshot required')
    commit=source_commit(a.worktree)
    admission_policy=load(a.admission_manifest)if a.admission_manifest else {}
    if manifest_path.exists():manifest=load(manifest_path)
    else:
        manifest={'schema':'v7-automatic-motion-quartet-batch-v1','output_root':str(out),
            'resource_budget_root':str(budget),'worktree':str(a.worktree.resolve()),'code_commit':commit,
            'dataset_role':'acceptance_only','training_dataset_merge':False,'CCI_transfer':False,
            'contract':str(a.contract.resolve()),'runtime_cpu':str(a.runtime.resolve()),
            'houses_cpu':str(a.houses.resolve()),'assets_cpu':str(a.assets.resolve()),
            'human_cpu':str(a.human_registry.resolve()),'human_registered_cpu':str(a.human_registry.resolve()),
            'cpu_ids':sorted(os.sched_getaffinity(0))[:16],
            'launch_disk_policy':admission_policy.get('launch_disk_policy',{}),
            'parents':[]}
        save(manifest_path,manifest)
    feedback=out/'inputs/demand_state.json'
    if not feedback.exists() and a.admission_manifest:
        baseline=Path(admission_policy['output_root'])/'inputs/demand_state.json'
        if baseline.is_file():save(feedback,load(baseline))
    houses=load(a.houses)['houses']
    from avengine.timeline.v7_automatic_feedback import from_passed_exports
    excluded=from_passed_exports(a.exclude_evidence_roots).get('observed_houses',[])if a.exclude_evidence_roots else []
    candidates=candidate_rooms(houses,load(a.contract),excluded)
    if not candidates:raise ValueError('no untouched legal training house admits the requested humans')
    pending=[]
    existing={(t['row']['house'],t['row']['room_id'],t.get('construction_attempt',0))for t in manifest['parents']}
    for attempt in range(3):
        for house,rooms in candidates:
            room=rooms[min(attempt,len(rooms)-1)]
            if (house['house'],room['room_label'],attempt)in existing:continue
            pending.append((house,room,attempt))
    active={};last_progress=0.;completed_at_start=sum((out/'evidence/jobs'/f'{t["slot"]:03d}.json').exists()for t in manifest['parents'])
    def report():
        totals=Counter();passed=[];failures=Counter();rows=[];attempted=0
        for task in manifest['parents']:
            path=out/'evidence/jobs'/f'{task["slot"]:03d}.json'
            receipt=load(path)if path.exists()else{'status':'not_started'}
            row={'slot':task['slot'],'house':task['row']['house'],'room_id':task['row']['room_id'],
                 'status':receipt['status'],'receipt':str(path),'reason':receipt.get('reason')}
            if receipt['status']in {'passed','failed'}:attempted+=1
            if receipt['status']=='passed':
                exported=receipt.get('export',{});counts=exported.get('question_family_counts',{})
                totals.update({f:counts.get(f,0)for f in FAMILIES});row['family_counts']=counts
                row['independent_questions']=receipt.get('independent_questions',0);passed.append(task)
            elif receipt['status']=='failed':failures[(receipt.get('reason')or'unknown').split(':')[0]]+=1
            rows.append(row)
        distinct={t['row']['house']for t in passed};families={t['family']for t in passed}
        meets=len(distinct)>=a.minimum_houses and {'hm3d','mp3d'}<=families and all(totals[f]>=a.minimum_per_family for f in FAMILIES)
        data={'time_utc':now(),'status':'complete'if meets else'running','code_commit':manifest['code_commit'],
              'independent_motion_family_counts':dict(totals),'distinct_passed_houses':len(distinct),
              'house_families':sorted(families),'native_parent_attempts_completed':attempted,
              'native_parent_passes':len(passed),'native_pass_rate':len(passed)/attempted if attempted else None,
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
                reason=receipt.get('reason','')
                if receipt.get('error_type')in {'ImportError','ModuleNotFoundError','SyntaxError','KeyError'}or reason.startswith('Installed Habitat/RLR runtime is unavailable'):
                    pending=[]
                    save(out/'INFRASTRUCTURE_FAILURE.json',{'status':'pause new dispatch until code fix','slot':slot,'reason':reason,'active_native_jobs_not_signalled':True})
                for audit in (out/'native/parents'/manifest['parents'][slot]['row']['scene_id']).glob('**/candidate_audit.json'):
                    data=load(audit);qa=data.get('v6_generation',{}).get('unmet_quota_by_qa',{})
                    save(out/'evidence'/f'candidate_audit_{slot:03d}.json',{'source':str(audit),'v6_generation.unmet_quota_by_qa':qa})
                if a.first_only and receipt['status']=='passed' and all(receipt.get('export',{}).get('question_family_counts',{}).get(f,0)>0 for f in FAMILIES):
                    result=report();print(json.dumps(result,ensure_ascii=False),flush=True);return
        if result['status']=='complete' and not active:return
        completed=sum(r['status']in {'passed','failed'}for r in result['parents'])
        smoke_ready=any(r['status']=='passed' and all(r.get('family_counts',{}).get(f,0)>0 for f in FAMILIES)for r in result['parents'])
        while pending and len(active)<(1 if a.first_only or not smoke_ready else a.workers) and len(manifest['parents'])<a.maximum_attempts and result['status']!='complete':
            if a.launch_until_utc and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(a.launch_until_utc):
                pending=[];break
            if psutil.virtual_memory().available<30_000_000_000 or shutil.disk_usage('/data').free<launch_minimum_free_bytes(manifest.get('launch_disk_policy')):break
            from avengine.timeline.v7_storage_budget import allocated_bytes
            if allocated_bytes(budget)+1_000_000_000>60_000_000_000:break
            if a.admission_manifest:
                from avengine.timeline.v7_live_resource_guard import observe
                limits=observe(load(a.admission_manifest))
                if limits['pause_new_parents'] or limits['own_pss_bytes']+30_000_000_000>admission_policy.get('maximum_pss_bytes',150_000_000_000) or limits['combined_pss_bytes']+30_000_000_000>admission_policy.get('maximum_combined_pss_bytes',190_000_000_000):break
            house,room,attempt=pending.pop(0);slot=len(manifest['parents'])
            seed=int.from_bytes(hashlib.sha256((str(a.seed)+house['house']+room['room_label']).encode()).digest()[:6],'big');identity=hashlib.sha256((house['house']+room['room_label']+str(seed)).encode()).hexdigest()[:14]
            if any(t['row']['house']==house['house'] and (out/'evidence/jobs'/f"{t['slot']:03d}.json").exists() and load(out/'evidence/jobs'/f"{t['slot']:03d}.json").get('status')=='passed'for t in manifest['parents']):continue
            row={'schema':'v7-visible-registered-pilot-phase2-v1','split':{'development':'dev'}.get(house['split'],house['split']),
                 'house':house['house'],'room_id':room['room_label'],'scene_seed':seed,
                 'scene_id':'motion_quartet_'+identity+'__attempt_'+str(attempt),'question_quota_base':10,'questions_per_window':14,
                 'window_count':5,'members':['v0_a0'],'trial_only':True}
            task={'slot':slot,'row':row,'device':'cpu','profile':'phase2','family':house['family'],
                  'constructor':'motion_quartet_template','focus':'motion_quartet_template','construction_attempt':attempt}
            manifest['parents'].append(task);save(manifest_path,manifest)
            logfile=out/'logs'/f'parent_{slot:03d}.log';stream=logfile.open('a')
            command=[sys.executable,str(a.worktree/'tools/timeline/run_v7_acceptance.py'),
                     '--manifest',str(manifest_path),'--worker',str(slot)]
            env=dict(os.environ);env['PYTHONPATH']=str(a.worktree/'src')+os.pathsep+str(a.worktree)+os.pathsep+env.get('PYTHONPATH','')
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);stream.close()
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
