"""Move only verified owned single-thread parents away from busy shared cores."""
from pathlib import Path
import json,os,time,datetime

def choose_cpu(current,usage,competitors,allowed,reserved=()):
    if current not in competitors or usage.get(current,0)<.8:return None
    candidates=[i for i in allowed if i not in reserved and i not in competitors and usage.get(i,1)<=.2]
    return min(candidates,key=lambda i:(usage[i],i))if candidates else None


def rebalance_owned_parents(manifest,manifest_path,interval_s=.3):
    import psutil
    root=Path(manifest['output_root']).resolve();allowed=set(manifest.get('cpu_ids')or os.sched_getaffinity(0))
    first=psutil.cpu_times(percpu=True);time.sleep(interval_s);last=psutil.cpu_times(percpu=True)
    usage={}
    for i,(a,b)in enumerate(zip(first,last)):
        total=sum(b[:8])-sum(a[:8]);idle=b.idle-a.idle+b.iowait-a.iowait
        usage[i]=max(0.,min(1.,(total-idle)/total))if total else 0.
    jobs=[]
    for path in(root/'evidence/jobs').glob('[0-9][0-9][0-9].json'):
        try:
            job=json.loads(path.read_text());pr=psutil.Process(job['pid']);args=pr.cmdline()
            if job.get('status')!='running'or pr.uids().real!=os.getuid()or pr.nice()<10:continue
            if not any(Path(a).name=='run_v7_acceptance.py'for a in args):continue
            if Path(args[args.index('--manifest')+1]).resolve()!=Path(manifest_path).resolve():continue
            if int(args[args.index('--worker')+1])!=job['slot']:continue
            if len(pr.cpu_affinity())==1:jobs.append((pr,job,pr.create_time()))
        except (psutil.Error,OSError,KeyError,ValueError):continue
    own={p.pid for p,_,_ in jobs};competitors=set();reserved={p.cpu_affinity()[0]for p,_,_ in jobs}
    for pr in psutil.process_iter():
        try:
            affinity=pr.cpu_affinity()
            if pr.pid not in own and pr.status()==psutil.STATUS_RUNNING and len(affinity)==1:competitors.add(affinity[0])
        except psutil.Error:continue
    changed=[]
    for pr,job,created in jobs:
        before=pr.cpu_affinity();target=choose_cpu(before[0],usage,competitors,allowed,reserved)
        if target is None:continue
        try:
            if not pr.is_running()or pr.create_time()!=created:continue
            pr.cpu_affinity([target]);reserved.add(target)
            item={'pid':pr.pid,'create_time':created,'scene_id':job['scene_id'],'old_cpu':before[0],'new_cpu':target,'nice':pr.nice(),'new_cpu_thread_limit':1,'signals_sent':False,'foreign_processes_modified':False}
            changed.append(item)
        except psutil.Error:continue
    if changed:
        text=datetime.datetime.now(datetime.timezone.utc).isoformat()+' owned_cpu_rebalance '+json.dumps(changed)+'\n'
        for path in(root/'WATCH_LOG.md',root/'PROGRESS.md'):
            with path.open('a')as f:f.write(text)
    return changed
