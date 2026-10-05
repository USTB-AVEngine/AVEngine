"""Read current-job training logs and control only this v7 coordinator.

The first 60 post-warmup samples are a documented same-job reference, not
claimed as a separately timed idle benchmark. No training files are written.
"""
import argparse,json,os,socket,statistics,time,uuid
from pathlib import Path

def atomic(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def rows(text):
    out=[]
    for line in text.splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r.get('elapsed_ms'),(int,float)) and r['elapsed_ms']>0:out.append(r)
    return out

def groups(records):
    data={}
    for row in records:data.setdefault(row.get('presentation','unknown'),[]).append(row['elapsed_ms'])
    return {k:{'n':len(v),'median_ms':statistics.median(v)} for k,v in data.items() if len(v)>=8}

def compare(baseline,current):
    out=[]
    for name,b in baseline.items():
        c=current.get(name)
        if not c or c['sample_step']==b['sample_step']:continue
        for modality,base in b['groups'].items():
            if modality in c['groups']:
                out.append({'path':name,'presentation':modality,
                    'throughput_drop':1-base['median_ms']/c['groups'][modality]['median_ms'],
                    'reference_median_ms':base['median_ms'],
                    'current_median_ms':c['groups'][modality]['median_ms']})
    return sorted(out,key=lambda x:x['throughput_drop'],reverse=True)

def read_samples(paths):
    baseline={};current={}
    for path in paths:
        p=Path(path)
        if not p.is_file() or time.time()-p.stat().st_mtime>300:continue
        with p.open('rb')as f:
            first=rows(f.read(200000).decode(errors='replace'))
            f.seek(max(0,p.stat().st_size-200000));last=rows(f.read().decode(errors='replace'))
        if len(first)<63 or len(last)<60:continue
        reference=first[3:63];tail=last[-60:]
        baseline[str(p)]={'groups':groups(reference),'sample_step':reference[-1].get('sample_step')}
        current[str(p)]={'groups':groups(tail),'sample_step':tail[-1].get('sample_step'),'age_seconds':time.time()-p.stat().st_mtime}
    return baseline,current

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-path-receipt',type=Path,required=True)
    parser.add_argument('--expected-job',required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=float,default=120.)
    a=parser.parse_args();root=a.root;guard_id='v7-r7-training-35pct-'+a.expected_job
    receipt=json.loads(a.training_path_receipt.read_text());expected=receipt['P19_owner']
    if expected['job_id']!=a.expected_job:raise ValueError('receipt job mismatch')
    paths=receipt['actual_current_training_paths'];allowed=root.parent.parent/'runs'
    for p in paths:
        if not Path(p).resolve().is_relative_to(allowed.resolve()) or a.expected_job not in p:raise ValueError('training path not from this P19 job')
    applied_steps=None;limit=40;log=root/'cpu_monitor/training_guard_r7.jsonl'
    while True:
        owner_path=root/'coordinator.lock/owner.json'
        if not owner_path.is_file():return
        owner=json.loads(owner_path.read_text())
        if owner!=expected:return
        control=root/'dispatch_control.json'
        if control.exists():
            existing=json.loads(control.read_text())
            if existing.get('observer_id')!=guard_id:
                raise RuntimeError('external dispatch control present; will not overwrite it')
            limit=min(limit,int(existing['workers']))
        baseline,current=read_samples(paths);comparison=compare(baseline,current)
        worst=comparison[0]['throughput_drop'] if comparison else None
        steps=sorted((p,r['sample_step'])for p,r in current.items())
        action=None
        if worst is not None and worst>.35 and steps!=applied_steps and limit>2:
            if json.loads(owner_path.read_text())!=expected:return
            limit=max(2,limit//2);applied_steps=steps
            action={'workers':limit,'observer_id':guard_id,'expected_owner':expected,
                'reason':'same-job comparable median throughput dropped more than 35 percent',
                'worst_comparable_throughput_drop':worst,'threshold':.35,
                'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            atomic(control,action)
        sample={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'observer_pid':os.getpid(),
            'observer_hostname':socket.gethostname(),'expected_owner':owner,'workers_limit':limit,
            'baseline':baseline,'current':current,'comparisons':comparison,
            'worst_comparable_throughput_drop':worst,'threshold':.35,'control_written':action,
            'reference_scope':'first 60 samples after 3 warmup samples in this reboot job; rows have no per-row wall timestamps; idle-only reference is unverified',
            'training_files_modified':False}
        with log.open('a')as f:f.write(json.dumps(sample,ensure_ascii=False)+'\n');f.flush()
        atomic(root/'cpu_monitor/training_guard_latest_r7.json',sample)
        if a.once:return
        time.sleep(a.interval)

if __name__=='__main__':main()
