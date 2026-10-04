#!/usr/bin/env python3
"""Complete CPU room views as owned acoustic house jobs finish."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save

def owned_child_alive(receipt):
    pid=receipt['pid'];p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text();fields=stat.rsplit(') ',1)[1].split()
        if fields[0]=='Z' or fields[19]!=str(receipt['proc_start_ticks']):return False
        argv=(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        expected=receipt['argv'][3:] if receipt['argv'][0]=='taskset' else receipt['argv']
        if argv!=[os.fsencode(x) for x in expected]:return False
        uid=next(x for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1]
        return int(uid)==os.getuid()
    except (FileNotFoundError,ProcessLookupError):return False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true');a=p.parse_args()
    excluded=set(read(a.output/'evidence/room_native_cpu_allocation.json')['affinity'])
    def sample():
        return {int(f[0][3:]):list(map(int,f[1:9])) for f in (s.split() for s in Path('/proc/stat').read_text().splitlines()) if f[0].startswith('cpu') and f[0][3:].isdigit()}
    x=sample();time.sleep(2);y=sample();busy={}
    for cpu,v in y.items():
        d=[u-v for u,v in zip(v,x[cpu])];n=sum(d);busy[cpu]=1-(d[3]+d[4])/n if n else 0
    available=sorted((cpu for cpu in busy if busy[cpu]<.2 and cpu not in excluded),key=lambda cpu:(busy[cpu],cpu))
    workers=min(a.workers,len(available));affinity=available[:workers]
    if not workers:raise RuntimeError('No spare CPU for room views')
    allocation_name='room_visual_cpu_allocation_resume.json' if a.resume else 'room_visual_cpu_allocation.json'
    save(a.output/'evidence'/allocation_name,dict(workers=workers,threads_per_worker=1,affinity=affinity,excluded_native_affinity=sorted(excluded),idle_below_20_percent=len(available),logical_cores=len(busy),busy_fraction_by_cpu=busy,cpu_only=True))
    houses=read(a.output/'inputs/candidates.json');done=set();futures={};results=[]
    def one(h):
        house=h['house'];log=a.output/'logs'/('visual_'+house+'.log')
        command=['taskset','-c',','.join(map(str,affinity)),sys.executable,str(REPO/'tools/acoustics/train_room_visuals.py'),'--output',str(a.output),'--house',house]
        with log.open('x') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            save(a.output/'evidence'/('visual_started_'+house+'.json'),dict(house=house,pid=proc.pid,argv=command,log=str(log),proc_start_ticks=Path('/proc/'+str(proc.pid)+'/stat').read_text().split()[21]))
            try:rc=proc.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                rc=-999
        r=dict(house=house,pid=proc.pid,returncode=rc,log=str(log),finished_marker=(a.output/'visuals'/house/'finished.json').exists())
        save(a.output/'evidence'/('visual_finished_'+house+'.json'),r)
        print('VISUAL_JOB_FINISHED',json.dumps(r),flush=True);return r
    def adopt(h,receipt):
        house=h['house']
        while owned_child_alive(receipt):time.sleep(2)
        p=a.output/'evidence'/('visual_finished_'+house+'.json')
        if p.exists():return read(p)
        r=dict(house=house,pid=receipt['pid'],returncode=None,log=receipt['log'],finished_marker=(a.output/'visuals'/house/'finished.json').exists(),adopted_running_owned_child=True,exit_status_unavailable='previous parent terminated; artifact completion marker is checked')
        save(p,r);print('ADOPTED_VISUAL_JOB_FINISHED',json.dumps(r),flush=True);return r
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if a.resume:
            for h in houses:
                house=h['house'];finished=a.output/'evidence'/('visual_finished_'+house+'.json');started=a.output/'evidence'/('visual_started_'+house+'.json')
                if finished.exists():results.append(read(finished));done.add(house)
                elif started.exists():futures[house]=pool.submit(adopt,h,read(started))
            if len(futures)>workers:raise ValueError('Resume worker limit smaller than running owned children')
            print('VISUAL_RESUME',len(done),'completed;',len(futures),'owned running children;',workers,'workers',flush=True)
        while len(done)<len(houses):
            for h in houses:
                house=h['house']
                if house in done or house in futures or len(futures)>=workers:continue
                native=a.output/'evidence'/('native_finished_'+house+'.json')
                if not native.exists():continue
                n=read(native)
                if not n['finished_marker']:
                    for room in h['acoustic_rooms']:
                        path=a.output/'houses'/house/'rooms'/room['room_label']/'result.json'
                        if not path.exists():save(path,dict(house=house,family=h['family'],room_label=room['room_label'],status='review',reasons=['NATIVE_WORKER_INCOMPLETE'],rirs=[],leakage=[],compute_device='CPU',native_worker=n))
                futures[house]=pool.submit(one,h)
            for house,f in list(futures.items()):
                if f.done():results.append(f.result());done.add(house);del futures[house]
            if len(done)<len(houses):time.sleep(3)
    save(a.output/'evidence/visual_driver_finished.json',dict(houses=len(done),results=results,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('ALL_ROOM_VISUAL_FINISHED',len(done),flush=True)
if __name__=='__main__':main()
