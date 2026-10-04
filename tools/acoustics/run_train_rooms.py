#!/usr/bin/env python3
"""Bounded CPU subprocess driver with identities and completion receipts."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save

def allocate(output,maximum_workers):
    def sample():
        return {int(f[0][3:]):list(map(int,f[1:9])) for f in (s.split() for s in Path('/proc/stat').read_text().splitlines()) if f[0].startswith('cpu') and f[0][3:].isdigit()}
    a=sample();time.sleep(2);b=sample();busy={}
    for cpu,v in b.items():
        d=[x-y for x,y in zip(v,a[cpu])];n=sum(d);busy[cpu]=1-(d[3]+d[4])/n if n else 0
    idle=sorted((c for c in busy if busy[c]<.2),key=lambda c:(busy[c],c))
    workers=min(maximum_workers,len(idle)//16,len(busy)//24)
    if workers<1:raise RuntimeError('Insufficient idle CPUs for an eight-thread worker with spare CPUs')
    affinity=idle[:workers*8]
    receipt=dict(logical_cores=len(busy),idle_below_20_percent=len(idle),measurement='two seconds /proc/stat delta, no guest double-count',workers=workers,threads_per_worker=8,affinity=affinity,busy_fraction_by_cpu=busy,cpu_only=True,generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(output/'evidence/room_native_cpu_allocation.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--house-timeout',type=int,default=7200);a=p.parse_args()
    allocation=allocate(a.output,a.workers);houses=read(a.output/'inputs/candidates.json')
    repeats={r['house'] for r in read(a.output/'inputs/repeat_rooms.json')['rooms']}
    first=[h for h in houses if h['house'] in repeats]
    rest=[h for h in houses if h['house'] not in repeats]
    hm=[h for h in rest if h['family']=='hm3d'];mp=[h for h in rest if h['family']=='mp3d'];ordered=first
    while hm or mp:
        if hm:ordered.append(hm.pop(0))
        if mp:ordered.append(mp.pop(0))
    print('NATIVE_ALLOCATION',json.dumps(allocation),flush=True)
    def one(h):
        house=h['house'];log=a.output/'logs'/('native_'+house+'.log')
        command=['taskset','-c',','.join(map(str,allocation['affinity'])),sys.executable,str(REPO/'tools/acoustics/train_room_acceptance.py'),'measure','--output',str(a.output),'--house',house]
        with log.open('x') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            started=dict(house=house,pid=proc.pid,argv=command,log=str(log),started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),proc_start_ticks=Path('/proc/'+str(proc.pid)+'/stat').read_text().split()[21])
            save(a.output/'evidence'/('native_started_'+house+'.json'),started)
            try:rc=proc.wait(timeout=a.house_timeout)
            except subprocess.TimeoutExpired:
                # This Popen object still owns the verified live child. No
                # name-based process matching or unrelated process stopping.
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                rc=-999
        r=dict(house=house,pid=proc.pid,returncode=rc,log=str(log),finished_marker=(a.output/'houses'/house/'finished.json').exists(),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(a.output/'evidence'/('native_finished_'+house+'.json'),r)
        print('HOUSE_JOB_FINISHED',json.dumps(r),flush=True);return r
    results=[]
    with ThreadPoolExecutor(max_workers=allocation['workers']) as executor:
        futures=[executor.submit(one,h) for h in ordered]
        for f in as_completed(futures):results.append(f.result())
    save(a.output/'evidence/native_driver_finished.json',dict(houses=len(results),results=results,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('ALL_ROOM_NATIVE_FINISHED',len(results),flush=True)
if __name__=='__main__':main()
