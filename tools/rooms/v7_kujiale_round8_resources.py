#!/usr/bin/env python3
"""One task-owned sampler for graphics processes and shared disk protection."""
from __future__ import annotations
import argparse,datetime,json,os,signal,shutil,time
from pathlib import Path
import psutil
from tools.rooms.v7_kujiale_ue_guard import gpu_snapshot


def allocation(roots):
    seen=set();allocated=logical=files=0
    for root in roots:
        for directory,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
            for name in names:
                p=Path(directory)/name
                try:
                    if p.is_symlink():continue
                    st=p.stat()
                except FileNotFoundError:continue
                logical+=st.st_size;files+=1;key=(st.st_dev,st.st_ino)
                if key not in seen:seen.add(key);allocated+=st.st_blocks*512
    return {'allocated_bytes':allocated,'logical_bytes':logical,'file_names':files,'unique_inodes':len(seen)}


def disk_violation(free,allocated,baseline,buffer=2_000_000_000):
    if free < baseline['minimum_shared_disk_free_bytes']+buffer:return 'shared_disk_80GiB_floor_buffer'
    if allocated-baseline['own_allocated_bytes_at_start'] > baseline['maximum_new_allocated_bytes']-buffer:return 'round8_40GB_growth_buffer'
    if allocated > baseline['maximum_total_task_allocated_bytes']-buffer:return 'task_80GB_allocation_buffer'
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--maximum-seconds',type=float,default=36000);a=p.parse_args()
    root=a.root.resolve();r8=root/'ue_round8';repo=Path(__file__).resolve().parents[2];baseline=json.loads((r8/'evidence/baseline.json').read_text())
    receipt_path=r8/'evidence/resource_sampler_identity.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        try:
            proc=psutil.Process(old['pid'])
            if proc.create_time()==old['create_time'] and proc.cmdline()==old['command']:raise RuntimeError('a task sampler is already active')
        except psutil.NoSuchProcess:pass
    mine=psutil.Process();receipt_path.write_text(json.dumps({'pid':mine.pid,'create_time':mine.create_time(),'command':mine.cmdline()},indent=2)+'\n')
    stop=False
    def terminate(_number,_frame):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    start=time.monotonic();tick=0;measured=None;scan_time=0
    with (r8/'evidence/disk_gpu_curve.jsonl').open('a') as stream:
        while not stop and time.monotonic()-start<a.maximum_seconds:
            began=time.monotonic()
            if measured is None or began-scan_time>=10:
                measured=allocation([root,repo]);scan_time=time.monotonic()
            free=shutil.disk_usage(root).free
            row={'schema':'kujiale_round8_resource_snapshot_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'sample_epoch':time.time(),'elapsed_seconds':time.monotonic()-start,'budget_root':str(root),'repo':str(repo),
                 'gpus':gpu_snapshot(),'shared_data_free_bytes':free,**measured,
                 'allocation_scan_age_seconds':time.monotonic()-scan_time,
                 'new_allocated_bytes':max(0,measured['allocated_bytes']-baseline['own_allocated_bytes_at_start']),
                 'disk_stop_reason':disk_violation(free,measured['allocated_bytes'],baseline)}
            temp=r8/'evidence/resource_snapshot.tmp';temp.write_text(json.dumps(row)+'\n');os.replace(temp,r8/'evidence/resource_snapshot.json')
            stream.write(json.dumps(row)+'\n');stream.flush();tick+=1
            time.sleep(max(0,start+tick-time.monotonic()))
    (r8/'evidence/resource_sampler_stopped.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start})+'\n')

if __name__=='__main__':main()
