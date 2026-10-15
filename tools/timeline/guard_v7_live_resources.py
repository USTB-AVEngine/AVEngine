#!/usr/bin/env python3
"""Continuously rank live native jobs and pause admission under committed policy."""
from pathlib import Path
import argparse,datetime,json,time,fcntl
from avengine.timeline.v7_live_resource_guard import observe,publish


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--until-utc',required=True)
    p.add_argument('--interval-s',type=float,default=10.);a=p.parse_args();deadline=datetime.datetime.fromisoformat(a.until_utc);last_log=0.;last_state=None
    if not 1<=a.interval_s<=60:raise ValueError('live resource interval must be between one and sixty seconds')
    first=json.loads(a.manifest.read_text());guard_lock=(Path(first['output_root'])/'evidence/live_resource_guard.lock').open('a')
    try:fcntl.flock(guard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    previous={}
    while True:
        m=json.loads(a.manifest.read_text());o=Path(m['output_root']);snapshot=observe(m)
        for row in snapshot['ranked_parent_trees']:
            key=str(row['root_pid'])+':'+row['root_start_ticks'];old=previous.get(key)
            row['cpu_growth_since_last_watch_s']=None if old is None else row['cpu_including_reaped_children_s']-old
        publish(m,snapshot)
        now=datetime.datetime.now(datetime.timezone.utc);state=tuple(snapshot['reasons'])
        if state!=last_state or time.monotonic()-last_log>=300 or snapshot['critical_below_20GB']and time.monotonic()-last_log>=30:
            with(o/'WATCH_LOG.md').open('a')as f:f.write(now.isoformat()+' committed_live_guard '+json.dumps(snapshot,ensure_ascii=False)+'\n')
            previous={str(r['root_pid'])+':'+r['root_start_ticks']:r['cpu_including_reaped_children_s']for r in snapshot['ranked_parent_trees']}
            print(now.isoformat(),'reasons',state,'own_GB',round(snapshot['own_pss_bytes']/1e9,2),'combined_GB',round(snapshot['combined_pss_bytes']/1e9,2),'available_GB',round(snapshot['MemAvailable_bytes']/1e9,2),flush=True);last_log=time.monotonic()
            if state!=last_state:
                with(o/'PROGRESS.md').open('a')as f:f.write(now.isoformat()+' 代码内持续资源守卫：'+json.dumps({'reasons':state,'own_GB':round(snapshot['own_pss_bytes']/1e9,2),'combined_GB':round(snapshot['combined_pss_bytes']/1e9,2),'available_GB':round(snapshot['MemAvailable_bytes']/1e9,2),'disk_free_GB':round(snapshot['data_free_bytes']/1e9,2)},ensure_ascii=False)+'；只暂停新起、不信号活动进程。\n')
        last_state=state
        if now>=deadline:break
        time.sleep(a.interval_s)
if __name__=='__main__':main()
