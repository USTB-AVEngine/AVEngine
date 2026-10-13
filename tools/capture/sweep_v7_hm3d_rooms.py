#!/usr/bin/env python3
"""Run one CPU HM3D room category witness job, without full media generation."""
import argparse,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--room-id',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--save-samples',action='store_true')
    args=p.parse_args()
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[os.getpid()%len(allowed)]})
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('CPU sweep requires nice >=10')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS']:
        if os.environ.get(key)!='1':raise RuntimeError(key+' must be 1 before importing native modules')
    from avengine.capture.v7_room_sweep import static_categories
    houses=json.loads((args.inputs/'houses.json').read_text());loc=houses['room_index'][args.room_id]
    h=next(h for h in houses['houses']if h['house']==loc['house']);room=h['checked_rooms'][loc['room_index']]
    runtime=json.loads((args.inputs/'runtime.json').read_text());assets=json.loads((args.inputs/'assets.json').read_text())['assets'];categories=json.loads((args.inputs/'categories.json').read_text())
    result=static_categories(h,room,runtime,assets,categories,args.output,args.save_samples)
    print(json.dumps({k:v for k,v in result.items()if k in {'status','house','room_id','error','wall_seconds','process_cpu_seconds','max_rss_kib'}},ensure_ascii=False),flush=True)
    return 0 if result['status']=='static_complete'else 2


if __name__=='__main__':raise SystemExit(main())
