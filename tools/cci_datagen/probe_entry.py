#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import time
from resources import probe,write_json
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);a=p.parse_args()
root=Path(a.output_root);rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));job=os.environ['DATAGEN_LAUNCH_ID']
folder=root/job
r=probe(folder/'io'/f'rank{rank}',True);write_json(folder/f'rank{rank:05d}.json',r)
if rank==0:
 end=time.monotonic()+30
 while len(list(folder.glob('rank*.json')))<world:
  if time.monotonic()>end:raise RuntimeError('missing rank reports; no probe completion marker written')
  time.sleep(.1)
 rows=[json.loads(p.read_text()) for p in sorted(folder.glob('rank*.json'))]
 write_json(root/'PROBE_COMPLETE.json',{'job':job,'reports':len(rows),'world_size':world,'hosts':sorted(set(r['host'] for r in rows)),'egl_render':'unverified; CPU-only probe does not initialize GPU','rank_reports':str(folder)})
print(json.dumps({'rank':rank,'host':r['host'],'cpu_cores':r['effective_cpu_cores'],'memory_bytes':r['memory_limit_bytes']}),flush=True)
