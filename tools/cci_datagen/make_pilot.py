#!/usr/bin/env python3
"""Build an explicit train-only launch manifest from explicit pilot requests."""
import argparse,json
from pathlib import Path
from resources import GIB,write_json
from generate import sha,files_ok
p=argparse.ArgumentParser();p.add_argument('--generator-profile');p.add_argument('--requests',required=True,help='JSON list of request JSON paths; prepared v6 pilot recipes or accepted v7 sampler handoff')
p.add_argument('--recommended-houses',required=True);p.add_argument('--code-root',required=True);p.add_argument('--output',required=True);p.add_argument('--world-count',type=int,default=64)
p.add_argument('--pre-rendered-captures',help='optional JSON mapping world_id to render receipt with request_sha256, attempt and files')
p.add_argument('--scratch-gib-per-world',type=float,default=4);p.add_argument('--media-mib-per-world',type=float,default=128)
a=p.parse_args();houses=json.loads(Path(a.recommended_houses).read_text());allowed={h['house'] for h in houses['houses'] if h['split']=='train'}
paths=json.loads(Path(a.requests).read_text())
if len(paths)!=a.world_count:raise ValueError(f'expected {a.world_count} independent requests; got {len(paths)}')
pre=json.loads(Path(a.pre_rendered_captures).read_text()) if a.pre_rendered_captures else {}
release=json.loads((Path(a.code_root).resolve().parent/'RELEASE.json').read_text());source_commit=release['source_commit']
worlds=[];seeds=set();profile=json.loads(Path(a.generator_profile).read_text()) if a.generator_profile else {};code=Path(profile.get('stage_adapter',str(Path(a.code_root)/'tools/cci_datagen/world_stage.py')))
for path in paths:
 request=json.loads(Path(path).read_text());house=request['room_id'];seed=request['seed']
 if house not in allowed:raise ValueError('pilot request uses a held-out or unaccepted house: '+house)
 key=(house,seed)
 if key in seeds:raise ValueError('duplicate house/seed world')
 seeds.add(key);wid=request['world_id']
 if not request.get('simulation_request'):raise ValueError('request lacks explicit RLR simulation settings')
 stages={}
 for phase in ['plan','render','audio','deliver']:
  argv=['{python}',str(code),'--phase',phase,'--request',str(Path(path).resolve()),'--attempt','{attempt}']
  if phase=='render':argv+=['--upstream','{plan_root}','--gpu','{gpu}']
  if phase=='audio':argv+=['--upstream','{render_root}']
  if phase=='deliver':argv+=['--upstream','{audio_root}','--world-root','{world_root}']
  outputs=['{world_root}/ARTIFACTS.json'] if phase=='deliver' else ['{attempt}/stage.json']
  stages[phase]={'argv':argv,'outputs':outputs}
 worlds.append({'world_id':wid,'house':house,'seed':seed,'split':'train','request':str(Path(path).resolve()),'request_sha256':sha(path),'scratch_bytes_peak':int(a.scratch_gib_per_world*GIB),'delivery_bytes_upper':int(a.media_mib_per_world*1024**2),'stages':stages})
if pre:
 if set(pre)!=set(w['world_id'] for w in worlds):raise ValueError('pre-rendered handoff must cover exactly this request list')
 for w in worlds:
  receipt=pre[w['world_id']]
  if receipt.get('request_sha256')!=w['request_sha256'] or not receipt.get('attempt') or not files_ok(receipt.get('files',[])):raise ValueError('capture/request handoff hash mismatch')
  w['precompleted']={'render':receipt}
version={}
if a.generator_profile:
 version={'generator_profile':str(Path(a.generator_profile).resolve()),'generator_profile_sha256':sha(a.generator_profile),'generator_release':profile['generator_release'],'generator_release_sha256':sha(profile['generator_release']),'generator_source_commit':json.loads(Path(profile['generator_release']).read_text())['source_commit']}
write_json(a.output,{**version,'schema':'cci_datagen_stages_v1','train_only':True,'formal_frozen_test':False,'source_commit':source_commit,'scope':'v6 chain infrastructure pilot; not a v7 sampler implementation','worlds':worlds})
print(a.output)
