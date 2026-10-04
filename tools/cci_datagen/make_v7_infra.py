#!/usr/bin/env python3
"""Bind 64 train-house v7 canaries to immutable infrastructure-only stages."""
import argparse,json,os
from pathlib import Path
from resources import GIB,write_json
from generate import sha
from v7_infra_stage import check_request

def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--recommended-houses',required=True);p.add_argument('--code-root',required=True);p.add_argument('--generator-profile',required=True);p.add_argument('--output',required=True);p.add_argument('--world-count',type=int,default=64);p.add_argument('--scratch-gib-per-world',type=float,default=8);a=p.parse_args()
 profile_path=Path(a.generator_profile);profile=json.loads(profile_path.read_text())
 if not profile.get('infrastructure_trial') or profile.get('production_allowed') or profile.get('training_eligible') is not False:raise ValueError('not an infrastructure-only profile')
 paths=json.loads(Path(a.requests).read_text())
 if len(paths)!=a.world_count or a.world_count!=64:raise ValueError('need exactly 64 explicit canary requests')
 release=json.loads(Path(profile['generator_release']).read_text());worlds=[]
 for path in paths:
  req=json.loads(Path(path).read_text());check_request(req)
  if req['generator_profile_sha256']!=sha(profile_path):raise ValueError('request belongs to a different profile')
  stages={}
  for phase in ('plan','render','audio','deliver'):
   argv=['{python}','-B',profile['stage_adapter'],'--phase',phase,'--request',path,'--attempt','{attempt}', '--scratch-parent', '{scratch_parent}']
   if phase!='plan':argv+=['--plan-root','{plan_root}']
   if phase in ('audio','deliver'):argv+=['--render-root','{render_root}']
   if phase=='deliver':argv+=['--audio-root','{audio_root}','--world-root','{world_root}']
   if phase=='render':argv+=['--gpu','{gpu}']
   stages[phase]={'argv':argv,'outputs':['{world_root}/ARTIFACTS.json' if phase=='deliver' else '{attempt}/stage.json']}
  worlds.append({'world_id':req['row']['scene_id'],'split':'train','request':path,'request_sha256':sha(path),'scratch_bytes_peak':int(a.scratch_gib_per_world*GIB),'delivery_bytes_upper':int(profile['delivery_mib_per_world']*1024**2),'stages':stages})
 if len({w['world_id'] for w in worlds})!=64:raise ValueError('duplicate parent IDs')
 manifest={'schema':'cci_datagen_stages_v1','scope':'infrastructure_trial','training_eligible':False,'train_only':True,'formal_frozen_test':False,'formal_dataset_count':0,'source_commit':release['source_commit'],'generator_base_commit':release['generator_base_commit'],'generator_profile':str(profile_path),'generator_profile_sha256':sha(profile_path),'generator_release':profile['generator_release'],'generator_release_sha256':sha(profile['generator_release']),'worlds':worlds}
 write_json(a.output,manifest)
if __name__=='__main__':main()
