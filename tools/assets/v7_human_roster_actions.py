"""Materialize real source-motion frames for native Habitat action-family probes."""
import argparse,copy,json
from pathlib import Path
from v7_human_roster_materials import json_write
from v7_human_roster_pose import materialize
def run(T,force=False):
 rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];cases=[]
 for row in rows:
  aid=row['asset_id'];color='red'if row['controlled_color']else'original';base=T/'assets/runtime'/aid/color/'runtime.glb';motion=T/'assets/motion'/aid;manifest=json.loads((motion/'motion.json').read_text())
  for family in ['walk','sit']:
   count=manifest[family]['frame_count'];frames=[0,count//2,count-1]if family=='walk'else[0]
   for fi in frames:
    label=family+'_'+str(fi).zfill(2);dest=T/'assets/action_frames'/aid/(label+'.glb')
    if dest.exists() and not force:ev=json.loads(dest.with_suffix('.pose.json').read_text())
    else:ev=materialize(base,motion,family,fi,dest);json_write(dest.with_suffix('.pose.json'),ev)
    cases.append({'asset_id':aid,'color':label,'controlled_color_source':color,'poses':{'stand':ev},'habitat_glb':str(dest.relative_to(T)),'review_family':family,'review_frame':fi})
  print('ACTION_FRAMES',aid,flush=True)
 json_write(T/'registrations/habitat_action_review_cases.json',cases);print('ACTION_CASE_COUNT',len(cases),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--force',action='store_true');a=p.parse_args();run(a.task_root,a.force)
