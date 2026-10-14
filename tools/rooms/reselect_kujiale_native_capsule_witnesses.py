#!/usr/bin/env python3
"""Reselect only already-frozen legal witnesses that clear the native capsule."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from copy import deepcopy
from avengine.capture.native_floor_reference import NativeFloorReference
from tools.rooms.check_kujiale_native_capsule_contact import query_capsule


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--rooms',nargs='+',default=['kujiale_0003:R0','kujiale_0065:R0'])
 p.add_argument('--output-directory',type=Path);p.add_argument('--selection-output',type=Path);a=p.parse_args()
 root=a.root/'ue_round6';records=[]
 destination=(a.output_directory or root/'capsule_reselected_houses').resolve()
 if a.root.resolve() not in destination.parents:raise ValueError('reselection output must remain in this task root')
 for target in a.rooms:
  house,room_id=target.split(':',1)
  if house not in {r['house'] for r in read(a.root/'acoustic_pass_train_candidates.json')['houses']}:raise ValueError('house is not acoustic-qualified')
  original=root/'houses'/house;plan=read(original/'acceptance_plan.json');reference=read(root/'native_floor_spatial_v1'/ (house+'.json'))
  surface=NativeFloorReference(reference['native_floor_surface']);room=next(r for r in plan['rooms'] if r['room_id']==room_id);checks=[]
  for index,witness in enumerate(room['witnesses']):
   row={'original_witness_index':index,'checks':[]}
   for role,height in (('camera',1.5),('source_1',1.2),('source_2',1.2)):
    point=list(witness[role+'_m']);point[1]-=height
    try:
     support=surface.sample(point)
     probe={'witness_index':index,'role':role,'capsule':{'center_ue_cm':[100*support[0],100*support[2],100*(support[1]+.75)],'radius_cm':20.,'half_height_cm':75.}}
     row['checks'].append(query_capsule(surface.mesh,probe))
    except ValueError as error:row['checks'].append({'status':'fail','reason':str(error),'role':role})
   row['status']='pass' if all(c['status']=='pass' for c in row['checks']) else 'fail';checks.append(row)
  selected=[r['original_witness_index'] for r in checks if r['status']=='pass'][:3]
  output=destination/house;output.mkdir(parents=True,exist_ok=False)
  result={'house':house,'room_id':room_id,'directory':str(output),'original_plan':str(original/'acceptance_plan.json'),'original_native_qa':str(original/'qa/result.json'),
          'checks':checks,'selected_original_witness_indices':selected,'status':'ready_for_native_recapture' if len(selected)==3 else 'insufficient_safe_frozen_witnesses',
          'method':'existing frozen legal candidates only; fixed native capsule 20cm radius / 75cm half-height and retained 2cm tolerance; no new geometry, source pair or relaxed rule'}
  write(output/'selection_receipt.json',result)
  if len(selected)==3:
   plan=deepcopy(plan);newroom=deepcopy(room);newroom['witnesses']=[{**room['witnesses'][i],'original_frozen_witness_index':i} for i in selected]
   plan['rooms']=[newroom];plan['witness_reselection_receipt']=str(output/'selection_receipt.json');write(output/'acceptance_plan.json',plan)
   cfg=read(original/'ue_config.json');cfg['acceptance_plan_path']=str(output/'acceptance_plan.json');write(output/'ue_config.json',cfg)
  print(house,selected,flush=True);records.append(result)
 selection_path=(a.selection_output or root/'capsule_witness_reselection.json').resolve()
 if a.root.resolve() not in selection_path.parents:raise ValueError('selection receipt must remain in this task root')
 write(selection_path,{'records':records})

if __name__=='__main__':main()
