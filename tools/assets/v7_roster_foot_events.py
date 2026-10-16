"""Export contact-onset events without confusing toe roll with a new footstep."""
import argparse,json
from pathlib import Path
def run(T):
 events=[]
 paths=list((T/'round2/assets/humans').glob('*/walking_*.json'))+list((T/'round2/assets/dogs').glob('*/*_Walking.json'))
 for p in paths:
  d=json.loads(p.read_text())
  if 'frames'not in d:continue
  previous={};house=d.get('house',p.stem.replace('walking_',''))
  for i,f in enumerate(d['frames']):
   for side,foot in f.get('feet',{}).items():
    active=foot['in_contact']
    if active and not previous.get(side,False):events.append({'asset_id':p.parent.name,'house':house,'kind':'initial_support'if i==0 else'contact_onset','side':side,'time_s':i/30,'support_world_m':foot.get('support_world_m'),'audio_alignment_status':'not_applicable_no_footstep_audio_admitted_in_this_task'})
    previous[side]=active
 p=T/'round3/registrations/foot_contact_events.json';p.write_text(json.dumps({'events':events,'note':'Foot rolling support-vertex changes are excluded; real audio admission and waveform alignment belong to v7-families.'},ensure_ascii=False,indent=2)+'\n')
 print('CONTACT_EVENTS',len(events),sum(e['kind']=='contact_onset'for e in events))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
