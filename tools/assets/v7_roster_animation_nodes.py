"""Discard only animation channels on unreachable, non-skin placeholder nodes."""
import argparse,json
from pathlib import Path
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_roster_contact_ik import animation_nodes
from v7_roster_disk import check
def run(T):
 rows=[]
 for path in sorted((T/'round2/assets/humans').glob('*/final_ue_root.glb')):
  d,b=read_glb(path);valid=animation_nodes(d);removed=[]
  for a in d['animations']:
   bad=[c for c in a['channels']if c['target']['node']not in valid];removed.append({'clip':a['name'],'removed_channels':len(bad),'nodes':sorted(set(c['target']['node']for c in bad))});a['channels']=[c for c in a['channels']if c['target']['node']in valid]
  check(T,64*1024**2);temp=path.with_name('reachable_animation_pending.glb');write_glb(temp,d,b);temp.replace(path);rows.append({'asset_id':path.parent.name,'status':'pass','removed':removed,'visible_geometry_skin_bind_and_reachable_node_animation_unchanged':True});print('REACHABLE_ANIMATION',path.parent.name,sum(r['removed_channels']for r in removed),flush=True)
 json_write(T/'round3/evidence/animation_node_repairs.json',rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
