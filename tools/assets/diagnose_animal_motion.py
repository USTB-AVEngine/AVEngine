"""Inspect all authored Walking sources and immutable old cat caches."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.profiles import Rig,paw_points,PAWS

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[];start=time.process_time()
    for record in json.loads(a.inventory.read_text())['assets']:
        if not record['has_walking']:continue
        try:
            rig=Rig.load(record);times=rig.samples();frames=[]
            for t in times:
                g=rig.globals(t);parts=rig.skin(g);marks=rig.landmarks(g);feet=paw_points(parts,rig.masks)
                front=(marks['head']-marks['tail'])[[0,2]];front/=np.linalg.norm(front)
                vertices=np.concatenate([x['positions'] for x in parts]);minimum=vertices[:,1].min()
                frames.append({'time_s':float(t),'head':marks['head'].tolist(),'tail':marks['tail'].tolist(),'body':marks['body'].tolist(),'mouth':marks['mouth'].tolist(),
                    'heading_error_to_minus_z_deg':float(np.degrees(np.arccos(np.clip(-front[1],-1,1)))),
                    'body_floor_min_m':float(minimum),'feet':{s:{'point':f['point'].tolist(),'height_after_rigid_ground_m':float(f['point'][1]-minimum)} for s,f in feet.items()}})
            feet=np.asarray([[f['feet'][s]['point'] for s in PAWS] for f in frames]);body=np.asarray([f['body'] for f in frames]);foot_relative=feet-body[:,None,:]
            dz=np.diff(foot_relative[:,:,2],axis=0)/np.diff(times)[:,None]
            row={'asset_id':rig.asset_id,'status':'diagnosed_not_admitted','provenance':rig.provenance,'frames':frames,
                 'maximum_heading_error_deg':max(f['heading_error_to_minus_z_deg'] for f in frames),
                 'paw_relative_z_speed_percentiles_m_s':np.percentile(dz,[5,25,50,75,95]).tolist(),
                 'maximum_four_paw_height_spread_m':float(np.ptp(feet[:,:,1],axis=1).max()),
                 'root_horizontal_displacement_m':float(np.linalg.norm((body[-1]-body[0])[[0,2]]))}
        except Exception as error:row={'asset_id':record['asset_id'],'status':'diagnostic_error','error':repr(error)}
        rows.append(row);print({k:v for k,v in row.items() if k not in ('frames','provenance')},flush=True)
        a.output.write_text(json.dumps({'assets':rows,'cpu_s':time.process_time()-start},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
