#!/usr/bin/env python3
"""Qualify moving RIR energy against retained native references (read-only)."""
from __future__ import annotations
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from avengine.timeline.continuous_scene import position_at,plan_rir_keyframes,interpolated_filter
from avengine.timeline.continuous_pilot import write_json
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);results=[]
    for house in sorted((a.pilot_root/'houses').iterdir()):
        cache=house/'validation_rir_cache'
        db=sqlite3.connect(f'file:{cache}/positions.sqlite3?mode=ro',uri=True)
        points=[(i,json.loads(x)) for i,x in db.execute('SELECT id,pose FROM rirs WHERE completed=1')];db.close()
        tree=cKDTree([x[1] for x in points]);memo={}
        def get(point):
            distance,index=tree.query(np.asarray(point))
            # This is floating-arithmetic equivalence of retained coordinates,
            # not spatial quantization or permission to choose a nearby RIR.
            if distance>1e-10:raise KeyError('independent reference missing at requested exact pose')
            key=points[index][0]
            if key not in memo:memo[key]=np.load(cache/f'rir_{key:06d}.npy',allow_pickle=False)
            return memo[key]
        for completed in sorted(house.rglob('completed.json')):
            receipt=json.loads(completed.read_text());scene=receipt['scene']
            if scene['duration_s']!=30:continue
            listener=np.asarray(scene['listener_m']);forward=np.asarray(scene['camera_forward'])
            track=scene['sources']['voice2']['trajectory']
            for seconds in (.25,.1):
                keys=plan_rir_keyframes(track,10+1/16000,listener,max_seconds=seconds,max_distance_m=.5)
                filters=[get(x['position_m']) for x in keys];samples=[x['sample'] for x in keys]
                checks=[]
                for t in np.arange(0.,10.00001,.05):
                    pos=position_at(track,t);ref=get(pos)
                    estimate=interpolated_filter(filters,samples,round(t*16000))
                    checks.append({'time_s':float(t),**compare_rirs(estimate,ref,source=pos,listener=listener,forward=forward)})
                result={'house':house.name,'scene_id':scene['scene_id'],'step_s':seconds,
                        'reference_step_s':.05,'reference_independent_of_coarse_keyframes':True,
                        'energy':qualify_energy(checks),'checks':checks,
                        'existing_house_thresholds_changed':False,'STARSS_waveforms_used':False}
                results.append(result)
                write_json(a.output/f'{scene["scene_id"]}__step{seconds}.json',result)
                print(json.dumps({k:result[k] for k in ('house','scene_id','step_s','energy')}),flush=True)
    write_json(a.output/'summary.json',{'runs':[{k:x[k] for k in ('house','scene_id','step_s','energy')} for x in results],
                'production_interpolation_admitted':all(x['energy']['status']=='pass' for x in results if x['step_s']==.25),
                'policy':'failed interpolation cannot publish formal media; refine natively and requalify'})

if __name__=='__main__':main()
