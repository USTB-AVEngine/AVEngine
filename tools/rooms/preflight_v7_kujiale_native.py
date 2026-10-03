#!/usr/bin/env python3
"""Select a measured witness whose unchanged seeded motion stays navigable."""
from __future__ import annotations
import argparse,json,time
from copy import deepcopy
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('house','seed-row','inputs','thresholds','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    read=lambda f:json.loads(f.read_text())
    house,row,inputs=read(a.house),read(a.seed_row),read(a.inputs)
    import yaml
    from avengine.acoustics.runtime import load_habitat_runtime
    from avengine.timeline.v7_candidate_generation import seeded_scene
    from avengine.capture.native_ue_trajectory import check_native_trajectories
    thresholds=yaml.safe_load(a.thresholds.read_text())
    source_height=float(thresholds['parameters']['source_height_m']['value'])
    source_clearance=float(thresholds['parameters']['placement_source_clearance_m']['value'])
    if source_height!=1.2 or source_clearance!=.15:
        raise ValueError('native v6 adapter does not support different frozen source dimensions')
    hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
    pf=hs.PathFinder()
    if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('navigation load failed')
    room=next(r for r in house['checked_rooms'] if r['room_label']==row['room_id'])
    tested=[];chosen=None;start=time.monotonic()
    for witness in room['native_witnesses']:
        for swap in (False,True):
            candidate=deepcopy(house)
            selected=next(r for r in candidate['checked_rooms'] if r['room_label']==row['room_id'])
            selected['placement']=deepcopy(witness['placement'])
            if swap:
                place=selected['placement']
                place['source_1_m'],place['source_2_m']=place['source_2_m'],place['source_1_m']
            working,scene=seeded_scene(row,candidate)
            check=check_native_trajectories(scene,pf,source_height_m=source_height,source_clearance_m=source_clearance)
            tested.append({'native_witness_index':witness['index'],'measured_source_roles_swapped':swap,'check':check})
            if check['status']=='pass':chosen=candidate;break
        if chosen:break
    result={'schema':'kujiale_native_continuous_cpu_preflight_v1','status':'pass' if chosen else 'fail',
        'frozen_threshold_source':str(a.thresholds),'tested_measured_witnesses':tested,
        'trajectory_changed':False,'seed_row_changed':False,'elapsed_seconds':time.monotonic()-start}
    if chosen:
        chosen['native_trajectory_preflight']=str(a.output/'preflight.json')
        chosen['native_trajectory_measured_witness_index']=tested[-1]['native_witness_index']
        chosen['native_trajectory_measured_source_roles_swapped']=tested[-1]['measured_source_roles_swapped']
        (a.output/'house.json').write_text(json.dumps(chosen,indent=2)+'\n')
        (a.output/'seed_row.json').write_bytes(a.seed_row.read_bytes())
    (a.output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    if not chosen:raise SystemExit(1)


if __name__=='__main__':main()
