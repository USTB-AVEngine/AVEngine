#!/usr/bin/env python3
"""Find a static four-body alpha layout in an already admitted native room."""
from __future__ import annotations
import argparse,json,math,itertools,time
from pathlib import Path
import numpy as np
from shapely.geometry import shape,Point
from avengine.capture.native_floor_reference import NativeFloorReference
from tools.rooms.check_kujiale_native_capsule_contact import query_capsule


def find_layout(house,inputs,room_id,*,search_new_cameras=False):
    from avengine.acoustics.runtime import load_habitat_runtime
    hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime']);pf=hs.PathFinder()
    if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('navmesh missing')
    room=next(r for r in house['checked_rooms'] if r['room_label']==room_id)
    floor=NativeFloorReference(house['input']['ue_config']['native_floor_surface'])
    plan=json.loads(Path(house['input']['ue_config']['acceptance_plan_path']).read_text());p=next(r for r in plan['rooms'] if r['room_id']==room_id)
    polygon=shape(p['floor_polygon']);minx,minz,maxx,maxz=polygon.bounds;native_y=float(room['placement']['source_1_m'][1]-1.2)
    candidates=[]
    for x in np.arange(minx+.15,maxx,.25):
        for z in np.arange(minz+.15,maxz,.25):
            if not polygon.covers(Point(float(x),float(z))):continue
            point=np.asarray(pf.snap_point(np.asarray([x,native_y,z],np.float32)),float)
            if not np.isfinite(point).all() or np.linalg.norm(point[[0,2]]-[x,z])>.05:continue
            if pf.distance_to_closest_obstacle(point.astype(np.float32),.5)<.2:continue
            try:root=np.asarray(floor.sample(point))
            except ValueError:continue
            probe={'witness_index':len(candidates),'role':'source','capsule':{'center_ue_cm':[root[0]*100,root[2]*100,(root[1]+.75)*100],'radius_cm':20.,'half_height_cm':75.}}
            capsule=query_capsule(floor.mesh,probe)
            if capsule['status']!='pass':continue
            candidates.append({'position_m':(point+[0.,1.2,0.]).tolist(),'root_m':root.tolist(),'capsule_check':capsule})
    cameras=[w['placement']['camera_m'] for w in room['native_witnesses']]
    if search_new_cameras:
        # Extend search, not frozen thresholds. Additional cameras stay inside
        # the admitted room polygon and need original >=25cm nav clearance.
        for x in np.arange(minx+.25,maxx,.5):
            for z in np.arange(minz+.25,maxz,.5):
                if not polygon.covers(Point(float(x),float(z))):continue
                nav=np.asarray(pf.snap_point(np.asarray([x,native_y,z],np.float32)),float)
                if not np.isfinite(nav).all() or np.linalg.norm(nav[[0,2]]-[x,z])>.05:continue
                if pf.distance_to_closest_obstacle(nav.astype(np.float32),.5)<.25:continue
                try:floor.sample(nav)
                except ValueError:continue
                cameras.append((nav+[0,1.5,0]).tolist())
    for camera in cameras:
        camera=np.asarray(camera,float);possible=[]
        for c in candidates:
            delta=np.asarray(c['position_m'])-camera;distance=float(np.linalg.norm(delta))
            if not 1.<=distance<=5.:continue
            # Exact native stage triangles, camera-to-body segment, unchanged geometry.
            hits,_,_=floor.mesh.ray.intersects_location(camera[None],delta[None]/distance,multiple_hits=False)
            if len(hits) and np.linalg.norm(hits[0]-camera)<distance-.15:continue
            angle=math.degrees(math.atan2(delta[0],-delta[2]));possible.append((angle,c))
        for yaw in np.arange(-180.,180.,10.):
            batch=[c for angle,c in possible if abs((angle-yaw+180)%360-180)<=37.5]
            batch.sort(key=lambda c:np.linalg.norm(np.asarray(c['position_m'])-camera))
            for first in batch:
                chosen=[first]
                # Farther points first reduce body overlap without moving the camera.
                order=sorted(batch,key=lambda c:min(np.linalg.norm((np.asarray(c['root_m'])-np.asarray(first['root_m']))[[0,2]]),5.),reverse=True)
                for c in order:
                    if all(np.linalg.norm((np.asarray(c['root_m'])-np.asarray(q['root_m']))[[0,2]])>=1.05 for q in chosen):chosen.append(c)
                    if len(chosen)==4:
                        a,b=np.asarray(chosen[0]['position_m']),np.asarray(chosen[1]['position_m'])
                        if not 1<=np.linalg.norm(a-b)<=5:break
                        return {'house':house['house'],'room_id':room_id,'status':'pass','listener_m':camera.tolist(),'center_yaw_deg':float(yaw),'valid':chosen,
                            'candidate_count':len(candidates),'frozen_source_clearance_m':.15,'capsule_radius_m':.20,'capsule_half_height_m':.75,
                            'placement_scope':'CPU nav/native capsule search; actual UE skin, colors and four-body bounds still require the alpha render',
                            'admission_thresholds_changed':False,'camera_search_scope':'same room polygon with frozen >=0.25m nav clearance' if search_new_cameras else 'three previously admitted UE witness cameras','camera_needs_native_scene_validation':search_new_cameras and not any(np.max(np.abs(camera-w['placement']['camera_m']))<1e-5 for w in room['native_witnesses'])}
    return {'house':house['house'],'room_id':room_id,'status':'blocked','candidate_count':len(candidates),'reason':'no_four_separated_native_capsules_in_frozen_rules_camera_sector' if search_new_cameras else 'no_four_separated_native_capsules_in_frozen_witness_camera_sector','camera_count':len(cameras)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['house','inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--room-id',default='R0');a=p.parse_args();start=time.perf_counter()
    result=find_layout(json.loads(a.house.read_text()),json.loads(a.inputs.read_text()),a.room_id);result['CPU_search_wall_seconds']=time.perf_counter()-start
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='valid'}),flush=True)
    if result['status']!='pass':raise SystemExit(2)
if __name__=='__main__':main()
