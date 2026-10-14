#!/usr/bin/env python3
"""Find frozen legal native witnesses with navigable seeded actor motion."""
from __future__ import annotations
import argparse,json,math,time
from copy import deepcopy
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','house','seed-row','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--attempts',type=int,default=48);p.add_argument('--witness-count',type=int,default=12)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda f:json.loads(f.read_text())
    plan,house,row,inputs=read(a.plan),read(a.house),read(a.seed_row),read(a.inputs)
    from avengine.acoustics.runtime import load_habitat_runtime
    from avengine.timeline.v7_candidate_generation import seeded_scene
    from avengine.capture.native_ue_trajectory import check_native_trajectories
    from tools.rooms.room_selection.navigation import sample_navigation,farthest_sample,components,ray_clear_batch
    from shapely.geometry import shape
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
    pf=hs.PathFinder()
    if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('navmesh missing')
    b=Path(plan['build_directory']);mesh=trimesh.Trimesh(vertices=np.load(b/'surface/vertices.npy',allow_pickle=False),
        faces=np.load(b/'surface/triangles.npy',allow_pickle=False),process=False)
    mesh.ray=RayMeshIntersector(mesh)
    params=plan['frozen_parameters'];target=next(r for r in plan['rooms'] if r['room_id']==row['room_id'])
    geometry=shape(target['floor_polygon']);start=time.monotonic()
    sampled,points,clearance,adj,comps=sample_navigation(pf,hs,geometry,target['source_floor_y_m'],params)
    # The frozen placement search returns its first legal pair. Enumerate the
    # same declared camera/source candidates and identical filters, then add
    # full trajectory rejection. The shared selector and scorer are untouched.
    main=max(components(adj),key=len,default=[])
    cams=farthest_sample(points,[i for i in main if clearance[i]>=params['placement_camera_clearance_m']],params['placement_camera_samples'])
    sources=farthest_sample(points,[i for i in main if clearance[i]>=params['placement_source_clearance_m']],params['placement_source_samples'])
    witnesses=[];attempts=[];motion_cache={};tested_pairs=0;raw_rays=0
    for ci in cams:
        camera=points[ci]+[0.,params['camera_height_m'],0.]
        ids=[i for i in sources if i!=ci];ends=points[ids]+[0.,params['source_height_m'],0.]
        lengths=np.linalg.norm(ends-camera,axis=1)
        keep=(lengths>=params['distance_min_m'])&(lengths<=params['distance_max_m'])
        ends=ends[keep];ids=np.asarray(ids)[keep]
        clear=ray_clear_batch(mesh,camera,ends,params['ray_endpoint_tolerance_m']);raw_rays+=len(ends)
        ends=ends[clear];ids=ids[clear]
        ii,jj=np.triu_indices(len(ends),1)
        dist=np.linalg.norm(ends[ii]-ends[jj],axis=1)
        left,right=ends[ii][:,[0,2]]-camera[[0,2]],ends[jj][:,[0,2]]-camera[[0,2]]
        angle=np.degrees(np.arccos(np.clip(np.sum(left*right,axis=1)/(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)),-1,1)))
        keep=(dist>=params['distance_min_m'])&(dist<=params['distance_max_m'])&(angle<=params['camera_hfov_deg'])
        for i,j,d,ang in zip(ii[keep],jj[keep],dist[keep],angle[keep],strict=True):
            tested_pairs+=1
            witness={'found':True,'camera_m':camera.tolist(),'source_1_m':ends[i].tolist(),'source_2_m':ends[j].tolist(),
                'horizontal_angle_deg':float(ang),'pairwise_distances_m':[float(np.linalg.norm(ends[i]-camera)),float(np.linalg.norm(ends[j]-camera)),float(d)],
                'camera_clearance_min_m':params['placement_camera_clearance_m'],'source_clearance_min_m':params['placement_source_clearance_m'],
                'search':'same frozen sampled candidates, distances, FOV and three clear raw mesh segments; additionally rejects unnavigable unchanged seeded motion'}
            key=(int(ids[i]),int(ids[j]))
            if key not in motion_cache:
                candidate=deepcopy(house);selected=next(r for r in candidate['checked_rooms'] if r['room_label']==row['room_id'])
                selected['placement']=witness;_,scene=seeded_scene(row,candidate)
                motion_cache[key]=check_native_trajectories(scene,pf,source_height_m=params['source_height_m'],source_clearance_m=params['placement_source_clearance_m'])
            check=motion_cache[key]
            if check['status']!='pass':continue
            if not ray_clear_batch(mesh,ends[i],[ends[j]],params['ray_endpoint_tolerance_m'])[0]:continue
            raw_rays+=1
            witnesses.append(witness);attempts.append({'static_placement':witness,'continuous_preflight':check});break
        if len(witnesses)>=a.witness_count:break
    if len(witnesses)<3:
        result={'status':'fail','witnesses_found':len(witnesses),'attempts':attempts,'elapsed_seconds':time.monotonic()-start}
        a.output.with_suffix('.failed.json').write_text(json.dumps(result,indent=2)+'\n')
        raise ValueError('not enough frozen legal navigable witnesses in declared search budget')
    views=[v for v in target['views'] if v['kind']=='orthographic']
    for i,w in enumerate(witnesses):
        camera=np.asarray(w['camera_m']);forward=(np.asarray(w['source_1_m'])+w['source_2_m'])/2-camera
        forward[1]=0.;forward/=np.linalg.norm(forward)
        views.append({'view_id':f'{target["room_id"]}_camera_{i+1}','kind':'perspective','position_m':camera.tolist(),
            'position_ue_cm':(100*camera[[0,2,1]]).tolist(),'yaw_deg':math.degrees(math.atan2(forward[2],forward[0])),
            'pitch_deg':0.,'witness':w})
    target.update(witnesses=witnesses,views=views,witness_count_status='pass')
    plan['continuous_motion_preparation']={'seed_row':str(a.seed_row),'room_id':row['room_id'],
        'status':'pass','witness_count':len(witnesses),'attempts':attempts,'elapsed_seconds':time.monotonic()-start,
        'tested_static_pairs':tested_pairs,'raw_mesh_ray_count':raw_rays,'unique_motion_pair_tests':len(motion_cache),
        'frozen_room_parameters_changed':False,'scene_trajectory_logic_changed':False,'static_raw_mesh_rays_reused':True}
    a.output.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'room_id':row['room_id'],'witness_count':len(witnesses),'attempt_count':len(attempts),'seconds':time.monotonic()-start}))


if __name__=='__main__':main()
