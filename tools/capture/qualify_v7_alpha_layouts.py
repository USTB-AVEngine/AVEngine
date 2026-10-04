#!/usr/bin/env python3
"""Qualify four static licensed bodies in already acoustically checked rooms.

CPU llvmpipe only. Native full-skin contact, both soles and a deliberately
penetrating control keep the existing 15mm criterion. This does not approve
walking or make an unmeasured house/room eligible for formal release.
"""
from __future__ import annotations
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,math,multiprocessing,os,resource,time
import numpy as np
from avengine.capture.v7_alpha_native import NativeAlphaVideo,standing_collision_probe
from avengine.timeline.continuous_pilot import write_json


def qualify(task):
    original,inputs,asset,output,wide=task;wall=time.perf_counter();cpu=time.process_time();results=[]
    for room in sorted(original['checked_rooms'],key=lambda r:-r.get('floor_area_m2',0.)):
        house=deepcopy(original);house['room']=room;pl=room['placement']
        listener=np.asarray(pl['camera_m']);a=np.asarray(pl['source_1_m']);b=np.asarray(pl['source_2_m'])
        forward=(a+b)/2-listener;forward[1]=0.;forward/=np.linalg.norm(forward)
        yaw=math.degrees(math.atan2(forward[0],-forward[2]))
        folder=Path(output)/original['house']/room['room_label'];folder.mkdir(parents=True,exist_ok=True)
        recipe={'house':house,'runtime_inputs':inputs,'asset':asset,'layout_revision':'four-static-actors-full-skin-native-v2','wide_bystander_views':wide}
        previous=folder/'summary.json'
        if previous.exists():
            result=json.loads(previous.read_text())
            if result.get('input_recipe')!=recipe:raise ValueError('completed static layout recipe changed; use a fresh output')
            results.append(result)
            if result['status']=='pass':return result
            continue
        scene={'duration_s':30.,'video_fps':15,'sample_rate_hz':16000,'listener_m':listener.tolist(),
            'camera_forward':forward.tolist(),'camera_yaw_knots':[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}],
            'sources':{'voice1':{'actor_id':1,'class_id':1,'trajectory':[{'time_s':0.,'position_m':a.tolist()},{'time_s':30.,'position_m':a.tolist()}]}},'events':[]}
        video=NativeAlphaVideo(house,scene,inputs,[asset],source_material_cache_directory=folder/'source_material_cache')
        valid=[];tried=[]
        try:
            candidates=[a,b]
            for distance in [1.6,2.,2.4,2.8,3.2,3.6,4.,4.4,4.8]:
                for offset in [-35.,-25.,-15.,-5.,5.,15.,25.,35.]:
                    angle=math.radians(yaw+offset);point=listener+distance*np.array([math.sin(angle),0.,-math.cos(angle)])
                    point[1]=a[1];candidates.append(point)
            if wide:
                for distance in [1.6,2.,2.4,2.8,3.2,3.6,4.,4.4,4.8]:
                    for offset in [-100.,-85.,-70.,-55.,-45.,45.,55.,70.,85.,100.]:
                        angle=math.radians(yaw+offset);point=listener+distance*np.array([math.sin(angle),0.,-math.cos(angle)])
                        point[1]=a[1];candidates.append(point)
            entry=video.actors[0]
            for index,point in enumerate(candidates):
                snap=np.asarray(video.pathfinder.snap_point(point-[0.,1.3,0.]),float)
                if not np.isfinite(snap).all() or np.linalg.norm(snap-(point-[0.,1.3,0.]))>.15:continue
                position=snap+[0.,1.3,0.];bearing=math.degrees(math.atan2(position[0]-listener[0],-(position[2]-listener[2])))
                if any(np.linalg.norm(position-np.asarray(r['position_m']))<.85 or abs((bearing-r['bearing_deg']+180)%360-180)<5 for r in valid):continue
                facing=math.degrees(math.atan2(listener[0]-point[0],-(listener[2]-point[2])))
                for orientation,turn in enumerate([0.,30.,-30.,90.,-90.,180.]):
                    source=scene['sources']['voice1'];source.update(body_yaw_deg=facing+turn,
                        trajectory=[{'time_s':0.,'position_m':position.tolist()},{'time_s':30.,'position_m':position.tolist()}])
                    entry['floor_y']=float(snap[1]);entry.pop('alpha_root_adjustment',None)
                    try:
                        rows=video._apply(scene,0)
                        proof=standing_collision_probe(video,entry,folder/f'point_{index:03d}_{orientation}')
                    except ValueError as error:
                        if not any(text in str(error) for text in ['native stage floor unavailable','static human floor correction outside','moving actor left admitted navigation support']):
                            raise
                        tried.append({'status':'blocked','reason':str(error),'index':index,
                            'orientation_index':orientation,'position_m':position.tolist()})
                        continue
                    pixels=int(np.count_nonzero(np.asarray(video.sim.get_sensor_observations()['semantic'])==entry['semantic_id']))
                    visibility_views=[{'yaw_deg':yaw,'pixels':pixels}]
                    if wide and len(valid)>=2 and pixels<200:
                        for view_delta in [-100.,-75.,-50.,50.,75.,100.]:
                            scene['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw+view_delta},{'time_s':30.,'yaw_deg':yaw+view_delta}]
                            video._apply(scene,0)
                            observed=int(np.count_nonzero(np.asarray(video.sim.get_sensor_observations()['semantic'])==entry['semantic_id']))
                            visibility_views.append({'yaw_deg':yaw+view_delta,'pixels':observed});pixels=max(pixels,observed)
                        scene['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]
                        video._apply(scene,0)
                    proof.update(visibility_probe_views=visibility_views,position_m=position.tolist(),observed_mouth_m=rows[0]['emitter_position_m'],
                        body_yaw_deg=facing+turn,bearing_deg=bearing,visible_pixels=pixels,index=index,orientation_index=orientation)
                    tried.append(proof)
                    if proof['status']=='pass' and pixels>=200:
                        lo,hi=np.asarray(proof['native_full_skin_bounds_m']);separate=True
                        for old in valid:
                            olo,ohi=np.asarray(old['native_full_skin_bounds_m'])
                            if np.all(np.minimum(hi,ohi)>np.maximum(lo,olo)):separate=False
                        if separate:
                            valid.append(proof);print('ADMITTED',original['house'],room['room_label'],len(valid),flush=True);break
                if len(valid)>=4:break
            result={'house':original['house'],'split':original['split'],'room_id':room['room_label'],
                'listener_m':listener.tolist(),'center_yaw_deg':yaw,'valid':valid,'tried_count':len(tried),
                'status':'pass' if len(valid)>=4 else 'blocked_insufficient_standing_positions',
                'stage_contact_geometry':video.stage_contact_geometry,'wall_seconds':time.perf_counter()-wall,
                'process_cpu_seconds':time.process_time()-cpu,'body_template_asset_id':asset['asset_id'],
                'motion_scope':'idle_pose0; no motion admission', 'contact_tolerance_changed':False,'input_recipe':recipe}
            write_json(folder/'summary.json',result);write_json(folder/'all_probes.json',tried);results.append(result)
            if result['status']=='pass':return result
        finally:video.close()
    return {'house':original['house'],'status':'blocked','room_attempts':results,
        'wall_seconds':time.perf_counter()-wall,'process_cpu_seconds':time.process_time()-cpu}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['houses','runtime-inputs','assets','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--house-ids',nargs='+',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--wide-bystander-views',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=64 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU worker/env bounds')
    houses=json.loads(a.houses.read_text())['houses'];by_id={h['house']:h for h in houses}
    if any(h not in by_id or by_id[h]['acoustic_status']!='pass' for h in a.house_ids):raise ValueError('unmeasured house requested')
    inputs=json.loads(a.runtime_inputs.read_text());assets=json.loads(a.assets.read_text());assets=assets['assets'] if isinstance(assets,dict) else assets
    asset=next(x for x in assets if x['endpoint_id']=='voice1');a.output.mkdir(parents=True,exist_ok=True)
    tasks=[(by_id[h],inputs,asset,str(a.output),a.wide_bystander_views) for h in a.house_ids];results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(qualify,task) for task in tasks]):
            result=future.result();results.append(result);print(result['house'],result['status'],flush=True)
    write_json(a.output/f'summary_{time.time_ns()}.json',{'houses':results,'status':'pass' if all(x['status']=='pass' for x in results) else 'blocked'})

if __name__=='__main__':main()
