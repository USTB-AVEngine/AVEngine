"""Static seated witness through the original phase-two seating compiler."""
from pathlib import Path
import math,time,traceback
import numpy as np
from avengine.assets.roster_motion.seating import compile_seated
from avengine.timeline.continuous_pilot import write_json


def seated_witness(video,scene,room,entry,directory,save_sample=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();video.actors=[entry]
    try:
        # Proposals only. Final body, chair, feet, clothing, stage clearance,
        # native floor agreement and private route preservation are checked by
        # the unchanged seated compiler. A standing admission is not assumed.
        listener=np.asarray(scene['listener_m']);floor=float(room['measured_floor_y_m'])
        cache=getattr(video,'_visible_room_support_cache',None)
        points=[p['point']for p in cache[2]['floor']]if cache else []
        points=list({tuple(round(float(x),5)for x in p):p for p in points}.values())
        points.sort(key=lambda p:abs(np.linalg.norm((np.asarray(p)-listener)[[0,2]])-2.5))
        points=[p for p in points if np.linalg.norm((np.asarray(p)-listener)[[0,2]])>=2.2][:16]
        if not points:raise ValueError('no measured floor proposal at visible seated camera distance')
        entry.update(resolved_root_m=np.asarray(points[0]),resolved_yaw_rad=0.)
        video._apply_entry(entry,scene)
        scene['sources'][entry['endpoint_id']]['body_yaw_deg']=0.
        scene['native_placement_candidates']=[]
        request={'action':'sit','chair_id':'v7_chair_stool','minimum_camera_distance_m':2.2,'native_chair_floor_points_m':points}
        record=compile_seated(video,entry,scene,video.motion_house,video.motion_inputs,directory/'native_seated',request)
        if len(set(record['pose_glbs']))!=1 or len({tuple(x)for x in record['world_roots_m']})!=1 or any(f['moving']for f in record['source_frames']):
            raise ValueError('seated feasibility only accepts the original static seated pose')
        entry['motion_episode']=record;readback=video._apply_motion_entry(entry,0)
        video.material_checks=[];video.color_checks=[];views=[];best=None;maximum_body=0;maximum_chair=0
        chair_entry=video.seat_material_entries[-1]
        for i in range(8):
            yaw=i*45.;forward=[math.sin(math.radians(yaw)),0.,-math.cos(math.radians(yaw))]
            pixels=video._native_pair_pixels(scene,[entry,chair_entry],forward)
            maximum_body=max(maximum_body,pixels[entry['declaration']['asset_id']]);maximum_chair=max(maximum_chair,pixels[chair_entry['declaration']['asset_id']])
            if min(pixels.values())<100:continue
            obs=video.sim.get_sensor_observations();rgb=np.asarray(obs['rgb'])[:,:,:3];semantic=np.asarray(obs['semantic'])
            video._check_source_frame(rgb,semantic,i);views.append({'yaw_deg':yaw,'native_pixels':pixels})
            if best is None:best=rgb.copy()
        measured=[m for m in video.material_checks if m['status']in {'pass','fail'}]
        if not views:raise ValueError('seated body and chair lack a simultaneous actual native 100 pixel view')
        if any(m['status']=='fail'for m in measured):raise ValueError('seated_native_material_failure')
        if {m['asset_id']for m in measured}!= {entry['declaration']['asset_id'],chair_entry['declaration']['asset_id']}:raise ValueError('seated appearance not measured for both original entities')
        sample=None
        if save_sample:
            from PIL import Image
            sample=directory/'witness.png';Image.fromarray(best).save(sample)
        result={'status':'pass','category':'chair_seated','asset_id':entry['declaration']['asset_id'],'chair_asset_id':chair_entry['declaration']['asset_id'],
            'seated_geometry_receipt':str(directory/'native_seated/seated_geometry.json'),'original_seating_request':request,
            'static_pose_frames':901,'unique_pose_meshes':1,'native_mouth_readback':readback,
            'observed_views':views,'maximum_native_body_pixels':maximum_body,'maximum_native_chair_pixels':maximum_chair,
            'material_checks':video.material_checks,'color_checks':video.color_checks,
            'chair_floor_deltas_m':record['native_support_deltas_m'],'seat_contact_gap_m':record['seat_contact_gap_m'],
            'worst_highlight_ratio':max(m.get('highlight_ratio',0.)for m in measured),'sample_image':str(sample)if sample else None,
            'wall_seconds':time.monotonic()-start,'scope':'original static seated geometry witness; no audio, RIR or formal parent admission'}
    except Exception as error:
        result={'status':'fail','category':'chair_seated','asset_id':entry['declaration']['asset_id'],'error':str(error),
            'seated_rejection_receipt':str(directory/'native_seated/seat_rejected.json'),'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start}
    write_json(directory/'result.json',result)
    return {'status':result['status'],'selected':result if result['status']=='pass'else None,'attempts':[str(directory/'result.json')],
        'failure_reasons':{}if result['status']=='pass'else {result['error']:1}}
