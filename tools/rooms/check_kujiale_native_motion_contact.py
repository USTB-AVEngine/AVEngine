#!/usr/bin/env python3
"""Measure registered native idle/walk skin contact without changing animation."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def depth_support(depth, origin_y, floor_y, fov_deg=85., tolerance_m=.02):
    """Recover skin height from native ray-distance depth, keeping the gate fixed."""
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0) & (depth < 65504))
    result = {'visible_body_pixels': len(y), 'native_depth_sha256': hashlib.sha256(np.asarray(depth, dtype='<f4').tobytes()).hexdigest()}
    valid = len(y) >= 100 and y.max() < depth.shape[0]-1 and x.min() > 0 and x.max() < depth.shape[1]-1
    if not valid:
        return {**result, 'status': 'fail', 'reason': 'full_body_depth_missing_or_cropped'}
    tangent = math.tan(math.radians(fov_deg)/2)
    rx = (2*(x+.5)/depth.shape[1]-1)*tangent
    ry = (1-2*(y+.5)/depth.shape[0])*tangent
    world_y = origin_y + ry/np.sqrt(1+rx*rx+ry*ry)*depth[y, x]
    return {**result, 'status': 'measured', 'lowest_skin_y_m': float(world_y.min()),
            'gap_m': float(world_y.min()-floor_y), 'below_floor_pixel_fraction': float(np.mean(world_y < floor_y-tolerance_m))}


def production_plan(registry, asset_id, entity_class, actor_id, base, origin, idle):
    """Use the real neutral-to-UE adapter; do not counterfeit its binding flag."""
    from avengine.capture.qa_plan_adapters import materialize_ue_episode_plan
    camera = {'position_m': origin.tolist(), 'basis': {'forward': [0., 0., 1.]}}
    state = {'actor_id': actor_id, 'root_transform': {'translation_m': base.tolist(), 'rotation_xyzw': [0., 0., 0., 1.]}, 'action_id': idle, 'action_phase': 0.}
    neutral = {'plan_coordinates': 'renderer_neutral', 'resources': {}, 'scene': {},
               'visual_plan': {'actors': [{'actor_id': actor_id, 'asset_id': asset_id, 'entity_class': entity_class}],
                               'camera': camera, 'frames': [{'frame_index': 0, 'camera_state': camera, 'actor_states': [state]}]}}
    return materialize_ue_episode_plan(neutral, registry)['visual_plan']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--house',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--opposite-view',action='store_true',help='Measure both front and rear native skin surfaces at the same phase.')
    parser.add_argument('--production-phases',action='store_true',help='Include the registered production walk phase period, in addition to 15FPS clip coverage.')
    parser.add_argument('--production-binding',action='store_true',help='Create actor declarations through materialize_ue_episode_plan.')
    args=parser.parse_args()
    house=json.loads(args.house.read_text());ue=house['input']['ue_config']
    import sys;sys.path.insert(0,ue['spear_ext_dir'])
    import cv2
    from avengine.capture.kujiale_spear_runtime import launch_native,camera_helpers,apply_camera,close_native,to_ue,from_ue
    from avengine.capture.native_ue_source_appearance import verify_original_ue_materials
    from avengine.rooms.qa_episode import source_declaration
    from avengine.runtime_profiles import resolve_source_asset_runtime_profile
    from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
    registry=json.loads(args.registry.read_text());records=[r for r in registry['assets'] if r['entity_class']=='articulated_human']
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result={'schema':'kujiale_native_human_motion_contact_v2','status':'started','records':[],'fps':15,'contact_tolerance_m':.02,
            'method':'native skinned depth lowest surface versus measured native support plane; no animation or per-phase floor correction',
            'production_binding':args.production_binding,'opposite_native_views':args.opposite_view,'production_phase_coverage':args.production_phases,
            'coverage':'registered idle/walk clips only; sampled support-plane contact, not arbitrary future actions or cloth/hand furniture collision'}
    instance=camera=components=None;runtimes={}
    try:
        instance,game,launch=launch_native(ue,house['input']['ue_map_path']);result['launch']=launch
        spawn,depth_reader,_,_,_,spawn_actors,apply_actor,destroy=camera_helpers()
        room=house['checked_rooms'][0]
        base=np.asarray(room['placement']['source_1_m'])-[0,1.2,0]
        base[1]+=room['floor_reference']['native_ue_floor_offset_m']
        front_origin=base+[0.,1.,-3.]
        views=[('front',front_origin,90.)]
        if args.opposite_view:views.append(('rear',base+[0.,1.,3.],-90.))
        with instance.begin_frame():
            camera,components=spawn(game,horizontal_fov_deg=85.,width=1024,height=1024)
            for name in ('rgb','object_ids'):components[name].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
        with instance.end_frame():pass
        for record in records:
            aid=record['asset_id'];row={'asset_id':aid,'actions':[]}
            declaration=source_declaration(registry,aid,'source1')
            state={'actor_id':'source1','translation_ue_cm':to_ue(base),'actor_yaw_ue_deg':0.,'action_id':declaration['idle_action_id'],'action_phase':0.,'ue_animation':declaration['animation_paths_by_action_id'][declaration['idle_action_id']]}
            visual_plan={'actors':[declaration],'frames':[{'actor_states':[state]}]}
            if args.production_binding:
                visual_plan=production_plan(registry,aid,record['entity_class'],'source1',base,front_origin,declaration['idle_action_id'])
                declaration=visual_plan['actors'][0];state=visual_plan['frames'][0]['actor_states'][0]
                row['neutral_runtime_binding']=visual_plan['ue_neutral_runtime_binding']
            try:
                with instance.begin_frame():
                    runtimes=spawn_actors(game,{'plan':visual_plan});native=runtimes['source1'];native['asset_id']=aid
                    row['original_material_readback']=verify_original_ue_materials(game,native,aid,ue.get('native_source_material_bindings'))
                    row['component_frame_correction']=native.get('component_frame_correction')
                    native['component'].set_property_value(property_name='bEnableUpdateRateOptimizations',property_value=False)
                    native['component'].set_property_value(property_name='VisibilityBasedAnimTickOption',property_value='AlwaysTickPoseAndRefreshBones')
                    row['native_pose_tick_policy']={k:native['component'].get_property_value(property_name=k) for k in ('bEnableUpdateRateOptimizations','VisibilityBasedAnimTickOption')}
                    components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
                    components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[native['visual_actor']])
                with instance.end_frame():pass
                timeline=resolve_source_asset_runtime_profile(registry,aid).get('timeline',{})
                for action in dict.fromkeys((declaration['idle_action_id'],declaration['walking_action_id'])):
                    length=float(native['lengths'][declaration['animation_paths_by_action_id'][action]])
                    count=max(2,math.ceil(length*15))
                    if args.production_phases and action==declaration['walking_action_id']:
                        count=max(count,int(timeline.get('walk_phase_period_frames',30)))
                    samples=[]
                    for index in range(count):
                        state.update(action_id=action,action_phase=index/count,ue_animation=declaration['animation_paths_by_action_id'][action])
                        measurements=[]
                        for name,origin,yaw in views:
                            with instance.begin_frame():
                                apply_camera(camera,{'position_m':origin.tolist(),'kind':'perspective','yaw_deg':yaw,'pitch_deg':0.})
                                _,animation=apply_actor(native,state,index)
                                pose=read_scene_component_pose(native['anchor_root'])
                            with instance.end_frame():depth=depth_reader(components['depth'])
                            for _ in range(2):
                                with instance.begin_frame():_,animation=apply_actor(native,state,index)
                                with instance.end_frame():depth=depth_reader(components['depth'])
                            with instance.begin_frame():post_tick_position=float(native['component'].GetPosition())
                            with instance.end_frame():pass
                            if abs(post_tick_position-animation['requested_position_seconds'])>.001:
                                raise ValueError('native stopped animation advanced during pose settling')
                            measurement={'view':name,'native_camera_position_m':origin.tolist(),'post_tick_native_position_seconds':post_tick_position,
                                         **depth_support(depth,float(origin[1]),float(base[1]))}
                            if index in (0,count//2,count-1):
                                z=np.where((depth>0)&(depth<65504),np.clip(depth/6,0,1)*255,0).astype('uint8')
                                witness=args.output/f'{aid}__{action}_{index:04d}_{name}_depth.png'
                                if not cv2.imwrite(str(witness),z):raise ValueError('native depth witness write failed')
                                measurement['depth_witness']=str(witness)
                            measurements.append(measurement)
                        sample={'frame_index':index,'phase':index/count,'native_animation':animation,'pose_settle_native_frames':2,
                                'root_position_m':from_ue(pose['location_cm']),'views':measurements,'status':'fail'}
                        if all(m['status']=='measured' for m in measurements):
                            minimum=min(m['lowest_skin_y_m'] for m in measurements)
                            sample.update(lowest_skin_y_m=minimum,measured_floor_y_m=float(base[1]),gap_m=minimum-float(base[1]),
                                          status='pass' if abs(minimum-float(base[1]))<=.02 else 'fail')
                        else:sample['reason']='full_body_depth_missing_or_cropped'
                        samples.append(sample)
                    unique_depth=len({tuple(m['native_depth_sha256'] for m in sample['views']) for sample in samples})
                    pose_observed=unique_depth>=2
                    row['actions'].append({'action_id':action,'native_clip_seconds':length,'sample_count':count,'samples':samples,
                                           'distinct_native_depth_frames':unique_depth,'native_motion_observed':pose_observed,
                                           'status':'pass' if all(s['status']=='pass' for s in samples) and pose_observed else 'fail',
                                           'failures':[] if pose_observed else ['sampled_native_pose_remained_static']})
                row['status']='pass' if row['actions'] and all(action['status']=='pass' for action in row['actions']) else 'fail'
            except Exception as error:row.update(status='fail',error=f'{type(error).__name__}: {error}')
            finally:
                if runtimes:destroy(instance,runtimes);runtimes={}
            result['records'].append(row);write(args.output/'live_result.json',result)
            print(aid,row['status'],row.get('error'),flush=True)
        result.update(status='completed',human_count=len(records),passed_humans=sum(r['status']=='pass' for r in result['records']))
    finally:
        if instance is not None:close_native(instance,camera,components)
        result['machine_seconds']=time.monotonic()-started;write(args.output/'result.json',result)

if __name__=='__main__':main()
