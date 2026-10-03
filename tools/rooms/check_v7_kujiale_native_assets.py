#!/usr/bin/env python3
"""Render all registered native source assets with fixed production appearance QA."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def vector_field(raw,name):
    from avengine.backends.spear_ue.research_runtime import _read_unreal_pose_vector
    for _ in range(3):
        lower={str(k).casefold():v for k,v in raw.items()}
        if name.casefold() in lower:
            return np.array(_read_unreal_pose_vector(lower[name.casefold()],keys=('X','Y','Z'),owner='native asset bounds'))
        if len(raw)==1 and isinstance(next(iter(raw.values())),dict):raw=next(iter(raw.values()))
        else:break
    raise ValueError('native asset bounds readback lacks '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--house',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit-assets',type=int)
    args=p.parse_args()
    house=json.loads(args.house.read_text());ue=house['input']['ue_config']
    import sys;sys.path.insert(0,ue['spear_ext_dir'])
    import cv2
    from avengine.capture.kujiale_spear_runtime import launch_native,camera_helpers,apply_camera,apply_native_rgb_profile,spawn_fill_lights,close_native,to_ue,from_ue
    from avengine.rooms.qa_episode import source_declaration
    from avengine.assets.source_materials import measure_source_appearance
    registry=json.loads(args.registry.read_text());inventory={a['asset_id']:a for a in json.loads(args.inventory.read_text())['assets']}
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result={'schema':'kujiale_native_source_asset_gallery_qa_v1','render_path':'ue_spear','records':[],
        'formal_training_asset_admission_count':0,'sampling_limitations':'three native camera azimuths, idle and sampled walk phase; full action/contact coverage remains separate'}
    instance=camera=components=None;runtimes={}
    try:
        instance,game,launch=launch_native(ue,house['input']['ue_map_path'])
        spawn,depth_reader,rgb_reader,_,_,spawn_actors,apply_actor,destroy=camera_helpers()
        room=house['checked_rooms'][0];base=np.asarray(room['placement']['source_1_m'])-[0.,1.2,0.]
        base[1]+=room['floor_reference']['native_ue_floor_offset_m']
        plan=json.loads(Path(ue['acceptance_plan_path']).read_text())
        with instance.begin_frame():
            camera,components=spawn(game,horizontal_fov_deg=85.,width=2560,height=1440)
            profile=apply_native_rgb_profile(components['rgb'],ue);lights=spawn_fill_lights(game,plan)
            original_source=components['rgb'].get_property_value(property_name='CaptureSource')
        with instance.end_frame():pass
        result.update(launch=launch,profile=profile,lights=lights)
        records=registry['assets'][:args.limit_assets] if args.limit_assets else registry['assets']
        for record in records:
            started=time.monotonic();aid=record['asset_id'];row={'asset_id':aid,'entity_class':record['entity_class'],'views':[],'status':'started'}
            try:
                declaration=source_declaration(registry,aid,'source1')
                static=declaration.get('motion_model')=='rigid_static'
                idle='static' if static else declaration['idle_action_id']
                state={'actor_id':'source1','translation_ue_cm':to_ue(base),'actor_yaw_ue_deg':0.,'action_id':idle,'action_phase':0.}
                if not static:state['ue_animation']=declaration['animation_paths_by_action_id'][idle]
                with instance.begin_frame():
                    runtimes=spawn_actors(game,{'plan':{'actors':[declaration],'frames':[{'actor_states':[state]}]}})
                    native=runtimes['source1']
                    bounds=native['visual_actor'].GetActorBounds(bOnlyCollidingComponents=False,bIncludeFromChildActors=True,as_dict=True)
                    center=np.array(from_ue(vector_field(bounds,'Origin')));extent=vector_field(bounds,'BoxExtent')/100
                    radius=float(np.linalg.norm(extent));distance=max(1.,radius*2.8)
                    for component in components.values():
                        component.set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
                        component.set_property_value(property_name='ShowOnlyActors',property_value=[native['visual_actor']])
                with instance.end_frame():pass
                row.update(native_bounds= bounds,native_radius_m=radius,native_binding={k:declaration.get(k) for k in ['blueprint_class_path','static_mesh_object_path','skeletal_mesh_path','actor_scale']})
                source=inventory[aid]['material_receipt'];profiles=source['profiles'];limit=min(v['max_highlight_ratio'] for v in profiles)
                for vi,yaw in enumerate((-45.,0.,45.)):
                    azimuth=math.radians(yaw);origin=center+[-distance*math.cos(azimuth),0.,-distance*math.sin(azimuth)]
                    camera_view={'position_m':origin.tolist(),'kind':'perspective','yaw_deg':yaw,'pitch_deg':0.}
                    action=idle if vi==0 or static else declaration['walking_action_id']
                    state['action_id']=action;state['action_phase']=0. if vi==0 or static else vi/4
                    if not static:state['ue_animation']=declaration['animation_paths_by_action_id'][action]
                    with instance.begin_frame():
                        apply_actor(native,state,vi);apply_camera(camera,camera_view)
                        components['rgb'].set_property_value(property_name='CaptureSource',property_value=original_source)
                    with instance.end_frame():pass
                    for _ in range(25):
                        with instance.begin_frame():pass
                        with instance.end_frame():rgb_reader(components['rgb'])
                    with instance.begin_frame():pass
                    with instance.end_frame():bgr=rgb_reader(components['rgb']);depth=depth_reader(components['depth'])
                    with instance.begin_frame():components['rgb'].set_property_value(property_name='CaptureSource',property_value='SCS_BaseColor')
                    with instance.end_frame():base_bgr=rgb_reader(components['rgb'])
                    mask=np.isfinite(depth)&(depth>0)&(depth<65504)
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);reference=cv2.cvtColor(base_bgr,cv2.COLOR_BGR2RGB)
                    metric=measure_source_appearance(rgb,mask,palette=np.asarray(source['source_palette_srgb']),max_highlight_ratio=limit,
                        reference_rgb=reference,metallic_surface=all(v['metallic']>.5 for v in profiles),strict=False)
                    albedo=measure_source_appearance(reference,mask,palette=np.asarray(source['source_palette_srgb']),max_highlight_ratio=limit,strict=False)
                    # BaseColor contains no specular term. Record its palette
                    # color test separately; highlight admission uses lit RGB
                    # and the same-view original BaseColor above.
                    color_fail=[x for x in albedo.get('failures',[]) if 'color' in x or 'rgb_error' in x]
                    yy,xx=np.nonzero(mask);foot={'status':'not_run'}
                    if len(yy)>=100 and yy.max()<depth.shape[0]-1:
                        tan=math.tan(math.radians(85.)/2);rx=(2*(xx+.5)/depth.shape[1]-1)*tan;ry=(1-2*(yy+.5)/depth.shape[0])*tan*depth.shape[0]/depth.shape[1]
                        surface_y=origin[1]+ry/np.sqrt(1+rx*rx+ry*ry)*depth[yy,xx]
                        gap=float(surface_y.min()-base[1]);foot={'lowest_y_m':float(surface_y.min()),'support_floor_y_m':float(base[1]),'gap_m':gap,
                            'status':'pass' if abs(gap)<=.02 else 'fail','raster_depth_tolerance_m':.02}
                    prefix=args.output/f'{aid}__view{vi}'
                    lit_path=str(prefix)+'.jpg';base_path=str(prefix)+'_basecolor.jpg'
                    cv2.imwrite(lit_path,cv2.resize(bgr,(1280,720)),[cv2.IMWRITE_JPEG_QUALITY,92]);cv2.imwrite(base_path,cv2.resize(base_bgr,(1280,720)),[cv2.IMWRITE_JPEG_QUALITY,92])
                    row['views'].append({'camera':camera_view,'action_id':action,'action_phase':state['action_phase'],'lit_path':lit_path,'basecolor_path':base_path,
                        'appearance_metric':metric,'native_albedo_palette_color_status':'fail' if color_fail else 'pass','native_albedo_palette_metric':albedo,'native_body_contact':foot})
                row['status']='pass' if all(v['appearance_metric']['status']=='pass' and v['native_albedo_palette_color_status']=='pass' for v in row['views']) else 'fail'
            except Exception as error:row.update(status='fail',error=f'{type(error).__name__}: {error}')
            finally:
                if runtimes:
                    destroy(instance,runtimes)
                    runtimes={}
            row['elapsed_seconds']=time.monotonic()-started;result['records'].append(row);write(args.output/'live_receipt.json',result)
            print(aid,row['status'],row.get('error'),row['elapsed_seconds'],flush=True)
        result.update(status='completed',asset_count=len(result['records']),appearance_pass_count=sum(r['status']=='pass' for r in result['records']))
    finally:
        if instance is not None:close_native(instance,camera,components)
        result['elapsed_seconds']=time.monotonic()-start;write(args.output/'result.json',result)


if __name__=='__main__':main()
