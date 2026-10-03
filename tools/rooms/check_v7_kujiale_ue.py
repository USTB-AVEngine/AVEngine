#!/usr/bin/env python3
"""Native UE room images, measured depth floor and exact exported geometry QA."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def geometry_parity(plan,audit):
    from scipy.spatial import cKDTree
    source=np.load(Path(plan['build_directory'])/'surface/vertices.npy',allow_pickle=False)
    geometry=audit['native_render_geometry']
    vertices=np.fromfile(geometry['world_vertices_ue_cm_f32'],dtype='<f4').reshape(-1,3)
    faces=np.fromfile(geometry['triangles_u32'],dtype='<u4').reshape(-1,3)
    if not len(vertices) or not len(faces):raise ValueError('native render geometry is empty')
    native=vertices[:,[0,2,1]]/100
    # UV/material seams duplicate render vertices. Compare unchanged point
    # sets once; exact de-duplication changes neither the metric nor tolerance.
    source_points=np.unique(source,axis=0)
    native_points=np.unique(native,axis=0)
    source_tree=cKDTree(source_points)
    native_tree=cKDTree(native_points)
    def bounded_distances(tree,points):
        distance=tree.query(points,workers=1,distance_upper_bound=.000500000001)[0]
        failed=np.flatnonzero(~np.isfinite(distance))
        if len(failed):
            # An exact bounded query already proves non-parity. Measure only
            # a labeled failure sample instead of an unbounded all-point scan.
            sample=points[failed[np.linspace(0,len(failed)-1,min(512,len(failed)),dtype=int)]]
            measured=tree.query(sample,workers=1)[0]
            return float(measured.max()),len(failed),False
        return float(distance.max()),0,True
    a,failed_native,complete_a=bounded_distances(source_tree,native_points)
    b,failed_source,complete_b=bounded_distances(native_tree,source_points)
    result={'status':'pass' if not (failed_native or failed_source) and max(a,b)<=.0005 else 'fail','source_vertex_count':len(source),
            'native_vertex_count':len(native),'native_face_count':len(faces),
            'source_unique_vertex_count':len(source_points),'native_unique_vertex_count':len(native_points),
            'max_native_to_source_vertex_error_m':a,'max_source_to_native_vertex_error_m':b,
            'max_vertex_error_m':max(a,b),'max_error_complete':complete_a and complete_b,
            'failure_sample_is_lower_bound':not (complete_a and complete_b),
            'native_unique_vertices_above_tolerance':failed_native,'source_unique_vertices_above_tolerance':failed_source,
            'tolerance_m':.0005,
            'native_export':geometry,'native_decode':'UE FVector cm -> canonical [X,Z,Y]/100',
            'alignment_fitting_used':False}
    return result


def view_quality(rgb,depth,view,room,params):
    from shapely.geometry import shape
    import shapely
    height,width=rgb.shape[:2]
    roi=np.ones((height,width),bool)
    if view.get('kind')=='orthographic':
        width_m=view['ortho_width_cm']/100
        x=view['position_m'][0]+((np.arange(width)+.5)/width-.5)*width_m
        z=view['position_m'][2]+((np.arange(height)+.5)/height-.5)*width_m*height/width
        xx,zz=np.meshgrid(x,z)
        roi=shapely.contains_xy(shape(room['floor_polygon']),xx,zz)
    n=int(roi.sum())
    if not n:raise ValueError('native review ROI is empty')
    pixels=rgb[roi]
    black=np.all(pixels<params['black_pixel_channel_lt'],axis=1)
    white=np.all(pixels>247,axis=1)
    grey=(np.ptp(pixels.astype(np.int16),axis=1)<=3)&(pixels.mean(1)>100)&(pixels.mean(1)<160)
    magenta=(pixels[:,0]>200)&(pixels[:,2]>200)&(pixels[:,1]<75)
    native_depth=depth[roi]
    empty=(~np.isfinite(native_depth))|(native_depth>=65500)|(native_depth<=0)
    ratio=lambda value:float(np.mean(value))
    values={'black_fraction':ratio(black),'overexposed_fraction':ratio(white),
            'default_grey_color_fraction':ratio(grey),'missing_texture_magenta_fraction':ratio(magenta),
            'native_no_surface_depth_fraction':ratio(empty),'rgb_std':float(pixels.std()),
            'roi_pixels':n,'roi_definition':'semantic floor polygon' if view.get('kind')=='orthographic' else 'whole native perspective image',
            'all_black':bool(black.all()),'all_white':bool(white.all()),'native_depth_median_m':float(np.nanmedian(native_depth))}
    failures=[]
    if values['black_fraction']>params['black_fraction_max']:failures.append('BLACK_FRACTION_ABOVE_FROZEN_015')
    if values['overexposed_fraction']>.15:failures.append('OVEREXPOSED_FRACTION_ABOVE_015')
    if values['missing_texture_magenta_fraction']>.15:failures.append('MISSING_TEXTURE_MAGENTA_ABOVE_015')
    if values['all_black'] or values['all_white']:failures.append('ALL_BLACK_OR_WHITE')
    # Exterior windows legitimately see no depth. Missing faces additionally
    # require a expected-geometry/native-depth comparison; this statistic alone
    # cannot pass that check or label windows as scan holes.
    values.update(status='fail' if failures else 'pixel_checks_pass',failures=failures,
                  grey_is_diagnostic_requires_native_material_identity=True,
                  missing_faces_geometry_crosscheck='required')
    return values,roi


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--editor-audit',type=Path,required=True)
    parser.add_argument('--ue-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit-rooms',type=int)
    args=parser.parse_args()
    import sys
    ue=json.loads(args.ue_config.read_text());sys.path.insert(0,ue['spear_ext_dir'])
    import cv2
    from avengine.capture.kujiale_spear_runtime import launch_native,camera_helpers,capture_view,close_native,spawn_fill_lights,to_ue,apply_native_rgb_profile
    from avengine.backends.spear_ue.research_runtime import apply_capture_exposure,read_scene_component_pose
    # Extension must precede the first optional client import.
    plan=json.loads(args.plan.read_text());audit=json.loads(args.editor_audit.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    parity=geometry_parity(plan,audit);write(args.output/'geometry_parity.json',parity)
    if parity['status']!='pass':raise RuntimeError('native/source vertices do not align')
    rooms=plan['rooms'][:args.limit_rooms] if args.limit_rooms else plan['rooms']
    receipt={'schema':'kujiale_native_ue_room_acceptance_v1','house':plan['house'],'rooms':[],
             'geometry_parity':parity,'status':'started','render_path':'ue_spear','plan':str(args.plan),
             'capture_resolution_hw':[1440,2560],'delivery_resolution_hw':[720,1280]}
    instance=camera=components=None
    try:
        instance,game,launch=launch_native(ue,plan['map_path'])
        receipt['launch']=launch
        spawn,*_=camera_helpers()
        with instance.begin_frame():
            camera,components=spawn(game,horizontal_fov_deg=85.,width=2560,height=1440)
            receipt['lights']=spawn_fill_lights(game,plan)
            game.get_unreal_object(uclass='UGameplayStatics').SetGamePaused(bPaused=False)
            receipt['exposure']=apply_native_rgb_profile(components['rgb'],ue)
        with instance.end_frame():pass
        instance.step(num_frames=120)
        for room in rooms:
            room_start=time.monotonic();records=[];floor_probes=[]
            tested=[];selected_witnesses=[]
            for view_original in room['views']:
                view=dict(view_original)
                if view.get('kind')=='orthographic':view['yaw_deg']=-90.
                bgr,depth=capture_view(instance,camera,components,view)
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                small=cv2.resize(rgb,(1280,720),interpolation=cv2.INTER_AREA)
                native_path=args.output/(view['view_id']+'_1440p.jpg')
                small_path=args.output/(view['view_id']+'_720p.jpg')
                cv2.imwrite(str(native_path),bgr,[cv2.IMWRITE_JPEG_QUALITY,92])
                cv2.imwrite(str(small_path),cv2.cvtColor(small,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
                np.savez_compressed(args.output/(view['view_id']+'_depth.npz'),depth_m=depth[::4,::4])
                metrics,roi=view_quality(rgb,depth,view,room,plan['frozen_parameters'])
                with instance.begin_frame():pose=read_scene_component_pose(camera.K2_GetRootComponent())
                with instance.end_frame():pass
                error=float(np.max(np.abs(np.asarray(pose['location_cm'])-to_ue(view['position_m']))))/100
                candidate={'view':view,'native_pose':pose,'native_position_error_m':error,
                                'path_1440p':str(native_path),'path_720p':str(small_path),'metrics':metrics}
                tested.append(candidate)
                if view['kind']=='orthographic':records.append(candidate)
                elif metrics['status']=='pixel_checks_pass' and error<.0005:
                    records.append(candidate)
                    witness_index=int(view['view_id'].rsplit('_',1)[1])-1
                    selected_witnesses.append((witness_index,room['witnesses'][witness_index]))
                if len(selected_witnesses)==3:break
                write(args.output/'live_receipt.json',receipt)
            for witness_index,witness in selected_witnesses:
                for role,height in [('camera',1.5),('source_1',1.2),('source_2',1.2)]:
                    nav=np.asarray(witness[role+'_m'],float)-[0,height,0]
                    position=nav+[0,.18,0]
                    view={'kind':'perspective','position_m':position.tolist(),'pitch_deg':-90.,'yaw_deg':-90.}
                    _bgr,depth=capture_view(instance,camera,components,view,warmup_frames=2)
                    h,w=depth.shape;center=depth[h//2-4:h//2+5,w//2-4:w//2+5]
                    finite=center[np.isfinite(center)&(center>0)&(center<1.)]
                    probe={'witness_index':witness_index,'role':role,'nav_point_m':nav.tolist(),'native_depth_valid_pixels':len(finite)}
                    if len(finite)<40:probe.update(status='fail',reason='LOW_CAMERA_NATIVE_FLOOR_DEPTH_MISSING')
                    else:
                        d=float(np.median(finite));floor_y=float(position[1]-d)
                        probe.update(status='measured',native_depth_m=d,native_visual_floor_y_m=floor_y,nav_to_visual_offset_m=floor_y-float(nav[1]))
                        with instance.begin_frame():
                            capsule_actor=game.unreal_service.spawn_actor(uclass='AActor',spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
                            capsule=game.unreal_service.create_scene_component_for_actor(owner=capsule_actor,scene_component_name='v7_floor_capsule',uclass='UCapsuleComponent')
                            capsule.SetCapsuleSize(InRadius=20.,InHalfHeight=75.,bUpdateOverlaps=True)
                            p=to_ue([nav[0],floor_y+.75,nav[2]])
                            capsule_actor.K2_SetActorLocation(NewLocation=dict(zip(('X','Y','Z'),p)),bSweep=False,bTeleport=True)
                            pose=read_scene_component_pose(capsule)
                            observed_half=float(capsule.GetScaledCapsuleHalfHeight())
                            gap=(pose['location_cm'][2]-observed_half)/100-floor_y
                            probe['capsule']={'center_ue_cm':pose['location_cm'],'half_height_cm':observed_half,
                                              'radius_cm':float(capsule.GetScaledCapsuleRadius()),'bottom_floor_gap_m':gap,
                                              'status':'pass' if abs(gap)<=.005 else 'fail'}
                            capsule_actor.K2_DestroyActor()
                        with instance.end_frame():pass
                    floor_probes.append(probe)
            valid=[p for p in floor_probes if p['status']=='measured']
            offsets=[p['nav_to_visual_offset_m'] for p in valid]
            floor={'room_id':room['room_id'],'method':'UE native depth at nine frozen witness support points',
                   'probes':floor_probes,'native_ue_floor_offset_m':float(np.median(offsets)) if offsets else None,
                   'status':'pass' if len(valid)==9 and all(p['capsule']['status']=='pass' for p in valid) else 'fail'}
            record={'room_id':room['room_id'],'room_type':room['room_type'],'views':records,'tested_views':tested,
                    'selected_witnesses':[{'index':i,'placement':w} for i,w in selected_witnesses],
                    'required_perspective_count':3,'floor_reference':floor,
                    'native_pose_alignment_status':'pass' if len(records)==4 and all(r['native_position_error_m']<.0005 for r in records) else 'fail',
                    'status':('review_pending_material_and_depth_geometry_checks' if len(selected_witnesses)==3 and all(v['metrics']['status']=='pixel_checks_pass' for v in records) else 'pixel_checks_fail'),'elapsed_seconds':time.monotonic()-room_start}
            receipt['rooms'].append(record)
            write(args.output/'live_receipt.json',receipt)
            write(args.output/'floor_reference.json',{'house':plan['house'],'rooms':[r['floor_reference'] for r in receipt['rooms']]})
        receipt['status']='native_pixels_and_floor_measured'
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if instance is not None:close_native(instance,camera,components)
        receipt['elapsed_seconds']=time.monotonic()-start
        write(args.output/'result.json',receipt)
    print(json.dumps({'house':plan['house'],'rooms':len(receipt['rooms']),'status':receipt['status'],'seconds':receipt['elapsed_seconds']}))


if __name__=='__main__':main()
