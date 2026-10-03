"""Native UE/SPEAR adapter for the existing v7 continuous candidate pipeline.

The shared generator consumes measured UE pixels and emitter-component poses.
This adapter never changes acoustic geometry, events, questions or admission
blockers. Registered v6 UE assets and animation driving are reused directly.
"""
from __future__ import annotations
from copy import deepcopy
import json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from avengine.timeline.continuous_scene import FPS, camera_basis, position_at
from avengine.timeline.continuous_pilot import write_json
from avengine.capture.kujiale_spear_runtime import (
    launch_native, camera_helpers, apply_camera, close_native, to_ue, from_ue,
    spawn_fill_lights,apply_native_rgb_profile,
)


class UENativeContinuousVideo:
    def __init__(self, house, scene, inputs, assets, *, width=1280, height=720,
                 source_material_cache_directory=None):
        import cv2
        from avengine.acoustics.runtime import load_habitat_runtime
        from avengine.runtime_profiles import load_default_source_asset_runtime_registry,resolve_source_asset_runtime_profile
        from avengine.capture.qa_plan_adapters import materialize_ue_episode_plan
        self.width,self.height,self.cv2=width,height,cv2
        # Current shared v7 callers pass a source cache directory. Native UE
        # consumes the isolated repaired MICs, rather than importing Habitat
        # GLB copies during capture; retain this input in the native receipt.
        self.source_material_cache_directory=source_material_cache_directory
        self.ue=deepcopy(house['input']['ue_config'])
        sys.path.insert(0,self.ue['spear_ext_dir'])
        from avengine.backends.spear_ue.research_runtime import apply_capture_exposure,attach_emitter_component
        self.instance=self.camera=self.components=None
        self.foot_camera=self.foot_components=None
        self.runtimes={}
        self.house,self.scene,self.assets=house,scene,assets
        hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
        self.pf=hs.PathFinder()
        if not self.pf.load_nav_mesh(house['input']['navmesh_source']):
            raise ValueError('CPU navigation input is missing')
        self.floor_offset=float(house['room']['floor_reference']['native_ue_floor_offset_m'])
        inventory=json.loads(Path(self.ue['source_material_inventory']).read_text())
        self.material_inventory={a['asset_id']:a for a in inventory['assets']}
        self.acceptance=json.loads(Path(self.ue['acceptance_plan_path']).read_text())
        registry=load_default_source_asset_runtime_registry()
        self.asset_profiles={a['asset_id']:resolve_source_asset_runtime_profile(registry,a['asset_id']) for a in assets}
        f,r,u=camera_basis(scene['camera_forward'])
        neutral_camera={'position_m':scene['listener_m'],
                        'basis':{'forward':f.tolist(),'right':r.tolist(),'up':u.tolist()},'hfov_deg':85.}
        neutral_actors=[];initial_states=[]
        for i,asset in enumerate(assets):
            native_id=f'source{i+1}'
            neutral_actors.append({'actor_id':native_id,'asset_id':asset['asset_id'],
                                   'entity_class':self.asset_profiles[asset['asset_id']]['entity_class']})
            state=self._state(asset,native_id,0)
            initial_states.append(state)
        neutral={'plan_coordinates':'renderer_neutral',
            'resources':{'room_package':{'visual_scene':{'map_path':house['input']['ue_map_path']}}},
            'scene':{},'visual_plan':{'actors':neutral_actors,'camera':neutral_camera,
                'frames':[{'frame_index':0,'camera_state':neutral_camera,'actor_states':initial_states}]}}
        driven=materialize_ue_episode_plan(neutral,registry)['visual_plan']
        self.declarations=driven['actors']
        self.initial=driven
        helpers=camera_helpers()
        self.spawn,self.read_depth,self.read_rgb,self.apply_old,self.actor_read,self.spawn_actors,self.apply_actor,self.destroy=helpers
        start=time.perf_counter()
        try:
            self.instance,self.game,launch=launch_native(self.ue,house['input']['ue_map_path'])
            with self.instance.begin_frame():
                self.camera,self.components=self.spawn(self.game,horizontal_fov_deg=85.,width=width*2,height=height*2)
                self.foot_camera,self.foot_components=self.spawn(self.game,horizontal_fov_deg=85.,width=1024,height=1024)
                for name in ('rgb','object_ids'):
                    self.foot_components[name].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
                apply_camera(self.camera,self._camera())
                lights=spawn_fill_lights(self.game,self.acceptance)
                self.original_capture_source=self.components['rgb'].get_property_value(property_name='CaptureSource')
                target=self.components['rgb'].get_property_value(property_name='TextureTarget',as_unreal_object=True)
                self.rgb_target_properties={name:target.get_property_value(property_name=name) for name in ['RenderTargetFormat','TargetGamma','bForceLinearGamma','SRGB']}
                exposure=apply_native_rgb_profile(self.components['rgb'],self.ue)
                self.runtimes=self.spawn_actors(self.game,{'plan':driven})
                for declaration in self.declarations:
                    runtime=self.runtimes[declaration['actor_id']]
                    if runtime.get('emitter_component') is None:
                        # The retained v6 registered mouth point is a native
                        # child component, not the planner's speaking centre.
                        runtime['emitter_component']=attach_emitter_component(self.game,
                            actor_id=declaration['actor_id'],anchor_root=runtime['anchor_root'],
                            emitter_local_ue_cm=declaration['emitter_local_ue_cm'])
            with self.instance.end_frame():pass
            for _ in range(80):
                with self.instance.begin_frame():pass
                with self.instance.end_frame():self.read_rgb(self.components['rgb'])
            self.receipt={'backend':'UE/SPEAR native','compute_device':'GPU',
                'graphics_adapter':self.ue['graphics_adapter'],'launch':launch,
                'capture_resolution_hw':[height*2,width*2],'resolution_hw':[height,width],
                'hfov_deg':85.,'fps':FPS,'exposure':exposure,'lights':lights,'rgb_target_properties':self.rgb_target_properties,
                'startup_and_warmup_seconds':time.perf_counter()-start,
                'canonical_to_ue':'[x,y,z] m -> [100x,100z,100y] cm',
                'registered_asset_ids':[d['asset_id'] for d in self.declarations],
                'emitter_authority':'native component attached using retained v6 source_asset_runtime_registry mouth binding',
                'animated_anatomical_mouth_contact_admission':'not_verified; shared formal blockers remain',
                'floor_reference':house['room']['floor_reference']}
        except BaseException:
            self.close()
            raise

    def _camera(self):
        forward=self.scene['camera_forward']
        return {'position_m':self.scene['listener_m'],'kind':'perspective',
            'yaw_deg':math.degrees(math.atan2(forward[2],forward[0])),
            'pitch_deg':math.degrees(math.atan2(forward[1],math.hypot(forward[0],forward[2])))}

    def _state(self,asset,native_id,index):
        t=index/FPS;track=self.scene['sources'][asset['endpoint_id']]['trajectory']
        p=position_at(track,t)
        foot=self.pf.snap_point(np.asarray([p[0],p[1]-1.2,p[2]],dtype=np.float32))
        if not np.isfinite(foot).all() or np.linalg.norm(np.asarray(foot)-[p[0],p[1]-1.2,p[2]])>.15:
            raise ValueError('UE actor root left admitted native navigation')
        root=np.asarray(foot,float);root[1]+=self.floor_offset
        delta=position_at(track,min(self.scene['duration_s'],t+1/FPS))-position_at(track,max(0.,t-1/FPS))
        delta[1]=0.;moving=bool(np.linalg.norm(delta)>1e-5)
        direction=delta if moving else np.asarray(self.scene['listener_m'])-p
        direction[1]=0.;direction/=np.linalg.norm(direction)
        # Registered neutral anatomical forward is +Z. The shared UE adapter
        # converts this canonical quaternion and applies the asset-frame delta.
        yaw=math.atan2(direction[0],direction[2])
        period=int(self.asset_profiles[asset['asset_id']]['timeline']['walk_phase_period_frames'])
        pose_seed_tick=int(self.scene.get('actor_pose_phase_ticks',{}).get(asset['endpoint_id'],0))
        phase=((index+pose_seed_tick)%period)/period if moving else 0.
        return {'actor_id':native_id,'root_transform':{'translation_m':root.tolist(),
                'rotation_xyzw':[0.,math.sin(yaw/2),0.,math.cos(yaw/2)]},
                'action_id':'walk' if moving else 'idle','action_phase':phase,'moving':moving}

    def _apply(self,index):
        from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
        rows=[]
        for asset,declaration in zip(self.assets,self.declarations,strict=True):
            state=self._state(asset,declaration['actor_id'],index)
            q=state['root_transform']['rotation_xyzw']
            state.update(translation_ue_cm=to_ue(state['root_transform']['translation_m']),
                actor_yaw_ue_deg=-math.degrees(2*math.atan2(q[1],q[3])),
                ue_animation=declaration['animation_paths_by_action_id'][state['action_id']])
            runtime=self.runtimes[declaration['actor_id']]
            observed,animation=self.apply_actor(runtime,state,index)
            pose=read_scene_component_pose(runtime['anchor_root'])
            emitter=read_scene_component_pose(runtime['emitter_component'])
            position=from_ue(pose['location_cm']);mouth=from_ue(emitter['location_cm'])
            error=float(np.max(np.abs(np.asarray(position)-state['root_transform']['translation_m'])))
            if error>.0005:raise ValueError('native UE root differs from applied registered state')
            rows.append({'endpoint_id':asset['endpoint_id'],'actor_id':self.scene['sources'][asset['endpoint_id']]['actor_id'],
                'native_actor_id':declaration['actor_id'],'asset_id':asset['asset_id'],
                'position_m':position,'emitter_position_m':mouth,'moving':state['moving'],
                'action_id':state['action_id'],'action_phase':state['action_phase'],'root_max_error':error,
                'native_animation':animation})
        return rows

    def _uncropped_foot_probe(self,actor):
        root=np.asarray(actor['position_m']);origin=root+[0.,1.,-3.]
        runtime=self.runtimes[actor['native_actor_id']]
        with self.instance.begin_frame():
            apply_camera(self.foot_camera,{'position_m':origin.tolist(),'kind':'perspective','yaw_deg':90.,'pitch_deg':0.})
            self.foot_components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
            self.foot_components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[runtime['visual_actor']])
        with self.instance.end_frame():target=self.read_depth(self.foot_components['depth'])
        yy,xx=np.nonzero(np.isfinite(target)&(target>0)&(target<65504))
        if len(yy)<100 or yy.max()>=target.shape[0]-1:
            return {'status':'not_run','reason':'independent_native_foot_probe_cropped_or_no_depth'}
        tan=math.tan(math.radians(85.)/2)
        rx=(2*(xx+.5)/target.shape[1]-1)*tan
        ry=(1-2*(yy+.5)/target.shape[0])*tan*target.shape[0]/target.shape[1]
        y=origin[1]+ry/np.sqrt(1+rx*rx+ry*ry)*target[yy,xx]
        lowest=float(y.min());gap=lowest-root[1]
        return {'method':'independent full-body native UE target-only ray depth, 1024x1024',
            'camera_position_m':origin.tolist(),'lowest_rendered_body_y_m':lowest,
            'support_floor_y_m':float(root[1]),'gap_m':float(gap),'raster_depth_tolerance_m':.02,
            'status':'pass' if abs(gap)<=.02 else 'fail'}

    def encode(self,scene,path,review_frames=None,*,collect_pixel_visibility=True):
        path=Path(path);start=time.perf_counter();frames=[];mouths={a['endpoint_id']:[] for a in self.assets}
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{self.width}x{self.height}','-framerate',str(FPS),'-i','pipe:0','-an',
            '-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(path)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        material_checks=[]
        reviews=set(review_frames or [0,45,75,90,120,149,225,300,375,449])
        visibility_truth=None;target_pose_max_error=0.;target_animation_max_error=0.;camera_position_max_error=0.
        try:
            for index in range(round(scene['duration_s']*FPS)):
                with self.instance.begin_frame():
                    actors=self._apply(index)
                    self.components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_RenderScenePrimitives')
                with self.instance.end_frame():
                    bgr=self.read_rgb(self.components['rgb']);normal=self.read_depth(self.components['depth'])
                semantic=np.zeros(normal.shape,dtype=np.uint16)
                targets={};ids={};pose_id=None
                for j,actor in enumerate(actors):
                    runtime=self.runtimes[actor['native_actor_id']]
                    with self.instance.begin_frame():
                        from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
                        import hashlib
                        camera_pose=read_scene_component_pose(self.camera.K2_GetRootComponent())
                        current_id=hashlib.sha256(json.dumps(camera_pose,sort_keys=True).encode()).hexdigest()
                        if pose_id is not None and current_id!=pose_id:raise ValueError('native modal/target-only camera pose changed')
                        pose_id=current_id
                        camera_error=float(np.max(np.abs(np.asarray(from_ue(camera_pose['location_cm']))-scene['listener_m'])))
                        camera_position_max_error=max(camera_position_max_error,camera_error)
                        if camera_error>.0005:raise ValueError('native target camera left listener coordinates')
                        root_pose=read_scene_component_pose(runtime['anchor_root'])
                        root_error=float(np.max(np.abs(np.asarray(from_ue(root_pose['location_cm']))-actor['position_m'])))
                        target_pose_max_error=max(target_pose_max_error,root_error)
                        if root_error>.0005:raise ValueError('native target-only actor root changed after modal capture')
                        if runtime.get('motion_model')!='rigid_static':
                            animation_error=abs(float(runtime['component'].GetPosition())-actor['native_animation']['observed_position_seconds'])
                            target_animation_max_error=max(target_animation_max_error,animation_error)
                            if animation_error>1e-4:raise ValueError('native target-only animation changed after modal capture')
                        self.components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
                        self.components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[runtime['visual_actor']])
                    with self.instance.end_frame():target=self.read_depth(self.components['depth'])
                    targets[str(actor['actor_id'])]=[np.ascontiguousarray(target[::2,::2])]
                    ids[str(actor['actor_id'])]=1001+j
                    visible=(target<65504)&(np.abs(normal-target)<=.01+.002*target)
                    # Reconstruct actual visible native body surfaces; this is
                    # a diagnostic contact measurement, not a bone/root proxy.
                    ty,tx=np.nonzero(target<65504)
                    if len(ty) and ty.max()<target.shape[0]-1:
                        tan=math.tan(math.radians(85.)/2)
                        rx=(2*(tx+.5)/target.shape[1]-1)*tan
                        ry=(1-2*(ty+.5)/target.shape[0])*tan*target.shape[0]/target.shape[1]
                        camera=self._camera()
                        pitch=math.radians(camera['pitch_deg'])
                        # Camera forward/up in canonical world coordinates.
                        # UE camera roll is zero; right.y is zero.
                        direction_y=(math.sin(pitch)+ry*math.cos(pitch))/np.sqrt(1+rx*rx+ry*ry)
                        surface_y=scene['listener_m'][1]+direction_y*target[ty,tx]
                        native_foot=float(surface_y.min())
                        floor_y=actor['position_m'][1]
                        actor['native_foot_contact']={'method':'native target-only ray depth reconstructed in world coordinates',
                            'lowest_rendered_body_y_m':native_foot,'support_floor_y_m':floor_y,
                            'gap_m':native_foot-floor_y,
                            'status':'pass' if abs(native_foot-floor_y)<=.02 else 'fail',
                            'raster_depth_tolerance_m':.02}
                    else:actor['native_foot_contact']=self._uncropped_foot_probe(actor)
                    semantic[visible]=1001+j
                    small_mask=self.cv2.resize(visible.astype('uint8'),(self.width,self.height),interpolation=self.cv2.INTER_NEAREST)>0
                    ys,xs=np.nonzero(small_mask)
                    actor.update(semantic_id=1001+j,visible_pixels=len(xs),
                        mask_centroid_px=[float(xs.mean()),float(ys.mean())] if len(xs) else None,
                        mask_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None)
                    mouths[actor['endpoint_id']].append({'time_s':index/FPS,'position_m':actor['emitter_position_m']})
                if collect_pixel_visibility:
                    from avengine.qa.pixel_visibility import compile_depth_pixel_visibility_truth
                    common={'renderer_backend':'ue_spear','rgb_renderer_backend':'ue_spear',
                        'camera_contract_id':'v7_native_camera_listener','semantic_id_namespace':'ue_native_depth_target_only_instance_v1',
                        'resolution_hw':[self.height,self.width],'frame_indices':[index],'camera_pose_ids':[pose_id]}
                    truth=compile_depth_pixel_visibility_truth(normal_depth_m_frames=[np.ascontiguousarray(normal[::2,::2])],
                        target_only_depth_m_frames_by_instance=targets,semantic_ids_by_instance=ids,
                        normal_context={'pass_kind':'modal_scene',**common},
                        target_only_contexts_by_instance={aid:{'pass_kind':'target_only','target_instance_id':aid,**common} for aid in ids},
                        target_only_background_depth_m=65504.,absolute_tolerance_m=.01,relative_tolerance=.002)
                    if visibility_truth is None:visibility_truth=truth
                    else:
                        visibility_truth['frame_indices'].extend(truth['frame_indices']);visibility_truth['camera_pose_ids'].extend(truth['camera_pose_ids'])
                        for aid,value in truth['per_instance'].items():
                            old=visibility_truth['per_instance'][aid];old['frames'].extend(value['frames'])
                            for key,count in value['state_counts'].items():old['state_counts'][key]+=count
                if index in reviews:
                    from avengine.assets.source_materials import measure_source_appearance
                    with self.instance.begin_frame():
                        self.components['rgb'].set_property_value(property_name='CaptureSource',property_value='SCS_BaseColor')
                    with self.instance.end_frame():base_bgr=self.read_rgb(self.components['rgb'])
                    with self.instance.begin_frame():
                        self.components['rgb'].set_property_value(property_name='CaptureSource',property_value=self.original_capture_source)
                    with self.instance.end_frame():pass
                    reference=self.cv2.cvtColor(base_bgr,self.cv2.COLOR_BGR2RGB)
                    self.cv2.imwrite(str(path.parent/f'native_basecolor_{index:04d}.png'),base_bgr)
                    rendered=self.cv2.cvtColor(bgr,self.cv2.COLOR_BGR2RGB)
                    for j,actor in enumerate(actors):
                        declaration=self.material_inventory.get(actor['asset_id'])
                        if declaration is None:raise ValueError('source material receipt is missing')
                        materials=declaration['material_receipt'];profiles=materials['profiles']
                        metric=measure_source_appearance(rendered,semantic==1001+j,
                            palette=np.asarray(materials['source_palette_srgb']),
                            max_highlight_ratio=min(p['max_highlight_ratio'] for p in profiles),
                            reference_rgb=reference,
                            metallic_surface=all(p['metallic']>0.5 for p in profiles),strict=False)
                        material_checks.append({'frame_index':index,'asset_id':actor['asset_id'],'metrics':metric,
                            'reference_kind':'same native UE BaseColor view; original UV palette retained',
                            'native_foot_contact':actor['native_foot_contact']})
                rgb=self.cv2.cvtColor(self.cv2.resize(bgr,(self.width,self.height),interpolation=self.cv2.INTER_AREA),self.cv2.COLOR_BGR2RGB)
                if float(rgb.std())<2:raise ValueError('native UE scene is empty')
                frames.append({'frame_index':index,'time_s':index/FPS,'actors':actors,'rgb_std':float(rgb.std())})
                if index in reviews:
                    self.cv2.imwrite(str(path.parent/f'native_frame_{index:04d}.png'),self.cv2.cvtColor(rgb,self.cv2.COLOR_RGB2BGR))
                    np.savez_compressed(path.parent/f'native_masks_{index:04d}.npz',
                        semantic=self.cv2.resize(semantic,(self.width,self.height),interpolation=self.cv2.INTER_NEAREST),
                        depth=self.cv2.resize(normal,(self.width,self.height),interpolation=self.cv2.INTER_NEAREST))
                process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('native UE frame stream encoding failed')
        except BaseException:
            if process.stdin and not process.stdin.closed:process.stdin.close()
            process.wait()
            raise
        for values in mouths.values():values.append({'time_s':scene['duration_s'],'position_m':values[-1]['position_m']})
        result={'renderer':self.receipt,'frames':frames,'observed_mouth_trajectories':mouths,
                'frame_count':len(frames),'wall_seconds':time.perf_counter()-start,
                'native_source_material_checks':material_checks,
                'native_source_material_status':'pass' if material_checks and all(x['metrics']['status']=='pass' for x in material_checks) else 'fail',
                'native_body_foot_status':'pass' if frames and all(a['native_foot_contact']['status']=='pass' for f in frames for a in f['actors']) else 'fail_or_unverified'}
        source_materials={'schema':'kujiale_native_source_material_bindings_v1','render_path':'ue_spear',
            'native_project':self.ue['uproject'],'source_material_inventory':self.ue['source_material_inventory'],
            'shared_v7_cache_directory_input':str(self.source_material_cache_directory) if self.source_material_cache_directory else None,
            'native_material_authority':'isolated UE asset bindings with applied reviewed PBR MIC repair; no Habitat GLB substitution',
            'assets':[{'asset_id':a['asset_id'],'native_declaration':d,'source_material_receipt':self.material_inventory[a['asset_id']]['material_receipt']}
                      for a,d in zip(self.assets,self.declarations,strict=True)]}
        material_receipt={'status':result['native_source_material_status'],'checks':material_checks}
        visibility={'pixel_visibility_truth':visibility_truth if visibility_truth is not None else {'status':'not_run_unqualified_native_stream'},
            'alignment':{'status':'pass' if visibility_truth is not None else 'not_run','mode':'same_UE_scene_per_instance_native_depth_rerender',
                'renderer_backend':'ue_spear','published_RGB_scene':self.house['input']['ue_map_path'],
                'rendered_native_target_depth_passes':len(frames)*len(self.assets),
                'root_max_error_m':max((a['root_max_error'] for f in frames for a in f['actors']),default=0.),
                'target_only_root_max_error_m':target_pose_max_error,'target_only_animation_max_error_s':target_animation_max_error,
                'camera_position_max_error_m':camera_position_max_error,
                'occlusion_criterion':'retained v6 depth tolerance 0.01m + 0.002 * target depth',
                'full_stream_qualification':'existing v6 compile_depth_pixel_visibility_truth reused; full M2 motion/contact and v7 formal blockers remain',
                'capture_resolution_hw':[self.height*2,self.width*2],'depth_delivery_sampling':'nearest sample [::2,::2], same as native mask delivery'}}
        write_json(path.parent/'source_materials.json',source_materials)
        write_json(path.parent/'source_material_checks.json',material_receipt)
        write_json(path.parent/'native_visibility_evidence.json',visibility)
        result.update(source_material_checks=material_receipt,pixel_visibility_truth=visibility['pixel_visibility_truth'],
                      native_visibility_alignment=visibility['alignment'])
        write_json(path.parent/'native_visual_evidence.json',result)
        return result

    def close(self):
        if self.instance is not None:
            if self.runtimes:
                try:
                    self.destroy(self.instance,self.runtimes)
                except Exception:pass
                self.runtimes={}
            if self.foot_camera is not None:
                from avengine.backends.spear_ue.research_runtime import close_scene_capture
                for component in self.foot_components.values():
                    try:close_scene_capture(instance=self.instance,game=self.game,camera=None,capture=component)
                    except Exception:pass
                try:
                    with self.instance.begin_frame():self.foot_camera.K2_DestroyActor()
                    with self.instance.end_frame():pass
                except Exception:pass
            close_native(self.instance,self.camera,self.components)
            self.instance=None
