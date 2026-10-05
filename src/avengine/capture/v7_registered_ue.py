"""Registered v7 static people in the isolated Kujiale UE/SPEAR scene.

The official sampler, recorded audio, RLR and question pipeline are shared.
Original registered UE packages and the existing UE pixel/contact checks are
used directly; a Habitat material conversion is never applied to UE assets.
"""
from pathlib import Path
from copy import deepcopy
import json,math,sys,time
import numpy as np
from avengine.capture.v7_alpha_ue import NativeUEAlphaVideo,assert_ue_alpha_visual
from avengine.capture.kujiale_spear_runtime import (launch_native,camera_helpers,apply_camera,
    from_ue,to_ue,configure_scene_lighting,apply_native_rgb_profile)
from avengine.timeline.continuous_pilot import write_json

REGISTERED_UE_VISUAL_REVISION='registered-native-UE-design-whitebalance-v2-nav-preflight'


class NativeRegisteredUEVideo(NativeUEAlphaVideo):
    def __init__(self,house,scene,inputs,assets,*,width=1280,height=720,source_material_cache_directory=None):
        import cv2
        from avengine.capture.kujiale_native_exposure import (bind_histogram_profile,
            require_authored_light_audit,require_histogram_room_visual_qa,PARAMETERS)
        from avengine.capture.native_floor_reference import NativeFloorReference
        from avengine.acoustics.runtime import load_habitat_runtime
        if house.get('family')!='kujiale' or inputs.get('render_backend')!='ue_spear':
            raise ValueError('registered UE adapter requires admitted Kujiale native inputs')
        for a in assets:
            if not a.get('standing_registered_qualified') or a.get('allowed_pose_for_this_task')!='stand':
                raise ValueError('registered static authority missing')
            if not a.get('ue_generated_class') or not a.get('registered_native_ue'):
                raise ValueError('registered UE class/material source missing')
        self.house,self.scene,self.assets=deepcopy(house),scene,assets
        self.width,self.height,self.cv2=width,height,cv2
        self.source_material_cache_directory=source_material_cache_directory
        self.ue=bind_histogram_profile(self.house['input']['ue_config'],scene_seed=scene['scene_id'])
        # The camera helper imports the native client; resolve the task-owned
        # extension before that import can cache an unavailable client.
        sys.path.insert(0,str(Path(self.ue['spear_ext_dir']).resolve()))
        self.ue['allow_invisible_source_material_samples']=True
        self.ue['source_material_inventory']=str(Path(inputs['native_registered_content_receipt']).parent.parent/'inputs/assets.json')
        self.ue['source_material_inventory']=str(Path(inputs['native_registered_content_receipt']).parent.parent/'inputs/assets.json')
        self.house['input']['ue_config']=self.ue
        require_authored_light_audit(self.ue,self.house['input']['ue_map_path'])
        require_histogram_room_visual_qa(self.ue,self.house['room']['room_label'],self.house['input']['ue_map_path'])
        content=json.loads(Path(inputs['native_registered_content_receipt']).read_text())
        if content['status']!='pass':raise ValueError('registered content copy not complete')
        self.acceptance=json.loads(Path(self.ue['acceptance_plan_path']).read_text())
        self.native_floor_surface=NativeFloorReference(self.ue['native_floor_surface'])
        self.native_floor_surface_check=self.native_floor_surface.validate_probes(self.house['room']['floor_reference'])
        hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
        self.pf=hs.PathFinder()
        if not self.pf.load_nav_mesh(self.house['input']['navmesh_source']):raise ValueError('native navigation missing')
        self.instance=self.camera=self.components=self.foot_camera=self.foot_components=None
        self.runtimes={};self.environment_nodes={};self.environment_frame_readbacks=[]
        self.placement_overrides={};self.alpha_time=0.;self.alpha_lower_endpoint=None;self.alpha_camera_rows={}
        self.initial_placement_preflight=self._seed_initial_navigation_placements()
        self.declarations=[{'actor_id':f'source{i+1}','asset_id':a['asset_id'],'ue_generated_class':a['ue_generated_class']} for i,a in enumerate(assets)]
        self.material_inventory={a['asset_id']:{'material_receipt':{'authority':'original registered UE package materials',
            'source_registry':a['source_registry'],'source_registry_sha256':a['source_registry_sha256'],
            'profiles':[{'max_highlight_ratio':.08}],'no_Habitat_PBR_overrides':True}} for a in assets}
        self.spawn,self.read_depth,self.read_rgb,_old,_actor,_spawn,_apply,_destroy=camera_helpers()
        self.destroy=self._destroy_registered
        start=time.perf_counter()
        try:
            self.instance,self.game,launch=launch_native(self.ue,self.house['input']['ue_map_path'])
            with self.instance.begin_frame():
                self.camera,self.components=self.spawn(self.game,horizontal_fov_deg=85.,width=width*2,height=height*2)
                self.foot_camera,self.foot_components=self.spawn(self.game,horizontal_fov_deg=85.,width=1024,height=1024)
                for name in ['rgb','object_ids']:self.foot_components[name].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
                apply_camera(self.camera,self._camera())
                lights=configure_scene_lighting(self.game,self.acceptance,self.ue)
                self.original_capture_source=self.components['rgb'].get_property_value(property_name='CaptureSource')
                exposure=apply_native_rgb_profile(self.components['rgb'],self.ue)
                for a,d in zip(assets,self.declarations,strict=True):self.runtimes[d['actor_id']]=self._spawn_registered(a,d)
                for ep,spec in scene['sources'].items():
                    if spec.get('native_visual_kind')=='audio_only':
                        actor=self.game.unreal_service.spawn_actor(uclass='AActor',spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
                        root=self.game.unreal_service.create_scene_component_for_actor(owner=actor,scene_component_name=ep+'_audio_only',uclass='USceneComponent')
                        root.SetMobility(NewMobility='Movable');actor.SetActorEnableCollision(bNewActorEnableCollision=False)
                        self.environment_nodes[ep]={'actor':actor,'root':root,'point':spec['trajectory'][0]['position_m'],'source_id':spec['actor_id']}
                self._apply(0)
            with self.instance.end_frame():pass
            for _ in range(PARAMETERS['warmup_frames']):
                with self.instance.begin_frame():pass
                with self.instance.end_frame():self.read_rgb(self.components['rgb'])
            self.receipt={'backend':'UE/SPEAR registered native standing','graphics_adapter':self.ue['graphics_adapter'],
                'compute_device':'GPU','launch':launch,'exposure':exposure,'lights':lights,'hfov_deg':85.,'fps':15,
                'capture_resolution_hw':[height*2,width*2],'resolution_hw':[height,width],
                'startup_and_warmup_seconds':time.perf_counter()-start,'original_materials':True,
                'source_material_overrides_applied':False,'source_extra_fill_applied':False,
                'initial_placement_preflight':self.initial_placement_preflight,
                'registered_asset_ids':[a['asset_id'] for a in assets],'native_floor_surface_validation':self.native_floor_surface_check,
                'registered_content_copy_receipt':inputs['native_registered_content_receipt'],
                'action_scope':'actual original registered static Blueprint; no walk/sit substitution'}
        except BaseException:self.close();raise

    def _spawn_registered(self,a,d):
        from avengine.backends.spear_ue.research_runtime import attach_emitter_component
        klass=self.game.unreal_service.load_class(uclass='AActor',name=a['ue_generated_class'])
        actor=self.game.unreal_service.spawn_actor(uclass=klass,spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
        actor.SetActorEnableCollision(bNewActorEnableCollision=False);actor.SetActorTickEnabled(bEnabled=False)
        root=actor.K2_GetRootComponent();root.SetMobility(NewMobility='Movable')
        components=self.game.unreal_service.get_components_by_class(actor=actor,uclass='UStaticMeshComponent')
        if isinstance(components,dict):components=list(components.values())
        if isinstance(components,dict):components=list(components.values())
        if not components:raise ValueError('registered Blueprint contains no native mesh')
        expected=set(a['registered_native_ue']['materials']);observed=[]
        objects=self.game.get_unreal_object(uclass='UKismetSystemLibrary')
        for c in components:
            c.SetMobility(NewMobility='Movable');c.SetCollisionEnabled(NewType='NoCollision')
            for i in range(int(c.GetNumMaterials())):
                material=c.GetMaterial(ElementIndex=i,as_unreal_object=True)
                observed.append(str(objects.GetPathName(Object=material)))
        if set(observed)!=expected:raise ValueError('registered native original material paths differ: '+str(observed)+' expected '+str(expected))
        emitter=attach_emitter_component(self.game,actor_id=d['actor_id'],anchor_root=root,emitter_local_ue_cm=a['registered_native_ue_mouth_local_cm'])
        return {'anchor':actor,'anchor_root':root,'visual_actor':actor,'component':components[0],
            'components':components,'emitter_component':emitter,'motion_model':'rigid_static',
            'original_material_readback':{'status':'pass','paths':observed,'source_registry':a['source_registry'],'material_overrides_applied':False}}

    def _navigation_foot(self,position):
        p=np.asarray(position,float)
        target=np.asarray([p[0],p[1]-1.3,p[2]],float)
        foot=np.asarray(self.pf.snap_point(target),float)
        distance=float(np.linalg.norm(foot-target))
        if not np.isfinite(foot).all() or distance>.15:raise ValueError('registered candidate off navigation')
        return foot,distance

    def _seed_initial_navigation_placements(self):
        # Native standing qualification already searches these same frozen
        # candidates. Seed the editor before its first _apply, otherwise an
        # invalid original fourth body aborts before that full check can run.
        # This preflight only uses the existing 15 cm navigation check; full
        # capsule, contact, visibility and acoustic gates run afterward.
        results=[]
        for asset in self.assets:
            ep=asset['endpoint_id'];original=self.scene['sources'][ep]['trajectory'][0]['position_m']
            for rank,p in enumerate([original,*self.scene.get('native_placement_candidates',[])]):
                try:_,distance=self._navigation_foot(p)
                except ValueError:continue
                if rank:self.placement_overrides[ep]=p
                results.append({'endpoint_id':ep,'candidate_rank':rank,'navigation_snap_delta_m':distance,
                                'unchanged_navigation_gate_m':.15,'initial_only':True})
                break
            else:raise ValueError('registered candidate off navigation')
        return results

    def _body(self,a):
        ep=a['endpoint_id'];spec=self.scene['sources'][ep]
        p=np.asarray(self.placement_overrides.get(ep,spec['trajectory'][0]['position_m']),float)
        foot,_=self._navigation_foot(p)
        root=np.asarray(self.native_floor_surface.sample(foot),float)
        if ep==self.alpha_lower_endpoint:root[1]-=.1
        direction=np.asarray(self.scene['listener_m'])-p
        yaw=math.degrees(math.atan2(direction[0],-direction[2]))
        # Registered source UE actor +90 degrees corresponds to canonical -Z.
        return root,yaw+90.

    def _apply(self,index):
        from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
        self.alpha_time=index/15;apply_camera(self.camera,self._camera());rows=[]
        for a,d in zip(self.assets,self.declarations,strict=True):
            root,yaw=self._body(a);native=self.runtimes[d['actor_id']]
            native['anchor'].K2_SetActorLocationAndRotation(NewLocation=dict(zip(['X','Y','Z'],to_ue(root))),NewRotation={'Roll':0.,'Pitch':0.,'Yaw':yaw},bSweep=False,bTeleport=True)
            pose=read_scene_component_pose(native['anchor_root']);emitter=read_scene_component_pose(native['emitter_component']);position=from_ue(pose['location_cm'])
            error=float(np.max(np.abs(np.asarray(position)-root)))
            if error>.0005:raise ValueError('registered UE root readback mismatch')
            rows.append({'endpoint_id':a['endpoint_id'],'actor_id':self.scene['sources'][a['endpoint_id']]['actor_id'],
                'native_actor_id':d['actor_id'],'native_object_id':int(native['visual_actor'].uobject),'asset_id':a['asset_id'],'position_m':position,
                'emitter_position_m':from_ue(emitter['location_cm']),'moving':False,'action_id':'registered_stand',
                'action_phase':0.,'root_max_error':error,'joint_state':'registered rigid Blueprint; no joint DOFs',
                'source_registry_sha256':a['source_registry_sha256']})
        env=[]
        for ep,native in self.environment_nodes.items():
            native['actor'].K2_SetActorLocationAndRotation(NewLocation=dict(zip(['X','Y','Z'],to_ue(native['point']))),NewRotation={'Roll':0.,'Pitch':0.,'Yaw':0.},bSweep=False,bTeleport=True)
            pose=read_scene_component_pose(native['root']);env.append({'endpoint_id':ep,'source_id':native['source_id'],
                'position_m':from_ue(pose['location_cm']),'native_component_pose':pose,'native_readback':'actual UE audio-only component; no visual geometry'})
        self.environment_frame_readbacks.append({'frame_index':index,'time_s':index/15,'emitters':env})
        self.alpha_camera_rows[index]={'frame_index':index,**self._camera_readback(index/15)}
        return rows

    def qualify_standing(self,directory):
        from tools.rooms.check_kujiale_native_capsule_contact import query_capsule
        from tools.rooms.check_v7_kujiale_native_assets import vector_field
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);checks=[];bounds=[];attempts=[]
        self.alpha_standing_refs={}
        for a,d in zip(self.assets,self.declarations,strict=True):
            accepted=None;ep=a['endpoint_id'];original=self.scene['sources'][ep]['trajectory'][0]['position_m']
            for p in [original,*self.scene.get('native_placement_candidates',[])]:
                self.placement_overrides[ep]=p
                try:
                    with self.instance.begin_frame():actor=next(x for x in self._apply(0) if x['endpoint_id']==ep)
                    with self.instance.end_frame():pass
                    root=np.asarray(actor['position_m']);probe={'witness_index':0,'role':ep,'capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':to_ue(root+[0.,.75,0.])}}
                    capsule=query_capsule(self.native_floor_surface.mesh,probe)
                    if capsule['status']!='pass':attempts.append({'endpoint_id':ep,'candidate':p,'capsule':capsule['status']});continue
                    native=self.runtimes[d['actor_id']]
                    with self.instance.begin_frame():b=native['visual_actor'].GetActorBounds(bOnlyCollidingComponents=False,bIncludeFromChildActors=True,as_dict=True)
                    with self.instance.end_frame():pass
                    center=np.asarray(from_ue(vector_field(b,'Origin')));ext=np.abs(np.asarray(from_ue(vector_field(b,'BoxExtent'))));lo,hi=center-ext,center+ext
                    if any(np.all(np.minimum(hi,bhi)>np.maximum(lo,blo)) for blo,bhi in bounds):continue
                    good=self._depth_contact_probe(actor);bad=self._depth_contact_probe(actor,lower=True)
                    detected=bad['status']=='fail' and bad['gap_m']<-.02
                    if good['status']!='pass' or not detected:
                        attempts.append({'endpoint_id':ep,'candidate':p,'good':good,'known_bad':bad});continue
                    gate=getattr(self,'acoustic_position_admission',None);acoustic=gate(ep,np.asarray(actor['emitter_position_m'])) if gate else {'status':'not_requested'}
                    if acoustic['status']=='blocked':continue
                    accepted={'status':'pass','endpoint_id':ep,'asset_id':a['asset_id'],'action':'registered_stand',
                        'contact_tolerance_m':.02,'tolerance_changed':False,'root_y_adjustment_m':0.,
                        'native_sole_depth':good,'native_capsule_static_geometry':capsule,'native_full_skin_bounds_m':[lo.tolist(),hi.tolist()],
                        'bounds_authority':'actual UE GetActorBounds of all original registered mesh components',
                        'known_bad_10cm_floor_contacts':bad,'known_bad_detected':detected,'acoustic_position_readback':acoustic}
                    bounds.append((lo,hi));self.alpha_standing_refs[ep]={'position_m':actor['position_m'],'reference':good};break
                except ValueError as e:attempts.append({'endpoint_id':ep,'candidate':p,'reason':str(e)})
            if accepted is None:
                write_json(directory/'placement_search_failed.json',{'status':'blocked','attempts':attempts});raise ValueError('registered UE actor lacks original qualified placement')
            checks.append(accepted)
        for ep,native in self.environment_nodes.items():
            success=False
            for p in [native['point'],*self.scene.get('native_placement_candidates',[])]:
                native['point']=p
                with self.instance.begin_frame():self._apply(0)
                with self.instance.end_frame():pass
                point=np.asarray(next(x['position_m'] for x in self.environment_frame_readbacks[-1]['emitters'] if x['endpoint_id']==ep))
                result=self.acoustic_position_admission(ep,point)
                if result['status']!='blocked':success=True;break
            if not success:raise ValueError('registered UE environmental emitter direction not admitted')
        write_json(directory/'standing_admission.json',{'status':'pass','checks':checks,'placement_attempts':attempts})
        self.alpha_camera_rows={};self.environment_frame_readbacks=[]
        return checks

    def _uncropped_foot_probe(self,actor):
        saved=self.alpha_standing_refs[actor['endpoint_id']]
        if actor['moving'] or actor['action_id']!='registered_stand' or np.max(np.abs(np.asarray(actor['position_m'])-saved['position_m']))>.0005:
            raise ValueError('registered rigid static support reference changed')
        ref=saved['reference'];root_y=actor['position_m'][1]
        return {'status':ref['status'],'gap_m':ref['gap_m'],'lowest_rendered_body_y_m':root_y+ref['gap_m'],
            'support_floor_y_m':root_y,'raster_depth_tolerance_m':.02,'method':'actual full native front/rear rigid registered depth; native root checked this frame',
            'measurement_reused':True,'reference_views':ref['views']}

    def encode(self,scene,path,review_frames=None,*,collect_pixel_visibility=True):
        result=super().encode(scene,path,review_frames,collect_pixel_visibility=collect_pixel_visibility)
        # Protocol extra camera samples are not RGB frames; retain precisely
        # the last actual rendering record for each of the 450 frame indices.
        rows={r['frame_index']:r for r in self.environment_frame_readbacks[:450]}
        if len(rows)!=450:raise ValueError('registered UE environmental clock incomplete')
        result['environmental_emitter_readbacks']=[rows[i] for i in range(450)]
        for ep in self.environment_nodes:
            result['observed_mouth_trajectories'][ep]=[{'time_s':r['time_s'],'position_m':next(e['position_m'] for e in r['emitters'] if e['endpoint_id']==ep)} for r in result['environmental_emitter_readbacks']]
        result['alignment']=result['native_visibility_alignment'];result['registered_visual_revision']=REGISTERED_UE_VISUAL_REVISION
        write_json(Path(path).parent/'native_environmental_emitters.json',{'frames':result['environmental_emitter_readbacks'],'visual_geometry':'none; original registered audio-only source points'})
        write_json(Path(path).parent/'native_registered_visual_evidence.json',result)
        return result

    @staticmethod
    def _destroy_registered(instance,runtimes):
        with instance.begin_frame():
            for r in runtimes.values():r['anchor'].K2_DestroyActor()
        with instance.end_frame():pass
