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

REGISTERED_UE_VISUAL_REVISION='registered-native-UE-design-whitebalance-v7-measured-basis-load'


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
        self.capture_scale=inputs.get('registered_ue_capture_scale',2)
        if type(self.capture_scale) is not int or self.capture_scale not in (1,2):
            raise ValueError('registered UE capture scale must be one or two native pixels per output dimension')
        self.source_material_cache_directory=source_material_cache_directory
        self.ue=bind_histogram_profile(self.house['input']['ue_config'],scene_seed=scene['scene_id'])
        if inputs.get('registered_offscreen_viewport_resolution_hw'):
            self.ue['offscreen_viewport_resolution_hw']=list(inputs['registered_offscreen_viewport_resolution_hw'])
        # The camera helper imports the native client; resolve the task-owned
        # extension before that import can cache an unavailable client.
        sys.path.insert(0,str(Path(self.ue['spear_ext_dir']).resolve()))
        self.ue['allow_invisible_source_material_samples']=True
        self.ue['source_material_inventory']=str(Path(inputs['native_registered_content_receipt']).parent.parent/'inputs/assets.json')
        self.ue['source_material_inventory']=str(Path(inputs['native_registered_content_receipt']).parent.parent/'inputs/assets.json')
        self.house['input']['ue_config']=self.ue
        if scene.get('require_current_native_environment'):
            # Same accepted authored lights, copied into this isolated launch root.
            import shutil
            source_light=Path(self.ue['native_lighting_audit'])
            own_light=Path(self.ue['output_root'])/'inputs/authored_lights'/house['house']/'native_lighting_audit.json'
            own_light.parent.mkdir(parents=True,exist_ok=True)
            if not own_light.exists():shutil.copy2(source_light,own_light)
            elif json.loads(own_light.read_text())!=json.loads(source_light.read_text()):raise ValueError('task light audit copy differs')
            self.ue['native_lighting_audit']=str(own_light)
            source_qa=Path(self.ue['native_room_visual_qa'])
            own_qa=Path(self.ue['output_root'])/'inputs/accepted_room_qa'/house['house']/'result_v2.json'
            expected_qa=json.loads(source_qa.read_text())
            source_capture=Path(expected_qa['inputs']['capture'])/'result.json'
            own_capture=own_qa.parent/'capture/result.json'
            own_capture.parent.mkdir(parents=True,exist_ok=True)
            if not own_capture.exists():shutil.copy2(source_capture,own_capture)
            elif json.loads(own_capture.read_text())!=json.loads(source_capture.read_text()):raise ValueError('task native capture copy differs')
            expected_qa['inputs']['capture']=str(own_capture.parent)
            if source_qa.resolve()!=own_qa.resolve():
                expected_qa['prior_admission_provenance']={'qa':str(source_qa),'capture':str(source_capture)}
            if not own_qa.exists():own_qa.write_text(json.dumps(expected_qa,ensure_ascii=False,indent=2)+'\n')
            elif json.loads(own_qa.read_text())!=expected_qa:raise ValueError('task room QA copy differs')
            self.ue['native_room_visual_qa']=str(own_qa)
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
        self.physical_environment_bindings={}
        self.current_native_environment=bool(scene.get('require_current_native_environment'))
        self.suspend_unused_capture_sensors=bool(inputs.get('suspend_unused_registered_capture_sensors',False))
        self.registered_actor_yaw_offset_deg=float(inputs.get('registered_actor_yaw_offset_deg',90.))
        self.construct_registered_dialogue_camera=bool(inputs.get('construct_registered_dialogue_camera',False))
        if not math.isfinite(self.registered_actor_yaw_offset_deg):raise ValueError('registered actor yaw offset must be finite')
        if inputs.get("kujiale_ue_asset_admission"):
            from avengine.capture.kujiale_registered_environment import environment_bindings, require_binding_project
            self.physical_environment_bindings=environment_bindings(scene,house["house"],inputs["kujiale_ue_asset_admission"],{a["endpoint_id"] for a in assets})
            if self.current_native_environment:
                inventory=json.loads(Path(inputs["kujiale_ue_asset_admission"]).read_text())
                byid={x['asset_id']:x for x in inventory['assets']}
                for i,plan in enumerate(scene.get('decorations',[])):
                    row=byid[plan['asset_id']]
                    if not row.get('silent_decor') or row.get('ue_native_physical_status')!='pass' or plan.get('status')!='pass_geometry_requires_current_native_scene':
                        raise ValueError('silent decoration requires an admitted native physical asset and current scene plan')
                    native={**plan,'emitter':{'actual_m':plan['expected_emitter_m']}}
                    self.physical_environment_bindings['decoration'+str(i+1)]={'asset':row,'native':native,'source_id':20001+i,'registration':inputs['kujiale_ue_asset_admission']}
            require_binding_project(self.physical_environment_bindings,self.ue)
        self.declarations=[{'actor_id':f'source{i+1}','asset_id':a['asset_id'],'ue_generated_class':a['ue_generated_class']} for i,a in enumerate(assets)]
        self.material_inventory={a['asset_id']:{'material_receipt':{'authority':'original registered UE package materials',
            'source_registry':a['source_registry'],'source_registry_sha256':a['source_registry_sha256'],
            'profiles':[{'max_highlight_ratio':.08}],'no_Habitat_PBR_overrides':True}} for a in assets}
        if self.current_native_environment:
            for declaration in self.material_inventory.values():
                declaration['material_receipt']['profiles']=[{'max_highlight_ratio':.02}]
        self.spawn,self.read_depth,self.read_rgb,_old,_actor,_spawn,_apply,_destroy=camera_helpers()
        self.destroy=self._destroy_registered
        start=time.perf_counter()
        try:
            self.instance,self.game,launch=launch_native(self.ue,self.house['input']['ue_map_path'])
            if self.current_native_environment or 'physical_gpu_index' in self.ue:
                from avengine.capture.kujiale_pilot_environment import verify_gpu
                launch['actual_graphics_gpu']=verify_gpu(self.ue,launch)
            with self.instance.begin_frame():
                self.camera,self.components=self.spawn(self.game,horizontal_fov_deg=85.,width=width*self.capture_scale,height=height*self.capture_scale)
                self.foot_camera,self.foot_components=self.spawn(self.game,horizontal_fov_deg=85.,width=1024,height=1024)
                # Semantic IDs are reconstructed from the complete native
                # modal/target depth passes; this extra capture is not consumed.
                if self.suspend_unused_capture_sensors:
                    self.components['object_ids'].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
                for name in ['rgb','object_ids']:self.foot_components[name].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
                apply_camera(self.camera,self._camera())
                lights=configure_scene_lighting(self.game,self.acceptance,self.ue)
                self.original_capture_source=self.components['rgb'].get_property_value(property_name='CaptureSource')
                exposure=apply_native_rgb_profile(self.components['rgb'],self.ue)
                for a,d in zip(assets,self.declarations,strict=True):self.runtimes[d['actor_id']]=self._spawn_registered(a,d)
                for ep,spec in scene['sources'].items():
                    if ep in self.physical_environment_bindings:
                        self.environment_nodes[ep]=self._spawn_physical_environment(ep,self.physical_environment_bindings[ep])
                    elif spec.get('native_visual_kind')=='audio_only':
                        actor=self.game.unreal_service.spawn_actor(uclass='AActor',spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
                        root=self.game.unreal_service.create_scene_component_for_actor(owner=actor,scene_component_name=ep+'_audio_only',uclass='USceneComponent')
                        root.SetMobility(NewMobility='Movable');actor.SetActorEnableCollision(bNewActorEnableCollision=False)
                        self.environment_nodes[ep]={'actor':actor,'root':root,'point':spec['trajectory'][0]['position_m'],'source_id':spec['actor_id']}
                for ep,binding in self.physical_environment_bindings.items():
                    if ep not in self.environment_nodes:self.environment_nodes[ep]=self._spawn_physical_environment(ep,binding)
                self._apply(0)
            with self.instance.end_frame():pass
            for _ in range(PARAMETERS['warmup_frames']):
                with self.instance.begin_frame():pass
                with self.instance.end_frame():self.read_rgb(self.components['rgb'])
            self.receipt={'backend':'UE/SPEAR registered native standing','graphics_adapter':self.ue['graphics_adapter'],
                'compute_device':'GPU','launch':launch,'exposure':exposure,'lights':lights,'hfov_deg':85.,'fps':15,
                'capture_resolution_hw':[height*self.capture_scale,width*self.capture_scale],'resolution_hw':[height,width],
                'startup_and_warmup_seconds':time.perf_counter()-start,'original_materials':True,
                'source_material_overrides_applied':False,'source_extra_fill_applied':False,
                'initial_placement_preflight':self.initial_placement_preflight,
                'registered_asset_ids':[a['asset_id'] for a in assets],'native_floor_surface_validation':self.native_floor_surface_check,
                'registered_content_copy_receipt':inputs['native_registered_content_receipt'],
                'action_scope':'actual original registered static Blueprint; no walk/sit substitution'}
        except BaseException:self.close();raise

    def _camera(self):
        view=super()._camera()
        view['yaw_deg']+=getattr(self,'current_camera_yaw_offset',0.)
        return view

    def _expected_camera_yaw(self,time_s):
        # The qualified binding view rotates the planned pan as a whole.
        # Retain the exact native pose tolerance; FOA and truth use readbacks.
        return super()._expected_camera_yaw(time_s)+getattr(self,'current_camera_yaw_offset',0.)

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
        from avengine.capture.v7_ue_registered_readback import prepare_readback
        measured=prepare_readback(self,a,d,actor,root,components)
        return {**measured,'anchor':actor,'anchor_root':root,'visual_actor':actor,'component':components[0],
            'components':components,'emitter_component':emitter,'motion_model':'rigid_static',
            'original_material_readback':{'status':'pass','paths':observed,'source_registry':a['source_registry'],'material_overrides_applied':False}}

    def _spawn_physical_environment(self,endpoint,binding):
        from avengine.backends.spear_ue.research_runtime import spawn_attached_static_actor
        row,native=binding["asset"],binding["native"];placement=native["placement"]
        runtimes=[spawn_attached_static_actor(self.game,actor_id=endpoint+"_"+str(i),static_mesh_object_path=path,
            position_ue_cm=to_ue(placement["root_m"]),yaw_ue_degrees=-math.degrees(placement["yaw_rad"]),
            actor_scale=1.,emitter_local_ue_cm=to_ue(row.get("emitter_local_m") or [0,0,0]))
            for i,path in enumerate(native["ue_asset_paths"])]
        return {"physical":True,"runtimes":runtimes,"source_id":binding["source_id"],
            "asset_id":row["asset_id"],"point":native["emitter"]["actual_m"],
            "native_admission":binding["registration"],"native_evidence":native}

    def _navigation_foot(self,position):
        p=np.asarray(position,float)
        target=np.asarray([p[0],p[1]-1.3,p[2]],float)
        foot=np.asarray(self.pf.snap_point(target),float)
        distance=float(np.linalg.norm(foot-target))
        if not np.isfinite(foot).all() or distance>.15:raise ValueError('registered candidate off navigation')
        # Reject furniture navigation before entering any SPEAR transaction.
        self.native_floor_surface.sample(foot)
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
        # Pose construction remains explicit in the recipe. The admitted GLB
        # import faces local UE +X; -90 makes it face the actual listener.
        return root,yaw+self.registered_actor_yaw_offset_deg

    def _apply(self,index):
        from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
        self.alpha_time=index/15;apply_camera(self.camera,self._camera());rows=[]
        for a,d in zip(self.assets,self.declarations,strict=True):
            root,yaw=self._body(a);native=self.runtimes[d['actor_id']]
            native['anchor'].K2_SetActorLocationAndRotation(NewLocation=dict(zip(['X','Y','Z'],to_ue(root))),NewRotation={'Roll':0.,'Pitch':0.,'Yaw':yaw},bSweep=False,bTeleport=True)
            pose=read_scene_component_pose(native['anchor_root']);emitter=read_scene_component_pose(native['emitter_component']);position=from_ue(pose['location_cm'])
            error=float(np.max(np.abs(np.asarray(position)-root)))
            if error>.0005:raise ValueError('registered UE root readback mismatch')
            from avengine.capture.v7_ue_registered_readback import measured_body_matrix
            matrix,basis_points=measured_body_matrix(native,position)
            expected=(np.asarray(matrix)@np.r_[a['mouth_local_glb_m'],1.])[:3]
            if np.linalg.norm(expected-np.asarray(from_ue(emitter['location_cm'])))>.002:raise ValueError('actual UE mouth differs from measured body basis')
            rows.append({'body_to_world_m':matrix,'body_basis_world_m':basis_points,
                'native_root_component_pose':pose,'native_load_backend':'ue_spear',
                'native_load_receipt':native['native_load_receipt'],
                'loaded_artifact_sha256':native['loaded_artifact_sha256'],'original_glb_sha256':native['original_glb_sha256'],
                'endpoint_id':a['endpoint_id'],'actor_id':self.scene['sources'][a['endpoint_id']]['actor_id'],
                'native_actor_id':d['actor_id'],'native_object_id':int(native['visual_actor'].uobject),'asset_id':a['asset_id'],'position_m':position,
                'emitter_position_m':from_ue(emitter['location_cm']),'moving':False,'action_id':'registered_stand',
                'action_phase':0.,'root_max_error':error,'joint_state':'registered rigid Blueprint; no joint DOFs',
                'source_registry_sha256':a['source_registry_sha256']})
        env=[]
        for ep,native in self.environment_nodes.items():
            if native.get('physical'):
                emitter=read_scene_component_pose(native['runtimes'][0]['emitter_component']);point=from_ue(emitter['location_cm'])
                if np.max(np.abs(np.asarray(point)-native['point']))>.0005:raise ValueError('native physical source emitter moved away from admitted sound position')
                env.append({'endpoint_id':ep,'source_id':native['source_id'],'asset_id':native['asset_id'],
                    'position_m':point,'native_component_pose':emitter,'native_readback':'actual UE visible StaticMesh and attached physical emitter',
                    'native_visual_admission':native['native_admission']})
                continue
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
        self.alpha_standing_refs={};self.current_camera_yaw_offset=0.
        for a,d in zip(self.assets,self.declarations,strict=True):
            accepted=None;ep=a['endpoint_id'];original=self.scene['sources'][ep]['trajectory'][0]['position_m']
            candidates=[original,*self.scene.get('native_placement_candidates',[])]
            if (self.current_native_environment and self.scene['production_scenario'].get('camera_policy')=='pilot_binding_coverage'
                    and len(self.assets)>2 and a is self.assets[2]):
                from avengine.capture.kujiale_pilot_environment import comparison_person_candidates
                points=[self.alpha_standing_refs[x['endpoint_id']]['position_m'] for x in self.assets[:2]]
                candidates=comparison_person_candidates(self.scene['listener_m'],points,candidates)
            for p in candidates:
                # SPEAR's begin_frame marks its transaction permanently in
                # error if a ValueError escapes _apply. Check the unchanged
                # 15 cm navigation rule before entering that transaction so
                # the original next-candidate search can actually continue.
                try:self._navigation_foot(p)
                except ValueError as e:
                    attempts.append({'endpoint_id':ep,'candidate':p,'reason':str(e)});continue
                if ((self.construct_registered_dialogue_camera or (self.current_native_environment and self.scene['production_scenario'].get('camera_policy')=='pilot_binding_coverage'))
                        and a is self.assets[1] and self.assets[0]['endpoint_id'] in self.alpha_standing_refs):
                    from avengine.capture.kujiale_pilot_environment import binding_pair_fits_view
                    first=self.alpha_standing_refs[self.assets[0]['endpoint_id']]['position_m']
                    if not binding_pair_fits_view(self.scene['listener_m'],first,p):
                        attempts.append({'endpoint_id':ep,'candidate':p,'reason':'both active speakers need separated bearings within the production binding view'});continue
                self.placement_overrides[ep]=p
                try:
                    with self.instance.begin_frame():actor=next(x for x in self._apply(0) if x['endpoint_id']==ep)
                    with self.instance.end_frame():pass
                    if (self.current_native_environment and self.scene['production_scenario'].get('camera_policy')=='pilot_binding_coverage'
                            and self.alpha_standing_refs):
                        from avengine.capture.kujiale_pilot_environment import binding_mouth_distance_contrast
                        mouths=[x['emitter_position_m'] for x in self.alpha_standing_refs.values()]
                        if not binding_mouth_distance_contrast(self.scene['listener_m'],actor['emitter_position_m'],mouths):
                            attempts.append({'endpoint_id':ep,'candidate':p,'reason':'actual native mouth distances need 0.6 m sampling contrast above unchanged 0.5 m QA gate'});continue
                    root=np.asarray(actor['position_m']);probe={'witness_index':0,'role':ep,'capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':to_ue(root+[0.,.75,0.])}}
                    capsule=query_capsule(self.native_floor_surface.mesh,probe)
                    if capsule['status']!='pass':attempts.append({'endpoint_id':ep,'candidate':p,'capsule':capsule['status']});continue
                    native=self.runtimes[d['actor_id']]
                    with self.instance.begin_frame():b=native['visual_actor'].GetActorBounds(bOnlyCollidingComponents=False,bIncludeFromChildActors=True,as_dict=True)
                    with self.instance.end_frame():pass
                    center=np.asarray(from_ue(vector_field(b,'Origin')));ext=np.abs(np.asarray(from_ue(vector_field(b,'BoxExtent'))));lo,hi=center-ext,center+ext
                    if any(np.all(np.minimum(hi,bhi)>np.maximum(lo,blo)) for blo,bhi in bounds):continue
                    if self.current_native_environment:
                        from avengine.capture.kujiale_pilot_environment import human_clear_of_environment
                        if not human_clear_of_environment(self,lo,hi):continue
                    good=self._depth_contact_probe(actor);bad=self._depth_contact_probe(actor,lower=True)
                    detected=bad['status']=='fail' and bad['gap_m']<-.02
                    if good['status']!='pass' or not detected or (self.current_native_environment and abs(good['gap_m'])>.015):
                        attempts.append({'endpoint_id':ep,'candidate':p,'good':good,'known_bad':bad});continue
                    gate=getattr(self,'acoustic_position_admission',None);acoustic=gate(ep,np.asarray(actor['emitter_position_m'])) if gate else {'status':'not_requested'}
                    if acoustic['status']=='blocked':continue
                    accepted={'status':'pass','endpoint_id':ep,'asset_id':a['asset_id'],'action':'registered_stand',
                        'contact_tolerance_m':.02,'tolerance_changed':False,'root_y_adjustment_m':0.,
                        'native_sole_depth':good,'native_capsule_static_geometry':capsule,'native_full_skin_bounds_m':[lo.tolist(),hi.tolist()],
                        'bounds_authority':'actual UE GetActorBounds of all original registered mesh components',
                        'known_bad_10cm_floor_contacts':bad,'known_bad_detected':detected,'acoustic_position_readback':acoustic}
                    bounds.append((lo,hi));self.alpha_standing_refs[ep]={'position_m':actor['position_m'],'emitter_position_m':actor['emitter_position_m'],'reference':good};break
                except ValueError as e:attempts.append({'endpoint_id':ep,'candidate':p,'reason':str(e)})
            if accepted is None:
                write_json(directory/'placement_search_failed.json',{'status':'blocked','attempts':attempts});raise ValueError('registered UE actor lacks original qualified placement')
            checks.append(accepted)
        for ep,native in self.environment_nodes.items():
            if native.get('physical'):
                result=self.acoustic_position_admission(ep,np.asarray(native['point']))
                if result['status']=='blocked':raise ValueError('visible native environmental emitter acoustics rejected')
                continue
            success=False
            for p in [native['point'],*self.scene.get('native_placement_candidates',[])]:
                native['point']=p
                with self.instance.begin_frame():self._apply(0)
                with self.instance.end_frame():pass
                point=np.asarray(next(x['position_m'] for x in self.environment_frame_readbacks[-1]['emitters'] if x['endpoint_id']==ep))
                result=self.acoustic_position_admission(ep,point)
                if result['status']!='blocked':success=True;break
            if not success:raise ValueError('registered UE environmental emitter direction not admitted')
        if self.current_native_environment:
            if self.scene['production_scenario'].get('camera_policy')=='pilot_binding_coverage':
                points=[np.array(self.alpha_standing_refs[a['endpoint_id']]['position_m']) for a in self.assets[:2]]
                listener=np.array(self.scene['listener_m'])
                from avengine.capture.kujiale_pilot_environment import binding_camera_yaw
                current=binding_camera_yaw(listener,points)
                original=math.degrees(math.atan2(self.scene['camera_forward'][0],-self.scene['camera_forward'][2]))
                self.current_camera_yaw_offset=(current-original+180)%360-180
            from avengine.capture.kujiale_pilot_environment import qualify_scene
            qualify_scene(self,directory.parent/'physical_environment',checks)
        if self.construct_registered_dialogue_camera:
            from avengine.capture.kujiale_pilot_environment import construct_dialogue_camera
            actual=[self.alpha_standing_refs[a['endpoint_id']]['emitter_position_m']for a in self.assets[:2]]
            plan=construct_dialogue_camera(self.scene['listener_m'],actual)
            original=math.degrees(math.atan2(self.scene['camera_forward'][0],-self.scene['camera_forward'][2]))
            self.current_camera_yaw_offset=(plan['yaw_deg']-original+180.)%360.-180.
            self.receipt['native_dialogue_camera_construction']=plan
        if self.current_native_environment or self.suspend_unused_capture_sensors:
            # Static foot support was measured front/rear and with a negative
            # 10-cm control. Every frame still verifies the identical rigid
            # pose/root before reusing that reference; these captures are not
            # read during the 450-frame static video.
            with self.instance.begin_frame():
                for component in self.foot_components.values():
                    component.set_property_value(property_name='bCaptureEveryFrame',property_value=False)
            with self.instance.end_frame():pass
            self.receipt['static_foot_capture_policy']='qualified front/rear rigid support cache; unchanged per-frame root/pose verification'
            self.receipt['qualified_binding_camera_yaw_offset_deg']=self.current_camera_yaw_offset
            self.receipt['isolated_depth_capture_policy']='every native target depth frame rendered; unused RGB/object-ID sensors suspended only during isolated passes and restored'
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
            if ep not in self.scene['sources']:continue
            result['observed_mouth_trajectories'][ep]=[{'time_s':r['time_s'],'position_m':next(e['position_m'] for e in r['emitters'] if e['endpoint_id']==ep)} for r in result['environmental_emitter_readbacks']]
        if self.current_native_environment:
            from avengine.capture.kujiale_pilot_environment import finish_scene
            finish_scene(self,result,path)
        result['alignment']=result['native_visibility_alignment'];result['registered_visual_revision']=REGISTERED_UE_VISUAL_REVISION
        copies=[]
        for asset in self.assets:
            frames=[a for f in result['frames']for a in f['actors']if a['asset_id']==asset['asset_id']]
            copies.append({**deepcopy(asset),'ue_measured_registration':{'house':self.house['house'],'room_id':self.house['room']['room_label'],
                'frames_checked':len(frames),'first_body_to_world_m':frames[0]['body_to_world_m'],
                'load_receipt':frames[0]['native_load_receipt'],'loaded_artifact_sha256':frames[0]['loaded_artifact_sha256'],
                'original_glb_sha256':frames[0]['original_glb_sha256'],'native_visual_evidence':str(Path(path).parent/'native_registered_visual_evidence.json')}})
        copy_path=Path(path).parent/'native_registered_assets.json'
        write_json(copy_path,{'schema':'v7-UE-measured-registration-copy-v1','original_registry_modified':False,'assets':copies})
        result['measured_registration_copy']=str(copy_path)
        write_json(Path(path).parent/'native_environmental_emitters.json',{'frames':result['environmental_emitter_readbacks'],'visual_geometry':'admitted original physical StaticMesh assets' if self.physical_environment_bindings else 'legacy audio-only source points'})
        write_json(Path(path).parent/'native_registered_visual_evidence.json',result)
        return result

    @staticmethod
    def _destroy_registered(instance,runtimes):
        with instance.begin_frame():
            for r in runtimes.values():r['anchor'].K2_DestroyActor()
        with instance.end_frame():pass
