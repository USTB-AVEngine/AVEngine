"""Native registered standing GLBs, retaining episode/contact/pixel gates.

A rigid standing snapshot has no joint DOFs. Static registration is never
promoted to walking admission. Registered mouth points are transformed from
the actual native root; collision probes include every original mesh vertex.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, math
import numpy as np
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.capture.v7_alpha_native import NativeAlphaVideo,forward_at,stage_sole_rays
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.continuous_scene import camera_basis

REGISTERED_VISUAL_REVISION='registered-contact-movement-native-emitter-v7-exact-target-mask'

class RigidStandingInstance:
    """Native rigid object with an explicitly empty articulated state."""
    def __init__(self,obj): self.native=obj
    def __getattr__(self,key): return getattr(self.native,key)
    @property
    def joint_positions(self): return np.empty(0)
    def get_link_ids(self): return []
    @property
    def motion_type(self): return self.native.motion_type
    @motion_type.setter
    def motion_type(self,value): self.native.motion_type=value
    @property
    def translation(self): return self.native.translation
    @translation.setter
    def translation(self,value): self.native.translation=value
    @property
    def rotation(self): return self.native.rotation
    @rotation.setter
    def rotation(self,value): self.native.rotation=value

def loaded_mesh(path):
    import trimesh
    scene=trimesh.load(path,force='scene',process=False)
    mesh=scene.to_geometry() if hasattr(scene,'to_geometry') else scene.dump(concatenate=True)
    if not len(mesh.vertices) or not len(mesh.faces):raise ValueError('registered original mesh empty')
    return mesh

def sole_regions(local,asset):
    if asset.get('entity_class')=='articulated_animal':
        x,z=np.median(local[:,0]),np.median(local[:,2])
        masks=[(local[:,0]<x)&(local[:,2]<z),(local[:,0]>=x)&(local[:,2]<z),(local[:,0]<x)&(local[:,2]>=z),(local[:,0]>=x)&(local[:,2]>=z)]
        if any(not m.any()for m in masks):raise ValueError('registered dog four-paw regions missing')
        return masks
    return [local[:,0]<0,local[:,0]>=0]

class NativeRegisteredVideo(NativeAlphaVideo):
    def __init__(self,house,scene,inputs,assets,**kwargs):
        self._native_restart_arguments=(house,scene,inputs,assets,dict(kwargs))
        self._native_pose_resource_paths=set()
        self._native_context_generation=0
        self._native_context_refreshing=False
        self._native_pose_cache_limit=int(inputs.get('native_motion_pose_cache_limit',64))
        if not 16<=self._native_pose_cache_limit<=256:raise ValueError('native pose cache limit must be 16..256')
        if inputs.get('render_backend') not in {'cpu_llvmpipe','habitat_gpu'}:raise ValueError('registered Habitat backend required')
        NativeContinuousVideo.__init__(self,house,scene,inputs,[],**kwargs)
        from avengine.assets.source_materials import prepare_source_glb
        self.camera_frame_readbacks=[];self.standing_checks=[];self.motion_house=house;self.motion_inputs=inputs
        self.environment_nodes={};self.environment_frame_readbacks=[]
        for ep,spec in scene['sources'].items():
            if spec.get('native_visual_kind')=='audio_only':
                node=self.sim.get_active_scene_graph().get_root_node().create_child()
                node.translation=self.mn.Vector3(spec['trajectory'][0]['position_m'])
                self.environment_nodes[ep]={'node':node,'source_id':spec['actor_id']}

        manager=self.sim.get_object_template_manager()
        for i,asset in enumerate(assets):
            if not asset.get('standing_registered_qualified') or asset['allowed_pose_for_this_task']!='stand':
                raise ValueError('registered standing authority missing')
            path=Path(asset['habitat_glb']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if asset['actual_artifact_sha256'].get(str(path))!=digest:raise ValueError('registered GLB changed')
            derived,material=prepare_source_glb(path,asset_id=asset['asset_id'],entity_class=asset.get('entity_class','articulated_human'),output_directory=self.source_material_cache_directory)
            attr=manager.create_new_template('v7_registered_'+str(i))
            attr.render_asset_handle=str(derived);attr.collision_asset_handle=str(derived)
            attr.is_collidable=False;attr.compute_COM_from_shape=False
            attr.shader_type='pbr'
            tid=manager.register_template(attr)
            obj=self.sim.get_rigid_object_manager().add_object_by_template_id(tid)
            obj.motion_type=self.hs.physics.MotionType.KINEMATIC;obj.semantic_id=1000+int(asset['native_instance_id'])
            mesh=loaded_mesh(path)
            self.actors.append({'object':RigidStandingInstance(obj),'endpoint_id':asset['endpoint_id'],'semantic_id':1000+int(asset['native_instance_id']),'declaration':asset,'source_material':material,'original_mesh':mesh,'applied_joint_input':np.empty(0),'native_actor_kind':'registered_rigid','prepared_glb':str(derived)})
        self.receipt.update(backend='Habitat-Sim registered standing GLB',source_materials=[a['source_material'] for a in self.actors],action_clocks=[],asset_admission_states=['formal_static_registration_episode_pending']*len(self.actors))

    def _apply_entry(self,entry,scene,frame_index=None):
        if frame_index is not None and entry.get("motion_episode"):
            return self._apply_motion_entry(entry,frame_index*2)
        from avengine.assets.habitat_capture import _apply_root_with_habitat
        asset=entry['declaration'];spec=scene['sources'][entry['endpoint_id']]
        if any(k['position_m']!=spec['trajectory'][0]['position_m'] for k in spec['trajectory']):raise ValueError('registered standing body moved without walking qualification')
        p=np.asarray(entry.get('placement_override_m',spec['trajectory'][0]['position_m']),float)
        a=math.radians(spec['body_yaw_deg']);f=np.array([math.sin(a),0.,-math.cos(a)])
        root=np.eye(4);root[:3,:3]=np.stack((np.cross(f,[0.,1.,0.]),[0.,1.,0.],-f),axis=1)
        root[:3,3]=[p[0],p[1]-1.3,p[2]]
        if 'floor_adjustment' not in entry:
            vertices=np.asarray(entry['original_mesh'].vertices)@root[:3,:3].T+root[:3,3]
            points=[]
            for side in sole_regions(np.asarray(entry['original_mesh'].vertices),asset):
                values=vertices[side];points.append(values[np.argmin(values[:,1])])
            soles=stage_sole_rays(self,points);delta=-min(x['signed_contact_error_m'] for x in soles)
            if abs(delta)>.5:raise ValueError('registered floor adjustment outside bounded support')
            entry['floor_adjustment']=delta
        root[1,3]+=entry['floor_adjustment']
        _apply_root_with_habitat(entry['object'],root,qt=self.qt,mn=self.mn)
        native=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
        error=float(np.max(np.abs(native-root)))
        if error>1e-5:raise ValueError('registered native body root differs from applied metres')
        # Use the actual rounded native root on every frame and in RIR.
        entry['applied_root_input']=native.copy()
        mouth=(native@np.r_[asset['mouth_local_glb_m'],1.])[:3]
        return {'endpoint_id':entry['endpoint_id'],'actor_id':asset['native_instance_id'],'native_object_id':int(entry['object'].object_id),'semantic_id':entry['semantic_id'],'asset_id':asset['asset_id'],'position_m':native[:3,3].tolist(),'body_to_world_m':native.tolist(),'emitter_position_m':mouth.tolist(),'moving':False,'action_id':'stand','action_sample_index':0,'action_tick':0,'root_max_error':error,'joint_max_error':0.,'joint_state':'no_joint_DOFs_registered_static_GLB','source_registry_sha256':asset['source_registry_sha256'],'loaded_artifact_sha256':asset['actual_artifact_sha256'][asset['habitat_glb']]}

    def _check_source_frame(self,rgb,semantic,frame_index):
        # Furniture has material checks but never becomes a speaking actor.
        actors=self.actors
        try:
            self.actors=[*actors,*getattr(self,'seat_material_entries',[])]
            super()._check_source_frame(rgb,semantic,frame_index)
        finally:self.actors=actors

    def _renew_motion_native_context(self):
        """Release native per-GLB resources, then reapply every actual actor.

        Motion geometry, original textures, exact matrices, source clocks and
        scene admission are unchanged. Every subsequent frame is independently
        read back by the existing native root/mouth/pixel gates.
        """
        if self._native_context_refreshing:return
        if getattr(self,'seat_material_entries',[]):
            raise ValueError('chunked native motion needs a registered seat rehydration adapter')
        import gc,time
        started=time.monotonic()
        house,scene,inputs,assets,kwargs=self._native_restart_arguments
        kwargs=dict(kwargs);kwargs['source_material_cache_directory']=self.source_material_cache_directory
        states=[(entry,entry.get('motion_pose_index',0),entry.get('motion_episode') is not None,
                 np.asarray(entry['object'].root_scene_node.absolute_transformation(),float).copy())
                for entry in self.actors]
        old_sim=self.sim
        for entry,_,_,_ in states:entry['object']=None
        old_sim.close()
        self.sim=None
        gc.collect()
        fresh=NativeRegisteredVideo(house,scene,inputs,assets,**kwargs)
        self.sim=fresh.sim
        fresh.sim=None
        self.pathfinder=fresh.pathfinder
        self.environment_nodes=fresh.environment_nodes
        by_endpoint={entry['endpoint_id']:entry for entry in fresh.actors}
        for entry,_,_,_ in states:
            replacement=by_endpoint[entry['endpoint_id']]
            entry['object']=replacement['object']
            entry.pop('motion_pose_path',None)
            entry.pop('motion_pose_index',None)
            entry.pop('contact_template',None)
        self._native_pose_resource_paths=set()
        self._native_context_generation+=1
        self._native_context_refreshing=True
        try:
            from avengine.capture.v7_registered_state import restore_actor_state
            from avengine.assets.habitat_capture import _apply_root_with_habitat
            for entry,index,moving,saved_root in states:
                restore_actor_state(self,entry,scene,index,moving,saved_root,_apply_root_with_habitat)
        finally:self._native_context_refreshing=False
        self.receipt.setdefault('native_context_renewals',[]).append({
            'generation':self._native_context_generation,'maximum_resident_unique_pose_assets':self._native_pose_cache_limit,
            'actors_reinstantiated':len(states),'wall_seconds':time.monotonic()-started,
            'same_original_house_assets_camera_and_motion_clock':True,
            'all_following_native_readback_gates_unchanged':True})

    def _apply_motion_entry(self,entry,index):
        from avengine.assets.habitat_capture import _apply_root_with_habitat
        import hashlib
        record=entry['motion_episode'];frame=record['source_frames'][index]
        path=record['pose_glbs'][index]
        if (path not in self._native_pose_resource_paths
                and len(self._native_pose_resource_paths)>=self._native_pose_cache_limit
                and not self._native_context_refreshing):
            self._renew_motion_native_context()
        self._native_pose_resource_paths.add(path)
        rm=self.sim.get_rigid_object_manager()
        if entry.get('motion_pose_path')!=path:
            rm.remove_object_by_id(entry['object'].object_id)
            tm=self.sim.get_object_template_manager()
            name='motion_'+entry['declaration']['asset_id']+'_'+str(index)
            template=tm.create_new_template(name);template.render_asset_handle=path;template.collision_asset_handle=path
            template.is_collidable=False;template.compute_COM_from_shape=False;template.shader_type='pbr'
            tid=tm.register_template(template);obj=rm.add_object_by_template_id(tid)
            obj.motion_type=self.hs.physics.MotionType.KINEMATIC;obj.semantic_id=entry['semantic_id']
            entry['object']=RigidStandingInstance(obj);entry['motion_pose_index']=index;entry['motion_pose_path']=path
        from avengine.assets.roster_motion.contact import yaw_matrix
        from avengine.assets.roster_motion.navmesh_execution import motion_yaw
        frame_yaw=motion_yaw(record,index);entry['resolved_yaw_rad']=frame_yaw
        root=np.eye(4);root[:3,:3]=yaw_matrix(frame_yaw);root[:3,3]=record['world_roots_m'][index]
        _apply_root_with_habitat(entry['object'],root,qt=self.qt,mn=self.mn)
        native=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
        child=entry['object'].root_scene_node.create_child();child.translation=self.mn.Vector3(frame['mouth_m'])
        mouth=np.asarray(child.absolute_transformation(),float)[:3,3];expected=(root@np.r_[frame['mouth_m'],1.])[:3]
        error=float(np.linalg.norm(mouth-expected))
        if error>.0001:raise ValueError('moving native mouth differs from posed lip point')
        entry['applied_root_input']=native.copy()
        a=entry['declaration']
        return {'endpoint_id':entry['endpoint_id'],'actor_id':a['native_instance_id'],
            'native_object_id':int(entry['object'].object_id),'semantic_id':entry['semantic_id'],'asset_id':a['asset_id'],
            'position_m':native[:3,3].tolist(),'body_to_world_m':native.tolist(),'emitter_position_m':mouth.tolist(),
            'moving':frame['moving'],'action_id':frame.get('action_id','walk' if frame['moving'] else 'planted_pause'),
            'action_sample_index':index,'action_tick':index,'root_max_error':float(np.max(np.abs(native-root))),
            'joint_max_error':0.,'joint_state':'30Hz CPU skin evaluated from admitted contact clip; native rigid posed mesh',
            'source_registry_sha256':a['source_registry_sha256'],
            'loaded_artifact_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'motion_master_sha256':record['lineage']['master_sha256'],'native_emitter_readback_error_m':error,
            'native_context_generation':self._native_context_generation}

    def _apply(self,scene,frame_index):
        rows=[self._apply_entry(entry,scene,frame_index) for entry in self.actors]
        f,r,u=camera_basis(forward_at(scene,frame_index/15))
        state=self.hs.AgentState();state.position=np.asarray(scene['listener_m'],dtype=np.float32)
        state.rotation=self.qt.from_rotation_matrix(np.stack((r,u,-f),axis=1));self.sim.get_agent(0).set_state(state)
        actual=self.sim.get_agent(0).get_state();matrix=self.qt.as_rotation_matrix(actual.rotation);actual_f=-matrix[:,2]
        self.camera_frame_readbacks.append({'frame_index':frame_index,'time_s':frame_index/15,'position_m':np.asarray(actual.position,float).tolist(),'yaw_deg':math.degrees(math.atan2(actual_f[0],-actual_f[2])),'forward_m':actual_f.tolist(),'orientation_wxyz':[float(actual.rotation.w),*actual.rotation.imag.tolist()],'camera_contract':'native observer rotation; listener position static'})
        if frame_index%15==0:
            obs=self.sim.get_sensor_observations();self._check_source_frame(np.asarray(obs['rgb'])[:,:,:3],np.asarray(obs['semantic']),frame_index)
        environmental=[]
        for ep,item in self.environment_nodes.items():
            actual=np.asarray(item['node'].absolute_transformation(),float)
            environmental.append({'endpoint_id':ep,'source_id':item['source_id'],'position_m':actual[:3,3].tolist(),'transform_m':actual.tolist(),'native_readback':'actual Habitat scenegraph audio-only child; no visual mesh'})
        self.environment_frame_readbacks.append({'frame_index':frame_index,'time_s':frame_index/15,'emitters':environmental})
        return rows

    def encode(self,scene,path,review_frames=None,*,collect_pixel_visibility=True):
        from avengine.timeline.v7_camera_evidence import review_frame_indices
        if review_frames is None:review_frames=review_frame_indices(scene)
        result=super().encode(scene,path,review_frames,collect_pixel_visibility=collect_pixel_visibility)
        if len(self.environment_frame_readbacks)!=450:raise ValueError('incomplete native environmental source clock')
        result['environmental_emitter_readbacks']=self.environment_frame_readbacks
        result['native_motion_admissions']={e['endpoint_id']:e['motion_episode'] for e in self.actors if e.get('motion_episode')}
        for ep,record in result['native_motion_admissions'].items():
            result['observed_mouth_trajectories'][ep]=[{'time_s':r['time_s'],'position_m':r['emitter_position_m']} for r in record['native_30Hz_readbacks']]

        for ep in self.environment_nodes:
            result['observed_mouth_trajectories'][ep]=[{'time_s':r['time_s'],'position_m':next(e['position_m'] for e in r['emitters'] if e['endpoint_id']==ep)} for r in self.environment_frame_readbacks]
        write_json(Path(path).parent/'native_environmental_emitters.json',{'frames':self.environment_frame_readbacks,'visual_geometry':'none; audio-only source points'})
        write_json(Path(path).parent/'native_registered_visual_evidence.json',result)
        return result

    def _probe_entry(self,entry,directory):
        from avengine.assets.world_contact import REFERENCE_MAXIMUM_CONTACT_HORIZONTAL_STEP_M as tolerance
        import trimesh
        from scipy.spatial import ConvexHull,QhullError
        native=entry['applied_root_input'];original=entry['original_mesh']
        local=np.asarray(original.vertices);vertices=local@native[:3,:3].T+native[:3,3]
        if 'contact_template' not in entry:
            center=(local.min(0)+local.max(0))/2;mesh=trimesh.Scene();groups={}
            cell=np.floor(local/.12).astype(int)
            for tri in original.faces:groups.setdefault(tuple(cell[int(tri[0])]),set()).update(int(k) for k in tri)
            covered=set()
            for i,ids in enumerate(groups.values()):
                covered.update(ids);points=local[sorted(ids)]-center
                try:
                    h=ConvexHull(points);part=trimesh.Trimesh(points,h.simplices,process=False);part.fix_normals()
                except QhullError:
                    part=trimesh.creation.box(np.maximum(np.ptp(points,axis=0),.001));part.apply_translation((points.min(0)+points.max(0))/2)
                mesh.add_geometry(part,geom_name='cell_'+str(i),node_name='cell_'+str(i))
            if len(covered)!=len(local):raise ValueError('registered contact proxy omitted original vertices')
            path=directory/(entry['endpoint_id']+'_registered_full_mesh.glb');path.write_bytes(mesh.export(file_type='glb'))
            manager=self.sim.get_object_template_manager();attr=manager.create_new_template('v7_registered_contact_'+entry['endpoint_id'])
            attr.render_asset_handle=str(path);attr.collision_asset_handle=str(path);attr.is_collidable=True;attr.join_collision_meshes=False;attr.margin=.001;attr.compute_COM_from_shape=False
            entry['contact_template']=(manager.register_template(attr),center,len(groups),str(path))
        tid,center,hulls,path=entry['contact_template'];rm=self.sim.get_rigid_object_manager();obj=rm.add_object_by_template_id(tid)
        from avengine.assets.habitat_capture import _apply_root_with_habitat
        matrix=native.copy();matrix[:3,3]=(native@np.r_[center,1])[:3]
        obj.motion_type=self.hs.physics.MotionType.DYNAMIC;_apply_root_with_habitat(obj,matrix,qt=self.qt,mn=self.mn)
        actor_ids={int(a['object'].object_id) for a in self.actors}
        def contacts():
            self.sim.perform_discrete_collision_detection();result=[]
            for c in self.sim.get_physics_contact_points():
                pair={int(c.object_id_a),int(c.object_id_b)}
                if int(obj.object_id) in pair and not pair.intersection(actor_ids):result.append({'object_id_a':int(c.object_id_a),'object_id_b':int(c.object_id_b),'contact_distance_m':float(c.contact_distance)})
            return result
        try:
            actual=contacts();obj.translation=self.mn.Vector3(matrix[:3,3]-[0.,.1,0.]);bad=contacts()
        finally:rm.remove_object_by_id(obj.object_id)
        points=[]
        for side in sole_regions(local,entry['declaration']):
            v=vertices[side];points.append(v[np.argmin(v[:,1])])
        soles=stage_sole_rays(self,points);floor_ok=all(abs(x['signed_contact_error_m'])<=tolerance for x in soles);skin_ok=all(x['contact_distance_m']>=-tolerance for x in actual);bad_detected=any(x['contact_distance_m']<-tolerance for x in bad)
        return {'status':'pass' if floor_ok and skin_ok and bad_detected else 'blocked','endpoint_id':entry['endpoint_id'],'asset_id':entry['declaration']['asset_id'],'action':'registered_stand','contact_tolerance_m':tolerance,'tolerance_changed':False,'sole_readbacks':soles,'native_skin_vertices':len(vertices),'native_triangles':len(original.faces),'full_skin_covered':True,'native_full_skin_bounds_m':[vertices.min(0).tolist(),vertices.max(0).tolist()],'compound_hulls':hulls,'compound_margin_m':.001,'contacts':actual,'known_bad_10cm_floor_contacts':bad,'known_bad_detected':bad_detected,'proxy_path':path,'source_original_glb':entry['declaration']['habitat_glb'],'motion_admission':'stand only; no walk/sit/seat qualification'}

    def qualify_standing(self,scene,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);checks=[];bounds=[];attempts=[]
        candidates=scene.get('native_placement_candidates',[])
        for entry in self.actors:
            original=scene['sources'][entry['endpoint_id']]['trajectory'][0]['position_m']
            accepted=None
            for position in [original,*candidates]:
                entry['placement_override_m']=position;entry.pop('floor_adjustment',None)
                try:
                    self._apply_entry(entry,scene)
                    check=self._probe_entry(entry,directory)
                    lo,hi=np.asarray(check['native_full_skin_bounds_m'])
                    intersects=any(np.all(np.minimum(hi,bhi)>np.maximum(lo,blo)) for blo,bhi in bounds)
                    attempts.append({'endpoint_id':entry['endpoint_id'],'candidate_m':position,'contact_status':check['status'],'known_bad_detected':check['known_bad_detected'],'pair_envelopes_overlap':bool(intersects),'minimum_contact_m':min((c['contact_distance_m'] for c in check['contacts']),default=None)})
                    if check['status']=='pass' and not intersects:
                        admit=getattr(self,'acoustic_position_admission',None)
                        mouth=(entry['applied_root_input']@np.r_[entry['declaration']['mouth_local_glb_m'],1.])[:3]
                        acoustic=admit(entry['endpoint_id'],mouth) if admit else {'status':'not_requested'}
                        attempts[-1]['acoustic_position_readback']=acoustic
                        if acoustic['status']=='blocked':continue
                        check['acoustic_position_readback']=acoustic
                        accepted=check;bounds.append((lo,hi));break
                except ValueError as error:
                    attempts.append({'endpoint_id':entry['endpoint_id'],'candidate_m':position,'status':'blocked','reason':str(error)})
            if accepted is None:
                write_json(directory/'placement_search_failed.json',{'status':'blocked','attempts':attempts})
                raise ValueError('registered actor has no native-qualified nonoverlapping room placement')
            accepted['resolved_body_root_world_m']=entry['applied_root_input'][:3,3].tolist();accepted['candidate_emitter_plan_m']=entry['placement_override_m'];checks.append(accepted)
        for ep,item in self.environment_nodes.items():
            accepted=None
            for point in [scene['sources'][ep]['trajectory'][0]['position_m'],*candidates]:
                item['node'].translation=self.mn.Vector3(point)
                actual=np.asarray(item['node'].absolute_transformation(),float)[:3,3]
                admit=getattr(self,'acoustic_position_admission',None)
                result=admit(ep,actual) if admit else {'status':'not_requested'}
                if result['status']=='pass':accepted=result;break
            if accepted is None:raise ValueError('audio-only source lacks unchanged native 5deg admission')
        requests=scene.get('production_scenario',{}).get('motion_requests',{})
        if requests:
            from avengine.assets.roster_motion.episode import compile_walking
            for entry in self.actors:
                ep=entry['endpoint_id']
                if ep not in requests:continue
                request=requests[ep]
                if request['action']=='walk'and entry['declaration'].get('entity_class')=='articulated_animal':
                    from avengine.assets.roster_motion.animal import compile_continuous_dog_walking
                    record=compile_continuous_dog_walking(self,entry,scene,self.motion_house,self.motion_inputs,directory/ep,request)
                elif request['action']=='walk':record=compile_walking(self,entry,scene,self.motion_house,self.motion_inputs,directory/ep,request)
                elif request['action']=='sit':
                    from avengine.assets.roster_motion.seating import compile_seated
                    record=compile_seated(self,entry,scene,self.motion_house,self.motion_inputs,directory/ep,request)
                else:raise ValueError('unsupported admitted motion action; no static substitution')
                entry['motion_episode']=record;actual=[]
                for index in range(901):
                    row=self._apply_motion_entry(entry,index);row.update(frame_index=index,time_s=index/30)
                    actual.append(row)
                record['native_30Hz_readbacks']=actual
                self._apply_motion_entry(entry,0)
                check=next(c for c in checks if c['endpoint_id']==ep)
                check.update(action='sit' if record.get('motion_kind')=='sit' else ('dog_continuous_walk'if record.get('species_id')=='dog'else'walk_and_planted_pause'),motion_admission=record,
                    native_full_skin_bounds_m=record['full_skin_bounds_m'],known_bad_detected=record['known_bad_10cm_floor_detected'])
                write_json(directory/ep/'native_motion_readbacks.json',{'status':'pass','frames':actual,'geometry_receipt':str(directory/ep/'motion_geometry.json')})
        self.environment_frame_readbacks=[]
        from avengine.timeline.v7_camera_evidence import recenter_on_actual_roots
        recenter_on_actual_roots(scene,self.actors)
        self.standing_checks=checks;write_json(directory/'standing_admission.json',{'status':'pass','checks':checks,'placement_attempts':attempts,'original_contact_gate_preserved':True});self.camera_frame_readbacks=[];return checks
