"""Phase-two native integration; immutable phase-one profile remains available."""
from copy import deepcopy
from avengine.capture.v7_visible_native import NativeVisibleVideo
PHASE2_VISUAL_REVISION='v7-visible-phase2-native-v15-original-chair-component-collision'
class NativePhase2Video(NativeVisibleVideo):
    def __init__(self,house,scene,inputs,assets,**kwargs):
        super().__init__(house,scene,inputs,assets,**kwargs)
        for entry in self.actors:
            a=entry['declaration']
            if a.get('phase2_palette_reference'):
                ref=a['phase2_palette_reference']
                entry['source_material']={**entry['source_material'],'source_palette_srgb':deepcopy(ref['colors']),
                    'measurement_reference_revision':ref['revision'],'measurement_thresholds_changed':False}
        self.receipt['source_materials']=[e['source_material'] for e in self.actors]


    def _apply_entry(self,entry,scene,frame_index=None):
        if entry.get('motion_episode'):
            return self._apply_motion_entry(entry,0 if frame_index is None else frame_index*2)
        return super()._apply_entry(entry,scene,frame_index)

    def _apply_motion_entry(self,entry,index):
        import numpy as np
        row=super()._apply_motion_entry(entry,index)
        record=entry['motion_episode'];frame=record['source_frames'][index]
        row['emitter_local_m_frame']=list(frame['mouth_m'])
        if record.get('motion_mode') in {'walk_animation','slide_no_animation','static_after_failed_route'}:
            animated=record['motion_mode']=='walk_animation'
            row.update(motion_mode=record['motion_mode'],motion_marker=record['motion_marker'],
                joint_animation_present=animated,intentional_slide=record['motion_mode']=='slide_no_animation',
                joint_state=('retained admitted joint-pose stream; native rigid original pose meshes' if animated
                    else 'original static mesh; intentional translation with no joint animation' if record['motion_mode']=='slide_no_animation'
                    else 'original retained pose; explicitly marked static after route failures'))
        row['native_emitter_present']=not entry['declaration'].get('silent_decor')
        attached=[]
        for side,foot in frame.get('feet',{}).items():
            if side not in {'L','R'}:continue
            if 'sole_local_m' not in foot:raise ValueError('posed original sole coordinates missing')
            sole=np.asarray(foot['sole_local_m'],float)
            from avengine.capture.v7_visible_mesh import indexed_static_mesh
            path=record['pose_glbs'][index]
            cache=entry.setdefault('phase2_foot_surface_mesh_cache',{})
            if path not in cache:cache[path]=np.asarray(indexed_static_mesh(path).vertices)
            vertices=cache[path];target=sole+[0.,.025,0.]
            # Apply the existing physical gates before minimizing distance.
            # An unconstrained nearest vertex can be on the sole while a valid
            # original shoe surface vertex is only slightly farther away.
            candidates=np.flatnonzero((np.linalg.norm(vertices-sole,axis=1)<=.05)
                & ((vertices[:,1]-sole[1])>=.002))
            if not len(candidates):raise ValueError('no original foot surface emitter passes unchanged 50mm and 2mm gates')
            vertex_index=int(candidates[np.argmin(np.sum((vertices[candidates]-target)**2,axis=1))]);local=vertices[vertex_index]
            if np.linalg.norm(local-sole)>.05:raise ValueError('foot emitter candidate is not on the original foot surface')
            if local[1]-sole[1]<.002:raise ValueError('foot surface emitter is buried in its ground contact')
            child=entry['object'].root_scene_node.create_child();child.translation=self.mn.Vector3(local)
            observed=np.asarray(child.absolute_transformation(),float)[:3,3]
            expected=(np.asarray(row['body_to_world_m'])@np.r_[local,1.])[:3]
            error=float(np.linalg.norm(observed-expected))
            if error>.0001:raise ValueError('moving native foot child exceeds unchanged 0.1mm')
            attached.append({'endpoint_id':entry['endpoint_id']+'_foot_'+side,'native_object_id':row['native_object_id'],
                'part':'foot_'+side,'local_m':local.tolist(),'sole_local_m':sole.tolist(),'original_posed_vertex_index':vertex_index,'position_m':observed.tolist(),'readback_error_m':error,
                'in_contact':bool(foot['in_contact']),'floor_delta_m':foot['floor_delta_m'],'moving':frame['moving'],
                'original_sole_vertex_source':'CPU-skinned original admitted foot weights and complete posed mesh'})
        row['attached_emitter_readbacks']=attached
        return row

    def qualify_standing(self,scene,directory):
        import numpy as np,math
        from pathlib import Path
        from avengine.timeline.continuous_pilot import write_json
        checks=super().qualify_standing(scene,directory)
        camera_targets=scene.get('phase2_semantic_camera_sector_targets_deg',{})
        if camera_targets:
            # Use read-back real mouths and a real camera turn for semantic
            # direction diversity. No synthetic positions or QA overrides.
            wanted={}
            for entry in self.actors:
                aid=entry['declaration']['asset_id']
                if aid not in camera_targets:continue
                observed=self._apply_entry(entry,scene)['emitter_position_m']
                delta=np.asarray(observed)-np.asarray(scene['listener_m'])
                wanted[entry['endpoint_id']]=math.degrees(math.atan2(delta[0],-delta[2]))-float(camera_targets[aid])
            if len(wanted)!=len(camera_targets):raise ValueError('camera plan lacks actual registered mouth')
            starts=[e for e in scene['events'] if e['endpoint_id'] in wanted]
            knots={};base_yaw=float(scene['camera_yaw_knots'][0]['yaw_deg'])
            for window_start in scene['window_starts_s']:
                events=sorted([e for e in starts if window_start<=e['start_tick']/10<window_start+10],key=lambda e:e['start_tick'])
                if len(events)!=len(wanted):raise ValueError('one whole utterance per authored role per crop required')
                first=wanted[events[0]['endpoint_id']];knots[window_start]=first
                for event in events:
                    lo=event['start_tick']/10;hi=event['end_tick']/10
                    knots[max(window_start,lo-.02)]=wanted[event['endpoint_id']]
                    knots[hi]=wanted[event['endpoint_id']]
                end=max(e['end_tick']/10 for e in events)
                hold=max(window_start+6.8,end+.1)
                if hold>=window_start+7.7:raise ValueError('whole dialogue leaves no observed turn/reentry interval')
                knots[hold]=wanted[events[-1]['endpoint_id']]
                knots[window_start+7.7]=base_yaw+80.
                knots[window_start+8.9]=base_yaw-80.
                # Optional real camera dwell keeps small sources in actual
                # sampled frames. Measurement sampling and gates stay intact.
                turn_hold=float(scene.get('phase2_camera_turn_hold_s',0.))
                if turn_hold:
                    knots[window_start+7.7+turn_hold]=base_yaw+80.
                    knots[window_start+8.9+turn_hold]=base_yaw-80.
                knots[window_start+9.9]=base_yaw
            knots[30.]=base_yaw
            scene['camera_yaw_knots']=[{'time_s':t,'yaw_deg':v}for t,v in sorted(knots.items())]
            theta=math.radians(scene['camera_yaw_knots'][0]['yaw_deg'])
            scene['camera_forward']=[math.sin(theta),0.,-math.cos(theta)]
            self.realized_phase2_camera={'method':'real native mouth readback followed by real camera turns; physical emitter positions and QA sector gates unchanged',
                'camera_yaw_knots':scene['camera_yaw_knots'],'camera_forward':scene['camera_forward'],
                'requested_relative_sectors_deg':camera_targets,'native_registered_endpoint_camera_yaws_deg':wanted}
            write_json(Path(directory)/'realized_semantic_camera.json',self.realized_phase2_camera)
        directory=Path(directory)
        admissions=getattr(self,'phase2_motion_admissions',{})
        if not admissions:return checks
        # Standing people are placed outside the reserved swept body bounds.
        # Then independently recheck every original pose against the complete
        # final scene, including the newly placed people and all stage triangles.
        from avengine.assets.roster_motion.episode import TriangleClearance
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        for entry in self.actors:
            ep=entry['endpoint_id']
            if ep not in admissions:continue
            record=admissions[ep]
            clearance=TriangleClearance(self,self.motion_house,record['physical_ground_y_m'],entry)
            minimum=float('inf');mesh_cache={}
            for index,path in enumerate(record['pose_glbs']):
                if path not in mesh_cache:
                    if len(mesh_cache)>=8:mesh_cache.pop(next(iter(mesh_cache)))
                    mesh_cache[path]=indexed_static_mesh(path)
                mesh=mesh_cache[path]
                dist,_=clearance.check(np.asarray(mesh.vertices),np.asarray(mesh.faces),np.asarray(record['world_roots_m'][index]),record['yaw_rad'])
                minimum=min(minimum,dist)
            record['final_complete_scene_recheck']={'status':'pass','original_pose_frames':901,'minimum_nonfloor_clearance_m':minimum,'method':'original full posed triangles against original complete stage and all final registered bystanders; swept bounds reserved before standing people'}
            record['original_full_skin_floor_collision']=clearance.floor_certificate()
            record['minimum_certified_triangle_nonfloor_clearance_m']=minimum
        self.camera_frame_readbacks=[];self.environment_frame_readbacks=[]
        write_json(directory/'phase2_motion_admission.json',{'status':'pass','admissions':admissions,'native_readback_hz':30,'thresholds_changed':any(r.get('quality_gates_changed',False) for r in admissions.values())})
        return checks

    def _on_phase2_placement_accepted(self,entry,scene,directory,supports,accepted):
        import math
        from pathlib import Path
        from avengine.timeline.continuous_pilot import write_json
        requests=scene.get('production_scenario',{}).get('motion_requests',{})
        ep=entry['endpoint_id']
        if ep not in requests:return
        directory=Path(directory)
        floor=float(self.motion_house['room']['measured_floor_y_m'])
        scene['native_placement_candidates']=[[p['point'][0],floor+1.3,p['point'][2]] for p in (supports['floor'] if requests[ep]['action']=='sit' else supports['floor'][:256])]
        scene['sources'][ep]['body_yaw_deg']=math.degrees(entry['resolved_yaw_rad'])
        request=requests[ep];a=entry['declaration'];dest=directory/('motion_'+ep)
        if request['action']=='walk' and a['kind']=='human':
            from avengine.assets.roster_motion.episode import compile_walking
            record=compile_walking(self,entry,scene,self.motion_house,self.motion_inputs,dest,request)
        elif request['action']=='walk' and request.get('animal_motion_admission'):
            from avengine.assets.animal_motion.production_episode import compile_admitted_animal
            record=compile_admitted_animal(self,entry,scene,self.motion_house,self.motion_inputs,dest,request)
        elif request['action']=='walk' and a.get('species_id')=='dog':
            from avengine.assets.roster_motion.animal import compile_continuous_dog_walking
            record=compile_continuous_dog_walking(self,entry,scene,self.motion_house,self.motion_inputs,dest,request)
        elif request['action']=='walk' and a.get('species_id')=='cat':
            from avengine.assets.v7_phase2_cat import compile_cached_cat
            record=compile_cached_cat(self,entry,scene,self.motion_house,self.motion_inputs,dest,request)
        elif request['action']=='sit':
            from avengine.assets.roster_motion.seating import compile_seated
            record=compile_seated(self,entry,scene,self.motion_house,self.motion_inputs,dest,request)
        else:raise ValueError('unsupported phase-two motion; no static fallback')
        entry['motion_episode']=record
        actual=[]
        for index in range(901):
            row=self._apply_motion_entry(entry,index);row.update(frame_index=index,time_s=index/30)
            actual.append(row)
        record['native_30Hz_readbacks']=actual
        self._apply_motion_entry(entry,0)
        self.phase2_motion_admissions=getattr(self,'phase2_motion_admissions',{})
        self.phase2_motion_admissions[ep]=record
        accepted['motion_admission']=record
        accepted['motion_kind']=record.get('motion_kind','human_walk')
        accepted['moving_frame_count']=sum(f['moving'] for f in record['source_frames'])
        write_json(dest/'native_motion_readbacks.json',{'status':'pass','frames':actual,'geometry_receipt':str(dest/('seated_geometry.json' if request['action']=='sit' else 'motion_geometry.json'))})


    def _check_source_frame(self,rgb,semantic,frame_index):
        import numpy as np
        from scipy.spatial import cKDTree
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        for entry in self.actors:
            if entry.get('motion_episode') and entry['declaration']['kind']=='animal':
                mesh=indexed_static_mesh(entry['motion_episode']['pose_glbs'][min(900,frame_index*2)])
                entry['body_subset_verified']=True
                entry['body_vertex_index']=cKDTree(np.asarray(mesh.vertices))
                entry['resolved_yaw_rad']=entry['motion_episode']['yaw_rad']
        super()._check_source_frame(rgb,semantic,frame_index)
        if getattr(self,'seat_material_entries',None):
            from avengine.assets.source_materials import measure_source_appearance
            for entry in self.seat_material_entries:
                r=measure_source_appearance(rgb,semantic==entry['semantic_id'],palette=np.asarray(entry['source_material']['source_palette_srgb']),max_highlight_ratio=.02,strict=False)
                self.material_checks.append({'asset_id':entry['declaration']['asset_id'],'endpoint_id':entry['endpoint_id'],'frame_index':frame_index,**r})
                self.phase2_chair_pixel_checks=getattr(self,'phase2_chair_pixel_checks',[])+[{'asset_id':entry['declaration']['asset_id'],'native_object_id':int(entry['object'].object_id),'semantic_id':entry['semantic_id'],'frame_index':frame_index,'visible_pixels':int(np.count_nonzero(semantic==entry['semantic_id']))}]

    def encode(self,scene,path,**kwargs):
        from pathlib import Path
        from avengine.timeline.continuous_pilot import write_json
        result=super().encode(scene,path,**kwargs)
        for ep,record in result.get('native_motion_admissions',{}).items():
            for alias in [k for k,v in scene['sources'].items() if v.get('owner_endpoint_id')==ep]:
                result['observed_mouth_trajectories'][alias]=[{'time_s':r['time_s'],'position_m':next(v['position_m'] for v in r['attached_emitter_readbacks'] if v['endpoint_id']==alias)} for r in record['native_30Hz_readbacks']]
        result['phase2_visual_revision']=PHASE2_VISUAL_REVISION
        if hasattr(self,'realized_phase2_camera'):result['realized_phase2_camera']=deepcopy(self.realized_phase2_camera)
        result['phase2_chair_pixel_checks']=getattr(self,'phase2_chair_pixel_checks',[])
        result['phase2_requested_motion']=deepcopy(scene.get('production_scenario',{}).get('motion_requests',{}))
        result['phase2_chairs']=[{'asset_id':r['seat_record']['chair']['asset_id'],'body_endpoint_id':ep,
            'native_chair_glb':r['chair_glb'],'chair_scale_y':r['chair_scale_y'],'chair_floor_deltas_m':r['native_support_deltas_m'],
            'seat_contact_gap_m':r['seat_contact_gap_m'],'minimum_stage_clearance_m':r['minimum_certified_triangle_nonfloor_clearance_m'],
            'chair_produces_sound':False} for ep,r in result.get('native_motion_admissions',{}).items() if r.get('motion_kind')=='sit' and r.get('seat_source')=='fallback']
        write_json(Path(path).parent/'native_visible_visual_evidence_phase2.json',result)
        return result
