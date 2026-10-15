"""CPU production native drawables with measured support and attached emitters."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
from avengine.capture.v7_registered_native import NativeRegisteredVideo
from avengine.capture.v7_visible_attachment import survey,rotation,check_support,support_points,body_mask_from_native_depth
from avengine.timeline.continuous_pilot import write_json

VISUAL_REVISION='v7-visible-static-pilot-native-v15-original-paw-sole-regions'


def contact_control_policy(vertices,surface,kind):
    if kind=='object':
        axis=0 if surface=='wall' else 1
        thickness=float(np.ptp(vertices[:,axis]))
        return {'burial_m':min(.03,max(.002,thickness/2)),
                'negative_distance_threshold_m':-.00101,
                'authority':'partial thickness burial; must detect penetration beyond the same 1mm Bullet margin used for actual mesh acceptance'}
    return {'burial_m':.1,'negative_distance_threshold_m':-.015,
            'authority':'existing 10cm whole-body buried negative control'}


def attachment_template(vertices,surface,kind=None,landmarks=None,mesh=None):
    axis=0 if surface=='wall' else 1;high=surface=='ceiling';values=vertices[:,axis]
    extreme=float(values.max() if high else values.min());band=min(.002,.05*float(np.ptp(values)))
    contact=vertices[(extreme-values if high else values-extreme)<=band+1e-9]
    if not len(contact):raise ValueError('original attachment-band vertices are missing')
    axes=tuple(x for x in range(3) if x!=axis);lo=contact.min(0);hi=contact.max(0);anchor=(lo+hi)/2;anchor[axis]=extreme
    selected=[];chosen=set();foot_projection=[]
    for da,db in [(0,0),(-.4,-.4),(-.4,.4),(.4,-.4),(.4,.4)]:
        target=anchor.copy();target[axes[0]]+=da*(hi-lo)[axes[0]];target[axes[1]]+=db*(hi-lo)[axes[1]]
        order=np.argsort(np.sum((contact-target)**2,axis=1));index=next((int(i) for i in order if int(i) not in chosen),int(order[0]))
        chosen.add(index);selected.append(contact[index])
    # Both human foot regions and all four quadruped/base regions are probed
    # independently. A global lowest band cannot detect the other foot floating.
    if surface=='floor' and kind in {'human','performer','animal'}:
        feet=[np.asarray(landmarks[k],float) for k in ('foot_0','foot_1','foot_2','foot_3') if landmarks and k in landmarks]
        if kind=='animal' and len(feet)==4:
            from scipy.spatial import cKDTree
            if mesh is None:
                distances,indices=cKDTree(vertices).query(feet);projected=vertices[indices]
            else:
                import trimesh
                projected,distances,_=trimesh.proximity.closest_point(mesh,np.asarray(feet))
            if max(distances)>.015:raise ValueError('registered animal feet are not near the original mesh')
            # Registration markers locate each paw region; their centroids
            # and nearest skin points can lie on the paw's upper surface. The
            # contact probe must be an original bottom vertex in that region.
            centers=np.asarray(feet);separation=np.linalg.norm(centers[:,None,[0,2]]-centers[None,:,[0,2]],axis=-1)
            np.fill_diagonal(separation,np.inf)
            for i,(p,near,d) in enumerate(zip(feet,projected,distances)):
                radius=min(.05,float(separation[i].min())/3)
                ids=np.flatnonzero(np.linalg.norm(vertices[:,[0,2]]-p[[0,2]],axis=1)<=radius)
                if not len(ids):raise ValueError('registered paw region has no original sole vertices')
                index=int(ids[np.argmin(vertices[ids,1])]);sole=vertices[index]
                selected.append(sole)
                foot_projection.append({'registration_point_m':p.tolist(),'nearest_original_surface_point_m':near.tolist(),'distance_m':float(d),
                    'original_surface_point_m':sole.tolist(),'original_vertex_index':index,'paw_region_radius_m':radius,'original_region_vertices':len(ids),
                    'definition':'lowest original vertex within <=50mm and <=one third of adjacent registered paw separation; marker is a region locator, not a contact point'})
        else:
            # Bounding-box axes are geometric; a dense tail/head may move a
            # vertex-density median away from the original four foot regions.
            split_x=float((vertices[:,0].min()+vertices[:,0].max())/2)
            split_z=float((vertices[:,2].min()+vertices[:,2].max())/2)
            regions=[vertices[:,0]<split_x,vertices[:,0]>=split_x]
            if kind=='animal':regions=[x&z for x in regions for z in [vertices[:,2]<split_z,vertices[:,2]>=split_z]]
            for region in regions:
                if not region.any():raise ValueError('original ground support region is empty')
                part=vertices[region];selected.append(part[np.argmin(part[:,1])])
    return anchor,selected,{'definition':'original GLB <=2mm outer attachment band; object-task algorithm',
        'axis':axis,'outer_band_m':band,'contact_vertices':len(contact),'distinct_probe_vertices':len(chosen),
        'probe_local_m':[p.tolist() for p in selected],'registered_foot_original_surface_projection':foot_projection}


class NativeVisibleVideo(NativeRegisteredVideo):
    def __init__(self,house,scene,inputs,assets,**kwargs):
        super().__init__(house,scene,inputs,assets,**kwargs)
        if self.environment_nodes:raise ValueError('integration pilot forbids audio-only SceneGraph nodes')
        self.animal_color_thresholds=inputs.get('animal_color_thresholds')
        self.color_checks=[];self.color_thresholds=inputs['color_thresholds'];self.attachment_checks=[]
        for entry in self.actors:
            a=entry['declaration']
            from avengine.capture.v7_visible_mesh import indexed_static_mesh
            entry['original_mesh']=indexed_static_mesh(a['habitat_glb']);vertices=np.asarray(entry['original_mesh'].vertices)
            if a['kind']=='object' and np.max(np.abs(np.ptp(vertices,axis=0)-np.asarray(a['geometry']['dimensions_m'])))>1e-5:raise ValueError('original GLB node-index dimensions differ from object registration')
            entry['attachment_meta']={**a,'_attachment_template':attachment_template(vertices,a['attachment_surface'],a['kind'],a.get('landmarks_local_m'),entry['original_mesh'])}
            if a['kind']=='animal' and a.get('body_only_glb'):
                from scipy.spatial import cKDTree
                body=indexed_static_mesh(a['body_only_glb']);dist=cKDTree(vertices).query(body.vertices,k=1)[0]
                entry['body_subset_verified']=bool(float(np.max(dist))<=1e-5)
                if entry['body_subset_verified']:entry['body_vertex_index']=cKDTree(np.asarray(body.vertices))

    def _apply_entry(self,entry,scene,frame_index=None):
        from avengine.assets.habitat_capture import _apply_root_with_habitat
        if 'resolved_root_m' not in entry:raise ValueError('drawable has no accepted native support')
        root=np.eye(4);root[:3,:3]=rotation(entry['resolved_yaw_rad']);root[:3,3]=entry['resolved_root_m']
        _apply_root_with_habitat(entry['object'],root,qt=self.qt,mn=self.mn)
        native=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
        error=float(np.max(np.abs(native-root)))
        if error>1e-5:raise ValueError('native drawable root did not apply')
        entry['applied_root_input']=native.copy();a=entry['declaration']
        if not a['silent_decor']:
            if 'emitter_node' not in entry:
                node=entry['object'].root_scene_node.create_child();node.translation=self.mn.Vector3(a['emitter_local_m']);entry['emitter_node']=node
            emitter=np.asarray(entry['emitter_node'].absolute_transformation(),float)[:3,3]
            expected=(native@np.r_[a['emitter_local_m'],1.])[:3];readback_error=float(np.linalg.norm(emitter-expected))
            if readback_error>.0001:raise ValueError('native attached emitter exceeds 0.1mm')
        else:emitter=native[:3,3];readback_error=None
        return {'endpoint_id':entry['endpoint_id'],'actor_id':a['native_instance_id'],
            'native_object_id':int(entry['object'].object_id),'semantic_id':entry['semantic_id'],'asset_id':a['asset_id'],
            'position_m':native[:3,3].tolist(),'body_to_world_m':native.tolist(),'emitter_position_m':emitter.tolist(),
            'native_emitter_readback_error_m':readback_error,'native_emitter_present':not a['silent_decor'],'moving':False,'action_id':'stand' if a['kind']=='human' else 'registered_static',
            'action_sample_index':0,'action_tick':0,'root_max_error':error,'joint_max_error':0.,
            'joint_state':'no_joint_DOFs_original_registered_static_GLB','source_registry_sha256':a['source_registry_sha256'],
            'loaded_artifact_sha256':a['actual_artifact_sha256'][a['habitat_glb']]}

    def _contacts(self,entry,directory,normal):
        """Same complete-mesh compound and Bullet tolerance as registered people."""
        import trimesh
        from scipy.spatial import ConvexHull,QhullError
        local=np.asarray(entry['original_mesh'].vertices);native=entry['applied_root_input'];center=(local.min(0)+local.max(0))/2
        if 'visible_contact_template' not in entry:
            groups={};cells=np.floor(local/.12).astype(int);mesh=trimesh.Scene();covered=set()
            for tri in entry['original_mesh'].faces:groups.setdefault(tuple(cells[int(tri[0])]),set()).update(int(k) for k in tri)
            for i,ids in enumerate(groups.values()):
                covered.update(ids);points=local[sorted(ids)]-center
                try:
                    h=ConvexHull(points);part=trimesh.Trimesh(points,h.simplices,process=False);part.fix_normals()
                except QhullError:
                    part=trimesh.creation.box(np.maximum(np.ptp(points,axis=0),.001));part.apply_translation((points.min(0)+points.max(0))/2)
                mesh.add_geometry(part,geom_name='cell_'+str(i),node_name='cell_'+str(i))
            if len(covered)!=len(local):raise ValueError('collision proxy omitted original mesh vertices')
            path=directory/(entry['endpoint_id']+'_full_mesh.glb');path.write_bytes(mesh.export(file_type='glb'))
            tm=self.sim.get_object_template_manager();attr=tm.create_new_template('visible_contact_'+entry['endpoint_id'])
            attr.render_asset_handle=str(path);attr.collision_asset_handle=str(path);attr.is_collidable=True
            attr.join_collision_meshes=False;attr.margin=.001;attr.compute_COM_from_shape=False
            entry['visible_contact_template']=(tm.register_template(attr),len(groups),str(path))
        tid,hulls,path=entry['visible_contact_template'];rm=self.sim.get_rigid_object_manager()
        from avengine.assets.habitat_capture import _apply_root_with_habitat
        root=native.copy();root[:3,3]=(native@np.r_[center,1.])[:3]
        ids={int(a['object'].object_id) for a in self.actors}
        def contacts(offset):
            # A fresh owned probe isolates each positive/negative control from
            # Bullet's persistent manifold left by the preceding buried pose.
            obj=rm.add_object_by_template_id(tid);matrix=root.copy();matrix[:3,3]+=offset
            obj.motion_type=self.hs.physics.MotionType.DYNAMIC
            _apply_root_with_habitat(obj,matrix,qt=self.qt,mn=self.mn)
            try:
                self.sim.perform_discrete_collision_detection();rows=[]
                for c in self.sim.get_physics_contact_points():
                    pair={int(c.object_id_a),int(c.object_id_b)}
                    if int(obj.object_id) in pair and not pair.intersection(ids):rows.append({'object_id_a':int(c.object_id_a),'object_id_b':int(c.object_id_b),'contact_distance_m':float(c.contact_distance)})
                return rows
            finally:rm.remove_object_by_id(obj.object_id)
        policy=contact_control_policy(local,entry['declaration']['attachment_surface'],entry['declaration']['kind']);burial=policy['burial_m']
        actual=contacts(np.zeros(3));bad=contacts(-normal*burial);raised=contacts(normal*.06)
        vertices=local@native[:3,:3].T+native[:3,3]
        return {'contacts':actual,'known_bad_contacts':bad,'raised_contacts':raised,
                'known_bad_detected':any(c['contact_distance_m']<policy['negative_distance_threshold_m'] for c in bad),
                'skin_pass':all(c['contact_distance_m']>=-.00101 for c in actual),'penetration_beyond_1mm_native_margin_m':max((max(0.,-c['contact_distance_m']-.001) for c in actual),default=0.),'penetration_numeric_epsilon_m':.00001,
                'off_support_clear':not any(c['contact_distance_m']<-.015 for c in raised),
                'native_full_skin_bounds_m':[vertices.min(0).tolist(),vertices.max(0).tolist()],
                'native_skin_vertices':len(local),'native_triangles':len(entry['original_mesh'].faces),
                'full_skin_covered':True,'compound_hulls':hulls,'compound_margin_m':.001,'proxy_path':path,'burial_control_m':burial,'negative_control_policy':policy}

    def _native_pair_pixels(self,scene,entries,forward=None):
        from avengine.timeline.continuous_scene import camera_basis
        f,r,u=camera_basis(np.asarray(scene['camera_forward'] if forward is None else forward,float))
        state=self.hs.AgentState();state.position=np.asarray(scene['listener_m'],dtype=np.float32)
        state.rotation=self.qt.from_rotation_matrix(np.stack((r,u,-f),axis=1));self.sim.get_agent(0).set_state(state)
        semantic=np.asarray(self.sim.get_sensor_observations()['semantic'])
        return {e['declaration']['asset_id']:int(np.count_nonzero(semantic==e['semantic_id'])) for e in entries}

    def _native_observed_pixels(self,scene,entries):
        from avengine.capture.v7_alpha_native import forward_at
        maximum={e['declaration']['asset_id']:0 for e in entries}
        observed=[];seen=set()
        for knot in scene['camera_yaw_knots']:
            yaw=round(knot['yaw_deg']%360.,6)
            if yaw in seen:continue
            seen.add(yaw);pixels=self._native_pair_pixels(scene,entries,forward_at(scene,knot['time_s']))
            observed.append({'time_s':knot['time_s'],'pixels':pixels})
            maximum={k:max(maximum[k],pixels[k]) for k in maximum}
            # Opt-in proposal search can stop after an actual positive witness
            # for every current entity. The original existential 100px gate
            # is identical; negative cases still inspect the complete path.
            # Final video and all material/color/clock observations are intact.
            if (self.motion_inputs.get('visible_placement_stop_after_native_visibility_witness',False) or self.motion_inputs.get('independent_probe_camera_search',False)) and all(n>=100 for n in maximum.values()):
                break
        return maximum,observed

    def qualify_standing(self,scene,directory):
        from shapely.geometry import shape,Point,Polygon,MultiPolygon
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        r=SimpleNamespace(habitat_sim=self.hs,magnum=self.mn,pathfinder=self.pathfinder)
        floor=float(self.motion_house['room']['measured_floor_y_m']);poly=shape(self.motion_house['room']['floor_polygon'])
        from avengine.capture.v7_visible_semantic_support import SemanticSupports
        semantic_support=getattr(self,'_original_semantic_support_cache',None)
        if semantic_support is None:
            semantic_support=SemanticSupports(self.motion_house)
            self._original_semantic_support_cache=semantic_support
        r.semantic_anchors=dict(semantic_support.anchors)
        origin=[scene['listener_m'][0],floor,scene['listener_m'][2]]
        # Independent probes park every inserted drawable below the house.
        # Their room-restricted, original semantic survey is immutable; cache
        # that query, while every proposed placement still rechecks its own
        # native support rays, labelled triangles, clearance, and contacts.
        probe_cache=bool(self.motion_inputs.get('cache_native_support_survey_for_independent_probes') or self.motion_inputs.get('reuse_native_room_support_survey',False))
        qualified_key=(tuple(origin),floor,poly.wkb_hex)
        qualified=getattr(self,'_independent_qualified_survey_cache',None)
        if probe_cache and qualified is not None and qualified[0]==qualified_key:
            from copy import deepcopy
            supports=deepcopy(qualified[1])
        else:
            cached=getattr(self,'_independent_probe_survey_cache',None)
            if probe_cache and cached is not None and cached[0]==origin:
                from copy import deepcopy
                supports=deepcopy(cached[1])
            else:
                supports=survey(self.sim,r,{'object_position':origin})
                if probe_cache:
                    from copy import deepcopy
                    self._independent_probe_survey_cache=(origin,deepcopy(supports))
            parts=list(poly.geoms) if hasattr(poly,'geoms') else [poly]
            room_extent=MultiPolygon([Polygon(p.exterior) for p in parts])
            room_extent_buffer=room_extent.buffer(.01)
            free_floor_buffer=poly.buffer(.01)
            # The floor polygon describes free floor. Mounts are admitted by the
            # original in-room ray origin, and tabletops by the room exterior, so
            # furniture holes and a navmesh wall clearance do not erase supports.
            def in_room(key,p):
                q=p.get('originating_native_floor_m',p['point'])
                authority=room_extent if key=='table' else poly
                return (room_extent_buffer if key=='table' else free_floor_buffer).covers(Point(q[0],q[2]))
            # Floor type is bound to the measured room level, never a bed/table top.
            for key in ('floor','table','wall','ceiling','doorbell_wall','high_wall','window_wall'):
                supports[key]=[p for p in supports[key] if in_room(key,p) and abs(p.get('floor_y',floor)-floor)<=.08]
            semantic_candidate_surfaces=('floor','table','wall','ceiling','doorbell_wall','high_wall','window_wall') if self.motion_inputs.get('candidate_floor_semantic_prefilter') else ('table','wall','ceiling','doorbell_wall','high_wall','window_wall')
            # Sampling budget only: query a deterministic spread of the
            # original surface proposals, then retain all original qualifiers.
            survey_limit=self.motion_inputs.get('candidate_semantic_survey_max_points_per_surface')
            if survey_limit is not None:
                survey_limit=int(survey_limit)
                if not 1<=survey_limit<=1024:raise ValueError('invalid semantic survey sampling budget')
                for key in semantic_candidate_surfaces:
                    points=supports[key]
                    if len(points)>survey_limit:
                        indices=np.linspace(0,len(points)-1,survey_limit,dtype=int)
                        supports[key]=[points[int(i)] for i in indices]
            for key in semantic_candidate_surfaces:
                kind='wall' if key.endswith('wall') else key
                batch_size=self.motion_inputs.get('candidate_semantic_filter_batch_size')
                if batch_size is None:
                    supports[key]=semantic_support.qualify_points(kind,supports[key])
                else:
                    # Candidate filtering only: pass every point, in its
                    # original order, to the unchanged semantic qualifier.
                    batch_size=int(batch_size)
                    if not 1<=batch_size<=1024:raise ValueError('invalid semantic candidate batch size')
                    points=supports[key]
                    supports[key]=[row for offset in range(0,len(points),batch_size)
                                   for row in semantic_support.qualify_points(kind,points[offset:offset+batch_size])]

            if probe_cache:
                from copy import deepcopy
                self._independent_qualified_survey_cache=(qualified_key,deepcopy(supports))
        write_json(directory/'native_support_survey.json',supports)
        bounds=[];checks=[];attempts=[];placed=[]
        # Pending original drawables are parked during layout search only.
        # Every final source must receive a checked placement before any media.
        for pending in self.actors:
            pending["object"].translation=self.mn.Vector3(0.,-1000.,0.)
            if scene.get('phase2_profile'):pending['phase2_layout_qualified']=False
        paired=[e for e in self.actors if e['declaration'].get('controlled_shirt')]
        require_pair=bool(self.motion_inputs.get('required_simultaneous_controlled_pair')) and len(paired)>1
        placed_pair=[]
        # Restrictive mount surfaces and large devices are assigned before
        # people. Original source ordering and endpoint identities stay fixed.
        def layout_order(entry):
            a=entry['declaration'];surface=a['attachment_surface']
            priority={'ceiling':0,'wall':1,'table':2,'floor':3}[surface]
            if surface=='floor' and a['kind']!='object':priority=4 if a['kind'] in {'human','performer'} else 5
            if scene.get('phase2_profile') and surface=='floor':
                priority=2.3 if a['kind']=='object' else 2.4 if a['kind']=='performer' else 2.5 if a['endpoint_id'] in scene['production_scenario'].get('motion_requests',{}) else 2.7
            if scene.get('phase2_profile') and a['asset_id'] in scene.get('native_attachment_supports',{}):
                priority=-1  # Reserve an explicitly requested qualified site; recheck every physical gate.
            from avengine.capture.v7_occlusion_demand import relation_roles
            targets,foreground=relation_roles(scene,a['asset_id'])
            if targets:priority=2.55
            if foreground:priority=2.6
            extent=np.ptp(np.asarray(entry['original_mesh'].vertices),axis=0)
            return priority,-float(np.prod(extent))
        for entry in sorted(self.actors,key=layout_order):
            a=entry['declaration'];meta=entry['attachment_meta'];key=a['attachment_surface']
            if a.get('object_type')=='doorbell_chime_unit':key='doorbell_wall'
            elif a.get('object_type')=='air_conditioner' and a.get('form_factor')=='wall_split':key='high_wall'
            candidates=list(supports.get(key,[]))
            pinned=scene.get('native_attachment_supports',{}).get(a['asset_id'])
            if pinned is not None:
                if pinned.get('surface')!=a['attachment_surface']:raise ValueError('pinned attachment surface differs from registration')
                candidates=[dict(pinned,point=pinned['original_surface_point_m']),*candidates]
            if key=='floor':candidates=list({tuple(round(v,5) for v in p['point']):p for p in candidates}.values())
            listener=np.asarray(scene['listener_m']);rejections={}
            def reject(reason):rejections[reason]=rejections.get(reason,0)+1
            from avengine.capture.v7_alpha_native import forward_at
            camera_path_vectors=[(forward_at(scene,k['time_s']),None) for k in scene['camera_yaw_knots']]
            camera_path_vectors=[(f,np.cross(f,[0.,1.,0.])) for f,_ in camera_path_vectors]
            def visible_from_path(delta):
                for f,right in camera_path_vectors:
                    depth=float(delta@f)
                    if depth>0 and abs(math.degrees(math.atan2(float(delta@right),depth)))<=37. and abs(math.degrees(math.atan2(delta[1],depth)))<=24.:return True
                return False
            target=scene.get('phase2_semantic_azimuth_targets_deg',{}).get(a['asset_id']) if scene.get('phase2_profile') else None
            distance_target=float(scene.get('native_attachment_distance_targets_m',{}).get(a['asset_id'],2.5)) if scene.get('phase2_profile') else 2.5
            def candidate_score(p):
                p=dict(p)
                if key=='floor':
                    delta=listener-np.asarray(p['point']);p['yaw_rad']=math.atan2(-delta[0],-delta[2]) if a['kind'] in {'human','performer','animal'} else math.atan2(-delta[2],delta[0])
                root,yaw,normal,_=support_points(meta,p);rep=root+rotation(yaw)@np.asarray(a['emitter_local_m']) if not a['silent_decor'] else root+[0,.5,0]
                delta=rep-listener;f=np.asarray(scene['camera_forward']);right=np.cross(f,[0.,1.,0.]);depth=float(delta@f)
                visible=visible_from_path(delta)
                central=depth>0 and abs(math.degrees(math.atan2(float(delta@right),depth)))<=20.
                az=math.degrees(math.atan2(float(delta@right),depth))
                from avengine.capture.v7_occlusion_demand import proposal_score
                relation_score=proposal_score(scene,entry,root,placed)
                if relation_score is not None:return (not visible,*relation_score)
                if target is not None:
                    return (not visible,(abs(az-float(target))/3.4)**2+(abs(float(np.linalg.norm(delta[[0,2]]))-distance_target)/.4)**2)
                return (not visible,not central if a['kind'] in {'human','performer'} or self.motion_inputs.get('independent_probe_camera_search') else False,abs(float(np.linalg.norm(delta[[0,2]]))-distance_target))
            candidates.sort(key=candidate_score)
            accepted=None;native_attempts=0;search_exhausted=False
            contact_budget=int(self.motion_inputs.get('visible_placement_max_contact_attempts_per_asset',64))
            if not 1<=contact_budget<=1280:raise ValueError('invalid native placement search budget')
            ordinary=[p for p in candidates if pinned is None or p!=dict(pinned,point=pinned['original_surface_point_m'])]
            proposals=([dict(pinned,point=pinned['original_surface_point_m'],_outward_clearance_m=pinned['_outward_clearance_m'])] if pinned is not None else [])
            support_budget=int(self.motion_inputs.get('visible_placement_max_support_candidates_per_asset',320))
            if not 1<=support_budget<=320:raise ValueError('invalid support candidate sampling budget')
            proposals.extend(dict(p,_outward_clearance_m=offset) for p in ordinary[:support_budget] for offset in (0.,.002,.005,.010,.014))
            proposal_budget=self.motion_inputs.get('visible_placement_max_proposals_per_asset')
            if proposal_budget is not None and not 1<=int(proposal_budget)<=1601:
                raise ValueError('invalid support proposal search budget')
            for proposal_index,support in enumerate(proposals):
                if proposal_budget is not None and proposal_index>=int(proposal_budget):
                    search_exhausted=True;reject('bounded_support_proposal_search_exhausted');break
                support=dict(support);support['original_surface_point_m']=support['point'];support['point']=(np.asarray(support['point'])+np.asarray(support['normal'])*support['_outward_clearance_m']).tolist()
                if key=='floor':
                    delta=listener-np.asarray(support['point'])
                    support['yaw_rad']=math.atan2(-delta[0],-delta[2]) if a['kind'] in {'human','performer','animal'} else math.atan2(-delta[2],delta[0])
                # Pure candidate filtering: the support transform is identical
                # to check_support's proposed root. Every surviving candidate
                # still runs all original rays, contacts and real native pixels.
                proposed_root,proposed_yaw,_,_=support_points(meta,support)
                proposed_rep=(proposed_root+rotation(proposed_yaw)@np.asarray(a['emitter_local_m'])) if not a['silent_decor'] else proposed_root+np.array([0.,.5,0.])
                if not visible_from_path(proposed_rep-listener):
                    reject('outside_observed_camera_path');continue
                placement=check_support(self.sim,r,meta,support)
                if placement is None:
                    reject("attachment_rays");continue
                # An optional recipe-side filter can request tighter quadruped
                # support. The original 15mm admission check remains unchanged.
                paw_limit=self.motion_inputs.get('quadruped_candidate_max_paw_gap_m')
                if paw_limit is not None and a['kind']=='animal' and len(a.get('landmarks_local_m',{})) and all(k in a['landmarks_local_m'] for k in ('foot_0','foot_1','foot_2','foot_3')):
                    paw_limit=float(paw_limit)
                    if not 0.<paw_limit<=.015:raise ValueError('paw candidate filter must be within the original 15mm gate')
                    if any(abs(v)>paw_limit for v in placement['contact_deltas_m'][-4:]):
                        reject('quadruped_paw_candidate_gap');continue
                if key=='doorbell_wall':
                    from avengine.capture.v7_visible_attachment import anchor_distance
                    point=placement['support']['original_surface_point_m']
                    anchors=supports['native_semantic_anchors']['door']
                    nearby=[d for d in anchors if anchor_distance(point,d)<=1.2]
                    if not nearby:
                        reject('doorbell_not_within_1_2m_of_actual_3d_door');continue
                    nearest=min(nearby,key=lambda d:anchor_distance(point,d))
                    placement['support']['native_door_anchor']=nearest
                    placement['support']['distance_to_native_door_aabb_m']=anchor_distance(point,nearest)
                if key=='floor' and any(abs(p[1]-floor)>.08 or not free_floor_buffer.covers(Point(p[0],p[2])) for p in placement['support_raycast_points_m']):
                    reject('ground_probes_outside_measured_free_floor');continue
                kind='wall' if key.endswith('wall') else key
                typed=semantic_support.qualify_points(kind,[{'point':p} for p in placement['support_raycast_points_m']])
                if len(typed)!=len(placement['support_raycast_points_m']):
                    reject('original_semantic_type_of_attachment_probes');continue
                placement['typed_support_probe_readbacks']=typed

                root=np.asarray(placement['root_m']);yaw=placement['yaw_rad']
                representative=(root+rotation(yaw)@np.asarray(a['emitter_local_m'])) if not a['silent_decor'] else root+np.array([0.,.5,0.])
                delta=representative-listener;forward=np.asarray(scene['camera_forward']);right=np.cross(forward,[0.,1.,0.]);depth=float(delta@forward)
                horizontal=math.degrees(math.atan2(float(delta@right),depth));vertical=math.degrees(math.atan2(delta[1],max(1e-9,depth)))
                if target is not None and abs(horizontal-float(target))>3.4:
                    reject('outside_requested_distinct_semantic_sector');continue
                if not visible_from_path(delta):
                    reject("outside_observed_camera_path");continue
                if require_pair and a.get('controlled_shirt') and (depth<=0 or abs(horizontal)>35.):
                    reject('outside_color_pair_hold_view');continue
                local=np.asarray(entry['original_mesh'].vertices)
                transformed=local@rotation(yaw).T+root;lo,hi=transformed.min(0),transformed.max(0)
                if any(np.all(np.minimum(hi,bhi)>np.maximum(lo,blo)) for blo,bhi in bounds):
                    reject("other_registered_entity_bounds");continue
                if np.linalg.norm((lo+hi)/2-listener)<.5:
                    reject("listener_clearance");continue
                if native_attempts>=contact_budget:
                    search_exhausted=True;reject('bounded_native_search_exhausted');break
                native_attempts+=1
                entry.update(resolved_root_m=root,resolved_yaw_rad=yaw);self._apply_entry(entry,scene)
                native_pixels,pixel_views=self._native_observed_pixels(scene,[*placed,entry])
                if any(n<100 for n in native_pixels.values()):
                    reject('actual_native_registered_or_stage_occlusion');continue
                from avengine.capture.v7_occlusion_demand import measure_requested_relations
                relation_witness=measure_requested_relations(self,scene,[*placed,entry])
                if not relation_witness['satisfies_request']:
                    write_json(directory/('occlusion_proposal_'+a['endpoint_id']+'_'+str(proposal_index).zfill(4)+'.json'),
                        {'status':'rejected_layout_proposal','candidate_asset_id':a['asset_id'],
                         'proposed_root_m':root.tolist(),'witness':relation_witness,
                         'qa_predicates_changed':False})
                    reject('requested_registered_occlusion_not_observed');continue
                native_contacts=self._contacts(entry,directory,np.asarray(placement['normal']))
                attempt={'asset_id':a['asset_id'],'endpoint_id':a['endpoint_id'],'placement':placement,
                         'skin_pass':native_contacts['skin_pass'],'known_bad_detected':native_contacts['known_bad_detected'],
                         'off_support_clear':native_contacts['off_support_clear'],'real_native_layout_pixels':native_pixels,'observed_pixel_views':pixel_views,'minimum_actual_contact_m':min((c['contact_distance_m'] for c in native_contacts['contacts']),default=None),'minimum_raised_contact_m':min((c['contact_distance_m'] for c in native_contacts['raised_contacts']),default=None)};attempts.append(attempt)
                if not native_contacts['skin_pass'] or not native_contacts['known_bad_detected'] or not native_contacts['off_support_clear']:
                    reject('native_contact_controls');continue
                emitter=self._apply_entry(entry,scene)['emitter_position_m'];admit=getattr(self,'acoustic_position_admission',None)
                acoustic={'status':'not_applicable_silent_decor_no_emitter'} if a['silent_decor'] else admit(a['endpoint_id'],np.asarray(emitter)) if admit else {'status':'not_requested'};attempt['acoustic']=acoustic
                if acoustic['status']=='blocked':
                    reject('native_RIR_direction');continue
                if require_pair and a.get('controlled_shirt'):
                    pixels=self._native_pair_pixels(scene,[*placed_pair,entry]);attempt['simultaneous_color_pair_pixels']=pixels
                    if any(n<100 for n in pixels.values()):
                        reject('actual_native_color_pair_occlusion');continue
                    placed_pair.append(entry)
                accepted={'status':'pass','asset_id':a['asset_id'],'endpoint_id':a['endpoint_id'],
                          'attachment_surface':a['attachment_surface'],'support_type':support['surface'],
                          'attachment_worst_error_m':max(abs(v) for v in placement['contact_deltas_m']),
                          'contact_tolerance_m':.015,'placement':placement,**native_contacts,
                          'resolved_body_root_world_m':entry['applied_root_input'][:3,3].tolist(),
                          'floor_type_authority':'native navmesh + all native support rays within 80mm of measured room floor/free polygon + original floor or ground-covering triangles <=15mm; contact itself <=15mm',
                          'table_type_authority':'actual native horizontal patch + original scan table/desk/counter-labelled triangles within <=15mm',
                          'motion_admission':'original static complete mesh; no new walking/sitting motion'}
                if relation_witness['relations']:accepted['requested_occlusion_witness']=relation_witness
                if scene.get('phase2_profile'):
                    entry['phase2_layout_qualified']=True
                    hook=getattr(self,'_on_phase2_placement_accepted',None)
                    if hook:hook(entry,scene,directory,supports,accepted)
                    if entry.get('motion_episode'):
                        motion_bounds=np.asarray(entry['motion_episode']['full_skin_bounds_m'])
                        lo,hi=motion_bounds[0]-.02,motion_bounds[1]+.02
                bounds.append((lo,hi));checks.append(accepted);placed.append(entry);break
            if accepted is None:
                write_json(directory/'attachment_rejection.json',{'status':'fail','failed_asset_id':a['asset_id'],'surface':key,'candidates':len(candidates),'attempts':attempts,'accepted_checks':checks,'rejection_counts':rejections})
                raise ValueError('no strict native '+key+' placement for '+a['asset_id'])
        if require_pair:
            pixels=self._native_pair_pixels(scene,paired)
            write_json(directory/'simultaneous_controlled_pair.json',{'status':'pass' if all(n>=100 for n in pixels.values()) else 'fail','native_semantic_pixels':pixels,'frame_time_s':0.,'minimum_pixels':100})
            if any(n<100 for n in pixels.values()):raise ValueError('final registered scene occludes a required simultaneous color pair')
        self.standing_checks=checks;self.attachment_checks=checks;self.camera_frame_readbacks=[];self.environment_frame_readbacks=[]
        write_json(directory/'standing_admission.json',{'status':'pass','checks':checks,'placement_attempts':attempts})
        return checks

    def _check_source_frame(self,rgb,semantic,frame_index):
        from avengine.assets.source_materials import measure_source_appearance
        from avengine.assets.v7_visible_color import measure
        from avengine.qa.v7_visual_admission import measure_native_color,COLOR_HUE_DEGREES
        obs=self.sim.get_sensor_observations();depth=np.asarray(obs['depth']);sensor=self.sim._sensors['rgb']._sensor_object
        P=np.asarray(sensor.render_camera.projection_matrix);W=np.asarray(sensor.node.absolute_transformation())
        for entry in self.actors:
            a=entry['declaration'];mask=semantic==entry['semantic_id'];material=entry['source_material']
            measured=measure_source_appearance(rgb,mask,palette=np.asarray(material['source_palette_srgb']),max_highlight_ratio=.02,strict=False)
            self.material_checks.append({'asset_id':a['asset_id'],'endpoint_id':entry['endpoint_id'],'frame_index':frame_index,**measured})
            if a.get('label_color') is None:continue
            if a['kind']=='object':
                body=body_mask_from_native_depth(mask,depth,P,W,entry['applied_root_input'][:3,3],entry['resolved_yaw_rad'],a)
                result=measure(rgb,body,self.color_thresholds)
            elif a['kind']=='animal':
                from avengine.assets.v7_visible_animal_color import measure as animal_measure
                body=mask.copy();yy,xx=np.where(mask)
                if not entry.get('body_subset_verified') or not self.animal_color_thresholds:
                    result={'status':'blocked','label':None,'reason':'original animal body subset/colour boundaries unavailable'}
                else:
                    z=depth[yy,xx];h,w=mask.shape;cam=np.column_stack([((xx+.5)/w*2-1)*z/P[0,0],(1-(yy+.5)/h*2)*z/P[1,1],-z,np.ones(len(z))]);world=cam@W.T;local=(world[:,:3]-entry['applied_root_input'][:3,3])@rotation(entry['resolved_yaw_rad']);dist=entry['body_vertex_index'].query(local,k=1)[0];body[yy,xx]=dist<=.006
                    result=animal_measure(rgb,body,self.animal_color_thresholds);result['label']=result.get('measured_label');result['roi_pixels']=result['body_pixels_after_exclusion'];result['body_ROI']='actual native semantic/depth pixels within 6mm of verified original body-only vertices; silent perch excluded'
            elif a.get('controlled_shirt') and a['label_color'] in COLOR_HUE_DEGREES:
                try:result=measure_native_color(rgb,semantic,entry['semantic_id'],a['label_color'])
                except IndexError:result={'status':'blocked','reason':'empty_native_shirt_ROI'}
                result['label']=a['label_color'] if result.get('status')=='pass' else None
            elif a.get('controlled_shirt'):
                yy,_=np.where(mask);body=mask.copy()
                if len(yy):
                    lo,hi=yy.min(),yy.max();grid=np.arange(mask.shape[0])[:,None];body&=(grid>=lo+.20*(hi-lo))&(grid<=lo+.55*(hi-lo))
                result=measure(rgb,body,self.color_thresholds)
            else:result=measure(rgb,mask,self.color_thresholds)
            result['roi_pixels']=int(result.get('roi_pixels',result.get('body_pixels',0)))
            self.color_checks.append({'asset_id':a['asset_id'],'actor_id':a['native_instance_id'],'frame_index':frame_index,
                'expected_color':a['label_color'],'registered_label_color':a['registered_label_color'],
                'matched':result.get('status')=='pass' and result.get('label')==a['label_color'],
                'engineering_only':a['label_color'] in {'gray','gray_silver','brown','wood_brown','beige'},**result})

    def encode(self,scene,path,**kwargs):
        result=super().encode(scene,path,**kwargs)
        from avengine.assets.v7_visible_color import rebuild_appearance_review
        assets=[e['declaration'] for e in self.actors]
        review,eligibility=rebuild_appearance_review(self.color_checks,assets,
            (self.animal_color_thresholds or {}).get('min_pixels',200))
        reviews=review['actors'];result['visible_color_eligibility']=eligibility
        result['native_appearance_review']={'actors':reviews};result['visible_color_checks']=self.color_checks
        result['visible_attachment_checks']=self.attachment_checks;result['visible_visual_revision']=VISUAL_REVISION
        write_json(Path(path).parent/'native_color_checks.json',{'checks':self.color_checks,'review':result['native_appearance_review'],'thresholds':self.color_thresholds})
        write_json(Path(path).parent/'native_visible_visual_evidence.json',result)
        return result
