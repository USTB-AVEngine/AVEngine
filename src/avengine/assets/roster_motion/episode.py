"""New-house contact compilation from admitted clips, with exact triangle gates.

Reference-route receipts do not admit a new placement. Every 30 Hz pose is
checked against the original stage triangles, actual floor and native NavMesh.
Ordinary routes are solved from endpoints through the shared room planner.
"""
from __future__ import annotations
import copy, hashlib, math
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.production_episode import OWNER_PRODUCTION_QUALITY
from avengine.assets.roster_motion.io import read_glb, viewbytes
from avengine.assets.roster_motion.pose import pose, animation_times, accessor, rigid_glb
from avengine.assets.roster_motion.contact import FootPlant, yaw_matrix, foot_min, foot_masks
from avengine.assets.roster_motion.appearance import apply_appearance
from avengine.capture.v7_registered_native import loaded_mesh
from avengine.timeline.continuous_pilot import write_json
from avengine.construction.contact_policy import (
    POLICY, FLOOR_STEP_RANGE_M, contact_quality_observation, MotionRouteUnavailable,
)

REVISION='admitted-contact-clip-new-house-full-triangle-native-v1'


class PhysicalFloor:
    def __init__(self, video, reference, ignore=(), reference_tolerance_m=None):
        if reference_tolerance_m is None:reference_tolerance_m=HUMAN_QUALITY.get('floor_level_tolerance_m',.015)
        self.video=video;self.reference=float(reference);self.ignore=set(ignore)
        self.reference_tolerance_m=max(float(reference_tolerance_m),FLOOR_STEP_RANGE_M)
        self.current_root_floor_m=None;self.missing_peripheral_support_samples=0
        self.original_ray=None;self.native_ray_agreements=[];self.native_ray_count=0
        if getattr(video,'motion_inputs',{}).get('phase2'):
            from avengine.assets.v7_phase2_floor import OriginalVerticalStageRay
            h=video.motion_house
            path=Path(h.get('input',{}).get('visual_source') or Path(h['input']['scene_directory'])/(h['input']['scan_id']+'.glb'))
            rays=getattr(video,'phase2_original_vertical_ray_cache',{})
            if str(path)not in rays:
                from avengine.assets.v7_phase2_floor import original_native_stage_mesh
                mesh,basis=original_native_stage_mesh(video,path)
                rays[str(path)]=OriginalVerticalStageRay(path,mesh=mesh,transform_evidence=basis)
            video.phase2_original_vertical_ray_cache=rays
            self.original_ray=rays[str(path)]
    def __call__(self, point):
        v=self.video;p=np.asarray(point,float)
        hits=v.sim.cast_ray(v.hs.geo.Ray(v.mn.Vector3(p+[0.,.2,0.]),v.mn.Vector3(0.,-1.,0.)),.85).hits
        hits=[h for h in hits if int(h.object_id) not in self.ignore]
        if not hits:
            if self.current_root_floor_m is None:raise ValueError('motion support ray has no actual stage floor')
            self.missing_peripheral_support_samples+=1
            return self.current_root_floor_m
        y=float(hits[0].point[1]);self.native_ray_count+=1
        if self.original_ray is not None:
            original=float(self.original_ray.many([p])[0])
            error=abs(original-y);self.native_ray_agreements.append(error)
            # Both readings are retained; the original visual mesh supplies Y.
            y=original
        if abs(y-self.reference)>self.reference_tolerance_m:raise ValueError('motion support ray hit furniture or a different floor level')
        return y
    def many(self, points):
        if self.original_ray is None:return np.array([self(p) for p in points])
        try:
            y=self.original_ray.many(points)
        except ValueError as error:
            if self.current_root_floor_m is None or 'no vertical support' not in str(error):raise
            values=[]
            for point in points:
                try:values.append(float(self.original_ray.many([point])[0]))
                except ValueError as missing:
                    if 'no vertical support' not in str(missing):raise
                    self.missing_peripheral_support_samples+=1
                    values.append(self.current_root_floor_m)
            y=np.asarray(values)
        if np.any(abs(y-self.reference)>self.reference_tolerance_m):raise ValueError('candidate support hit furniture or a different floor level')
        return y

    def ground_roots(self, positions):
        # No fallback at body roots: every frame must have actual ground.
        self.current_root_floor_m=None
        result=np.asarray(positions,float).copy()
        result[:,1]=self.many(result)
        return result

    def observation(self):
        return contact_quality_observation(
            native_original_worst_difference_m=max(self.native_ray_agreements,default=0.),
            missing_peripheral_support_samples=self.missing_peripheral_support_samples,
            peripheral_missing_support_policy='IK uses the independently measured current root floor; no hit fabricated',
            root_support_verified=True)



def triangle_mesh(d,b,parts):
    vertices=[];faces=[];offset=0
    for part in parts:
        prim=d['meshes'][d['nodes'][part['node']]['mesh']]['primitives'][part['primitive']]
        vertices.append(part['positions']);ix=accessor(d,b,prim['indices']).astype(np.int32).reshape(-1,3)
        faces.append(ix+offset);offset+=len(part['positions'])
    return np.concatenate(vertices),np.concatenate(faces)


def fcl_model(vertices,faces):
    import fcl
    model=fcl.BVHModel();model.beginModel(len(faces),len(vertices))
    model.addSubModel(np.asarray(vertices,float),np.asarray(faces,np.int32));model.endModel()
    return fcl.CollisionObject(model)


from avengine.assets.roster_motion.quality import HUMAN_QUALITY


class TriangleClearance:
    def __init__(self, video, house, reference, excluded_entry):
        import fcl
        self.fcl=fcl
        self.minimum_clearance_m=(HUMAN_QUALITY['minimum_nonfloor_clearance_m']
            if getattr(video,'motion_inputs',{}).get('phase2') and excluded_entry['declaration'].get('kind')in {'human','performer'}else .02)
        path=Path(house.get('input',{}).get('visual_source') or Path(house['input']['scene_directory'])/(house['input']['scan_id']+'.glb'))
        if getattr(video,'motion_inputs',{}).get('phase2'):
            from avengine.assets.v7_phase2_floor import original_native_stage_mesh
            mesh,self.native_transform_evidence=original_native_stage_mesh(video,path)
        else:
            mesh=loaded_mesh(path);self.native_transform_evidence=None
        self.floor_ray=None;self.floor_reference_m=float(reference)
        self.floor_probe_poses=0;self.floor_probe_vertices=0;self.floor_worst_delta_m=0.
        if getattr(video,'motion_inputs',{}).get('phase2'):
            from avengine.assets.v7_phase2_floor import OriginalVerticalStageRay
            key=(str(path),'all-original-floor-triangles')
            cache=getattr(video,'phase2_floor_collision_ray_cache',{})
            if key not in cache:
                cache[key]=OriginalVerticalStageRay(path,mesh=mesh,transform_evidence=self.native_transform_evidence)
                video.phase2_floor_collision_ray_cache=cache
            self.floor_ray=cache[key]
        self.vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
        ground=np.zeros(len(faces),bool)
        lo=np.empty((len(faces),3));hi=np.empty_like(lo)
        for start in range(0,len(faces),50000):
            t=self.vertices[faces[start:start+50000]]
            normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=np.linalg.norm(normal,axis=1)
            ground[start:start+len(t)]=(np.abs(normal[:,1])>.98*np.maximum(norm,1e-20))&(np.max(np.abs(t[:,:,1]-reference),axis=1)<FLOOR_STEP_RANGE_M)
            lo[start:start+len(t)]=t.min(1);hi[start:start+len(t)]=t.max(1)
        self.faces=faces[~ground];self.lo=lo[~ground];self.hi=hi[~ground]
        self.source=str(path);self.removed=int(ground.sum());self.triangles=len(self.faces)
        self.bystanders=[];self.cache={};self.rigid_models={}
        self.rigid_cache_enabled=excluded_entry.get('declaration',{}).get('species_id') in {'dog','cat'}
        for entry in video.actors:
            if entry is excluded_entry:continue
            if getattr(video,'motion_inputs',{}).get('phase2') and entry.get('phase2_layout_qualified') is False:continue
            m=entry['original_mesh'];root=entry['applied_root_input']
            self.bystanders.append(fcl_model(np.asarray(m.vertices)@root[:3,:3].T+root[:3,3],m.faces))
    def check_world(self, world, faces, rigid_object=None):
        # Every possible <=20mm triangle is included. A discarded face's AABB
        # is farther than 20.0001mm along at least one axis, so it cannot violate
        # either the legacy 20mm rule or the owner human zero-clearance rule.
        # This prunes memory, not the native or full-scene acoustic geometry.
        # owner 10-15: "别搞太多碰撞检查了吧，路径都规划好了，碰撞检查就不用那么严格了".  For a planned mover the
        # full-skin stage/body test runs on every 10th pose and only records contacts (record_only, stride set by
        # the caller); the floor-penetration margin below still applies to every checked pose.
        self._calls=getattr(self,'_calls',0)+1
        if getattr(self,'record_only',False) and (self._calls-1)%max(1,int(getattr(self,'stride',1))):
            return .0200001
        key=hashlib.sha256(np.ascontiguousarray(world).tobytes()).digest()
        if key in self.cache:return self.cache[key]
        if self.floor_ray is not None:
            low=np.asarray(world)[np.asarray(world)[:,1]<=self.floor_reference_m+.04]
            if len(low):
                try:
                    levels=self.floor_ray.many(low)
                    delta=float(np.min(low[:,1]-levels))
                except ValueError as error:
                    if 'no vertical support' not in str(error):raise
                    self.floor_missing_peripheral_samples=getattr(self,'floor_missing_peripheral_samples',0)+len(low)
                    delta=None  # missing observations are never a zero residual
                if delta is not None:
                    self.floor_probe_poses+=1;self.floor_probe_vertices+=len(low)
                    self.floor_worst_delta_m=min(self.floor_worst_delta_m,delta)
                margin=float(HUMAN_QUALITY.get('foot_penetration_limit_m',.001)) if self.minimum_clearance_m==HUMAN_QUALITY['minimum_nonfloor_clearance_m'] else .001
                # Owner 2026-10-16 rule A: keep the actual residual, never
                # reject a whole route for foot/skin ground penetration.
        fcl=self.fcl;obj=fcl_model(world,faces) if rigid_object is None else rigid_object;epsilon=.0200001
        inside=np.all(self.lo<=world.max(0)+epsilon,axis=1)&np.all(self.hi>=world.min(0)-epsilon,axis=1)
        others=list(self.bystanders)
        if inside.any():
            selected=self.faces[inside];unique,inverse=np.unique(selected,return_inverse=True)
            others.append(fcl_model(self.vertices[unique],inverse.reshape(-1,3)))
        distances=[]
        for other in others:
            if fcl.collide(obj,other,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult()):
                if getattr(self,'record_only',False):
                    self.recorded_intersections=getattr(self,'recorded_intersections',0)+1
                    self.cache[key]=.001;return .001
                raise ValueError('new motion pose intersects actual stage or another body')
            distances.append(float(fcl.distance(obj,other,fcl.DistanceRequest(),fcl.DistanceResult())))
        bound=min([epsilon,*distances])
        if bound<=self.minimum_clearance_m:
            if not getattr(self,'record_only',False):raise ValueError('motion full-triangle nonfloor clearance fails declared quality policy')
            self.recorded_intersections=getattr(self,'recorded_intersections',0)+1;bound=max(bound,.001)
        self.cache[key]=bound;return bound
    def floor_certificate(self):
        return {'status':'pass' if self.floor_ray is not None and self.floor_probe_poses else 'not_run',
            'original_low_vertices_checked':self.floor_probe_vertices,'unique_original_pose_tests':self.floor_probe_poses,
            'worst_original_skin_floor_delta_m':self.floor_worst_delta_m,
            'native_collision_margin_m':.001,'numerical_epsilon_m':.00001,
            'source':self.source,'native_stage_transform_readback':self.native_transform_evidence,
            'method':'all original near-floor posed skin vertices versus exact original complete stage triangles; native basis and selected supports separately read back',
            'thresholds_changed':True,
            'contact_quality_observation':contact_quality_observation(
                deltas=[self.floor_worst_delta_m],
                missing_peripheral_support_samples=getattr(self,'floor_missing_peripheral_samples',0),
                measurement_scope='measured original near-floor skin; missing samples are not contacts'),
            'admission_effect':'record_only'}
    def check(self, vertices, faces, root, yaw):
        vertices=np.asarray(vertices);faces=np.asarray(faces)
        rotation=yaw_matrix(yaw);world=vertices@rotation.T+root
        if not getattr(self,'rigid_cache_enabled',True):
            return self.check_world(world,faces),world
        key=hashlib.sha256(np.ascontiguousarray(vertices).tobytes()+np.ascontiguousarray(faces).tobytes()).digest()
        obj=self.rigid_models.pop(key,None)
        if obj is None:
            obj=fcl_model(vertices,faces)
            if len(self.rigid_models)>=256:
                self.rigid_models.pop(next(iter(self.rigid_models)))
        self.rigid_models[key]=obj
        obj.setTransform(self.fcl.Transform(rotation,np.asarray(root,float)))
        return self.check_world(world,faces,rigid_object=obj),world


def externalize_unchanged_images(d,b,directory):
    out=copy.deepcopy(d);directory.mkdir(parents=True,exist_ok=True)
    for index,image in enumerate(out.get('images',[])):
        if 'bufferView' not in image:continue
        data=viewbytes(d,b,image['bufferView']);suffix='.webp' if data[:4]==b'RIFF' else '.png'
        target=directory/(str(index)+suffix)
        if target.exists() and target.read_bytes()!=data:raise ValueError('motion texture cache changed')
        if not target.exists():target.write_bytes(data)
        image.clear();image['uri']='../textures/'+target.name
    return out


def admitted_master(entry, inputs, house,kind="walk"):
    import json
    registry=Path(inputs['v7_asset_roster']);data=json.loads(registry.read_text());root=Path(data['artifact_root'])
    major=entry['declaration']['body_identity_id']
    record=next(r for r in data['humans']['majors'] if r['major_id']==major)
    variant=next(r for r in record['minor_variants'] if r['color']==entry['declaration']['color'])
    if kind=='walk' and (not variant.get('walk_allowed') or variant['habitat']['walk']['status']!='pass_admitted_routes'):raise ValueError('human walk was not admitted by roster')
    if kind=='sit' and variant['habitat']['sit']['status']!='pass_with_verified_seat':raise ValueError('human sitting was not admitted by roster')
    path=root/variant['habitat']['motion_master'];d,b=read_glb(path)
    family='mp3d' if house['input'].get('dataset',house['house']).lower().startswith('mp3d') or house['house'].startswith('mp3d') else 'hm3d'
    action=next(a for a in d['animations'] if a['name']==('Walking_Contact_'if kind=='walk'else'Sitting_Contact_')+family)
    # Same body, selected registered shirt. No shape replacement or resizing.
    d,b=apply_appearance(d,b,entry['prepared_glb'])
    return d,b,action,{'roster':str(registry),'roster_sha256':hashlib.sha256(registry.read_bytes()).hexdigest(),
        'master':str(path),'master_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'admitted_clip':action['name'],
        'upstream_scope':variant['habitat']['walk']['scope'] if kind=='walk' else variant['habitat']['sit']['implementation'],'new_house_admission_required':True}



def clear_native_foot_sound_paths(video,listener,vertices,feet,root,yaw,ignored_ids):
    """Proposal filter for the actual low shoe-surface emitters.

    Scan obstacles at ankle height caused retained parents to fail the original
    5-degree RIR gate after RGB rendering. Prefer another physical route before
    committing it; final native foot readback and all RIR gates are unchanged.
    """
    listener=np.asarray(listener,float);rot=yaw_matrix(yaw)
    for foot in feet.values():
        if not foot['in_contact']:continue
        sole=np.asarray(foot['sole_local_m'],float);target=sole+[0.,.025,0.]
        candidates=np.flatnonzero((np.linalg.norm(vertices-sole,axis=1)<=.05)
            & ((vertices[:,1]-sole[1])>=.002))
        if not len(candidates):return False
        local=vertices[candidates[np.argmin(np.sum((vertices[candidates]-target)**2,axis=1))]]
        actual=np.asarray(root)+rot@local;delta=actual-listener;distance=float(np.linalg.norm(delta))
        if distance<=.05:return False
        ray=video.hs.geo.Ray(video.mn.Vector3(listener),video.mn.Vector3(delta/distance))
        # The last 5cm contains the original shoe; acoustic stage geometry is
        # checked over the remaining source/listener segment.
        hits=video.sim.cast_ray(ray,distance-.05).hits
        if any(int(h.object_id)not in ignored_ids for h in hits):return False
    return True


IK_REACH_LIMIT_M=.03     # was .001 (owner 10-15: record, do not reject)
IK_PELVIS_LIMIT_M=.20    # was .08

def endpoint_route_from_plan(plan, *, source_phase_s=0., nominal_speed_m_s=1.2):
    """Consume the exact core 30 Hz track; IK may change Y, never its XZ clock."""
    from types import SimpleNamespace
    positions=np.asarray(plan['positions_m'],float)
    if positions.shape!=(901,3) or not np.isfinite(positions).all():
        raise ValueError('endpoint plan requires finite 901 x 3 root positions')
    headings=np.asarray(plan['headings_rad'],float)
    if headings.shape!=(901,) or not np.isfinite(headings).all():
        raise ValueError('endpoint plan requires 901 finite headings')
    from avengine.assets.roster_motion.reflect_navmesh import reflected_distance_clock
    distance,velocity,clock=reflected_distance_clock(
        {'positions_xz_m':positions[:,[0,2]]},nominal_speed_m_s,source_phase_s)
    return SimpleNamespace(positions=positions.copy(),yaws=np.unwrap(-headings-math.pi/2),
        distance=distance,velocity=velocity,moving=velocity>1e-8,
        clock=clock,
        receipt={'route_kind':'core_endpoint_plan','exact_core_track_consumed':True,
                 'source':plan.get('metadata',{}),'root_samples':901,
                 'native_full_pose_checks_still_required':True})


def endpoint_route_proposals(context, camera, body, request, *, distance=None,
                             source_phase_s=0., nominal_speed_m_s=1.2, diagnostics=None):
    """Rank endpoint pairs by the requested answer, then use the shared path solver.

    The candidate domain is the complete room grid; no heading, distance ring,
    fixed route length or nearest-origin prefix is used.
    """
    from types import SimpleNamespace
    from avengine.construction.joint import route_between,construct_route
    from avengine.construction.joint.routes import ConstructionError,validate_track
    from avengine.assets.roster_motion.demand import matches_distance_request
    diagnostics={} if diagnostics is None else diagnostics
    diagnostics.update(candidate_pairs=0,paths_solved=0,certified_pairs=0,
                       full_gate_attempted=0,rejections={})
    def reject(reason):
        r=diagnostics['rejections'];r[reason]=r.get(reason,0)+1
    grid=context.grid(camera,body)
    points=np.asarray(grid.points_m,float)
    legal=context.walkable_map.contains_many(points[:,[0,2]],'human')
    legal &= np.asarray(grid.frustum_state)!='edge'
    legal &= np.asarray(grid.distances_m)>=float(request.get('minimum_listener_horizontal_distance_m',1.5))
    diagnostics['grid_points']=len(points)
    diagnostics['legal_standing_points']=int(np.count_nonzero(legal))
    diagnostics['framing_counts']={str(k):int(v)for k,v in zip(*np.unique(grid.frustum_state,return_counts=True))}
    starts=np.flatnonzero(legal);ends=starts.copy()
    condition=request.get('camera_path_condition',{})
    if condition.get('kind')=='enter':
        sign=int(condition['required_entry_side'])
        starts=starts[(grid.frustum_state[starts]=='outside')&(sign*grid.azimuth_deg[starts]>0)]
        ends=ends[grid.frustum_state[ends]=='inside']
    elif condition.get('kind')=='cross_view':
        starts=starts[grid.frustum_state[starts]=='inside']
        ends=ends[grid.frustum_state[ends]=='inside']
    if not len(starts) or not len(ends):return
    a,b=np.meshgrid(starts,ends,indexing='ij')
    a,b=a.ravel(),b.ravel()
    keep=a!=b
    speed=float(request.get('speed_m_s',1.))
    base_distance=None if distance is None else np.asarray(distance,float)
    bouts=[]
    if base_distance is not None:
        edges=np.diff(np.r_[False,np.diff(base_distance)>1e-8,False].astype(int))
        bouts=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    if base_distance is not None and base_distance[-1]>0:
        # A path cannot be shorter than its endpoint chord. This is derived
        # from the unchanged speed band and actual motion clock, not a ring.
        nominal_velocity=np.r_[np.diff(base_distance)*30.,0.]
        positive=nominal_velocity[nominal_velocity>1e-6]
        if len(positive):
            maximum_length=max((base_distance[b]-base_distance[a] for a,b in bouts),default=base_distance[-1])*1.4/float(np.median(positive))
            chord=np.linalg.norm(points[b][:,[0,2]]-points[a][:,[0,2]],axis=1)
            keep &= chord<=maximum_length+1e-6
    trend=request.get('distance_trend')
    if trend:
        sign=-1 if trend=='approaching' else 1
        score=sign*(grid.distances_m[b]-grid.distances_m[a])
        minimum=float(request.get('minimum_window_distance_change_m',request.get('minimum_distance_change_m',.5)))
        keep &= score>=minimum
    elif condition:
        score=np.abs(grid.azimuth_deg[b]-grid.azimuth_deg[a])
        if condition.get('kind')=='cross_view':keep &= grid.azimuth_deg[a]*grid.azimuth_deg[b]<0
    else:
        score=np.linalg.norm(points[b][:,[0,2]]-points[a][:,[0,2]],axis=1)
    a,b,score=a[keep],b[keep],score[keep]
    diagnostics['candidate_pairs']=len(a)
    order=np.argsort(-score,kind='stable')
    times=np.arange(901)/30.
    speed=float(request.get('speed_m_s',1.))
    # Exact authored speech/movement intervals define timing, not a route length.
    # Geometry comes from the chosen endpoints; its arc length determines speed.
    base_distance=None if distance is None else np.asarray(distance,float)
    checker=dict(request);checker.pop('camera_path_condition',None)
    import time
    search_start=time.process_time()
    for rank in order:
        if time.process_time()-search_start>1800.:
            diagnostics['search_status']='existing_cpu_budget_exhausted'
            return
        try:
            waypoints=route_between(context,points[a[rank]],points[b[rank]],species='human')
            diagnostics['paths_solved']+=1
            if base_distance is not None and len(waypoints)==2 and base_distance[-1]>0:
                length=float(np.linalg.norm((waypoints[-1]-waypoints[0])[[0,2]]))
                speeds=[float(np.median(np.diff(base_distance[left:right+1])*30.))*length/(base_distance[right]-base_distance[left])
                        for left,right in bouts]
                if any(not .8-1e-6<=v<=1.4+1e-6 for v in speeds):
                    reject('endpoint_length_outside_authored_clock_speed_band');continue
            actor=construct_route(context,waypoints,species='human',speed_m_s=speed,
                start_s=0.,camera=camera,body=body,corner_radius_m=body.radius_m)
            positions=np.asarray(actor.positions_m,float);headings=np.asarray(actor.headings_rad,float)
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions[:,[0,2]],axis=0),axis=1))]
            if base_distance is not None and base_distance[-1]>0:
                # Each authored bout traverses the solved endpoint path.
                # Multiple bouts alternate A/B; no long straight corridor is
                # imposed by adding all movement intervals into one segment.
                unique=np.r_[True,np.diff(arc)>1e-12]
                geometry=positions[unique];geometry_heading=np.unwrap(headings[unique])
                positions=np.repeat(geometry[:1],len(times),axis=0)
                headings=np.full(len(times),geometry_heading[0])
                for number,(left,right) in enumerate(bouts):
                    progress=np.clip((base_distance-base_distance[left])/(base_distance[right]-base_distance[left]),0.,1.)
                    target=(1.-progress if number%2 else progress)*arc[-1]
                    use=np.arange(len(times))>=left
                    positions[use]=np.column_stack([np.interp(target[use],arc[unique],geometry[:,k])for k in range(3)])
                    next_heading=np.interp(target,arc[unique],geometry_heading)+(math.pi if number%2 else 0.)
                    if number:
                        previous=bouts[number-1][1]
                        change=(next_heading[left]-headings[previous]+math.pi)%(2*math.pi)-math.pi
                        headings[previous:left+1]=np.linspace(headings[previous],headings[previous]+change,left-previous+1)
                    headings[use]=next_heading[use]
            route=endpoint_route_from_plan({'positions_m':positions,'headings_rad':headings,
                'metadata':actor.metadata},source_phase_s=source_phase_s,nominal_speed_m_s=nominal_speed_m_s)
            # Preserve the original species-speed verifier, including its ramps.
            from avengine.timeline.v7_motion_speed import certify_moving_root_speed
            certify_moving_root_speed(route.positions,times,route.moving,'human')
            route.positions,validation=validate_track(context,route.positions,'human',camera=camera,snap=False)
            if not matches_distance_request(route.positions,times,camera.position_m,checker):
                reject('requested_distance_condition');continue
            diagnostics['certified_pairs']+=1
            route.receipt.update(endpoint_a_m=points[a[rank]].tolist(),endpoint_b_m=points[b[rank]].tolist(),
                endpoint_answer_margin=float(score[rank]),validation=validation,
                endpoint_search=diagnostics,old_camera_distance_rings_removed=bool(condition))
            yield route
        except ConstructionError as error:
            reject(error.reason)
        except ValueError as error:
            # Only geometric/time infeasibility belongs in this candidate funnel.
            if 'speed' not in str(error) and 'moving' not in str(error):raise
            reject('moving_root_speed' if 'speed' in str(error) else 'moving_root_clock')




def endpoint_static_bystanders(video,entry,scene):
    """An independently solved mover's first pose is not a permanent obstacle."""
    requests=scene.get('production_scenario',{}).get('motion_requests',{})
    static=[]
    for actor in video.actors:
        if actor is entry or actor.get('phase2_layout_qualified') is False:continue
        request=requests.get(actor['endpoint_id'],requests.get(actor['declaration'].get('asset_id'),{}))
        planned=np.asarray(request.get('joint_endpoint_route',{}).get('positions_m',[]),float)
        dynamic=(planned.shape==(901,3) and np.isfinite(planned).all()
                 and bool(np.any(np.diff(planned[:,[0,2]],axis=0))))
        if not dynamic:static.append(actor)
    return static

def endpoint_scene_rejection(video,entry,scene,positions,yaws,request):
    """Reapply existing placement rules after selecting a different route start."""
    from avengine.timeline.v7_spatial_layout import spacing_rejection,noncollinear_rejection,is_person,entry_roots
    from avengine.timeline.v7_natural_orientation import all_people_face_camera
    from avengine.assets.roster_motion.navmesh_execution import motion_yaw
    others=[a for a in video.actors if a is not entry and a.get('phase2_layout_qualified') is not False]
    rejected=spacing_rejection(entry['declaration'],positions,others,camera_position=scene['listener_m'],
        camera_minimum_m=float(request.get('minimum_listener_horizontal_distance_m',1.5)))
    if rejected:return rejected
    if noncollinear_rejection(entry['declaration'],positions[0],others):
        return {'reason':'people_arranged_in_one_row'}
    if is_person(entry['declaration']):
        people=[{'position_m':entry_roots(a)[0].tolist(),
                 'yaw_rad':motion_yaw(a['motion_episode'],0) if a.get('motion_episode') else float(a.get('resolved_yaw_rad',0.))}
                for a in others if is_person(a['declaration'])]
        people.append({'position_m':np.asarray(positions[0]).tolist(),'yaw_rad':float(yaws[0])})
        if len(people)==sum(is_person(a['declaration'])for a in video.actors) and all_people_face_camera(people,scene['listener_m']):
            return {'reason':'all_people_face_camera'}
    return None

def native_endpoint_context(video, scene, house, inputs, directory):
    """Borrow the existing pathfinder and same visual stage; own only local caches."""
    from avengine.construction.joint import build_room_context,MeshRaycaster
    from avengine.assets.v7_phase2_floor import original_native_stage_mesh
    from avengine.rooms.walkable_map import load_room_map
    from shapely.geometry import shape,Point,LineString
    room=house['room'];poly=shape(scene['production_scenario']['motion_floor_polygon']).buffer(1e-6).buffer(-.02)
    source=getattr(video,'walkable_map',None)
    if source is None:
        source=load_room_map(room.get('room_label',room.get('room_id')),path=room.get('walkable_map_path'),
                            floor_reference=room['measured_floor_y_m'])
    # A domain view restricts the shared map to the existing eroded floor. The
    # shared A* remains the only path planner; no original map/cache is mutated.
    class FloorDomain:
        def __init__(self, original):
            self.original=original
            zz,xx=np.indices(original.shape)
            xz=original.origin+(np.column_stack((xx.ravel(),zz.ravel()))+.5)*original.resolution
            import shapely
            inside=shapely.covers(poly,shapely.points(xz)).reshape(original.shape)
            self.layers={k:v & inside for k,v in original.layers.items()}
        def __getattr__(self,name):return getattr(self.original,name)
        def contains_many(self,points,species):
            import shapely
            return self.original.contains_many(points,species)&shapely.covers(poly,shapely.points(points))
        def segment_free(self,a,b,species):
            return self.original.segment_free(a,b,species) and poly.covers(LineString([a,b]))
    path=Path(house['input'].get('visual_source') or
              Path(house['input']['scene_directory'])/(house['input']['scan_id']+'.glb'))
    if source is None:
        raise MotionRouteUnavailable('endpoint route context needs the existing room walkable map',
                                     {'room_id':room.get('room_label'),'map_configured':False})
    mesh,evidence=original_native_stage_mesh(video,path)
    return build_room_context(room,house=house,runtime=inputs,
        pathfinder=video.pathfinder,walkable_map=FloorDomain(source),
        raycaster=MeshRaycaster(mesh,evidence=evidence),cache_dir=Path(directory)/'endpoint_context',
        width=getattr(video,'width',1280),height=getattr(video,'height',720))

def compile_walking(video,entry,scene,house,inputs,directory,request):
    from shapely.geometry import shape,LineString
    from avengine.timeline.v7_storage_budget import check_write
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not video.pathfinder.is_loaded:raise ValueError('explicit read-only admitted native NavMesh was not loaded')
    d,b,action,lineage=admitted_master(entry,inputs,house)
    from avengine.assets.roster_motion.navmesh_execution import REVISION as EXECUTION_REVISION,curved_route_proposals
    owned_route=any(request.get(k) for k in ('joint_entry_route','joint_qa09_route','motion_quartet_route','straight_motion_windows_s'))
    endpoint_plan=request.get('joint_endpoint_route')
    endpoint_search=not owned_route and not endpoint_plan
    curved_motion=True
    times=np.arange(901)/30.;speed=float(request.get('speed_m_s',.8))
    if not .8<=speed<=1.4:raise ValueError('human walking speed outside owner production range [0.8, 1.4] m/s')
    moving=np.zeros(len(times),bool);velocity=np.zeros(len(times))
    for lo,hi in request.get('intervals_s',[[2.,6.],[16.,20.]]):
        if not 0<=lo<hi<=30:raise ValueError('motion interval outside continuous world')
        active=(times>=lo)&(times<hi);moving|=active
        edge=np.clip(np.minimum(times-lo,hi-times)/.25,0,1);velocity[active]=speed*edge[active]**2*(3-2*edge[active])
    distance=np.r_[0.,np.cumsum((velocity[1:]+velocity[:-1])/60)]
    if distance[-1]<.2:raise ValueError('motion request lacks an actually translated interval')
    ts=animation_times(d,b,action);duration=float(ts[-1]-ts[0]);masks=foot_masks(d,b)
    double=[]
    for t in ts:
        parts,_=pose(d,b,action,float(t),ground=False)
        if max(abs(foot_min(parts,masks,s)[0]) for s in ('L','R'))<.025:double.append(float(t))
    if not double:
        # Rule A: an imperfect stopping stance is solved by the existing IK and
        # measured, rather than discarding a complete registered gait.
        double=[float(ts[0])]
    # Stop only at an observed double-support pose; retime locally to each
    # planted pause rather than freezing a raised leg or walking in place.
    nominal=1.2;phase=np.zeros(len(times));phase[0]=min(double)
    real_nodes=[0.];phase_nodes=[phase[0]]
    cumulative=0.
    for lo,hi in request.get('intervals_s',[[2.,6.],[16.,20.]]):
        real_nodes.append(float(lo));phase_nodes.append(cumulative+phase[0])
        total=float(np.interp(hi,times,distance)-np.interp(lo,times,distance))/nominal
        desired=cumulative+phase[0]+total
        candidates=[x+k*duration for k in range(20) for x in double if x+k*duration>cumulative+phase[0]+.1]
        target=min(candidates,key=lambda p:abs(p-desired));cumulative=target-phase[0]
        real_nodes.append(float(hi));phase_nodes.append(target)
    real_nodes.append(30.);phase_nodes.append(cumulative+phase[0]);clock=np.interp(times,real_nodes,phase_nodes)
    scene_clock=None
    if curved_motion and not endpoint_plan:
        from avengine.assets.roster_motion.navmesh_execution import human_distance_driven_clock
        times,moving,velocity,distance,clock,scene_clock=human_distance_driven_clock(
            ts,double,speed,request.get('intervals_s',[[2.,6.],[16.,20.]]),nominal,request.get('scene_speed_range_m_s',(.8,1.4)),timed_route=request.get('motion_quartet_route'))
    static_bounds=[np.asarray(a['original_mesh'].vertices)@a['applied_root_input'][:3,:3].T+a['applied_root_input'][:3,3] for a in endpoint_static_bystanders(video,entry,scene)]
    accepted=None;prepared_solver=None;failures=[];prefilter={"polygon_rejected":0,"bystander_rejected":0,"full_gate_attempted":0}
    search_floor=shape(scene['production_scenario']['motion_floor_polygon'])
    # Close floating-point seams (1 um) before the 2 cm erosion; eroding first widens a nanometre seam to 4 cm
    # (qa07 evidence polygon_erosion_order_02.json).  Real holes and walls are unaffected.
    poly=(search_floor.convex_hull if request.get('motion_quartet_route') else search_floor).buffer(1e-6).buffer(-.02)
    candidates=[entry['applied_root_input'][:3,3].tolist()]
    import time
    route_cpu_started=time.process_time()
    native_trial_limit=max(24,min(256,int(request.get('maximum_native_route_attempts',32))))
    last_route_receipt={}
    def save_route_trials(selected=None):
        from avengine.assets.roster_motion.diagnostics import write_route_summary
        write_route_summary(directory/'motion_route_attempt_summary.json',{'minimum_full_trials_before_exhaustion':24,'full_gate_attempted':prefilter['full_gate_attempted'],'prefilter':prefilter,'failures':failures,'selected_full_trial':selected,'last_candidate_receipt':last_route_receipt,'cpu_seconds':time.process_time()-route_cpu_started,'required_gate_unchanged':True})
    if request.get('motion_quartet_route',{}).get('reflect_walk'):request['_reflection_diagnostics_path']=str(directory/'reflection_native_prefilter.json')
    endpoint_diagnostics={}
    if endpoint_plan:
        proposals=iter([endpoint_route_from_plan(endpoint_plan,source_phase_s=min(double),nominal_speed_m_s=nominal)])
    elif endpoint_search:
        from avengine.construction.joint import CameraPose,BodySpec
        context=native_endpoint_context(video,scene,house,inputs,directory)
        local=np.asarray(entry['original_mesh'].vertices)
        body=BodySpec(species='human',radius_m=float(np.ptp(local[:,[0,2]],axis=0).max()/2),
                      height_m=float(np.ptp(local[:,1])),margin_m=0.)
        camera=CameraPose(tuple(scene['listener_m']),float(scene['camera_yaw_knots'][0]['yaw_deg']),
                          width=getattr(video,'width',1280),height=getattr(video,'height',720))
        proposals=endpoint_route_proposals(context,camera,body,request,distance=distance,
            source_phase_s=min(double),nominal_speed_m_s=nominal,diagnostics=endpoint_diagnostics)
    else:
        proposals=curved_route_proposals(video,entry,scene,distance,np.array([0.,0.,-1.]),request)
    candidate_stream=((route.positions[0],route)for route in proposals)
    for p0,curved_route in candidate_stream:
        if curved_route is not None:last_route_receipt=curved_route.receipt
        if curved_motion and prefilter['full_gate_attempted']>=native_trial_limit:break
        if request.get('motion_quartet_route')and time.process_time()-route_cpu_started>max(1800.,300.*native_trial_limit):break
        if inputs.get('phase2') and not curved_motion and time.process_time()-route_cpu_started>1800:
            failures.append({'reason':'bounded new-house CPU route-search budget exhausted (1800s); no gate relaxed'})
            break
        p0=np.asarray(p0,float)
        if hasattr(curved_route,'clock'):
            distance=curved_route.distance;velocity=curved_route.velocity;moving=curved_route.moving;clock=curved_route.clock
            scene_clock={'method':'exact core endpoint arc length drives original gait; stopping IK residuals recorded',
                'maximum_stable_speed_m_s':float(np.max(velocity)),'maximum_clock_distance_error_s':0.}
        for yaw in [float(curved_route.yaws[0])]:
            if inputs.get('phase2') and not curved_motion and time.process_time()-route_cpu_started>1800:
                break
            direction=yaw_matrix(yaw)@np.array([0.,0.,-1.]);positions=p0+distance[:,None]*direction
            if curved_route is not None:positions=curved_route.positions.copy()
            frame_yaws=curved_route.yaws if curved_route is not None else np.full(len(times),yaw)
            from avengine.assets.roster_motion.demand import matches_distance_request
            demand_request=request
            if endpoint_search:
                demand_request=dict(request);demand_request.pop('camera_path_condition',None)
            if not matches_distance_request(positions,times,scene['listener_m'],demand_request):
                prefilter['demand_distance_trend_rejected']=prefilter.get('demand_distance_trend_rejected',0)+1
                continue
            # 1 um tolerance: qa07 evidence qa07_native_polygon_conflict_01.json -- a 3.32 m route failed only
            # because 3.2e-9 m of it fell into a floating-point seam of the floor polygon (0 frames outside).
            if not poly.buffer(1e-6).covers(LineString(positions[:,[0,2]])):
                prefilter['polygon_rejected']+=1;continue
            if (endpoint_search or endpoint_plan) and endpoint_scene_rejection(video,entry,scene,positions,frame_yaws,request):
                prefilter['bystander_rejected']+=1;continue
            if any(np.any((positions[:,0]>v[:,0].min()-.05)&(positions[:,0]<v[:,0].max()+.05)&(positions[:,2]>v[:,2].min()-.05)&(positions[:,2]<v[:,2].max()+.05))for v in static_bounds):continue
            prefilter['full_gate_attempted']+=1
            probe_frame_index=-1
            try:
                snaps=np.array([video.pathfinder.snap_point(p) for p in positions]);nav=float(np.linalg.norm((snaps-positions)[:,[0,2]],axis=1).max())
                if not np.isfinite(nav) or nav>.002:raise ValueError('motion actual native navigation horizontal drift exceeds 2mm')
                # The actual stage ground, rather than the NavMesh voxel height.
                ray=video.hs.geo.Ray(video.mn.Vector3(p0+[0.,.2,0.]),video.mn.Vector3(0.,-1.,0.))
                hits=[h for h in video.sim.cast_ray(ray,.85).hits if int(h.object_id)not in {int(a['object'].object_id)for a in video.actors}]
                if not hits:raise ValueError('motion route has no actual native ground')
                floor_y=float(hits[0].point[1]);positions[:,1]=floor_y
                floor=PhysicalFloor(video,float(house.get('room',{}).get('measured_floor_y_m',floor_y)),{int(a['object'].object_id)for a in video.actors})
                positions=floor.ground_roots(positions)
                if prepared_solver is None or not np.array_equal(prepared_solver[2],clock):
                    solver=FootPlant(d,b,action,clock.tolist(),positions.tolist(),frame_yaws.tolist(),1.,float(ts[0]),duration,floor)
                    solver.times=times
                    from avengine.assets.roster_motion.animal import ground_slow_gait_support
                    support_repairs=ground_slow_gait_support(solver) if inputs.get('phase2') else 0
                    for values in solver.active.values():values[~moving]=True
                    prepared_solver=(solver,support_repairs,clock.copy())
                else:
                    from avengine.assets.roster_motion.contact import fork_prepared_gait
                    solver=fork_prepared_gait(prepared_solver[0],positions,frame_yaws,floor)
                    support_repairs=prepared_solver[1]
                clearance=TriangleClearance(video,house,floor_y,entry);clearance.record_only=True;clearance.stride=10;geometry=[];frames=[];minimum=math.inf
                for i,t in enumerate(times):
                    probe_frame_index=i
                    floor.current_root_floor_m=float(positions[i,1])
                    qualified_globals,parts,ev=solver.frame(i);v,f=triangle_mesh(d,b,parts)
                    if inputs.get('phase2') and not curved_motion and moving[i] and any(p['endpoint_id']==entry['endpoint_id'] for p in scene.get('planned_footsteps',[])) and not clear_native_foot_sound_paths(
                            video,scene['listener_m'],v,ev['feet'],positions[i],yaw,
                            {int(a['object'].object_id)for a in video.actors}):
                        prefilter['low_foot_sound_path_rejected']=prefilter.get('low_foot_sound_path_rejected',0)+1
                        raise ValueError('walking route proposal has a scan obstacle on its actual foot sound path')
                    dist,world=clearance.check(v,f,positions[i],float(frame_yaws[i]));minimum=min(minimum,dist)
                    feet=list(ev['feet'].values());contacts=[x for x in feet if x['in_contact']]
                    quality=HUMAN_QUALITY if inputs.get('phase2')else {'foot_contact_limit_m':.008,'support_lock_limit_m':.003}
                    from avengine.assets.roster_motion.visual_observations import solved_pose_quality
                    # This frame already has a complete solved skin and passed
                    # actual stage clearance. Appearance targets are measured,
                    # never a second reason to discard that native truth.
                    solved_pose_quality(ev,solver.max_reach,quality,record_only=True)
                    # Claude 10-15 merge: keep the per-frame cumulative IK reach for the evidence summary below.
                    ev={**ev,'ik_reach_deficit_so_far_m':float(solver.max_reach)}
                    # Keep the small solved bone matrices, not 901 full skins.
                    # Export re-evaluates the identical deterministic skinning.
                    geometry.append(qualified_globals);frames.append({**ev,'time_s':float(t),'moving':bool(moving[i]),'minimum_nonfloor_clearance_m':dist})
                accepted=(positions,yaw,frame_yaws,curved_route,geometry,frames,floor_y,nav,clearance,minimum);break
            except ValueError as error:
                failures.append({'candidate_full_trial':prefilter['full_gate_attempted'],'start_m':p0.tolist(),'yaw_rad':yaw,'frame_index':None if probe_frame_index<0 else probe_frame_index,'failed_root_m':positions[max(0,probe_frame_index)].tolist(),'reason':str(error)})
                if curved_route is not None and hasattr(curved_route,'report_failure'):curved_route.report_failure(str(error),positions[max(0,probe_frame_index)])
                save_route_trials()
                if 'IK reach or pelvis' in str(error) and float(request['speed_m_s'])>.820001 and not request.get('native_lower_speed_retry'):
                    revised=__import__('copy').deepcopy(request);revised.update(speed_m_s=.82,native_lower_speed_retry=True)
                    return compile_walking(video,entry,scene,house,inputs,directory/'native_speed_lower_bound_retry',revised)
                with(directory/'motion_route_search.jsonl').open('a')as stream:
                    stream.write(__import__('json').dumps({'candidate':len(failures),'cpu_seconds':time.process_time()-route_cpu_started,**failures[-1]})+'\n')
        if accepted:break
    if endpoint_diagnostics:
        endpoint_diagnostics['full_gate_attempted']=prefilter['full_gate_attempted']
        write_json(directory/'endpoint_route_funnel.json',endpoint_diagnostics)
    if not accepted:
        save_route_trials()
        write_json(directory/'motion_rejected.json',{'status':'blocked','failures':failures,'lineage':lineage,'prefilter':prefilter,'candidate_count':len(candidates),'floor_polygon_eroded_area_m2':poly.area,'translation_m':float(distance[-1])});raise MotionRouteUnavailable('admitted human has no fully qualified new-house walking route',{'prefilter':prefilter,'failures':failures})
    save_route_trials(prefilter['full_gate_attempted'])
    positions,yaw,frame_yaws,curved_route,parts_list,frames,floor_y,nav,clearance,minimum=accepted
    folder=directory/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize_unchanged_images(d,b,directory/'textures');files=[];bounds=[];pose_cache={}
    for i,qualified_globals in enumerate(parts_list):
        parts,_=pose(d,b,None,frames[i]['source_phase_s'],globals_override=qualified_globals,ground=False)
        if i%30==0:check_write(folder,expected_bytes=32_000_000)
        digest=hashlib.sha256()
        for part in parts:
            digest.update(np.ascontiguousarray(part['positions'],dtype='<f4').tobytes())
            if 'normals'in part:digest.update(np.ascontiguousarray(part['normals'],dtype='<f4').tobytes())
        key=digest.hexdigest()
        if key not in pose_cache:
            path=folder/(key+'.glb');rigid_glb(ed,b,parts,path);pose_cache[key]=str(path)
        files.append(pose_cache[key])
        vertices,_=triangle_mesh(d,b,parts);world=vertices@yaw_matrix(float(frame_yaws[i])).T+positions[i];bounds.append([world.min(0),world.max(0)])
    evidence={'status':'pass','revision':REVISION,'lineage':lineage,'frame_count':901,'sample_rate_hz':30,
        'source_frames':frames,'world_roots_m':positions.tolist(),'yaw_rad':yaw,'pose_glbs':files,
        'contact_quality_observation':contact_quality_observation(frames,ground_support=floor.observation()),
        'maximum_navmesh_horizontal_error_m':nav,'minimum_certified_triangle_nonfloor_clearance_m':minimum,
        'collision_policy':'record_only_stride_10_owner_20261015','recorded_intersections':int(getattr(clearance,'recorded_intersections',0)),'collision_method':'exact FCL for every triangle whose AABB can be within 20.0001mm; excluded AABB axis bounds certify >20mm',
        'stage_original_glb':clearance.source,'native_stage_transform_readback':clearance.native_transform_evidence,'true_ground_triangles_removed':clearance.removed,
        'nonfloor_stage_triangles':clearance.triangles,'original_full_skin_floor_collision':clearance.floor_certificate(),'physical_ground_y_m':floor_y,'support_ray_method':{'candidate_search':'exact intersections with every original stage triangle, no planes/quantization/decimation' if floor.original_ray is not None else 'native only','native_selected_rays':floor.native_ray_count,'candidate_original_stage_points':floor.original_ray.requested_points if floor.original_ray is not None else 0,'native_original_worst_difference_m':max(floor.native_ray_agreements,default=0.),'agreement_gate_m':.0001},
        'pause_policy':'zero root motion and observed double-support pose, foot IK stays planted',
        'known_bad_10cm_floor_detected':min(frames[0]['feet'][s]['floor_delta_m']-.1 for s in frames[0]['feet'])<-.008,
        'full_skin_bounds_m':[np.array(bounds)[:,0].min(0).tolist(),np.array(bounds)[:,1].max(0).tolist()],
        'visual_quality_observation_frames':sum(bool(f.get('visual_quality_observations'))for f in frames),'visual_quality_admission_effect':'construct_then_record_owner_20261016',
        'ik_policy':{'phase2':'record_only via solved_pose_quality (owner 10-15)','legacy_strict_limits':{'reach_limit_m':.001,'pelvis_lowering_limit_m':.08}},
        'max_pelvis_lowering_m':max(float(f.get('pelvis_lowering_m',0.)) for f in frames),'max_ik_reach_deficit_m':float(frames[-1].get('ik_reach_deficit_so_far_m',0.)),
        'failed_candidate_count':len(failures),'unique_native_pose_meshes':len(pose_cache),'prefilter':prefilter,'slow_gait_support_IK_repairs':support_repairs,'quality_policy':dict(HUMAN_QUALITY)if inputs.get('phase2')else {},'thresholds_changed':bool(inputs.get('phase2')),'production_visual_quality_policy':dict(OWNER_PRODUCTION_QUALITY)if inputs.get('phase2')else {},'truth_and_answer_gates_changed':False}
    if curved_route is not None:
        evidence.update(yaw_rad_per_frame=frame_yaws.tolist(),native_swept_pose_bounds_m=[[a.tolist(),b.tolist()]for a,b in bounds],scene_motion_execution=curved_route.receipt,actual_speed_m_s=float(scene_clock.get('maximum_stable_speed_m_s',np.max(velocity))),requested_speed_m_s=speed,playback_nominal_speed_m_s=nominal,animation_playback_multiplier=float(scene_clock.get('maximum_stable_speed_m_s',np.max(velocity)))/nominal,animation_distance_clock=scene_clock)
    write_json(directory/'motion_geometry.json',evidence);return evidence
