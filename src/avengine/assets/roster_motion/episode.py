"""New-house contact compilation from admitted clips, with exact triangle gates.

Reference-route receipts do not admit a new placement. Every 30 Hz pose is
checked against the original stage triangles, actual floor and native NavMesh.
Only flat, clear, straight routes are supported by this first adapter.
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

REVISION='admitted-contact-clip-new-house-full-triangle-native-v1'


class PhysicalFloor:
    def __init__(self, video, reference, ignore=()):
        self.video=video;self.reference=float(reference);self.ignore=set(ignore)
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
        if not hits:raise ValueError('motion support ray has no actual stage floor')
        y=float(hits[0].point[1]);self.native_ray_count+=1
        if self.original_ray is not None:
            original=float(self.original_ray.many([p])[0])
            error=abs(original-y);self.native_ray_agreements.append(error)
            if error>.0001:raise ValueError('original full-stage support differs from native floor ray by >0.1mm')
        if abs(y-self.reference)>.015:raise ValueError('motion support ray hit furniture or a different floor level')
        return y
    def many(self, points):
        if self.original_ray is None:return np.array([self(p) for p in points])
        y=self.original_ray.many(points)
        if np.any(abs(y-self.reference)>.015):raise ValueError('candidate support hit furniture or a different floor level')
        return y


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
            ground[start:start+len(t)]=(np.abs(normal[:,1])>.98*np.maximum(norm,1e-20))&(np.max(np.abs(t[:,:,1]-reference),axis=1)<.02)
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
        key=hashlib.sha256(np.ascontiguousarray(world).tobytes()).digest()
        if key in self.cache:return self.cache[key]
        if self.floor_ray is not None:
            low=np.asarray(world)[np.asarray(world)[:,1]<=self.floor_reference_m+.04]
            if len(low):
                levels=self.floor_ray.many(low)
                delta=float(np.min(low[:,1]-levels))
                self.floor_probe_poses+=1;self.floor_probe_vertices+=len(low)
                self.floor_worst_delta_m=min(self.floor_worst_delta_m,delta)
                if delta<-.00101:
                    raise ValueError('actual original posed skin penetrates original floor beyond native 1mm margin plus 10um epsilon')
        fcl=self.fcl;obj=fcl_model(world,faces) if rigid_object is None else rigid_object;epsilon=.0200001
        inside=np.all(self.lo<=world.max(0)+epsilon,axis=1)&np.all(self.hi>=world.min(0)-epsilon,axis=1)
        others=list(self.bystanders)
        if inside.any():
            selected=self.faces[inside];unique,inverse=np.unique(selected,return_inverse=True)
            others.append(fcl_model(self.vertices[unique],inverse.reshape(-1,3)))
        distances=[]
        for other in others:
            if fcl.collide(obj,other,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult()):raise ValueError('new motion pose intersects actual stage or another body')
            distances.append(float(fcl.distance(obj,other,fcl.DistanceRequest(),fcl.DistanceResult())))
        bound=min([epsilon,*distances])
        if bound<=self.minimum_clearance_m:raise ValueError('motion full-triangle nonfloor clearance fails declared quality policy')
        self.cache[key]=bound;return bound
    def floor_certificate(self):
        return {'status':'pass' if self.floor_ray is not None and self.floor_probe_poses else 'not_run',
            'original_low_vertices_checked':self.floor_probe_vertices,'unique_original_pose_tests':self.floor_probe_poses,
            'worst_original_skin_floor_delta_m':self.floor_worst_delta_m,
            'native_collision_margin_m':.001,'numerical_epsilon_m':.00001,
            'source':self.source,'native_stage_transform_readback':self.native_transform_evidence,
            'method':'all original near-floor posed skin vertices versus exact original complete stage triangles; native basis and selected supports separately read back',
            'thresholds_changed':False}
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


def compile_walking(video,entry,scene,house,inputs,directory,request):
    from shapely.geometry import shape,LineString
    from avengine.timeline.v7_storage_budget import check_write
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not video.pathfinder.is_loaded:raise ValueError('explicit read-only admitted native NavMesh was not loaded')
    d,b,action,lineage=admitted_master(entry,inputs,house)
    from avengine.assets.roster_motion.navmesh_execution import REVISION as EXECUTION_REVISION,curved_route_proposals
    curved_motion=request.get('motion_execution_revision')==EXECUTION_REVISION
    times=np.arange(901)/30.;speed=float(request.get('speed_m_s',.5))
    if not .15<=speed<=(1.4 if curved_motion else 1.2):raise ValueError('human walking speed outside explicit slow-to-normal production support')
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
    if not double:raise ValueError('admitted walk clip has no double-support pause sample')
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
    if curved_motion:
        from avengine.assets.roster_motion.navmesh_execution import human_distance_driven_clock
        times,moving,velocity,distance,clock,scene_clock=human_distance_driven_clock(
            ts,double,speed,request.get('intervals_s',[[2.,6.],[16.,20.]]),nominal,request.get('scene_speed_range_m_s',(.8,1.4)),timed_route=request.get('motion_quartet_route'))
    static_bounds=[np.asarray(a['original_mesh'].vertices)@a['applied_root_input'][:3,:3].T+a['applied_root_input'][:3,3] for a in video.actors if a is not entry and a.get('phase2_layout_qualified') is not False]
    accepted=None;prepared_solver=None;failures=[];prefilter={"polygon_rejected":0,"bystander_rejected":0,"full_gate_attempted":0}
    poly=shape(scene['production_scenario']['motion_floor_polygon']).buffer(-.02)
    x0,z0,x1,z1=poly.bounds
    from shapely.geometry import Point
    extra=[[float(x),float(entry['applied_root_input'][1,3]),float(z)] for x in np.arange(x0,x1,.2) for z in np.arange(z0,z1,.2) if poly.covers(Point(x,z))]
    extra.sort(key=lambda p:float(np.linalg.norm(np.asarray(p)[[0,2]]-entry['applied_root_input'][[0,2],3])))
    candidates=[entry['applied_root_input'][:3,3].tolist(),*extra]
    import time
    route_cpu_started=time.process_time()
    proposals=curved_route_proposals(video,entry,scene,distance,np.array([0.,0.,-1.]),request) if curved_motion else None
    candidate_stream=((route.positions[0],route)for route in proposals) if curved_motion else ((point,None)for point in candidates[:512])
    for p0,curved_route in candidate_stream:
        if inputs.get('phase2') and not curved_motion and time.process_time()-route_cpu_started>1800:
            failures.append({'reason':'bounded new-house CPU route-search budget exhausted (1800s); no gate relaxed'})
            break
        p0=np.asarray(p0,float)
        for yaw in ([float(curved_route.yaws[0])] if curved_route is not None else [0.,math.pi/2,math.pi,-math.pi/2]):
            if inputs.get('phase2') and not curved_motion and time.process_time()-route_cpu_started>1800:
                break
            direction=yaw_matrix(yaw)@np.array([0.,0.,-1.]);positions=p0+distance[:,None]*direction
            if curved_route is not None:positions=curved_route.positions.copy()
            frame_yaws=curved_route.yaws if curved_route is not None else np.full(len(times),yaw)
            trend=request.get('distance_trend')
            if trend is not None and curved_route is None:
                if trend not in {'approaching','receding'}:raise ValueError('unsupported demanded distance trend')
                radii=np.linalg.norm(positions[:,[0,2]]-np.asarray(scene['listener_m'])[[0,2]],axis=1)
                diffs=np.diff(radii);net=float(radii[-1]-radii[0])
                if ((trend=='approaching' and (net>-.2 or np.any(diffs>1e-7)))
                        or (trend=='receding' and (net<.2 or np.any(diffs<-1e-7)))):
                    prefilter['demand_distance_trend_rejected']=prefilter.get('demand_distance_trend_rejected',0)+1
                    continue
            if not poly.covers(LineString([positions[0][[0,2]],positions[-1][[0,2]]])):
                prefilter['polygon_rejected']+=1;continue
            if any(np.any((positions[:,0]>v[:,0].min()-.05)&(positions[:,0]<v[:,0].max()+.05)&(positions[:,2]>v[:,2].min()-.05)&(positions[:,2]<v[:,2].max()+.05))for v in static_bounds):continue
            prefilter['full_gate_attempted']+=1
            try:
                snaps=np.array([video.pathfinder.snap_point(p) for p in positions]);nav=float(np.linalg.norm((snaps-positions)[:,[0,2]],axis=1).max())
                if not np.isfinite(nav) or nav>.002:raise ValueError('motion actual native navigation horizontal drift exceeds 2mm')
                # The actual stage ground, rather than the NavMesh voxel height.
                ray=video.hs.geo.Ray(video.mn.Vector3(p0+[0.,.2,0.]),video.mn.Vector3(0.,-1.,0.))
                hits=[h for h in video.sim.cast_ray(ray,.85).hits if int(h.object_id)not in {int(a['object'].object_id)for a in video.actors}]
                if not hits:raise ValueError('motion route has no actual native ground')
                floor_y=float(hits[0].point[1]);positions[:,1]=floor_y
                floor=PhysicalFloor(video,floor_y,{int(a['object'].object_id)for a in video.actors})
                if prepared_solver is None:
                    solver=FootPlant(d,b,action,clock.tolist(),positions.tolist(),frame_yaws.tolist(),1.,float(ts[0]),duration,floor)
                    solver.times=times
                    from avengine.assets.roster_motion.animal import ground_slow_gait_support
                    support_repairs=ground_slow_gait_support(solver) if inputs.get('phase2') else 0
                    for values in solver.active.values():values[~moving]=True
                    prepared_solver=(solver,support_repairs)
                else:
                    from avengine.assets.roster_motion.contact import fork_prepared_gait
                    solver=fork_prepared_gait(prepared_solver[0],positions,frame_yaws,floor)
                    support_repairs=prepared_solver[1]
                clearance=TriangleClearance(video,house,floor_y,entry);geometry=[];frames=[];minimum=math.inf
                for i,t in enumerate(times):
                    qualified_globals,parts,ev=solver.frame(i);v,f=triangle_mesh(d,b,parts)
                    if inputs.get('phase2') and not curved_motion and moving[i] and any(p['endpoint_id']==entry['endpoint_id'] for p in scene.get('planned_footsteps',[])) and not clear_native_foot_sound_paths(
                            video,scene['listener_m'],v,ev['feet'],positions[i],yaw,
                            {int(a['object'].object_id)for a in video.actors}):
                        prefilter['low_foot_sound_path_rejected']=prefilter.get('low_foot_sound_path_rejected',0)+1
                        raise ValueError('walking route proposal has a scan obstacle on its actual foot sound path')
                    dist,world=clearance.check(v,f,positions[i],float(frame_yaws[i]));minimum=min(minimum,dist)
                    feet=list(ev['feet'].values());contacts=[x for x in feet if x['in_contact']]
                    quality=HUMAN_QUALITY if inputs.get('phase2')else {'foot_contact_limit_m':.008,'support_lock_limit_m':.003}
                    if min(x['floor_delta_m']for x in feet)<-.008 or any(
                            abs(x['floor_delta_m'])>quality['foot_contact_limit_m']+1e-12
                            or x['support_lock_error_m']>quality['support_lock_limit_m']+1e-12 for x in contacts):
                        raise ValueError('motion source foot contact or lock violates declared human quality policy')
                    if solver.max_reach>=.001 or ev['pelvis_lowering_m']>=.08:raise ValueError('motion IK reach or pelvis bound failed')
                    # Keep the small solved bone matrices, not 901 full skins.
                    # Export re-evaluates the identical deterministic skinning.
                    geometry.append(qualified_globals);frames.append({**ev,'time_s':float(t),'moving':bool(moving[i]),'minimum_nonfloor_clearance_m':dist})
                accepted=(positions,yaw,frame_yaws,curved_route,geometry,frames,floor_y,nav,clearance,minimum);break
            except ValueError as error:
                failures.append({'start_m':p0.tolist(),'yaw_rad':yaw,'reason':str(error)})
                with(directory/'motion_route_search.jsonl').open('a')as stream:
                    stream.write(__import__('json').dumps({'candidate':len(failures),'cpu_seconds':time.process_time()-route_cpu_started,**failures[-1]})+'\n')
        if accepted:break
    if not accepted:
        write_json(directory/'motion_rejected.json',{'status':'blocked','failures':failures,'lineage':lineage,'prefilter':prefilter,'candidate_count':len(candidates),'floor_polygon_eroded_area_m2':poly.area,'translation_m':float(distance[-1])});raise ValueError('admitted human has no fully qualified new-house walking route')
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
        'maximum_navmesh_horizontal_error_m':nav,'minimum_certified_triangle_nonfloor_clearance_m':minimum,'collision_method':'exact FCL for every triangle whose AABB can be within 20.0001mm; excluded AABB axis bounds certify >20mm',
        'stage_original_glb':clearance.source,'native_stage_transform_readback':clearance.native_transform_evidence,'true_ground_triangles_removed':clearance.removed,
        'nonfloor_stage_triangles':clearance.triangles,'original_full_skin_floor_collision':clearance.floor_certificate(),'physical_ground_y_m':floor_y,'support_ray_method':{'candidate_search':'exact intersections with every original stage triangle, no planes/quantization/decimation' if floor.original_ray is not None else 'native only','native_selected_rays':floor.native_ray_count,'candidate_original_stage_points':floor.original_ray.requested_points if floor.original_ray is not None else 0,'native_original_worst_difference_m':max(floor.native_ray_agreements,default=0.),'agreement_gate_m':.0001},
        'pause_policy':'zero root motion and observed double-support pose, foot IK stays planted',
        'known_bad_10cm_floor_detected':min(frames[0]['feet'][s]['floor_delta_m']-.1 for s in frames[0]['feet'])<-.008,
        'full_skin_bounds_m':[np.array(bounds)[:,0].min(0).tolist(),np.array(bounds)[:,1].max(0).tolist()],
        'failed_candidate_count':len(failures),'unique_native_pose_meshes':len(pose_cache),'prefilter':prefilter,'slow_gait_support_IK_repairs':support_repairs,'quality_policy':dict(HUMAN_QUALITY)if inputs.get('phase2')else {},'thresholds_changed':bool(inputs.get('phase2')),'production_visual_quality_policy':dict(OWNER_PRODUCTION_QUALITY)if inputs.get('phase2')else {},'truth_and_answer_gates_changed':False}
    if curved_route is not None:
        evidence.update(yaw_rad_per_frame=frame_yaws.tolist(),native_swept_pose_bounds_m=[[a.tolist(),b.tolist()]for a,b in bounds],scene_motion_execution=curved_route.receipt,actual_speed_m_s=scene_clock['maximum_stable_speed_m_s'],requested_speed_m_s=speed,playback_nominal_speed_m_s=nominal,animation_playback_multiplier=scene_clock['maximum_stable_speed_m_s']/nominal,animation_distance_clock=scene_clock)
    write_json(directory/'motion_geometry.json',evidence);return evidence
