"""Recorded dog emitter intake, preserving the admitted Rocketbox body.

The first dog action is continuous slow walking in a straight clear corridor.
An admitted canine gait has no four-paw pause sample; human two-foot pause
logic must not be applied to it. Pauses remain unsupported, not fabricated.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import accessor
from avengine.assets.roster_motion.contact import find_bone
from avengine.assets.roster_motion.appearance import apply_appearance

def dog_chains_masks(d,b):
    chains={s:[find_bone(d,' '+side+' '+p)for p in parts]for s,side,parts in [('FL','L',['UpperArm','Forearm','Hand','Finger0']),('FR','R',['UpperArm','Forearm','Hand','Finger0']),('HL','L',['Thigh','Calf','Foot','Toe0']),('HR','R',['Thigh','Calf','Foot','Toe0'])]}
    masks=[]
    for node in d['nodes']:
        if 'mesh'not in node:continue
        for prim in d['meshes'][node['mesh']]['primitives']:
            joints=d['skins'][node['skin']]['joints'];js=accessor(d,b,prim['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,prim['attributes']['WEIGHTS_0'])
            masks.append({s:(np.isin(js,[joints.index(j)for j in c[2:]])*ws).sum(1)>.15 for s,c in chains.items()})
    if any(not any(m[s].any()for m in masks)for s in chains):raise ValueError('dog paw weights missing')
    return chains,masks

def dog_master(entry,inputs,house):
    registry=Path(inputs['v7_asset_roster']);data=json.loads(registry.read_text());root=Path(data['artifact_root'])
    record=next(v for major in data['animals']if major['major_id']=='dog'for v in major['minor_variants']if v['asset_id']==entry['declaration']['asset_id'])
    if not record.get('walk_allowed')or not record.get('dual_backend_static_usable')or not record.get('original_target_geometry_preserved'):raise ValueError('dog roster motion authority missing')
    path=root/record['motion_master_glb'];d,b=read_glb(path)
    family='mp3d'if house['house'].startswith('mp3d')else'hm3d';name='Walking_Contact_'+family
    if name not in record['admitted_motion_clips']['walk']:raise ValueError('canine contact clip not admitted')
    action=next(a for a in d['animations']if a['name']==name);d,b=apply_appearance(d,b,entry['prepared_glb'])
    return d,b,action,{'roster':str(registry),'roster_sha256':hashlib.sha256(registry.read_bytes()).hexdigest(),'master':str(path),'master_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'admitted_clip':name,'upstream_scope':record['route_scope'],'source_license':record['source_license'],'new_house_admission_required':True}

from avengine.assets.roster_motion.episode import (PhysicalFloor,TriangleClearance,triangle_mesh,externalize_unchanged_images,REVISION)
from avengine.assets.roster_motion.pose import animation_times,rigid_glb
from avengine.assets.roster_motion.contact import FootPlant,yaw_matrix
from avengine.timeline.continuous_pilot import write_json
import math,time

def ground_slow_gait_support(solver):
    """Plant a near-ground authored paw when re-timing removes a stance label.

    The source Contact clips already contain IK rotations. Their ankle
    derivatives are not a reliable stance classifier after a second retiming.
    This changes the generated pose through the existing world-space IK;
    it never fabricates a contact certificate or changes its tolerances.
    """
    changed=[]
    sides=list(solver.chains)
    for i,source in enumerate(solver.source):
        if any(solver.active[side][i] for side in sides):continue
        heights=source[2]
        near=[side for side in sides if heights[side]<.025]
        if not near:raise ValueError("slow canine gait has no near-ground authored support candidate")
        side=min(near,key=lambda key:heights[key])
        solver.active[side][i]=True
        changed.append({"frame_index":i,"paw":side,"source_floor_height_m":float(heights[side])})
    return changed


def compile_continuous_dog_walking(video,entry,scene,house,inputs,directory,request):
    from shapely.geometry import shape,LineString
    from avengine.timeline.v7_storage_budget import check_write
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not video.pathfinder.is_loaded:raise ValueError('explicit read-only admitted native NavMesh was not loaded')
    d,b,action,lineage=dog_master(entry,inputs,house)
    times=np.arange(901)/30.;speed=float(request.get('speed_m_s',.05))
    if not .03<=speed<=.3:raise ValueError('dog speed outside explicitly supported slow walk')
    if request.get('intervals_s')not in (None,[[0.,30.]]):raise ValueError('dog pause is unsupported; no raised-paw freeze')
    moving=np.ones(len(times),bool);distance=times*speed
    ts=animation_times(d,b,action);duration=float(ts[-1]-ts[0]);chains,masks=dog_chains_masks(d,b)
    clock=float(ts[0])+times*speed/.3
    static_bounds=[np.asarray(a['original_mesh'].vertices)@a['applied_root_input'][:3,:3].T+a['applied_root_input'][:3,3] for a in video.actors if a is not entry]
    accepted=None;failures=[];prefilter={"polygon_rejected":0,"bystander_rejected":0,"full_gate_attempted":0}
    search_start=time.process_time();max_attempts=8;cpu_budget_s=1800.
    def search_progress(**extra):
        write_json(directory/('motion_search_progress_'+str(time.time_ns())+'.json'),{'status':'searching_not_admitted','cpu_seconds':time.process_time()-search_start,'prefilter':dict(prefilter),'failures':list(failures),'max_native_route_attempts':max_attempts,'search_CPU_budget_s':cpu_budget_s,'thresholds_changed':False,**extra})
    poly=shape(scene['production_scenario']['motion_floor_polygon']).buffer(-.02)
    x0,z0,x1,z1=poly.bounds
    from shapely.geometry import Point
    extra=[[float(x),float(entry['applied_root_input'][1,3]),float(z)] for x in np.arange(x0,x1,.2) for z in np.arange(z0,z1,.2) if poly.covers(Point(x,z))]
    extra.sort(key=lambda p:float(np.linalg.norm(np.asarray(p)[[0,2]]-entry['applied_root_input'][[0,2],3])))
    candidates=[entry['applied_root_input'][:3,3].tolist(),*extra]
    for p0 in candidates[:512]:
        p0=np.asarray(p0,float)
        for yaw in [0.,math.pi/2,math.pi,-math.pi/2]:
            direction=yaw_matrix(yaw)@np.array([0.,0.,-1.]);positions=p0+distance[:,None]*direction
            if not poly.covers(LineString([positions[0][[0,2]],positions[-1][[0,2]]])):
                prefilter['polygon_rejected']+=1;continue
            if any(np.any((positions[:,0]>v[:,0].min()-.05)&(positions[:,0]<v[:,0].max()+.05)&(positions[:,2]>v[:,2].min()-.05)&(positions[:,2]<v[:,2].max()+.05))for v in static_bounds):continue
            if prefilter['full_gate_attempted']>=max_attempts or time.process_time()-search_start>cpu_budget_s:
                write_json(directory/'motion_rejected.json',{'status':'blocked','reason':'bounded route search exhausted; no physical admission','failures':failures,'lineage':lineage,'prefilter':prefilter,'thresholds_changed':False})
                raise ValueError('dog native route search budget exhausted without admission')
            prefilter['full_gate_attempted']+=1
            search_progress(start_m=p0.tolist(),yaw_rad=yaw,frame_index=None)
            try:
                snaps=np.array([video.pathfinder.snap_point(p) for p in positions]);nav=float(np.linalg.norm((snaps-positions)[:,[0,2]],axis=1).max())
                if not np.isfinite(nav) or nav>.002:raise ValueError('motion actual native navigation horizontal drift exceeds 2mm')
                # The actual stage ground, rather than the NavMesh voxel height.
                ray=video.hs.geo.Ray(video.mn.Vector3(p0+[0.,.2,0.]),video.mn.Vector3(0.,-1.,0.))
                hits=[h for h in video.sim.cast_ray(ray,.85).hits if int(h.object_id)not in {int(a['object'].object_id)for a in video.actors}]
                if not hits:raise ValueError('motion route has no actual native ground')
                floor_y=float(hits[0].point[1]);positions[:,1]=floor_y
                floor=PhysicalFloor(video,floor_y,{int(a['object'].object_id)for a in video.actors})
                solver=FootPlant(d,b,action,clock.tolist(),positions.tolist(),[yaw]*len(times),1.,float(ts[0]),duration,floor,basis=np.eye(4),chains=chains,masks=masks,emitter_bone='Xtra Mouth',canonical_flip=False,backward_threshold=.08)
                solver.times=times
                support_repairs=ground_slow_gait_support(solver)
                for values in solver.active.values():values[~moving]=True
                clearance=TriangleClearance(video,house,floor_y,entry);geometry=[];frames=[];minimum=math.inf
                for i,t in enumerate(times):
                    if i%90==0:search_progress(start_m=p0.tolist(),yaw_rad=yaw,frame_index=i)
                    if time.process_time()-search_start>cpu_budget_s:raise ValueError('dog route CPU budget exhausted; no admission')
                    _,parts,ev=solver.frame(i);v,f=triangle_mesh(d,b,parts);dist,world=clearance.check(v,f,positions[i],yaw);minimum=min(minimum,dist)
                    feet=list(ev['feet'].values());contacts=[x for x in feet if x['in_contact']]
                    if not contacts:raise ValueError('dog gait has no planted paw at an actual 30Hz sample')
                    if min(x['floor_delta_m']for x in feet)<-.008 or any(abs(x['floor_delta_m'])>=.008 or x['support_lock_error_m']>=.003 for x in contacts):raise ValueError('motion source foot contact or lock violates unchanged 8mm/3mm admission')
                    if solver.max_reach>=.001 or ev['pelvis_lowering_m']>=.08:raise ValueError('motion IK reach or pelvis bound failed')
                    geometry.append(parts);frames.append({**ev,'time_s':float(t),'moving':bool(moving[i]),'minimum_nonfloor_clearance_m':dist})
                accepted=(positions,yaw,geometry,frames,floor_y,nav,clearance,minimum,support_repairs);break
            except ValueError as error:failures.append({'start_m':p0.tolist(),'yaw_rad':yaw,'reason':str(error)})
        if accepted:break
    if not accepted:
        write_json(directory/'motion_rejected.json',{'status':'blocked','failures':failures,'lineage':lineage,'prefilter':prefilter,'candidate_count':len(candidates),'floor_polygon_eroded_area_m2':poly.area,'translation_m':float(distance[-1])});raise ValueError('admitted dog has no fully qualified new-house walking route')
    positions,yaw,parts_list,frames,floor_y,nav,clearance,minimum,support_repairs=accepted
    folder=directory/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize_unchanged_images(d,b,directory/'textures');files=[];bounds=[];pose_cache={}
    for i,parts in enumerate(parts_list):
        if i%30==0:check_write(folder,expected_bytes=32_000_000)
        digest=hashlib.sha256()
        for part in parts:
            digest.update(np.ascontiguousarray(part['positions']).tobytes())
            if 'normals'in part:digest.update(np.ascontiguousarray(part['normals']).tobytes())
        key=digest.hexdigest()
        if key not in pose_cache:
            path=folder/(key+'.glb');rigid_glb(ed,b,parts,path);pose_cache[key]=str(path)
        files.append(pose_cache[key])
        vertices,_=triangle_mesh(d,b,parts);world=vertices@yaw_matrix(yaw).T+positions[i];bounds.append([world.min(0),world.max(0)])
    evidence={'status':'pass','revision':REVISION,'lineage':lineage,'frame_count':901,'sample_rate_hz':30,
        'source_frames':frames,'world_roots_m':positions.tolist(),'yaw_rad':yaw,'pose_glbs':files,
        'maximum_navmesh_horizontal_error_m':nav,'minimum_certified_triangle_nonfloor_clearance_m':minimum,'collision_method':'exact FCL for every triangle whose AABB can be within 20.0001mm; excluded AABB axis bounds certify >20mm',
        'stage_original_glb':clearance.source,'true_ground_triangles_removed':clearance.removed,
        'nonfloor_stage_triangles':clearance.triangles,'physical_ground_y_m':floor_y,
        'motion_kind':'dog_continuous_walk','slow_gait_support_IK_repairs':support_repairs,'pause_policy':'continuous slow walk only; no four-paw stop is declared','species_id':'dog',
        'known_bad_10cm_floor_detected':min(frames[0]['feet'][s]['floor_delta_m']-.1 for s in frames[0]['feet'])<-.008,
        'full_skin_bounds_m':[np.array(bounds)[:,0].min(0).tolist(),np.array(bounds)[:,1].max(0).tolist()],
        'failed_candidate_count':len(failures),'unique_native_pose_meshes':len(pose_cache),'prefilter':prefilter,'thresholds_changed':False}
    write_json(directory/'motion_geometry.json',evidence);return evidence
