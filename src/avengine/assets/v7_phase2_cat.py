"""New-room qualification of the animal task's immutable Habitat pose caches."""
from pathlib import Path
from copy import deepcopy
import json,math,hashlib
import numpy as np
from avengine.capture.v7_visible_mesh import indexed_static_mesh
from avengine.assets.roster_motion.episode import PhysicalFloor,TriangleClearance
from avengine.assets.roster_motion.contact import yaw_matrix,rotation_between
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.timeline.continuous_pilot import write_json

def compile_cached_cat(video,entry,scene,house,inputs,directory,request):
    from shapely.geometry import shape,LineString
    from avengine.timeline.v7_storage_budget import check_write
    a=entry['declaration'];family='mp3d' if str(house['family']).lower()=='mp3d' else 'hm3d'
    caches=a.get('walking_frame_cache') or {};cache=Path(caches.get(family,''))
    if not a.get('walk_allowed') or not cache.is_dir():raise ValueError('cat lacks an admitted immutable Habitat pose cache')
    asset_root=Path(a['source_registry']).parents[1]
    candidate_path=asset_root/'append_20261012/walking_frames'/a['asset_id']/'candidate.json'
    candidate=json.loads(candidate_path.read_text())
    upstream_path=asset_root/'append_20261012/evidence/walking'/family/(a['asset_id']+'_Walking.json')
    upstream=json.loads(upstream_path.read_text())
    if upstream['status']!='pass' or upstream['frame_count']!=41:raise ValueError('cat native source cache not qualified')
    cpu=next(x for x in candidate['CPU_actions'] if x['family']=='Walking')['frames']
    native=upstream['frames'];prepared=[];dirpath=Path(directory);dirpath.mkdir(parents=True,exist_ok=True)
    poses=dirpath/'frames';poses.mkdir()
    # Exact source meshes remain untouched. A bounded rigid contact correction
    # in this episode is applied to every point, including the head emitter.
    for ix,record in enumerate(native):
        source=cache/(str(ix).zfill(4)+'.glb');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=record['rigid_pose_sha256']:raise ValueError('cat cached pose SHA changed')
        mesh=indexed_static_mesh(source);v=np.asarray(mesh.vertices)
        feet={s:np.asarray(f['support_local_m'],float) for s,f in record['feet'].items()}
        active=[p for s,p in feet.items() if record['feet'][s]['in_contact']]
        measured_support_recovered=False
        flags={side:bool(value['in_contact'])for side,value in record['feet'].items()}
        if not active:
            # Cache frame 11, for example, has a native-measured hind paw
            # exactly on the ground although its authored phase flag is false.
            # Recover only the actual lowest registered paw, without changing
            # the original pose or its upstream metadata.
            side=min(feet,key=lambda k:feet[k][1])
            if abs(record['feet'][side]['native_delta_m'])>.012:
                raise ValueError('cache has neither an authored nor a measured grounded paw')
            active=[feet[side]];flags[side]=True;measured_support_recovered=True
        r=np.eye(3)
        if len(active)>=2:
            delta=active[1]-active[0];target=delta.copy();target[1]=0.
            if np.linalg.norm(target)<.02:raise ValueError('cat stance points do not define a safe rigid ground correction')
            r=rotation_between(delta,target)
        tilt=math.degrees(math.acos(float(np.clip(r[1,1],-1,1))))
        if tilt>10:raise ValueError('cached cat rigid contact correction would tilt body beyond 10deg')
        tv=v@r.T;shift=-float(tv[:,1].min())
        matrix=np.eye(4);matrix[:3,:3]=r;matrix[1,3]=shift
        corrected={s:(matrix@np.r_[p,1.])[:3] for s,p in feet.items()}
        if any(abs(corrected[s][1])>.012 for s in feet if flags[s]):
            raise ValueError('cat stance fails unchanged upstream 12mm contact gate after rigid grounding')
        d,b=read_glb(source)
        texture_dir=dirpath/'textures';texture_dir.mkdir(exist_ok=True)
        for image in d.get('images',[]):
            if image.get('uri'):
                original=(source.parent/image['uri']).resolve()
                raw=original.read_bytes();name=hashlib.sha256(raw).hexdigest()+original.suffix
                target=texture_dir/name
                if not target.exists():target.write_bytes(raw)
                if target.read_bytes()!=raw:raise ValueError('copied cat texture differs')
                image['uri']='../textures/'+name
        roots=list(d['scenes'][d.get('scene',0)]['nodes']);j=len(d['nodes'])
        d['nodes'].append({'name':'OwnedEpisodeRigidCatGround','matrix':matrix.T.ravel().tolist(),'children':roots});d['scenes'][d.get('scene',0)]['nodes']=[j]
        check_write(poses,expected_bytes=source.stat().st_size+10000)
        out=poses/(str(ix).zfill(4)+'.glb');write_glb(out,d,b)
        extra=float(record.get('native_additional_whole_frame_vertical_translation_m',0))
        mouth=np.asarray(cpu[ix]['mouth_local_m'],float)+[0,extra,0];mouth=(matrix@np.r_[mouth,1.])[:3]
        prepared.append({'path':str(out),'mesh':indexed_static_mesh(out),'mouth_m':mouth.tolist(),'feet':corrected,'flags':flags,'original_authored_contact_flags':{s:bool(f['in_contact']) for s,f in record['feet'].items()},'native_measured_support_recovered':measured_support_recovered,'source':str(source),'source_sha256':digest,'rigid_correction_m':matrix.tolist(),'tilt_deg':tilt})
    times=np.arange(901)/30.;speed=float(request.get('speed_m_s',.05))
    if not .03<=speed<=.15:raise ValueError('cached cat route must be an explicit slow walk')
    if request.get('intervals_s') not in (None,[[0.,30.]]):raise ValueError('cached cat planted pauses are not admitted')
    period=float(cpu[-1]['time_s']);phase_indices=np.rint((times*speed/.15%period)/period*40).astype(int)%40
    poly=shape(scene['production_scenario']['motion_floor_polygon']).buffer(-.05)
    failures=[];accepted=None;prefilter={'polygon_rejected':0,'native_navmesh_rejected':0,'full_gate_attempted':0}
    import time
    search_started_cpu=time.process_time()
    start=entry['applied_root_input'][:3,3]
    raw_candidates=[start,*[np.asarray(p)-[0,1.3,0] for p in scene['native_placement_candidates'][:256]]]
    candidates=list({tuple(round(float(x),5)for x in p[[0,2]]):p for p in raw_candidates}.values())
    other_centers=[np.asarray(e['applied_root_input'][:3,3])for e in video.actors if e is not entry]
    def candidate_score(p):
        from shapely.geometry import Point
        border=poly.boundary.distance(Point(float(p[0]),float(p[2])))
        others=min((np.linalg.norm((p-q)[[0,2]])for q in other_centers),default=100.)
        return -min(border,others)
    candidates.sort(key=candidate_score)
    original_ids={int(x['object'].object_id) for x in video.actors}
    for p0 in candidates:
        for yaw in [0.,math.pi/2,math.pi,-math.pi/2]:
            direction=yaw_matrix(yaw)@np.array([0.,0.,-1.]);positions=np.asarray(p0)+times[:,None]*speed*direction
            if time.process_time()-search_started_cpu>1800:break
            if not poly.covers(LineString([positions[0][[0,2]],positions[-1][[0,2]]])):
                prefilter['polygon_rejected']+=1;continue
            # Reject a path outside the real admitted NavMesh before spending
            # any of the bounded original-mesh/contact attempts.
            snaps=np.array([video.pathfinder.snap_point(p) for p in positions]);nav=float(np.linalg.norm((snaps-positions)[:,[0,2]],axis=1).max())
            if not math.isfinite(nav) or nav>.002:
                prefilter['native_navmesh_rejected']+=1;continue
            prefilter['full_gate_attempted']+=1
            try:
                floor=PhysicalFloor(video,float(house['room']['measured_floor_y_m']),original_ids)
                positions[:,1]=floor(p0)
                clearance=TriangleClearance(video,house,positions[0,1],entry);clearance.record_only=True;clearance.stride=10
                frames=[];bounds=[];minimum=math.inf
                for i,t in enumerate(times):
                    pose=prepared[int(phase_indices[i])];rot=yaw_matrix(yaw);mesh=pose['mesh'];distance,world=clearance.check(mesh.vertices,mesh.faces,positions[i],yaw);minimum=min(minimum,distance)
                    if world[:,1].min()<positions[i,1]-.00101:raise ValueError('cat original whole body penetrates floor beyond native margin')
                    feet={}
                    for side,local in pose['feet'].items():
                        wp=positions[i]+rot@local;level=floor(wp);delta=float(wp[1]-level)
                        contact=pose['flags'][side]
                        if delta<-.00101 or (contact and abs(delta)>.012):raise ValueError('actual cat paw fails floor/12mm stance criterion')
                        feet[side]={'in_contact':contact,'sole_local_m':local.tolist(),'sole_world_m':wp.tolist(),'floor_y_m':level,'floor_delta_m':delta}
                    frames.append({'time_s':float(t),'moving':True,'mouth_m':pose['mouth_m'],'feet':feet,'world_root_m':positions[i].tolist(),'yaw_rad':yaw,'source_cache_frame':int(phase_indices[i]),'minimum_nonfloor_clearance_m':distance,'upright_tilt_deg':pose['tilt_deg']})
                    bounds.append([world.min(0),world.max(0)])
                accepted=(positions,yaw,frames,bounds,minimum,nav,clearance);break
            except ValueError as error:
                failures.append({'start_m':np.asarray(p0).tolist(),'yaw_rad':yaw,'error':str(error)})
                if len(failures)>=12:break
        if accepted or len(failures)>=12 or time.process_time()-search_started_cpu>1800:break
    if not accepted:
        write_json(dirpath/'motion_rejected.json',{'status':'blocked','failures':failures,'prefilter':prefilter,'thresholds_changed':False})
        raise ValueError('no cached-cat route passes actual new-room gates')
    positions,yaw,frames,bounds,minimum,nav,clearance=accepted
    evidence={'status':'pass','revision':'cached-original-cat-new-room-rigid-ground-v1','species_id':'cat','motion_kind':'cat_cached_walk','frame_count':901,'sample_rate_hz':30,
        'source_frames':frames,'world_roots_m':positions.tolist(),'yaw_rad':yaw,'pose_glbs':[prepared[int(i)]['path'] for i in phase_indices],
        'maximum_navmesh_horizontal_error_m':nav,'original_full_skin_floor_collision':clearance.floor_certificate(),'minimum_certified_triangle_nonfloor_clearance_m':minimum,
        'physical_ground_y_m':float(positions[0,1]),'full_skin_bounds_m':[np.asarray(bounds)[:,0].min(0).tolist(),np.asarray(bounds)[:,1].max(0).tolist()],
        'known_bad_10cm_floor_detected':min(f['floor_delta_m']-.1 for row in frames for f in row['feet'].values())<-.012,'thresholds_changed':False,
        'source_cache_frame_count':41,'native_stage_transform_readback':clearance.native_transform_evidence,'source_cache_prepared':[{k:({s:q.tolist() for s,q in v.items()} if k=='feet' else v) for k,v in p.items() if k!='mesh'} for p in prepared],
        'lineage':{'master':candidate['motion_runtime_glb'],'master_sha256':candidate['motion_runtime_sha256'],'roster':a['source_registry'],'roster_sha256':a['source_registry_sha256'],'cache_certificate':str(upstream_path),'cache_certificate_sha256':hashlib.sha256(upstream_path.read_bytes()).hexdigest(),'original_source_cache_unchanged':True},
        'contact_gate_m':.012,'body_floor_penetration_gate_m':.00101,'upright_tilt_gate_deg':10.,
        'failed_candidate_count':len(failures),'prefilter':prefilter,'collision_method':'whole posed original triangles, FCL nonfloor clearance >20mm'}
    write_json(dirpath/'motion_geometry.json',evidence);return evidence
