"""New-house contact compilation from admitted clips, with exact triangle gates.

Reference-route receipts do not admit a new placement. Every 30 Hz pose is
checked against the original stage triangles, actual floor and native NavMesh.
Only flat, clear, straight routes are supported by this first adapter.
"""
from __future__ import annotations
import copy, hashlib, math
from pathlib import Path
import numpy as np
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
    def __call__(self, point):
        v=self.video;p=np.asarray(point,float)
        hits=v.sim.cast_ray(v.hs.geo.Ray(v.mn.Vector3(p+[0.,.2,0.]),v.mn.Vector3(0.,-1.,0.)),.85).hits
        hits=[h for h in hits if int(h.object_id) not in self.ignore]
        if not hits:raise ValueError('motion support ray has no actual stage floor')
        y=float(hits[0].point[1])
        if abs(y-self.reference)>.015:raise ValueError('motion support ray hit furniture or a different floor level')
        return y
    def many(self, points):return np.array([self(p) for p in points])


def triangle_mesh(d,b,parts):
    vertices=[];faces=[];offset=0
    for part in parts:
        prim=d['meshes'][d['nodes'][part['node']]['mesh']]['primitives'][part['primitive']]
        vertices.append(part['positions']);ix=accessor(d,b,prim['indices']).astype(np.int32).reshape(-1,3)
        faces.append(ix+offset);offset+=len(part['positions'])
    return np.concatenate(vertices),np.concatenate(faces)


def fcl_model(vertices,faces):
    import fcl
    model=fcl.BVHModel();model.beginModel(len(vertices),len(faces))
    model.addSubModel(np.asarray(vertices,float),np.asarray(faces,np.int32));model.endModel()
    return fcl.CollisionObject(model)


class TriangleClearance:
    def __init__(self, video, house, reference, excluded_entry):
        import fcl
        self.fcl=fcl
        path=Path(house.get('input',{}).get('visual_source') or Path(house['input']['scene_directory'])/(house['input']['scan_id']+'.glb'))
        mesh=loaded_mesh(path);v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
        tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n=np.linalg.norm(normal,axis=1)
        ground=(np.abs(normal[:,1])>.98*np.maximum(n,1e-20)) & (np.max(np.abs(tri[:,:,1]-reference),axis=1)<.02)
        self.stage=fcl_model(v,f[~ground]);self.source=str(path);self.removed=int(ground.sum());self.triangles=int((~ground).sum())
        self.bystanders=[]
        for entry in video.actors:
            if entry is excluded_entry:continue
            mesh=entry['original_mesh'];root=entry['applied_root_input']
            self.bystanders.append(fcl_model(np.asarray(mesh.vertices)@root[:3,:3].T+root[:3,3],mesh.faces))
    def check(self, vertices,faces,root,yaw):
        fcl=self.fcl;world=np.asarray(vertices)@yaw_matrix(yaw).T+root;obj=fcl_model(world,faces)
        distances=[]
        for other in [self.stage,*self.bystanders]:
            result=fcl.CollisionResult()
            if fcl.collide(obj,other,fcl.CollisionRequest(num_max_contacts=1),result):raise ValueError('new motion pose intersects actual stage or another body')
            distances.append(float(fcl.distance(obj,other,fcl.DistanceRequest(),fcl.DistanceResult())))
        if min(distances)<=.02:raise ValueError('motion full-triangle nonfloor clearance is not above frozen 2cm reference criterion')
        return min(distances),world


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


def compile_walking(video,entry,scene,house,inputs,directory,request):
    from shapely.geometry import shape,LineString
    from avengine.timeline.v7_storage_budget import check_write
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    d,b,action,lineage=admitted_master(entry,inputs,house)
    times=np.arange(901)/30.;speed=float(request.get('speed_m_s',.5))
    if not .15<=speed<=1.2:raise ValueError('human walking speed outside explicit slow-to-normal production support')
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
    static_bounds=[np.asarray(a['original_mesh'].vertices)@a['applied_root_input'][:3,:3].T+a['applied_root_input'][:3,3] for a in video.actors if a is not entry]
    accepted=None;failures=[]
    candidates=[entry['applied_root_input'][:3,3].tolist(),*[np.asarray(p)-[0.,1.3,0.] for p in scene['native_placement_candidates']]]
    for p0 in candidates[:96]:
        p0=np.asarray(p0,float)
        for yaw in [0.,math.pi/2,math.pi,-math.pi/2]:
            direction=yaw_matrix(yaw)@np.array([0.,0.,-1.]);positions=p0+distance[:,None]*direction
            poly=shape(scene['production_scenario']['motion_floor_polygon']).buffer(-.55)
            if not poly.covers(LineString([positions[0][[0,2]],positions[-1][[0,2]]])):continue
            if any(np.any((positions[:,0]>v[:,0].min()-.7)&(positions[:,0]<v[:,0].max()+.7)&(positions[:,2]>v[:,2].min()-.7)&(positions[:,2]<v[:,2].max()+.7))for v in static_bounds):continue
            try:
                snaps=np.array([video.sim.pathfinder.snap_point(p) for p in positions]);nav=float(np.linalg.norm((snaps-positions)[:,[0,2]],axis=1).max())
                if not np.isfinite(nav) or nav>.002:raise ValueError('motion actual native navigation horizontal drift exceeds 2mm')
                # The actual stage ground, rather than the NavMesh voxel height.
                ray=video.hs.geo.Ray(video.mn.Vector3(p0+[0.,.2,0.]),video.mn.Vector3(0.,-1.,0.))
                hits=[h for h in video.sim.cast_ray(ray,.85).hits if int(h.object_id)not in {int(a['object'].object_id)for a in video.actors}]
                if not hits:raise ValueError('motion route has no actual native ground')
                floor_y=float(hits[0].point[1]);positions[:,1]=floor_y
                floor=PhysicalFloor(video,floor_y,{int(a['object'].object_id)for a in video.actors})
                solver=FootPlant(d,b,action,clock.tolist(),positions.tolist(),[yaw]*len(times),1.,float(ts[0]),duration,floor)
                solver.times=times
                for values in solver.active.values():values[~moving]=True
                clearance=TriangleClearance(video,house,floor_y,entry);geometry=[];frames=[];minimum=math.inf
                for i,t in enumerate(times):
                    _,parts,ev=solver.frame(i);v,f=triangle_mesh(d,b,parts);dist,world=clearance.check(v,f,positions[i],yaw);minimum=min(minimum,dist)
                    feet=list(ev['feet'].values());contacts=[x for x in feet if x['in_contact']]
                    if min(x['floor_delta_m']for x in feet)<-.008 or any(abs(x['floor_delta_m'])>=.008 or x['support_lock_error_m']>=.003 for x in contacts):raise ValueError('motion source foot contact or lock violates unchanged 8mm/3mm admission')
                    if solver.max_reach>=.001 or ev['pelvis_lowering_m']>=.08:raise ValueError('motion IK reach or pelvis bound failed')
                    geometry.append(parts);frames.append({**ev,'time_s':float(t),'moving':bool(moving[i]),'minimum_nonfloor_clearance_m':dist})
                accepted=(positions,yaw,geometry,frames,floor_y,nav,clearance,minimum);break
            except ValueError as error:failures.append({'start_m':p0.tolist(),'yaw_rad':yaw,'reason':str(error)})
        if accepted:break
    if not accepted:
        write_json(directory/'motion_rejected.json',{'status':'blocked','failures':failures,'lineage':lineage});raise ValueError('admitted human has no fully qualified new-house walking route')
    positions,yaw,parts_list,frames,floor_y,nav,clearance,minimum=accepted
    folder=directory/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize_unchanged_images(d,b,directory/'textures');files=[];bounds=[]
    for i,parts in enumerate(parts_list):
        if i%30==0:check_write(folder,expected_bytes=32_000_000)
        path=folder/(f'{i:04d}.glb');rigid_glb(ed,b,parts,path);files.append(str(path))
        vertices,_=triangle_mesh(d,b,parts);world=vertices@yaw_matrix(yaw).T+positions[i];bounds.append([world.min(0),world.max(0)])
    evidence={'status':'pass','revision':REVISION,'lineage':lineage,'frame_count':901,'sample_rate_hz':30,
        'source_frames':frames,'world_roots_m':positions.tolist(),'yaw_rad':yaw,'pose_glbs':files,
        'maximum_navmesh_horizontal_error_m':nav,'minimum_exact_triangle_nonfloor_clearance_m':minimum,
        'stage_original_glb':clearance.source,'true_ground_triangles_removed':clearance.removed,
        'nonfloor_stage_triangles':clearance.triangles,'physical_ground_y_m':floor_y,
        'pause_policy':'zero root motion and observed double-support pose, foot IK stays planted',
        'known_bad_10cm_floor_detected':min(frames[0]['feet'][s]['floor_delta_m']-.1 for s in frames[0]['feet'])<-.008,
        'full_skin_bounds_m':[np.array(bounds)[:,0].min(0).tolist(),np.array(bounds)[:,1].max(0).tolist()],
        'failed_candidate_count':len(failures),'thresholds_changed':False}
    write_json(directory/'motion_geometry.json',evidence);return evidence
