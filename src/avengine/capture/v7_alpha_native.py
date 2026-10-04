"""CPU native alpha scenes: approved static poses, measured contact, camera pan.

Camera orientation changes never move an unqualified person. Exact native
mouths, camera poses, modal/target-only masks and source materials are retained.
The standing tolerance is the existing 15 mm asset contact check; a conservative
full-skin compound must pass, including a known bad floor-penetration control.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.timeline.continuous_pilot import write_json

ALPHA_VISUAL_REVISION='v7-alpha-native-static-grounded-camera-v1'


def yaw_at(scene, time_s):
    knots=scene['camera_yaw_knots']
    return float(np.interp(time_s,[k['time_s'] for k in knots],[k['yaw_deg'] for k in knots]))


def forward_at(scene,time_s):
    yaw=math.radians(yaw_at(scene,time_s))
    return np.array([math.sin(yaw),0.,-math.cos(yaw)])


def rotate_world_foa(samples,camera_frames,sample_rate=16000,fps=15):
    """Interpolate observed unwrapped yaw, preserving per-sample FOA energy."""
    a=np.asarray(samples,float)
    if a.ndim!=2 or a.shape[0]!=4 or not camera_frames:raise ValueError('native FOA/camera stream missing')
    times=np.array([r['frame_index']/fps for r in camera_frames])
    angles=np.unwrap(np.radians([r['yaw_deg'] for r in camera_frames]))
    yaw=np.interp(np.arange(a.shape[1])/sample_rate,times,angles)
    sn=np.sin(yaw);cs=np.cos(yaw);xyz=a[[3,1,2]]/math.sqrt(3)
    x,y,z=xyz
    return np.stack((a[0],-cs*x-sn*z,y,sn*x-cs*z))


def native_skin(entry):
    from avengine.assets.glb import load_glb
    from avengine.assets.skinning import compile_skinning
    from avengine.qa.v7_human_admission import skin_native_vertices
    if 'alpha_skin' not in entry:
        doc=load_glb(entry['bundle'].paths_by_role['visual']);compiled=compile_skinning(doc)
        names=[doc.json['nodes'][j].get('name','') for j in compiled.joint_node_indices]
        links={entry['object'].get_link_name(k):int(k) for k in entry['object'].get_link_ids()}
        if any(n not in links and n!='AVEngine Human Root' for n in names):raise ValueError('unbound full-skin joint')
        entry['alpha_skin']=(doc,compiled,names,links)
    doc,compiled,names,links=entry['alpha_skin'];obj=entry['object']
    matrices=np.stack([np.asarray(obj.root_scene_node.absolute_transformation() if n=='AVEngine Human Root'
        else obj.get_link_scene_node(links[n]).absolute_transformation(),float) for n in names])
    vertices,weights=skin_native_vertices(compiled,matrices)
    return vertices,weights,names,doc,compiled


def sole_points(vertices,weights,names):
    result=[]
    for side in ['L','R']:
        ordinals=[j for j,n in enumerate(names) if n.startswith('Bip01 '+side+' Foot') or n.startswith('Bip01 '+side+' Toe')]
        mask=np.concatenate([np.sum(w*np.isin(j,ordinals),axis=1)>=.5 for j,w in weights])
        if not mask.any():raise ValueError('native foot vertices missing')
        values=vertices[mask];result.append(values[np.argmin(values[:,1])])
    return result


def stage_sole_rays(video,points):
    rows=[]
    actor_ids={int(a['object'].object_id) for a in video.actors}
    for point in points:
        origin=point+[0.,.25,0.]
        ray=video.hs.geo.Ray(video.mn.Vector3(*origin),video.mn.Vector3(0.,-1.,0.))
        hits=[h for h in video.sim.cast_ray(ray).hits if int(h.object_id) not in actor_ids]
        if not hits or hits[0].ray_distance>.6:raise ValueError('native stage floor unavailable')
        floor=np.asarray(hits[0].point,float)
        rows.append({'sole_world_m':point.tolist(),'floor_world_m':floor.tolist(),
            'signed_contact_error_m':float(point[1]-floor[1]),'object_id':int(hits[0].object_id)})
    return rows


def _triangles(doc,index):
    a=doc.json['accessors'][index];v=doc.json['bufferViews'][a['bufferView']]
    dtype={5121:'u1',5123:'<u2',5125:'<u4'}[a['componentType']]
    if a['type']!='SCALAR' or a.get('sparse'):raise ValueError('unsupported original triangles')
    start=int(v.get('byteOffset',0))+int(a.get('byteOffset',0))
    return np.ndarray((a['count'],),dtype=dtype,buffer=doc.binary,offset=start,
        strides=(v.get('byteStride',np.dtype(dtype).itemsize),)).astype(np.int64).reshape(-1,3)


def standing_collision_probe(video,entry,directory):
    """Every original triangle is wholly included in a per-bone convex hull."""
    import trimesh
    from scipy.spatial import ConvexHull,QhullError
    from avengine.assets.world_contact import REFERENCE_MAXIMUM_CONTACT_HORIZONTAL_STEP_M as tolerance
    vertices,weights,names,doc,compiled=native_skin(entry)
    soles=stage_sole_rays(video,sole_points(vertices,weights,names))
    centered=(vertices.min(0)+vertices.max(0))/2
    groups={};offset=0;covered=set();triangles=0
    for primitive,(ordinals,w) in zip(compiled.primitives,weights,strict=True):
        p=doc.json['meshes'][primitive.mesh_index]['primitives'][primitive.primitive_index]
        if p.get('mode',4)!=4:raise ValueError('native skin primitive is not triangles')
        dominant=ordinals[np.arange(len(w)),np.argmax(w,axis=1)]
        for triangle in _triangles(doc,p['indices']):
            groups.setdefault(int(dominant[int(triangle[0])]),set()).update(int(v)+offset for v in triangle);triangles+=1
        for i,bone in enumerate(dominant):groups.setdefault(int(bone),set()).add(i+offset);covered.add(i+offset)
        offset+=len(primitive.positions)
    if len(covered)!=len(vertices):raise ValueError('incomplete native skin coverage')
    mesh=trimesh.Scene()
    for bone,ids in sorted(groups.items()):
        points=vertices[sorted(ids)]-centered
        try:
            hull=ConvexHull(points);part=trimesh.Trimesh(vertices=points,faces=hull.simplices,process=False);part.fix_normals()
        except QhullError:
            part=trimesh.creation.box(np.maximum(np.ptp(points,axis=0),.001));part.apply_translation((points.min(0)+points.max(0))/2)
        mesh.add_geometry(part,geom_name='bone_'+str(bone),node_name='bone_'+str(bone))
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(entry['endpoint_id']+'_static_full_skin.glb')
    path.write_bytes(mesh.export(file_type='glb'))
    manager=video.sim.get_object_template_manager();attr=manager.create_new_template('v7_alpha_skin_'+entry['endpoint_id'])
    attr.render_asset_handle=str(path);attr.collision_asset_handle=str(path);attr.is_collidable=True
    attr.join_collision_meshes=False;attr.margin=.001
    if not attr.use_mesh_for_collision:raise ValueError('native full-skin proxy is not mesh-backed')
    # The exported mesh is centered explicitly. Do not let an automatic COM
    # recenter a differently weighted compound and move its collision shape.
    properties={}
    for name in ['COM','compute_COM_from_shape']:
        if hasattr(attr,name):
            properties[name]=str(getattr(attr,name))
    if hasattr(attr,'COM'):attr.COM=video.mn.Vector3(0.,0.,0.)
    if hasattr(attr,'compute_COM_from_shape'):attr.compute_COM_from_shape=False
    tid=manager.register_template(attr);rigid=video.sim.get_rigid_object_manager();body=rigid.add_object_by_template_id(tid)
    body.motion_type=video.hs.physics.MotionType.DYNAMIC;body.translation=video.mn.Vector3(*centered)
    actor_ids={int(a['object'].object_id) for a in video.actors}
    def contacts():
        video.sim.perform_discrete_collision_detection();rows=[]
        for contact in video.sim.get_physics_contact_points():
            pair={int(contact.object_id_a),int(contact.object_id_b)}
            if int(body.object_id) in pair and not pair.intersection(actor_ids):
                rows.append({'object_id_a':int(contact.object_id_a),'object_id_b':int(contact.object_id_b),
                    'contact_distance_m':float(contact.contact_distance)})
        return rows
    try:
        actual=contacts();body.translation=video.mn.Vector3(*(centered-[0.,.1,0.]));bad=contacts()
    finally:rigid.remove_object_by_id(body.object_id)
    floor_ok=all(abs(r['signed_contact_error_m'])<=tolerance for r in soles)
    skin_ok=all(r['contact_distance_m']>=-tolerance for r in actual)
    bad_detected=any(r['contact_distance_m']<-tolerance for r in bad)
    return {'status':'pass' if floor_ok and skin_ok and bad_detected else 'blocked',
        'endpoint_id':entry['endpoint_id'],'asset_id':entry['declaration']['asset_id'],
        'action':'idle_pose0','contact_tolerance_m':tolerance,'tolerance_changed':False,
        'root_y_adjustment_m':entry.get('alpha_root_adjustment',0.),'sole_readbacks':soles,
        'native_skin_vertices':len(vertices),'native_triangles':triangles,'full_skin_covered':True,
        'native_full_skin_bounds_m':[vertices.min(0).tolist(),vertices.max(0).tolist()],
        'compound_hulls':len(groups),'compound_margin_m':.001,'contacts':actual,
        'known_bad_10cm_floor_contacts':bad,'known_bad_detected':bad_detected,
        'native_template_original_COM_properties':properties,'explicit_center_of_mass':[0.,0.,0.],
        'proxy_path':str(path),'motion_admission':'static pose only; walking not admitted'}


class NativeAlphaVideo(NativeContinuousVideo):
    def __init__(self,house,scene,inputs,assets,**kwargs):
        if inputs.get('render_backend','cpu_llvmpipe')!='cpu_llvmpipe':raise ValueError('only approved CPU llvmpipe backend')
        for source in scene['sources'].values():
            if any(not np.array_equal(k['position_m'],source['trajectory'][0]['position_m']) for k in source['trajectory']):
                raise ValueError('alpha human trajectories must be static; walking is not admitted')
        super().__init__(house,scene,inputs,assets,**kwargs)
        self.camera_frame_readbacks=[];self.standing_checks=[]

    def _apply(self,scene,frame_index):
        from avengine.assets.habitat_capture import _apply_root_with_habitat,transform_to_matrix
        from avengine.capture.mp3d_multi_actor import _emitter_position
        from avengine.timeline.continuous_scene import camera_basis
        rows=super()._apply(scene,frame_index)
        for entry,row in zip(self.actors,rows,strict=True):
            body_yaw=scene['sources'][entry['endpoint_id']].get('body_yaw_deg')
            if body_yaw is not None:
                angle=math.radians(body_yaw);f=np.array([math.sin(angle),0.,-math.cos(angle)])
                actor_root=entry['applied_root_input']@np.linalg.inv(entry['bundle'].actor_from_skin_root)
                actor_root[:3,:3]=np.stack((np.cross(f,[0.,1.,0.]),[0.,1.,0.],-f),axis=1)
                applied=actor_root@np.asarray(entry['bundle'].actor_from_skin_root)
                _apply_root_with_habitat(entry['object'],applied,qt=self.qt,mn=self.mn)
                entry['applied_root_input']=applied
            if 'alpha_root_adjustment' not in entry:
                vertices,weights,names,*_=native_skin(entry)
                floor=stage_sole_rays(self,sole_points(vertices,weights,names))
                adjustment=-min(r['signed_contact_error_m'] for r in floor)
                if abs(adjustment)>.5:raise ValueError('static human floor correction outside bounded native support')
                entry['alpha_root_adjustment']=adjustment
            applied=entry['applied_root_input'].copy();applied[1,3]+=entry['alpha_root_adjustment']
            _apply_root_with_habitat(entry['object'],applied,qt=self.qt,mn=self.mn)
            entry['applied_root_input']=applied
            native=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)
            error=float(np.max(np.abs(native-applied)))
            if error>1e-5:raise ValueError('native standing root correction did not apply')
            mouth=_emitter_position(entry['object'],entry['mouth_link'],transform_to_matrix(entry['anchor']['joint_from_anchor']))
            row.update(position_m=native[:3,3].tolist(),emitter_position_m=mouth.tolist(),root_max_error=error,
                alpha_action='idle_pose0_only',native_floor_corrected=True)
        f,r,u=camera_basis(forward_at(scene,frame_index/15))
        state=self.hs.AgentState();state.position=np.asarray(scene['listener_m'],dtype=np.float32)
        state.rotation=self.qt.from_rotation_matrix(np.stack((r,u,-f),axis=1));self.sim.get_agent(0).set_state(state)
        actual=self.sim.get_agent(0).get_state();matrix=self.qt.as_rotation_matrix(actual.rotation);actual_f=-matrix[:,2]
        yaw=math.degrees(math.atan2(actual_f[0],-actual_f[2]))
        self.camera_frame_readbacks.append({'frame_index':frame_index,'time_s':frame_index/15,
            'position_m':np.asarray(actual.position,float).tolist(),'yaw_deg':yaw,'forward_m':actual_f.tolist(),
            'orientation_wxyz':[float(actual.rotation.w),*actual.rotation.imag.tolist()],
            'camera_contract':'native observer rotation; listener position static'})
        if frame_index%15==0:
            obs=self.sim.get_sensor_observations();self._check_source_frame(np.asarray(obs['rgb'])[:,:,:3],np.asarray(obs['semantic']),frame_index)
        return rows

    def qualify_standing(self,scene,directory):
        self._apply(scene,0)
        self.standing_checks=[standing_collision_probe(self,e,directory) for e in self.actors]
        write_json(Path(directory)/'standing_admission.json',{'checks':self.standing_checks,
            'status':'pass' if all(r['status']=='pass' for r in self.standing_checks) else 'blocked'})
        # Probe frames are not part of the published video clock.
        self.camera_frame_readbacks=[]
        return self.standing_checks

    def encode(self,scene,path,review_frames=None,*,collect_pixel_visibility=True):
        checks=self.qualify_standing(scene,Path(path).parent/'standing')
        if not all(r['status']=='pass' for r in checks):raise ValueError('alpha standing full-skin/native contact admission failed')
        result=super().encode(scene,path,review_frames=list(range(0,450,15)),collect_pixel_visibility=collect_pixel_visibility)
        result.update(camera_frame_readbacks=self.camera_frame_readbacks,standing_admission=checks,
            alpha_visual_revision=ALPHA_VISUAL_REVISION)
        protocol_cameras=[]
        from avengine.timeline.continuous_scene import camera_basis
        for tick in range(301):
            t=tick/10;f,r,u=camera_basis(forward_at(scene,t))
            state=self.hs.AgentState();state.position=np.asarray(scene['listener_m'],dtype=np.float32)
            state.rotation=self.qt.from_rotation_matrix(np.stack((r,u,-f),axis=1));self.sim.get_agent(0).set_state(state)
            actual=self.sim.get_agent(0).get_state();matrix=self.qt.as_rotation_matrix(actual.rotation);observed=-matrix[:,2]
            protocol_cameras.append({'time_s':t,'position_m':np.asarray(actual.position,float).tolist(),
                'yaw_deg':math.degrees(math.atan2(observed[0],-observed[2])),
                'forward_m':observed.tolist(),'measurement':'native AgentState 100ms readback'})
        result['protocol_camera_readbacks']=protocol_cameras
        # The base receipt is retained; the alpha extension has a separate name.
        write_json(Path(path).parent/'native_alpha_visual_evidence.json',result)
        return result
