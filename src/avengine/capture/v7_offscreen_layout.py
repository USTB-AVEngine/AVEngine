"""Admit deliberately off-camera drawables using actual native projection.

Nothing assigns a QA visibility state. The normal full native target/modal
mask stream remains the truth authority, including zero target pixels.
"""
import math
import numpy as np


def offscreen_role(scene,asset_id):
    role=scene.get('native_view_roles',{}).get(asset_id,{})
    return role if role.get('view')=='outside'else None


def azimuth_degrees(delta,forward):
    return math.degrees(math.atan2(float(np.asarray(delta)@np.cross(forward,[0.,1.,0.])),float(np.asarray(delta)@forward)))


def proposal_view_ok(scene,asset_id,delta,forward,ordinary_visible):
    role=offscreen_role(scene,asset_id)
    if role:return abs(azimuth_degrees(delta,forward))>=float(role.get('minimum_abs_azimuth_deg',60.))
    return ordinary_visible


def native_pixels_ok(scene,entries,counts,minimum_visible_pixels=100):
    for entry in entries:
        count=counts.get(entry['declaration']['asset_id'],0)
        if offscreen_role(scene,entry['declaration']['asset_id']):
            if count!=0:return False
        elif count<minimum_visible_pixels:return False
    return True


def vertices_outside_camera(vertices,world_to_camera,projection):
    vertices=np.asarray(vertices,float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)or not np.isfinite(vertices).all():raise ValueError('actual original drawable vertices required')
    homogeneous=np.column_stack([vertices,np.ones(len(vertices))])
    clip=homogeneous@np.asarray(world_to_camera,float).T@np.asarray(projection,float).T
    w=clip[:,3]
    return bool(np.all(w<=0)or any(np.all(clip[:,axis]>w)or np.all(clip[:,axis]<-w)for axis in (0,1)))


def native_projection_proof(video,scene,entry,visible_pixels):
    if not offscreen_role(scene,entry['declaration']['asset_id']):return None
    sensor=video.sim._sensors['rgb']._sensor_object
    projection=np.asarray(sensor.render_camera.projection_matrix,float)
    world_to_camera=np.linalg.inv(np.asarray(sensor.node.absolute_transformation(),float))
    local=np.asarray(entry['original_mesh'].vertices,float);transform=np.asarray(entry['applied_root_input'],float)
    vertices=local@transform[:3,:3].T+transform[:3,3]
    outside=vertices_outside_camera(vertices,world_to_camera,projection)
    observed=video._apply_entry(entry,scene)
    if visible_pixels!=0 or not outside or observed.get('native_object_id')is None:return {'status':'blocked','reason':'registered body is not wholly outside actual native camera projection'}
    return {'status':'pass','asset_id':entry['declaration']['asset_id'],'native_object_id':observed['native_object_id'],
        'native_visible_pixels':int(visible_pixels),'original_mesh_vertices':len(local),'all_vertices_outside_native_frustum_plane':True,
        'native_projection_matrix':projection.tolist(),'native_world_to_camera':world_to_camera.tolist(),
        'source_is_real_drawable':True,'QA_visibility_overridden':False}


def coverage_errors(scene,assets,visual,visible_frame_counts):
    errors=[];byid={a['asset_id']:a for a in assets};proofs={a['asset_id']:a.get('intentional_offscreen_proof')for a in visual.get('visible_attachment_checks',[])}
    for aid,count in visible_frame_counts.items():
        role=offscreen_role(scene,aid)
        if not role:
            if not count:errors.append('registered entity never reached 100 native visible pixels')
            continue
        proof=proofs.get(aid)
        if (aid not in byid or not proof or proof.get('status')!='pass'or not proof.get('source_is_real_drawable')
                or not proof.get('all_vertices_outside_native_frustum_plane')or proof.get('native_object_id')is None):
            errors.append('intentional offscreen body lacks actual native drawable/projection proof')
        if role.get('static',True)and count:errors.append('static offscreen body entered the actual image')
    if set(scene.get('native_view_roles',{}))-set(byid):errors.append('offscreen role has no actual registered body')
    return errors
