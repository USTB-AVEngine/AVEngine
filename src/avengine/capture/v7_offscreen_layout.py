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


UNCONSTRAINED_OUTSIDE_AZIMUTH_DEG=60.


def visibility_required(scene,declaration):
    """owner 10-15: the question decides whether a body must be in view, not a blanket all-visible rule.

    Required only by an explicit 'inside' role / required_visible_sometime, an explicit scene list, or a legacy
    all-visible scene flag.  The placement loop adds requested sectors, colour pairs, movers and pinned sites.
    """
    aid=declaration['asset_id'];role=scene.get('native_view_roles',{}).get(aid,{})
    if role.get('view')=='outside':return False
    if role.get('view')=='inside' or role.get('required_visible_sometime'):return True
    if aid in set(scene.get('native_visibility_required_asset_ids',[])):return True
    return bool(scene.get('native_all_entities_visible'))


def proposal_view_ok(scene,asset_id,delta,forward,ordinary_visible,required=True):
    band=scene.get('native_distance_bands_m',{}).get(asset_id)
    if band is not None:
        distance=float(np.linalg.norm(np.asarray(delta)[[0,2]]))
        if not band[0]<=distance<=band[1]:return False
    if scene.get('native_view_roles',{}).get(asset_id,{}).get('view')=='unconstrained_motion':return True
    role=offscreen_role(scene,asset_id)
    if role:return abs(azimuth_degrees(delta,forward))>=float(role.get('minimum_abs_azimuth_deg',60.))
    if required:return ordinary_visible
    # Unconstrained bodies stand clearly in view or clearly outside it, never on the image edge where "in view"
    # would be ambiguous for the questions that read visibility from the native pixels.
    return ordinary_visible or abs(azimuth_degrees(delta,forward))>=UNCONSTRAINED_OUTSIDE_AZIMUTH_DEG


def native_pixels_ok(scene,entries,counts,minimum_visible_pixels=100,unconstrained=(),new_entry_id=None):
    for entry in entries:
        aid=entry['declaration']['asset_id'];count=counts.get(aid,0)
        if aid in unconstrained and aid!=new_entry_id:continue      # accepted earlier and not needed in view
        if scene.get('native_view_roles',{}).get(aid,{}).get('view')=='unconstrained_motion':
            if getattr(entry.get('object'),'object_id',None) is None:return False
            continue
        role=offscreen_role(scene,aid)
        if role:
            if role.get('static',True) is False:
                # Dynamic 'outside' role (QA-07 entry): outside at frame zero, proven on the actual posed mesh in
                # refresh_actual_offscreen_proofs.  Before the walk is compiled (no motion_episode yet) the body
                # stands at its frame-zero pose, so it must show 0 px; after compilation `counts` are whole-clip
                # maxima and only the required later entry is checked (Claude 10-16; qa07 evidence
                # moving_offscreen_view_bug_01.json, dynamic_outside_initial_phase_02.json).
                if not entry.get('motion_episode'):
                    if count!=0:return False
                elif role.get('required_visible_sometime') and count<minimum_visible_pixels:return False
                continue
            if count!=0:return False
        elif aid in unconstrained and count==0:continue      # naturally off screen; a role is assigned on acceptance
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
    mesh=entry['original_mesh']
    if entry.get('motion_episode'):
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        mesh=indexed_static_mesh(entry['motion_pose_path'])
    local=np.asarray(mesh.vertices,float);transform=np.asarray(entry['applied_root_input'],float)
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
        if scene.get('native_view_roles',{}).get(aid,{}).get('view')=='unconstrained_motion':
            check=next((a for a in visual.get('visible_attachment_checks',[])if a['asset_id']==aid),{})
            proof=check.get('moving_drawable_view_proof',{})
            if aid not in byid or proof.get('native_object_id')is None or not proof.get('actual_per_frame_visibility_required'):
                errors.append('moving view fallback lacks actual registered native drawable proof')
            continue
        role=offscreen_role(scene,aid)
        proof=proofs.get(aid)
        if not role and not count and aid not in scene.get('native_view_roles',{}):
            # Placement runs in a fresh native process. Its approved static
            # offscreen fallback is measured in the returned evidence even
            # when the caller's original scene has no such role yet.
            asset=byid.get(aid,{})
            requests=scene.get('production_scenario',{}).get('motion_requests',{})
            moving=aid in requests or asset.get('endpoint_id')in requests
            if (not moving and proof and proof.get('status')=='pass'
                    and proof.get('source_is_real_drawable')
                    and proof.get('all_vertices_outside_native_frustum_plane')
                    and proof.get('native_object_id')is not None
                    and proof.get('native_visible_pixels')==0
                    and proof.get('QA_visibility_overridden')is False):
                role={'view':'outside','static':True,'authority':'actual returned native drawable/projection proof'}
        if not role:
            if not count:
                declaration=byid.get(aid,{'asset_id':aid})
                if (not visibility_required(scene,declaration)
                        and aid not in scene.get('phase2_semantic_azimuth_targets_deg',{})):
                    # owner 10-15: only bodies the question plan needs in view must appear; a mover may walk out of
                    # (or never into) the fixed view and a bystander may stand off screen.  Their native placement,
                    # contact and readback still prove they are real registered bodies; visibility truths read pixels.
                    # Merge 10-16: returned drawable evidence, when present, must still name the actual native object.
                    if proof and (proof.get('native_object_id')is None or not proof.get('source_is_real_drawable')):
                        errors.append('zero-pixel body evidence lacks its actual registered native drawable')
                    continue
                errors.append('registered entity never reached 100 native visible pixels')
            continue
        if (aid not in byid or not proof or proof.get('status')!='pass'or not proof.get('source_is_real_drawable')
                or not proof.get('all_vertices_outside_native_frustum_plane')or proof.get('native_object_id')is None):
            errors.append('intentional offscreen body lacks actual native drawable/projection proof')
        if role.get('static',True)and count:errors.append('static offscreen body entered the actual image')
        if role.get('required_visible_sometime')and not count:errors.append('entry body never reached 100 actual native visible pixels')
    if set(scene.get('native_view_roles',{}))-set(byid):errors.append('offscreen role has no actual registered body')
    return errors


def refresh_actual_offscreen_proofs(video,scene,checks):
    if not scene.get('native_view_roles'):return
    entries=video.actors
    for entry in entries:video._apply_entry(entry,scene)
    counts,observed=video._native_observed_pixels(scene,entries)
    frame_zero=next((o['pixels'] for o in observed if float(o.get('time_s',-1.))==0.),None)
    # Bodies the question plan does not need in view (placement flag) are not re-required here (owner 10-15).
    unconstrained=tuple(e['declaration']['asset_id'] for e in entries if not e.get('_required_view',True))
    if not native_pixels_ok(scene,entries,counts,unconstrained=unconstrained):raise ValueError('final native frame-zero view roles do not match actual pixels')
    byid={c['asset_id']:c for c in checks}
    for entry in entries:
        aid=entry['declaration']['asset_id']
        role=offscreen_role(scene,aid)
        if not role:continue
        pixels=counts[aid]
        if role.get('static',True) is False:
            # The proof is taken on the frame-zero pose (restored by _native_observed_pixels); a dynamic role is
            # judged on its frame-zero pixels, not on the later entry that the role requires.
            if frame_zero is None:raise ValueError('dynamic offscreen role needs an actual frame-zero native observation')
            pixels=int(frame_zero.get(aid,0))
        proof=native_projection_proof(video,scene,entry,pixels)
        if proof['status']!='pass':raise ValueError('final actual posed body is not outside native camera projection')
        byid[aid]['intentional_offscreen_proof']=proof


def placement_construction_fallback(scene,declaration,surface,target,require_pair):
    """Construct instead of failing when no strict placement passed (owner 10-15: "缺什么补什么", off-screen allowed).

    Stage 'camera_1_2m': a person may stand at the unchanged 1.2 m person/camera gate instead of the 1.5 m
    construction preference.  Stage 'offscreen': an entity the question plan does not need in view becomes a real
    registered body wholly outside the fixed camera (|azimuth| >= 60 deg, zero native pixels, projection proof by
    refresh_actual_offscreen_proofs).  Sector targets, controlled colour pairs, movers and entities that already have
    a view role keep their requirements.  All physical/contact/acoustic gates and all truths are unchanged; truths
    are read from the actual native scene.  Returns the applied stage, or None when nothing is left to relax.
    """
    aid=declaration['asset_id']
    log=scene.setdefault('construction_fallbacks',[])
    done={f['stage'] for f in log if f.get('asset_id')==aid}
    if surface!='floor':return None
    person=declaration.get('kind') in {'human','performer'}
    if person and 'camera_1_2m' not in done:
        scene.setdefault('native_person_camera_construction_m',{})[aid]=1.2
        log.append({'asset_id':aid,'stage':'camera_1_2m','policy':'owner-20261015-construct-not-fail'})
        return 'camera_1_2m'
    movers=set(scene.get('production_scenario',{}).get('motion_requests',{}))
    roles=scene.setdefault('native_view_roles',{})
    moving=(aid in movers or declaration.get('endpoint_id') in movers)
    if (moving and aid not in roles and target is None and not visibility_required(scene,declaration)
            and not (require_pair and declaration.get('controlled_shirt'))):
        roles[aid]={'view':'unconstrained_motion','static':False,'real_registered_drawable_required':True,
                    'actual_per_frame_visibility_required':True,
                    'construction':'owner 10-15 placement fallback: allow a real mover to enter or leave the fixed view'}
        log.append({'asset_id':aid,'stage':'moving_view_construction','policy':'owner-20261015-construct-not-fail'})
        return 'moving_view_construction'
    if ('offscreen' not in done and aid not in roles and target is None and not visibility_required(scene,declaration)
            and not (require_pair and declaration.get('controlled_shirt'))
            and aid not in movers and declaration.get('endpoint_id') not in movers):
        roles[aid]={'view':'outside','static':True,'minimum_abs_azimuth_deg':60.,'real_registered_drawable_required':True,
                    'construction':'owner 10-15 off-screen fallback: no in-view floor placement passed'}
        if person:scene.setdefault('native_person_camera_construction_m',{})[aid]=1.2
        log.append({'asset_id':aid,'stage':'offscreen','policy':'owner-20261015-construct-not-fail'})
        return 'offscreen'
    return None


def record_placement_rejection(directory,asset_id,record,stage=None):
    """Preserve every fallback receipt without replacing immutable evidence."""
    from pathlib import Path
    from avengine.timeline.continuous_pilot import write_json
    name=('attachment_rejection_before_'+stage+'_'+asset_id+'.json') if stage else 'attachment_rejection.json'
    target=Path(directory)/name
    write_json(target,{**record,'status':'retry_with_construction_fallback' if stage else 'fail',**({'stage':stage} if stage else {})})
    return target
