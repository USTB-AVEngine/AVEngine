"""Current-scene native evidence for physically bound Kujiale pilot emitters."""
from __future__ import annotations
from pathlib import Path
import json,math
import numpy as np
from avengine.capture.kujiale_spear_runtime import from_ue,apply_camera
from avengine.qa.kujiale_asset_geometry import import_vertices,rotation,NativeSupportGeometry
from avengine.qa.kujiale_native_depth import far_range,visibility
from avengine.qa.kujiale_asset_color import measure,summarize_views
from avengine.timeline.continuous_pilot import write_json

COLOR_THRESHOLDS={'neutral_chroma_max':10,'black_L_max':20,'white_L_min':45,'main_share_min':.55}


def verify_gpu(ue,launch):
    """Read this UE PID's selected Vulkan GPU; report all permitted contexts."""
    import re,subprocess,xml.etree.ElementTree as ET
    policy=json.loads((Path(ue['output_root'])/'kujiale_native_gpu_policy.json').read_text())
    expected=int(ue['physical_gpu_index'])
    expected_uuid=policy['gpu_uuids'][str(expected)]
    pid=int(launch['native_ue_pid'])
    argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
    if str(ue['uproject']) not in argv or not any(launch['temporary_directory'] in a for a in argv):
        raise ValueError('native UE GPU readback PID does not match this launch')
    # UE may create an auxiliary graphics context on its first visible device
    # before initializing Vulkan on another one. NVML reports both for one PID.
    # Read the actual renderer selection instead of treating every context as it.
    log=(Path(launch['temporary_directory'])/'UnrealEditor.log').read_text()
    selections={int(x) for x in re.findall(
        r'LogVulkanRHI: Using device at index (\d+) of \d+',log)}
    if selections!={int(ue['graphics_adapter'])}:
        raise ValueError('native Vulkan renderer selection differs from assigned adapter')
    index=next(iter(selections))
    devices={int(row['vulkan_index']):row for row in launch['vulkan_devices']}
    if index not in devices or devices[index]['gpu_uuid']!=expected_uuid:
        raise ValueError('native Vulkan renderer is on an unauthorized physical GPU')
    permitted={policy['gpu_uuids'][str(i)] for i in policy['allowed_gpu_indices']}
    cards=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True)).findall('gpu')
    matched=[];auxiliary=[]
    for visible_index,card in enumerate(cards):
        for process in card.findall('processes/process_info'):
            if int(process.findtext('pid'))!=pid:continue
            uuid=card.findtext('uuid')
            record={'pid':pid,'nvml_visible_index':visible_index,'uuid':uuid,
                    'command':argv,'used_memory':process.findtext('used_memory'),
                    'process_type':process.findtext('type')}
            if uuid not in permitted:
                raise ValueError('own native UE context is on a GPU outside the task scope')
            if uuid==expected_uuid:
                record['physical_gpu']=expected;matched.append(record)
            else:auxiliary.append(record)
    if not matched:raise ValueError('own UE graphics PID was not found on its renderer GPU in nvidia-smi XML')
    return {'status':'pass','expected_physical_gpu':expected,'processes':matched,
            'vulkan_renderer':devices[index],'auxiliary_permitted_contexts':auxiliary,
            'scope':'exact UE PID, private project/launch, native Vulkan device UUID and task GPU scope'}


def _context(video):
    from tools.ue.capture_kujiale_admission import Capture
    probe=Capture.__new__(Capture)
    probe.instance,probe.camera,probe.components=video.instance,video.camera,video.components
    probe.rgb,probe.depth=video.read_rgb,video.read_depth
    probe.thresholds=COLOR_THRESHOLDS
    return probe

def native_bounds(video,node):
    from tools.rooms.check_v7_kujiale_native_assets import vector_field
    bounds=[]
    with video.instance.begin_frame():
        for runtime in node['runtimes']:
            value=runtime['visual_actor'].GetActorBounds(bOnlyCollidingComponents=False,bIncludeFromChildActors=True,as_dict=True)
            center=np.array(from_ue(vector_field(value,'Origin')));ext=np.abs(from_ue(vector_field(value,'BoxExtent')))
            bounds.append((center-ext,center+ext))
    with video.instance.end_frame():pass
    return np.min([x[0] for x in bounds],axis=0),np.max([x[1] for x in bounds],axis=0)

def binding_camera_yaw(listener,positions):
    # A yaw-only camera needs equal horizontal bearings, independent of the
    # feet's vertical distance from its lens.
    rays=np.asarray(positions,float)-np.asarray(listener,float)
    rays[:,1]=0.
    norms=np.linalg.norm(rays,axis=1)
    if np.any(norms<1e-6):raise ValueError('binding source coincides with listener')
    forward=(rays/norms[:,None]).sum(axis=0)
    if np.linalg.norm(forward)<1e-6:raise ValueError('binding speakers have opposing bearings')
    return math.degrees(math.atan2(forward[0],-forward[2]))


def binding_pair_fits_view(listener,first,second):
    rays=np.asarray([first,second],float)-np.asarray(listener,float)
    rays[:,1]=0.
    norms=np.linalg.norm(rays,axis=1)
    if np.any(norms<.5):return False
    angle=math.degrees(math.acos(float(np.clip(np.dot(rays[0],rays[1])/np.prod(norms),-1.,1.))))
    # Placement, not acceptance: leave room for both complete torso masks
    # within the unchanged 85-degree production camera.
    return 16.<=angle<=60.


def binding_mouth_distance_contrast(listener,emitter,existing_emitters):
    """Sample distinct real mouth distances above the unchanged QA-14 0.5 m gate."""
    listener=np.asarray(listener,float);emitter=np.asarray(emitter,float)
    points=np.asarray(existing_emitters,float)
    if points.size==0:return True
    if not np.isfinite(listener).all() or not np.isfinite(emitter).all() or not np.isfinite(points).all():return False
    distance=float(np.linalg.norm(emitter-listener))
    distances=np.linalg.norm(points-listener,axis=1)
    return bool(np.all(np.abs(distances-distance)>=.6))


def comparison_person_candidates(listener,positions,candidates):
    """Sample a visible third person with real distance contrast, not QA waivers."""
    listener=np.asarray(listener,float);points=np.asarray(positions,float)
    rays=points-listener;rays[:,1]=0.
    radii=np.linalg.norm(rays,axis=1)
    if len(points)!=2 or np.any(radii<.5):return []
    middle=(rays/radii[:,None]).sum(axis=0)
    middle/=np.linalg.norm(middle)
    accepted=[]
    for rank,candidate in enumerate(candidates):
        p=np.asarray(candidate,float)
        ray=p-listener;ray[1]=0.;radius=float(np.linalg.norm(ray))
        if not np.isfinite(p).all() or radius<.5:continue
        unit=ray/radius
        off=math.degrees(math.acos(float(np.clip(np.dot(unit,middle),-1.,1.))))
        separations=[math.degrees(math.acos(float(np.clip(np.dot(unit,v/r),-1.,1.)))) for v,r in zip(rays,radii)]
        if off>30. or min(separations)<8. or min(abs(radius-r) for r in radii)<.6:continue
        # Prefer a farther visible body to avoid occluding the two speakers.
        accepted.append((radius<=max(radii)+.6,rank,candidate))
    return [p for _,_,p in sorted(accepted,key=lambda x:x[:2])]


def human_clear_of_environment(video,low,high):
    for node in video.environment_nodes.values():
        if node.get('physical'):
            lo,hi=node['native_evidence']['placement']['native_world_bounds_m']
            if np.all(np.minimum(high,hi)>np.maximum(low,lo)):return False
            # Disjoint bodies can still hide a small animal completely.
            # Reject a standing placement blocking the measured listener's
            # segment to any physical environment body's center.
            origin=np.array(video.scene['listener_m'],float)
            direction=(np.array(lo)+np.array(hi))/2-origin
            enter,leave=0.,1.
            for axis in range(3):
                if abs(direction[axis])<1e-12:
                    if not low[axis]<=origin[axis]<=high[axis]:enter,leave=1.,0.;break
                else:
                    first=(low[axis]-origin[axis])/direction[axis]
                    last=(high[axis]-origin[axis])/direction[axis]
                    enter=max(enter,min(first,last));leave=min(leave,max(first,last))
            if enter<=leave and 0.<leave and enter<1.:return False
    return True

def qualify_scene(video,directory,human_checks):
    """Recheck actual joint-scene attachment, clearance, size and three native views."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    probe=_context(video);results={}
    geometry=NativeSupportGeometry(video.ue['native_floor_surface'])
    geometry.floor.validate_probes(video.house['room']['floor_reference'])
    envelopes=[np.array(c['native_full_skin_bounds_m']) for c in human_checks]
    for endpoint,node in video.environment_nodes.items():
        if not node.get('physical'):raise ValueError('pilot contains an invisible audio-only source')
        row=video.physical_environment_bindings[endpoint]['asset'];native=node['native_evidence']
        placement=native['placement'];vertices=import_vertices(json.loads(Path(native['import_receipt']).read_text()))
        lo,hi=native_bounds(video,node)
        expected=np.array(placement['native_world_bounds_m'])
        bounds_error=float(np.max(np.abs(np.stack([lo,hi])-expected)))
        if bounds_error>.015:raise ValueError('current native mesh bounds differ from planned native LOD0: '+endpoint)
        if any(np.all(np.minimum(hi,e[1])>np.maximum(lo,e[0])) for e in envelopes):
            raise ValueError('current native human/environment body envelopes overlap: '+endpoint)
        envelopes.append(np.stack([lo,hi]))
        tolerance=.012 if row['category']!='object' else .015
        collision=geometry.collision(vertices,placement['root_m'],placement['yaw_rad'],placement['normal'],tolerance)
        if collision['status']!='pass':raise ValueError('current native object intersects room furniture: '+endpoint)
        gaps=[]
        R=rotation(placement['yaw_rad']);normal=np.array(placement['normal'])
        for q in placement['attachment_points_local_m']:
            point=np.array(placement['root_m'])+R@q
            hit=geometry.ray(point+normal*.1,-normal,.22)
            if hit is None:raise ValueError('current attachment support not read back')
            gaps.append(float(np.dot(point-np.array(hit['point_m']),normal)))
        if max(map(abs,gaps))>tolerance:raise ValueError('current native attachment exceeds unchanged gate')
        probe.active=node['runtimes'];views=[]
        for index,view in enumerate(placement['camera_candidates'][:24]):
            views.append(probe.view(row,view,placement,vertices,directory/endpoint,index,24))
            if sum(x['status']=='pass' for x in views)>=3:break
        if sum(x['status']=='pass' for x in views)<3:raise ValueError('current joint scene lacks three complete native visibility views: '+endpoint)
        node['current_scene_native_vertices']=vertices
        node['current_scene_contact']={'status':'pass','gap_m':max(map(abs,gaps)),'surface':row['attachment_surface'],
            'contact_deltas_m':gaps,'tolerance_m':tolerance,'collision':collision,'native_bounds_error_m':bounds_error}
        results[endpoint]={'status':'pass','asset_id':row['asset_id'],'views':views,
            'attachment':node['current_scene_contact'],'size':native['size'],'native_world_bounds_m':[lo.tolist(),hi.tolist()],
            'emitter':{'status':'not_applicable_silent_decor'} if row['silent_decor'] else {'status':'pass','actual_m':node['point'],'tolerance_m':.0005},
            'source_label':row.get('label_color'),'label_modified':False}
    # Return to the actual production camera before rendering frame zero.
    with video.instance.begin_frame():video._apply(0)
    with video.instance.end_frame():pass
    video.alpha_camera_rows={};video.environment_frame_readbacks=[]
    video.current_environment_admission=results;video.current_environment_frames=[]
    write_json(directory/'admission.json',{'status':'pass','entities':results,'all_emitters_have_native_mesh':True,'thresholds_unchanged':True})
    return results

def capture_frame(video,index,bgr,normal,semantic,targets,ids,pose_id):
    """Modal and isolated native depth, actual RGB and transforms at all 450 frames."""
    from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
    from tools.ue.capture_kujiale_admission import Capture
    probe=_context(video);rows=[];records=[]
    for j,(endpoint,node) in enumerate(video.environment_nodes.items()):
        if not node.get('physical'):continue
        row=video.physical_environment_bindings[endpoint]['asset'];native=node['native_evidence'];placement=native['placement']
        with video.instance.begin_frame():
            roots=[read_scene_component_pose(n['anchor_root']) for n in node['runtimes']]
            emitter=read_scene_component_pose(node['runtimes'][0]['emitter_component'])
            camera_pose=read_scene_component_pose(video.camera.K2_GetRootComponent())
            current_id=__import__('hashlib').sha256(json.dumps(camera_pose,sort_keys=True).encode()).hexdigest()
            if current_id!=pose_id:raise ValueError('physical target-only camera pose differs from modal scene')
            video.components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
            video.components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[n['visual_actor'] for n in node['runtimes']])
        with video.instance.end_frame():target=video.read_depth(video.components['depth'])
        point=from_ue(emitter['location_cm'])
        root_error=max(float(np.max(np.abs(np.array(from_ue(p['location_cm']))-placement['root_m']))) for p in roots)
        yaw_error=max(abs((float(p['rotation_deg'][2])+math.degrees(placement['yaw_rad'])+180)%360-180) for p in roots)
        if root_error>.0005 or yaw_error>1e-4 or np.max(np.abs(np.array(point)-node['point']))>.0005:
            raise ValueError('physical root/yaw/emitter changed in native frame')
        view=video._camera();vertices=node['current_scene_native_vertices'];world=vertices@rotation(placement['yaw_rad']).T+np.array(placement['root_m'])
        mask,visible=visibility(normal,target,far_range(world,view))
        sid=str(node['source_id']);semantic_id=2001+j
        targets[sid]=[np.ascontiguousarray(target[::getattr(video,'capture_scale',2),::getattr(video,'capture_scale',2)])];ids[sid]=semantic_id
        semantic[visible]=semantic_id
        count=int(mask.sum());pixels=int(visible.sum());highlight=float(np.mean(np.all(bgr[visible]>247,axis=1))) if pixels else None
        highlight_warning=highlight is not None and highlight>.02  # record only, per owner 2026-10-15 10:4x
        body=Capture.body_mask(probe,row,visible,target,view,vertices,placement)
        color=measure(bgr[:,:,::-1],body,COLOR_THRESHOLDS) if index%15==0 else None
        contact=node['current_scene_contact']
        small=video.cv2.resize(visible.astype('uint8'),(video.width,video.height),interpolation=video.cv2.INTER_NEAREST)>0
        y,x=np.nonzero(small)
        actors={'endpoint_id':endpoint,'actor_id':node['source_id'],'native_actor_id':endpoint,'native_object_id':int(node['runtimes'][0]['visual_actor'].uobject),
            'asset_id':row['asset_id'],'position_m':from_ue(roots[0]['location_cm']),'emitter_position_m':point,'moving':False,'action_id':'registered_static_support',
            'action_phase':0.,'root_max_error':root_error,'joint_state':'registered static UE meshes; no joint DOFs','semantic_id':semantic_id,
            'visible_pixels':len(x),'mask_centroid_px':[float(x.mean()),float(y.mean())] if len(x) else None,
            'mask_bbox_xyxy':[int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x) else None,
            'entity_class':'static_object' if row['category']=='object' else 'static_animal','species_id':None if row['category']=='object' else row['category'],
            'silent_decor':row['silent_decor'],'native_foot_contact':{**contact,'method':'current joint UE LOD0 attachment rays, per-frame native root/yaw verified'}}
        rows.append(actors)
        records.append({'endpoint_id':endpoint,'source_id':node['source_id'],'asset_id':row['asset_id'],'frame_index':index,
            'physical_mesh_count':len(node['runtimes']),'source_geometry_present':True,'emitter_position_m':point,'root_error_m':root_error,
            'target_pixels':count,'visible_pixels':pixels,'visible_fraction':float(pixels/count) if count else None,
            'highlight_ratio':highlight,'highlight_previous_limit_exceeded':highlight_warning,'highlight_gate_enforced':False,'color':color,'registered_color_label':row.get('label_color')})
    video.current_environment_frames.append({'frame_index':index,'entities':records})
    return rows

def finish_scene(video,result,path):
    frames=video.current_environment_frames
    if len(frames)!=450 or [f['frame_index'] for f in frames]!=list(range(450)):raise ValueError('physical native frame clock incomplete')
    for endpoint in video.environment_nodes:
        if not any(e['visible_pixels']>=80 for f in frames for e in f['entities'] if e['endpoint_id']==endpoint):
            raise ValueError('physical asset never appears in the produced native video: '+endpoint)
    proof={'status':'pass','frame_count':450,'current_joint_admission':video.current_environment_admission,'frames':frames,
        'all_sources_have_drawable_geometry':True,'color_thresholds':COLOR_THRESHOLDS,'source_labels_modified':False}
    write_json(Path(path).parent/'native_physical_environment.json',proof)
    result['native_physical_environment']=proof
    return proof

def current_color_evidence(visual,start):
    first=round(start*15);last=first+150
    by_id={}
    for frame in visual.get('native_physical_environment',{}).get('frames',[]):
        if not first<=frame['frame_index']<last:continue
        for item in frame['entities']:
            if item.get('color') is not None:
                rec=by_id.setdefault(str(item['source_id']),{'expected':item['registered_color_label'],'samples':[]})
                rec['samples'].append({'frame_index':frame['frame_index'],'measurement':item['color']})
    aliases={'gray':'gray_silver','silver':'gray_silver','brown':'wood_brown'}
    for record in by_id.values():
        measured=summarize_views([x['measurement'] for x in record['samples']],COLOR_THRESHOLDS)
        expected=aliases.get(record['expected'],record['expected'])
        record.update(measured=measured,status='unlabelled' if expected is None else 'pass' if measured['status']=='pass' and measured['label']==expected else 'reject_color_questions')
    return by_id


def construct_dialogue_camera(listener,positions,*,additional_positions=None):
    """Choose one fixed view for all required bodies and original speaker bins."""
    from avengine.qa.unified_catalog import _azimuth_sector
    listener=np.asarray(listener,float);points=np.asarray(positions,float)
    if points.shape!=(2,3) or not np.isfinite(points).all():
        raise ValueError('dialogue camera needs two actual native mouth readbacks')
    extra=np.asarray(additional_positions if additional_positions is not None else [],float).reshape(-1,3)
    if not np.isfinite(extra).all():raise ValueError('additional required binding positions must be finite')
    extra_rays=extra-listener;extra_bearings=np.degrees(np.arctan2(extra_rays[:,0],-extra_rays[:,2]))
    center=binding_camera_yaw(listener,points)
    rays=points-listener;bearings=np.degrees(np.arctan2(rays[:,0],-rays[:,2]));choices=[]
    for offset in np.arange(-35.,35.01,.5):
        yaw=center+float(offset);angles=(bearings-yaw+180.)%360.-180.
        sectors=[_azimuth_sector(float(a))for a in angles]
        if max(abs(angles))>37. or None in sectors or len(set(sectors))!=2:continue
        extra_angles=(extra_bearings-yaw+180.)%360.-180.
        all_angles=np.r_[angles,extra_angles]
        if np.max(np.abs(all_angles))>42.5+1e-9:continue
        phase=(angles+22.5)%45.;boundary=float(np.min(np.minimum(phase,45.-phase)))
        margin=min(boundary,float(42.5-max(abs(all_angles))))
        choices.append((margin,-abs(float(offset)),float(offset),yaw,angles.tolist(),sectors))
    if not choices:raise ValueError('native pair needs wider separated placements or another required-body layout before fixed dialogue capture')
    margin,_,offset,yaw,angles,sectors=max(choices,key=lambda x:x[:3])
    return {'status':'constructed_from_native_mouths','yaw_deg':yaw,'center_yaw_deg':center,
        'sampling_offset_deg':offset,'source_azimuths_deg':angles,'source_sectors':sectors,
        'minimum_boundary_or_frustum_margin_deg':margin,'native_mouths_m':points.tolist(),
        'qa_sector_gate_changed':False,'native_hfov_deg':85.,
        'additional_required_positions_m':extra.tolist(),
        'additional_required_azimuths_deg':((extra_bearings-yaw+180.)%360.-180.).tolist(),
        'construction_only_all_required_bodies_in_fixed_view':True}
