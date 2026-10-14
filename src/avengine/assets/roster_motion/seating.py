"""Seat-first intake: actual native semantic surfaces or verified silent chairs."""
from pathlib import Path
import copy,json,math,hashlib,time
import numpy as np
from avengine.assets.roster_motion.episode import admitted_master,PhysicalFloor,TriangleClearance,triangle_mesh,externalize_unchanged_images,fcl_model
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.assets.roster_motion.pose import rigid_glb
from avengine.assets.roster_motion.contact import FootPlant,yaw_matrix
from avengine.assets.roster_motion.collision import clipped_surface_minimum,intersections
from avengine.timeline.continuous_pilot import write_json



def original_factor_palette_srgb(doc):
    """These authored untextured chairs use linear glTF baseColorFactor.

    Only the reference input is encoded into the RGB observation space. The
    original GLB, rendered PBR material, measurement function and gates stay
    unchanged. Texture-backed furniture needs a separate explicit path.
    """
    rows=[]
    for material in doc['materials']:
        pbr=material.get('pbrMetallicRoughness',{})
        if pbr.get('baseColorTexture'):raise ValueError('factor-only chair reference cannot handle a texture')
        value=np.array(pbr.get('baseColorFactor',[1,1,1,1])[:3],float)
        if not np.isfinite(value).all()or np.any(value<0)or np.any(value>1):raise ValueError('invalid original chair factor')
        rows.append(np.where(value<=.0031308,12.92*value,1.055*value**(1/2.4)-.055))
    if not rows:raise ValueError('original chair has no declared material colors')
    return np.array(rows)


def existing_native_seats(video,scene,directory):
    from avengine.timeline.continuous_scene import camera_basis
    labels={i:o.category.name().lower()for i,o in enumerate(video.sim.semantic_scene.objects or [])if o is not None and o.category is not None}
    eligible={i:c for i,c in labels.items()if c in {'chair','sofa','couch','bench','stool','armchair','seat','ottoman'}}
    result={'status':'not_available_native_semantic_seat_labels' if not eligible else 'native_surface_scan',
        'native_semantic_objects':len(labels),'semantic_seat_objects':len(eligible),'seats':[],
        'policy':'only actual RGB-D-semantic horizontal surfaces; no AABB top, no unknown furniture relabelled'}
    if eligible:
        samples={}
        for i,yaw in enumerate(np.arange(0,360,30)):
            a=math.radians(yaw);f,r,u=camera_basis([math.sin(a),0,-math.cos(a)])
            state=video.hs.AgentState();state.position=np.asarray(scene['listener_m'],np.float32)
            state.rotation=video.qt.from_rotation_matrix(np.stack((r,u,-f),axis=1));video.sim.get_agent(0).set_state(state)
            obs=video.sim.get_sensor_observations();sem=np.asarray(obs['semantic']);dep=np.asarray(obs['depth'])
            rc=video.sim._sensors['depth']._sensor_object.render_camera
            invp=np.linalg.inv(np.asarray(rc.projection_matrix));invv=np.linalg.inv(np.asarray(rc.camera_matrix))
            def world(x,y):
                p=invp@np.array([2*(x+.5)/video.width-1,1-2*(y+.5)/video.height,-1,1]);p/=p[3]
                p[:3]*=-float(dep[y,x])/p[2];p[3]=1;return (invv@p)[:3]
            floor=min(float(e['applied_root_input'][1,3])for e in video.actors)
            for sid,category in eligible.items():
                yy,xx=np.where(sem==sid)
                for j in np.linspace(0,len(xx)-1,min(80,len(xx)),dtype=int)if len(xx)else []:
                    x,y=int(xx[j]),int(yy[j])
                    if x<2 or y<2 or x>=video.width-2 or y>=video.height-2 or not 0<float(dep[y,x])<4:continue
                    wp=world(x,y);normal=np.cross(world(x+2,y)-wp,world(x,y+2)-wp)
                    if .25<wp[1]-floor<.65 and abs(normal[1])>.85*np.linalg.norm(normal):samples.setdefault(sid,[]).append(wp)
        for sid,values in samples.items():
            points=np.asarray(values)
            if len(points)>=5 and np.ptp(points[:,1])<.06:result['seats'].append({'kind':'existing','semantic_id':sid,'category':eligible[sid],'center_m':np.median(points,axis=0).tolist(),'surface_samples_m':points.tolist()})
    write_json(directory/'existing_native_seat_survey.json',result);return result


def compile_seated(video,entry,scene,house,inputs,directory,request):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not video.pathfinder.is_loaded:raise ValueError('explicit read-only admitted native NavMesh was not loaded')
    d,b,action,lineage=admitted_master(entry,inputs,house,kind='sit')
    roster=json.loads(Path(inputs['v7_asset_roster']).read_text());root=Path(roster['artifact_root'])
    survey=existing_native_seats(video,scene,directory)
    original_navmesh=directory/'original_private_navmesh.navmesh'
    video.pathfinder.save_nav_mesh(str(original_navmesh))
    if not original_navmesh.is_file() or original_navmesh.stat().st_size==0:raise ValueError('cannot snapshot private seating navigation')
    major=entry['declaration']['body_identity_id'];reference=json.loads((root/'round2/assets/humans'/major/'seated_contact/hm3d/seated.json').read_text())
    compat=next(r for r in json.loads((root/'round3/evidence/chair_body_compatibility.json').read_text())if r['asset_id']==major)
    styles={r['chair_id']:r for r in compat['styles']if r['status']=='pass'}
    # A native verified existing seat must be tried before adding furniture.
    candidates=[]
    cx,_,cz=reference['chair_boxes'][0]['center_m']
    for seat in ([] if request.get('chair_id') else survey['seats']):
        p=np.asarray(seat['center_m']);candidates.append((seat,[p[0],p[1]-reference['seat_height_m'],p[2]],[-cx,0,-cz]))
    native_points=[np.asarray(p)-[0,1.3,0]for p in scene['native_placement_candidates']]
    if inputs.get('phase2') and request.get('minimum_camera_distance_m'):
        native_points=[p for p in native_points if np.linalg.norm((p-np.asarray(scene['listener_m']))[[0,2]])>=float(request['minimum_camera_distance_m'])]
    floor_proposals=request.get('native_chair_floor_points_m',[])
    offset_proposals=request.get('chair_horizontal_offset_proposals_m',[[0.,0.,0.]])
    if floor_proposals and not inputs.get('phase2'):raise ValueError('explicit chair proposals require phase two')
    if not offset_proposals:raise ValueError('empty seated physical offset proposals')
    for value in floor_proposals:
        if np.asarray(value).shape!=(3,) or not np.isfinite(value).all():raise ValueError('invalid physical chair floor proposal')
    for value in offset_proposals:
        vector=np.asarray(value,float)
        if vector.shape!=(3,) or not np.isfinite(vector).all() or vector[1]!=0. or np.linalg.norm(vector)>.100001:
            raise ValueError('chair horizontal proposal exceeds unchanged physical model scope')
    for point in [*floor_proposals,entry['applied_root_input'][:3,3].tolist(),*native_points[:96]]:
        for chair in roster['chairs']:
            if chair['asset_id']in styles and chair.get('habitat_status')=='pass_reference_placements' and (not request.get('chair_id') or chair['asset_id']==request['chair_id']):
                for delta in offset_proposals:
                    offset=(np.asarray(styles[chair['asset_id']]['human_local_offset_m'])+np.asarray(delta)).tolist()
                    candidates.append(({'kind':'fallback','chair':chair},point,offset))
    accepted=None;failures=[]
    all_ids={int(a['object'].object_id)for a in video.actors}
    protected=[np.asarray(scene['listener_m'])-np.array([0,1.3,0]),*[a['applied_root_input'][:3,3]for a in video.actors if a is not entry and a.get('phase2_layout_qualified') is not False]]
    search_cpu_started=time.process_time()
    for seat,point,offset in candidates:
        if inputs.get('phase2') and time.process_time()-search_cpu_started>1800:
            failures.append({'reason':'bounded seating search CPU budget exhausted (1800s); no gate relaxed'});break
        cr=np.asarray(point,float);offset=np.asarray(offset,float)
        # The seated mouth can be visible while a close chair is below the
        # original camera frustum. This opt-in candidate constraint selects
        # farther physical seats; final actual 100-pixel gate stays unchanged.
        if inputs.get('phase2') and np.linalg.norm((cr-np.asarray(scene['listener_m']))[[0,2]])<float(request.get('minimum_camera_distance_m',0.)):
            continue
        for yaw in [math.radians(scene['sources'][entry['endpoint_id']]['body_yaw_deg']),0.,math.pi/2,math.pi,-math.pi/2]:
            rot=yaw_matrix(yaw);hr=cr+rot@offset
            if any(np.linalg.norm((hr-p)[[0,2]])<.9 for p in protected[1:]):continue
            chair_obj=None;surface=None;scale=None;gap=None;own_hits=[]
            try:
                hits=[h for h in video.sim.cast_ray(video.hs.geo.Ray(video.mn.Vector3(hr+[0,.2,0]),video.mn.Vector3(0,-1,0)),.85).hits if int(h.object_id)not in all_ids]
                if not hits:raise ValueError('seated body has no actual physical floor')
                ground=float(hits[0].point[1]);floor=PhysicalFloor(video,ground,all_ids)
                chair_grounding=None
                if seat['kind']=='fallback' and inputs.get('phase2'):
                    # The scan floor varies under the different chair legs.
                    # A room-level Y is only a proposal, never the chair floor.
                    # Ground the original full registered bottom footprint first;
                    # still reject every native support gap >12mm and any full
                    # original-skin penetration beyond the fixed native margin.
                    from avengine.capture.v7_visible_mesh import indexed_static_mesh
                    registered=seat['chair']
                    original_chair=indexed_static_mesh(registered['habitat_glb'])
                    vertices=np.asarray(original_chair.vertices)
                    bottom=vertices[vertices[:,1]<=vertices[:,1].min()+.00001]
                    if len(bottom)<4:raise ValueError('registered chair has no complete bottom footprint')
                    candidate=bottom@rot.T+cr
                    levels=floor.many(candidate)
                    old_y=float(cr[1])
                    cr=cr.copy();cr[1]=float(np.max(levels-bottom[:,1]))+.0005
                    hr=cr+rot@offset
                    chair_grounding={'method':'rigid chair root fitted to highest exact original scan floor under every original bottom vertex; final native rays and complete mesh checked separately',
                        'original_bottom_vertices':len(bottom),'proposed_root_y_m':old_y,'grounded_root_y_m':float(cr[1]),'rigid_shift_y_m':float(cr[1]-old_y),
                        'floor_min_y_m':float(levels.min()),'floor_max_y_m':float(levels.max()),'precision_headroom_m':.0005,'thresholds_changed':False}
                solver=FootPlant(d,b,action,[0.,1/30],[hr.tolist()]*2,[yaw]*2,0.,0.,1/30,floor)
                for values in solver.active.values():values[:]=True
                _,parts,ev=solver.frame(0);v,f=triangle_mesh(d,b,parts);tri=(v+offset)[f]
                if solver.max_reach>=.001 or max(abs(e['floor_delta_m'])for e in ev['feet'].values())>=.008:raise ValueError('seated foot contact or reach failed unchanged 8mm/1mm gate')
                clearance=TriangleClearance(video,house,ground,entry)
                # Existing support is not stripped from the exact stage model.
                # If it cannot satisfy the conservative new-house collision
                # gate it remains rejected, then a separately checked chair.
                minimum,world=clearance.check(v,f,hr,yaw)
                folder=directory/'frames';folder.mkdir(parents=True,exist_ok=True)
                cd,cb=None,None;chair_path=None;scale=None;legs=[]
                if seat['kind']=='fallback':
                    chair=seat['chair'];surface=clipped_surface_minimum(tri,[0,0],[chair['seat_width_m'],chair['seat_depth_m']],.25)
                    if surface is None:raise ValueError('actual posed clothing has no seat footprint contact')
                    height=float(surface[1]);scale=(height-(.001 if inputs.get('phase2') else 0.))/chair['seat_height_m']
                    if not .7<scale<1.3:raise ValueError('chair seat fit outside admitted ergonomic scaling')
                    for box in chair['solid_boxes']:
                        c=np.array(box['center_m']);size=np.array(box['size_m']);c[1]*=scale;size[1]*=scale
                        if len(intersections(tri,c,size,box.get('yaw_rad',0),.00101 if inputs.get('phase2') else .004)):raise ValueError('actual seated skin intersects chair inner solid triangles')
                    cd,cb=read_glb(Path(chair['habitat_glb']));cd=copy.deepcopy(cd)
                    for node in cd['nodes']:
                        if 'mesh'in node:node['scale']=[1,scale,1]
                    chair_path=directory/'verified_chair.glb';write_glb(chair_path,cd,cb)
                    from avengine.assets.source_materials import prepare_source_glb
                    derived,chair_material=prepare_source_glb(chair_path,asset_id=chair['asset_id'],entity_class='furniture',output_directory=video.source_material_cache_directory)
                    chair_material={**chair_material,
                        'legacy_factor_palette_mislabeled_srgb':chair_material['source_palette_srgb'],
                        'source_palette_srgb':original_factor_palette_srgb(cd).tolist(),
                        'reference_color_revision':'original-gltf-linear-factor-to-srgb-v1',
                        'reference_source':'https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html',
                        'measurement_function_changed':False,'thresholds_changed':False}

                    from avengine.capture.v7_registered_native import loaded_mesh
                    cm=loaded_mesh(chair_path);cv=np.asarray(cm.vertices)@rot.T+cr
                    clearance.check_world(cv,cm.faces)
                    for p in chair['support_points_m']:
                        wp=cr+rot@np.asarray(p);legs.append(float(wp[1]-floor(wp)))
                    if any(abs(x)>.012 for x in legs):raise ValueError('new chair support does not meet actual 12mm native floor gate')
                    tm=video.sim.get_object_template_manager();attr=tm.create_new_template('verified_seat_'+entry['endpoint_id'])
                    attr.render_asset_handle=str(derived);attr.collision_asset_handle=str(derived);attr.is_collidable=True;attr.compute_COM_from_shape=False;attr.shader_type='pbr';attr.margin=.001
                    tid=tm.register_template(attr);chair_obj=video.sim.get_rigid_object_manager().add_object_by_template_id(tid)
                    from avengine.assets.habitat_capture import _apply_root_with_habitat
                    matrix=np.eye(4);matrix[:3,:3]=rot;matrix[:3,3]=cr
                    _apply_root_with_habitat(chair_obj,matrix,qt=video.qt,mn=video.mn);chair_obj.motion_type=video.hs.physics.MotionType.STATIC;chair_obj.semantic_id=6100000+int(entry['declaration']['native_instance_id'])
                    wp=cr+rot@surface
                    hits=video.sim.cast_ray(video.hs.geo.Ray(video.mn.Vector3(wp+[0,.15,0]),video.mn.Vector3(0,-1,0)),.4).hits
                    own_hits=[hit for hit in hits if int(hit.object_id)==int(chair_obj.object_id)]
                    if not own_hits:raise ValueError('no actual native registered seat surface at clothing contact')
                    gap=float(wp[1]-own_hits[0].point[1])
                    if not -.00101<=gap<=.012:raise ValueError('actual native clothing/seat gap fails unchanged support gate')
                else:
                    surface=clipped_surface_minimum(tri,[0,0],[.4,.28],.25)
                    if surface is None:raise ValueError('existing seat actual clothing contact absent')
                    wp=cr+rot@surface;hits=video.sim.cast_ray(video.hs.geo.Ray(video.mn.Vector3(wp+[0,.2,0]),video.mn.Vector3(0,-1,0)),.5).hits
                    hits=[h for h in hits if int(h.object_id)not in all_ids]
                    gap=float(wp[1]-hits[0].point[1])if hits else math.inf
                    if abs(gap)>.012:raise ValueError('existing native seat gap exceeds 12mm')
                if chair_obj is not None and inputs.get('phase2') and request.get('require_observed_chair_candidate'):
                    # Reject hidden candidates before committing a physical seat.
                    # The final native 100px chair/person and material gates still run.
                    chair_view={'semantic_id':int(chair_obj.semantic_id),
                        'declaration':{'asset_id':seat['chair']['asset_id']}}
                    visible,chair_views=video._native_observed_pixels(scene,[chair_view])
                    if visible[seat['chair']['asset_id']]<100:
                        raise ValueError('fallback chair candidate below unchanged native 100px visibility')
                # A private navigation rebuild only: preserve every protected
                # endpoint and its originally available connection.
                nav=video.pathfinder;before=[np.asarray(nav.snap_point(p))for p in protected]
                ns=video.hs.NavMeshSettings();ns.set_defaults()
                preserved_nav_settings={}
                if inputs.get('phase2') and hasattr(nav,'nav_mesh_settings'):
                    previous=nav.nav_mesh_settings
                    for field in dir(previous):
                        if field.startswith('_'):continue
                        value=getattr(previous,field)
                        if isinstance(value,(bool,int,float)):
                            setattr(ns,field,value)
                            if getattr(ns,field)!=value:raise ValueError('private navigation setting copy changed '+field)
                            preserved_nav_settings[field]=value
                ns.include_static_objects=True
                if not video.sim.recompute_navmesh(nav,ns):raise ValueError('verified seat private navigation rebuild failed')
                after=[np.asarray(nav.snap_point(p))for p in protected];ne=max(float(np.linalg.norm((a-b)[[0,2]]))for a,b in zip(before,after))
                if not math.isfinite(ne)or ne>.002:raise ValueError('seat displaced a protected navigation endpoint beyond 2mm')
                for p in after[1:]:
                    path=video.hs.ShortestPath();path.requested_start=after[0];path.requested_end=p
                    if not nav.find_path(path):raise ValueError('seat blocked a required native route')
                ed=externalize_unchanged_images(d,b,directory/'textures');body=folder/'seated.glb';rigid_glb(ed,b,parts,body)
                ev.update(moving=False,action_id='sit',world_root_m=hr.tolist(),yaw_rad=yaw)
                frames=[{**ev,'time_s':i/30}for i in range(901)]
                record={'status':'pass','revision':'native-seat-first-exact-contact-v1','lineage':lineage,'motion_kind':'sit',
                    'frame_count':901,'sample_rate_hz':30,'source_frames':frames,'world_roots_m':[hr.tolist()]*901,
                    'yaw_rad':yaw,'pose_glbs':[str(body)]*901,'physical_ground_y_m':ground,
                    'full_skin_bounds_m':[world.min(0).tolist(),world.max(0).tolist()],
                    'known_bad_10cm_floor_detected':min(e['floor_delta_m']-.1 for e in ev['feet'].values())<-.008,
                    'seat_source':seat['kind'],'seat_record':seat,'human_local_offset_m':offset.tolist(),'chair_world_root_m':cr.tolist(),'chair_grounding':chair_grounding,'chair_glb':str(chair_path)if chair_path else None,
                    'chair_scale_y':scale,'native_support_deltas_m':legs,'seat_contact_gap_m':gap,
                    'private_navigation_original_settings':preserved_nav_settings,'original_full_skin_floor_collision':clearance.floor_certificate(),'maximum_navmesh_horizontal_error_m':ne,'minimum_certified_triangle_nonfloor_clearance_m':minimum,
                    'existing_seat_survey':survey,'thresholds_changed':False,'chair_produces_sound':False,
                    'anatomical_buttock_surface':'not independently segmented; lowest original clothing triangles in seat footprint as upstream'}
                accepted=record;break
            except ValueError as error:
                failures.append({'seat_kind':seat['kind'],'start_m':cr.tolist(),'reason':str(error),'human_local_offset_m':offset.tolist(),'yaw_rad':yaw,'proposed_clothing_contact_chair_local_m':surface.tolist()if surface is not None else None,'proposed_chair_scale_y':scale,'native_contact_gap_m':gap,'native_own_seat_hit_points_m':[list(h.point)for h in own_hits[:3]]})
                if chair_obj is not None:video.sim.get_rigid_object_manager().remove_object_by_id(chair_obj.object_id)
                video.pathfinder.load_nav_mesh(str(original_navmesh))
        if accepted:break
    if not accepted:
        write_json(directory/'seat_rejected.json',{'status':'blocked','failures':failures,'existing_seat_survey':survey});raise ValueError('no existing or fallback chair passed native seated episode gates')
    if chair_obj is not None:
        from avengine.capture.v7_registered_native import RigidStandingInstance
        item={'object':RigidStandingInstance(chair_obj),'native_actor_kind':'registered_rigid','endpoint_id':'silent_seat_'+entry['endpoint_id'],
            'semantic_id':int(chair_obj.semantic_id),'declaration':{'asset_id':seat['chair']['asset_id']},'source_material':chair_material}
        video.seat_material_entries=getattr(video,'seat_material_entries',[])+[item]
        video.receipt['source_materials'].append(chair_material)
    if accepted.get('seat_source')=='fallback':write_json(directory/'chair_material_reference.json',chair_material)
    accepted['chair_inner_solid_tolerance_m']=.00101 if inputs.get('phase2') else .004
    accepted['failed_candidate_count']=len(failures);write_json(directory/'seated_geometry.json',accepted);return accepted
