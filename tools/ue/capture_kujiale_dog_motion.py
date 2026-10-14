"""Native skeletal motion and per-paw metric-depth admission in Kujiale."""
from __future__ import annotations
import json,math,traceback
from pathlib import Path
import numpy as np
from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,rotation
from avengine.qa.kujiale_dog_motion import calibrate
from avengine.capture.kujiale_spear_runtime import apply_camera,to_ue,from_ue
from avengine.backends.spear_ue.research_runtime import attach_skeletal_emitter_component,read_scene_component_pose
from capture_kujiale_admission import write,vector
from avengine.qa.kujiale_native_depth import world_pixels,REVISION
from tools.rooms.check_v7_kujiale_native_assets import vector_field

SIDES=['FL','FR','HL','HR']

def spawn(c,row,native):
    actor=c.game.unreal_service.spawn_actor(uclass='ASkeletalMeshActor',spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'})
    actor.SetActorEnableCollision(bNewActorEnableCollision=False)
    component=c.game.unreal_service.get_component_by_class(actor=actor,uclass='USkeletalMeshComponent')
    component.SetMobility(NewMobility='Movable')
    mesh=c.game.unreal_service.load_object(uclass='USkeletalMesh',name=native['skeletal_mesh'])
    component.SetSkeletalMeshAsset(NewMesh=mesh)
    component.SetCollisionEnabled(NewType='NoCollision')
    component.SetCastShadow(NewCastShadow=True)
    component.set_property_value(property_name='VisibilityBasedAnimTickOption',property_value='AlwaysTickPoseAndRefreshBones')
    component.SetComponentTickEnabled(bEnabled=True)
    animation=c.game.unreal_service.load_object(uclass='UAnimSequence',name=native['animation'])
    component.PlayAnimation(NewAnimToPlay=animation,bLooping=False)
    component.SetPlayRate(Rate=0.)
    component.Stop()
    emitter=attach_skeletal_emitter_component(c.game,actor_id=row['asset_id'],skeletal_component=component,attachment={'attachment_type':'bone','name':native['selected_bones']['mouth'],'local_offset_cm':[0.,0.,0.]})
    c.active=[{'visual_actor':actor,'anchor':actor,'component':component,'emitter_component':emitter}]
    return actor,component,emitter

def read_bones(component,names):
    return {key:from_ue(vector_field(component.GetSocketLocation(InSocketName=name,as_dict=True),'ReturnValue'))
            for key,name in names.items()}

def bounds(component):
    raw=component.GetLocalBounds(as_dict=True)
    return [from_ue(vector(raw,k)) for k in ['Min','Max']]

def check_route(g,plan,authored,offset=0.):
    # Admission plays every authored pose and yaw in place. Spatial production
    # trajectories still require their own swept-room gate; no new scenes here.
    for placement in plan['placements']:
        root=np.asarray(placement['root_m']).copy()
        root[1]=float(placement['support']['point_m'][1])+offset
        base=float(placement['yaw_rad'])-float(authored['yaw_rad'][0])
        failures=[]
        for index,frame in enumerate(authored['frames']):
            lo,hi=np.asarray(frame['bounds_min_m']),np.asarray(frame['bounds_max_m'])
            corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            yaw=base+float(authored['yaw_rad'][index])
            result=g.collision(corners,root,yaw,[0,1,0],.012)
            if result['status']!='pass':
                failures.append({'frame':index,**result});break
        if not failures:return root,base,placement
    raise RuntimeError('no collision-free native skeletal admission placement for all authored keys')

def foot_depth(c,foot_camera,foot_components,actor,bones,g,row,dest,index):
    rows=[];patches={}
    for side in SIDES:
        paw=np.asarray(bones[side])
        hit=g.ray(paw+[0,.05,0],[0,-1,0],.60)
        if hit is None or '/Meshes/floor/' not in hit['material_path']:
            rows.append({'paw':side,'status':'fail','reason':'paw is not above room floor','support':hit});continue
        floor_y=float(hit['point_m'][1]);eye=paw.copy();eye[1]=floor_y-.22
        view={'position_m':eye.tolist(),'yaw_deg':0.,'pitch_deg':90.,'hfov_deg':85.}
        def setup():
            apply_camera(foot_camera,view)
            foot_components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList')
            foot_components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[actor])
        c.frame(setup);c.frame()
        with c.instance.begin_frame():pass
        with c.instance.end_frame():d=c.depth(foot_components['depth'])
        h,w=d.shape;yy,xx=np.indices(d.shape);f=w/(2*math.tan(math.radians(85)/2));D=d/np.sqrt(1+((yy+.5-h/2)/f)**2+((xx+.5-w/2)/f)**2)
        x=eye[0]+D*(yy+.5-h/2)/f;z=eye[2]+D*(xx+.5-w/2)/f;y=eye[1]+D
        dist=np.stack([(x-bones[s][0])**2+(z-bones[s][2])**2 for s in SIDES])
        own=np.argmin(dist,axis=0)==SIDES.index(side)
        mask=np.isfinite(d)&(d>.05)&(d<.90)&own&(dist[SIDES.index(side)]<.09**2)
        if not mask.any():
            rows.append({'paw':side,'status':'fail','reason':'native rendered sole missing in the declared paw region','support':hit});continue
        idx=np.flatnonzero(mask)[np.argmin(y[mask])];py,px=np.unravel_index(idx,d.shape)
        # Swing paws can rise beyond 12 cm: trace to the real first surface.
        # Increasing query range preserves the 12 mm support/penetration gate.
        sole=np.array([x[py,px],y[py,px],z[py,px]]);sole_hit=g.ray(sole+[0,.10,0],[0,-1,0],.75)
        gap=None if sole_hit is None else float(sole[1]-sole_hit['point_m'][1])
        floor_ok=sole_hit is not None and '/Meshes/floor/' in sole_hit['material_path']
        expected=row['feet'][side];contact=bool(expected['in_contact'])
        # A swinging paw can be above the floor. A support paw must touch it;
        # every paw must avoid penetration, and one sole must support the body.
        okay=floor_ok and gap>=-.012 and (not contact or abs(gap)<=.012)
        rows.append({'paw':side,'status':'pass' if okay else 'fail','native_sole_world_m':sole.tolist(),'floor_gap_m':gap,'expected_in_contact':contact,'registered_min_world_y_delta_m':expected['floor_delta_m'],'support':sole_hit,'tolerance_m':.012,'native_valid_sole_pixels':int(mask.sum()),'minimum_depth_pixel':[int(px),int(py)]})
        patches[side]=d[max(0,py-40):min(h,py+41),max(0,px-40):min(w,px+41)].astype('float16')
    if patches:np.savez_compressed(dest/f'frame_{index:04d}_native_sole_depth.npz',**patches)
    gaps=[x['floor_gap_m'] for x in rows if x.get('floor_gap_m') is not None]
    support=len(gaps)==4 and min(abs(v) for v in gaps)<=.012
    return {'status':'pass' if support and len(rows)==4 and all(x['status']=='pass' for x in rows) else 'fail',
            'four_paws':rows,'at_least_one_native_support_sole':support,
            'floor_ray_range_m':.75,'method':'four native UE isolated metric-depth views from below, nonoverlapping 9 cm paw regions, actual sole-to-native-room-floor first-hit rays; 12 mm contact/penetration gate'}

def run(c):
    g=NativeSupportGeometry(c.ue['native_floor_surface']);g.floor.validate_probes(c.house['checked_rooms'][0]['floor_reference'])
    rows=[x for x in c.manifest['assets'] if x['walk_allowed']]
    if c.args.only:rows=[x for x in rows if x['asset_id'] in c.args.only]
    results=[]
    foot_camera=foot_components=None
    try:
        with c.instance.begin_frame():
            foot_camera,foot_components=c.spawn(c.game,horizontal_fov_deg=85,width=512,height=512)
            for name in ['rgb','object_ids']:foot_components[name].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
        with c.instance.end_frame():pass
        for row in rows:
            for family in ['Idle','Walking']:
                ev=c.root/'evidence/native_motion'/c.args.house_id/(row['asset_id']+'_'+family+'.json')
                if ev.exists() and json.loads(ev.read_text()).get('status')=='pass_native_motion' and json.loads(ev.read_text()).get('native_skin_floor_calibration')==json.loads((c.root/'inputs/native_skin_floor_calibration.json').read_text()).get(row['body_id'],{'fixed_root_y_offset_m':0.}):results.append(json.loads(ev.read_text()));continue
                rec={'asset_id':row['asset_id'],'family':family,'house':c.args.house_id,'measurement_revision':REVISION,'status':'fail','frames':[],'launch_receipt':str(c.root/'evidence'/f'native_launch_{c.revision}_{c.run_label}.json')}
                try:
                    n=json.loads((c.root/'inputs/native_motion_editor_snapshot'/(row['asset_id']+'_'+family+'.json')).read_text())
                    authored=json.loads(Path(n['authored_contact_plan']).read_text())
                    calibration=calibrate(n,authored)
                    if calibration['status']!='pass':raise RuntimeError('imported native pose differs from authored dog rig')
                    # Calibration is below 0.2 mm with identity axes in this import.
                    if np.max(np.abs(np.asarray(calibration['rotation_ue'])-np.eye(3)))>1e-4 or abs(calibration['uniform_scale']-1)>1e-4:raise RuntimeError('nonidentity dog import requires an explicitly applied fixed calibration')
                    plan=json.loads((c.root/'evidence/plans/final'/c.args.house_id/(row['asset_id']+'.json')).read_text())
                    floor_calibration=json.loads((c.root/"inputs/native_skin_floor_calibration.json").read_text()).get(row["body_id"],{"fixed_root_y_offset_m":0.})
                    root,base,placement=check_route(g,plan,authored,floor_calibration["fixed_root_y_offset_m"])
                    with c.instance.begin_frame():
                        actor,component,emitter=spawn(c,row,n)
                        c.components['rgb'].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
                        c.components['depth'].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
                    with c.instance.end_frame():pass
                    dest=c.root/'renders/native_motion'/c.args.house_id/row['asset_id']/family;dest.mkdir(parents=True,exist_ok=True)
                    for index,key in enumerate(authored['frames']):
                        yaw=base+authored['yaw_rad'][index]
                        def update():
                            actor.K2_SetActorLocationAndRotation(NewLocation=dict(zip(['X','Y','Z'],to_ue(root))),NewRotation={'Roll':0.,'Pitch':0.,'Yaw':-math.degrees(yaw)},bSweep=False,bTeleport=True)
                            component.SetPosition(InPos=index/30.,bFireNotifies=False)
                        c.frame(update);c.frame()
                        with c.instance.begin_frame():
                            bone=read_bones(component,n['selected_bones'])
                            pose=read_scene_component_pose(emitter)
                            animation_time=float(component.GetPosition())
                        with c.instance.end_frame():pass
                        expected={k:(root+rotation(yaw)@np.asarray(v)[[0,2,1]]/100.).tolist() for k,v in n['frames'][index]['native_component_bones_cm'].items()}
                        bone_error=max(float(np.linalg.norm(np.asarray(bone[k])-expected[k])) for k in expected)
                        emitter_error=float(np.linalg.norm(np.asarray(from_ue(pose['location_cm']))-bone['mouth']))
                        spine=np.asarray(bone['neck'])-bone['pelvis'];horizontal=float(np.linalg.norm(spine[[0,2]]));tilt=math.degrees(math.atan2(abs(spine[1]),horizontal))
                        soles=foot_depth(c,foot_camera,foot_components,actor,bone,g,key,dest,index)
                        valid=bone_error<=.003 and emitter_error<=.0005 and abs(animation_time-index/30.)<=.002 and tilt<=45 and soles['status']=='pass'
                        fr={'frame':index,'time_s':index/30.,'status':'pass' if valid else 'fail','native_bones_world_m':bone,'bone_readback_max_error_m':bone_error,'mouth_emitter_error_m':emitter_error,'native_animation_time_s':animation_time,'body_axis_tilt_deg':tilt,'body_axis_tilt_max_deg':45,'sole_depth':soles}
                        rec['frames'].append(fr)
                        if index%30==0 or index==len(authored['frames'])-1:
                            with c.instance.begin_frame():
                                c.components['rgb'].set_property_value(property_name='bCaptureEveryFrame',property_value=True)
                                apply_camera(c.camera,placement['camera_candidates'][0])
                            with c.instance.end_frame():pass
                            for _ in range(24):c.frame()
                            with c.instance.begin_frame():pass
                            with c.instance.end_frame():rgb=c.rgb(c.components['rgb'])
                            import cv2
                            path=dest/f'frame_{index:04d}.jpg';cv2.imwrite(str(path),cv2.resize(rgb,(1280,720)),[cv2.IMWRITE_JPEG_QUALITY,90]);fr['rgb']=str(path)
                            c.frame(lambda:c.components['rgb'].set_property_value(property_name='bCaptureEveryFrame',property_value=False))
                        if index%20==0:write(ev,rec);print('MOTION_FRAME',row['asset_id'],family,index,fr['status'],bone_error,soles['status'],flush=True)
                    passed=all(x['status']=='pass' for x in rec['frames']) and len(rec['frames'])==len(authored['frames'])
                    rec.update(status='pass_native_motion' if passed else 'fail',frame_count=len(rec['frames']),failed_frames=[x['frame'] for x in rec['frames'] if x['status']!='pass'],rig_calibration=calibration,skeletal_mesh=n['skeletal_mesh'],animation=n['animation'],root_m=root.tolist(),native_skin_floor_calibration=floor_calibration,route_mode='in_place_animation_admission',scope='Every authored Idle/Walking key, actual native skeleton and rendered paw soles. Production spatial walk trajectories are not generated or admitted by this test.')
                except Exception:
                    rec['error']=traceback.format_exc();write(ev,rec)
                if c.active:c.cleanup()
                c.frame(lambda:[c.components[x].set_property_value(property_name='bCaptureEveryFrame',property_value=True) for x in ['rgb','depth']])
                write(ev,rec);results.append(rec);write(c.root/'evidence'/f'native_motion_{c.run_label}.json',results)
    finally:
        if foot_camera is not None:
            from avengine.backends.spear_ue.research_runtime import close_scene_capture
            for name in ("object_ids","depth","rgb"):
                close_scene_capture(instance=c.instance,game=c.game,camera=None,capture=foot_components[name])
            with c.instance.begin_frame():foot_camera.K2_DestroyActor()
            with c.instance.end_frame():pass
    return results
