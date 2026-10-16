"""Machine qualification in native MP3D and HM3D, with actual RGB body masks."""
import argparse,json,math,sys,traceback,hashlib
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from avengine.assets.source_materials import configure_source_pbr,measure_source_appearance,source_palette,read_glb,prepare_texture_reference_glb
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
from v7_human_roster_habitat import camera,project
from v7_animal_variants_color import measure
from v7_pet_birds_hy3d_habitat import make_object,yaw_matrix

def dump(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def floor_at(sim,runtime,point,ignore=()):
    point=np.asarray(point);hits=sim.cast_ray(runtime.habitat_sim.geo.Ray(runtime.magnum.Vector3(point+[0,.15,0]),runtime.magnum.Vector3(0,-1,0)),.6).hits;hits=[h for h in hits if h.object_id not in ignore];return None if not hits else float(hits[0].point[1])
def feature_exclusion(runtime,sim,uuid,meta,mask):
    h=meta['dimensions_m'][1]-(meta.get('perch') or {}).get('perch_top_m',0);head=np.array(meta['head_center_local_m']);mouth=np.array(meta['emit_point_local_m']);root=np.array(meta['_root']);marks=[mouth,head+[h*.05,h*.02,0],head+[-h*.05,h*.02,0]];yy,xx=np.indices(mask.shape);out=np.zeros(mask.shape,bool)
    for p in marks:
        pixel=project(runtime,sim,uuid,root+p);edge=project(runtime,sim,uuid,root+p+[h*.025,0,0]);rad=max(2,min(8,float(np.linalg.norm(np.array(pixel)-edge))));out|=(xx-pixel[0])**2+(yy-pixel[1])**2<=rad**2
    return out

def qualify(sim,r,uuids,T,meta,house,placements,shader,contact_threshold_m=None):
    D=T/'assets'/meta['asset_id'];height=meta['dimensions_m'][1];marks=meta['landmarks_local_m'];supports=[np.array(p) for k,p in marks.items() if k.startswith('foot_')]
    if meta.get('perch'):supports=[np.array([x,0,z]) for x,z in [(0,0),(-.055,0),(.055,0),(0,-.055),(0,.055)]]
    sid=7400100;obj=None;chosen=None;attempts=[]
    for candidate in [x for x in placements if x['house']==house and x['status']=='pass_candidate'][:8]:
        root=np.array(candidate['root_m'],float);fy=floor_at(sim,r,root)
        if fy is None:continue
        root[1]=fy;fs=[floor_at(sim,r,root+p) for p in supports]
        if any(y is None for y in fs):continue
        deltas=[float(root[1]+p[1]-y) for p,y in zip(supports,fs)]
        obj=make_object(sim,r,Path(meta['runtime_glb']),root,0,sid);obj.override_collision_group(r.habitat_sim.physics.CollisionGroups.Dynamic);obj.translation=r.magnum.Vector3(root+[0,.06,0]);raised=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root-[0,.03,0]);buried=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root);at_floor=bool(obj.contact_test());rec={'root_m':root.tolist(),'support_deltas_m':deltas,'buried_3cm':buried,'raised_6cm':raised};attempts.append(rec)
        if max(map(abs,deltas)) <= (contact_threshold_m if contact_threshold_m is not None else (.005 if meta.get('perch') else .015)) and buried and not raised:
            # A placement must expose a complete silhouette at four camera families.
            target=root+[0,((meta.get('perch') or {}).get('perch_top_m',0)+height)/2 if meta.get('perch') else height*.58,0];distance=max(.45,max(meta['dimensions_m'])*2.4);visible_views=0
            for angle in [0,-math.pi/4,math.pi/4,-math.pi/2]:
                visible=False
                for shift in [0,.12,-.12,.25,-.25]:
                    if visible:break
                    for mult in [.65,.8,1.,.5,1.2,1.5]:
                        if visible:break
                        for elevation in [.70,.95,1.15]:
                            eye=root+np.array([math.sin(angle+shift)*distance*mult,((meta.get('perch') or {}).get('perch_top_m',0)+(height-(meta.get('perch') or {}).get('perch_top_m',0))*elevation+.03) if meta.get('perch') else height*elevation,-math.cos(angle+shift)*distance*mult]);sim.initialize_agent(0,camera(r,eye,target));mask=np.asarray(sim.get_sensor_observations()[uuids['semantic']])==sid;yy,xx=np.where(mask)
                            if len(xx)>(600 if meta.get('perch') else 1000) and min(xx.min(),yy.min(),639-xx.max(),639-yy.max())>3:visible=True;break
                visible_views+=int(visible)
            rec['full_silhouette_preflight_views']=visible_views
            if visible_views==4:chosen=(root,deltas,raised,buried,at_floor);break
        sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);obj=None
    if chosen is None:raise RuntimeError('native floor/collision placement gate:'+json.dumps(attempts))
    root,deltas,raised,buried,at_floor=chosen;child=obj.root_scene_node.create_child();local=np.array(meta['emit_point_local_m']);child.translation=r.magnum.Vector3(local);native=np.array(child.absolute_translation);error=float(np.linalg.norm(native-root-local));delta=np.array([.1,.02,-.07]);yaw=.35;obj.translation=r.magnum.Vector3(root+delta);obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(yaw),r.magnum.Vector3.y_axis());follow=float(np.linalg.norm(np.array(child.absolute_translation)-(root+delta+yaw_matrix(yaw)@local)));obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(0),r.magnum.Vector3.y_axis());obj.translation=r.magnum.Vector3(root);floor_delta=float(obj.aabb.min[1]);native_height=float(obj.aabb.max[1]-obj.aabb.min[1]);sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
    body_path=Path(meta['body_only_glb']);reference_path=D/'appearance_reference'/(hashlib.sha256(body_path.read_bytes()).hexdigest()[:16]+'_unlit_minimal_v2.glb');reference_path.parent.mkdir(exist_ok=True)
    if not reference_path.exists():
        prepare_texture_reference_glb(body_path,reference_path)
        # Unlit diagnostics require no layered specular/normal extensions.
        # Magnum aborts on Rocketbox KHR_materials_specular + unlit layers.
        from v7_human_roster_materials import write_glb
        rd,rb=read_glb(reference_path)
        for material in rd.get('materials',[]):
            material['extensions']={'KHR_materials_unlit':{}}
            material.pop('normalTexture',None);material.pop('occlusionTexture',None)
        rd['extensionsUsed']=['KHR_materials_unlit']
        rd.pop('extensionsRequired',None)
        write_glb(reference_path,rd,rb)
    body=make_object(sim,r,body_path,root,0,sid);stand=None
    if meta.get('perch'):
        stand_path=D/'perch_only.glb'
        if not stand_path.exists():
            scene=trimesh.load(meta['runtime_glb'],force='scene',process=False);out=trimesh.Scene()
            for name,g in scene.geometry.items():
                if name.startswith('Perch'):out.add_geometry(g,geom_name=name)
            if not out.geometry:raise RuntimeError('perch body semantic split unavailable')
            out.export(stand_path)
        stand=make_object(sim,r,stand_path,root,0,sid+1)
    meta['_root']=root.tolist();doc,blob=read_glb(body_path);palette=source_palette(doc,blob,body_path);cal=json.loads((T/'evidence/color_calibration.json').read_text());assert cal['status']=='pass';images=[];pooled_pixels=[];target=root+[0,((meta.get('perch') or {}).get('perch_top_m',0)+height)/2 if meta.get('perch') else height*.58,0];distance=max(.45,max(meta['dimensions_m'])*2.4)
    for name,angle in [('front',0),('quarter_left',-math.pi/4),('quarter_right',math.pi/4),('side',-math.pi/2)]:
        accepted=None
        for shift in [0,.12,-.12,.25,-.25]:
            if accepted:break
            for mult in [.65,.8,1.,.5,1.2,1.5]:
                if accepted:break
                for elevation in [.70,.95,1.15]:
                    dist=distance*mult;eye=root+np.array([math.sin(angle+shift)*dist,((meta.get('perch') or {}).get('perch_top_m',0)+(height-(meta.get('perch') or {}).get('perch_top_m',0))*elevation+.03) if meta.get('perch') else height*elevation,-math.cos(angle+shift)*dist]);sim.initialize_agent(0,camera(r,eye,target));obs=sim.get_sensor_observations();mask=np.asarray(obs[uuids['semantic']])==sid;yy,xx=np.where(mask)
                    if len(xx)>(600 if meta.get('perch') else 1000) and min(xx.min(),yy.min(),639-xx.max(),639-yy.max())>3:accepted=(eye,obs,mask,dist,angle+shift);break
        if accepted is None:raise RuntimeError('native complete visibility failed:'+name)
        eye,obs,mask,dist,angle_used=accepted;rgb=np.asarray(obs[uuids['rgb']])[...,:3];version=meta['runtime_sha256'][:12]+'_'+hashlib.sha256((T/'evidence/color_calibration.json').read_bytes()).hexdigest()[:10]+'_v7';dest=D/'screenshots'/version/house;dest.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(dest/(name+'.png'));Image.fromarray(rgb).save(dest/(name+'.jpg'),quality=88);Image.fromarray(np.uint8(mask)*255).save(dest/(name+'_body_mask.png'));excluded=feature_exclusion(r,sim,uuids['rgb'],meta,mask);Image.fromarray(np.uint8(excluded&mask)*255).save(dest/(name+'_feature_exclusion.png'));color=measure(rgb,mask,cal['thresholds'],excluded);pooled_pixels.append(rgb[mask&~excluded]);sim.get_rigid_object_manager().remove_object_by_id(body.object_id);reference_object=make_object(sim,r,reference_path,root,0,sid);reference_rgb=np.asarray(sim.get_sensor_observations()[uuids['rgb']])[...,:3].copy();sim.get_rigid_object_manager().remove_object_by_id(reference_object.object_id);body=make_object(sim,r,body_path,root,0,sid);Image.fromarray(reference_rgb).save(dest/(name+'_original_texture_reference.png'));appearance=measure_source_appearance(rgb,mask,palette=palette,reference_rgb=reference_rgb,max_highlight_ratio=.02,strict=False);sat=float(np.mean(np.all(rgb[mask]>=245,axis=1)));excluded_fraction=float((excluded&mask).sum()/mask.sum());row={'name':name,'image':str(dest/(name+'.jpg')),'rgb_png':str(dest/(name+'.png')),'body_mask':str(dest/(name+'_body_mask.png')),'feature_exclusion':str(dest/(name+'_feature_exclusion.png')),'rgb_sha256':hashlib.sha256((dest/(name+'.png')).read_bytes()).hexdigest(),'body_mask_sha256':hashlib.sha256((dest/(name+'_body_mask.png')).read_bytes()).hexdigest(),'feature_exclusion_sha256':hashlib.sha256((dest/(name+'_feature_exclusion.png')).read_bytes()).hexdigest(),'source_pixels':int(mask.sum()),'minimum_frame_margin_pixels':int(min(xx.min(),yy.min(),639-xx.max(),639-yy.max())),'eye_m':eye.tolist(),'target_m':target.tolist(),'distance_m':dist,'azimuth_rad':angle_used,'original_texture_reference':str(dest/(name+'_original_texture_reference.png')),'original_texture_reference_sha256':hashlib.sha256((dest/(name+'_original_texture_reference.png')).read_bytes()).hexdigest(),'original_texture_reference_glb':str(reference_path),'production_appearance':appearance,'highlight_gate_status':'pass' if appearance['highlight_ratio']<=.02 and appearance['color_observable'] else 'fail','saturated_white_pixels_scope':'diagnostic; intrinsically white albedo saturation is not specular gloss','saturated_rgb_ge245_ratio':sat,'color':color,'excluded_small_feature_body_fraction':excluded_fraction,'status':'pass' if appearance['highlight_ratio']<=.02 and appearance['color_observable'] and color['body_pixels_after_exclusion']>=cal['thresholds']['min_pixels'] and excluded_fraction<=.08 else 'fail'};images.append(row)
    sim.get_rigid_object_manager().remove_object_by_id(body.object_id)
    if stand:sim.get_rigid_object_manager().remove_object_by_id(stand.object_id)
    labels={x['color']['measured_label'] for x in images};pooled_array=np.concatenate(pooled_pixels);pooled_color=measure(pooled_array[:,None,:],np.ones((len(pooled_array),1),bool),cal['thresholds']);major=pooled_color['measured_label'];agree=sum(x['color']['measured_label']==major for x in images);passed=all(x['status']=='pass' for x in images) and (pooled_color['status']=='pass' or meta.get('cross_requirement_exempt_companion')) and abs(floor_delta)<1e-5 and error<1e-4 and follow<1e-4 and meta['emit_head_status']=='pass'

    return {'asset_id':meta['asset_id'],'house':house,'status':'pass' if passed else 'fail','root_m':root.tolist(),'support_deltas_m':deltas,'maximum_support_delta_m':max(map(abs,deltas)),'grounding_error_m':abs(floor_delta),'at_floor_contact':at_floor,'buried_3cm_positive_contact_control':buried,'raised_6cm_scene_collision':raised,'placements_attempted':attempts,'native_emitter_world_m':native.tolist(),'emit_readback_error_m':error,'emit_translation_and_yaw_follow_error_m':follow,'emit_head_status':meta['emit_head_status'],'native_height_m':native_height,'dominant_label':major,'pooled_body_color':pooled_color,'color_policy':'same-house four complete native silhouettes pooled after eye/nose/beak exclusions; pooled dominant fraction >= calibrated threshold; each view retained diagnostically','views_agreeing':agree,'semantic_body_mask_method':'native object semantic ID; separate authored silent stand ID; projected eye/nose/beak exclusion circles','pose_gate':meta['normalization'],'shader':shader,'screenshots':images,'audio':'not_run'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--views',type=Path,required=True);p.add_argument('--only');p.add_argument('--only-missing',action='store_true');a=p.parse_args();T=a.task_root;views=json.loads(a.views.read_text());placements=json.loads((a.reference_root/'round3/evidence/chair_native_placements.json').read_text());plan=json.loads((T/'plan.json').read_text());records=[]
    for view in views:
        house=view['house'];r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=view['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(view['room_manifest']),Path(view['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,T/'evidence/habitat'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
        for agent in cfg.agents:
            for spec in agent.sensor_specifications:spec.resolution=[640,640]
        with r.habitat_sim.Simulator(cfg) as sim:
            shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator)
            for item in plan['animals']:
                if a.only and a.only!=item['asset_id']:continue
                path=T/'assets'/item['asset_id']/'asset.json';receipt=T/'evidence/habitat'/house/(item['asset_id']+'.json')
                if not path.exists():continue
                meta=json.loads(path.read_text())
                if a.only_missing and receipt.exists():
                    previous=json.loads(receipt.read_text())
                    if previous.get('status')=='pass' and previous.get('runtime_sha256')==meta['runtime_sha256'] and previous.get('color_calibration_sha256')==hashlib.sha256((T/'evidence/color_calibration.json').read_bytes()).hexdigest() and previous.get('qa_protocol') in ['native_four_views_v6','native_four_views_v7']:continue
                if meta.get('status')=='rejected_normalization':continue
                rec={'asset_id':item['asset_id'],'house':house,'scene':str(scene),'runtime_sha256':meta['runtime_sha256'],'color_calibration_sha256':hashlib.sha256((T/'evidence/color_calibration.json').read_bytes()).hexdigest(),'qa_protocol':'native_four_views_v7'}
                try:rec.update(qualify(sim,r,uuids,T,meta,house,placements,shader))
                except BaseException as e:
                    rec.update(status='fail',error=repr(e),traceback=traceback.format_exc())
                    for obj in list(sim.get_rigid_object_manager().get_objects_by_handle_substring('').values()):sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
                dump(receipt,rec);records.append(rec);dump(T/'evidence/habitat_results.json',records);print('ANIMAL_NATIVE',item['asset_id'],house,rec['status'],rec.get('error',''),flush=True)
    return 0 if all(x['status']=='pass' for x in records) else 1
if __name__=='__main__':raise SystemExit(main())
