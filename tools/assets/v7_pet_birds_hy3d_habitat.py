"""Qualify static Hunyuan birds in native MP3D/HM3D using production PBR preparation."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from avengine.assets.source_materials import configure_source_pbr,measure_source_appearance,source_palette,read_glb
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
from v7_human_roster_habitat import camera,project
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
def make_object(sim,r,path,pos,yaw,semantic):
    manager=sim.get_object_template_manager();handle=str(path)+".bird_margin_1mm"
    attributes=manager.create_template(handle,register_template=False)
    attributes.render_asset_handle=str(path);attributes.collision_asset_handle=str(path)
    attributes.compute_COM_from_shape=False;attributes.shader_type="pbr";attributes.semantic_id=semantic
    attributes.margin=.001;attributes.scale=r.magnum.Vector3(1,1,1);manager.register_template(attributes,handle)
    obj=sim.get_rigid_object_manager().add_object_by_template_handle(handle);assert obj is not None
    obj.motion_type=r.habitat_sim.physics.MotionType.KINEMATIC;obj.translation=r.magnum.Vector3(pos)
    obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(yaw),r.magnum.Vector3.y_axis())
    obj.semantic_id=semantic;obj.root_scene_node.semantic_id=semantic;return obj
def yaw_matrix(yaw):
    c,s=math.cos(yaw),math.sin(yaw);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def floor_rays(sim,runtime,root,radius,objid=None):
    out=[]
    for x,z in [(0,0),(-radius,0),(radius,0),(0,-radius),(0,radius)]:
        start=root+np.array([x,.15,z]);hits=sim.cast_ray(runtime.habitat_sim.geo.Ray(runtime.magnum.Vector3(start),runtime.magnum.Vector3(0,-1,0)),.5).hits
        hits=[h for h in hits if h.object_id!=objid]
        out.append(None if not hits else float(hits[0].point[1]))
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--task-root",type=Path,required=True);p.add_argument("--reference-root",type=Path,required=True);p.add_argument("--views",type=Path,required=True);a=p.parse_args();T=a.task_root
    views=json.loads(a.views.read_text());placements=json.loads((a.reference_root/"round3/evidence/chair_native_placements.json").read_text());records=[]
    for view in views:
        house=view["house"]
        runtime=prepare_installed_habitat_runtime(runtime_prefix="/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z",magnum_python_site="/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages",mp3d_root=view["data_root"],rlr_sdk_root="/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg")
        inputs=load_and_validate_inputs(Path(view["room_manifest"]),Path(view["capture_request"]))
        cfg,uuids,_,scene=_make_configuration(inputs,None,T/"evidence/habitat"/house,mp3d_root=runtime.mp3d_root,include_audio_sensor=False,physics_config_path=runtime.physics_config_path)
        cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
        for agent in cfg.agents:
            for spec in agent.sensor_specifications:spec.resolution=[640,640]
        with runtime.habitat_sim.Simulator(cfg) as sim:
            shader=configure_source_pbr(cfg,runtime.habitat_sim,metadata_mediator=sim.metadata_mediator)
            for index,kind in enumerate(["canary","budgerigar"]):
                D=T/"assets"/kind;meta=json.loads((D/"asset.json").read_text());root=np.array(next(x["root_m"] for x in placements if x["house"]==house and x["status"]=="pass_candidate"),float)
                radius=meta["perch"]["base_radius_m"]*.7;floor=floor_rays(sim,runtime,root,radius)
                if any(x is None for x in floor):raise RuntimeError("Stage floor missing: "+house)
                root[1]=float(np.median(floor));sid=6400100+index;path=D/"runtime.glb";obj=make_object(sim,runtime,path,root,0,semantic=sid)
                # Default Kinematic masks exclude the static stage; override only this owned object.
                obj.override_collision_group(runtime.habitat_sim.physics.CollisionGroups.Dynamic)
                child=obj.root_scene_node.create_child();local=np.array(meta["emit_point_local_m"]);child.translation=runtime.magnum.Vector3(local)
                native=np.array(child.absolute_translation);error=float(np.linalg.norm(native-root-local));delta=np.array([.1,.02,-.07]);yaw=.35
                obj.translation=runtime.magnum.Vector3(root+delta);obj.rotation=runtime.magnum.Quaternion.rotation(runtime.magnum.Rad(yaw),runtime.magnum.Vector3.y_axis())
                follow=float(np.linalg.norm(np.array(child.absolute_translation)-(root+delta+yaw_matrix(yaw)@local)))
                obj.rotation=runtime.magnum.Quaternion.rotation(runtime.magnum.Rad(0),runtime.magnum.Vector3.y_axis());obj.translation=runtime.magnum.Vector3(root)
                grounded_contact=bool(obj.contact_test());obj.translation=runtime.magnum.Vector3(root-[0,.03,0]);buried_control=bool(obj.contact_test());obj.translation=runtime.magnum.Vector3(root+[0,.06,0]);raised=bool(obj.contact_test());obj.translation=runtime.magnum.Vector3(root)
                floor=floor_rays(sim,runtime,root,radius,obj.object_id);support=[None if x is None else float(root[1]-x) for x in floor]
                doc,blob=read_glb(path);palette=source_palette(doc,blob,path)
                images=[];height=meta["dimensions_m"][1];target=root+[0,height*.50,0]
                for name,angle in [("front",0),("quarter_left",-math.pi/6),("quarter_right",math.pi/6),("side",-math.pi/3)]:
                    accepted=None
                    for shift in [0,.12,-.12,.25,-.25]:
                        if accepted:break
                        for dist in [.65,.55,.75,.85,1.0]:
                            eye=root+np.array([math.sin(angle+shift)*dist,height*.65,-math.cos(angle+shift)*dist])
                            sim.initialize_agent(0,camera(runtime,eye,target));obs=sim.get_sensor_observations();mask=np.array(obs[uuids["semantic"]])==sid;yy,xx=np.where(mask)
                            if len(xx)>1000 and min(xx.min(),yy.min(),639-xx.max(),639-yy.max())>3:
                                accepted=(eye,angle+shift,dist,obs,mask);break
                    if accepted is None:raise RuntimeError(f"No complete visible native camera: {kind}/{house}/{name}")
                    eye,angle_used,dist,obs,mask=accepted;rgb=np.array(obs[uuids["rgb"]])[...,:3];ip=D/"screenshots"/house/(name+".jpg");ip.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(ip,quality=90)
                    appearance=measure_source_appearance(rgb,mask,palette=palette,max_highlight_ratio=.02,strict=False)
                    saturated=float(np.mean(np.all(rgb[mask]>=245,axis=1)));pixel=project(runtime,sim,uuids["rgb"],native)
                    images.append({"name":name,"image":str(ip),"native_pixels_unannotated":True,"jpeg_quality":90,"eye_m":eye.tolist(),"target_m":target.tolist(),"azimuth_rad":angle_used,"distance_m":dist,"source_pixels":int(mask.sum()),"emit_point_pixel":pixel,"saturated_rgb_ge245_ratio":saturated,"production_appearance":appearance,"status":"pass" if appearance["status"]=="pass" and saturated<=.02 else "fail"})
                bbox=obj.aabb;floor_delta=float(bbox.min[1]);native_height=float(bbox.max[1]-bbox.min[1])
                passed=buried_control and not raised and max(abs(x) for x in support if x is not None)<=.005 and all(x is not None for x in support) and error<.0001 and follow<.0001 and abs(floor_delta)<.00001 and all(x["status"]=="pass" for x in images)
                rec={"asset_id":meta["asset_id"],"kind":kind,"house":house,"status":"pass" if passed else "fail","scene":str(scene),"root_m":root.tolist(),"base_support_deltas_m":support,"at_floor_contact":grounded_contact,"buried_3cm_positive_contact_control":buried_control,"actual_enable_physics":bool(cfg.sim_cfg.enable_physics),"collision_group":"Dynamic override on this owned kinematic object only; Static stage included","built_with_bullet":bool(runtime.habitat_sim.built_with_bullet),"owned_object_margin_m":.001,"raised_probe_m":.06,"raised_6cm_scene_collision":raised,"collision_scope":"native whole rigid-object contact proxy after lifting 6cm; owned object margin 1mm to exclude intended base/floor support; not a complete scene-physics admission","emit_point_world_m":native.tolist(),"emit_readback_error_m":error,"emit_translation_and_yaw_follow_error_m":follow,"native_local_min_y_m":floor_delta,"native_height_m":native_height,"shader":shader,"screenshots":images,"audio":"not_run; static visual asset acceptance"}
                records.append(rec);dump(T/"evidence/habitat_results.json",records);print("BIRD_NATIVE",kind,house,rec["status"],support,flush=True)
                sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
    return 0 if len(records)==4 and all(row["status"]=="pass" for row in records) else 1
if __name__=="__main__":raise SystemExit(main())
