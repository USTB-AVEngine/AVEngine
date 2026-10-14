#!/usr/bin/env python3
"""Native UE render availability and visibility of room placement witnesses.

This command does not produce parent scenes, questions, audio or training rows.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, time, traceback
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from avengine.capture.kujiale_spear_runtime import (
    launch_native, camera_helpers, apply_camera, from_ue, apply_native_rgb_profile,
    configure_scene_lighting, capture_view, close_native)
from avengine.capture.kujiale_pilot_environment import verify_gpu
from avengine.capture.kujiale_native_exposure import bind_histogram_profile

def read(path):
    return json.loads(Path(path).read_text())

def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".pending."+str(os.getpid())+"."+str(time.time_ns()))
    try:
        with temporary.open("x") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.write("\n")
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

def measure_point(depth,view,point,occlusion_tolerance_m=.005):
    """Native depth is radial metres. A visible empty placement point is not an asset mask."""
    eye=np.asarray(view["position_m"],float);point=np.asarray(point,float)
    yaw,pitch=map(math.radians,[view["yaw_deg"],view["pitch_deg"]])
    forward=np.array([math.cos(pitch)*math.cos(yaw),math.sin(pitch),math.cos(pitch)*math.sin(yaw)])
    side=np.array([-math.sin(yaw),0.,math.cos(yaw)]);up=np.cross(side,forward)
    ray=point-eye;axial=float(ray@forward);radial=float(np.linalg.norm(ray))
    if axial<=0:
        return {"status":"fail","reason":"target_behind_native_camera","radial_distance_m":radial}
    tangent=math.tan(math.radians(view.get("hfov_deg",85))/2)
    height,width=depth.shape
    x=float(ray@side/(axial*tangent));y=float(ray@up/(axial*tangent*height/width))
    u=int(round((x+1)*width/2-.5));v=int(round((1-y)*height/2-.5))
    if not 2<=u<width-2 or not 2<=v<height-2:
        return {"status":"fail","reason":"target_outside_native_frame","pixel":[u,v]}
    patch=np.asarray(depth[v-1:v+2,u-1:u+2],float);valid=np.isfinite(patch)&(patch>0)
    # Infinite/65504 background means no intervening geometry, but does not
    # independently prove the room rendered. That is checked on the whole frame.
    samples=patch[valid];distance=float(np.median(samples)) if len(samples) else None
    passed=distance is not None and distance>=radial-occlusion_tolerance_m
    return {"status":"pass" if passed else "fail","point_m":point.tolist(),"pixel":[u,v],
            "native_depth_radial_m":distance,"radial_distance_m":radial,
            "occlusion_tolerance_m":occlusion_tolerance_m,"valid_neighbourhood_pixels":int(valid.sum()),
            "reason":None if passed else "native_room_geometry_occludes_placement_point",
            "scope":"view of the specified empty placement point; individual rendered asset masks remain a production gate"}

def read_camera(camera,view):
    from avengine.backends.spear_ue.research_runtime import read_scene_component_pose
    pose=read_scene_component_pose(camera.K2_GetRootComponent())
    position=from_ue(pose["location_cm"]);rotation=pose["rotation_deg"]
    error=float(np.max(np.abs(np.array(position)-view["position_m"])))
    yaw_error=abs((float(rotation[2])-view["yaw_deg"]+180)%360-180)
    pitch_error=abs(float(rotation[1])-view["pitch_deg"])
    if error>.0005 or yaw_error>1e-4 or pitch_error>1e-4:
        raise ValueError("native room camera differs from commanded pose")
    return {"status":"pass","native_component_pose":pose,"position_m":position,
            "position_max_error_m":error,"yaw_error_deg":yaw_error,"pitch_error_deg":pitch_error,
            "position_tolerance_m":.0005,"angle_tolerance_deg":1e-4}

def run(args):
    root=args.root.resolve();output=args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("room evidence must stay inside pilot output")
    if args.gpu!=2 or os.nice(0)<10:
        raise ValueError("room native checks require authorized physical GPU2 and nice>=10")
    free=os.statvfs(root);free=free.f_bavail*free.f_frsize
    memory=next(int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
    if free<150_000_000_000 or memory<40_000_000_000:
        raise ValueError("room native launch resource floor failed")
    runtime=read(args.runtime);cv2.setNumThreads(1);completed=0
    for original in runtime["houses"]:
        if args.house and original["house"]!=args.house:
            continue
        if original["split"]!="train":
            raise ValueError("native room legality uses admitted train rooms only")
        house=deepcopy(original);rooms=[]
        for room in house["checked_rooms"]:
            if args.room_labels and room["room_label"] not in args.room_labels:
                continue
            label=house["house"]+"_"+room["room_label"];dest=output/(label+".json")
            if dest.exists():
                print("RESUME_NATIVE_ROOM",label,flush=True);continue
            cpu=args.cpu_rooms/(label+".json")
            if not cpu.exists():
                raise ValueError("CPU legality witness missing: "+str(cpu))
            rooms.append((room,read(cpu),dest))
        if not rooms:
            continue
        ue=house["input"]["ue_config"];ue.update(uproject=str(root/"ue_v1/runtime/SpearSim/SpearSim.uproject"),
            output_root=str(root),physical_gpu_index=2,graphics_adapter=args.graphics_adapter,
            unreal_editor=str(root/"scripts/ue_shared/ue_gpu2"))
        lighting=root/"inputs/room_legality_lights_v1"/house["house"]/"native_lighting_audit.json"
        lighting.parent.mkdir(parents=True,exist_ok=True)
        if not lighting.exists():
            shutil.copy2(ue["native_lighting_audit"],lighting)
        ue["native_lighting_audit"]=str(lighting)
        ue=bind_histogram_profile(ue,scene_seed="kujiale-room-legality-"+house["house"])
        instance=camera=components=None;house_started=time.time()
        try:
            instance,game,launch=launch_native(ue,house["input"]["ue_map_path"])
            launch["actual_graphics_gpu"]=verify_gpu(ue,launch)
            spawn,depth_reader,rgb_reader,*_=camera_helpers()
            with instance.begin_frame():
                camera,components=spawn(game,horizontal_fov_deg=85,width=1280,height=720)
                exposure=apply_native_rgb_profile(components["rgb"],ue)
                lights=configure_scene_lighting(game,read(ue["acceptance_plan_path"]),ue)
                components["object_ids"].set_property_value(property_name="bCaptureEveryFrame",property_value=False)
            with instance.end_frame():
                pass
            warmed=False
            for room,cpu,dest in rooms:
                label=house["house"]+"_"+room["room_label"]
                result={"schema":"kujiale_room_native_render_visibility_v1","house":house["house"],
                    "room_id":room["room_label"],"split":"train","ue_map_path":house["input"]["ue_map_path"],
                    "renderer":"native UE/SPEAR Vulkan, production design lights and histogram exposure",
                    "launch":launch,"exposure":exposure,"native_lights":lights,"category_views":[],
                    "status":"fail","parent_scenes_started":0,"questions_generated":0}
                checks=[c for c in cpu["category_checks"] if c["status"]=="available_CPU_native_LOD_witness"]
                for cell in checks:
                    row={"category_key":cell["category_key"],"coarse_category":cell["coarse_category"],
                         "representative_asset_id":cell["representative_asset_id"],"status":"fail","views":[]}
                    for index,view in enumerate(cell["placement"]["camera_candidates"][:3]):
                        trace=[]
                        bgr,depth=capture_view(instance,camera,components,view,
                            warmup_frames=120 if not warmed else 8,exposure_trace=trace)
                        warmed=True
                        with instance.begin_frame():
                            pose=read_camera(camera,view)
                        with instance.end_frame():
                            pass
                        visibility=measure_point(depth,view,view["target_m"])
                        render={"rgb_std_255":float(bgr.std()),"rgb_mean_255":float(bgr.mean()),
                            "native_positive_finite_depth_fraction":float(np.mean(np.isfinite(depth)&(depth>0)&(depth<1000)))}
                        render["status"]="pass" if render["rgb_std_255"]>1 and render["native_positive_finite_depth_fraction"]>.05 else "fail"
                        record={"view":view,"camera_readback":pose,"placement_point_visibility":visibility,
                                "native_render_availability":render,"exposure_trace":trace}
                        row["views"].append(record)
                        if visibility["status"]=="pass" and render["status"]=="pass":
                            image=root/"renders/room_legality_v1"/label/(cell["category_key"]+"_view_"+str(index)+"_"+str(time.time_ns())+".jpg")
                            image.parent.mkdir(parents=True,exist_ok=True)
                            if image.exists():
                                raise ValueError("completed room render retained: "+str(image))
                            if not cv2.imwrite(str(image),cv2.resize(bgr,(960,540)),[cv2.IMWRITE_JPEG_QUALITY,88]):
                                raise ValueError("native room JPEG write failed")
                            record["rgb"]=str(image);row.update(status="pass",selected_view=record);break
                    if row["status"]!="pass":
                        row["reason"]="no_native_visible_placement_point_at_the_retained_camera_or_render_checks"
                    result["category_views"].append(row)
                    print("NATIVE_CELL",label,cell["category_key"],row["status"],flush=True)
                result["native_room_render_status"]="pass" if any(v["native_render_availability"]["status"]=="pass" for c in result["category_views"] for v in c["views"]) else "fail"
                result["status"]="pass" if result["native_room_render_status"]=="pass" and all(c["status"]=="pass" for c in result["category_views"]) else "category_failure"
                result["scope"]="native room render plus line of sight to CPU placement point; physical placement proven by audited LOD rays and OBBs, individual masks/actions/RIR/question truth still require the existing production gates"
                publish(dest,result);completed+=1
                with (root/"PROGRESS.md").open("a") as stream:
                    stream.write("- "+time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())+"："+label+" 本轮GPU2原生出图/相机摆放点读回复核 "+result["status"]+"；"+str(sum(c["status"]=="pass" for c in result["category_views"]))+"/"+str(len(result["category_views"]))+"细类可见；未启动父场景/出题。\n")
                print("NATIVE_ROOM_DONE",label,result["status"],flush=True)
        finally:
            if instance is not None:
                close_native(instance,camera,components)
            print("NATIVE_HOUSE_CLOSED",house["house"],"seconds",time.time()-house_started,flush=True)
    return completed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","runtime","cpu-rooms","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--graphics-adapter",type=int,default=0)
    p.add_argument("--house");p.add_argument("--room-labels",nargs="+")
    run(p.parse_args())

if __name__=="__main__":
    main()
