"""All-frame Habitat validation with bounded pose buffers and native caches.

Uses the committed animal_motion solver, native floor/clearance functions and
unchanged checks. A fresh simulator precedes reuse of a pose-buffer filename.
Only cache lifetime and stored pose buffers differ from the standard runner.
"""
import argparse,copy,gc,json,math,pathlib,subprocess,time
import numpy as np
from avengine.assets.animal_motion.profiles import Rig,PAWS
from avengine.assets.animal_motion.walk import ConstrainedWalk
from avengine.assets.animal_motion.native import StageClearance,camera,instantiate,overlay
from avengine.assets.animal_motion.stage import original_native_stage_mesh,OriginalVerticalStageRay
from avengine.assets.animal_motion.checks import frame_checks,direction_check,direction_negative_controls
from avengine.assets.roster_motion.episode import triangle_mesh,externalize_unchanged_images
from avengine.assets.roster_motion.pose import rigid_glb
from avengine.assets.roster_motion.contact import yaw_matrix
from avengine.capture.continuous_native_video import NativeContinuousVideo

def run(record,house,inputs,route,output,*,speed=.05,period=2.4,seconds=4.8,slots=8,mode="walk_animation",animation_attempts=None):
    output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=False);(output/"frames").mkdir()
    p0=np.array(route["root_m"],float);yaw=float(route["yaw_rad"]);rotation=yaw_matrix(yaw)
    eye=np.asarray(house["room"]["placement"]["camera_m"],float);target=p0+rotation@np.array([0,.4,-speed*seconds/2])
    forward=target-eye;forward/=np.linalg.norm(forward);scene={"listener_m":eye.tolist(),"camera_forward":forward.tolist(),"sources":{}}
    def fresh():
        v=NativeContinuousVideo(house,scene,inputs,[],width=512,height=384,source_material_cache_directory=output/"material_cache")
        cf=camera(v,eye,target);return v,cf
    video=None;encoder=None;start=time.process_time()
    code_root=pathlib.Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(["git","-C",str(code_root),"rev-parse","HEAD"],text=True).strip()
    result={"asset_id":record["asset_id"],"status":"pending","backend":"Habitat CPU llvmpipe","house":house["house"],"motion_mode":mode,"joint_animation_present":mode=="walk_animation","animation_attempts":animation_attempts or [],"frames":[],"audio_status":"not_run","native_validation_code_commit":commit,"bounded_adapter":{"pose_slots":slots,"reset_before_buffer_reuse":True,"all_frame_checks_retained":True,"thresholds_changed":False},"native_direction_negative_frames":{"180":[],"90":[]}}
    try:
        video,(eye,cf,cr,cu)=fresh();renderer=copy.deepcopy(video.receipt)
        stagepath=pathlib.Path(house["input"].get("visual_source") or pathlib.Path(house["input"]["scene_directory"])/(house["input"]["scan_id"]+".glb"))
        mesh,transform=original_native_stage_mesh(video,stagepath);rays=OriginalVerticalStageRay(stagepath,mesh=mesh,transform_evidence=transform);clearance=StageClearance(mesh,p0[1])
        class RouteFloor:
            def many(self,points):return rays.many(np.asarray(points)@rotation.T+p0)-p0[1]
            def __call__(self,p):return float(self.many([p])[0])
        floor=RouteFloor()
        if mode=="walk_animation":
            rig=Rig.load(record);walk=ConstrainedWalk(rig,speed=speed,period=period,floor=floor)
        else:
            from avengine.assets.animal_motion.slide import StandingSlide
            walk=StandingSlide(record,speed=speed,floor=floor);rig=walk.rig
        ed=externalize_unchanged_images(rig.document,rig.binary,output/"textures")
        n=round(seconds*30)+1;selected={0,n//4,n//2,3*n//4,n-1};images=[];resets=[]
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-n","-f","rawvideo","-pix_fmt","rgb24","-s","512x384","-r","30","-i","pipe:0","-an","-c:v","libx264","-threads","1","-preset","veryfast","-crf","20","-pix_fmt","yuv420p",str(output/"motion.mp4")]
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        for i,t in enumerate(np.arange(n)/30):
            if i and i%slots==0:
                video.close();video=None;gc.collect()
                video,(eye,cf,cr,cu)=fresh();resets.append(i)
            parts,row=walk.frame(float(t));v,f=triangle_mesh(rig.document,rig.binary,parts);root=p0+rotation@np.asarray(row["world_root_m"]);world=v@rotation.T+root
            row["minimum_nonfloor_clearance_m"]=clearance.check(world,f)
            near=world[world[:,1]<=p0[1]+.08]
            if not len(near):raise ValueError("no_near_floor_skin_vertices")
            row["body_floor_min_m"]=float(np.min(near[:,1]-rays.many(near)))
            snap=np.asarray(video.pathfinder.snap_point(root));nav=float(np.linalg.norm((snap-root)[[0,2]]))
            if not np.isfinite(nav):raise ValueError("native_navigation_missing")
            row["navmesh_horizontal_error_m"]=nav
            matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=root
            path=output/"frames"/f"pose_slot_{i%slots:02d}.glb"
            # The simulator holding the prior use of this filename was closed.
            rigid_glb(ed,rig.binary,parts,path)
            obj=instantiate(video,path,matrix);native=np.asarray(obj.root_scene_node.absolute_transformation(),float)
            # Check actual native geometry bounds, including after slot reuse.
            bbox=obj.aabb
            native_bounds=np.array([np.asarray(bbox.min,float),np.asarray(bbox.max,float)])
            expected_bounds=np.array([v.min(0),v.max(0)])
            geometry_error=float(np.max(np.abs(native_bounds-expected_bounds)))
            if geometry_error>2e-5:raise ValueError("native_geometry_bounds_stale_or_mismatched")
            row.update(native_geometry_bounds_m=native_bounds.tolist(),expected_pose_bounds_m=expected_bounds.tolist(),native_geometry_bounds_error_m=geometry_error)
            bone=obj.root_scene_node.create_child();bone.transformation=video.mn.Matrix4(np.asarray(row["mouth_bone_matrix"]))
            mouth=bone.create_child();mouth.translation=video.mn.Vector3(row["mouth_bone_local_m"]);actual_mouth=np.asarray(mouth.absolute_transformation(),float)[:3,3]
            hn=obj.root_scene_node.create_child();hn.translation=video.mn.Vector3(row["head_m"]);tn=obj.root_scene_node.create_child();tn.translation=video.mn.Vector3(row["tail_m"])
            row.update(frame_index=i,root_world_m=native[:3,3].tolist(),head_world_m=np.asarray(hn.absolute_translation).tolist(),tail_world_m=np.asarray(tn.absolute_translation).tolist(),mouth_world_m=actual_mouth.tolist(),pose_glb=str(path),pose_buffer_is_transient=True,pose_reconstruction={"source_master":str(rig.path),"source_inventory":record.get("walking_candidate"),"time_s":float(t),"speed_m_s":speed,"period_s":period,"route":route},native_root_error_m=float(np.max(np.abs(native-matrix))),native_emitter_readback_error_m=float(np.linalg.norm(actual_mouth-(native@np.r_[row["mouth_m"],1])[:3])))
            for foot in row["feet"].values():
                for k in ("support_world_m","support_target_world_m","sole_world_m"):foot[k]=(native@np.r_[np.asarray(foot[k])-np.asarray(row["world_root_m"]),1])[:3].tolist()
                wp=np.asarray(foot["sole_world_m"]);ray=video.hs.geo.Ray(video.mn.Vector3(wp+[0,.2,0]),video.mn.Vector3(0,-1,0));hits=[h for h in video.sim.cast_ray(ray,.85).hits if int(h.object_id)!=obj.object_id]
                if not hits:raise ValueError("native_paw_floor_ray_missing")
                fy=float(hits[0].point[1]);exact=float(rays.many([wp])[0])
                if abs(fy-exact)>.0001:raise ValueError("original_and_native_floor_ray_disagree")
                foot.update(floor_delta_m=float(wp[1]-fy),native_floor_y_m=fy,original_floor_y_m=exact)
            obs=video.sim.get_sensor_observations();rgb=np.asarray(obs["rgb"])[...,:3];mask=np.asarray(obs["semantic"])==700001
            yy,xx=np.where(mask);row["visible_pixels"]=int(mask.sum());row["complete_silhouette_in_frame"]=bool(len(xx)>100 and min(xx.min(),yy.min(),511-xx.max(),383-yy.max())>=3)
            encoder.stdin.write(rgb.tobytes())
            if i in selected:
                ip=output/f"frame_{i:04d}.png";overlay(rgb,eye,(cf,cr,cu),row["head_world_m"],row["tail_world_m"],np.asarray(row["root_world_m"])+[0,.02,0],rotation@np.array([0,0,-speed])).save(ip);images.append(str(ip))
            result["frames"].append(row);video.sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
            for angle in (180,90):
                badmatrix=matrix.copy();badmatrix[:3,:3]=rotation@yaw_matrix(math.radians(angle));bad=instantiate(video,path,badmatrix,semantic=700002);br=np.asarray(bad.root_scene_node.absolute_transformation(),float)
                bh=bad.root_scene_node.create_child();bh.translation=video.mn.Vector3(row["head_m"]);bt=bad.root_scene_node.create_child();bt.translation=video.mn.Vector3(row["tail_m"])
                neg={"time_s":float(t),"root_world_m":br[:3,3].tolist(),"head_world_m":np.asarray(bh.absolute_translation).tolist(),"tail_world_m":np.asarray(bt.absolute_translation).tolist(),"applied_native_yaw_delta_deg":angle,"physically_rotated_native_drawable":True,"body_rotation_error_deg":float(np.degrees(np.arccos(np.clip((np.trace(native[:3,:3].T@br[:3,:3])-1)/2,-1,1))))}
                result["native_direction_negative_frames"][str(angle)].append(neg)
                if i==0:
                    nrgb=np.asarray(video.sim.get_sensor_observations()["rgb"])[...,:3];overlay(nrgb,eye,(cf,cr,cu),neg["head_world_m"],neg["tail_world_m"],br[:3,3]+[0,.02,0],rotation@np.array([0,0,-speed])).save(output/f"negative_{angle}.png")
                video.sim.get_rigid_object_manager().remove_object_by_id(bad.object_id)
            if i%30==0:print("BOUNDED_NATIVE_FRAME",record["asset_id"],i,"floor",row["body_floor_min_m"],"pixels",row["visible_pixels"],flush=True)
        encoder.stdin.close();rc=encoder.wait();encoder=None
        if rc:raise RuntimeError("video_encoder_failed")
        checks=frame_checks(result["frames"],native=True,mode=mode);controls=direction_negative_controls(result["frames"]);actual={}
        for angle,rows in result["native_direction_negative_frames"].items():
            actual[angle]={"check":direction_check([r["time_s"] for r in rows],[r["root_world_m"] for r in rows],[r["head_world_m"] for r in rows],[r["tail_world_m"] for r in rows]),"physically_rotated_native_drawable":True,"body_rotation_error_deg":rows[0]["body_rotation_error_deg"],"all_frame_max_rotation_error_deg":max(abs(r["body_rotation_error_deg"]-float(angle)) for r in rows),"native_frame_count":len(rows)}
        actual["status"]="pass" if actual["180"]["check"]["status"]==actual["90"]["check"]["status"]=="fail" else "fail"
        if not all(r["complete_silhouette_in_frame"] for r in result["frames"]):checks["status"]="fail";checks["failures"].append("native_silhouette_clipped_or_under_100_pixels")
        result.update(status="geometry_pass_audio_pending" if checks["status"]==controls["status"]==actual["status"]=="pass" else "geometry_fail",checks=checks,direction_negative_controls=controls,native_direction_negative_controls=actual,renderer=renderer,stage_transform=transform,source_master=str(rig.path),source_provenance=rig.provenance,support_plane=walk.support_plane,listener_m=eye.tolist(),camera_forward=cf.tolist(),images=images,frame_count=len(result["frames"]),native_checks_run=True,native_stage_floor_triangles=clearance.floor_triangles,max_native_geometry_bounds_error_m=max(r["native_geometry_bounds_error_m"] for r in result["frames"]),max_emitter_error_m=max(r["native_emitter_readback_error_m"] for r in result["frames"]),min_visible_pixels=min(r["visible_pixels"] for r in result["frames"]),simulator_reset_frame_indices=resets)
    except Exception as e:
        import traceback
        result.update(status="failed",error=repr(e),traceback=traceback.format_exc(),failure_kind="motion_rejected" if isinstance(e,ValueError) else "execution_error")
    finally:
        if encoder is not None:
            encoder.stdin.close();encoder.wait()
        if video is not None:video.close()
        result["cpu_s"]=time.process_time()-start;(output/"native_motion.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("inventory","house","runtime","route","output"):p.add_argument("--"+flag,type=pathlib.Path,required=True)
    p.add_argument("--asset",required=True);p.add_argument("--speed",type=float,default=.05);p.add_argument("--period",type=float,default=2.4);p.add_argument("--seconds",type=float,default=4.8);p.add_argument("--pose-slots",type=int,default=8)
    p.add_argument("--mode",choices=("walk_animation","slide_no_animation"),default="walk_animation");p.add_argument("--animation-attempts",type=pathlib.Path)
    a=p.parse_args();record=next(r for r in json.loads(a.inventory.read_text())["assets"] if r["asset_id"]==a.asset)
    result=run(record,json.loads(a.house.read_text()),json.loads(a.runtime.read_text()),json.loads(a.route.read_text()),a.output,speed=a.speed,period=a.period,seconds=a.seconds,slots=a.pose_slots,mode=a.mode,animation_attempts=json.loads(a.animation_attempts.read_text()) if a.animation_attempts else None)
    print({k:result.get(k) for k in ("asset_id","status","error","frame_count","max_emitter_error_m","min_visible_pixels","cpu_s")},flush=True)
    if result["status"]!="geometry_pass_audio_pending":raise SystemExit(2)
if __name__=="__main__":main()
