"""Independent CPU measurements; native Habitat/audio are separate statuses."""
import argparse,json,pathlib,subprocess,time
import numpy as np
from avengine.assets.roster_motion.pose import pose,accessor
from avengine.assets.roster_motion.episode import triangle_mesh
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=pathlib.Path,required=True);p.add_argument("--asset",required=True);p.add_argument("--output",type=pathlib.Path,required=True);p.add_argument("--speed",type=float,default=.05);p.add_argument("--period",type=float,default=2.4);p.add_argument("--seconds",type=float,default=4.8);p.add_argument("--reference-code-root",type=pathlib.Path);p.add_argument("--blender",type=pathlib.Path);a=p.parse_args()
 reference=None
 if a.reference_code_root:
  import avengine.assets
  avengine.assets.__path__.append(str(a.reference_code_root/"src/avengine/assets"))
  reference={"path":str(a.reference_code_root),"git_commit":subprocess.check_output(["git","-C",str(a.reference_code_root),"rev-parse","HEAD"],text=True).strip(),"uncommitted_preliminary_only":True,"native_admission_claim":False}
 from avengine.assets.animal_motion.profiles import Rig
 from avengine.assets.animal_motion.walk import ConstrainedWalk
 from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls
 a.output.mkdir(parents=True,exist_ok=False);start=time.process_time();frames=[];deform=[]
 result={"asset_id":a.asset,"status":"pending","reference_code":reference,"native_habitat":"not_run","native_audio":"not_run","thresholds_changed":False}
 try:
  r=next(x for x in json.loads(a.inventory.read_text())["assets"] if x["asset_id"]==a.asset)
  rig=Rig.load(r);walk=ConstrainedWalk(rig,speed=a.speed,period=a.period)
  v0,faces=triangle_mesh(rig.document,rig.binary,walk.reference_parts)
  face0=v0[faces];e0=np.max(np.linalg.norm(face0-np.roll(face0,1,axis=1),axis=2),axis=1)
  from pathlib import Path
  for i,t in enumerate(np.arange(round(a.seconds*30)+1)/30):
   parts,row=walk.frame(float(t));root=np.array(row["world_root_m"]);row.update(root_world_m=root.tolist(),head_world_m=(np.array(row["head_m"])+root).tolist(),tail_world_m=(np.array(row["tail_m"])+root).tolist())
   frames.append(row)
   v,f=triangle_mesh(rig.document,rig.binary,parts)
   tri=v[f];longest=np.max(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2),axis=1)
   area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
   ratios=longest/np.maximum(e0,1e-12);shard=ratios>=2.
   deform.append({"frame":i,"time_s":float(t),"maximum_edge_growth":float(ratios.max()),"area_in_edges_doubled_share":float(area[shard].sum()/max(area.sum(),1e-20))})
   if i%30==0:print("CPU_FRAME",i,"floor",row["body_floor_min_m"],"pawgap",max(abs(x["floor_delta_m"]) for x in row["feet"].values()),"lock",max(x["support_lock_error_m"] for x in row["feet"].values()),flush=True)
   if float(ratios.max())>1.5+1e-6 and i>=1:
    result["rejection_witness"]={"reason":"owner_edge_growth_limit_1_5","frame":i,"maximum_edge_growth":float(ratios.max()),"required_frame_count":round(a.seconds*30)+1,"full_sequence_checked":False}
    print("REJECTED_EDGE_GROWTH_WITNESS",result["rejection_witness"],flush=True);break
  check=frame_checks(frames);negative=direction_negative_controls(frames)
  # Owner 02:3x explicitly limits longest triangle-edge growth to 1.5.
  result.update(status="cpu_motion_pass_native_pending" if check["status"]==negative["status"]=="pass" else "cpu_motion_fail",checks=check,direction_negative_controls=negative,support_plane=walk.support_plane,deformation_diagnostics=deform,frames=frames,frame_count=len(frames),speed_m_s=a.speed,period_s=a.period,source_master=str(rig.path),source_provenance=rig.provenance,self_intersection_status="not_run")
  result["edge_growth_limit"]=1.5
  result["maximum_edge_growth"]=max(x["maximum_edge_growth"] for x in deform)
  if result["maximum_edge_growth"]>1.5+1e-6:
   result["status"]="cpu_motion_fail";result["deformation_failure"]="owner_edge_growth_limit_1_5"
  if a.blender and result["status"]=="cpu_motion_pass_native_pending":
   from avengine.assets.animal_motion.export import export_walk
   export=export_walk(rig,walk,np.arange(round(a.seconds*30)+1)/30,a.output/"Walking_Repaired.glb")
   (a.output/"repaired_export.json").write_text(json.dumps(export,indent=2))
   surfacepath=a.output/"surface.json"
   cmd=[str(a.blender),"-b","--factory-startup","--disable-autoexec","-t","1","--python-exit-code","2","--python",str(pathlib.Path(__file__).with_name("blender_check_quadruped_surface.py")),"--","--glb",str(a.output/"Walking_Repaired.glb"),"--output",str(surfacepath),"--frames",str(len(frames))]
   registration_request=rig.path.parent/"request.json"
   if registration_request.exists():cmd+=["--registration-request",str(registration_request)]
   code=subprocess.call(cmd)
   result["surface_checks"]=json.loads(surfacepath.read_text())
   if code or result["surface_checks"]["status"]!="pass":result["status"]="cpu_motion_fail";result["surface_failure"]="visible_regional_penetration_or_edge_growth_or_uv_tear"
   if result["surface_checks"]["worst_shard_share_area"]>.025 or any(x["nonfinite_vertices"] for x in result["surface_checks"]["frames"]):
    result["status"]="cpu_motion_fail";result["deformation_failure"]="original_calibrated_2_5_percent_shard_gate_or_nonfinite_vertices"
   result["self_intersection_status"]=result["surface_checks"]["status"]
   deformation=a.output/"deformation.json";gatepath=a.output/"deformation_gate.json"
   cmd=[str(a.blender),"-b","--factory-startup","--disable-autoexec","-t","1","--python",str(pathlib.Path(__file__).with_name("measure_walk_deformation.py")),"--",str(a.output/"Walking_Repaired.glb"),str(deformation),"Walking_Repaired",str(len(frames))]
   subprocess.run(cmd,check=True)
   gatecmd=[__import__("sys").executable,str(pathlib.Path(__file__).with_name("gate_rigged_asset.py")),str(deformation)]
   gated=subprocess.run(gatecmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);gatecode=gated.returncode
   print(gated.stdout,flush=True)
   lines=[line.split(" ",1)[1] for line in gated.stdout.splitlines() if line.startswith("RIGGED_GATE_")]
   verdict=json.loads(lines[-1]) if lines else {"status":"execution_failed","returncode":gatecode}
   gatepath.write_text(json.dumps(verdict,indent=2));result["deformation_gate"]=verdict
   if gatecode:result["status"]="cpu_motion_fail"
 except Exception as e:
  import traceback
  result.update(status="failed",error=repr(e),traceback=traceback.format_exc(),frames=frames)
 result["cpu_s"]=time.process_time()-start
 (a.output/"checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print({k:v for k,v in result.items() if k not in ("frames","checks","direction_negative_controls","source_provenance","support_plane","traceback","deformation_diagnostics","surface_checks","deformation_gate")},flush=True)
 if result["status"]!="cpu_motion_pass_native_pending":raise SystemExit(2)
if __name__=="__main__":main()
