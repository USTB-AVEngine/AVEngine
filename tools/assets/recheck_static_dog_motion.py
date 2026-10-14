"""Recheck owned static-dog bindings with the explicitly authorized motion-only policy."""
import argparse,datetime,gc,json,pathlib,runpy,signal,sys,time,psutil
from avengine.assets.quadruped_motion_policy import OWNER093X, policy_frame_checks, policy_direction_check
from avengine.assets.quadruped_penetration import visible_penetration_gate
ROOT=pathlib.Path("/data/jzy/tmp/claude_hardbank_20260923/v7_rig_dogs_20261014")
WT=pathlib.Path(__file__).resolve().parents[2]
def read(p):return json.loads(pathlib.Path(p).read_text())
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,d):
 t=p.with_name(p.name+".tmp."+str(__import__("os").getpid()));t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def progress(s):
 with (ROOT/"PROGRESS.md").open("a") as f:f.write("\n"+stamp()+" "+s+"\n")
def tool(name,args):
 old=sys.argv;sys.argv=[str(WT/"tools/assets"/name),*map(str,args)]
 try:runpy.run_path(sys.argv[0],run_name="__main__");return 0
 except SystemExit as e:return int(e.code or 0)
 finally:sys.argv=old;gc.collect()
class BindingBudget(BaseException):pass
def budget(signum,frame):
 owned=[]
 for p in psutil.Process().children(recursive=True):
  try:owned.append((p,p.create_time(),p.cmdline()))
  except psutil.Error:pass
 for p,created,command in reversed(owned):
  try:
   if p.create_time()==created and p.cmdline()==command:p.terminate()
  except psutil.Error:pass
 _,alive=psutil.wait_procs([x[0] for x in owned],timeout=3)
 for p in alive:
  old=next((x for x in owned if x[0].pid==p.pid),None)
  try:
   if old and p.create_time()==old[1] and p.cmdline()==old[2]:p.kill()
  except psutil.Error:pass
 raise BindingBudget()
def controls(m):
 rows=m.get("native_direction_negative_frames",{})
 out={}
 for angle in ("180","90"):
  f=rows.get(angle,[])
  if len(f)<2:return {"status":"fail","reason":"retained_actual_negative_frames_missing"}
  out[angle]=policy_direction_check([x["time_s"] for x in f],[x["root_world_m"] for x in f],[x["head_world_m"] for x in f],[x["tail_world_m"] for x in f],policy=OWNER093X)
 return {"status":"pass" if all(x["status"]=="fail" for x in out.values()) else "fail","checks":out,"retained_physically_rotated_native_drawables":True}
def surface_pass(cp):
 s=cp.get("surface_checks",{})
 return cp.get("frame_count")==145 and cp.get("maximum_edge_growth",float("inf"))<=2.000001 and s.get("frame_count")==145 and s.get("status")=="pass" and not any(visible_penetration_gate(f["regional_penetration_measurements"],cross_region_area_limit_m2=400e-6,depth_limit_m=.01,same_region_depth_is_criterion=False) for f in s["frames"])
def candidates(aid,coverage,legacy):
 paths=[]
 for a in coverage.get("animation_attempts",[]):paths.append(a.get("directory"))
 for a in legacy.get("animation_attempts",[]):paths.append(a.get("directory"))
 for a in legacy.get("checks",[]):paths.append(a.get("directory"))
 if aid=="v7_dog_labrador_black_20261012":paths.append(str(ROOT/"derived/labrador_black_original_capsule_v7"))
 unique=[]
 for p in paths:
  if not p:continue
  p=pathlib.Path(p)
  if not p.resolve().is_relative_to(ROOT) or not (p/"inventory.json").exists():continue
  if p not in unique:unique.append(p)
 def score(p):
  files=sorted(p.glob("cpu_check*/checks.json"),key=lambda x:x.stat().st_mtime,reverse=True)
  cp=read(files[0]) if files else {}
  s=cp.get("surface_checks",{});rows=s.get("frames",[])
  area=max((sum(g["new_area_m2"] for g in f.get("regional_penetration_measurements",[]) if g["region_pair"][0]!=g["region_pair"][1]) for f in rows),default=0)
  return (0 if surface_pass(cp) else 1 if cp.get("maximum_edge_growth",0)<=2.000001 else 2,area,-p.stat().st_mtime)
 return sorted(unique,key=score)[:3]
def cp_for_motion(m):
 for a in m.get("animation_attempts",[]):
  if not a.get("checks"):continue
  cp=read(a["checks"])
  if cp.get("source_master")==m.get("source_master") and surface_pass(cp):return a["checks"],cp
 return None,None
def qualify_native(path,cp_path=None,reused=False):
 m=read(path)
 if (m.get("frame_count")!=145 or len(m.get("frames",[]))!=145
     or not m.get("native_checks_run") or m.get("checks",{}).get("status")!="pass"):return None
 ch=policy_frame_checks(m["frames"],mode=m["motion_mode"],native=True,policy=OWNER093X);neg=controls(m)
 if ch["status"]!=neg["status"] or ch["status"]!="pass":return None
 if m.get("motion_mode")=="walk_animation":
  cp=read(cp_path) if cp_path else None
  if cp is None or not surface_pass(cp):return None
 return {"asset_id":m["asset_id"],"status":"pass","classification":"有动画" if m["motion_mode"]=="walk_animation" else "只能滑行","motion_mode":m["motion_mode"],"native_motion":str(path),"native_policy_checks":ch,"native_direction_negative_controls":neg,"cpu_checks":str(cp_path) if cp_path else None,"native_evidence_reused":reused,"native_validation_code_commit":m.get("native_validation_code_commit"),"policy":OWNER093X,"per_asset_audio_required":False,"production_audio_screening_required":True,"completed":stamp()}
def publish(aid,adm):
 path=ROOT/"validation"/aid/"owner093x_motion_admission.json";atomic(path,adm);return str(path)
def existing(aid):
 walks=[];slides=[]
 for path in (ROOT/"validation"/aid).glob("*/native_motion.json"):
  m=read(path)
  if m.get("checks",{}).get("status")!="pass" or m.get("frame_count")!=145:continue
  if m.get("motion_mode")=="walk_animation":
   cp_path,_=cp_for_motion(m)
   if cp_path:walks.append((path,cp_path))
  else:slides.append((path,None))
 for path,cp_path in [*walks,*slides]:
  adm=qualify_native(path,cp_path,reused=True)
  if adm:return adm
 return None
def screen(aid,directory):
 source=directory/"source_preservation_owner093x_v1.json"
 if not source.exists():tool("check_dog_source_preservation.py",["--rigging",directory/"rigging.json","--output",source])
 if read(source)["status"]!="pass":return {"status":"source_preservation_fail","source_preservation":str(source)}
 output=directory/"cpu_check_owner093x_v1"
 if not (output/"checks.json").exists():
  if output.exists():return {"status":"incomplete_previous_screen","directory":str(output)}
  tool("check_static_dog_rig.py",["--inventory",directory/"inventory.json","--asset",aid,"--output",output,"--speed",".03","--period","1.6","--seconds","4.8","--blender","/data/jzy/blender/blender-4.5.13-linux-x64/blender","--motion-policy",OWNER093X])
 cp=read(output/"checks.json")
 if cp.get("status")=="failed":raise RuntimeError("checker_execution_failure:"+str(output)+":"+str(cp.get("error")))
 return {"directory":str(directory),"checks":str(output/"checks.json"),"status":cp["status"],"maximum_edge_growth":cp.get("maximum_edge_growth"),"frame_count":cp.get("frame_count"),"source_preservation":str(source),"surface_failures":cp.get("surface_checks",{}).get("relative_failures",[])[:1],"motion_failures":cp.get("checks",{}).get("failures",[])[:5]}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--asset");ap.add_argument("--screen-only",action="store_true");args=ap.parse_args()
 path=ROOT/"batch/owner093x_rechecks.json";targets=read(ROOT/"diagnostics/targets.json")["assets"]
 doc=read(path) if path.exists() else {"policy":OWNER093X,"started":stamp(),"assets":{}}
 coverage=read(ROOT/"batch/coverage.json")["assets"];legacy=read(ROOT/"batch/legacy_rechecks.json")["assets"]
 for target in targets:
  aid=target["asset_id"]
  if aid in doc["assets"] and doc["assets"][aid].get("finished"):continue
  base=existing(aid)
  if base and base["classification"]=="有动画":
   doc["assets"][aid]={"classification":"有动画","status":"existing_walk_requalified","admission":publish(aid,base),"finished":stamp(),"attempts":[]};atomic(path,doc);progress(aid+" 新动作门槛复判有动画；复用完整原生逐帧/真实反向负例，声学交量产。")
 for target in targets:
  aid=target["asset_id"]
  if args.asset and args.asset!=aid:continue
  if doc["assets"].get(aid,{}).get("finished"):continue
  rec=doc["assets"].setdefault(aid,{"asset_id":aid,"attempts":[],"started":stamp()});rec["state"]="screening";atomic(path,doc)
  winner=None;start=time.monotonic();old=signal.signal(signal.SIGALRM,budget);signal.alarm(900)
  try:
   for directory in candidates(aid,coverage.get(aid,{}),legacy.get(aid,{})):
    result=screen(aid,directory);rec["attempts"].append(result);atomic(path,doc)
    print("OWNER093X_SCREEN",aid,result["status"],directory,flush=True)
    if result["status"]=="cpu_motion_pass_native_pending":winner=directory;rec["winning_checks"]=result["checks"];break
    if len(rec["attempts"])>=2:
     prior=rec["attempts"][-2:]
     reasons=[(r.get("surface_failures") or r.get("motion_failures") or [r["status"]])[0] for r in prior]
     rec["two_failure_diagnosis"]={"evidence":[r.get("checks") for r in prior],"witnesses":reasons,"selected_policy":OWNER093X,"action":"stop repeating this failure route; preserve original mesh and qualified glide"}
     # Try another existing weight scheme only after recording concrete witnesses.
     atomic(path,doc)
  except BindingBudget:rec["timebox"]="900_second_existing_binding_rescreen";progress(aid+" 新门槛复判15分钟时间盒到，保留部分证据和原滑行。")
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);rec["screen_wall_s"]=time.monotonic()-start;gc.collect()
  if args.screen_only:
   rec["state"]="native_pending" if winner else "screened";atomic(path,doc);continue
  if winner:
   rec["state"]="native";atomic(path,doc)
   from tools.assets.validate_static_dog_native_bounded import run
   record=read(winner/"inventory.json")["assets"][0];out=ROOT/"validation"/aid/"walk_owner093x_habitat_v1"
   if not (out/"native_motion.json").exists():
    if out.exists():raise RuntimeError("incomplete prior native output; preserve instead of overwrite:"+str(out))
    run(record,read(ROOT/"inputs/house.json"),read(ROOT/"inputs/runtime.json"),read(ROOT/"inputs/route.json"),out,speed=.03,period=1.6,seconds=4.8,slots=8,mode="walk_animation",animation_attempts=rec["attempts"],motion_policy=OWNER093X)
   nm=read(out/"native_motion.json");rec["walking_native_motion"]=str(out/"native_motion.json");rec["walking_native_status"]=nm.get("status");rec["walking_native_failures"]=nm.get("checks",{}).get("failures",[])[:10];rec["walking_native_error"]=nm.get("error");atomic(path,doc)
   if nm.get("status")=="failed" and nm.get("failure_kind")=="execution_error":raise RuntimeError("native_execution_failure:"+str(out)+":"+str(nm.get("error")))
   adm=qualify_native(out/"native_motion.json",rec["winning_checks"])
  else:adm=None
  if adm is None:
   adm=existing(aid)
   if adm is None:adm={"asset_id":aid,"status":"blocked","classification":"不能移动","reason":"no_qualified_native_case","policy":OWNER093X,"per_asset_audio_required":False,"production_audio_screening_required":True}
   adm["binding_recheck_evidence"]=str(path);adm["binding_recheck_attempts"]=rec["attempts"]
  rec.update(classification=adm["classification"],admission=publish(aid,adm),state="finished",finished=stamp());atomic(path,doc);progress(aid+" 新动作门槛最终："+adm["classification"]+"；"+rec["admission"])
  print("OWNER093X_FINAL",aid,adm["classification"],flush=True);gc.collect()
 doc["finished"]=len([r for r in doc["assets"].values() if r.get("finished")])==len(targets);doc["updated"]=stamp();atomic(path,doc)
 print("OWNER093X_SUMMARY",{c:sum(r.get("classification")==c for r in doc["assets"].values()) for c in ("有动画","只能滑行","不能移动")},flush=True)
if __name__=="__main__":main()
