"""Timeboxed static-dog binding coverage and native walking/slide triage."""
import argparse,copy,gc,json,pathlib,runpy,sys,time,datetime,fcntl,signal,os
ROOT=pathlib.Path("/data/jzy/tmp/claude_hardbank_20260923/v7_rig_dogs_20261014")
WT=pathlib.Path(__file__).resolve().parents[2]
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
 tmp=path.with_name(path.name+".tmp."+str(os.getpid()));tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def progress(message):
 with (ROOT/"PROGRESS.md").open("a") as f:f.write("\n"+stamp()+" "+message+"\n")
def tool(name,args):
 saved=sys.argv;sys.argv=[str(WT/"tools/assets"/name),*map(str,args)]
 try:
  runpy.run_path(sys.argv[0],run_name="__main__");return 0
 except SystemExit as e:return int(e.code or 0)
 except Exception as e:
  import traceback;traceback.print_exc();return 70
 finally:sys.argv=saved;gc.collect()
def enqueue(native):
 path=ROOT/"batch/audio_queue.json"
 with (ROOT/"batch/queue.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  data=json.loads(path.read_text()) if path.exists() else {"items":[]}
  if not any(x["native_motion"]==str(native) for x in data["items"]):
   data["items"].append({"native_motion":str(native),"state":"pending","created":stamp()})
  atomic(path,data)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--skip",default="v7_dog_labrador_black_20261012");a=p.parse_args()
 targets=json.loads((ROOT/"diagnostics/targets.json").read_text())["assets"]
 house=json.loads((ROOT/"inputs/house.json").read_text());runtime=json.loads((ROOT/"inputs/runtime.json").read_text());route=json.loads((ROOT/"inputs/route.json").read_text())
 from tools.assets.validate_static_dog_native_bounded import run as native
 statepath=ROOT/"batch/coverage.json"
 state=json.loads(statepath.read_text()) if statepath.exists() else {"schema":"static_dogs_timeboxed_coverage_v1","targets":25,"assets":{},"binding_max_seconds_per_asset":2700,"maximum_weight_recipes":3,"speed_m_s":.03,"period_s":1.6,"seconds":4.8}
 for target in targets:
  aid=target["asset_id"]
  if aid==a.skip or state["assets"].get(aid,{}).get("state") in ("audio_queued","cannot_move_in_test_room"):continue
  rec={"asset_id":aid,"started":stamp(),"state":"binding","animation_attempts":[],"classification":"pending"}
  state["assets"][aid]=rec;atomic(statepath,state);begin=time.monotonic();success=None;same_failure=0;previous_reason=None
  def timeout(signum,frame):
   import psutil
   parent=psutil.Process()
   owned=[(x,x.create_time(),x.cmdline()) for x in parent.children(recursive=True)]
   for x,created,cmd in reversed(owned):
    try:
     if x.create_time()==created and x.cmdline()==cmd:x.terminate()
    except psutil.Error:pass
   raise TimeoutError("per_asset_binding_45_minute_timebox")
  old_handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(2700)
  try:
   for index,method in enumerate(("envelope","blender_envelope","heat"),1):
    if time.monotonic()-begin>=2700:break
    directory=ROOT/"derived"/aid/(method+"_v1")
    if directory.exists():raise RuntimeError("unexpected existing attempt directory; do not overwrite")
    args=["--targets",ROOT/"diagnostics/targets.json","--asset",aid,"--template",ROOT/"template/Dog.blend","--blender","/data/jzy/blender/blender-4.5.13-linux-x64/blender","--weights",method,"--motion-amplitude",".03","--output",directory]
    code=tool("build_static_dog_rig.py",args)
    attempt={"method":method,"sequence":index,"binding_returncode":code,"directory":str(directory),"started":stamp()}
    rec["animation_attempts"].append(attempt);atomic(statepath,state)
    if code:
     attempt.update(status="binding_failed",reason="target_binding_failed")
    else:
     preservation=directory/"source_preservation.json"
     preserve_code=tool("check_dog_source_preservation.py",["--rigging",directory/"rigging.json","--output",preservation])
     attempt["source_preservation"]=str(preservation)
     if preserve_code:attempt.update(status="rejected",reason="source_appearance_or_geometry_changed")
     else:
      output=directory/"cpu_check_v1"
      code=tool("check_static_dog_rig.py",["--inventory",directory/"inventory.json","--asset",aid,"--output",output,"--speed",".03","--period","1.6","--seconds","4.8","--blender","/data/jzy/blender/blender-4.5.13-linux-x64/blender"])
      check=json.loads((output/"checks.json").read_text()) if (output/"checks.json").exists() else {}
      reason=check.get("surface_failure") or check.get("deformation_failure") or str(check.get("checks",{}).get("failures") or check.get("error") or "execution_without_receipt")
      attempt.update(status=check.get("status","execution_failed"),checks=str(output/"checks.json"),reason=None if code==0 else reason)
      if code==0 and check.get("status")=="cpu_motion_pass_native_pending":success=directory;break
    reason=attempt.get("reason");same_failure=same_failure+1 if reason==previous_reason else 1;previous_reason=reason
    if same_failure>=2:
     rec["diagnosis"]={"cause":reason,"evidence":[x.get("checks",x["directory"]) for x in rec["animation_attempts"][-2:]],"action":"stop_this_failure_route_and_fall_back; no third identical retry"}
     progress(aid+" 同因两次，停止该绑定路线；诊断证据="+str(rec["diagnosis"]["evidence"]))
     break
  except TimeoutError as e:rec["timebox_reason"]=str(e);progress(aid+" 绑定45分钟时间盒到期，转滑行")
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
  attemptsfile=ROOT/"batch"/(aid+"_attempts.json");atomic(attemptsfile,rec["animation_attempts"])
  rec["binding_elapsed_s"]=time.monotonic()-begin;rec["state"]="native";atomic(statepath,state)
  modes=[("walk_animation",success)] if success else []
  modes.append(("slide_no_animation",None))
  for mode,directory in modes:
   record=json.loads((directory/"inventory.json").read_text())["assets"][0] if directory else copy.deepcopy(target["inventory_record"])
   output=ROOT/"validation"/aid/(mode+"_habitat_v1")
   result=native(record,house,runtime,route,output,speed=.03,period=1.6,seconds=4.8,slots=8,mode=mode,animation_attempts=rec["animation_attempts"])
   rec.setdefault("native_attempts",[]).append({"mode":mode,"status":result["status"],"native_motion":str(output/"native_motion.json"),"error":result.get("error"),"checks":result.get("checks")})
   atomic(statepath,state)
   if result.get("status")=="geometry_pass_audio_pending":
    rec.update(state="audio_queued",motion_mode=mode,classification="有动画并通过（音频待验）" if mode=="walk_animation" else "只能滑行（音频待验）",native_motion=str(output/"native_motion.json"),reason=None if mode=="walk_animation" else "binding_checks_rejected_or_timeboxed")
    enqueue(output/"native_motion.json");break
   if result.get("failure_kind")=="execution_error":
    rec.update(state="execution_error",classification="未完成",reason=result.get("error"));break
  else:rec.update(state="cannot_move_in_test_room",classification="不能移动",reason="original_native_geometry_checks_rejected_in_this_HM3D_room")
  rec["finished"]=stamp();atomic(statepath,state);progress(aid+" "+rec["classification"]+"；native="+str(rec.get("native_motion",rec.get("reason"))))
  gc.collect()
 state["binding_and_geometry_finished"]=stamp();atomic(statepath,state);progress("25狗绑定/几何覆盖队列完成（首只单独滑行作业另记），音频队列继续独立验收。")
if __name__=="__main__":main()
