"""Qualify queued native dogs with fresh RIRs and a 45 minute asset timebox."""
import argparse,datetime,fcntl,gc,json,os,pathlib,runpy,sys,time
import numpy as np
ROOT=pathlib.Path("/data/jzy/tmp/claude_hardbank_20260923/v7_rig_dogs_20261014")
WT=pathlib.Path(__file__).resolve().parents[2]
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
 tmp=path.with_name(path.name+".tmp."+str(os.getpid()));tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def progress(message):
 with (ROOT/"PROGRESS.md").open("a") as f:f.write("\n"+stamp()+" "+message+"\n")
def queue(change=None):
 path=ROOT/"batch/audio_queue.json"
 with (ROOT/"batch/queue.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);data=json.loads(path.read_text()) if path.exists() else {"items":[]}
  if change:change(data);atomic(path,data)
  return data
def update(path,fields):
 def f(data):
  row=next(x for x in data["items"] if x["native_motion"]==str(path));row.update(fields)
 queue(f)
def tool(name,args):
 saved=sys.argv;sys.argv=[str(WT/"tools/assets"/name),*map(str,args)]
 try:runpy.run_path(sys.argv[0],run_name="__main__");return 0
 except SystemExit as e:return int(e.code or 0)
 except Exception:
  import traceback;traceback.print_exc();return 70
 finally:sys.argv=saved;gc.collect()
def cores():
 jobs=json.loads((ROOT/"JOBS.json").read_text())["jobs"]
 running=any(x["name"].startswith("dog_coverage_batch") and x["status"]=="running" for x in jobs)
 return "101,102" if running else "101,102,103"
def refinement_diagnosis(previous):
 p=previous/"progress.json"
 if not p.exists():return {"permitted":False,"reason":"no_interval_evidence"}
 report=json.loads(p.read_text());failed=report.get("failed_intervals",[]);rows=[]
 for interval in failed:
  mid=round(interval["midpoint_s"]*16000);files=[previous/f"reference_{mid:06d}_{i}.npy" for i in range(3)]
  if not all(f.exists() for f in files):return {"permitted":False,"reason":"three_fresh_references_missing"}
  waves=[np.load(f,allow_pickle=False) for f in files];same=all(np.array_equal(waves[0],w) for w in waves[1:])
  rows.append({"midpoint_s":interval["midpoint_s"],"references_samplewise_equal":same,"failure_metrics":interval["qualification"]["failures"],"interval_s":interval["end_s"]-interval["start_s"]})
 return {"permitted":bool(rows) and all(r["references_samplewise_equal"] and r["interval_s"]>1/60 for r in rows),"failed_intervals":rows,"action":"promote only identical measured midpoints; rerun three independent references at new midpoints; unchanged gates"}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--maximum-rounds",type=int,default=3);a=p.parse_args()
 completed_idle=0
 while True:
  items=queue()["items"];row=next((r for r in items if r["state"]=="pending"),None)
  if row is None:
   coverage=ROOT/"batch/coverage.json";c=json.loads(coverage.read_text()) if coverage.exists() else {}
   if c.get("binding_and_geometry_finished"):
    progress("移动RIR待验队列已处理；全部结果在batch/audio_queue.json，失败项未准入。");break
   time.sleep(10);continue
  motion=pathlib.Path(row["native_motion"]);m=json.loads(motion.read_text());aid=m["asset_id"];start=time.monotonic()
  update(motion,{"state":"running","started":stamp(),"asset_id":aid,"motion_mode":m["motion_mode"]})
  endpoint=motion.parent/"audio_endpoints_v1";common=["--motion",motion,"--house",ROOT/"inputs/house.json","--runtime",ROOT/"inputs/runtime.json"]
  endpoint_code=tool("prepare_animal_audio_endpoints.py",[*common,"--output",endpoint,"--cores",cores()])
  rounds=[];audio=None;last=None
  if endpoint_code==0:
   previous=endpoint
   for number in range(1,a.maximum_rounds+1):
    if time.monotonic()-start>2700:
     update(motion,{"timebox":"45_minute_audio_asset_budget"});break
    output=motion.parent/f"audio_v{number}"
    flags=["--refine-failed"] if number>1 else []
    rc=tool("qualify_animal_audio_parallel.py",[*common,"--previous",previous,"--output",output,"--cores",cores(),*flags])
    last=output/"audio.json";result=json.loads(last.read_text()) if last.exists() else {"status":"execution_failed"}
    rounds.append({"round":number,"returncode":rc,"audio_json":str(last),"status":result["status"]})
    if result.get("status")=="pass":audio=last;break
    diag=refinement_diagnosis(output);atomic(output/"diagnosis.json",diag)
    progress(aid+" RIR两个失败区间已停并诊断："+json.dumps(diag,ensure_ascii=False)[:600])
    if not diag["permitted"]:break
    previous=output
  if audio:
   from avengine.assets.animal_motion.service import completed_checks
   result=json.loads(audio.read_text());joint=completed_checks(m,result,m["motion_mode"])
   physical=m.get("native_direction_negative_controls",{}).get("status")=="pass"
   passed=joint["status"]=="pass" and physical
   admission={"asset_id":aid,"species":"dog","status":"pass" if passed else "blocked","classification":"有动画并通过" if passed and m["motion_mode"]=="walk_animation" else "只能滑行" if passed else "未准入","motion_mode":m["motion_mode"],"joint_animation_present":m["motion_mode"]=="walk_animation","reason":None if m["motion_mode"]=="walk_animation" else "automatic_binding_attempts_rejected_or_timeboxed","animation_attempts":m.get("animation_attempts",[]),"native_motion":str(motion),"audio":str(audio),"combined_checks":joint,"native_physically_rotated_controls_pass":physical,"thresholds_changed":False,"completed":stamp()}
   atomic(motion.parent/"admission.json",admission);update(motion,{"state":"passed" if passed else "blocked","admission":str(motion.parent/"admission.json"),"audio":str(audio),"rounds":rounds,"finished":stamp()})
   progress(aid+" "+admission["classification"]+"；联合验收="+str(motion.parent/"admission.json"))
  else:
   update(motion,{"state":"blocked","endpoint_returncode":endpoint_code,"rounds":rounds,"last_audio":str(last) if last else None,"finished":stamp(),"reason":"moving_audio_not_qualified_within_timebox","fallback_geometry_needed":m["motion_mode"]=="walk_animation"})
   progress(aid+" 移动RIR未准入，保留原生几何结果并转下一只；"+str(last))
if __name__=="__main__":main()
