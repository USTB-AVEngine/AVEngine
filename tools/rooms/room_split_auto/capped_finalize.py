"""Immutable correction of implementation exceptions after native cap35 run."""
from pathlib import Path
import argparse,datetime,hashlib,json,multiprocessing,os,resource,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.rooms.room_split_auto.capped_split import worker,dump

def run(root,workers=8):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    root=Path(root);base=root/"delivery_all_v4"
    if not (base/"completed.json").exists():raise RuntimeError("native run incomplete")
    plan=json.loads((base/"revision_plan.json").read_text())
    failed={p.name for p in (base/"regions").glob("*.json") if json.loads(p.read_text()).get("traceback")}
    trial=root/"revision_cap35_20261009_v1/native_repair_v1";trial.mkdir(exist_ok=False);(trial/"regions").mkdir()
    jobs=[]
    for j in plan["jobs"]:
        sources=[s for s in j["sources"] if Path(s).name in failed]
        if sources:jobs.append(dict(j,sources=sources))
    dump(trial/"plan.json",dict(affected_sources=sorted(failed),jobs=jobs,
                               correction="gallery orthographic metadata adapter supplies sensor span_m; geometry/cut/placement thresholds unchanged"))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,j,root,trial,plan["parameters"],plan["selected_parameter"],plan["max_room_area_m2"],plan["overhead_cache"]):j["house"] for j in jobs}
        for f in as_completed(futures):
            try:receipts.append(f.result())
            except Exception as e:receipts.append(dict(house=futures[f],error=repr(e),traceback=traceback.format_exc()))
    dump(trial/"runtime_receipt.json",receipts)
    out=base/"final_v1";out.mkdir(exist_ok=False);(out/"regions").mkdir();(out/"rooms").mkdir()
    errors=[]
    for path in sorted((base/"regions").glob("*.json")):
        correction=trial/"regions"/path.name
        if path.name in failed and not correction.exists():raise RuntimeError("missing corrected source "+path.name)
        source=correction if correction.exists() else path
        d=json.loads(source.read_text())
        if d.get("traceback"):errors.append(dict(source=path.name,error=d.get("unresolved_reasons"),traceback=d["traceback"]))
        dump(out/"regions"/path.name,d)
        for b in d["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
    dump(out/"revision_plan.json",dict(plan,immutable_base_delivery=str(base),implementation_repair_sources=sorted(failed)))
    dump(out/"native_receipt.json",json.loads((base/"native_receipt.json").read_text())+receipts)
    dump(out/"completed.json",dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  source_regions=len(list((out/"regions").glob("*.json"))),workers=workers,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
                                  implementation_repair_sources=len(failed),remaining_implementation_exceptions=errors))
    files=[Path(__file__).with_name(n) for n in ["capped_split.py","capped_review.py","capped_finalize.py"]]
    dump(out/"algorithm_snapshot.json",dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                                           max_room_area_m2=35,corrected_sensor_metadata=True))
    print("V4_FINAL_NATIVE_DONE",len(failed),"REMAINING_ERRORS",len(errors),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--workers",type=int,default=8);v=a.parse_args();run(v.root,v.workers)
