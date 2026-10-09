"""Recompute only post-audit affected sources; preserve every preceding delivery."""
from pathlib import Path
import argparse,datetime,json,multiprocessing,os,resource,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.rooms.room_split_auto.connected_refine import worker
from tools.rooms.room_split_auto.connected_split import dump

def run(root,workers=4):
    resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3));root=Path(root)
    affected=json.loads((root/"revision_connectivity_20261009_v1/final_correction_sources_v1.json").read_text())
    plan=json.loads((root/"processing_plan_v1.json").read_text());parameter=json.loads((root/"selected_parameter_v1.json").read_text())["selected_parameter"]
    trial=root/"revision_connectivity_20261009_v1/final_correction_native_v1";trial.mkdir(exist_ok=False);(trial/"regions").mkdir()
    jobs={}
    for name in affected:
        old=json.loads((root/"delivery_all_v2/regions"/name).read_text());house=old["house"]
        if house not in jobs:
            j=next(x.copy() for x in plan["jobs"] if x["house"]==house);j.pop("rows",None);j["sources"]=[];jobs[house]=j
        jobs[house]["sources"].append(str(root/"delivery_all_v2/regions"/name))
    ordered=sorted(jobs.values(),key=lambda j:-sum(json.loads(Path(p).read_text())["source_floor_area_m2"] for p in j["sources"]))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        fut={pool.submit(worker,j,root,trial,plan["parameters"],parameter,50.,plan["overhead_cache"]):j["house"] for j in ordered}
        for f in as_completed(fut):
            try:receipts.append(f.result())
            except Exception as e:receipts.append(dict(house=fut[f],error=repr(e)))
    dump(trial/"runtime_receipt.json",receipts)
    out=root/"delivery_all_v3/final_v1";out.mkdir(exist_ok=False);(out/"regions").mkdir();(out/"rooms").mkdir()
    used=[]
    for path in (root/"delivery_all_v3/regions").glob("*.json"):
        corrected=trial/"regions"/path.name;source=corrected if corrected.exists() else path
        d=json.loads(source.read_text())
        if path.name in affected and not corrected.exists():raise RuntimeError("post-audit correction is missing: "+path.name)
        dump(out/"regions"/path.name,d)
        for b in d["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
        if corrected.exists():used.append(path.name)
    for name in ["revision_plan.json","completed.json"]:shutil.copyfile(root/"delivery_all_v3"/name,out/name)
    base_receipts=json.loads((root/"delivery_all_v3/native_receipt.json").read_text())
    dump(out/"native_receipt.json",base_receipts+receipts)
    dump(out/"post_audit_correction_receipt.json",dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_delivery=str(root/"delivery_all_v3"),new_delivery=str(out),corrected_sources=used,reason="full-scene furniture guidance and step/steps stair aliases; immutable older delivery retained",workers=workers,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0)))
    print("FINAL_NATIVE_READY",out,used,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--workers",type=int,default=4);a=p.parse_args();run(a.root,a.workers)
