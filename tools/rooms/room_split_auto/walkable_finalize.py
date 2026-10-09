"""Immutable second pass for broad-corridor construction failures in v5."""
from pathlib import Path
import argparse,datetime,hashlib,json,multiprocessing,os,resource,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.rooms.room_split_auto import walkable_split as w
dump=w.dump

def key(reg):
    return (sum(b["floor_area_m2"] for b in reg["blocks"] if b["decision"]=="unresolved"),
            -sum(b["floor_area_m2"] for b in reg["blocks"] if b["decision"]=="retain"),
            sum(c.get("furniture_intersection_length_m",0) for c in reg["cut_lines"]))

def run(root,workers=4):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    root=Path(root);base=root/"delivery_all_v5"
    if not (base/"completed.json").exists():raise RuntimeError("first native v5 pass incomplete")
    plan=json.loads((base/"revision_plan.json").read_text())
    originals={p.name:json.loads(p.read_text()) for p in (base/"regions").glob("*.json")}
    wanted={name for name,r in originals.items() if r["requires_split"] and any(
        any(reason in b.get("unresolved_reasons",[]) for reason in ("CORRIDOR_WIDE_PART_AXIS_SEPARATION_UNRESOLVED","NO_AXIS_CUT_WITH_2_4_M_DISKS","AXIS_CUT_CPU_BUDGET_UNRESOLVED","NATIVE_HOUSE_JOB_FAILURE","V5_PROCESS_FAILURE")) for b in r["blocks"])}
    trial=root/"revision_seam_nav_v5_20261009_v1/corridor_refinement_v1";trial.mkdir(exist_ok=False);(trial/"regions").mkdir()
    jobs=[]
    for j in plan["jobs"]:
        sources=[s for s in j["sources"] if Path(s).name in wanted]
        if sources:jobs.append(dict(j,sources=sources,corridor_refine=True))
    dump(trial/"plan.json",dict(affected_sources=sorted(wanted),jobs=jobs,
        refinement="corridor-only finite wall-to-wall candidate lookahead up to room scale2.4m; exact narrow discard width still<1.5; two-axis-leg vertices added; no frozen threshold changed"))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(w.worker,j,root,trial,plan["parameters"],plan["selected_parameter"],35,plan["overhead_cache"]):j["house"] for j in jobs}
        for f in as_completed(futures):
            try:receipts.append(f.result())
            except Exception as exc:receipts.append(dict(house=futures[f],error=repr(exc),traceback=traceback.format_exc()))
    dump(trial/"runtime_receipt.json",receipts)
    out=base/"final_v1";out.mkdir(exist_ok=False);(out/"regions").mkdir();(out/"rooms").mkdir()
    selections=[];errors=[]
    for name,initial in sorted(originals.items()):
        candidate=trial/"regions"/name;refined=json.loads(candidate.read_text()) if candidate.exists() else None
        choose=refined is not None and not refined.get("traceback") and key(refined)<key(initial)
        d=refined if choose else initial
        if d.get("traceback"):errors.append(dict(source=name,reasons=d.get("unresolved_reasons"),traceback=d["traceback"]))
        dump(out/"regions"/name,d)
        for b in d["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
        if name in wanted:selections.append(dict(source=name,chosen_refinement=choose,initial_quality=key(initial),refined_quality=key(refined) if refined else None))
    dump(out/"revision_plan.json",dict(plan,immutable_initial_pass=str(base),corridor_refinement_sources=sorted(wanted),
        corridor_refinement_selection=selections,final_regions_source_map={name:("corridor_refinement" if any(s["source"]==name and s["chosen_refinement"] for s in selections) else "initial_v5") for name in originals}))
    dump(out/"native_receipt.json",json.loads((base/"native_receipt.json").read_text())+receipts)
    dump(out/"completed.json",dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_regions=len(originals),workers=workers,
        remaining_implementation_exceptions=errors,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),correction_selection=selections))
    dump(out/"algorithm_snapshot.json",dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py") if p.name.startswith(("walkable","seam","capped","native"))},
        pipeline="initial v5 corridor lookahead1.5/uniform2legs, then unresolved broad-corridor refinement lookahead2.4/boundary vertices; selection favors less unresolved area, more retained area, then less furniture crossing",
        frozen_cap35=True,global_minimum_unverified=True))
    print("V5_FINAL_NATIVE_DONE","REFINED",sum(s["chosen_refinement"] for s in selections),"ERRORS",len(errors),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--workers",type=int,default=4);v=a.parse_args();run(v.root,v.workers)
