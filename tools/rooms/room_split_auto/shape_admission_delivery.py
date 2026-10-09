"""Apply native shape admission to scoped source rooms and preserve other bytes."""
from __future__ import annotations
import argparse,copy,csv,datetime,hashlib,json,multiprocessing,os,resource,time,traceback,faulthandler
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import shapely
from shapely.geometry import shape
from tools.rooms.room_selection.measurements import load_scene,navmesh_triangles
from tools.rooms.room_split_auto.pipeline import load_native,raw_collision
from tools.rooms.room_split_auto.shape_quality_geometry import defects,wrap_count
from tools.rooms.room_split_auto.shape_quality_repair import process,enforce_room_disk_admission,room_disk_certificate,wide_body_candidates,prepare_wide_body_recovery
from tools.rooms.room_split_auto.shape_quality_delivery import same_layout,construction_lost,canonical_interfaces,csv_records
from tools.rooms.room_split_auto.native_finalize import copy_record

def dump(p,x):
    with Path(p).open("x") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def get(p):return json.loads(Path(p).read_text())
def key(r):return r["house"]+"/"+r["source_region"]
def quality(reg):
    kept=[b for b in reg["blocks"] if b["decision"]=="retain"]
    counts=dict(NECK=0,CORRIDOR=0,WRAP=0,FURNITURE=0,NODISK=0,WIDE_DROPPED=0)
    for b in kept:
        g=shape(b["floor_polygon_xz_m"]);d=defects(g)
        counts["NECK"]+=d["neck_count"];counts["CORRIDOR"]+=d["corridor_count"]
        counts["NODISK"]+=not room_disk_certificate(g)["fits"]
    for fid in {b["floor_id"] for b in kept}:
        counts["WRAP"]+=wrap_count([shape(b["floor_polygon_xz_m"]) for b in kept if b["floor_id"]==fid])
    counts["FURNITURE"]=sum(c.get("active_in_final_partition",True) and (c.get("furniture_intersection_length_m") or 0)>.5 for c in reg["cut_lines"])
    counts["WIDE_DROPPED"]=sum(len(wide_body_candidates(shape(b["floor_polygon_xz_m"]))) for b in reg["blocks"] if b["decision"]=="discard" and "STAIRS" not in b.get("discard_reasons",[]))
    return counts
def fallback(old,reason):
    r=copy.deepcopy(old);r["disk_admission_audit"]=enforce_room_disk_admission(r["blocks"])
    r["shape_admission_attempt_failure"]=reason
    r["retained_new_rooms"]=sum(b["decision"]=="retain" for b in r["blocks"])
    r["status"]="processed_with_local_disk_discard"
    return r
def worker(job,parameters,root,out,admission_only=False):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    assert 10<=os.getpriority(os.PRIO_PROCESS,0)<=15
    started=time.time();receipts=[]
    try:
        hs=load_native();pf,np_,ny=navmesh_triangles(hs,Path(job["navmesh"]))
        mesh,ray=raw_collision(job["scene_directory"]);objects=load_scene(job["scene_directory"])
        setup_error=None
    except Exception:
        setup_error=traceback.format_exc();ray=None
    for filename in job["sources"]:
        old=get(filename);t=time.time();error=setup_error
        if error:r=fallback(old,error)
        else:
            faulthandler.dump_traceback_later(90,repeat=True)
            try:
                working=old
                if admission_only:
                    from tools.rooms.room_split_auto import connected_split as cs
                    working=fallback(old,"bounded recovery after previous complete boundary search failed")
                    by_floor={x["floor_id"]:cs.furniture_for(objects,old["source_region_id"],x["floor_y_m"],parameters,shape(x["floor_polygon"])) for x in old["source_geometry"]["floors"]}
                    working,audit=prepare_wide_body_recovery(working,by_floor)
                    working["post_disk_wide_body_recovery_pass"]=True
                    working["post_disk_wide_body_recovery_audit"]=audit
                r=process(working,mesh,pf,hs,np_,ny,objects,parameters,root)
                lost=construction_lost(old,r)
                if lost:
                    error=dict(reason="previous admitted floor became unresolved; keep its valid cells",lost=lost)
                    r=fallback(old,error)
            except Exception:
                error=traceback.format_exc();r=fallback(old,error)
            finally:faulthandler.cancel_dump_traceback_later()
        dump(Path(out)/"regions"/Path(filename).name,r)
        receipts.append(dict(source=key(old),seconds=time.time()-t,error=error,before=quality(old),after=quality(r),retained=sum(b["decision"]=="retain" for b in r["blocks"]),disk_discard=r.get("disk_admission_audit",[])))
        print("SHAPE_ADMISSION_SOURCE",key(old),round(time.time()-t,2),json.dumps(receipts[-1]["after"]),flush=True)
    return dict(house=job["house"],pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-started,sources=receipts,ray_receipt=ray,cpu_only=True,address_space_limit_gib=8)
def run_attempt(root,baseline,out,plan,parameters,workers,admission_only=False):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    assert 1<=workers<=7
    out.mkdir();(out/"regions").mkdir()
    sources=set(plan["six_sources"]+[plan["optional_source"]]);jobs={}
    for p in (baseline/"regions").glob("*.json"):
        r=get(p)
        if key(r) not in sources:continue
        row=r["source_geometry"]
        j=jobs.setdefault(r["house"],dict(house=r["house"],scene_directory=row["scene_directory"],navmesh=row["navmesh_source"],sources=[]))
        j["sources"].append(str(p))
    assert sum(len(j["sources"]) for j in jobs.values())==len(sources)
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,j,parameters,str(root),str(out),admission_only):j for j in jobs.values()}
        for f in as_completed(futures):
            try:results.append(f.result())
            except Exception:
                j=futures[f];error=traceback.format_exc();results.append(dict(house=j["house"],error=error))
                for source in j["sources"]:
                    dest=out/"regions"/Path(source).name
                    if not dest.exists():dump(dest,fallback(get(source),error))
    dump(out/"native_receipt.json",results)
    dump(out/"completed.json",dict(workers=workers,compute_processes_max=workers+2,tree_address_space_upper_gib=8*(workers+2),cpu_only=True,nice=os.getpriority(os.PRIO_PROCESS,0)))
def assemble(root,baseline,attempt,out,plan,draft_path):
    out.mkdir();(out/"regions").mkdir();(out/"rooms").mkdir()
    required=set(plan["six_sources"]);optional=plan["optional_source"];changes=[];sha=[];selected={};counts={};rooms=[]
    for p in sorted((baseline/"regions").glob("*.json")):
        old=get(p);source=key(old);candidate=None;reason="outside authorized sources"
        if source in required or source==optional:
            candidate=get(attempt/"regions"/p.name);reason="authorized native shape admission"
            if source==optional:
                before=quality(old);after=quality(candidate)
                if same_layout(old,candidate) or sum(after.values())>=sum(before.values()):
                    candidate=None;reason="optional retry did not improve a physical partition; keep v6 bytes"
        if candidate is not None:
            new=canonical_interfaces(candidate);changes.append(dict(source=source,before=quality(old),after=quality(new),reason=reason))
            dump(out/"regions"/p.name,new)
            for b in new["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
        else:
            new=old;copied=[(p,out/"regions"/p.name)]
            copied += [(baseline/"rooms"/(b["id"]+".json"),out/"rooms"/(b["id"]+".json")) for b in old["blocks"]]
            for a,b in copied:
                import shutil
                shutil.copyfile(a,b)
                sha.append(dict(source=source,relative_path=str(b.relative_to(out)),v6_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),v6b_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),identical=a.read_bytes()==b.read_bytes()))
        selected[p.name]=new
        if new.get("requires_split"):
            rooms+=new["blocks"]
            q=quality(new)
            for k,n in q.items():counts[k]=counts.get(k,0)+n
    assert len({b["id"] for r in selected.values() for b in r["blocks"]})==sum(len(r["blocks"]) for r in selected.values())
    kept=[b for b in rooms if b["decision"]=="retain"]
    summary=dict(retained_rooms=len(kept),retained_area_m2=sum(b["floor_area_m2"] for b in kept),quality_counts=counts,source_regions=len(selected),split_sources=sum(r.get("requires_split",False) for r in selected.values()),changed_sources=len(changes),unchanged_source_regions=len({x["source"] for x in sha}),unchanged_files=len(sha),all_unchanged_bytes_identical=all(x["identical"] for x in sha),unresolved=[b["id"] for b in rooms if b["decision"]=="unresolved"],acoustics="not_run",gpu="not_used",production_integrated=False,lists_modified=False)
    dump(out/"summary.json",summary);dump(out/"source_changes.json",changes);dump(out/"unchanged_sha256.json",sha)
    csv_records(out/"unchanged_sha256.csv",sha,["source","relative_path","v6_sha256","v6b_sha256","identical"])
    csv_records(out/"rooms_summary.csv",rooms,["id","house","source_region","floor_id","floor_y_m","floor_area_m2","short_side_m","nav_walkable_area_m2","black_fraction","decision","discard_reasons","unresolved_reasons","placement_witness","floor_polygon_xz_m"])
    dump(out/"no_disk_discard.json",[b for b in rooms if b["decision"]=="discard" and "NO_2_4M_DISK" in b.get("discard_reasons",[])])
    dump(out/"recovered_rooms.json",[b for b in kept if b.get("recovered_from_discard_id")])
    # Native 775 records are a direct value copy from the last approved draft.
    previous=get(root/"final_room_list_draft_v2/room_list_draft_v2.json")
    native=[r for r in previous["rooms"] if r["source"]!="new_cut"];assert len(native)==775
    source_info={(r["house"],r["source_region"]):r for r in previous["rooms"] if r["source"]=="new_cut"}
    added=[]
    for b in kept:
        r=copy_record(b);r["source"]="new_cut";original=source_info.get((b["house"],b["source_region"]),{})
        r["origin_list"]=original.get("origin_list","new");r["source_selection"]=original.get("source_selection")
        added.append(r)
    records=native+added;assert len({r["id"] for r in records})==len(records)
    draft_path.mkdir()
    dump(draft_path/"room_list_draft_v3.json",dict(schema="HM3D_room_list_draft_v3_cap35",rooms=records,production_integrated=False,leakage="pending separate acoustics task"))
    csv_records(draft_path/"rooms.csv",records,["id","house","source_region","source","floor_area_m2","short_side_m","floor_polygon_xz_m","placement_witness","origin_list","source_selection","leakage"])
    pending=[b for b in get(root/"final_room_list_draft_v2/pending.json") if b.get("decision")=="pending_legacy"]+[b for b in rooms if b["decision"]=="unresolved"]
    dump(draft_path/"pending.json",pending)
    dump(draft_path/"summary.json",dict(total_rooms=len(records),native_rooms=775,new_cut_rooms=len(added),total_area_m2=sum(r["floor_area_m2"] for r in records),native_records_identical_to_v2=native==[r for r in previous["rooms"] if r["source"]!="new_cut"],pending=len(pending),acoustics="not_run",production_integrated=False,lists_modified=False))
    print("SHAPE_ADMISSION_DELIVERY",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--baseline",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--plan",type=Path,required=True);p.add_argument("--parameters",type=Path,required=True);p.add_argument("--workers",type=int,default=4);p.add_argument("--assemble",action="store_true");p.add_argument("--attempt",type=Path);p.add_argument("--draft",type=Path);p.add_argument("--admission-only",action="store_true")
    args=p.parse_args();plan=get(args.plan)
    if args.assemble:assemble(args.root,args.baseline,args.attempt,args.output,plan,args.draft)
    else:run_attempt(args.root,args.baseline,args.output,plan,get(args.parameters)["parameters"],args.workers,args.admission_only)
