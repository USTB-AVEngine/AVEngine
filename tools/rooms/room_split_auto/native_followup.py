"""Native HM3D connectivity cleanup and cap35 draft; never edits production lists."""
from __future__ import annotations
import argparse,copy,csv,datetime,html,json,math,multiprocessing,os,resource,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
import shapely
from shapely.geometry import shape,mapping,Point,Polygon,GeometryCollection
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.measurements import navmesh_triangles,load_scene
from tools.rooms.room_selection.navigation import sample_navigation,placement
from tools.rooms.room_split_auto.pipeline import load_native,raw_collision,nav_scope_at,measured_black,finalize_interfaces,adjacency
from tools.rooms.room_split_auto.seam_connectivity import Connectivity,exterior
from tools.rooms.room_split_auto import walkable_split as w,capped_split as c,capped_review as cr,revision_review as rr,connected_split as cs
from tools.rooms.room_split_auto.size_gallery import render
from tools.rooms.room_split_auto.contours import body_width
dump=cs.dump
PREP=w.PREP

def historical_witness(room):
    path=Path(room["source_list_record"]["result_path"])
    d=json.loads(path.read_text())
    found=next(x for x in d["input"]["acoustic_rooms"] if x["room_label"]==room["room_label"])
    return dict(found["placement"],validation_provenance=str(path),reused_frozen_witness=True)

def width_scope(raw,ctx):
    """Diagnostic: remove >=6m2 structural voids; do not let scan speckles dominate."""
    env=ctx.envelope(raw);holes=[]
    for part in polygons(raw):
        for ring in part.interiors:
            h=Polygon(ring)
            if h.area>=6:holes.append(h)
    return env.difference(shapely.union_all(holes)),holes

def native_block(room,g,index,decision,witness,black,reasons,unresolved,ctx,main=False):
    return dict(schema="native_room_connectivity_cap35_v1",
        id=room["house"]+"__"+room["room_label"]+"__"+room["selected_floor_id"]+f"__E{index:03d}",
        house=room["house"],source_region=room["room_label"],source_region_id=room["region_id"],
        floor_id=room["selected_floor_id"],floor_y_m=room["floor_y_m"],
        floor_polygon_xz_m=mapping(g),floor_area_m2=float(g.area),short_side_m=short_side(g),
        black_fraction=black["black_fraction"],black_measurement=black,placement_witness=witness,
        decision=decision,discard_reasons=reasons,unresolved_reasons=unresolved,
        room_type=room["room_type"],new_room=not main,native_main=main,
        source="original" if main else "native_detached_candidate",
        source_selection=room["source_list_name"],origin_list=("strict" if "strict" in room["source_list_name"] else "review band"),
        connectivity_certificate=ctx.certificate(g),original_room_id=room["room_id"],
        area_method=room["floor_area_method"],acoustics="not_run",leakage=None,
        inherited_original_circle_gate=False,inherited_original_visibility_gate=False)

def witness_in(g,witness):
    return bool(witness.get("found")) and all(g.buffer(1e-7).covers(Point(witness[k][0],witness[k][2])) for k in ("camera_m","source_1_m","source_2_m"))

def check_piece(room,g,index,ctx,mesh,pf,hs,p,frame,old_witness,main):
    reasons=[];unknown=[]
    if g.area<6-1e-8:reasons.append("DETACHED_FRAGMENT")
    if g.area>35+1e-8:unknown.append("NATIVE_CAP_REPLACEMENT_REQUIRED")
    if short_side(g)<2.4-1e-8:reasons.append("SHORT_SIDE_BELOW_2_4")
    if ctx.count(g)!=1:unknown.append("CONNECTIVITY_0_6_UNRESOLVED")
    black=measured_black(g,room["floor_y_m"],frame,p)
    if black["black_fraction"] is None:unknown.append("SCAN_BLACK_FRACTION_UNVERIFIED")
    elif black["black_fraction"]>.15:reasons.append("SCAN_BLACK_FRACTION_ABOVE_15_PERCENT")
    # Re-run the frozen CPU placement search for every changed >=6m2 component.
    if g.area>=6-1e-8:
        _,points,clearance,adj,_=sample_navigation(pf,hs,g,room["floor_y_m"],p)
        inside=np.flatnonzero(shapely.contains_xy(g,points[:,0],points[:,2]))
        witness=placement(mesh,points,clearance,adj,p,inside.tolist())
        witness["reused_frozen_witness"]=False
        if not witness["found"]:reasons.append("PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET")
    else:witness=dict(found=False,not_run_reason="DETACHED_FRAGMENT")
    decision="discard" if reasons else "unresolved" if unknown else "retain"
    return native_block(room,g,index,decision,witness,black,reasons,unknown,ctx,main)

def old_minimum_recovery(g,outputs):
    saved=[];short_saved=[]
    for rawpart in sorted(polygons(g),key=lambda x:-x.area):
        if rawpart.area>=6 and short_side(rawpart)>=2.4:continue
        overlap=sum(rawpart.intersection(shape(b["floor_polygon_xz_m"])).area for b in outputs if b["decision"]=="retain")
        if overlap>1e-8:
            if rawpart.area<6:saved.append(overlap)
            else:short_saved.append(overlap)
    return sum(saved),sum(short_saved)

def worker(job,root,out,params):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    start=time.time();root=Path(root);out=Path(out);hs=load_native()
    pf,np_,ny=navmesh_triangles(hs,Path(job["navmesh"]));mesh=None;objects=None;rows=[];errors=[]
    for room in job["rooms"]:
        try:
            g=shape(room["floor_polygon_xz_m"]);fy=room["floor_y_m"]
            nav=nav_scope_at(np_,ny,fy,exterior(g).buffer(.3),params)
            ctx=Connectivity(nav);record=ctx.record(g)
            widths,holes=width_scope(g,ctx)
            metrics=dict(room_id=room["room_id"],house=room["house"],room_label=room["room_label"],
                floor_area_m2=g.area,listed_area_m2=room["listed_floor_area_m2"],short_side_m=short_side(g),
                raw_parts=len(list(polygons(g))),direct_groups=record["direct_group_count"],
                opened_components=ctx.count(g),old_raw_component_areas_m2=sorted([q.area for q in polygons(g)],reverse=True),
                new_direct_component_areas_m2=[q.area for q in ctx.groups(g)],
                body_width_m=body_width(widths),raw_holed_body_width_m=body_width(g),
                filled_body_width_m=body_width(ctx.envelope(g)),
                preserved_large_void_count=len(holes),preserved_large_void_area_m2=sum(h.area for h in holes),
                width_method="existing medial body_width; seam-joined exterior, scan holes filled except original voids>=6m2; raw and fully-filled widths also reported; diagnostic only",
                floor_y_m=fy,source_list_name=room["source_list_name"])
            affected=ctx.count(g)!=1
            outputs=[];cuts=[]
            if g.area>35+1e-8:
                status="replaced_by_v5_cap_cut";affected=False
            elif not affected:
                witness=historical_witness(room);fraction=float(room["source_list_record"]["black_fraction"])
                legacy=[]
                if g.area<6-1e-8:legacy.append("HISTORICAL_ADMISSION_BELOW_6")
                if short_side(g)<2.4-1e-8:legacy.append("HISTORICAL_ADMISSION_SHORT_SIDE_BELOW_2_4")
                if fraction>.15:legacy.append("HISTORICAL_ADMISSION_BLACK_ABOVE_15_PERCENT")
                if not witness.get("found"):legacy.append("HISTORICAL_ADMISSION_WITNESS_MISSING")
                pending=bool(legacy)
                # Preserve historic admissions and explicitly flag old <6m2 exceptions.
                status="historical_frozen_threshold_exception" if pending else "unchanged"
                block=native_block(room,g,0,"pending_legacy" if pending else "retain",witness,
                    dict(black_fraction=fraction,source=room["source_list_record"].get("visual_path"),method="inherited unchanged original ground visual measurement"),
                    [],legacy,ctx,True)
                outputs=[block]
            else:
                if mesh is None:mesh,_=raw_collision(job["scene_directory"]);objects=load_scene(job["scene_directory"])
                w.install(ctx);furniture=cs.furniture_for(objects,room["region_id"],fy,params,g)
                pieces,cuts=c.connected_parts(g,furniture,c.main_axis(g));pieces=sorted(pieces,key=lambda x:-x.area)
                old_witness=historical_witness(room);frame=frame_for_native(room,root,out)
                for i,piece in enumerate(pieces):
                    outputs.append(check_piece(room,piece,i,ctx,mesh,pf,hs,params,frame,old_witness,i==0))
                status="connectivity_cleaned" if all(x["decision"]!="unresolved" for x in outputs) else "partially_unresolved"
            # Exact old-rule loss saved by the new joins: <6m2 raw islands now in
            # a retained native main/component. Larger old fragments were merely
            # candidates, so their entire area must not be called lost.
            small_saved,short_saved=old_minimum_recovery(g,outputs)
            metrics.update(status=status,affected=affected,
                old_detached_fragment_area_now_retained_m2=small_saved,
                old_short_side_fragment_area_now_retained_m2=short_saved,
                old_frozen_geometric_fragment_area_now_retained_m2=small_saved+short_saved,
                old_separation_avoided=metrics["raw_parts"]>1 and record["direct_group_count"]==1)
            region=dict(house=room["house"],source_region=room["room_label"],source_region_id=room["region_id"],source_floor_area_m2=float(g.area),source_geometry=dict(
                house=room["house"],room_label=room["room_label"],region_id=room["region_id"],floors=[dict(floor_id=room["selected_floor_id"],floor_y_m=fy,floor_area_m2=g.area,floor_polygon=mapping(g))]),
                requires_split=affected,status=status,blocks=outputs,cut_lines=[],metrics=metrics,
                source_native_polygon_manifest=str(PREP),floor_overheads={room["selected_floor_id"]:native_frame_reference(room,root,out)})
            for i,(line,m,kind,stage,n,*rest) in enumerate(cuts):
                region["cut_lines"].append(dict(id=f"E_L{i:03d}",floor_id=room["selected_floor_id"],type=kind,stage=stage,line_xz_m=list(line.coords),line_geometry_xz_m=mapping(line),segment_count=len(line.coords)-1,**m))
            finalize_interfaces(region["blocks"],region["cut_lines"]);adjacency(region["blocks"],region["cut_lines"])
            dump(out/"regions"/(room["house"]+"__"+room["room_label"]+".json"),region)
            for b in outputs:dump(out/"rooms"/(b["id"]+".json"),b)
            rows.append(metrics)
            print("NATIVE_CONNECTIVITY",room["room_id"],status,"parts",len(outputs),flush=True)
        except Exception as exc:
            errors.append(dict(room_id=room["room_id"],error=repr(exc),traceback=traceback.format_exc()))
    return dict(house=job["house"],rows=rows,errors=errors,seconds=time.time()-start,
                pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def native_frame_reference(room,root,out):
    candidate=[]
    for folder in (out/"raw",root/"size_gallery_30_50_v1/raw"):
        for path in folder.glob(room["house"]+"__*.json"):
            meta=json.loads(path.read_text())
            if abs(meta["floor_y_m"]-room["floor_y_m"])<=.05:
                candidate.append((abs(meta["floor_y_m"]-room["floor_y_m"]),path,meta))
    if not candidate:return None
    _,path,meta=min(candidate,key=lambda x:x[0])
    return dict(metadata_path=str(path),image_path=meta["path"],source="whole-house real CPU llvmpipe")
def frame_for_native(room,root,out):
    ref=native_frame_reference(room,root,out)
    if ref is None:return None
    from PIL import Image
    meta=json.loads(Path(ref["metadata_path"]).read_text())
    meta=dict(meta,images=[dict(path=meta["path"],projection=meta["projection"],span_m=meta["span_m"])])
    return meta,meta["images"][0],Image.open(ref["image_path"]).convert("RGB"),ref

def prepare(root):
    root=Path(root);out=root/"existing_rooms_connectivity_v1";out.mkdir(exist_ok=False)
    for folder in ("raw","regions","rooms","media"): (out/folder).mkdir()
    rooms=json.loads(PREP.read_text())["rooms"];jobs={};renders={}
    inventory={p.stem:json.loads(p.read_text()) for p in (root/"inventory_v1").glob("*.json")}
    for room in rooms:
        house=room["house"]
        if house not in jobs:jobs[house]=dict(house=house,scene_directory=room["scene_directory"],navmesh=room["navmesh_source"],rooms=[])
        jobs[house]["rooms"].append(room)
        if native_frame_reference(room,root,out):continue
        if house not in renders:
            geoms=[shape(f["floor_polygon"]) for row in inventory[house]["rows"] for f in row["floors"]]
            x0,z0,x1,z1=shapely.union_all(geoms).bounds
            renders[house]=dict(house=house,scene_directory=room["scene_directory"],bounds=[x0,z0,x1,z1],
                cx=(x0+x1)/2,cz=(z0+z1)/2,span=max(x1-x0,z1-z0)*1.12+1,floors=[])
        floors=renders[house]["floors"]
        if not any(abs(x["y"]-room["floor_y_m"])<=.05 for x in floors):
            floors.append(dict(key=f"Y{len(floors):03d}",y=room["floor_y_m"]))
    dump(out/"plan.json",dict(native_rooms=len(rooms),jobs=list(jobs.values()),render_jobs=list(renders.values()),
        max_room_area_m2=35,parameters=json.loads((root/"processing_plan_v1.json").read_text())["parameters"],
        source_native_polygon_manifest=str(PREP),native_gt35_delegated_to_v5=True,
        original_subcap_no_circle_or_visibility_gate=True,cpu_only=True))
    print("NATIVE_PREPARED",len(rooms),"HOUSES",len(jobs),"MISSING_FRAMES",sum(len(j["floors"]) for j in renders.values()),flush=True)

def run(root,workers=8):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    root=Path(root);out=root/"existing_rooms_connectivity_v1";plan=json.loads((out/"plan.json").read_text())
    v5=root/"delivery_all_v5/final_v1"
    if not (v5/"validation.json").exists() or not json.loads((v5/"validation.json").read_text())["passed"]:raise RuntimeError("v5 acceptance required before native followup")
    render_receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(render,j,out/"raw"):j["house"] for j in plan["render_jobs"]}
        for f in as_completed(futures):
            try:render_receipts.append(f.result())
            except Exception as exc:render_receipts.append(dict(house=futures[f],error=repr(exc),traceback=traceback.format_exc()))
    dump(out/"render_receipt.json",render_receipts)
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,j,root,out,plan["parameters"]):j["house"] for j in plan["jobs"]}
        for f in as_completed(futures):
            try:results.append(f.result())
            except Exception as exc:results.append(dict(house=futures[f],rows=[],errors=[dict(error=repr(exc),traceback=traceback.format_exc())]))
    rows=[x for r in results for x in r["rows"]]
    dump(out/"runtime_receipt.json",[{k:v for k,v in r.items() if k not in ("rows",)} for r in results])
    dump(out/"metrics.json",rows);dump(out/"completed.json",dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        native_room_count=len(rows),expected_native_room_count=plan["native_rooms"],errors=[e for r in results for e in r["errors"]],
        pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),workers=workers,max_address_space_gib=8*(workers+1)))
    print("NATIVE_CONNECTIVITY_DONE",len(rows),"ERRORS",sum(len(r["errors"]) for r in results),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--prepare",action="store_true");a.add_argument("--workers",type=int,default=8);v=a.parse_args()
    if v.prepare:prepare(v.root)
    else:run(v.root,v.workers)
