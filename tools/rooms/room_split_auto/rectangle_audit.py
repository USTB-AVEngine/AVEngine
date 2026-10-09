"""Read-only audit of native floor shape and rectangular sampling scope."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,datetime,html,json,os,resource,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
import shapely
from shapely.geometry import shape,mapping,box
from tools.rooms.room_selection.measurements import navmesh_triangles
from tools.rooms.room_selection.geometry import union_projected_polygons
from tools.rooms.room_split_auto.pipeline import load_native

def dump(p,d):
    with Path(p).open("x") as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2)

def scope_metrics(ground,rectangle,nav,other_ground):
    walk=nav.intersection(rectangle);other=walk.intersection(other_ground).difference(ground)
    return dict(real_floor_area_m2=float(ground.area),bbox_area_m2=float(rectangle.area),
        floor_over_bbox_ratio=float(ground.area/rectangle.area) if rectangle.area else None,
        floor_outside_bbox_area_m2=float(ground.difference(rectangle).area),
        bbox_walkable_area_m2=float(walk.area),walkable_on_other_ground_m2=float(other.area),
        on_other_fraction_of_bbox_walkable=float(other.area/walk.area) if walk.area else None,
        walkable_outside_own_ground_m2=float(walk.difference(ground).area),
        ground_label_overlap_m2=float(walk.intersection(ground).intersection(other_ground).area))

def worker(job,root):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    start=time.time();inv=json.loads(Path(job["inventory"]).read_text());rows=inv["rows"];hs=load_native()
    pf,nav_polys,nav_ys=navmesh_triangles(hs,Path(rows[0]["navmesh_source"]))
    floors=[(r["room_label"],f["floor_y_m"],shape(f["floor_polygon"])) for r in rows for f in r["floors"]]
    output=[];errors=[];nav_cache={};other_cache={};legacy_cache={}
    for selection in job["selection"]:
        try:
            row=next(r for r in rows if r["room_label"]==selection["room_label"])
            fy=float(selection["floor_y_m"])
            eligible=[f for f in row["floors"] if abs(f["floor_y_m"]-fy)<=.3]
            g=union_projected_polygons([shape(f["floor_polygon"]) for f in eligible])
            measurement_path=Path(selection["result_path"]);production_scope_source=None
            if measurement_path.exists():
                admission=json.loads(measurement_path.read_text())
                admitted=next((x for x in admission.get("input",{}).get("acoustic_rooms",[]) if x["room_label"]==selection["room_label"]),None)
                if admitted and admitted.get("floor_polygon"):g=shape(admitted["floor_polygon"])
                production_scope_source=admitted.get("source_registry") if admitted else None
            if g.is_empty:raise ValueError("no admitted same-floor semantic ground")
            key=round(fy,7)
            if key not in nav_cache:
                nav_cache[key]=union_projected_polygons([nav_polys[int(i)] for i in np.flatnonzero(abs(nav_ys-fy)<=.3)])
            nav=nav_cache[key]
            other=union_projected_polygons([f for label,y,f in floors if label!=selection["room_label"] and abs(y-fy)<=.3])
            rect=box(*g.bounds);metrics=scope_metrics(g,rect,nav,other)
            item=dict(house=selection["house"],region=selection["room_label"],region_id=row["region_id"],
                floor_y_m=fy,selection_file=selection["selection_file"],listed_area_m2=float(selection["floor_area_m2"]),
                ground_measurement_source=str(measurement_path) if measurement_path.exists() else job["inventory"],
                semantic_source=row["semantic_source"],navmesh_source=row["navmesh_source"],
                floor_bbox_xz_m=[[g.bounds[0],g.bounds[1]],[g.bounds[2],g.bounds[3]]],
                **metrics)
            try:
                legacy_path=Path(production_scope_source or row["legacy_rooms_source"])
                if str(legacy_path).startswith("/data/smy/"):raise ValueError("forbidden original smy registry source; not read")
                if str(legacy_path) not in legacy_cache:legacy_cache[str(legacy_path)]=json.loads(legacy_path.read_text())
                legacy=legacy_cache[str(legacy_path)]
                reg=next((r for r in legacy.get("rooms",[]) if int(r["region_id"])==row["region_id"]),None)
                if reg and reg.get("bbox_xz_m"):
                    a,b=reg["bbox_xz_m"];prod=box(a[0],a[1],b[0],b[1]);prodmet=scope_metrics(g,prod,nav,other)
                    item.update(registry_bbox_xz_m=reg["bbox_xz_m"],registry_source=str(legacy_path))
                    item.update({"registry_"+k:v for k,v in prodmet.items()})
                else:item["registry_scope_unverified"]="no matching legacy registry region bbox"
            except Exception as e:
                item["registry_scope_unverified"]=repr(e)
            output.append(item)
        except Exception as e:errors.append(dict(house=selection["house"],region=selection["room_label"],error=repr(e),traceback=traceback.format_exc()))
    return dict(house=job["house"],rows=output,errors=errors,seconds=time.time()-start,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def distribution(values):
    v=np.array([x for x in values if x is not None],float)
    if not len(v):return dict(n=0)
    return dict(n=len(v),min=float(v.min()),p05=float(np.quantile(v,.05)),p25=float(np.quantile(v,.25)),median=float(np.median(v)),p75=float(np.quantile(v,.75)),p95=float(np.quantile(v,.95)),max=float(v.max()),mean=float(v.mean()),
        zero_count=int((v<=1e-8).sum()))

def run(root,workers=8):
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));root=Path(root);out=root/"rectangle_scope_audit_v1";out.mkdir(exist_ok=False)
    selection=json.loads((root/"size_gallery_30_50_v1/selection_rows_v1.json").read_text());unique={(r["house"],r["room_label"]):r for r in selection};byhouse={}
    for r in unique.values():byhouse.setdefault(r["house"],[]).append(r)
    jobs=[dict(house=h,selection=v,inventory=str(root/"inventory_v1"/(h+".json"))) for h,v in sorted(byhouse.items())]
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        fut={pool.submit(worker,j,root):j["house"] for j in jobs}
        for f in as_completed(fut):
            try:res=f.result();receipts.append(res);print("C_HOUSE",res["house"],len(res["rows"]),len(res["errors"]),flush=True)
            except Exception as e:receipts.append(dict(house=fut[f],rows=[],errors=[dict(error=repr(e))]))
    rows=sorted([r for rec in receipts for r in rec["rows"]],key=lambda r:(r["house"],int(r["region"][1:])))
    errors=[e for rec in receipts for e in rec["errors"]]
    metrics=["floor_over_bbox_ratio","on_other_fraction_of_bbox_walkable","walkable_on_other_ground_m2"]
    summary=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),requested_rooms=len(unique),measured_rooms=len(rows),unresolved=len(errors),registry_scope_measured_rooms=sum("registry_floor_over_bbox_ratio" in r for r in rows),houses=len(jobs),cpu_only=True,workers=workers,nice=10,max_address_space_gib=(workers+1)*6,
        distributions={k:distribution([r.get(k) for r in rows]) for k in metrics+["registry_"+k for k in metrics]},
        rooms_with_other_floor_ground=sum(r["walkable_on_other_ground_m2"]>1e-6 for r in rows),
        registry_rooms_with_other_floor_ground=sum(r.get("registry_walkable_on_other_ground_m2",0)>1e-6 for r in rows),
        methods=dict(floor="original holed admitted semantic-floor projection; no furniture subtraction added",rectangle="both floor envelope and current selection source_registry member-face AABB measured",
            navigation="exact union of native navmesh triangle horizontal projections within frozen +/-0.3m floor window; continuous area, not a random sampling probability",
            other_room="other region original same-floor ground; subtract own ground to avoid double-counting ambiguous labels",
            geometry_and_area_source="selection admission floor_polygon when available, otherwise same-floor inventory union",production="read-only code audit; no production change"))
    worst=dict(floor_bbox_lowest_ratio=sorted(rows,key=lambda r:r["floor_over_bbox_ratio"])[:20],floor_bbox_highest_other_fraction=sorted(rows,key=lambda r:-(r["on_other_fraction_of_bbox_walkable"] or 0))[:20],
        registry_bbox_lowest_ratio=sorted([r for r in rows if r.get("registry_floor_over_bbox_ratio") is not None],key=lambda r:r["registry_floor_over_bbox_ratio"])[:20],
        registry_bbox_highest_other_fraction=sorted([r for r in rows if r.get("registry_on_other_fraction_of_bbox_walkable") is not None],key=lambda r:-r["registry_on_other_fraction_of_bbox_walkable"])[:20])
    dump(out/"summary.json",summary);dump(out/"rooms.json",rows);dump(out/"worst20.json",worst);dump(out/"unresolved.json",errors)
    dump(out/"runtime_receipt.json",[{k:v for k,v in rec.items() if k not in ["rows","errors"]} for rec in receipts])
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/"rooms.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report="现有名单矩形取点范围包含其他语义房间的可走地面；以下为实测，不改生产取点或名单。\n\n"
    report+=f"覆盖 {len(rows)}/{len(unique)} 间、{len(jobs)} 套 HM3D；未判定 {len(errors)}。机器 48g，nice=10，CPU {workers} workers。\n\n"
    report+="口径：真实地面仍带洞；主指标为地面外接矩形，另列 selection result.input.acoustic_rooms.source_registry 的实际成员面范围框。navmesh 连续水平面积不等于生产随机取样概率，未验证楼层锁定之外的动态路线。\n\n"
    report+="来源：rooms.json / rooms.csv；分布 summary.json；四组最差20间 worst20.json。\n\n"
    for k,d in summary["distributions"].items():report+=f"{k}: n={d['n']}, min={d.get('min')}, median={d.get('median')}, p95={d.get('p95')}, max={d.get('max')}\n\n"
    for title,items in worst.items():
        report+=title+"\n\n| 房间 | 地面/框 | 框内其他房间可走面积 m² | 其他占比 |\n|---|---:|---:|---:|\n"
        prefix="registry_" if title.startswith("registry_") else ""
        for r in items:report+=f"| {r['house']}/{r['region']} | {r[prefix+'floor_over_bbox_ratio']:.4f} | {r[prefix+'walkable_on_other_ground_m2']:.3f} | {r[prefix+'on_other_fraction_of_bbox_walkable'] or 0:.4f} |\n"
        report+="\n"
    report+="代码：src/avengine/rooms/walkable_space.py:76–82（相机/声源 region 框内取点），:98–111（camera_grid 框），:69–74（完整navmesh shortest_path）。几何、生产名单、声学包均只读。\n"
    with (out/"REPORT_zh.md").open("x") as f:f.write(report)
    with (root/"PROGRESS_zh.md").open("a") as f:f.write("\nC_READY "+str(out/"REPORT_zh.md")+"；量测="+str(len(rows))+"/"+str(len(unique))+"，未判定="+str(len(errors))+"。\n")
    print("C_DONE",len(rows),len(errors),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--workers",type=int,default=8);v=a.parse_args();run(v.root,v.workers)
