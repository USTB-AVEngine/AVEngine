"""Independent full-scene furniture audit of actual holed-ground cut interfaces."""
from pathlib import Path
import argparse,csv,json,multiprocessing,os,resource,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import shapely
from shapely.geometry import shape,LineString,GeometryCollection
from tools.rooms.room_selection.measurements import load_scene
from tools.rooms.room_screening.geometry import shape_preserving_projected_footprint
from tools.rooms.room_split_auto.connected_split import cut_measure,FurnitureList,dump

def linear_only(g):
    if g.geom_type=="LineString":return g
    if hasattr(g,"geoms"):return shapely.union_all([linear_only(p) for p in g.geoms])
    return GeometryCollection()

def worker(job):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));started=time.time()
    scene=load_scene(job["scene"]);cache={};rows=[]
    for path in job["regions"]:
        reg=json.loads(Path(path).read_text());floors={f["floor_id"]:f for f in reg["source_geometry"]["floors"]}
        for cut in reg["cut_lines"]:
            fid=cut["floor_id"];fy=floors[fid]["floor_y_m"];ground=shape(floors[fid]["floor_polygon"]);key=round(fy,8)
            if key not in cache:
                items=[]
                for instance in scene.instances:
                    category=instance["category"]
                    if instance["role"]!="blocker" or category in ("floor","step","steps") or any(t in category for t in ["wall","ceiling","door","window","stair"]):continue
                    footprint=shape_preserving_projected_footprint(instance["triangles"],fy,2.4)
                    if not footprint.is_empty:items.append(dict(instance_id=instance["instance_id"],region_id=instance["region_id"],category=category,geometry=footprint))
                cache[key]=items
            items=cache[key];allf=FurnitureList(items);known=FurnitureList([x for x in items if x["region_id"] in (reg["source_region_id"],-1)])
            actual=linear_only(shape(cut["final_interface_geometry_xz_m"])).intersection(ground)
            allm=cut_measure(actual,ground,allf);same=cut_measure(actual,ground,known)
            design=LineString(cut["line_xz_m"])
            upper=cut_measure(design,ground,allf)
            # Original cut scope is the final holed ground interface. The upper
            # bound deliberately also includes source portions outside the cut's
            # former parent, and must not be interpreted as the actual cut area.
            band=actual.buffer(.125,cap_style="flat",join_style="mitre").intersection(ground)
            allarea=band.intersection(allf.union).area
            rows.append(dict(source_region=reg["house"]+"/"+reg["source_region"],floor_id=fid,cut_id=cut["id"],type=cut["type"],designed_straight_segment_count=len(cut["line_xz_m"])-1,active=cut["active_in_final_partition"],actual_holed_ground_interface_length_m=actual.length,full_scene_furniture_intersection_length_m=allm["furniture_intersection_length_m"],full_scene_cut_strip_original_ground_intersection_area_m2=allarea,cut_strip_width_m=.25,known_source_actual_interface_furniture_length_m=same["furniture_intersection_length_m"],additional_other_region_furniture_length_m=max(0.,allm["furniture_intersection_length_m"]-same["furniture_intersection_length_m"]),original_applied_known_source_design_length_m=cut["furniture_intersection_length_m"],original_applied_known_source_design_strip_area_m2=cut["furniture_intersection_area_m2"],full_source_design_upper_bound_all_scene_furniture_length_m=upper["furniture_intersection_length_m"],full_source_design_upper_bound_all_scene_strip_area_m2=upper["furniture_intersection_area_m2"],actual_interface_furniture_intersections=allm["furniture_intersections"],furniture_projection="all scene known semantic blocker instances; canonical shape-preserving 2.4m slab footprint; no source-region filtering",floor_polygon_measurement="actual interface clipped to original holed semantic ground",global_minimum_furniture_crossing_unverified=True))
    return dict(house=job["house"],rows=rows,seconds=time.time()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0))

def run(root,workers=2,delivery_dir="delivery_all_v3"):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));root=Path(root);out=root/delivery_dir/"cut_scene_furniture_audit_v1";out.mkdir(exist_ok=False)
    jobs={}
    for p in (root/delivery_dir/"regions").glob("*.json"):
        reg=json.loads(p.read_text())
        if not reg["requires_split"] or not reg["cut_lines"]:continue
        h=reg["house"]
        if h not in jobs:jobs[h]=dict(house=h,scene=reg["source_geometry"]["scene_directory"],regions=[])
        jobs[h]["regions"].append(str(p))
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        fut={pool.submit(worker,j):j["house"] for j in jobs.values()}
        for f in as_completed(fut):
            try:r=f.result();results.append(r);print("SCENE_FURNITURE",r["house"],len(r["rows"]),flush=True)
            except Exception as e:results.append(dict(house=fut[f],rows=[],error=repr(e)))
    rows=sorted([x for r in results for x in r["rows"]],key=lambda x:(x["source_region"],x["cut_id"]))
    summary=dict(cut_count=len(rows),designed_max_segments=max((x["designed_straight_segment_count"] for x in rows),default=0),actual_interfaces_crossing_all_scene_furniture=sum(x["full_scene_furniture_intersection_length_m"]>1e-7 for x in rows),cuts_with_additional_other_region_furniture=sum(x["additional_other_region_furniture_length_m"]>1e-7 for x in rows),errors=[r for r in results if "error" in r],global_minimum_unverified=True,method="all scene canonical semantic furniture footprints; original holed ground final actual interfaces; .25m strip area and exact line length both reported; full source designed line is an explicitly separate upper bound")
    dump(out/"summary.json",summary);dump(out/"cut_intersections.json",rows);dump(out/"runtime_receipt.json",[{k:v for k,v in r.items() if k!="rows"} for r in results])
    fields=[k for k in rows[0] if k!="actual_interface_furniture_intersections"] if rows else ["cut_id"]
    with (out/"cut_intersections.csv").open("x",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    report="切线的全场景家具复量已完成；实际切线使用原始带洞地面上的最终接口。\n\n"
    report+=f"切线 {len(rows)}，设计最多 {summary['designed_max_segments']} 段；实际接口与全场景家具相交 {summary['actual_interfaces_crossing_all_scene_furniture']} 条，因其他region家具额外相交 {summary['cuts_with_additional_other_region_furniture']} 条；错误 {len(summary['errors'])}。逐条出处cut_intersections.json/csv。\n\n"
    report+="原算法为同region及未分区的已知家具提示，此复量使用整场景全部已知语义家具，不按region过滤。家具投影仍用原共享shape-preserving后端，2.4m体高切片仅作切线家具审计。实际接口剪在原带洞地面；面积为0.25m宽条带与原地面、家具的交集，另列精确线穿家具长度。\n\n"
    report+="full_source_design_upper_bound_* 是设计线延长至整个来源区域后的保守上界，可能包含原切分子块外的地面，不应与实际切线相交面积混为一谈。\n\n"
    report+="家具相交的数学全局最小值未验证；若有额外跨region相交，本轮没有再调门槛或改生产，报告保留这些例外供owner审核。\n"
    with (out/"REPORT_zh.md").open("x") as f:f.write(report)
    print("SCENE_FURNITURE_DONE",summary,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--workers",type=int,default=2);p.add_argument("--delivery-dir",default="delivery_all_v3");a=p.parse_args();run(a.root,a.workers,a.delivery_dir)
