#!/usr/bin/env python3
"""Preflight admitted Kujiale rooms using audited native LOD geometry.

Produces placement witnesses, not episode admission or question truth.
No UE, audio generator, or GPU work is started by this command.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry, import_vertices, rotation, camera_candidates
from avengine.qa.kujiale_asset_paws import landmark_contacts
from avengine.capture.kujiale_room_inputs import prepare_room_inputs, admission_floor_reference, floor_reference_for_planning

COARSE = ("human", "dog", "cat", "bird_with_perch",
          "floor_object", "table_object", "wall_object", "ceiling_object")

def read(path):
    return json.loads(Path(path).read_text())

def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("completed evidence retained: " + str(path))
    temporary=path.with_name(path.name+".pending."+str(os.getpid())+"."+str(time.time_ns()))
    try:
        with temporary.open("x") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.write("\n")
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

def category(asset):
    if asset["category"] == "bird":
        return "bird_with_perch"
    if asset["category"] != "object":
        return asset["category"]
    return asset["attachment_surface"] + "_object"

def groups(assets):
    result = defaultdict(list)
    for asset in assets:
        key = ("animal." + asset["category"] if asset["category"] != "object"
               else "object." + asset["object_type"] + "." + asset["attachment_surface"])
        result[key].append(asset)
    return dict(sorted(result.items()))

def corners(low, high):
    return np.array([[x,y,z] for x in (low[0],high[0])
                     for y in (low[1],high[1]) for z in (low[2],high[2])])

def complete_camera(geometry, vertices, placement, polygon):
    """Check full envelope in frame and clear native rays at actual candidate cameras."""
    from shapely.geometry import Point
    world = vertices @ rotation(placement["yaw_rad"]).T + placement["root_m"]
    low,high = world.min(0),world.max(0); target=(low+high)/2
    bound = corners(low,high)
    result=[]; tangent=math.tan(math.radians(85)/2)
    for view in camera_candidates(vertices,placement,polygon):
        eye=np.array(view["position_m"]); floor_y=placement["support"]["floor_y_m"]
        try:
            support=geometry.floor.sample([eye[0],floor_y+.01,eye[2]])
        except ValueError:
            continue
        # The eye must occupy clear space; the supporting floor alone is allowed below.
        camera_box=corners(np.array([-.05,0.,-.05]),np.array([.05,.10,.05]))
        collision=geometry.collision(camera_box,eye-[0,.05,0],0.,[0,1,0],.005)
        if collision["status"]!="pass":
            continue
        yaw,pitch=map(math.radians,[view["yaw_deg"],view["pitch_deg"]])
        forward=np.array([math.cos(pitch)*math.cos(yaw),math.sin(pitch),math.cos(pitch)*math.sin(yaw)])
        side=np.array([-math.sin(yaw),0.,math.cos(yaw)]);up=np.cross(side,forward)
        delta=bound-eye;distance=delta@forward
        if np.any(distance<=.15):
            continue
        if np.max(np.abs(delta@side/(distance*tangent)))>.96 or np.max(np.abs(delta@up/(distance*tangent*9/16)))>.96:
            continue
        d=target-eye;length=float(np.linalg.norm(d))
        if geometry.ray(eye,d/length,length-.005) is not None:
            continue
        # Support contact corners can coincide with the support surface itself.
        clear=True
        for point in bound:
            ray=point-eye;length=float(np.linalg.norm(ray))
            if geometry.ray(eye,ray/length,max(0.,length-.015)) is not None:
                clear=False;break
        if clear:
            view=deepcopy(view);view.update(native_LOD_camera_clearance_m=.05,
                full_envelope_inside_frame=True,center_and_eight_envelope_corner_rays_clear=True,
                camera_floor_identity=deepcopy(geometry.floor.last_support_readback))
            result.append(view)
            if len(result)==3:
                return result
    return result

def bounded_supports(rows, maximum=64):
    unique={}
    for row in rows:
        key=tuple(round(float(x),2) for x in row["point_m"])+tuple(row["normal"])
        unique.setdefault(key,row)
    rows=list(unique.values())
    if len(rows)<=maximum:
        return rows
    # Cover the entire sampled room rather than only the old source neighbourhood.
    return [rows[i] for i in np.linspace(0,len(rows)-1,maximum).astype(int)]

def representative_order(rows):
    ordered=sorted(rows,key=lambda a:float(np.prod(a["dimensions_m"])),reverse=True)
    choices=[ordered[0]]
    if ordered[-1]["asset_id"]!=ordered[0]["asset_id"]:
        choices.append(ordered[-1])
    if len(ordered)>2:
        choices.append(ordered[len(ordered)//2])
    seen=set();result=[]
    for row in choices:
        shape=tuple(round(float(x),4) for x in row["dimensions_m"])
        if shape not in seen:
            result.append(row);seen.add(shape)
    return result

def find_witness(geometry, polygon, asset, vertices, catalog, limit, cell_seconds):
    start=time.monotonic();surface=asset["attachment_surface"]
    supports=bounded_supports(catalog[surface],limit);failures=defaultdict(int);tested=0
    if not supports:
        return None,{"reason":"no_sampled_"+surface+"_support",
                     "claim":"current native LOD sampling found no support; physical impossibility is not proven",
                     "tested_supports":0,"failures":{}}
    tolerance=.02 if asset["category"]=="human" else (.015 if asset["category"]=="object" else .012)
    for support in supports:
        if time.monotonic()-start>cell_seconds:
            failures["bounded_search_timebox_reached"]+=1;break
        if asset.get("object_type")=="air_conditioner" and surface=="wall" and support["point_m"][1]-support["floor_y_m"]<2.2:
            failures["wall_air_conditioner_support_below_2_2m"]+=1;continue
        normal=np.array(support["normal"])
        angles=[math.atan2(-normal[2],normal[0])] if surface=="wall" else [0.,math.pi/2]
        for yaw in angles:
            tested+=1
            place=geometry.placement(vertices,support,yaw,tolerance)
            if place is None:
                failures["attachment_or_native_LOD_envelope_collision"]+=1;continue
            if surface=="floor":
                try:
                    actual=geometry.floor.sample(support["point_m"])
                except ValueError as error:
                    failures["support_is_not_measured_room_floor"]+=1;continue
                place["verified_room_floor_identity"]=deepcopy(geometry.floor.last_support_readback)
                place["floor_height_error_m"]=abs(float(actual[1]-geometry.floor.room_floor_y_m))
            paws=landmark_contacts(geometry,asset,vertices,place) if asset["category"] in ("dog","cat","bird") else {"status":"not_applicable"}
            if paws["status"]=="fail":
                failures["animal_landmark_contact"]+=1;continue
            views=complete_camera(geometry,vertices,place,polygon)
            if len(views)<3:
                failures["fewer_than_three_clear_complete_CPU_views"]+=1;continue
            world=vertices@rotation(yaw).T+place["root_m"]
            place.update(camera_candidates=views,paw_contacts=paws,
                         native_world_bounds_m=[world.min(0).tolist(),world.max(0).tolist()])
            return place,{"tested_support_yaw_pairs":tested,"elapsed_seconds":time.monotonic()-start,
                          "failures_before_witness":dict(failures)}
    return None,{"reason":"no_qualified_witness_in_bounded_native_LOD_search",
                 "claim":"only this tested asset, sampled supports and search limit are rejected",
                 "tested_support_yaw_pairs":tested,"sampled_support_count":len(supports),
                 "elapsed_seconds":time.monotonic()-start,"failures":dict(failures)}

def run(args):
    root=args.root.resolve()
    if os.nice(0)<10:
        raise ValueError("room jobs require nice >=10")
    free=os.statvfs(root);free=free.f_bavail*free.f_frsize
    memory=next(int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
    if free<150_000_000_000 or memory<40_000_000_000:
        raise ValueError("insufficient task disk or available memory")
    output=args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("room evidence must stay inside the authorized pilot output")
    runtime=read(args.runtime);manifest=read(args.assets)["assets"];native={a["asset_id"]:a for a in read(args.native_assets)["assets"]}
    category_groups=groups(manifest);human=read(args.human_geometry)
    floor_lineages=read(args.floor_lineages) if args.floor_lineages else {}
    human_asset={"asset_id":human["asset_id"],"category":"human","attachment_surface":"floor",
                 "dimensions_m":human["dimensions_m"],"silent_decor":False}
    category_groups={"human.standing":[human_asset],**category_groups}
    vertices_cache={}
    for house in runtime["houses"]:
        if house["split"]!="train":
            raise ValueError("room legality is restricted to admitted train houses")
        geometry=None
        for room in house["checked_rooms"]:
            label=house["house"]+"_"+room["room_label"];dest=output/(label+".json")
            if dest.exists():
                print("RESUME_ROOM",label,flush=True);continue
            saved=read(args.catalogs/(label+".json"))
            declaration=deepcopy(saved["native_floor_surface"])
            declaration.update(floor_lineages.get(house["house"],{}))
            if geometry is None:
                geometry=NativeSupportGeometry(declaration)
            room_inputs=prepare_room_inputs(room,cpu_floor_reference=saved["floor_reference"])
            floor=geometry.floor.validate_probes(floor_reference_for_planning(room_inputs))
            from shapely.geometry import shape
            plan=read(house["input"]["ue_config"]["acceptance_plan_path"])
            polygon=shape(next(p["floor_polygon"] for p in plan["rooms"] if p["room_id"]==room["room_label"]))
            record={"schema":"kujiale_room_category_CPU_legality_v1",
                    "house":house["house"],"room_id":room["room_label"],"split":"train",
                    "ue_map_path":house["input"]["ue_map_path"],"native_room_admission":house["native_qa_result"],
                    "original_native_floor_admission":admission_floor_reference(room_inputs),
                    "CPU_planning_floor_reference":deepcopy(saved["floor_reference"]),
                    "floor_support_check":floor,"native_floor_surface":declaration,
                    "source_catalog":str(args.catalogs/(label+".json")),
                    "support_candidate_counts":{s:len(rows) for s,rows in saved["catalog"].items()},
                    "category_checks":[],"native_UE_render":{"status":"not_run"},
                    "scope":"tested representatives and single-asset placement witnesses; simultaneous placement, action, audio and question gates remain mandatory"}
            for key,rows in category_groups.items():
                cell={"category_key":key,"coarse_category":category(rows[0]),
                      "attachment_surface":rows[0]["attachment_surface"],"available_individuals":len(rows),
                      "status":"unavailable_in_bounded_search","tested_assets":[]}
                for asset in representative_order(rows):
                    ident=asset["asset_id"]
                    if ident not in vertices_cache:
                        if asset["category"]=="human":
                            vertices_cache[ident]=np.array(human["vertices_local_m"])
                        else:
                            receipt=args.imports/(ident+".json")
                            vertices_cache[ident]=import_vertices(read(receipt))
                    vertices=vertices_cache[ident]
                    relative=np.abs(np.ptp(vertices,axis=0)-asset["dimensions_m"])/np.maximum(asset["dimensions_m"],1e-5)
                    if relative.max()>.15:
                        cell["tested_assets"].append({"asset_id":ident,"status":"fail_native_size","relative_errors":relative.tolist()});continue
                    place,search=find_witness(geometry,polygon,asset,vertices,saved["catalog"],args.maximum_supports,args.cell_seconds)
                    tested={"asset_id":ident,"dimensions_m":np.ptp(vertices,axis=0).tolist(),
                            "size_relative_errors":relative.tolist(),"search":search,
                            "status":"pass_CPU_native_LOD_witness" if place else "not_found"}
                    cell["tested_assets"].append(tested)
                    if place:
                        cell.update(status="available_CPU_native_LOD_witness",representative_asset_id=ident,
                                    placement=place,dimensions_m=tested["dimensions_m"],
                                    ue_asset_paths=native[ident]["ue_asset_paths"] if asset["category"]!="human" else human["ue_asset_paths"],
                                    geometry_source=str(args.imports/(ident+".json")) if asset["category"]!="human" else human["geometry_source"],
                                    emitter_local_m=asset.get("emitter_local_m"),silent_decor=asset.get("silent_decor",False))
                        break
                if cell["status"]!="available_CPU_native_LOD_witness":
                    cell["reason"]=cell["tested_assets"][-1]["search"]["reason"] if cell["tested_assets"] and "search" in cell["tested_assets"][-1] else "no_size_qualified_representative"
                record["category_checks"].append(cell)
                print("CELL",label,key,cell["status"],flush=True)
            record["coarse_categories"]={coarse:{"status":"available_CPU_native_LOD_witness" if any(c["status"]=="available_CPU_native_LOD_witness" for c in record["category_checks"] if c["coarse_category"]==coarse) else "unavailable_in_bounded_search",
                "available_category_keys":[c["category_key"] for c in record["category_checks"] if c["coarse_category"]==coarse and c["status"]=="available_CPU_native_LOD_witness"],
                "rejected_category_keys":[c["category_key"] for c in record["category_checks"] if c["coarse_category"]==coarse and c["status"]!="available_CPU_native_LOD_witness"]} for coarse in COARSE}
            publish(dest,record)
            with (root/"PROGRESS.md").open("a") as stream:
                stream.write("- "+time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())+"："+label+" 房间×类别CPU原生LOD合法性完成；"+str(sum(c["status"]=="available_CPU_native_LOD_witness" for c in record["category_checks"]))+"/"+str(len(record["category_checks"]))+"细类有实测摆放证据，本轮UE出图待测。\n")
            print("ROOM_DONE",label,flush=True)
        del geometry;gc.collect()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","runtime","assets","native-assets","human-geometry","catalogs","imports","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--floor-lineages",type=Path)
    p.add_argument("--maximum-supports",type=int,default=48)
    p.add_argument("--cell-seconds",type=float,default=30.)
    run(p.parse_args())

if __name__=="__main__":
    main()
