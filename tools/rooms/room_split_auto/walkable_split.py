"""CPU v5 cut construction with walkable seams and bounded fallback directions."""
from __future__ import annotations
import argparse,copy,datetime,json,math,multiprocessing,os,resource,time,traceback
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import shapely
from shapely.geometry import shape,mapping,LineString,Point
from shapely.affinity import rotate
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.measurements import load_scene,navmesh_triangles
from tools.rooms.room_split_auto import capped_split as c,connected_split as cs
from tools.rooms.room_split_auto.seam_connectivity import Connectivity,exterior
from tools.rooms.room_split_auto.pipeline import nav_scope_at,load_native,raw_collision,finalize_interfaces,adjacency
from tools.rooms.room_split_auto.scene import structural_instances
BASE_CANDIDATES=c.candidates
BASE_PROCESS=c.process
BASE_SOLVE=c.Cutter.solve
BASE_ROOM_CIRCLE=c.room_circle
CORRIDOR_JUNCTION_SEARCH_M=1.5
CORRIDOR_ELBOW_VERTEX_REFINEMENT=False
dump=c.dump
PREP=Path("/data/jzy/tmp/claude_hardbank_20260923/room_polygon_placement_20261009/hm3d_prep_v1/polygon_manifest/manifest.json")

@lru_cache(maxsize=512)
def efficient_room_circle(g):
    """Certified existence test; avoid optimizing the radius of every bad trial."""
    ext=c.outline(g);points=[]
    if not ext.is_empty:
        points=[ext.centroid,ext.representative_point()]
    for shrink in (1.2,1.2-2e-7):
        core=ext.buffer(-shrink,join_style=1)
        if not core.is_empty:points.extend(p.representative_point() for p in polygons(core) if not p.is_empty)
        if any(not q.is_empty and ext.covers(q) and ext.boundary.distance(q)>=1.2-1e-7 for q in points):break
    points=[q for q in points if not q.is_empty and ext.covers(q)]
    centre=max(points,key=lambda q:ext.boundary.distance(q)) if points else None
    radius=float(ext.boundary.distance(centre)) if centre is not None and ext.covers(centre) else 0.
    return dict(fits=radius>=1.2-1e-7,diameter_m=2.4,radius_m=radius,
        centre_xz_m=[centre.x,centre.y] if centre is not None else None,
        shape_basis="filled joined exterior; exact centre-to-boundary circle certificate; raw floor unchanged",
        failed_trial_radius_not_maximized=True)

def two_leg_candidates(g,furniture,cap,axis,mode="rooms"):
    local=rotate(g,-axis,origin=(0,0));x0,y0,x1,y1=local.bounds;e=max(x1-x0,y1-y0)+2
    choices=[];parent=c.connection_count(g)
    xs=list(np.linspace(x0+.5,x1-.5,9));ys=list(np.linspace(y0+.5,y1-.5,9))
    if mode=="corridor" and CORRIDOR_ELBOW_VERTEX_REFINEMENT:
        vertices=[xy for part in polygons(rotate(c.outline(g),-axis,origin=(0,0)).simplify(.05,preserve_topology=True)) for xy in part.exterior.coords]
        xs=sorted(set(xs)|{a for a,b in vertices if x0+.05<a<x1-.05})
        ys=sorted(set(ys)|{b for a,b in vertices if y0+.05<b<y1-.05})
        if len(xs)>24:xs=[xs[int(i)] for i in np.linspace(0,len(xs)-1,24,dtype=int)]
        if len(ys)>24:ys=[ys[int(i)] for i in np.linspace(0,len(ys)-1,24,dtype=int)]
    for x in xs:
        for y in ys:
            for a,b in [((x,y0-e),(x1+e,y)),((x,y0-e),(x0-e,y)),((x,y1+e),(x1+e,y)),((x,y1+e),(x0-e,y))]:
                line=rotate(LineString([a,(x,y),b]),axis,origin=(0,0));children=cs.split_geometry(g,line)
                if not children:continue
                main=[p for p in children if p.area>=6-1e-8]
                if mode=="rooms":
                    if len(main)<2 or any(short_side(p)<2.4-1e-7 or not c.room_circle(p)["fits"] or c.connection_count(p)!=1 for p in main):continue
                elif mode=="connectivity":
                    if max(c.connection_count(p) for p in children)>=parent:continue
                else:
                    broad=[p for p in main if c.room_circle(p)["fits"]]
                    if not broad or len(broad)==len(children) or any(c.broad_components(p) for p in children if p not in broad):continue
                m=cs.cut_measure(line,g,furniture);target=g.area/max(2,math.ceil((g.area-1e-8)/cap))
                score=(m["furniture_intersection_length_m"]>1e-7,m["furniture_intersection_length_m"],
                       sum(math.ceil((p.area-1e-8)/cap) for p in main),sum(abs(p.area-target) for p in main),
                       2,m["furniture_intersection_area_m2"],line.length)
                choices.append((score,line,children,m))
    return sorted(choices,key=lambda x:x[0])

def candidates(g,furniture,cap,axis,mode="rooms",doglegs=True):
    strict=BASE_CANDIDATES(g,furniture,cap,axis,mode=mode,doglegs=False)
    if mode=="connectivity":return strict
    if strict:
        if doglegs and not any(x[3]["furniture_intersection_length_m"]<=1e-7 for x in strict):
            strict+=two_leg_candidates(g,furniture,cap,axis,mode)
        return sorted(strict,key=lambda x:x[0])
    tilted=[]
    if mode=="rooms":
        for delta in [-3,3,-6,6,-9,9,-12,12,-15,15]:
            tilted.extend(BASE_CANDIDATES(g,furniture,cap,axis+delta,mode=mode,doglegs=False))
        if tilted:return sorted(tilted,key=lambda x:x[0])
    return two_leg_candidates(g,furniture,cap,axis,mode) if doglegs else []

def finite_corridor_candidates(g,furniture,axis):
    """Local finite wall-to-wall chords cannot cut a distant broad room."""
    from scipy.spatial import Voronoi,QhullError
    from tools.rooms.room_split_auto.contours import chord
    ext=c.outline(g);samples=[]
    for p in polygons(ext):
        ring=LineString(p.exterior.coords).simplify(.015,preserve_topology=True)
        n=max(4,math.ceil(ring.length/.10))
        samples.extend((q.x,q.y) for q in (ring.interpolate(i/n,normalized=True) for i in range(n)))
    samples=np.unique(np.round(samples,7),axis=0)
    if len(samples)>12000:samples=samples[np.linspace(0,len(samples)-1,12000,dtype=int)]
    try:
        vor=Voronoi(samples);centres=list(vor.vertices)
        centres.extend((vor.vertices[a]+vor.vertices[b])/2 for a,b in vor.ridge_vertices if a>=0 and b>=0)
    except (QhullError,ValueError):return []
    centres=np.asarray(centres)
    if not len(centres):return []
    centres=centres[shapely.contains_xy(ext,centres[:,0],centres[:,1])]
    widths=2*shapely.distance(shapely.points(centres),ext.boundary)
    centres=centres[(widths>.08)&(widths<=CORRIDOR_JUNCTION_SEARCH_M+1e-8)]
    choices=[];seen=set()
    for centre in centres:
        key=tuple(np.round(centre/.15).astype(int))
        if key in seen:continue
        seen.add(key);hit=chord(ext,centre,[axis,axis+90])
        if hit is None or hit["width_m"]>CORRIDOR_JUNCTION_SEARCH_M+1e-8:continue
        children=c.split_at_chord(g,hit["line"],0.)
        if not children:continue
        broad=[q for q in children if q.area>=6-1e-8 and c.room_circle(q)["fits"] and c.broad_components(q)]
        narrow=[q for q in children if q not in broad]
        if not broad or not narrow or any(c.broad_components(q) for q in narrow):continue
        if any(c.connection_count(q)!=1 for q in broad):continue
        measure=cs.cut_measure(hit["line"],g,furniture)
        score=(measure["furniture_intersection_length_m"]>1e-7,
               measure["furniture_intersection_length_m"],-sum(q.area for q in narrow),
               len(broad),hit["width_m"])
        choices.append((score,hit["line"],children,measure))
    ordered=sorted(choices,key=lambda x:x[0])
    if CORRIDOR_ELBOW_VERTEX_REFINEMENT:
        evaluated=[]
        for score,line,children,measure in ordered[:64]:
            broad=[q for q in children if q.area>=6 and c.room_circle(q)["fits"] and c.broad_components(q)]
            remaining_narrow=0.
            for q in broad:
                rect=q.minimum_rotated_rectangle
                lengths=np.linalg.norm(np.diff(np.asarray(rect.exterior.coords),axis=0),axis=1)
                if c.body_width(c.outline(q))<1.5 and max(lengths)/max(short_side(q),1e-9)>=3:remaining_narrow+=q.area
            rank=(remaining_narrow>0,remaining_narrow,score[0],score[1],-sum(q.area for q in broad),score)
            evaluated.append((rank,line,children,measure))
        return sorted(evaluated,key=lambda x:x[0])
    return ordered

def split_corridor(g,furniture,axis):
    if not c.broad_components(g):return [(g,False)],[]
    pending=[g];parts=[];cuts=[]
    while pending:
        q=pending.pop(0)
        if not c.broad_components(q):parts.append((q,False));continue
        rect=q.minimum_rotated_rectangle
        lengths=np.linalg.norm(np.diff(np.asarray(rect.exterior.coords),axis=0),axis=1)
        aspect=max(lengths)/max(short_side(q),1e-9)
        if c.body_width(c.outline(q))>=1.5 or aspect<3:parts.append((q,False));continue
        choices=finite_corridor_candidates(q,furniture,axis)
        if not choices:choices=candidates(q,furniture,35,axis,mode="corridor",doglegs=True)
        if not choices or len(cuts)>=12:parts.append((q,True));continue
        _,line,children,measure=choices[0]
        cuts.append((line,measure,"narrow","finite_axis_chord_narrow_corridor_only",len(choices)))
        pending[:0]=children
    return parts,cuts

def refined_solve(self,g,depth=0):
    leaves,cuts=BASE_SOLVE(self,g,depth)
    if g.area<=self.cap+1e-8 or depth>=10:return leaves,cuts
    if not any(error in ("NO_AXIS_CUT_WITH_2_4_M_DISKS","AXIS_CUT_CPU_BUDGET_UNRESOLVED") for part,vis,error in leaves):return leaves,cuts
    pieces,extra=split_corridor(g,self.furniture,self.axis)
    if not extra or any(unresolved for part,unresolved in pieces):return leaves,cuts
    replacement=[];newcuts=list(extra)
    for part,unresolved in pieces:
        if part.area<g.area-1e-6:
            fresh=c.Cutter(self.mesh,self.points,self.nav,self.p,self.parameter,self.cap,self.furniture,self.axis)
            child_leaves,child_cuts=fresh.solve(part,depth+1)
            self.audit.extend(fresh.audit)
            replacement.extend(child_leaves);newcuts.extend(child_cuts)
        else:return leaves,cuts
    if sum(part.area for part,vis,error in replacement if error)<sum(part.area for part,vis,error in leaves if error):
        return replacement,newcuts
    return leaves,cuts

def install(ctx):
    c.outline=ctx.envelope;cs.filled_footprint=ctx.envelope
    c.room_circle=efficient_room_circle;BASE_ROOM_CIRCLE.cache_clear()
    c.connection_count=lambda g,width=.6:ctx.count(g)
    def splitting(g,line):
        children=ctx.split(g,line)
        c.room_circle.cache_clear();c.opened.cache_clear();c.broad_components.cache_clear()
        return children
    cs.split_geometry=splitting;c.candidates=candidates;c.split_corridor=split_corridor
    def finite(g,line,minimum_each=0.):
        xy=np.asarray(line.coords);u=xy[-1]-xy[0];length=np.linalg.norm(u)
        if length<1e-9:return None
        u=u/length;children=splitting(g,LineString([xy[0]-u*.003,xy[-1]+u*.003]))
        if len(children)<2 or any(p.area<minimum_each-1e-8 for p in children):return None
        return children
    c.split_at_chord=finite
    def connected(g,furniture,axis,max_depth=16):
        groups=ctx.groups(g);pending=[(p,0) for p in groups];out=[];cuts=[]
        if g.wkb in ctx.masks:
            for p in groups:ctx.masks[p.wkb]=ctx.masks[g.wkb]
        while pending:
            p,depth=pending.pop(0);count=ctx.count(p)
            if count<=1 or depth>=max_depth:out.append(p);continue
            choices=candidates(p,furniture,35,axis,mode="connectivity",doglegs=False)
            if not choices:out.append(p);continue
            _,line,children,m=choices[0]
            cuts.append((line,m,"narrow","true_neck_after_seam_and_navmesh_merge",len(choices)))
            pending[:0]=[(ch,depth+1) for ch in children]
        return out,cuts
    c.connected_parts=connected
    c.Cutter.solve=refined_solve if CORRIDOR_ELBOW_VERTEX_REFINEMENT else BASE_SOLVE
    c.room_circle.cache_clear();c.opened.cache_clear();c.broad_components.cache_clear()

def process(old,mesh,pf,hs,nav_polys,nav_ys,objects,markers,door_info,p,parameter,cap,cache,root):
    original=old["source_geometry"];result=copy.deepcopy(old)
    result.update(blocks=[],cut_lines=[],revision="walkable_seam_nav_v5",status="processed",requires_split=True,
        floor_overheads={},wall_axes={},stair_exact_exclusion=[],stair_separation_audit=[],merge_audit=[],
        straight_partition_audit=[],corridor_salvage_audit=[],connection_source_audit=[],door_instance_audit_v5=[],
        parameter=dict(parameter,max_room_area_m2=cap))
    for floor in original["floors"]:
        scope=shape(floor["floor_polygon"]);fy=floor["floor_y_m"]
        nav=nav_scope_at(nav_polys,nav_ys,fy,exterior(scope).buffer(.3),p)
        ctx=Connectivity(nav);before=ctx.certificate(scope);install(ctx)
        one=copy.deepcopy(old);one["source_geometry"]=dict(original,floors=[floor],floor_area_sum_m2=scope.area)
        part=BASE_PROCESS(one,mesh,pf,hs,nav_polys,nav_ys,objects,markers,door_info,p,parameter,cap,cache,root)
        result["connection_source_audit"].append(dict(floor_id=floor["floor_id"],**before))
        for b in part["blocks"]:
            g=shape(b["floor_polygon_xz_m"]);cert=ctx.certificate(g)
            b.update(schema="hm3d_auto_room_split_v5",connectivity_rule="seam_nav_v5",
                connectivity_certificate=cert,connectivity_core_count=ctx.count(g),raw_semantic_part_count=len(list(polygons(g))))
            if b.get("inscribed_circle") and not b["inscribed_circle"].get("fits") and g.area>=6:
                b["inscribed_circle"]=BASE_ROOM_CIRCLE(g)
                b["inscribed_circle"]["failed_final_radius_measured_once"]=True
            if b.get("inscribed_circle"):b["inscribed_circle"]["shape_basis"]="filled joined exterior with seam/local-native-nav bridges; raw ground unchanged"
            diag=list(dict.fromkeys(b.get("unverified_diagnostics",[])+b.get("unresolved_reasons",[])))
            construction=any("AXIS_CUT" in x or "CORRIDOR_WIDE_PART" in x for x in diag)
            actual_narrow=any(x in ("CORRIDOR_BODY_WIDTH_BELOW_1_5","DETACHED_FRAGMENT","STAIRS") for x in b.get("discard_reasons",[]))
            construction=construction or (not actual_narrow and "NO_2_4_M_ROOM_DISK_AXIS_CONSTRUCTION_UNRESOLVED" in diag)
            if b["floor_area_m2"]>=6 and construction:
                b["construction_failure_secondary_checks"]=b.get("discard_reasons",[])
                b.update(decision="unresolved",discard_reasons=[],unresolved_reasons=diag)
            result["blocks"].append(b)
        for cut in part["cut_lines"]:
            error=max(cut["segment_angle_errors_deg"],default=0)
            cut.update(angle_fallback_deg=error,angle_fallback_allowed=error<=15+1e-6,
                direction_policy="wall axes, then absent feasible straight axes +-15deg, then two axis legs")
            result["cut_lines"].append(cut)
        result["floor_overheads"].update(part["floor_overheads"]);result["wall_axes"].update(part["wall_axes"])
        for key in ["stair_exact_exclusion","stair_separation_audit","merge_audit","straight_partition_audit","corridor_salvage_audit"]:
            result[key].extend(part.get(key,[]))
        result["door_instance_audit_v5"].extend(part.get("door_instance_audit_v4",[]))
    result["retained_new_rooms"]=sum(b["decision"]=="retain" for b in result["blocks"])
    result["status"]="partially_unresolved" if any(b["decision"]=="unresolved" for b in result["blocks"]) else "processed"
    result["area_partition_error_m2"]=abs(sum(b["floor_area_m2"] for b in result["blocks"])-original["floor_area_sum_m2"])
    finalize_interfaces(result["blocks"],result["cut_lines"]);adjacency(result["blocks"],result["cut_lines"])
    return result

def worker(job,root,out,p,parameter,cap,cache):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    global CORRIDOR_JUNCTION_SEARCH_M,CORRIDOR_ELBOW_VERTEX_REFINEMENT
    if job.get("corridor_refine"):
        CORRIDOR_JUNCTION_SEARCH_M=2.4;CORRIDOR_ELBOW_VERTEX_REFINEMENT=True
    t=time.time();root=Path(root);out=Path(out);hs=load_native()
    pf,np_,ny=navmesh_triangles(hs,Path(job["navmesh"]));mesh,receipt=raw_collision(job["scene_directory"])
    markers,door_info=structural_instances(job["scene_directory"]);objects=load_scene(job["scene_directory"])
    marker_ids={m["instance_id"] for m in markers}
    markers.extend(dict(x,semantic_source=str(Path(job["scene_directory"])/(Path(job["scene_directory"]).name.split("-",1)[1]+".semantic.glb"))) for x in objects.instances if x["category"] in ("step","steps") and x["instance_id"] not in marker_ids)
    statuses=[]
    for source in job["sources"]:
        old=json.loads(Path(source).read_text());start=time.time()
        try:d=process(old,mesh,pf,hs,np_,ny,objects,markers,door_info,p,parameter,cap,cache,root)
        except Exception as exc:
            d=copy.deepcopy(old);d.update(status="unresolved",revision="walkable_seam_nav_v5",retained_new_rooms=0,
                cut_lines=[],unresolved_reasons=[repr(exc)],traceback=traceback.format_exc(),requires_split=True)
            for b in d["blocks"]:b.update(decision="unresolved",discard_reasons=[],unresolved_reasons=["V5_PROCESS_FAILURE"],new_room=True)
        dump(out/"regions"/Path(source).name,d)
        statuses.append(dict(source=d["source_region"],status=d["status"],seconds=time.time()-start,retained=d["retained_new_rooms"]))
        print("V5_REGION",job["house"],d["source_region"],d["status"],"seconds",round(time.time()-start,1),"kept",d["retained_new_rooms"],flush=True)
    return dict(house=job["house"],seconds=time.time()-t,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ray_receipt=receipt,statuses=statuses)

def prepare(root):
    root=Path(root);out=root/"delivery_all_v5";out.mkdir(exist_ok=False)
    (out/"regions").mkdir();(out/"rooms").mkdir();(out/"source_inputs").mkdir()
    plan=json.loads((root/"delivery_all_v4/revision_plan.json").read_text())
    prepared=json.loads(PREP.read_text())["rooms"];bykey={(r["house"],r["room_label"]):r for r in prepared}
    targets={Path(s).name:json.loads(Path(s).read_text()) for j in plan["jobs"] for s in j["sources"]}
    native_deltas=[]
    for name,d in targets.items():
        if d.get("native_selection"):
            x=bykey[(d["house"],d["source_region"])];row=d["source_geometry"];oldarea=row["floor_area_sum_m2"]
            floor=dict(floor_id=x["selected_floor_id"],floor_y_m=x["floor_y_m"],height_range_m=x["height_range_m"],
                       face_count=x["ground_face_count"],floor_area_m2=x["floor_area_m2"],short_side_m=x["short_side_m"],floor_polygon=x["floor_polygon_xz_m"])
            row.update(floors=[floor],floor_area_sum_m2=x["floor_area_m2"],ground_measurement=x["floor_area_method"],
                       native_polygon_manifest=str(PREP),source_selection=x["source_list_name"])
            d["source_floor_area_m2"]=x["floor_area_m2"]
            native_deltas.append(dict(source=name,previous_area_m2=oldarea,manifest_area_m2=x["floor_area_m2"],delta_m2=x["floor_area_m2"]-oldarea))
    jobs={}
    for name,d in targets.items():
        path=out/"source_inputs"/name;dump(path,d);row=d["source_geometry"];house=d["house"]
        if house not in jobs:jobs[house]=dict(house=house,scene_directory=row["scene_directory"],navmesh=row["navmesh_source"],sources=[])
        jobs[house]["sources"].append(str(path))
    previous=root/"delivery_all_v4/final_v1/regions"
    if not previous.exists():previous=root/"delivery_all_v4/regions"
    for path in previous.glob("*.json"):
        if path.name not in targets:dump(out/"regions"/path.name,json.loads(path.read_text()))
    ordered=sorted(jobs.values(),key=lambda j:-sum(targets[Path(s).name]["source_floor_area_m2"] for s in j["sources"]))
    plan.update(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=ordered,
        source="delivery_all_v4 corrected metadata plus owner-authorized native polygon manifest",
        native_polygon_manifest=str(PREP),native_geometry_deltas=native_deltas,
        connectivity="<=0.05m seams; otherwise connected same-floor navmesh intersection of parts expanded0.3m; then filled joined exterior0.6m test",
        cut_fallback="axes, absent feasible axes +/-15deg, two wall-axis legs",
        visibility_policy="no extra subcap visibility admission or splitting; whole-scene diagnostic only",
        unchanged_policy="846 originals<=35 not cut until separate authorized native connectivity work")
    dump(out/"revision_plan.json",plan)
    print("V5_PREPARED",len(targets),"HOUSES",len(jobs),flush=True)

def run(root,workers=8,pilot_dir=None):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    root=Path(root);out=root/"delivery_all_v5";plan=json.loads((out/"revision_plan.json").read_text());jobs=plan["jobs"]
    if pilot_dir:
        out=root/"revision_seam_nav_v5_20261009_v1"/pilot_dir;out.mkdir(exist_ok=False);(out/"regions").mkdir()
        wanted={("00638","R5"),("00150","R12"),("00475","R0"),("00707","R5"),("00327","R6"),("00414","R10"),("00149","R5")}
        if pilot_dir=="pilot_v3":wanted={("00150","R12")}
        if pilot_dir=="pilot_corridor_v1":wanted={("00238","R0"),("00099","R2")}
        jobs=[dict(j,sources=[s for s in j["sources"] if any(h in j["house"] and json.loads(Path(s).read_text())["source_region"]==rid for h,rid in wanted)]) for j in jobs]
        jobs=[dict(j,corridor_refine=pilot_dir=="pilot_corridor_v1") for j in jobs if j["sources"]]
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,j,root,out,plan["parameters"],plan["selected_parameter"],35,plan["overhead_cache"]):j["house"] for j in jobs}
        for f in as_completed(futures):
            try:receipts.append(f.result())
            except Exception as exc:receipts.append(dict(house=futures[f],error=repr(exc),traceback=traceback.format_exc()))
    dump(out/"native_receipt.json",receipts)
    missing=[s for j in jobs for s in j["sources"] if not (out/"regions"/Path(s).name).exists()]
    if not pilot_dir:
        for s in missing:
            d=json.loads(Path(s).read_text());d.update(status="unresolved",requires_split=True,cut_lines=[],retained_new_rooms=0,unresolved_reasons=["NATIVE_HOUSE_JOB_FAILURE"])
            for b in d["blocks"]:b.update(decision="unresolved",discard_reasons=[],unresolved_reasons=["NATIVE_HOUSE_JOB_FAILURE"],new_room=True)
            dump(out/"regions"/Path(s).name,d)
        for path in (out/"regions").glob("*.json"):
            d=json.loads(path.read_text())
            for b in d["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
    dump(out/"completed.json",dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,
        max_compute_processes=workers+2,max_address_space_gib=10*(workers+1),source_regions=len(list((out/"regions").glob("*.json"))),
        native_house_jobs=len(jobs),missing_sources=missing,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0)))
    print("V5_NATIVE_DONE",out,flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--prepare",action="store_true")
    a.add_argument("--workers",type=int,default=8);a.add_argument("--pilot-dir");v=a.parse_args()
    if v.prepare:prepare(v.root)
    else:run(v.root,v.workers,v.pilot_dir)
