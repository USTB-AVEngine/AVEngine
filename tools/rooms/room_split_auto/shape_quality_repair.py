"""Repair only authorized v5 source regions; all other room bytes stay intact."""
from __future__ import annotations
import argparse,copy,datetime,json,math,os,resource,time,traceback,multiprocessing,shutil,hashlib,signal,faulthandler
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import numpy as np
import shapely
from shapely.geometry import shape,mapping,LineString,GeometryCollection
from PIL import Image
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.measurements import load_scene,navmesh_triangles
from tools.rooms.room_selection.navigation import sample_navigation,placement
from tools.rooms.room_split_auto import capped_split as cap,connected_split as cs
from tools.rooms.room_split_auto.pipeline import load_native,raw_collision,nav_scope_at,measured_black,block_type,finalize_interfaces,adjacency
from tools.rooms.room_split_auto.shape_quality_geometry import own_outline,defects,disk,Repair,PartConnectivity,raw_cells,wrap_count,surviving_design

def dump(p,d):
    with Path(p).open("x") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def key(reg):return reg["house"]+"/"+reg["source_region"]
def fresh_room_id(house,region,fid,used,start=0):
    """Reserve every input ID before creating cells during an incremental retry."""
    index=start
    while True:
        bid=house+"__"+region+"__"+fid+f"__Q{index:03d}"
        if bid not in used:
            used.add(bid)
            return bid
        index+=1

def unique_block_ids(result):
    """Repair imported retry ID collisions without changing any geometry."""
    used={b["id"] for b in result["blocks"]};seen=set();audit=[]
    for b in result["blocks"]:
        original=b["id"]
        if original in seen:
            b["id"]=fresh_room_id(result["house"],result["source_region"],b["floor_id"],used)
            audit.append(dict(original_id=original,id=b["id"],floor_id=b["floor_id"],
                raw_polygon_sha256=hashlib.sha256(json.dumps(b["floor_polygon_xz_m"],sort_keys=True).encode()).hexdigest()))
        seen.add(b["id"])
    if audit:
        result["room_id_repair_audit"]=audit
        adjacency(result["blocks"],[c for c in result["cut_lines"] if c.get("active_in_final_partition",True)])
    return audit

def room_disk_certificate(g,tolerance=.0005):
    """The room's own outline, with the reviewer's maximum-circle precision."""
    envelope=own_outline(g)
    choices=[shapely.maximum_inscribed_circle(q,tolerance=tolerance) for q in polygons(envelope)]
    circle=max(choices,key=lambda q:q.length) if choices else None
    radius=float(circle.length) if circle is not None else 0.
    centre=list(circle.coords[0]) if circle is not None else None
    return dict(fits=radius>=1.2,diameter_m=2.4,radius_m=radius,centre_xz_m=centre,
        maximum_circle_tolerance_m=tolerance,
        shape_basis="room own exterior; <=0.05m seams closed, holes filled; no navmesh or neighbour floor")

def wide_body_candidates(g):
    """Detect real-sized wide cores without replacing any raw measured floor."""
    if g.area<6:return []
    opening=own_outline(g).buffer(-.75,join_style=2).buffer(.75,join_style=2)
    return [q for q in polygons(opening) if q.area>=6 and shapely.maximum_inscribed_circle(q,tolerance=.0005).length>=1.2]

def recover_wide_dropped_candidates(blocks):
    """Re-open eligible discards for native admission; stairs stay unchanged.

    Returned candidates are planning records, not an admission claim. The
    process() consumer recomputes raw geometry metrics and the frozen witness.
    """
    result=[];audit=[]
    for block in blocks:
        b=copy.deepcopy(block)
        if b["decision"]=="discard" and "STAIRS" not in b.get("discard_reasons",[]):
            g=shape(b["floor_polygon_xz_m"]);cores=wide_body_candidates(g)
            if cores:
                audit.append(dict(id=b["id"],old_reasons=b.get("discard_reasons",[]),
                    raw_floor_area_m2=g.area,wide_outline_areas_m2=[q.area for q in cores],
                    method="raw floor re-enters normal room/neck/corridor/cap pipeline"))
                b["decision"]="retain";b["discard_reasons"]=[];b["unresolved_reasons"]=[]
                b["wide_body_recovery_requires_native_measurement"]=True
                b["wide_body_recovery_original_id"]=block["id"]
        result.append(b)
    return result,audit

def merge_nodisk_candidates(blocks,cap_m2=35.):
    """Prefer a legal adjacent merge before boundary and bent-line searches."""
    result=copy.deepcopy(blocks);audit=[]
    for _ in range(len(blocks)):
        invalid=[i for i,b in enumerate(result) if b["decision"]=="retain" and not room_disk_certificate(shape(b["floor_polygon_xz_m"]))["fits"]]
        if not invalid:break
        merged=False
        for i in invalid:
            a=result[i];ga=shape(a["floor_polygon_xz_m"])
            for j,b in enumerate(result):
                if i==j or b["decision"]!="retain" or a["floor_id"]!=b["floor_id"]:continue
                gb=shape(b["floor_polygon_xz_m"])
                if ga.distance(gb)>.600001:continue
                union=ga.union(gb)
                if not(6<=union.area<=cap_m2) or short_side(union)<2.4 or not room_disk_certificate(union)["fits"]:continue
                d=defects(union)
                if d["neck_count"] or d["corridor_count"]:continue
                other=[shape(x["floor_polygon_xz_m"]) for k,x in enumerate(result) if k not in (i,j) and x["decision"]=="retain" and x["floor_id"]==a["floor_id"]]
                if wrap_count([union]+other):continue
                new=copy.deepcopy(a);new["floor_polygon_xz_m"]=mapping(union)
                new["floor_area_m2"]=float(union.area)
                new["disk_merge_requires_native_measurement"]=True
                new["disk_merge_source_ids"]=[a["id"],b["id"]]
                audit.append(dict(source_ids=[a["id"],b["id"]],merged_area_m2=union.area))
                result=[x for k,x in enumerate(result) if k not in (i,j)]+[new]
                merged=True;break
            if merged:break
        if not merged:break
    return result,audit

def enforce_room_disk_admission(blocks):
    """An unsuccessful disk construction discards that cell, never the source."""
    audit=[]
    for b in blocks:
        known_disk_failure=b["decision"]=="unresolved" and "NO_2_4_M_OWN_ROOM_DISK" in b.get("unresolved_reasons",[])
        if b["decision"]!="retain" and not known_disk_failure:continue
        certificate=room_disk_certificate(shape(b["floor_polygon_xz_m"]))
        b["inscribed_circle"]=certificate
        if certificate["fits"]:continue
        b["decision"]="discard";b["discard_reasons"]=["NO_2_4M_DISK"];b["unresolved_reasons"]=[]
        reason=b.get("shape_repair_remaining_reason","finite legal boundary/axis/angle/bent/merge search did not yield a 2.4m own-outline disk")
        b["disk_admission_failure"]=dict(reason=reason,global_impossibility_proved=False)
        audit.append(dict(id=b["id"],area_m2=b["floor_area_m2"],radius_m=certificate["radius_m"],reason=reason))
    return audit

def wide_body_recovery_plan(g,furniture,axis,neighbors=(),cap_m2=35.,budget_seconds=30.):
    """Plan on raw floor; opening cores only suggest cuts, never add floor."""
    from shapely.affinity import rotate
    started=time.monotonic();best=None
    cores=wide_body_candidates(g)
    for angle in [axis,axis+90,axis-15,axis+15,axis+75,axis+105]:
        local=rotate(own_outline(g),-angle,origin=(0,0))
        if local.is_empty:continue
        a,b,c,d=local.bounds;extra=max(c-a,d-b)+2
        positions=set(np.arange(a+.1,c-.1,.15))
        for core in cores:
            q=rotate(core,-angle,origin=(0,0));positions.update([q.bounds[0],q.bounds[2]])
        for position in sorted(positions,key=lambda x:abs(x-(a+c)/2)):
            if time.monotonic()-started>budget_seconds:return best
            line=rotate(LineString([(position,b-extra),(position,d+extra)]),angle,origin=(0,0))
            cells=raw_cells(g,line)
            if len(cells)!=2:continue
            kept=[];discarded=[];valid=True
            for cell in cells:
                certificate=room_disk_certificate(cell);flags=defects(cell)
                if 6<=cell.area<=cap_m2 and short_side(cell)>=2.4 and certificate["fits"] and not flags["neck_count"] and not flags["corridor_count"]:
                    kept.append(cell)
                elif wide_body_candidates(cell):valid=False;break
                elif not certificate["fits"] or cell.area<6:
                    discarded.append(cell)
                else:valid=False;break
            if not valid or not kept or wrap_count(kept+list(neighbors))>wrap_count(neighbors):continue
            measurement=cs.cut_measure(line,g,furniture)
            if measurement["furniture_intersection_length_m"]>.5:continue
            score=(-sum(q.area for q in kept),measurement["furniture_intersection_length_m"],max(cap.angle_errors(line,axis)))
            if best is None or score<best["score"]:
                best=dict(score=score,line=line,retained=kept,discarded=discarded,measurement=measurement)
    return best

def prepare_wide_body_recovery(result,furniture_by_floor,cap_m2=35.):
    """Recover a wide discard after disk rejection, possibly moving its neighbour.

    Returns planning cells requiring native witness/black/nav measurement.
    No neighbour floor outside this source or floor is borrowed.
    """
    planned=copy.deepcopy(result);audit=[]
    used={b["id"] for b in planned["blocks"]}
    for discarded in list(planned["blocks"]):
        if discarded not in planned["blocks"] or discarded["decision"]!="discard" or "STAIRS" in discarded.get("discard_reasons",[]):continue
        g=shape(discarded["floor_polygon_xz_m"])
        if not wide_body_candidates(g):continue
        fid=discarded["floor_id"];axis=planned.get("wall_axes",{}).get(fid,{}).get("primary_deg",cap.main_axis(g))
        options=[[]]+[[b] for b in planned["blocks"] if b["decision"]=="retain" and b["floor_id"]==fid and g.distance(shape(b["floor_polygon_xz_m"]))<=.600001]
        best=None
        for neighbours in options:
            union=shapely.union_all([g]+[shape(b["floor_polygon_xz_m"]) for b in neighbours])
            other=[shape(b["floor_polygon_xz_m"]) for b in planned["blocks"] if b["decision"]=="retain" and b["floor_id"]==fid and all(b is not n for n in neighbours)]
            candidate=wide_body_recovery_plan(union,furniture_by_floor[fid],axis,other,cap_m2)
            if candidate and (best is None or candidate["score"]<best[0]["score"]):best=(candidate,neighbours,union)
        if best is None:
            audit.append(dict(id=discarded["id"],status="no_legal_raw_floor_recovery_cut",reason="bounded raw axis/angle search found no legal disk room without a remaining wide discard"))
            continue
        candidate,neighbours,union=best
        removed={discarded["id"]}|{b["id"] for b in neighbours}
        planned["blocks"]=[b for b in planned["blocks"] if b["id"] not in removed]
        created=[]
        for retained,cells in [(True,candidate["retained"]),(False,candidate["discarded"])]:
            for cell in cells:
                b=copy.deepcopy(discarded)
                b["id"]=fresh_room_id(planned["house"],planned["source_region"],fid,used)
                b["floor_polygon_xz_m"]=mapping(cell);b["floor_area_m2"]=float(cell.area);b["short_side_m"]=short_side(cell)
                b["inscribed_circle"]=room_disk_certificate(cell);b["unresolved_reasons"]=[]
                b["placement_witness"]=dict(found=False,not_run_reason="native admission pending" if retained else "local geometry rejection")
                b["black_fraction"]=None;b["black_measurement"]=dict(status="pending native remeasurement")
                b["decision"]="retain" if retained else "discard"
                b["discard_reasons"]=[] if retained else ["NO_2_4M_DISK" if cell.area>=6 else "FLOOR_AREA_BELOW_6"]
                if retained:b.pop("disk_admission_failure",None)
                elif cell.area>=6:
                    b["disk_admission_failure"]=dict(reason="remaining raw floor after a legal wide-body boundary reallocation has no 2.4m own-outline disk",global_impossibility_proved=False)
                b["wide_body_recovery_original_id"]=discarded["id"]
                b["wide_body_recovery_requires_native_measurement"]=retained
                b["admission_recovery_local_discard"]=not retained
                b["new_room"]=True;b["cut_ids"]=[];b["adjacent_rooms"]=[]
                planned["blocks"].append(b);created.append(b["id"])
        apply_cuts(planned,[(candidate["line"],"WIDE_RECOVERY",candidate["measurement"],union)],fid,axis)
        audit.append(dict(id=discarded["id"],status="planned_raw_recovery",neighbour_ids=[b["id"] for b in neighbours],created_ids=created,
            raw_parent_area_m2=union.area,retained_area_m2=sum(q.area for q in candidate["retained"]),furniture_intersection_length_m=candidate["measurement"]["furniture_intersection_length_m"]))
    planned["shape_admission_reserved_ids"]=sorted(used)
    return planned,audit

def shape_flags(leaves):
    retained=[x["g"] for x in leaves if not x.get("forced") and (not x.get("error") or x.get("original",{}).get("decision")=="retain")]
    return sum(defects(g)["neck_count"]+defects(g)["corridor_count"] for g in retained)+wrap_count(retained)
def cut_record(fid,axis,line,kind,m,parent,index):
    return dict(id=fid+f"_Q{index:03d}",floor_id=fid,type="narrow" if kind in ("NECK","CORRIDOR") else "visibility",
        stage="own_outline_"+kind.lower()+"_shape_repair",line_xz_m=list(map(list,line.coords)),
        line_geometry_xz_m=mapping(line),segment_count=len(line.coords)-1,wall_axis_deg=axis,
        segment_angle_errors_deg=cap.angle_errors(line,axis),angle_fallback_deg=max(cap.angle_errors(line,axis),default=0.),
        direction_policy="wall axes then +/-15degrees; navigation never shapes room outline",
        applied_parent_geometry_xz_m=mapping(parent),**m)
def apply_cuts(result,cuts,fid,axis):
    for line,kind,m,parent in cuts:
        result["cut_lines"].append(cut_record(fid,axis,line,kind,m,parent,len(result["cut_lines"])))
def local_wrap_repair(leaves,repair):
    cuts=[];audit=[]
    for _ in range(12):
        live=[(i,x) for i,x in enumerate(leaves) if not x.get("forced") and (not x.get("error") or x.get("original",{}).get("decision")=="retain")]
        hit=None
        for i,a in live:
            aa=own_outline(a["g"])
            for j,b in live:
                if i==j:continue
                bb=own_outline(b["g"])
                if aa.convex_hull.intersection(bb).area/bb.area>=.3:hit=(i,j);break
            if hit:break
        if not hit:break
        i,j=hit;g=shapely.union_all([leaves[i]["g"],leaves[j]["g"]])
        choices=repair.choices(g,neighbors=[x["g"] for k,x in live if k not in hit])
        best=None
        for score,line,m in choices[:8]:
            children=raw_cells(g,line)
            if len(children)!=2:continue
            out=[];extra=[]
            for child in children:
                rr,cc=repair.solve(child);out+=rr;extra+=cc
            trial=[x for k,x in enumerate(leaves) if k not in hit]+out
            count=shape_flags(trial)
            if best is None or count<best[0]:best=(count,trial,[(line,"WRAP",m,g)]+extra)
            if count==0:break
        if best is None or best[0]>=shape_flags(leaves):
            audit.append(dict(status="unresolved",reason="NO_IMPROVING_WALL_AXIS_WRAP_CUT",pair=list(hit)));break
        audit.append(dict(status="repaired",old_flags=shape_flags(leaves),new_flags=best[0]))
        leaves=best[1];cuts+=best[2]
    return leaves,cuts,audit

def evaluate_interfaces(result,furniture_by_floor):
    finalize_interfaces(result["blocks"],result["cut_lines"])
    for cut in result["cut_lines"]:
        if not cut.get("active_in_final_partition"):continue
        measured=shape(cut["final_interface_geometry_xz_m"])
        parent=own_outline(shape(cut["applied_parent_geometry_xz_m"])) if cut.get("applied_parent_geometry_xz_m") else None
        if parent is not None:measured=measured.intersection(parent)
        design=LineString(cut["line_xz_m"])
        current=surviving_design(design,measured)
        if parent is not None:current=current.intersection(parent)
        cut["active_in_final_partition"]=current.length>1e-4
        cut["current_design_span_geometry_xz_m"]=mapping(current)
        cut["final_interface_geometry_xz_m"]=mapping(current)
        cut["current_design_span_length_m"]=current.length
        cut.setdefault("v5_or_design_furniture_intersection_length_m",cut.get("furniture_intersection_length_m"))
        if not cut["active_in_final_partition"]:
            cut["inactive_reason"]="crossing point only; no surviving collinear partition boundary"
            continue
        scope=shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in result["blocks"] if b["floor_id"]==cut["floor_id"]])
        furniture=furniture_by_floor[cut["floor_id"]]
        cut["actual_interface_furniture_measurement"]=cs.cut_measure(measured,scope,furniture)
        cut.update(cs.cut_measure(current,scope,furniture))
        cut["furniture_measurement_basis"]="current collinear design span; internal scan gaps bridged per leg; all-scene furniture; original v5/design metric retained separately"
    active={c["id"] for c in result["cut_lines"] if c.get("active_in_final_partition")}
    for block in result["blocks"]:block["cut_ids"]=[cid for cid in block.get("cut_ids",[]) if cid in active]
    adjacency(result["blocks"],result["cut_lines"])

def local_circle_repair(prior,furniture,axis):
    """Move the boundary of a deficient cell; do not discard it for the circle."""
    leaves=[dict(g=shape(b["floor_polygon_xz_m"]),forced=None,error=None,original=b) for b in prior]
    cuts=[];audit=[];changed=False
    for _ in range(len(prior)):
        bad=[i for i,x in enumerate(leaves) if not disk(x["g"])["fits"]]
        if not bad:break
        i=bad[0];g=leaves[i]["g"]
        neighbours=sorted((j for j in range(len(leaves)) if j!=i and g.distance(leaves[j]["g"])<=.600001),key=lambda j:g.distance(leaves[j]["g"]))
        improved=False
        for j in neighbours:
            union=shapely.union_all([g,leaves[j]["g"]]);others=[x["g"] for k,x in enumerate(leaves) if k not in (i,j)]
            solver=Repair(furniture,axis)
            try:choices=solver.choices(union,neighbors=others)
            except (shapely.GEOSException,TimeoutError):continue
            for score,line,m in choices[:16]:
                children=raw_cells(union,line)
                if len(children)!=2:continue
                if any(not (6-1e-8<=x.area<=35+1e-8) or short_side(x)<2.4-1e-7 or not disk(x)["fits"] for x in children):continue
                try:
                    if any(defects(x)["neck_count"] or defects(x)["corridor_count"] for x in children):continue
                    if wrap_count(children+others)>wrap_count([x["g"] for x in leaves]):continue
                except shapely.GEOSException:continue
                new=[dict(g=x,forced=None,error=None,original=max([leaves[i]["original"],leaves[j]["original"]],key=lambda b:x.intersection(shape(b["floor_polygon_xz_m"])).area)) for x in children]
                leaves=[x for k,x in enumerate(leaves) if k not in (i,j)]+new
                cuts.append((line,"CIRCLE",m,union))
                audit.append(dict(status="local_circle_boundary_repositioned",pair=[prior_id for prior_id in [new[0]["original"]["id"],new[1]["original"]["id"]]],candidate_search=solver.audit))
                improved=True;changed=True;break
            if improved:break
        if not improved:break
    return leaves,cuts,audit,changed and all(disk(x["g"])["fits"] for x in leaves)

def repair_floor(old,floor,nav,furniture):
    fid=floor["floor_id"];axis=old.get("wall_axes",{}).get(fid,{}).get("primary_deg",cap.main_axis(shape(floor["floor_polygon"])))
    solver=Repair(furniture,axis)
    prepared,recovery_audit=recover_wide_dropped_candidates([b for b in old["blocks"] if b["floor_id"]==fid])
    prepared,merge_disk_audit=merge_nodisk_candidates(prepared)
    old=copy.deepcopy(old)
    old["blocks"]=[b for b in old["blocks"] if b["floor_id"]!=fid]+prepared
    stable=[copy.deepcopy(b) for b in old["blocks"] if b["floor_id"]==fid and b["decision"] not in ("retain","unresolved")]
    prior=[b for b in old["blocks"] if b["floor_id"]==fid and b["decision"] in ("retain","unresolved")]
    leaves=[];cuts=[];audit=[dict(status="wide_body_recovery_candidates",entries=recovery_audit),dict(status="disk_merge_candidates",entries=merge_disk_audit)]
    joint_done=False
    live_prior=[b for b in prior if b["decision"]=="retain"]
    if len(live_prior)==len(prior) and len(prior)>=2 and any(not disk(shape(b["floor_polygon_xz_m"]))["fits"] for b in prior):
        rr,cc,aa,ok=local_circle_repair(prior,furniture,axis)
        audit+=aa
        if ok:leaves=rr;cuts=cc;joint_done=True
    if not joint_done and len(live_prior)>=2 and any(not disk(shape(b["floor_polygon_xz_m"]))["fits"] for b in live_prior):
        union=shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in prior])
        joint=Repair(furniture,axis)
        try:
            rr,cc=joint.solve(union)
            if not any(x["error"] for x in rr) and any(not x["forced"] for x in rr):
                for x in rr:
                    x["original"]=max(prior,key=lambda b:x["g"].intersection(shape(b["floor_polygon_xz_m"])).area)
                leaves=rr;cuts=cc;joint_done=True
                audit.append(dict(status="joint_floor_replan",reason="an old cell has no 2.4m circle on its own outline",old_rooms=len(live_prior),new_candidates=sum(not x["forced"] for x in rr),candidate_search=joint.audit))
            else:audit.append(dict(status="joint_floor_replan_failed",reasons=[x["error"] for x in rr if x["error"]],candidate_search=joint.audit))
        except TimeoutError as e:
            audit.append(dict(status="joint_floor_replan_failed",reason=str(e)))
    if not joint_done:
        for b in prior:
            g=shape(b["floor_polygon_xz_m"])
            if b["decision"]=="unresolved" and "SEMANTIC_PALETTE_AMBIGUOUS" in b.get("unresolved_reasons",[]):
                leaves.append(dict(g=g,forced=None,error="SEMANTIC_PALETTE_AMBIGUOUS",original=b));continue
            forbidden=shapely.union_all([shape(c["floor_polygon_xz_m"]) for c in prior if c is not b and c["decision"]=="retain"])
            ctx=PartConnectivity(nav,forbidden)
            groups=ctx.groups(g)
            for part in groups:
                if len(groups)>1 and part.area<6:
                    leaves.append(dict(g=part,forced="DETACHED_FRAGMENT",error=None,original=b));continue
                solver.nodes=0
                rr,cc=solver.solve(part)
                for x in rr:x["original"]=b
                if any(x["error"] for x in rr):
                    leaves.append(dict(g=part,forced=None,error=b.get("unresolved_reasons",["SHAPE_REPAIR_SEARCH_UNRESOLVED"])[0] if b["decision"]=="unresolved" else "SHAPE_REPAIR_SEARCH_UNRESOLVED",original=b))
                    audit.append(dict(room=b["id"],status="unresolved",reasons=[x["error"] for x in rr if x["error"]]))
                else:leaves+=rr;cuts+=cc
    before_wrap=shape_flags(leaves)
    solver.nodes=0
    leaves,wc,wa=local_wrap_repair(leaves,solver);cuts+=wc;audit+=wa
    # If a local repair makes no satisfactory shape partition, replan the union
    # with full-width half-plane cuts. This never borrows neighbouring ground.
    if shape_flags(leaves)>0:
        live=[x for x in leaves if not x.get("forced") and (not x.get("error") or x.get("original",{}).get("decision")=="retain")]
        if live:
            union=shapely.union_all([x["g"] for x in live]);solver.nodes=0
            rr,cc=solver.solve(union)
            trial=[x for x in leaves if x not in live]+rr
            if not any(x["error"] for x in rr) and shape_flags(trial)<shape_flags(leaves):
                audit.append(dict(status="full_width_replan",old_flags=shape_flags(leaves),new_flags=shape_flags(trial)))
                leaves=trial;cuts+=cc
    # Shift problematic existing boundaries, or choose the other wall axis.
    oldbad=[c for c in old["cut_lines"] if c["floor_id"]==fid and c.get("active_in_final_partition",True) and (c.get("furniture_intersection_length_m") or 0)>.5]
    for oc in oldbad:
        line=LineString(oc["line_xz_m"])
        touched=[i for i,x in enumerate(leaves) if not x.get("forced") and not x.get("error") and x["g"].boundary.intersection(line.buffer(.01)).length>.1]
        if len(touched)<2:continue
        union=shapely.union_all([leaves[i]["g"] for i in touched])
        choices=solver.choices(union,near_line=line,neighbors=[x["g"] for i,x in enumerate(leaves) if i not in touched and not x.get("forced") and not x.get("error")])
        best=None
        for score,ll,mm in choices[:10]:
            pp=raw_cells(union,ll)
            if not pp:continue
            rr=[];cc=[];solver.nodes=0
            for child in pp:
                a,b=solver.solve(child);rr+=a;cc+=b
            if any(x["error"] or x["forced"] for x in rr):continue
            trial=[x for i,x in enumerate(leaves) if i not in touched]+rr
            if shape_flags(trial)>shape_flags(leaves):continue
            # Both the exact raw geometry and circles must pass, not previews.
            if any(x["g"].area>35+1e-8 or short_side(x["g"])<2.4-1e-7 or not disk(x["g"])["fits"] for x in rr):continue
            if mm["furniture_intersection_length_m"]>=oc["furniture_intersection_length_m"]-1e-7:continue
            best=(trial,[(ll,"FURNITURE",mm,union)]+cc,mm);break
        if best:
            leaves=best[0];cuts+=best[1]
            audit.append(dict(status="furniture_repositioned",old_cut=oc["id"],before_length_m=oc["furniture_intersection_length_m"],candidate_after_length_m=best[2]["furniture_intersection_length_m"]))
        else:audit.append(dict(status="unresolved",old_cut=oc["id"],reason="NO_SHORTER_FEASIBLE_WITHIN_0_6_OR_OTHER_WALL_AXIS"))
    audit.append(dict(status="candidate_search_summary",entries=solver.audit))
    return stable,leaves,cuts,axis,audit

def process(old,mesh,pf,hs,nav_polys,nav_ys,objects,p,root):
    row=old["source_geometry"];result=copy.deepcopy(old);result["revision"]="own_shape_quality_v6"
    result["blocks"]=[];result["shape_repair_audit"]=[];furniture_by_floor={}
    reserved_ids={b["id"] for b in old["blocks"]}|set(old.get("shape_admission_reserved_ids",[]))
    for floor in row["floors"]:
        fid=floor["floor_id"];fy=floor["floor_y_m"];scope=shape(floor["floor_polygon"])
        nav=nav_scope_at(nav_polys,nav_ys,fy,own_outline(scope).buffer(.3),p)
        furniture=cs.furniture_for(objects,row["region_id"],fy,p,scope);furniture_by_floor[fid]=furniture
        stable,leaves,cuts,axis,audit=repair_floor(old,floor,nav,furniture)
        direct=[]
        for x in leaves:
            if x.get("forced") or x.get("error"):direct.append(x);continue
            other=shapely.union_all([z["g"] for z in leaves if z is not x and not z.get("forced") and not z.get("error")])
            groups=PartConnectivity(nav,other).groups(x["g"])
            for part in groups:
                child=dict(x,g=part)
                if len(groups)>1 and part.area<6:child.update(forced="DETACHED_FRAGMENT",error=None)
                direct.append(child)
        leaves=direct
        result["blocks"]+=stable;apply_cuts(result,cuts,fid,axis);result["shape_repair_audit"]+=audit
        if not leaves:continue
        _,points,clearance,adj,_=sample_navigation(pf,hs,scope,fy,p)
        ref=old.get("floor_overheads",{}).get(fid)
        if not ref or not Path(ref["metadata_path"]).exists() or not Path(ref["image_path"]).exists():
            raise RuntimeError("CPU_FRAME_MISSING_NO_WRITE_TO_OLD_ATLAS: "+fid)
        metadata=json.loads(Path(ref["metadata_path"]).read_text())
        if "images" in metadata:
            sensor=next((z for z in metadata["images"] if z["path"].endswith("_overview.png")),metadata["images"][0])
            sensor=dict(sensor)
            sensor.setdefault("span_m",metadata.get("span_m",2/np.linalg.norm(np.asarray(sensor["projection"])[0,:3])))
        else:
            sensor=dict(path=ref["image_path"],projection=metadata["projection"],span_m=metadata["span_m"])
        rgb=(metadata,sensor,Image.open(ref["image_path"]).convert("RGB"),ref)
        for stable_block in stable:
            if stable_block.get("admission_recovery_local_discard"):
                stable_g=shape(stable_block["floor_polygon_xz_m"])
                stable_black=measured_black(stable_g,fy,rgb,p)
                stable_block["black_fraction"]=stable_black["black_fraction"]
                stable_block["black_measurement"]=stable_black
                stable_block["nav_walkable_area_m2"]=float(nav.intersection(stable_g).area)
        for i,x in enumerate(leaves):
            g=x["g"];template=x.get("original")
            if template and g.symmetric_difference(shape(template["floor_polygon_xz_m"])).area<=1e-8 and not x.get("forced") and not template.get("wide_body_recovery_requires_native_measurement") and not template.get("disk_merge_requires_native_measurement"):
                # A failed retry preserves the full old block record.
                b=copy.deepcopy(template)
                if b["decision"]=="retain" and x.get("error"):
                    b["shape_repair_remaining_reason"]=x["error"]
                result["blocks"].append(b);continue
            forced=x.get("forced");err=x.get("error");ids=np.flatnonzero(shapely.contains_xy(g,points[:,0],points[:,2]))
            witness=placement(mesh,points,clearance,adj,p,ids.tolist()) if not forced and not err else dict(found=False,not_run_reason=forced or err,acoustics="not_run")
            black=measured_black(g,fy,rgb,p)
            kind,evidence=block_type(g,fy,[z for z in objects.instances if "stair" not in z["category"] and z["category"] not in ("step","steps")],[],row)
            short=short_side(g);dc=disk(g);reasons=[];unknown=[]
            if forced:reasons.append(forced)
            if not forced and g.area<6-1e-8:reasons.append("FLOOR_AREA_BELOW_6")
            if g.area>35+1e-8:unknown.append("STILL_ABOVE_AREA_CAP_UNRESOLVED")
            if short<2.4-1e-7 and not forced:reasons.append("SHORT_SIDE_BELOW_2_4")
            if not dc["fits"] and not forced:unknown.append("NO_2_4_M_OWN_ROOM_DISK")
            if kind=="outdoor":reasons.append("OUTDOOR_OR_BALCONY_SEMANTIC_AND_LOCAL_GEOMETRY")
            if black["black_fraction"] is None:unknown.append("SCAN_BLACK_FRACTION_UNVERIFIED")
            elif black["black_fraction"]>.15:reasons.append("SCAN_BLACK_FRACTION_ABOVE_15_PERCENT")
            if not witness["found"] and not forced and not err:reasons.append("PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET")
            if err:unknown.append(err)
            decision="discard" if reasons else "unresolved" if unknown else "retain"
            bid=fresh_room_id(old["house"],old["source_region"],fid,reserved_ids,i)
            b=dict(schema="hm3d_auto_room_shape_v6",id=bid,house=old["house"],source_region=old["source_region"],source_region_id=old["source_region_id"],
                source_region_origins=row["origins"],floor_id=fid,floor_y_m=fy,floor_height_range_m=floor["height_range_m"],
                floor_polygon_xz_m=mapping(g),floor_area_m2=float(g.area),short_side_m=short,nav_walkable_area_m2=float(nav.intersection(g).area),
                nav_grid_point_count=len(ids),black_fraction=black["black_fraction"],black_measurement=black,placement_witness=witness,
                visibility_coverage_fraction=None,visibility=dict(status="not_required",reason="no added subcap visibility split or gate"),visibility_admission_gate=False,
                inscribed_circle=dc,decision=decision,discard_reasons=reasons if decision=="discard" else [],unresolved_reasons=unknown if decision=="unresolved" else [],
                room_type=kind,type_evidence=evidence,cut_ids=[],adjacent_rooms=[],new_room=True,max_room_area_m2=35,acoustics="not_run_per_task",
                measurement_source=row["semantic_source"],area_method=row["ground_measurement"],source_selection=row.get("source_selection"))
            if template and template.get("wide_body_recovery_original_id"):
                b["recovered_from_discard_id"]=template["wide_body_recovery_original_id"]
            if template and template.get("disk_merge_source_ids"):
                b["disk_merge_source_ids"]=template["disk_merge_source_ids"]
            result["blocks"].append(b)
    result["disk_admission_audit"]=enforce_room_disk_admission(result["blocks"])
    # Disk rejection may expose a wide discarded body; recover on raw ground,
    # then re-enter native admission once with the newly planned valid cells.
    if not old.get("post_disk_wide_body_recovery_pass"):
        prepared,recovery=prepare_wide_body_recovery(result,furniture_by_floor)
        if any(x["status"]=="planned_raw_recovery" for x in recovery):
            prepared["post_disk_wide_body_recovery_pass"]=True
            prepared["post_disk_wide_body_recovery_audit"]=recovery
            return process(prepared,mesh,pf,hs,nav_polys,nav_ys,objects,p,root)
        result["post_disk_wide_body_recovery_audit"]=recovery
    # Final certificates exclude all other retained polygons on this floor.
    for b in result["blocks"]:
        if b["decision"]!="retain":continue
        g=shape(b["floor_polygon_xz_m"]);fy=b["floor_y_m"]
        nav=nav_scope_at(nav_polys,nav_ys,fy,own_outline(g).buffer(.3),p)
        other=shapely.union_all([shape(x["floor_polygon_xz_m"]) for x in result["blocks"] if x is not b and x["decision"]=="retain" and x["floor_id"]==b["floor_id"]])
        b["connectivity_certificate"]=PartConnectivity(nav,other).certificate(g)
        b["connectivity_rule"]="distinct_part_seam_nav_v6"
    evaluate_interfaces(result,furniture_by_floor)
    result["retained_new_rooms"]=sum(b["decision"]=="retain" for b in result["blocks"])
    result["status"]="partially_unresolved" if any(b["decision"]=="unresolved" for b in result["blocks"]) else "processed"
    result["area_partition_error_m2"]=abs(sum(b["floor_area_m2"] for b in result["blocks"])-row["floor_area_sum_m2"])
    if result["area_partition_error_m2"]>1e-5:raise RuntimeError("raw floor partition area changed: "+str(result["area_partition_error_m2"]))
    return result

def worker(job,root,out,p):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    assert 10<=os.getpriority(os.PRIO_PROCESS,0)<=15
    t=time.time();root=Path(root);out=Path(out)
    hs=load_native();pf,np_,ny=navmesh_triangles(hs,Path(job["navmesh"]))
    mesh,ray=raw_collision(job["scene_directory"]);objects=load_scene(job["scene_directory"]);status=[]
    for source in job["sources"]:
        old=json.loads(Path(source).read_text());start=time.time()
        faulthandler.dump_traceback_later(120,repeat=True)
        try:r=process(old,mesh,pf,hs,np_,ny,objects,p,root)
        except Exception as e:
            r=copy.deepcopy(old)
            r["v6_attempt_failure"]=dict(reason=repr(e),traceback=traceback.format_exc(),retained_input_unchanged=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
        dump(out/"regions"/Path(source).name,r)
        status.append(dict(source=key(old),seconds=time.time()-start,retained=sum(b["decision"]=="retain" for b in r["blocks"]),error=r.get("v6_attempt_failure")))
        print("V6_REGION",key(old),round(time.time()-start,2),"kept",status[-1]["retained"],"error",bool(status[-1]["error"]),flush=True)
    return dict(house=job["house"],status=status,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),seconds=time.time()-t,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ray_receipt=ray,cpu_only=True,address_space_limit_gib=8)

def interface_worker(job,root,out,p):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    from tools.rooms.room_split_auto.shape_quality_delivery import same_layout
    root=Path(root);out=Path(out)
    objects=load_scene(job["scene_directory"]);status=[]
    for source in job["sources"]:
        path=Path(source);r=json.loads(path.read_text())
        baseline=json.loads((root/"delivery_all_v5/final_v1/regions"/path.name).read_text())
        if r.get("v6_attempt_failure") or same_layout(baseline,r):
            shutil.copyfile(path,out/"regions"/path.name)
            status.append(dict(source=key(r),normalized=False));continue
        by_floor={}
        for floor in r["source_geometry"]["floors"]:
            by_floor[floor["floor_id"]]=cs.furniture_for(objects,r["source_region_id"],floor["floor_y_m"],p,shape(floor["floor_polygon"]))
        evaluate_interfaces(r,by_floor)
        r["interface_revision"]="surviving_collinear_design_spans_v6"
        dump(out/"regions"/path.name,r)
        status.append(dict(source=key(r),normalized=True))
    return dict(house=job["house"],pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,status=status)

def normalize(root,source_attempt,destination,workers=4):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    assert 1<=workers<=7
    root=Path(root);out=root/"delivery_all_v6"/destination;out.mkdir();(out/"regions").mkdir()
    pp=json.loads((root/"delivery_all_v5/revision_plan.json").read_text())["parameters"]
    jobs={}
    for f in (root/"delivery_all_v6"/source_attempt/"regions").glob("*.json"):
        d=json.loads(f.read_text());row=d["source_geometry"]
        j=jobs.setdefault(d["house"],dict(house=d["house"],scene_directory=row["scene_directory"],sources=[]))
        j["sources"].append(str(f))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(interface_worker,j,root,out,pp) for j in jobs.values()]
        for f in as_completed(futures):receipts.append(f.result())
    dump(out/"native_receipt.json",receipts)
    dump(out/"completed.json",dict(source_attempt=source_attempt,workers=workers,compute_process_upper_bound=workers+2,total_address_space_upper_bound_gib=8*(workers+2)))
    print("V6_INTERFACES_DONE",destination,flush=True)

def run(root,attempt="attempt_v1",workers=4,pilot=False,sources_json=None,input_dir=None):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    root=Path(root);out=root/"delivery_all_v6"/attempt;out.mkdir();(out/"regions").mkdir()
    plan=json.loads((root/"delivery_all_v6/revision_plan_v1.json").read_text())
    pp=json.loads((root/"delivery_all_v5/revision_plan.json").read_text())["parameters"]
    targets=set(json.loads(Path(sources_json).read_text())) if sources_json else set(plan["attempted_sources"])
    if pilot:targets={s for s in targets if any("_"+n+"_" in s for n in ["00016","00210","00250","00466","00732","00876"])}
    assert 1<=workers<=7,"parent and resource tracker must fit within nine total processes"
    assert targets<=set(plan["attempted_sources"]),"sources outside authorization"
    jobs={}
    source_root=Path(input_dir) if input_dir else root/"delivery_all_v5/final_v1/regions"
    for f in source_root.glob("*.json"):
        d=json.loads(f.read_text())
        if key(d) not in targets:continue
        row=d["source_geometry"];j=jobs.setdefault(d["house"],dict(house=d["house"],scene_directory=row["scene_directory"],navmesh=row["navmesh_source"],sources=[]))
        j["sources"].append(str(f))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        future={pool.submit(worker,j,root,out,pp):j for j in jobs.values()}
        for f in as_completed(future):
            try:receipts.append(f.result())
            except Exception as e:
                j=future[f];receipts.append(dict(house=j["house"],error=repr(e),traceback=traceback.format_exc()))
                for s in j["sources"]:
                    dst=out/"regions"/Path(s).name
                    if not dst.exists():
                        fallback=json.loads(Path(s).read_text())
                        fallback["v6_attempt_failure"]=dict(reason=repr(e),retained_input_unchanged=True,worker_failure=True)
                        dump(dst,fallback)
    dump(out/"native_receipt.json",receipts)
    dump(out/"completed.json",dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,compute_process_upper_bound=workers+2,total_address_space_upper_bound_gib=8*(workers+2),sources=len(targets),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0)))
    print("V6_ATTEMPT_DONE",str(out),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--attempt",default="attempt_v1");a.add_argument("--workers",type=int,default=4);a.add_argument("--pilot",action="store_true");a.add_argument("--sources-json");a.add_argument("--input-dir");a.add_argument("--normalize-source");v=a.parse_args()
    normalize(v.root,v.normalize_source,v.attempt,v.workers) if v.normalize_source else run(v.root,v.attempt,v.workers,v.pilot,v.sources_json,v.input_dir)
