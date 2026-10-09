"""Connected CPU HM3D partitions using finite straight or three-segment cuts."""
from __future__ import annotations
import argparse,copy,datetime,json,math,os,resource,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from functools import lru_cache
import numpy as np
import shapely
from shapely.geometry import shape,mapping,Polygon,LineString,Point,GeometryCollection
from shapely.ops import split,nearest_points
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side,infer_type
from tools.rooms.room_selection.measurements import navmesh_triangles,load_scene,layer_furniture
from tools.rooms.room_selection.navigation import sample_navigation,placement
from tools.rooms.room_split_auto.pipeline import load_native,raw_collision,stair_partition,nav_scope_at,block_type,measured_black,overhead_for,finalize_interfaces,adjacency
from tools.rooms.room_split_auto.scene import structural_instances
from tools.rooms.room_split_auto.contours import body_width
from tools.rooms.room_split_auto.visibility import assess_visibility,grid_atoms
from tools.rooms.room_split_auto.atlas import floor_overhead
from tools.rooms.room_split_auto.camera_view_proof import filled_footprint

filled_footprint=lru_cache(maxsize=128)(filled_footprint)

class FurnitureList(list):
    def __init__(self,items):
        super().__init__(items)
        self.union=shapely.union_all([x['geometry'] for x in self]) if self else GeometryCollection()
        self.tree=shapely.STRtree([x['geometry'] for x in self]) if self else None

def dump(p,d):
    with Path(p).open("x") as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2)

def connection_count(g,width=.6):
    # The input must itself be one exact measured ground polygon. Erosion alone
    # must not ignore another small disconnected polygon that has no clear core.
    outline=filled_footprint(g)
    raw=list(polygons(outline))
    if len(raw)!=1:return len(raw)
    opened=outline.buffer(-width/2,quad_segs=8).buffer(width/2,quad_segs=8)
    return len([x for x in polygons(opened) if x.area>1e-8])

def split_geometry(g,line):
    try:parts=list(polygons(split(g,line)))
    except Exception:return []
    if len(parts)<2:return []
    if abs(sum(p.area for p in parts)-g.area)>1e-6:return []
    return sorted(parts,key=lambda x:(-x.area,x.bounds))

def cutting_line(g,angle,offset):
    a=math.radians(angle);normal=np.array([math.cos(a),math.sin(a)]);unit=np.array([-normal[1],normal[0]])
    centre=np.array([(g.bounds[0]+g.bounds[2])/2,(g.bounds[1]+g.bounds[3])/2])
    centre+=normal*(offset-float(centre@normal))
    extent=max(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1])*2+2
    return LineString([centre-unit*extent,centre+unit*extent])

def cut_measure(line,g,furniture,band_width=.25):
    scope=filled_footprint(g)
    geom=furniture.union if isinstance(furniture,FurnitureList) else shapely.union_all([x["geometry"] for x in furniture]) if furniture else GeometryCollection()
    interior=line.intersection(scope)
    exact=interior.intersection(geom)
    band=line.buffer(band_width/2,cap_style="flat",join_style="mitre").intersection(scope)
    overlaps=[]
    for x in ([furniture[int(i)] for i in furniture.tree.query(band)] if isinstance(furniture,FurnitureList) and furniture.tree is not None else furniture):
        length=interior.intersection(x["geometry"]).length
        area=band.intersection(x["geometry"]).area
        if length>1e-8 or area>1e-8:overlaps.append(dict(instance_id=x["instance_id"],category=x["category"],line_overlap_length_m=float(length),cut_band_overlap_area_m2=float(area)))
    return {"furniture_intersection_length_m":float(exact.length),"furniture_intersection_area_m2":float(band.intersection(geom).area),"intersection_area_band_width_m":band_width,"furniture_intersections":overlaps,"geometry_source":"shape-preserving all-scene semantic furniture projections in height slab; cut strip width equals frozen grid step"}

def connected_parts(g,furniture=(),width=.6,max_depth=12):
    """Explode raw islands first; resolve narrow necks with finite straight cuts."""
    out=[];cuts=[]
    pending=[(x,0) for x in sorted(polygons(g),key=lambda x:-x.area)]
    while pending:
        part,depth=pending.pop(0);count=connection_count(part,width)
        if count<=1 or depth>=max_depth:
            out.append(part);continue
        cores=sorted(polygons(filled_footprint(part).buffer(-width/2,quad_segs=8)),key=lambda x:-x.area)
        if len(cores)<2:
            out.append(part);continue
        a,b=nearest_points(cores[0],cores[1]);delta=np.array([b.x-a.x,b.y-a.y])
        angle=math.degrees(math.atan2(delta[1],delta[0]));mid=np.array([(a.x+b.x)/2,(a.y+b.y)/2])
        normal=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
        choices=[]
        for shift in [0.,-.125,.125,-.25,.25]:
            line=cutting_line(part,angle,float(mid@normal)+shift)
            children=split_geometry(part,line)
            if not children:continue
            child_counts=[connection_count(x,width) for x in children]
            if max(child_counts)>=count:continue
            m=cut_measure(line,part,furniture)
            choices.append(((m["furniture_intersection_length_m"]>1e-7,m["furniture_intersection_area_m2"],sum(child_counts),-min(x.area for x in children)),line,children,m))
        if not choices:out.append(part);continue
        _,line,children,m=min(choices,key=lambda x:x[0]);cuts.append((line,m,"narrow_neck_lt_0_6_separation"))
        pending[:0]=[(x,depth+1) for x in children]
    return out,cuts

def line_candidates(g,furniture,cap,allow_small=False,max_dogleg_axis=20):
    angles={0.,90.}
    rect=g.minimum_rotated_rectangle
    if rect.geom_type=="Polygon":
        p=np.asarray(rect.exterior.coords);v=p[1]-p[0]
        angle=math.degrees(math.atan2(v[1],v[0]))%180
        angles.update([round(angle,3),round((angle+90)%180,3)])
    # Added directions are deterministic search resolution, not admission rules.
    angles.update([15.,30.,45.,60.,75.,105.,120.,135.,150.,165.])
    coords=np.array([[g.bounds[0],g.bounds[1]],[g.bounds[0],g.bounds[3]],[g.bounds[2],g.bounds[1]],[g.bounds[2],g.bounds[3]]])
    choices=[];target=g.area/max(2,math.ceil((g.area-1e-8)/cap));seen=set()
    def add(line):
        key=tuple(np.round(np.asarray(line.coords).ravel(),4))
        if key in seen:return
        seen.add(key);children=split_geometry(g,line)
        main=[x for x in children if x.area>=6-1e-8]
        if len(main)<2:return
        if any(short_side(x)<2.4-1e-7 for x in main):return
        m=cut_measure(line,g,furniture)
        rooms=sum(max(1,math.ceil((x.area-1e-8)/cap)) for x in main)
        balanced=abs(min(x.area for x in main)-target)
        score=(m["furniture_intersection_length_m"]>1e-7,m["furniture_intersection_area_m2"] if m["furniture_intersection_length_m"]>1e-7 else 0.,rooms,balanced,m["furniture_intersection_area_m2"],line.intersection(g).length)
        choices.append((score,line,children,m))
    for angle in sorted(angles):
        normal=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))]);vals=coords@normal
        for offset in np.arange(vals.min()+.5,vals.max()-.5,.25):add(cutting_line(g,angle,float(offset)))
    # A monotone three-segment dogleg can route around a furniture group.
    if not any(c[0][0]==False for c in choices) and furniture:
        x0,z0,x1,z1=g.bounds
        boxes=[x["geometry"].bounds for x in furniture if not x["geometry"].intersection(g).is_empty]
        xs=sorted({round((x0+x1)/2,3),*[round(v,3) for b in boxes for v in [b[0]-.25,b[2]+.25]]})
        zs=sorted({round((z0+z1)/2,3),*[round(v,3) for b in boxes for v in [b[1]-.25,b[3]+.25]]})
        xs=[x for x in xs if x0+.5<x<x1-.5];zs=[z for z in zs if z0+.5<z<z1-.5]
        if max_dogleg_axis<20:
            # Evenly cover the complete object-boundary axis under a CPU budget.
            xs=[xs[int(i)] for i in np.linspace(0,len(xs)-1,min(max_dogleg_axis,len(xs)),dtype=int)]
            zs=[zs[int(i)] for i in np.linspace(0,len(zs)-1,min(max_dogleg_axis,len(zs)),dtype=int)]
        else:xs=xs[:20];zs=zs[:20]
        for a in xs:
            for b in xs:
                if abs(a-b)<.25:continue
                for z in zs:add(LineString([(a,z0-2),(a,z),(b,z),(b,z1+2)]))
        for a in zs:
            for b in zs:
                if abs(a-b)<.25:continue
                for x in xs:add(LineString([(x0-2,a),(x,a),(x,b),(x1+2,b)]))
    return sorted(choices,key=lambda x:x[0]),len(choices)

class VisibilityCutter:
    def __init__(self,mesh,points,nav,p,parameter,cap,furniture):
        self.mesh=mesh;self.points=points;self.nav=nav;self.p=p;self.parameter=parameter;self.cap=cap;self.furniture=furniture;self.cache={};self.attempts=0;self.audit=[]
    def visibility(self,g):
        key=g.wkb
        if key not in self.cache:
            ids=np.flatnonzero(shapely.contains_xy(g,self.points[:,0],self.points[:,2]));pts=self.points[ids]
            _,_,weights=grid_atoms(g,self.nav.intersection(g),pts,self.p["grid_step_m"])
            self.cache[key]=assess_visibility(self.mesh,pts,weights,self.p,self.parameter["coverage"],self.parameter["max_distance"])
        return self.cache[key]
    def solve(self,g,depth=0):
        if g.area<=self.cap+1e-8:
            vis=self.visibility(g)
            if vis["meets_visibility"]:return [(g,vis)],[],None
        if depth>=8 or g.area<12-1e-8 or self.attempts>=36:return [(g,None)],[],"VISIBILITY_STRAIGHT_PARTITION_UNRESOLVED"
        choices,count=line_candidates(g,self.furniture,self.cap)
        if not choices:return [(g,None)],[],"NO_FEASIBLE_STRAIGHT_OR_THREE_SEGMENT_CUT"
        best=None
        for score,line,children,m in choices[:5]:
            if self.attempts>=36:break
            self.attempts+=1;leaves=[];cuts=[];failed=None
            for child in children:
                parts,conncuts=connected_parts(child,self.furniture)
                for c,cm,reason in conncuts:cuts.append((c,cm,"narrow",reason,count))
                for part in parts:
                    if part.area<6:leaves.append((part,None));continue
                    ll,cc,err=self.solve(part,depth+1);leaves.extend(ll);cuts.extend(cc)
                    if err:failed=err;break
                if failed:break
            if failed:continue
            cuts.append((line,m,"visibility","straight_or_three_segment_visibility_partition",count))
            value=(len([x for x,v in leaves if x.area>=6]),sum(c[1]["furniture_intersection_area_m2"] for c in cuts))
            if best is None or value<best[0]:best=(value,leaves,cuts)
            if value[0]==math.ceil((g.area-1e-8)/self.cap):break
        self.audit.append({"floor_area_m2":g.area,"candidate_count":count,"attempts_total":self.attempts,"feasible":best is not None,"candidate_minimum_furniture_overlap_area_m2":min(x[3]["furniture_intersection_area_m2"] for x in choices),"global_minimum_unverified":True})
        if best:return best[1],best[2],None
        return [(g,None)],[],"VISIBILITY_STRAIGHT_PARTITION_NO_VALID_CONSTRUCTION_WITHIN_CPU_BUDGET"


def structural_partition(scope,doors,floor_y,width_min,width_max,furniture):
    """Detect walls/neck widths on filled outlines; split measured holed ground."""
    from tools.rooms.room_split_auto.contours import chord,medial_candidates,split_at_chord
    parts=[scope];cuts=[];audits=[]
    for door in sorted(doors,key=lambda d:d["instance_id"]):
        rec={k:v for k,v in door.items() if k!="triangles"};lo,hi=door["height_range_m"]
        if lo>floor_y+1.8 or hi<floor_y+.3:rec["status"]="other_floor";audits.append(rec);continue
        applied=False
        for i,g in enumerate(parts):
            candidate=chord(filled_footprint(g),door["centre_xz_m"],np.arange(door["axis_angle_deg"]-30,door["axis_angle_deg"]+31,3))
            if candidate is None or candidate["width_m"]>max(door["extent_xz_m"])*1.7+.4:continue
            children=split_at_chord(g,candidate["line"],.01)
            if not children:continue
            cuts.append(dict(type="door",line_xz_m=list(candidate["line"].coords),width_m=candidate["width_m"],semantic_instance_id=door["instance_id"],semantic_category=door["category"],semantic_source=door["semantic_source"],geometry_source="semantic door plus filled exterior contour; measured holed ground split",side_area_m2=[x.area for x in children]))
            parts[i:i+1]=children;applied=True;break
        rec["status"]="cut_applied" if applied else "unreliable_or_not_internal; narrow_only_fallback";audits.append(rec)
    pending=list(parts);parts=[]
    while pending:
        g=pending.pop(0);candidates=medial_candidates(filled_footprint(g),width_min,width_max) if g.area>=12 else []
        feasible=[]
        for c in candidates:
            children=split_at_chord(g,c["line"],6.)
            if not children:continue
            m=cut_measure(c["line"],g,furniture)
            feasible.append(((m["furniture_intersection_length_m"]>1e-7,m["furniture_intersection_area_m2"] if m["furniture_intersection_length_m"]>1e-7 else 0.,c["width_m"],-min(x.area for x in children)),c,children))
        if not feasible:parts.append(g);continue
        _,c,children=min(feasible,key=lambda x:x[0])
        cuts.append(dict(type="narrow",line_xz_m=list(c["line"].coords),width_m=c["width_m"],side_area_m2=[x.area for x in children],geometry_source="medial axis of filled exterior; minimum furniture crossing among finite neck candidates; original holed ground split",semantic_source=None,tested_neck_candidates=len(feasible)))
        pending.extend(children)
    return parts,cuts,audits


def furniture_for(scene,rid,fy,p,scope=None):
    # Cut guidance uses all scene furniture, including neighbours' annotations.
    # Floor geometry is never reduced by furniture, and placement remains frozen.
    from tools.rooms.room_screening.geometry import shape_preserving_projected_footprint
    items=[];bounds=scope.bounds if scope is not None and not scope.is_empty else None
    for x in scene.instances:
        category=x["category"]
        if x["role"]!="blocker" or category in ("floor","step","steps") or any(t in category for t in ["wall","ceiling","door","window","stair"]):continue
        tri=x["triangles"]
        if bounds:
            xs=tri[:,:,0];zs=tri[:,:,2]
            if xs.max()<bounds[0]-.125 or xs.min()>bounds[2]+.125 or zs.max()<bounds[1]-.125 or zs.min()>bounds[3]+.125:continue
        g=shape_preserving_projected_footprint(tri,fy,2.4)
        if not g.is_empty:items.append(dict(instance_id=x["instance_id"],region_id=x["region_id"],category=category,geometry=g))
    return FurnitureList(items)

def partition_stair_aliases(scope,floor_y,markers,height_separation):
    normalized=[dict(s,category="stairs") if s["category"] in ("step","steps") else s for s in markers if "stair" in s["category"] or s["category"] in ("step","steps")]
    return stair_partition(scope,floor_y,normalized,height_separation)

def process(old,mesh,pf,hs,nav_polys,nav_ys,objects,structural,door_info,p,parameter,cap,cache,root):
    row=old["source_geometry"];result=copy.deepcopy(old)
    result.update(blocks=[],cut_lines=[],status="processed",merge_audit=[],connectivity_separation_audit=[],straight_partition_audit=[],stair_separation_audit=[],parameter=dict(parameter,max_room_area_m2=cap),revision="connected_straight_v3")
    result["floor_overheads"]={};index=0
    for floor in row["floors"]:
        scope=shape(floor["floor_polygon"]);fy=floor["floor_y_m"];fid=floor["floor_id"]
        rgb=floor_overhead(row,floor,cache,root/"overhead_cpu_atlas_v3",overhead_for(row,cache))
        result["floor_overheads"][fid]=rgb[3] if rgb else None
        nav=nav_scope_at(nav_polys,nav_ys,fy,scope,p)
        _,points,clearance,adj,_=sample_navigation(pf,hs,scope,fy,p)
        furniture=furniture_for(objects,row["region_id"],fy,p,scope) if scope.area>=6 else FurnitureList([])
        error="SEMANTIC_PALETTE_AMBIGUOUS" if row.get("ambiguous_semantic_palette") else "ZERO_AREA_SEMANTIC_GROUND_FACES" if door_info["invalid_ground_zero_area_faces_by_region"].get(row["region_id"],0) else None
        normal,stairs,stair_audit=partition_stair_aliases(scope,fy,structural,p["floor_height_separation_m"])
        result["stair_separation_audit"].extend(dict(x,floor_id=fid,periphery_expansion_m=0.,original_semantic_category=next((m["category"] for m in structural if m["instance_id"]==x["semantic_instance_id"]),x["category"])) for x in stair_audit)
        result.setdefault("stair_exact_exclusion",[]).append(dict(floor_id=fid,geometry_xz_m=mapping(shapely.union_all(stairs)),area_m2=sum(x.area for x in stairs),periphery_expansion_m=0.))
        final=[dict(g=x,forced="STAIRS",visibility=None,error=None) for x in stairs]
        def cut_record(line,m,kind,stage,candidate_count=None):
            cid=fid+f'_L{len(result["cut_lines"]):03d}'
            rec=dict(id=cid,floor_id=fid,type=kind,line_xz_m=np.asarray(line.coords).tolist(),line_geometry_xz_m=mapping(line),segment_count=len(line.coords)-1,stage=stage,**m)
            if candidate_count is not None:rec["tested_candidate_count"]=candidate_count
            rec["fallback_crosses_furniture"]=m["furniture_intersection_length_m"]>1e-7
            result["cut_lines"].append(rec)
        if error:final.append(dict(g=normal,forced=None,visibility=None,error=error))
        else:
            components,conncuts=connected_parts(normal,furniture)
            for line,m,stage in conncuts:cut_record(line,m,"narrow",stage)
            components.sort(key=lambda x:-x.area)
            for part in components:
                if part.area<6:
                    final.append(dict(g=part,forced="DETACHED_FRAGMENT",visibility=None,error=None));continue
                pieces,cuts,audits=structural_partition(part,[s for s in structural if "door" in s["category"]],fy,parameter["width_min"],parameter["width_max"],furniture)
                result.setdefault("door_instance_audit_v3",[]).extend(audits)
                for c in cuts:
                    line=LineString(c["line_xz_m"]);m=cut_measure(line,part,furniture)
                    cut_record(line,m,c["type"],"door_and_narrow_cues")
                    result["cut_lines"][-1].update({k:v for k,v in c.items() if k not in ["id","line_xz_m","type"]})
                for piece in pieces:
                    connected,extra=connected_parts(piece,furniture)
                    for line,m,stage in extra:cut_record(line,m,"narrow",stage)
                    for child in connected:
                        if child.area<6:final.append(dict(g=child,forced="DETACHED_FRAGMENT",visibility=None,error=None));continue
                        if child.area<=cap+1e-8:final.append(dict(g=child,forced=None,visibility=None,error=None));continue
                        cutter=VisibilityCutter(mesh,points,nav,p,parameter,cap,furniture)
                        leaves,newcuts,err=cutter.solve(child)
                        result["straight_partition_audit"].extend(dict(x,floor_id=fid) for x in cutter.audit)
                        for line,m,kind,stage,n in newcuts:cut_record(line,m,kind,stage,n)
                        for g,vis in leaves:final.append(dict(g=g,forced="DETACHED_FRAGMENT" if g.area<6 else None,visibility=vis,error=err))
        # Connected postcondition is applied after every selected cut. Large
        # detached pieces become fresh candidates; no nearest-seed island ownership.
        expanded=[]
        for item in final:
            if item["forced"]=="STAIRS" or item["error"]:expanded.append(item);continue
            comps,extra=connected_parts(item["g"],furniture)
            for line,m,stage in extra:cut_record(line,m,"narrow",stage)
            for g in comps:expanded.append(dict(item,g=g,forced="DETACHED_FRAGMENT" if g.area<6 else item["forced"],visibility=None if len(comps)>1 else item["visibility"]))
        final=expanded
        # Wide-open merges are allowed only if connectivity and visibility pass.
        changed=True
        while changed and not error:
            changed=False
            for i in range(len(final)):
                if changed:break
                if final[i]["forced"] or final[i]["error"]:continue
                for j in range(i+1,len(final)):
                    if final[j]["forced"] or final[j]["error"]:continue
                    a,b=final[i]["g"],final[j]["g"]
                    if a.area+b.area>cap+1e-8:continue
                    interface=a.boundary.intersection(b.boundary)
                    if interface.length<=1.6:continue
                    if any(c["type"]=="door" and interface.intersection(LineString(c["line_xz_m"]).buffer(.001)).length>.01 for c in result["cut_lines"] if c["floor_id"]==fid):continue
                    merged=a.union(b)
                    if connection_count(merged)!=1:continue
                    vc=VisibilityCutter(mesh,points,nav,p,parameter,cap,furniture);vis=vc.visibility(merged)
                    result["merge_audit"].append(dict(floor_id=fid,area_m2=merged.area,opening_width_m=interface.length,meets_visibility=vis["meets_visibility"],merged=vis["meets_visibility"]))
                    if vis["meets_visibility"]:
                        final[i]=dict(g=merged,forced=None,visibility=vis,error=None);final.pop(j);changed=True;break
        for item in final:
            g=item["g"];vis=item["visibility"];ids=np.flatnonzero(shapely.contains_xy(g,points[:,0],points[:,2]));navg=nav.intersection(g)
            witness=placement(mesh,points,clearance,adj,p,ids.tolist()) if not item["forced"] else dict(found=False,not_run_reason=item["forced"],acoustics="not_run")
            black=measured_black(g,fy,rgb,p)
            kind,evidence=block_type(g,fy,[x for x in objects.instances if "stair" not in x["category"] and x["category"] not in ("step","steps")],[x for x in structural if "stair" not in x["category"] and x["category"] not in ("step","steps")],row)
            if kind=="stairs":kind="unknown";evidence=["stair semantic footprint already excluded; adjacent flat ground re-evaluated"]
            if item["forced"]=="STAIRS":kind="stairs";evidence=["direct semantic stair ground intersection only"]
            reasons=[];unknown=[]
            if item["forced"]:reasons.append(item["forced"])
            if g.area<6-1e-8 and not item["forced"]:reasons.append("FLOOR_AREA_BELOW_6")
            if g.area>cap+1e-8:unknown.append("STILL_ABOVE_AREA_CAP_UNRESOLVED")
            short=short_side(g);width=body_width(filled_footprint(g)) if g.area>=6 else 0.
            rect=g.minimum_rotated_rectangle;long=max(np.linalg.norm(np.diff(np.array(rect.exterior.coords),axis=0),axis=1)) if rect.geom_type=="Polygon" else 0.
            aspect=long/short if short else None;conn=connection_count(g)
            if short<2.4-1e-8:reasons.append("SHORT_SIDE_BELOW_2_4")
            if width<1.5 and aspect and aspect>=3:reasons.append("CORRIDOR_BODY_WIDTH_BELOW_1_5")
            if kind=="outdoor":reasons.append("OUTDOOR_OR_BALCONY_SEMANTIC_AND_LOCAL_GEOMETRY")
            if conn!=1 and not item["forced"]:unknown.append("CONNECTIVITY_0_6_UNRESOLVED")
            if black["black_fraction"] is None:unknown.append("SCAN_BLACK_FRACTION_UNVERIFIED")
            elif black["black_fraction"]>.15:reasons.append("SCAN_BLACK_FRACTION_ABOVE_15_PERCENT")
            if not witness["found"] and not item["forced"]:reasons.append("PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET")
            if item["error"]:unknown.append(item["error"])
            decision="discard" if reasons else "unresolved" if unknown else "retain"
            bid=row["house"]+"__"+row["room_label"]+"__"+fid+f'__S{index:03d}';index+=1
            result["blocks"].append(dict(schema="hm3d_auto_room_split_v1",id=bid,house=row["house"],source_region_id=row["region_id"],source_region=row["room_label"],source_region_origins=row["origins"],floor_id=fid,floor_y_m=fy,floor_height_range_m=floor["height_range_m"],floor_polygon_xz_m=mapping(g),floor_area_m2=g.area,short_side_m=short,body_width_m_proxy=width,aspect_ratio=aspect,nav_walkable_area_m2=navg.area,nav_grid_point_count=len(ids),black_fraction=black["black_fraction"],black_measurement=black,visibility_coverage_fraction=vis["coverage_fraction"] if vis else None,visibility=vis or dict(status="not_required",reason="only residual blocks above area cap enter visibility splitting"),placement_witness=witness,decision=decision,discard_reasons=reasons if decision=="discard" else [],unresolved_reasons=unknown if decision=="unresolved" else [],unverified_diagnostics=unknown,room_type=kind,type_evidence=evidence,cut_ids=[],adjacent_rooms=[],new_room=True,acoustics="not_run_per_task",measurement_source=row["semantic_source"],area_method=row["ground_measurement"],connectivity_core_count=conn,connectivity_width_m=.6,max_room_area_m2=cap))
    result["retained_new_rooms"]=sum(x["decision"]=="retain" for x in result["blocks"])
    result["status"]="partially_unresolved" if any(x["decision"]=="unresolved" for x in result["blocks"]) else "processed"
    result["area_partition_error_m2"]=abs(sum(x["floor_area_m2"] for x in result["blocks"])-row["floor_area_sum_m2"])
    finalize_interfaces(result["blocks"],result["cut_lines"]);adjacency(result["blocks"],result["cut_lines"])
    return result

def worker(job,root,out,p,parameter,cap,cache):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    started=time.time();root=Path(root);out=Path(out)
    hs=load_native();pf,nav_polys,nav_ys=navmesh_triangles(hs,Path(job["navmesh"]))
    mesh,ray_receipt=raw_collision(job["scene_directory"])
    structural,door_info=structural_instances(job["scene_directory"]);objects=load_scene(job["scene_directory"])
    marker_ids={m["instance_id"] for m in structural}
    structural.extend(dict(x,semantic_source=str(Path(job["scene_directory"])/(Path(job["scene_directory"]).name.split("-",1)[1]+".semantic.glb"))) for x in objects.instances if x["category"] in ("step","steps") and x["instance_id"] not in marker_ids)
    statuses=[]
    for source in job["sources"]:
        old=json.loads(Path(source).read_text())
        try:result=process(old,mesh,pf,hs,nav_polys,nav_ys,objects,structural,door_info,p,parameter,cap,cache,root)
        except Exception as e:
            result=copy.deepcopy(old);result.update(status="unresolved",revision="connected_straight_v3",unresolved_reasons=[type(e).__name__+": "+str(e)],traceback=traceback.format_exc())
            result["cut_lines"]=[];result["retained_new_rooms"]=0
            for b in result["blocks"]:
                b["decision"]="unresolved";b["discard_reasons"]=[];b["unresolved_reasons"]=["V3_PROCESS_FAILURE"];b["new_room"]=True
        dump(out/"regions"/Path(source).name,result);statuses.append(dict(source=old["source_region"],status=result["status"]))
        print("V3_REGION",job["house"],old["source_region"],result["status"],flush=True)
    return dict(house=job["house"],statuses=statuses,seconds=time.time()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ray_receipt=ray_receipt,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0))

def run(root,workers=8,cap=50.):
    resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    root=Path(root);out=root/"delivery_all_v3";out.mkdir(exist_ok=False);(out/"regions").mkdir();(out/"rooms").mkdir()
    if not (root/"size_gallery_30_50_v1/COMPLETED.txt").exists():raise RuntimeError("A gallery must finish before B")
    plan=json.loads((root/"processing_plan_v1.json").read_text());p=plan["parameters"];parameter=json.loads((root/"selected_parameter_v1.json").read_text())["selected_parameter"]
    by_house={j["house"]:j for j in plan["jobs"]};jobs={}
    for path in sorted((root/"delivery_all_v2/regions").glob("*.json")):
        d=json.loads(path.read_text())
        if d["source_floor_area_m2"]<=cap+1e-8:
            dump(out/"regions"/path.name,d)
        else:
            if d["house"] not in jobs:jobs[d["house"]]={k:v for k,v in by_house[d["house"]].items() if k!="rows"};jobs[d["house"]]["sources"]=[]
            jobs[d["house"]]["sources"].append(str(path))
    dump(out/"revision_plan.json",dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source="delivery_all_v2",max_room_area_m2=cap,workers=workers,cpu_only=True,nice=10,max_compute_processes=workers+1,max_address_space_gib=10*(workers+1),unchanged_policy="Original regions <=cap copied byte-equivalent JSON semantics; no visibility or connectivity rejection added",jobs=list(jobs.values())))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,j,root,out,p,parameter,cap,plan["overhead_cache"]):j["house"] for j in jobs.values()}
        for f in as_completed(futures):
            try:receipts.append(f.result())
            except Exception as e:receipts.append(dict(house=futures[f],error=repr(e),traceback=traceback.format_exc()))
    # Every authorized source remains represented even when a native house job fails.
    for source in sorted((root/"delivery_all_v2/regions").glob("*.json")):
        if (out/"regions"/source.name).exists():continue
        failed=json.loads(source.read_text());failed.update(status="unresolved",revision="connected_straight_v3",retained_new_rooms=0,cut_lines=[],unresolved_reasons=["NATIVE_HOUSE_JOB_FAILURE"])
        for b in failed["blocks"]:
            b.update(decision="unresolved",discard_reasons=[],unresolved_reasons=["NATIVE_HOUSE_JOB_FAILURE"],new_room=True)
        dump(out/"regions"/source.name,failed)
    dump(out/"native_receipt.json",receipts)
    for path in (out/"regions").glob("*.json"):
        d=json.loads(path.read_text())
        for b in d.get("blocks",[]):dump(out/"rooms"/(b["id"]+".json"),b)
    dump(out/"completed.json",dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),native_house_jobs=len(jobs),house_receipts=len(receipts),source_regions=len(list((out/"regions").glob("*.json")))))
    print("V3_NATIVE_DONE",len(receipts),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--workers",type=int,default=8);p.add_argument("--max-room-area-m2",type=float,default=50.)
    a=p.parse_args();run(a.root,a.workers,a.max_room_area_m2)
