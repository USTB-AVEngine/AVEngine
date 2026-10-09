"""CPU cap-35 room revision: orthogonal cuts, room disks, and narrow-only corridors.

Only cut inputs authorized by the v4 scope. Geometry area always uses the exact
semantic-ground polygons including holes. Filled exteriors are geometric shape
proxies for connectivity and cut selection, never replacements for output area.
"""
from __future__ import annotations
import argparse,copy,datetime,json,math,multiprocessing,os,resource,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
import shapely
from shapely.affinity import rotate
from shapely.geometry import Polygon,LineString,Point,GeometryCollection,shape,mapping
from shapely.ops import nearest_points
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.measurements import load_scene,navmesh_triangles
from tools.rooms.room_selection.navigation import sample_navigation,placement
from tools.rooms.room_split_auto import connected_split as cs
from tools.rooms.room_split_auto.contours import chord,split_at_chord,body_width
from tools.rooms.room_split_auto.pipeline import (
 load_native,raw_collision,nav_scope_at,block_type,measured_black,
 finalize_interfaces,adjacency,overhead_for)
from tools.rooms.room_split_auto.scene import structural_instances
from tools.rooms.room_split_auto.visibility import assess_visibility,grid_atoms
from tools.rooms.room_split_auto.atlas import floor_overhead

CAP=35.
MIN=6.
DIAMETER=2.4
WIDTH=.6
dump=cs.dump
outline=cs.filled_footprint

@lru_cache(maxsize=512)
def opened(g,width=WIDTH):
    # Match Claude's independent checker: fill every hole, MITRE joins.
    return outline(g).buffer(-width/2,join_style=2).buffer(width/2,join_style=2)

def connection_count(g,width=WIDTH):
    raw=list(polygons(g))
    if len(raw)!=1:return len(raw)
    return len([p for p in polygons(opened(g,width)) if p.area>1e-8])

@lru_cache(maxsize=512)
def room_circle(g):
    """A certified radius in filled exterior; exact boundary distance verifies it."""
    ext=outline(g)
    if ext.is_empty:return dict(fits=False,diameter_m=DIAMETER,radius_m=0.,centre_xz_m=None)
    eroded=ext.buffer(-DIAMETER/2,join_style=1)
    candidates=[]
    if not eroded.is_empty:
        candidates=[p.representative_point() for p in polygons(eroded)]
    # Also handle exactly 2.4 m wide shapes and near-threshold/complex boundaries.
    if not candidates or max(ext.boundary.distance(p) for p in candidates)<DIAMETER/2-1e-8:
        try:
            line=shapely.maximum_inscribed_circle(ext,tolerance=.002)
            if not line.is_empty:candidates.append(Point(line.coords[0]))
        except (shapely.GEOSException,ValueError):pass
    if not candidates:return dict(fits=False,diameter_m=DIAMETER,radius_m=0.,centre_xz_m=None)
    centre=max(candidates,key=lambda p:ext.boundary.distance(p))
    radius=float(ext.boundary.distance(centre)) if ext.covers(centre) else 0.
    return dict(fits=radius>=DIAMETER/2-1e-7,diameter_m=DIAMETER,
                radius_m=radius,centre_xz_m=[centre.x,centre.y],
                shape_basis="filled exterior only; raw holed area is unchanged")

def main_axis(g):
    parts=list(polygons(outline(g)))
    if not parts:return 0.
    rect=max(parts,key=lambda p:p.area).minimum_rotated_rectangle
    xy=np.asarray(rect.exterior.coords)
    delta=np.diff(xy,axis=0);v=delta[int(np.argmax(np.linalg.norm(delta,axis=1)))]
    return math.degrees(math.atan2(v[1],v[0]))%90.

def angle_errors(line,axis):
    xy=np.asarray(line.coords,float);delta=np.diff(xy,axis=0);ans=[]
    for v in delta:
        if np.linalg.norm(v)<1e-9:continue
        a=math.degrees(math.atan2(v[1],v[0]))
        ans.append(abs((a-axis+45)%90-45))
    return ans

def world_line(local,axis):
    return rotate(local,axis,origin=(0,0))

def candidates(g,furniture,cap,axis,mode="rooms",doglegs=True):
    """Finite axis-only family. No other angle can enter the candidate set."""
    local=rotate(g,-axis,origin=(0,0));x0,y0,x1,y1=local.bounds
    f_local=[rotate(x["geometry"],-axis,origin=(0,0)) for x in furniture
             if not x["geometry"].intersection(outline(g)).is_empty]
    fx=[v for a in f_local for v in [a.bounds[0]-.15,a.bounds[2]+.15]]
    fy=[v for a in f_local for v in [a.bounds[1]-.15,a.bounds[3]+.15]]
    xs=sorted({float(v) for v in np.arange(x0+.3,x1-.3,.25)}|set(fx)|{(x0+x1)/2})
    ys=sorted({float(v) for v in np.arange(y0+.3,y1-.3,.25)}|set(fy)|{(y0+y1)/2})
    xs=[v for v in xs if x0+.05<v<x1-.05]
    ys=[v for v in ys if y0+.05<v<y1-.05]
    target=g.area/max(2,math.ceil((g.area-1e-8)/cap));choices=[];seen=set()
    parent_count=connection_count(g)
    def add(ll):
        line=world_line(ll,axis)
        key=tuple(np.round(np.asarray(line.coords).ravel(),7))
        if key in seen:return
        seen.add(key)
        children=cs.split_geometry(g,line)
        if not children:return
        large=[p for p in children if p.area>=MIN-1e-8]
        if mode=="rooms":
            # Do not intentionally create skinny or undersize wedges.
            if len(large)<2:return
            if any(short_side(p)<DIAMETER-1e-7 or not room_circle(p)["fits"] for p in large):return
            if any(connection_count(p)!=1 for p in large):return
        elif mode=="connectivity":
            if max(connection_count(p) for p in children)>=parent_count:return
        elif mode=="corridor":
            if not large:return
            broad=[p for p in large if room_circle(p)["fits"]]
            if not broad or len(broad)==len(children):return
            # Every non-room child must lack a broad >=6 m2 region.
            if any(broad_components(p) for p in children if p not in broad):return
        m=cs.cut_measure(line,g,furniture)
        room_budget=sum(max(1,math.ceil((p.area-1e-8)/cap)) for p in large)
        balance=sum(abs(p.area-target) for p in large)
        if mode=="connectivity":
            score=(max(connection_count(p) for p in children),
                   sum(connection_count(p) for p in children),
                   m["furniture_intersection_length_m"]>1e-7,
                   m["furniture_intersection_area_m2"],-min(p.area for p in children))
        elif mode=="corridor":
            score=(-sum(p.area for p in large if room_circle(p)["fits"]),
                   m["furniture_intersection_length_m"]>1e-7,
                   m["furniture_intersection_area_m2"],len(children))
        else:
            score=(m["furniture_intersection_length_m"]>1e-7,
                   m["furniture_intersection_length_m"] if m["furniture_intersection_length_m"]>1e-7 else 0.,
                   room_budget,balance,
                   len(line.coords)-1,m["furniture_intersection_area_m2"],
                   line.intersection(outline(g)).length)
        choices.append((score,line,children,m))
    extent=max(x1-x0,y1-y0)+3
    for x in xs:add(LineString([(x,y0-extent),(x,y1+extent)]))
    for y in ys:add(LineString([(x0-extent,y),(x1+extent,y)]))
    # Only seek doglegs if straight room cuts cross furniture or fail.
    if doglegs and (not choices or not any(c[3]["furniture_intersection_length_m"]<1e-7 for c in choices)):
        def budget(values):
            if len(values)<=8:return values
            return [values[int(i)] for i in np.linspace(0,len(values)-1,8,dtype=int)]
        dx=budget(sorted(set(fx)|{(x0+x1)/2}|set(xs[::max(1,len(xs)//5)])))
        dy=budget(sorted(set(fy)|{(y0+y1)/2}|set(ys[::max(1,len(ys)//5)])))
        dx=[v for v in dx if x0+.05<v<x1-.05]
        dy=[v for v in dy if y0+.05<v<y1-.05]
        for a in dx:
            for b in dx:
                if abs(a-b)<.1:continue
                for y in dy:add(LineString([(a,y0-extent),(a,y),(b,y),(b,y1+extent)]))
        for a in dy:
            for b in dy:
                if abs(a-b)<.1:continue
                for x in dx:add(LineString([(x0-extent,a),(x,a),(x,b),(x1+extent,b)]))
    return sorted(choices,key=lambda c:c[0])

def connected_parts(g,furniture,axis,max_depth=16):
    """Islands are new candidates; true narrow bridges use only the wall axes."""
    pending=[(p,0) for p in sorted(polygons(g),key=lambda x:-x.area)]
    result=[];cuts=[]
    while pending:
        p,depth=pending.pop(0)
        count=connection_count(p)
        if count<=1 or depth>=max_depth:
            result.append(p);continue
        choices=candidates(p,furniture,CAP,axis,mode="connectivity",doglegs=False)
        if not choices:
            result.append(p);continue
        _,line,children,m=choices[0]
        cuts.append((line,m,"narrow","true_exterior_passage_below_0_6_m",len(choices)))
        pending[:0]=[(x,depth+1) for x in children]
    return result,cuts

@lru_cache(maxsize=256)
def broad_components(g):
    """A corridor cannot discard an >=6m2, 2.4m-disk-compatible broad lobe."""
    ext=outline(g)
    widened=ext.buffer(-DIAMETER/2,join_style=1).buffer(DIAMETER/2,join_style=1)
    wide=g.intersection(widened)
    return tuple(p for p in polygons(wide) if p.area>=MIN-1e-8 and room_circle(p)["fits"])

def split_corridor(g,furniture,axis):
    broad=broad_components(g)
    if not broad:return [(g,False)],[]
    choices=candidates(g,furniture,CAP,axis,mode="corridor",doglegs=True)
    if not choices:return [(g,True)],[]
    _,line,children,m=choices[0]
    return [(p,False) for p in children],[(line,m,"narrow","separate_narrow_corridor_from_broad_room",len(choices))]

def structural_parts(scope,doors,fy,parameter,furniture,axis):
    # Both semantic doors and structural necks are checked, using the same axes.
    parts=[scope];cuts=[];audit=[]
    for door in sorted(doors,key=lambda d:d["instance_id"]):
        rec={k:v for k,v in door.items() if k!="triangles"}
        lo,hi=door["height_range_m"]
        if lo>fy+1.8 or hi<fy+.3:rec["status"]="other_floor";audit.append(rec);continue
        applied=False
        for i,g in enumerate(parts):
            c=chord(outline(g),door["centre_xz_m"],[axis,axis+90])
            if c is None or c["width_m"]>max(door["extent_xz_m"])*1.7+.4:continue
            children=split_at_chord(g,c["line"],MIN)
            if not children or any(not room_circle(p)["fits"] or short_side(p)<DIAMETER-1e-7 for p in children):continue
            m=cs.cut_measure(c["line"],g,furniture)
            cuts.append((c["line"],m,"door","semantic_door_main_wall_axis",1,
                          {"semantic_instance_id":door["instance_id"],"semantic_category":door["category"],
                           "semantic_source":door["semantic_source"],"width_m":c["width_m"]}))
            parts[i:i+1]=children;applied=True;break
        rec["status"]="cut_applied" if applied else "unreliable_or_no_room_shape_cut; narrow_only_fallback"
        audit.append(rec)
    # Axis-only shortest narrow chords, sampled from the original Voronoi medial
    # points. The 3-degree chord family in old v3 is deliberately not imported.
    from scipy.spatial import Voronoi,QhullError
    pending=list(parts);parts=[]
    while pending:
        g=pending.pop(0);ext=outline(g);samples=[]
        if g.area<2*MIN:parts.append(g);continue
        for poly in polygons(ext):
            ring=LineString(poly.exterior.coords).simplify(.015,preserve_topology=True)
            n=max(4,math.ceil(ring.length/.10))
            samples.extend([(q.x,q.y) for q in (ring.interpolate(i/n,normalized=True) for i in range(n))])
        samples=np.unique(np.round(samples,7),axis=0)
        if len(samples)>12000:samples=samples[np.linspace(0,len(samples)-1,12000,dtype=int)]
        try:
            vor=Voronoi(samples);centres=list(vor.vertices)
            centres.extend((vor.vertices[a]+vor.vertices[b])/2 for a,b in vor.ridge_vertices if a>=0 and b>=0)
        except (QhullError,ValueError):parts.append(g);continue
        centres=np.asarray(centres)
        if not len(centres):parts.append(g);continue
        inside=shapely.contains_xy(ext,centres[:,0],centres[:,1]);centres=centres[inside]
        widths=2*shapely.distance(shapely.points(centres),ext.boundary)
        centres=centres[(widths>=parameter["width_min"]-.1)&(widths<=parameter["width_max"]+.1)]
        seen=set();feasible=[]
        for centre in centres:
            key=tuple(np.round(centre/.15).astype(int))
            if key in seen:continue
            seen.add(key)
            c=chord(ext,centre,[axis,axis+90])
            if c is None or not parameter["width_min"]<=c["width_m"]<=parameter["width_max"]:continue
            children=split_at_chord(g,c["line"],MIN)
            if not children or any(short_side(p)<DIAMETER-1e-7 or not room_circle(p)["fits"] for p in children):continue
            m=cs.cut_measure(c["line"],g,furniture)
            feasible.append(((m["furniture_intersection_length_m"]>1e-7,m["furniture_intersection_length_m"],
                              c["width_m"],-min(p.area for p in children)),c,children,m))
        if not feasible:parts.append(g);continue
        _,c,children,m=min(feasible,key=lambda x:x[0])
        cuts.append((c["line"],m,"narrow","filled_exterior_medial_neck_main_wall_axis",len(feasible),
                     {"width_m":c["width_m"],"semantic_source":None}))
        pending.extend(children)
    return parts,cuts,audit

class Cutter:
    """Capped axis partition; no extra 80% trigger on a sub-cap block."""
    def __init__(self,mesh,points,nav,p,parameter,cap,furniture,axis):
        self.mesh=mesh;self.points=points;self.nav=nav;self.p=p;self.parameter=parameter
        self.cap=cap;self.furniture=furniture;self.axis=axis;self.audit=[];self.attempts=0;self.cache={}
    def visibility(self,g):
        if g.wkb not in self.cache:
            ids=np.flatnonzero(shapely.contains_xy(g,self.points[:,0],self.points[:,2]))
            pts=self.points[ids]
            _,_,weights=grid_atoms(g,self.nav.intersection(g),pts,self.p["grid_step_m"])
            self.cache[g.wkb]=assess_visibility(self.mesh,pts,weights,self.p,self.parameter["coverage"],self.parameter["max_distance"])
        return self.cache[g.wkb]
    def solve(self,g,depth=0):
        if g.area<MIN-1e-8:return [(g,None,None)],[]
        if g.area<=self.cap+1e-8:
            # Owner explicitly disabled visibility-only extra splitting. Measured
            # coverage stays in JSON as a diagnostic, without a new admission gate.
            return [(g,self.visibility(g),None)],[]
        if depth>=10 or self.attempts>=36:return [(g,None,"AXIS_CUT_CPU_BUDGET_UNRESOLVED")],[]
        choices=candidates(g,self.furniture,self.cap,self.axis)
        if not choices:return [(g,None,"NO_AXIS_CUT_WITH_2_4_M_DISKS")],[]
        best=None
        for score,line,children,m in choices[:4]:
            if self.attempts>=36:break
            self.attempts+=1;leaves=[];cuts=[]
            for child in children:
                ll,cc=self.solve(child,depth+1);leaves.extend(ll);cuts.extend(cc)
            unresolved=sum(p.area for p,v,e in leaves if e)
            # Visibility informs the construction among geometry-feasible cuts.
            visibility_deficit=sum((1.-v.get("coverage_fraction",0.))*p.area for p,v,e in leaves if v)
            key=(unresolved,len(leaves),score[:2],visibility_deficit,score[3:])
            cuts.append((line,m,"visibility","cap_partition_main_wall_axis_no_subcap_visibility_trigger",len(choices)))
            if best is None or key<best[0]:best=(key,leaves,cuts)
            if not unresolved and len(leaves)==math.ceil((g.area-1e-8)/self.cap):break
        self.audit.append(dict(area_m2=g.area,candidate_count=len(choices),attempts_total=self.attempts,
                               selected_unresolved_area_m2=best[0][0] if best else g.area,
                               wall_axis_deg=self.axis,only_wall_axes=all(max(angle_errors(line,self.axis),default=0)<1e-6 for score,line,children,measure in choices),candidate_max_wall_angle_error_deg=max((max(angle_errors(line,self.axis),default=0) for score,line,children,measure in choices),default=0),global_minimum_unverified=True))
        if best:return best[1],best[2]
        return [(g,None,"AXIS_CUT_CPU_BUDGET_UNRESOLVED")],[]

def rgb_for(row,floor,cache,root,old):
    # Preserve existing CPU mosaics, including all-scene contents outside scope.
    fid=floor["floor_id"];ref=old.get("floor_overheads",{}).get(fid)
    if ref:
        from PIL import Image
        meta=Path(ref["metadata_path"]);im=Path(ref["image_path"])
        if meta.exists() and im.exists():
            d=json.loads(meta.read_text());sensor=next((s for s in d["images"] if s["path"].endswith("_overview.png")),d["images"][0])
            if abs(d["floor_y_m"]-floor["floor_y_m"])<=.3:
                sensor=dict(sensor)
                sensor.setdefault("span_m",d.get("span_m",2/np.linalg.norm(np.asarray(sensor["projection"])[0,:3])))
                return d,sensor,Image.open(im).convert("RGB"),ref
    # Newly admitted 35-50m2 native sources already have full-house CPU renders.
    candidates_raw=[]
    for path in (root/"size_gallery_30_50_v1/raw").glob(row["house"]+"__*.json"):
        d=json.loads(path.read_text())
        if abs(d["floor_y_m"]-floor["floor_y_m"])<=.3:candidates_raw.append((abs(d["floor_y_m"]-floor["floor_y_m"]),path,d))
    if candidates_raw:
        from PIL import Image
        _,path,d=min(candidates_raw,key=lambda x:x[0])
        entry=dict(d,images=[dict(path=d["path"],projection=d["projection"],span_m=d["span_m"])])
        ref=dict(metadata_path=str(path),image_path=d["path"],source="existing CPU whole-house gallery frame",metadata_format="whole_house_gallery_v1")
        return entry,entry["images"][0],Image.open(d["path"]).convert("RGB"),ref
    return floor_overhead(row,floor,cache,root/"overhead_cpu_atlas_v4",overhead_for(row,cache))

def process(old,mesh,pf,hs,nav_polys,nav_ys,objects,markers,door_info,p,parameter,cap,cache,root):
    row=old["source_geometry"];result=copy.deepcopy(old)
    result.update(blocks=[],cut_lines=[],status="processed",revision="cap35_axis_v4",requires_split=True,
                  merge_audit=[],connectivity_separation_audit=[],straight_partition_audit=[],
                  stair_separation_audit=[],stair_exact_exclusion=[],corridor_salvage_audit=[],
                  parameter=dict(parameter,max_room_area_m2=cap),floor_overheads={},wall_axes={})
    index=0
    for floor in row["floors"]:
        scope=shape(floor["floor_polygon"]);fy=floor["floor_y_m"];fid=floor["floor_id"];axis=main_axis(scope)
        result["wall_axes"][fid]=dict(primary_deg=axis,perpendicular_deg=axis+90,source="dominant filled ground minimum rotated rectangle; frozen within floor")
        rgb=rgb_for(row,floor,cache,root,old);result["floor_overheads"][fid]=rgb[3] if rgb else None
        nav=nav_scope_at(nav_polys,nav_ys,fy,scope,p)
        _,points,clearance,adj,_=sample_navigation(pf,hs,scope,fy,p)
        furniture=cs.furniture_for(objects,row["region_id"],fy,p,scope) if scope.area>=MIN else cs.FurnitureList([])
        error="SEMANTIC_PALETTE_AMBIGUOUS" if row.get("ambiguous_semantic_palette") else "ZERO_AREA_SEMANTIC_GROUND_FACES" if door_info["invalid_ground_zero_area_faces_by_region"].get(row["region_id"],0) else None
        normal,stairs,sa=cs.partition_stair_aliases(scope,fy,markers,p["floor_height_separation_m"])
        result["stair_separation_audit"].extend(dict(s,floor_id=fid,periphery_expansion_m=0.) for s in sa)
        result["stair_exact_exclusion"].append(dict(floor_id=fid,geometry_xz_m=mapping(shapely.union_all(stairs)),area_m2=sum(x.area for x in stairs),periphery_expansion_m=0.))
        final=[dict(g=g,forced="STAIRS",visibility=None,error=None) for g in stairs]
        def add_cut(item):
            line,m,kind,stage,n,*rest=item;cid=fid+f'_L{len(result["cut_lines"]):03d}'
            rec=dict(id=cid,floor_id=fid,type=kind,stage=stage,line_xz_m=np.asarray(line.coords).tolist(),
                     line_geometry_xz_m=mapping(line),segment_count=len(line.coords)-1,wall_axis_deg=axis,
                     segment_angle_errors_deg=angle_errors(line,axis),tested_candidate_count=n,
                     fallback_crosses_furniture=m["furniture_intersection_length_m"]>1e-7,**m)
            if rest:rec.update(rest[0])
            result["cut_lines"].append(rec)
        if error:final.append(dict(g=normal,forced=None,visibility=None,error=error))
        else:
            components,cc=connected_parts(normal,furniture,axis)
            for item in cc:add_cut(item)
            for part in components:
                if part.area<MIN:final.append(dict(g=part,forced="DETACHED_FRAGMENT",visibility=None,error=None));continue
                pieces,sc,da=structural_parts(part,[s for s in markers if "door" in s["category"]],fy,parameter,furniture,axis)
                result.setdefault("door_instance_audit_v4",[]).extend(da)
                for item in sc:add_cut(item)
                for piece in pieces:
                    pending,extra=connected_parts(piece,furniture,axis)
                    for item in extra:add_cut(item)
                    for child in pending:
                        if child.area<MIN:final.append(dict(g=child,forced="DETACHED_FRAGMENT",visibility=None,error=None));continue
                        cutter=Cutter(mesh,points,nav,p,parameter,cap,furniture,axis)
                        if child.area>cap+1e-8:
                            leaves,newcuts=cutter.solve(child)
                            for item in newcuts:add_cut(item)
                            result["straight_partition_audit"].extend(dict(x,floor_id=fid) for x in cutter.audit)
                        else:leaves=[(child,None,None)]
                        for g,vis,err in leaves:final.append(dict(g=g,forced=None,visibility=vis,error=err))
        # Never assign a flying component back to a room by nearest-seed ownership.
        expanded=[]
        for item in final:
            if item["forced"]=="STAIRS" or item["error"]:expanded.append(item);continue
            comps,extra=connected_parts(item["g"],furniture,axis)
            for cut in extra:add_cut(cut)
            for g in comps:expanded.append(dict(item,g=g,forced="DETACHED_FRAGMENT" if g.area<MIN else None,
                                                visibility=None if len(comps)>1 else item["visibility"]))
        final=expanded
        # Separate broad rooms from a corridor instead of discarding a broad comb.
        revised=[]
        for item in final:
            g=item["g"]
            if item["forced"] or item["error"] or g.area<MIN:revised.append(item);continue
            width=body_width(outline(g));short=short_side(g);rect=g.minimum_rotated_rectangle
            lengths=np.linalg.norm(np.diff(np.asarray(rect.exterior.coords),axis=0),axis=1)
            aspect=float(max(lengths)/short) if short else math.inf
            if width>=1.5 or aspect<3:revised.append(item);continue
            pieces,cc=split_corridor(g,furniture,axis)
            result["corridor_salvage_audit"].append(dict(floor_id=fid,source_area_m2=g.area,
                broad_parts=[p.area for p in broad_components(g)],parts=[dict(area_m2=x.area,unresolved=e) for x,e in pieces]))
            for cut in cc:add_cut(cut)
            for child,unresolved in pieces:
                revised.append(dict(item,g=child,visibility=None,
                                    error="CORRIDOR_WIDE_PART_AXIS_SEPARATION_UNRESOLVED" if unresolved else None))
        final=revised
        # Open-zone merges can use visibility, but never trigger an extra subcap split.
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
                    if any(c["type"]=="door" and c["floor_id"]==fid and interface.intersection(LineString(c["line_xz_m"]).buffer(.001)).length>.01 for c in result["cut_lines"]):continue
                    merged=a.union(b)
                    if connection_count(merged)!=1 or not room_circle(merged)["fits"]:continue
                    vis=Cutter(mesh,points,nav,p,parameter,cap,furniture,axis).visibility(merged)
                    result["merge_audit"].append(dict(floor_id=fid,area_m2=merged.area,opening_width_m=interface.length,
                                                     meets_visibility=vis["meets_visibility"],merged=vis["meets_visibility"]))
                    if vis["meets_visibility"]:
                        final[i]=dict(g=merged,forced=None,visibility=vis,error=None);final.pop(j);changed=True;break
        for item in final:
            g=item["g"];vis=item["visibility"]
            if g.is_empty or g.area<=1e-10:continue
            ids=np.flatnonzero(shapely.contains_xy(g,points[:,0],points[:,2]));navg=nav.intersection(g)
            witness=placement(mesh,points,clearance,adj,p,ids.tolist()) if not item["forced"] else dict(found=False,not_run_reason=item["forced"],acoustics="not_run")
            black=measured_black(g,fy,rgb,p)
            kind,evidence=block_type(g,fy,[x for x in objects.instances if "stair" not in x["category"] and x["category"] not in ("step","steps")],[x for x in markers if "stair" not in x["category"] and x["category"] not in ("step","steps")],row)
            if kind=="stairs":kind="unknown";evidence=["exact stair mask excluded; adjacent flat ground follows normal checks"]
            if item["forced"]=="STAIRS":kind="stairs";evidence=["direct semantic stair ground only"]
            reasons=[];unknown=[]
            if item["forced"]:reasons.append(item["forced"])
            if g.area<MIN-1e-8 and not item["forced"]:reasons.append("FLOOR_AREA_BELOW_6")
            if g.area>cap+1e-8:unknown.append("STILL_ABOVE_AREA_CAP_UNRESOLVED")
            short=short_side(g);width=body_width(outline(g)) if g.area>=MIN else 0.
            rect=g.minimum_rotated_rectangle
            long=max(np.linalg.norm(np.diff(np.asarray(rect.exterior.coords),axis=0),axis=1)) if rect.geom_type=="Polygon" else 0.
            aspect=float(long/short) if short else None;conn=connection_count(g);circle=room_circle(g) if g.area>=MIN else dict(fits=False,diameter_m=DIAMETER)
            if short<DIAMETER-1e-8:reasons.append("SHORT_SIDE_BELOW_2_4")
            if width<1.5 and aspect and aspect>=3 and not item["forced"]:
                if broad_components(g):unknown.append("CORRIDOR_WIDE_PART_AXIS_SEPARATION_UNRESOLVED")
                else:reasons.append("CORRIDOR_BODY_WIDTH_BELOW_1_5")
            if kind=="outdoor":reasons.append("OUTDOOR_OR_BALCONY_SEMANTIC_AND_LOCAL_GEOMETRY")
            if conn!=1 and not item["forced"]:unknown.append("CONNECTIVITY_0_6_UNRESOLVED")
            # Disk rule selects new cuts; it is NOT applied to existing <=35 rooms.
            if not circle["fits"] and not item["forced"]:unknown.append("NO_2_4_M_ROOM_DISK_AXIS_CONSTRUCTION_UNRESOLVED")
            if black["black_fraction"] is None:unknown.append("SCAN_BLACK_FRACTION_UNVERIFIED")
            elif black["black_fraction"]>.15:reasons.append("SCAN_BLACK_FRACTION_ABOVE_15_PERCENT")
            if not witness["found"] and not item["forced"]:reasons.append("PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET")
            if item["error"]:unknown.append(item["error"])
            # An unsalvaged broad corridor must remain unresolved, never 'discard'.
            corridor_wide="CORRIDOR_WIDE_PART_AXIS_SEPARATION_UNRESOLVED" in unknown
            decision="unresolved" if corridor_wide else "discard" if reasons else "unresolved" if unknown else "retain"
            bid=row["house"]+"__"+row["room_label"]+"__"+fid+f'__S{index:03d}';index+=1
            result["blocks"].append(dict(schema="hm3d_auto_room_split_v4",id=bid,house=row["house"],
                source_region_id=row["region_id"],source_region=row["room_label"],source_region_origins=row["origins"],
                floor_id=fid,floor_y_m=fy,floor_height_range_m=floor["height_range_m"],floor_polygon_xz_m=mapping(g),
                floor_area_m2=float(g.area),short_side_m=short,body_width_m_proxy=width,aspect_ratio=aspect,
                nav_walkable_area_m2=float(navg.area),nav_grid_point_count=len(ids),black_fraction=black["black_fraction"],
                black_measurement=black,visibility_coverage_fraction=vis["coverage_fraction"] if vis else None,
                visibility=vis or dict(status="not_required",reason="no added visibility-only subcap split or admission gate"),
                visibility_admission_gate=False,placement_witness=witness,decision=decision,
                discard_reasons=reasons if decision=="discard" else [],unresolved_reasons=list(dict.fromkeys(unknown)) if decision=="unresolved" else [],
                unverified_diagnostics=list(dict.fromkeys(unknown)),room_type=kind,type_evidence=evidence,cut_ids=[],adjacent_rooms=[],
                new_room=True,acoustics="not_run_per_task",measurement_source=row["semantic_source"],area_method=row["ground_measurement"],
                connectivity_core_count=conn,connectivity_width_m=WIDTH,connectivity_join_style="mitre",
                inscribed_circle=circle,max_room_area_m2=cap,source_selection=row.get("source_selection")))
    result["retained_new_rooms"]=sum(b["decision"]=="retain" for b in result["blocks"])
    result["status"]="partially_unresolved" if any(b["decision"]=="unresolved" for b in result["blocks"]) else "processed"
    result["area_partition_error_m2"]=abs(sum(b["floor_area_m2"] for b in result["blocks"])-row["floor_area_sum_m2"])
    finalize_interfaces(result["blocks"],result["cut_lines"]);adjacency(result["blocks"],result["cut_lines"])
    return result

def worker(job,root,out,p,parameter,cap,cache):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    started=time.time();root=Path(root);out=Path(out)
    hs=load_native();pf,np_,ny=navmesh_triangles(hs,Path(job["navmesh"]))
    mesh,receipt=raw_collision(job["scene_directory"]);markers,door_info=structural_instances(job["scene_directory"]);objects=load_scene(job["scene_directory"])
    marker_ids={m["instance_id"] for m in markers}
    markers.extend(dict(x,semantic_source=str(Path(job["scene_directory"])/(Path(job["scene_directory"]).name.split("-",1)[1]+".semantic.glb"))) for x in objects.instances if x["category"] in ("step","steps") and x["instance_id"] not in marker_ids)
    statuses=[]
    for source in job["sources"]:
        old=json.loads(Path(source).read_text());t=time.time()
        try:result=process(old,mesh,pf,hs,np_,ny,objects,markers,door_info,p,parameter,cap,cache,root)
        except Exception as e:
            result=copy.deepcopy(old);result.update(status="unresolved",revision="cap35_axis_v4",requires_split=True,cut_lines=[],
                retained_new_rooms=0,unresolved_reasons=[repr(e)],traceback=traceback.format_exc())
            for b in result["blocks"]:b.update(decision="unresolved",discard_reasons=[],unresolved_reasons=["V4_PROCESS_FAILURE"],new_room=True)
        dump(out/"regions"/Path(source).name,result)
        statuses.append(dict(source=old["source_region"],status=result["status"],seconds=time.time()-t,
                             retained=result["retained_new_rooms"],error=result.get("unresolved_reasons")))
        print("V4_REGION",job["house"],old["source_region"],result["status"],"seconds",round(time.time()-t,1),"kept",result["retained_new_rooms"],flush=True)
    return dict(house=job["house"],statuses=statuses,seconds=time.time()-started,pid=os.getpid(),
                nice=os.getpriority(os.PRIO_PROCESS,0),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                ray_receipt=receipt,cpu_only=True)

def prepare(root,cap=CAP):
    root=Path(root);out=root/"delivery_all_v4";out.mkdir(exist_ok=False)
    (out/"regions").mkdir();(out/"rooms").mkdir();inputs=out/"source_inputs";inputs.mkdir()
    plan=json.loads((root/"processing_plan_v1.json").read_text())
    old={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v3/final_v1/regions").glob("*.json")}
    selection=json.loads((root/"size_gallery_30_50_v1/selection_rows_v1.json").read_text())
    native=[r for r in selection if float(r["floor_area_m2"])>cap+1e-8]
    by_key={(d["house"],d["source_region"]):(name,d) for name,d in old.items()}
    inventory={p.stem:json.loads(p.read_text()) for p in (root/"inventory_v1").glob("*.json")}
    targets={name:copy.deepcopy(d) for name,d in old.items() if d["requires_split"]}
    manifest=[]
    for name,d in targets.items():manifest.append(dict(house=d["house"],region=d["source_region"],source=name,origin="original_gt50",area_m2=d["source_floor_area_m2"]))
    for r in native:
        key=(r["house"],r["room_label"]);name=r["house"]+"__"+r["room_label"]+".json"
        inv=inventory[r["house"]];row=copy.deepcopy(next(x for x in inv["rows"] if x["room_label"]==r["room_label"]))
        row.update(origins=["existing_training_native_gt35"],reference_fold=None,requires_split=True,
                   source_selection=r["selection_file"],native_selection_row=r)
        if key in by_key:
            name,previous=by_key[key];d=copy.deepcopy(previous)
            row["origins"]=list(dict.fromkeys(previous["origins"]+row["origins"]))
        else:
            d=dict(house=r["house"],source_region=r["room_label"],source_region_id=row["region_id"],
                   origins=row["origins"],reference_fold=None,source_floor_area_m2=row["floor_area_sum_m2"],blocks=[],
                   cut_lines=[],floor_overheads={},status="pending",requires_split=True)
        d.update(source_geometry=row,source_floor_area_m2=row["floor_area_sum_m2"],requires_split=True,origins=row["origins"],native_selection=r)
        if not d["blocks"]:
            for f in row["floors"]:
                d["blocks"].append(dict(id=name[:-5]+"__"+f["floor_id"]+"__NATIVE",house=r["house"],source_region=r["room_label"],
                    floor_id=f["floor_id"],floor_y_m=f["floor_y_m"],floor_polygon_xz_m=f["floor_polygon"],floor_area_m2=f["floor_area_m2"],
                    short_side_m=f["short_side_m"],decision="unchanged",new_room=False,room_type=r["room_type"],
                    placement_witness={},discard_reasons=[],unresolved_reasons=[]))
        targets[name]=d
        manifest.append(dict(house=r["house"],region=r["room_label"],source=name,origin="existing_training_native_gt35",
                             listed_area_m2=float(r["floor_area_m2"]),area_m2=row["floor_area_sum_m2"],selection=r["selection_file"]))
    for name,d in old.items():
        if name not in targets:dump(out/"regions"/name,d)
    jobs={}
    for name,d in targets.items():
        row=d["source_geometry"];dump(inputs/name,d);house=d["house"]
        if house not in jobs:jobs[house]=dict(house=house,scene_directory=row["scene_directory"],navmesh=row["navmesh_source"],sources=[])
        jobs[house]["sources"].append(str(inputs/name))
    ordered=sorted(jobs.values(),key=lambda j:-sum(targets[Path(p).name]["source_floor_area_m2"] for p in j["sources"]))
    revision=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source="delivery_all_v3/final_v1",max_room_area_m2=cap,min_room_area_m2=MIN,min_short_side_m=DIAMETER,
        room_disk_diameter_m=DIAMETER,original_gt50_sources=sum(d["requires_split"] for d in old.values()),
        native_gt35_sources=len(native),native_35_40=sum(float(r["floor_area_m2"])<40 for r in native),
        native_40_50=sum(40<=float(r["floor_area_m2"])<=50 for r in native),target_sources=len(targets),
        unchanged_prior_sources=len(old)-sum(name in old for name in targets),total_sources=len(old)+sum(name not in old for name in targets),
        scope_manifest=manifest,jobs=ordered,parameters=plan["parameters"],selected_parameter=json.loads((root/"selected_parameter_v1.json").read_text())["selected_parameter"],
        overhead_cache=plan["overhead_cache"],cpu_only=True,nice=10,
        visibility_policy="No visibility-only split/admission for subcap rooms; existing P13 visibility diagnostics guide cuts and wide merges; untouched <=35 native rooms get no circle rejection",
        connectivity="fill every interior hole, erode/dilate 0.3m with mitre joins; raw islands separated; area uses raw geometry",
        owner_update="20261009 round5 cap35 supersedes cap50; immutable prior deliveries")
    dump(out/"revision_plan.json",revision)
    print("V4_PREPARED",revision["target_sources"],"HOUSES",len(jobs),"NATIVE",len(native),flush=True)

def run(root,workers=8,pilot=False,pilot_dir="pilot_v1"):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3))
    root=Path(root);out=root/"delivery_all_v4";plan=json.loads((out/"revision_plan.json").read_text())
    if pilot:
        out=root/"revision_cap35_20261009_v1"/pilot_dir;out.mkdir(exist_ok=False);(out/"regions").mkdir()
        wanted={("00081","R5"),("00831","R6"),("00821","R1"),("00022","R24"),("00149","R5")}
        jobs=[]
        for j in plan["jobs"]:
            ss=[s for s in j["sources"] if any(h in j["house"] and json.loads(Path(s).read_text())["source_region"]==rid for h,rid in wanted)]
            if ss:jobs.append(dict(j,sources=ss))
    else:jobs=plan["jobs"]
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        fs={pool.submit(worker,j,root,out,plan["parameters"],plan["selected_parameter"],plan["max_room_area_m2"],plan["overhead_cache"]):j["house"] for j in jobs}
        for f in as_completed(fs):
            try:receipts.append(f.result())
            except Exception as e:receipts.append(dict(house=fs[f],error=repr(e),traceback=traceback.format_exc()))
    dump(out/"native_receipt.json",receipts)
    if not pilot:
        for j in jobs:
            for s in j["sources"]:
                name=Path(s).name
                if (out/"regions"/name).exists():continue
                d=json.loads(Path(s).read_text());d.update(status="unresolved",requires_split=True,cut_lines=[],retained_new_rooms=0,unresolved_reasons=["NATIVE_HOUSE_JOB_FAILURE"])
                for b in d["blocks"]:b.update(decision="unresolved",discard_reasons=[],unresolved_reasons=["NATIVE_HOUSE_JOB_FAILURE"],new_room=True)
                dump(out/"regions"/name,d)
        for path in (out/"regions").glob("*.json"):
            d=json.loads(path.read_text())
            for b in d["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
    dump(out/"completed.json",dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,
        compute_processes_with_tracker=workers+2,max_address_space_gib=10*(workers+1),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
        native_house_jobs=len(jobs),source_regions=len(list((out/"regions").glob("*.json"))),pilot=pilot))
    print("V4_NATIVE_DONE",out,"HOUSE_JOBS",len(jobs),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--prepare",action="store_true");a.add_argument("--pilot",action="store_true");a.add_argument("--workers",type=int,default=8);a.add_argument("--pilot-dir",default="pilot_v1")
    v=a.parse_args()
    if v.prepare:prepare(v.root)
    else:run(v.root,v.workers,v.pilot,v.pilot_dir)
