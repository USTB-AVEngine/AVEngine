"""Floor-part connectivity with exact seams and local existing-navmesh support."""
from __future__ import annotations
import hashlib,math
import numpy as np
import shapely
from shapely.geometry import Polygon,GeometryCollection,LineString,mapping
from shapely.ops import split
from tools.rooms.room_selection.media import polygons

PROXY_REPAIRS=0
def proxy_op(operation,a,b):
    """Only diagnostic contours retry at numerical precision; raw floor unchanged."""
    global PROXY_REPAIRS
    a=shapely.make_valid(a);b=shapely.make_valid(b)
    fn=getattr(shapely,operation)
    try:return fn(a,b)
    except shapely.GEOSException:
        PROXY_REPAIRS+=1
        return fn(a,b,grid_size=1e-8)
def proxy_union(a,b):return proxy_op("union",a,b)
def proxy_intersection(a,b):return proxy_op("intersection",a,b)
def proxy_difference(a,b):return proxy_op("difference",a,b)
def clip_proxy_to_mask(g,mask):
    """Avoid a costly overlay when exact containment proves clipping is a no-op."""
    if mask is None or g.is_empty or mask.covers(g):return g
    return proxy_intersection(g,mask)

def exterior(g):
    parts=[q for p in polygons(g) for q in polygons(shapely.make_valid(Polygon(p.exterior)))]
    return shapely.union_all(parts)

class Connectivity:
    """Bridge shapes are diagnostic only. Raw ground area is never enlarged."""
    def __init__(self,nav,seam=.05,expansion=.3,width=.6):
        self.nav=nav;self.seam=seam;self.expansion=expansion;self.width=width
        self.cache={};self.pair_cache={};self.expanded_cache={};self.masks={};self.audit=[]

    def expanded(self,g):
        key=hashlib.sha256(g.wkb).digest()
        if key not in self.expanded_cache:
            if len(self.expanded_cache)>4096:self.expanded_cache.clear()
            self.expanded_cache[key]=g.buffer(self.expansion,join_style=2)
        return self.expanded_cache[key]

    def _pair(self,a,b,mask=None):
        key=tuple(sorted([hashlib.sha256(a.wkb).digest(),hashlib.sha256(b.wkb).digest()]))
        if key not in self.pair_cache:
            dist=float(a.distance(b))
            if a.covers(b) or b.covers(a):
                rec=dict(kind="seam_contained_inside_filled_outline",distance_m=dist,bridge=GeometryCollection(),accepted=True)
            elif dist<=self.seam+1e-9:
                expanded_a=self.expanded(a);expanded_b=self.expanded(b)
                bridge=proxy_difference(proxy_intersection(expanded_a,expanded_b),proxy_union(a,b))
                # Stay inside the original outer hull; a seam cannot enlarge the
                # room on its external side. It only fills the missing join.
                bridge=proxy_intersection(bridge,proxy_union(a,b).convex_hull)
                rec=dict(kind="seam_distance_le_0_05",distance_m=dist,bridge=bridge,accepted=True)
            elif dist<=2*self.expansion+1e-9:
                expanded_a=self.expanded(a);expanded_b=self.expanded(b)
                support=proxy_intersection(self.nav,proxy_union(expanded_a,expanded_b))
                supports=[p for p in polygons(support) if proxy_intersection(p,a).area>1e-10 and proxy_intersection(p,b).area>1e-10]
                if supports:
                    connected=shapely.union_all(supports)
                    bridge=proxy_difference(proxy_intersection(connected,proxy_intersection(expanded_a,expanded_b)),proxy_union(a,b))
                    bridge=proxy_intersection(bridge,proxy_union(a,b).convex_hull)
                    rec=dict(kind="native_navmesh_local_direct_passage",distance_m=dist,bridge=bridge,accepted=True,
                             nav_support=connected)
                else:rec=dict(kind="no_connected_local_navmesh_support",distance_m=dist,bridge=GeometryCollection(),accepted=False)
            else:rec=dict(kind="outside_local_0_3_expansion",distance_m=dist,bridge=GeometryCollection(),accepted=False)
            self.pair_cache[key]=rec
        base=self.pair_cache[key];rec=dict(base)
        if mask is not None:
            rec["bridge"]=clip_proxy_to_mask(base["bridge"],mask)
            if rec["accepted"] and "nav_support" in base:
                support=clip_proxy_to_mask(base["nav_support"],mask)
                supports=[p for p in polygons(support) if proxy_intersection(p,a).area>1e-10 and proxy_intersection(p,b).area>1e-10]
                rec["accepted"]=bool(supports)
                if supports:rec["nav_support"]=shapely.union_all(supports)
            elif rec["accepted"] and base["bridge"].area>1e-10 and rec["bridge"].is_empty:
                # A designed cut may not be crossed by a phantom seam bridge.
                rec["accepted"]=a.distance(b)<=1e-9 and not a.intersection(b).is_empty
        return rec

    def record(self,g):
        mask=self.masks.get(g.wkb)
        key=(g.wkb,mask.wkb if mask is not None else None)
        if key in self.cache:return self.cache[key]
        raw=sorted(polygons(g),key=lambda p:(-p.area,p.bounds))
        ext=[exterior(p) for p in raw];parent=list(range(len(raw)));edges=[];bridges=[]
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        if ext:
            tree=shapely.STRtree(ext)
            for i,a in enumerate(ext):
                for j in sorted(map(int,tree.query(a.buffer(2*self.expansion+1e-8)))):
                    if j<=i:continue
                    rec=self._pair(a,ext[j],mask)
                    if not rec["accepted"]:continue
                    ai,bi=find(i),find(j)
                    if ai!=bi:parent[bi]=ai
                    bridges.append(rec["bridge"])
                    edges.append(dict(part_a=i,part_b=j,distance_m=rec["distance_m"],kind=rec["kind"],
                        expansion_m=self.expansion,bridge_geometry_xz_m=mapping(rec["bridge"]),
                        nav_support_geometry_xz_m=mapping(rec["nav_support"]) if "nav_support" in rec else None))
        groups={}
        for i,p in enumerate(raw):groups.setdefault(find(i),[]).append(p)
        grouped=[shapely.union_all(parts) for parts in groups.values()]
        try:envelope=shapely.union_all([*ext,*bridges])
        except shapely.GEOSException:
            global PROXY_REPAIRS
            PROXY_REPAIRS+=1
            envelope=shapely.union_all([*ext,*bridges],grid_size=1e-8)
        if mask is not None:envelope=clip_proxy_to_mask(envelope,mask)
        # Fill only shape holes, including holes introduced by joining scan parts.
        envelope=exterior(envelope)
        opening=envelope.buffer(-(self.width/2-1e-8),join_style=2).buffer(self.width/2-1e-8,join_style=2)
        count=len([p for p in polygons(opening) if p.area>1e-8])
        result=dict(groups=sorted(grouped,key=lambda p:-p.area),envelope=envelope,links=edges,
                    raw_part_count=len(raw),direct_group_count=len(grouped),opened_count=count,opening=opening)
        if len(self.cache)>2048:self.cache.clear()
        self.cache[key]=result
        return result

    def groups(self,g):
        return self.record(g)["groups"]

    def envelope(self,g):
        return self.record(g)["envelope"]

    def count(self,g):
        r=self.record(g)
        if r["direct_group_count"]!=1:return r["direct_group_count"]
        return r["opened_count"]

    def split(self,g,line):
        """Split merged contours, then clip RAW measured floor to the cells."""
        proxy=self.envelope(g)
        try:cells=list(polygons(split(proxy,line)))
        except shapely.GEOSException:return []
        if len(cells)<2:return []
        rawcells=[]
        for cell in cells:
            raw=shapely.union_all(list(polygons(g.intersection(cell))))
            if raw.is_empty or raw.area<=1e-10:continue
            rawcells.append((raw,cell))
        if len(rawcells)<2:return []
        if abs(sum(raw.area for raw,cell in rawcells)-g.area)>1e-6:return []
        children=[]
        for raw,cell in rawcells:
            self.masks[raw.wkb]=cell
            for child in self.groups(raw):
                self.masks[child.wkb]=cell
                children.append(child)
        if len(children)<2:return []
        if shapely.union_all(children).symmetric_difference(g).area>1e-6:return []
        return sorted(children,key=lambda p:(-p.area,p.bounds))

    def certificate(self,g):
        rec=self.record(g)
        return dict(rule="seam_nav_v5",seam_gap_m=self.seam,part_expansion_m=self.expansion,
                    min_passage_m=self.width,numeric_opening_tolerance_m=1e-8,
                    raw_part_count=rec["raw_part_count"],direct_connected_groups=rec["direct_group_count"],
                    opened_components=rec["opened_count"],connection_count=self.count(g),
                    shape_envelope_xz_m=mapping(rec["envelope"]),links=rec["links"],
                    partition_cell_xz_m=mapping(self.masks[g.wkb]) if g.wkb in self.masks else None,
                    geometry_proxy_numeric_repairs=PROXY_REPAIRS,geometry_proxy_retry_precision_m=1e-8,
                    area_basis="raw holed semantic ground unchanged; bridge masks are connectivity/shape diagnostics only")
