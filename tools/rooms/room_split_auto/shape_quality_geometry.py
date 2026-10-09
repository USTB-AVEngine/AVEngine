"""Room-only shape repairs. Navigation joins never enlarge the shape test."""
from __future__ import annotations
import math,time
from functools import lru_cache
import numpy as np
import shapely
from shapely.geometry import Polygon,LineString,Point,GeometryCollection,box
from shapely.affinity import rotate
from shapely.ops import split,nearest_points
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_split_auto.seam_connectivity import Connectivity,exterior,proxy_intersection,proxy_difference,proxy_union
from tools.rooms.room_split_auto.connected_split import cut_measure

@lru_cache(maxsize=1024)
def own_outline(g):
    # Exactly the reviewer's seam closing and hole filling; no navigation input.
    p=exterior(g.buffer(0))
    p=p.buffer(.03,join_style=2).buffer(-.03,join_style=2)
    return exterior(p)

@lru_cache(maxsize=512)
def planning_outline(g):
    # Quantization belongs only to candidate ranking, never stored floor or area.
    p=own_outline(g)
    try:p=shapely.set_precision(p,.005)
    except shapely.GEOSException:pass
    return exterior(p.simplify(.025,preserve_topology=True).buffer(0))

def long_side(g):
    r=g.minimum_rotated_rectangle
    return max((math.dist(a,b) for a,b in zip(r.exterior.coords,list(r.exterior.coords)[1:])),default=0.) if r.geom_type=="Polygon" else 0.

@lru_cache(maxsize=2048)
def defects(g):
    p=own_outline(g)
    o=sorted((q for q in polygons(p.buffer(-.3,join_style=2).buffer(.3,join_style=2)) if q.area>=.05),key=lambda q:-q.area)
    neck=[q for q in o[1:] if q.area>=.5]
    corridor=[q for q in polygons(p.difference(p.buffer(-.75,join_style=2).buffer(.75,join_style=2))) if q.area>=3 and long_side(q)>=3.5]
    return dict(neck=neck,corridor=corridor,neck_count=len(neck),corridor_count=len(corridor))

@lru_cache(maxsize=2048)
def disk(g,tolerance=.001):
    p=own_outline(g)
    erosion=p.buffer(-1.2,join_style=1)
    pts=[q.representative_point() for q in polygons(erosion)]
    if not p.is_empty:pts.append(p.centroid)
    if not p.minimum_rotated_rectangle.is_empty:pts.append(p.minimum_rotated_rectangle.centroid)
    if not pts:
        # A bounded exact-boundary fallback, including exactly 2.4m rectangles.
        r=p.minimum_rotated_rectangle
        if not r.is_empty:pts.append(r.centroid)
        pts.extend(q.representative_point() for q in polygons(p))
    good=[(float(p.boundary.distance(x)),x) for x in pts if p.covers(x)]
    radius,centre=max(good,key=lambda q:q[0]) if good else (0.,None)
    if radius<1.2-1e-7 and hasattr(shapely,"maximum_inscribed_circle") and not p.is_empty:
        for part in polygons(p):
            circle=shapely.maximum_inscribed_circle(part,tolerance=tolerance)
            candidate=Point(circle.coords[0])
            candidate_radius=float(part.boundary.distance(candidate))
            if candidate_radius>radius:radius,centre=candidate_radius,candidate
        if tolerance<=.001 and 0<1.2-radius<=2*tolerance:
            for part in polygons(p):
                circle=shapely.maximum_inscribed_circle(part,tolerance=.00001)
                candidate=Point(circle.coords[0]);candidate_radius=float(part.boundary.distance(candidate))
                if candidate_radius>radius:radius,centre=candidate_radius,candidate
    return dict(fits=radius>=1.2-1e-7,diameter_m=2.4,radius_m=radius,
        centre_xz_m=[centre.x,centre.y] if centre else None,
        shape_basis="room own exterior; <=0.05m floor seams closed, holes filled; no navmesh or neighbour floor")

class PartConnectivity(Connectivity):
    """Local pair bridge only: expanded(i) intersect expanded(j) intersect nav."""
    def __init__(self,nav,forbidden=None):
        if forbidden is not None and not forbidden.is_empty:nav=proxy_difference(nav,forbidden)
        super().__init__(nav)
        self.forbidden=forbidden
    def _pair(self,a,b,mask=None):
        key=tuple(sorted((a.wkb,b.wkb)))+(mask.wkb if mask is not None else b"",)
        if key in self.pair_cache:return self.pair_cache[key]
        dist=a.distance(b);bridge=GeometryCollection();accepted=False;support=None
        if dist<=self.seam+1e-9:
            accepted=True;kind="original_parts_seam_le_0_05"
            joined=proxy_intersection(a.buffer(.03,join_style=2),b.buffer(.03,join_style=2))
            bridge=proxy_difference(joined,proxy_union(a,b))
        elif dist<=2*self.expansion+1e-9:
            support=proxy_intersection(proxy_intersection(self.expanded(a),self.expanded(b)),self.nav)
            if mask is not None:support=proxy_intersection(support,mask)
            supported=[p for p in polygons(support) if p.intersection(a).area>1e-10 and p.intersection(b).area>1e-10]
            accepted=bool(supported);kind="distinct_parts_pair_overlap_native_nav"
            if accepted:
                support=shapely.union_all(supported);bridge=proxy_difference(support,proxy_union(a,b))
        else:kind="outside_pair_expansion"
        if self.forbidden is not None and not bridge.is_empty:bridge=proxy_difference(bridge,self.forbidden)
        if mask is not None:bridge=proxy_intersection(bridge,mask)
        result=dict(kind=kind,distance_m=float(dist),bridge=bridge,accepted=accepted)
        if accepted and support is not None:result["nav_support"]=support
        self.pair_cache[key]=result
        return result
    def record(self,g):
        mask=self.masks.get(g.wkb);key=(g.wkb,mask.wkb if mask is not None else None)
        if key in self.cache:return self.cache[key]
        raw=sorted(polygons(g),key=lambda p:(-p.area,p.bounds))
        parent=list(range(len(raw)));edges=[]
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        if raw:
            tree=shapely.STRtree(raw)
            for i,a in enumerate(raw):
                for j in sorted(map(int,tree.query(a.buffer(.60000001)))):
                    if j<=i:continue
                    rec=self._pair(a,raw[j],mask)
                    if not rec["accepted"]:continue
                    ai,bi=find(i),find(j)
                    if ai!=bi:parent[bi]=ai
                    edges.append(dict(part_a=i,part_b=j,distance_m=rec["distance_m"],kind=rec["kind"],
                        expansion_m=self.expansion,bridge_geometry_xz_m=shapely.geometry.mapping(rec["bridge"]),
                        nav_support_geometry_xz_m=shapely.geometry.mapping(rec["nav_support"]) if "nav_support" in rec else None))
        groups={}
        for i,p in enumerate(raw):groups.setdefault(find(i),[]).append(p)
        grouped=sorted([shapely.union_all(parts) for parts in groups.values()],key=lambda p:-p.area)
        env=own_outline(g);opening=env.buffer(-.3,join_style=2).buffer(.3,join_style=2)
        result=dict(groups=grouped,envelope=env,links=edges,raw_part_count=len(raw),
            direct_group_count=len(grouped),opened_count=len([p for p in polygons(opening) if p.area>=.05]),opening=opening)
        if len(self.cache)>512:self.cache.clear()
        self.cache[key]=result
        return result
    def envelope(self,g):return own_outline(g)
    def certificate(self,g):
        r=super().certificate(g)
        r.update(rule="distinct_part_seam_nav_v6",shape_envelope_xz_m=shapely.geometry.mapping(own_outline(g)),
            shape_basis="own outline only; navigation bridges are separate passage evidence",
            bridge_policy="dilate(part_i,.3) intersect dilate(part_j,.3) intersect nav, i!=j; excluded other retained polygons",
            forbidden_other_retained_area_m2=self.forbidden.area if self.forbidden is not None else 0.)
        return r

def linear_parts(g):
    if g.geom_type=="LineString":return [g]
    return [p for q in getattr(g,"geoms",()) for p in linear_parts(q)]

def surviving_design(line,interfaces):
    """Span surviving collinear interfaces per designed leg, bridging scan holes.

    A crossing point is not an active old boundary. The span deliberately counts
    furniture in gaps between measured floor faces, rather than hiding it.
    """
    out=[]
    for aa,bb in zip(line.coords,list(line.coords)[1:]):
        a=np.array(aa);b=np.array(bb);vector=b-a;length=float(np.linalg.norm(vector))
        if length<1e-6:continue
        unit=vector/length;intervals=[]
        segment=LineString([aa,bb])
        for p in linear_parts(interfaces.intersection(segment.buffer(2e-6,cap_style=2))):
            coords=np.asarray(p.coords)
            for x,y in zip(coords,coords[1:]):
                delta=y-x;size=float(np.linalg.norm(delta))
                if size<1e-4 or abs(float(delta@unit))/size<.9999:continue
                if max(abs(float(unit[0]*(x[1]-a[1])-unit[1]*(x[0]-a[0]))),abs(float(unit[0]*(y[1]-a[1])-unit[1]*(y[0]-a[0]))))>3e-6:continue
                intervals.extend([max(0.,min(length,float((x-a)@unit))),max(0.,min(length,float((y-a)@unit)))])
        if intervals and max(intervals)-min(intervals)>1e-4:
            out.append(LineString([a+unit*min(intervals),a+unit*max(intervals)]))
    return shapely.union_all(out)

def raw_cells(g,line):
    # A full planar partition; no raw part is assigned by nearest seed.
    x0,z0,x1,z1=g.bounds
    b=box(x0-1,z0-1,x1+1,z1+1)
    try:cells=list(polygons(split(b,line)))
    except shapely.GEOSException:return []
    if len(cells)!=2:return []
    try:out=[shapely.union_all(list(polygons(g.intersection(p)))) for p in cells]
    except shapely.GEOSException:return []
    if any(x.is_empty or x.area<1e-8 for x in out):return []
    if abs(sum(x.area for x in out)-g.area)>1e-6:return []
    return out

def wrap_count(geoms):
    p=[own_outline(g) for g in geoms]
    return sum(a.convex_hull.intersection(b).area/b.area>=.3 for i,a in enumerate(p) for j,b in enumerate(p) if i!=j and b.area>0)

def thin_only(g):
    p=own_outline(g)
    return p.buffer(-.75,join_style=2).area<=1e-8

class Repair:
    def __init__(self,furniture,axis,cap=35.,max_nodes=24):
        self.furniture=furniture;self.axis=axis;self.cap=cap
        self.nodes=0;self.max_nodes=max_nodes;self.audit=[];self.deadline=time.monotonic()+120
    def primitive_lines(self,g,angles=None,near_line=None):
        p=planning_outline(g)
        angles=angles or [self.axis,self.axis+90]
        for angle in angles:
            loc=rotate(p,-angle,origin=(0,0));a,b,c,d=loc.bounds
            extent=max(c-a,d-b)+3.
            values=set(np.arange(a+.15,c-.15,.15))|{(a+c)/2}
            # Junctions and furniture edges avoid coarse-grid coincidences.
            verts=np.asarray(loc.exterior.coords) if loc.geom_type=="Polygon" else np.concatenate([np.asarray(x.exterior.coords) for x in polygons(loc)])
            if len(verts)>0:
                values.update(verts[::max(1,len(verts)//70),0])
            for item in self.furniture:
                f=rotate(item["geometry"],-angle,origin=(0,0))
                if f.intersects(loc):
                    values.update([f.bounds[0]-.04,f.bounds[2]+.04])
            dd=defects(p)
            cores=list(polygons(p.buffer(-.3,join_style=2)))
            if len(cores)>1:
                for q in cores[1:]:
                    u,v=nearest_points(cores[0],q);mid=rotate(Point((u.x+v.x)/2,(u.y+v.y)/2),-angle,origin=(0,0))
                    values.update([mid.x-.05,mid.x,mid.x+.05])
            broad=p.buffer(-.75,join_style=2).buffer(.75,join_style=2)
            for q in polygons(broad):
                s=rotate(q,-angle,origin=(0,0))
                values.update([s.bounds[0],s.bounds[2]])
            values=sorted(x for x in values if a+.04<x<c-.04)
            if len(values)>160:values=[values[int(i)] for i in np.linspace(0,len(values)-1,160)]
            for x in values:
                ll=LineString([(x,b-extent),(x,d+extent)])
                line=rotate(ll,angle,origin=(0,0))
                if near_line is not None:
                    n=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
                    old=np.asarray(near_line.coords).mean(axis=0)@n
                    old_direction=math.degrees(math.atan2(near_line.coords[-1][1]-near_line.coords[0][1],near_line.coords[-1][0]-near_line.coords[0][0]))%180
                    new_direction=(angle+90)%180
                    parallel_error=min(abs(old_direction-new_direction),180-abs(old_direction-new_direction))
                    if parallel_error<=15.00001 and abs(x-old)>.60001:continue
                yield line
    def bent_lines(self,g,near_line=None):
        """At most three wall-axis segments, with every end outside the floor."""
        p=rotate(planning_outline(g),-self.axis,origin=(0,0))
        a,b,c,d=p.bounds;extra=max(c-a,d-b)+3
        coords=[z for q in polygons(p) for z in q.exterior.coords]
        levels=[]
        broad=p.buffer(-.75,join_style=2).buffer(.75,join_style=2)
        for dim,lo,hi in [(0,a,c),(1,b,d)]:
            vals={(lo+hi)/2}
            for z in coords:vals.add(round(z[dim],3))
            for q in polygons(broad):vals.update([q.bounds[dim],q.bounds[dim+2]])
            for f in self.furniture:
                fg=rotate(f["geometry"],-self.axis,origin=(0,0))
                if fg.intersects(p):vals.update([fg.bounds[dim]-.04,fg.bounds[dim+2]+.04])
            vals=sorted(v for v in vals if lo+.06<v<hi-.06)
            if len(vals)>14:vals=[vals[int(k)] for k in np.linspace(0,len(vals)-1,14)]
            levels.append(vals)
        xs,ys=levels
        for x in xs:
            for y in ys:
                for yy in [b-extra,d+extra]:
                    for xx in [a-extra,c+extra]:
                        yield rotate(LineString([(x,yy),(x,y),(xx,y)]),self.axis,origin=(0,0))
        # A stepped boundary may go around furniture without forming an L room.
        for angle in [self.axis,self.axis+90]:
            pp=rotate(planning_outline(g),-angle,origin=(0,0))
            aa,bb,cc,dd=pp.bounds
            local_old=rotate(near_line,-angle,origin=(0,0)) if near_line is not None else None
            centre=(local_old.bounds[0]+local_old.bounds[2])/2 if local_old is not None else (aa+cc)/2
            xvals={centre-.6,centre-.3,centre,centre+.3,centre+.6}
            yvals={(bb+dd)/2}
            for f in self.furniture:
                fg=rotate(f["geometry"],-angle,origin=(0,0))
                if fg.intersects(pp) and fg.bounds[0]-.6<=centre<=fg.bounds[2]+.6:
                    xvals.update([fg.bounds[0]-.04,fg.bounds[2]+.04])
                    yvals.update([fg.bounds[1]-.04,fg.bounds[3]+.04])
            xvals=sorted(x for x in xvals if aa+.06<x<cc-.06 and abs(x-centre)<=.60001)
            yvals=sorted(y for y in yvals if bb+.06<y<dd-.06)
            if len(xvals)>9:xvals=[xvals[int(k)] for k in np.linspace(0,len(xvals)-1,9)]
            if len(yvals)>12:yvals=[yvals[int(k)] for k in np.linspace(0,len(yvals)-1,12)]
            for x1 in xvals:
                for x2 in xvals:
                    if abs(x1-x2)<.04:continue
                    for y in yvals:
                        yield rotate(LineString([(x1,bb-extra),(x1,y),(x2,y),(x2,dd+extra)]),angle,origin=(0,0))
    def classify(self,g,cause=None):
        if cause=="CORRIDOR" and thin_only(g):return "CORRIDOR"
        if g.area<6-1e-8:return "NECK_FRAGMENT" if cause=="NECK" else "FLOOR_AREA_BELOW_6"
        if cause=="NECK" and not disk(g)["fits"]:return "NECK_FRAGMENT"
        return None
    def choices(self,g,cause=None,near_line=None,neighbors=(),fallback=True):
        preview=planning_outline(g)
        ratio=g.area/max(preview.area,1e-9)
        answer=[]
        original_narrow=shapely.union_all(defects(preview)["corridor"]) if cause=="CORRIDOR" else GeometryCollection()
        opened=sorted(polygons(preview.buffer(-.3,join_style=2).buffer(.3,join_style=2)),key=lambda x:-x.area)
        main_core=opened[0] if opened else GeometryCollection()
        hanging_cores=shapely.union_all(opened[1:]) if cause=="NECK" else GeometryCollection()
        angle_batches=[[self.axis,self.axis+90],None]
        if fallback:angle_batches.append([self.axis+v+k for v in [-15,-10,-5,5,10,15] for k in [0,90]])
        for angles in angle_batches:
            for line in (self.bent_lines(g,near_line) if angles is None else self.primitive_lines(g,angles,near_line)):
                if time.monotonic()>self.deadline:raise TimeoutError("FINITE_SHAPE_CANDIDATE_BUDGET_120_SECONDS")
                try:
                    cells=raw_cells(preview,line)
                    if not cells:continue
                    estimated=[q.area*ratio for q in cells]
                    discard=[];bad=0;large=[]
                    for q,area in zip(cells,estimated):
                        if area<6 and cause not in ("NECK","CORRIDOR"):bad+=10
                        if area<6 and cause=="CORRIDOR" and not thin_only(q):bad+=10
                        forced="NECK_FRAGMENT" if cause=="NECK" and (area<6 or not disk(q,tolerance=.01)["fits"]) else "CORRIDOR" if cause=="CORRIDOR" and thin_only(q) else "FLOOR_AREA_BELOW_6" if area<6 else None
                        if forced:discard.append(area);continue
                        if not disk(q,tolerance=.01)["fits"] or short_side(q)<2.4-1e-7:bad+=10
                        bad+=defects(q)["neck_count"]+defects(q)["corridor_count"]
                        large.append(q)
                    if not large or bad>=10:continue
                    # No newly designed cut deliberately creates a wedge.
                    try:m=cut_measure(line,preview,self.furniture)
                    except shapely.GEOSException:continue
                    wraps=wrap_count(large+list(neighbors)) if neighbors else wrap_count(large)
                    over=sum(max(0,math.ceil((x-1e-7)/self.cap)-1) for x in estimated if x>=6)
                    # Cut at the actual junction, rather than hiding a remnant
                    # just below the reviewer's area/length reporting threshold.
                    junction_error=sum(proxy_intersection(q,original_narrow).area for q in large) if not original_narrow.is_empty else 0.
                    if cause=="NECK":
                        junction_error=sum(min(proxy_intersection(q,main_core).area,proxy_intersection(q,hanging_cores).area) for q in cells)
                    score=(wraps,bad, m["furniture_intersection_length_m"]>.5+1e-7,
                           junction_error,sum(discard),over,m["furniture_intersection_length_m"],
                           abs(estimated[0]-estimated[1]),len(line.coords)-1)
                    answer.append((score,line,m))
                except shapely.GEOSException:
                    continue
            if answer and min(x[0][:4] for x in answer)==(0,0,False,0.):break
        answer.sort(key=lambda x:x[0])
        self.audit.append(dict(cause=cause or "CAP",feasible_candidates=len(answer),best_score=list(answer[0][0]) if answer else None,
            best_furniture_length_m=answer[0][2]["furniture_intersection_length_m"] if answer else None))
        return answer
    def solve(self,g,cause=None,depth=0):
        self.nodes+=1
        forced=self.classify(g,cause)
        if forced:return [dict(g=g,forced=forced,error=None)],[]
        dd=defects(g)
        if g.area<=self.cap+1e-8 and short_side(g)>=2.4-1e-7 and disk(g)["fits"] and not dd["neck_count"] and not dd["corridor_count"]:
            return [dict(g=g,forced=None,error=None)],[]
        # Own disconnected contours are separate shape candidates even if a
        # threshold on navmesh connects the measured parts.
        own_parts=list(polygons(own_outline(g)))
        significant=[q for q in own_parts if q.area>=.5]
        if len(significant)>1:
            parts=[g.intersection(q) for q in own_parts]
            out=[];cuts=[]
            for p in parts:
                if p.area<1e-9:continue
                child,cs=self.solve(p,"NECK",depth+1);out+=child;cuts+=cs
            if abs(sum(x["g"].area for x in out)-g.area)<=1e-6:return out,cuts
        if depth>=10 or self.nodes>self.max_nodes:
            return [dict(g=g,forced=None,error="FINITE_SHAPE_SEARCH_BUDGET")],[]
        mode="NECK" if dd["neck_count"] else "CORRIDOR" if dd["corridor_count"] else None
        choices=self.choices(g,mode)
        if choices and choices[0][0][1]>=dd["neck_count"]+dd["corridor_count"] and g.area<=self.cap and mode:
            return [dict(g=g,forced=None,error="NO_IMPROVING_OWN_OUTLINE_JUNCTION")],[]
        best=None
        for score,line,m in choices[:4]:
            parts=raw_cells(g,line)
            if not parts:continue
            out=[];cuts=[];good=True
            for p in parts:
                forced=self.classify(p,mode)
                if not forced and (not disk(p)["fits"] or short_side(p)<2.4-1e-7):
                    good=False;break
                leaves,extra=self.solve(p,mode,depth+1);out+=leaves;cuts+=extra
            if not good:continue
            errs=sum(x["error"] is not None for x in out)
            final_score=(errs,sum(x["g"].area for x in out if x["forced"]),sum(cut[2]["furniture_intersection_length_m"]>.5 for cut in cuts)+int(m["furniture_intersection_length_m"]>.5))
            cuts=[(line,mode or "CAP",m,g)]+cuts
            if best is None or final_score<best[0]:best=(final_score,out,cuts)
            if errs==0:break
        if best:return best[1],best[2]
        return [dict(g=g,forced=None,error="NO_FEASIBLE_OWN_OUTLINE_AXIS_SHAPE_REPAIR")],[]
