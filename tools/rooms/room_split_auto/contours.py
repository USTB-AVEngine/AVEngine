"""Door and medial-axis cuts; measured geometry never uses a raster area."""
from __future__ import annotations
import math
import numpy as np
import shapely
from shapely.geometry import Point, LineString, Polygon, GeometryCollection
from shapely.ops import split, unary_union
from scipy.spatial import Voronoi, QhullError
from tools.rooms.room_selection.media import polygons


def chord(scope, centre, angles=None):
    """Shortest boundary-to-boundary chord through an interior medial point."""
    pt=Point(centre)
    if not scope.covers(pt):return None
    angles=np.arange(0,180,3) if angles is None else angles
    size=max(scope.bounds[2]-scope.bounds[0],scope.bounds[3]-scope.bounds[1])*2+1
    best=None
    for angle in angles:
        unit=np.array([math.cos(math.radians(float(angle))),math.sin(math.radians(float(angle)))])
        line=LineString([np.asarray(centre)-unit*size,np.asarray(centre)+unit*size])
        hit=scope.intersection(line)
        for part in (hit.geoms if hasattr(hit,'geoms') else [hit]):
            if part.geom_type!='LineString' or part.distance(pt)>1e-7:continue
            coords=np.asarray(part.coords)
            if len(coords)<2:continue
            length=float(part.length)
            if length<1e-6:continue
            if best is None or length<best['width_m']:
                best={'line':LineString([coords[0],coords[-1]]),'width_m':length,'centre_xz_m':list(map(float,centre)),'angle_deg':float(angle)}
    return best


def split_at_chord(scope, line, minimum_each=0.0):
    """Split only the polygon crossed by this finite chord; retain every face."""
    a,b=np.asarray(line.coords)[[0,-1]]
    unit=(b-a)/np.linalg.norm(b-a)
    cutter=LineString([a-unit*0.003,b+unit*0.003])
    parts=list(polygons(scope));crossed=[];other=[]
    for poly in parts:
        # A finite door line must not slice remote rooms on the same infinite axis.
        result=list(polygons(split(poly,cutter)))
        if len(result)>1:crossed.extend(result)
        else:other.append(poly)
    if len(crossed)!=2:return None
    groups=[[crossed[0]],[crossed[1]]]
    for poly in other:
        groups[int(crossed[1].distance(poly)<crossed[0].distance(poly))].append(poly)
    children=[unary_union(g) for g in groups]
    if any(g.area<minimum_each-1e-8 for g in children):return None
    if children[0].intersection(children[1]).area>1e-7:return None
    if unary_union(children).symmetric_difference(scope).area>1e-7:return None
    return children


def medial_candidates(scope, width_min, width_max):
    """Boundary Voronoi medial-axis vertices and edge midpoints, in world X/Z."""
    points=[]
    for poly in polygons(scope):
        for ring in [poly.exterior,*poly.interiors]:
            line=LineString(ring.coords).simplify(0.015,preserve_topology=True)
            n=max(4,math.ceil(line.length/0.10))
            points.extend([(p.x,p.y) for p in (line.interpolate(i/n,normalized=True) for i in range(n))])
    if len(points)<4:return []
    points=np.unique(np.round(points,7),axis=0)
    # Bound memory for unusually intricate scan outlines; this is detection resolution only.
    if len(points)>12000:points=points[np.linspace(0,len(points)-1,12000,dtype=int)]
    try:vor=Voronoi(points)
    except QhullError:return []
    samples=list(vor.vertices)
    for edge in vor.ridge_vertices:
        if len(edge)==2 and min(edge)>=0:
            samples.append((vor.vertices[edge[0]]+vor.vertices[edge[1]])/2)
    samples=np.asarray(samples)
    inside=shapely.contains_xy(scope,samples[:,0],samples[:,1])
    samples=samples[inside]
    if not len(samples):return []
    widths=2*shapely.distance(shapely.points(samples),scope.boundary)
    samples=samples[(widths>=width_min-0.10)&(widths<=width_max+0.10)]
    candidates=[];seen=set()
    for centre in samples:
        key=tuple(np.round(centre/0.15).astype(int))
        if key in seen:continue
        seen.add(key)
        candidate=chord(scope,centre)
        if candidate and width_min<=candidate['width_m']<=width_max:
            children=split_at_chord(scope,candidate['line'],6.0)
            if children:
                candidate['side_area_m2']=[float(g.area) for g in children]
                candidates.append(candidate)
    return sorted(candidates,key=lambda c:(c['width_m'],-min(c['side_area_m2']),c['centre_xz_m']))


def structural_partition(scope, doors, floor_y, width_min=0.6,width_max=1.6):
    parts=[scope];cuts=[];door_audit=[]
    for door in sorted(doors,key=lambda d:d['instance_id']):
        rec={k:v for k,v in door.items() if k!='triangles'}
        lo,hi=door['height_range_m']
        if lo>floor_y+1.8 or hi<floor_y+0.3:
            rec['status']='other_floor';door_audit.append(rec);continue
        centre=door['centre_xz_m'];angle=door['axis_angle_deg']
        found=False
        for i,g in enumerate(parts):
            # Open door leaves can be rotated; only locally supported contour chords qualify.
            angles=np.arange(angle-30,angle+31,3)
            c=chord(g,centre,angles)
            if c is None or c['width_m']>max(door['extent_xz_m'])*1.7+0.4:continue
            children=split_at_chord(g,c['line'],0.01)
            if not children:continue
            cid=f'C{len(cuts):03d}'
            cuts.append({'id':cid,'type':'door','line_xz_m':list(c['line'].coords),'width_m':c['width_m'],'semantic_instance_id':door['instance_id'],'semantic_category':door['category'],'semantic_source':door['semantic_source'],'geometry_source':'semantic GLB instance plus structural floor contour','side_area_m2':[float(x.area) for x in children]})
            parts[i:i+1]=children;found=True;break
        rec['status']='cut_applied' if found else 'unreliable_or_not_internal; narrow_only_fallback'
        door_audit.append(rec)
    # Both cues run. Narrow cuts do not depend on finding a semantic door.
    pending=list(parts);parts=[]
    while pending:
        g=pending.pop(0)
        candidates=medial_candidates(g,width_min,width_max) if g.area>=12 else []
        if not candidates:parts.append(g);continue
        c=candidates[0];children=split_at_chord(g,c['line'],6.0)
        if children is None:parts.append(g);continue
        cuts.append({'id':f'C{len(cuts):03d}','type':'narrow','line_xz_m':list(c['line'].coords),'width_m':c['width_m'],'side_area_m2':[float(x.area) for x in children],'geometry_source':'0.10 m boundary-sampled Voronoi medial axis; shortest floor-boundary chord among 3-degree directions; unrounded floor union areas','semantic_source':None})
        pending.extend(children)
    return parts,cuts,door_audit


def body_width(scope):
    """Length-weighted median medial-axis width, ignoring short corner branches."""
    samples=[]
    for poly in polygons(scope):
        for ring in [poly.exterior,*poly.interiors]:
            line=LineString(ring.coords).simplify(0.015,preserve_topology=True)
            n=max(4,math.ceil(line.length/0.10))
            samples.extend([(p.x,p.y) for p in (line.interpolate(i/n,normalized=True) for i in range(n))])
    if len(samples)<4:return 0.0
    boundary=np.unique(np.round(samples,7),axis=0)
    if len(boundary)>12000:boundary=boundary[np.linspace(0,len(boundary)-1,12000,dtype=int)]
    try:vor=Voronoi(boundary)
    except QhullError:return 0.0
    values=[];weights=[]
    for edge in vor.ridge_vertices:
        if len(edge)!=2 or min(edge)<0:continue
        a,b=vor.vertices[edge];mid=(a+b)/2
        if not scope.contains(Point(mid)):continue
        length=float(np.linalg.norm(a-b))
        if length<0.08:continue
        width=float(2*scope.boundary.distance(Point(mid)))
        values.append(width);weights.append(length*max(width,0.01))
    if not values:return 0.0
    order=np.argsort(values);cum=np.cumsum(np.asarray(weights)[order]);i=min(np.searchsorted(cum,cum[-1]/2),len(order)-1)
    return float(np.asarray(values)[order[i]])
