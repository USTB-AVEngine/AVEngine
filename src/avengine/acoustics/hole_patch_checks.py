"""Native navigation, candidate placement and original-scan opening checks."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from shapely.geometry import shape

from avengine.acoustics.hole_patch import (quad_grid, check_direct_paths, nav_prism_intersections,
                                         ray_quad_intersections, RULES)


def placement_candidates(house, registry, thresholds, habitat):
    """Reconstruct every bounded camera/source pool using the original sampler."""
    from tools.rooms.room_selection.navigation import sample_navigation, farthest_sample
    p={k:v['value'] for k,v in yaml.safe_load(Path(thresholds).read_text())['parameters'].items()}
    pf=habitat.PathFinder()
    if not pf.load_nav_mesh(house['navmesh_source']):raise ValueError('NATIVE_NAVMESH_LOAD_FAILED')
    nav_vertices=np.asarray(pf.build_navmesh_vertices(),float)
    nav_triangles=np.asarray(pf.build_navmesh_vertex_indices(),int).reshape(-1,3)
    rooms=[json.loads(s) for s in Path(registry).read_text().splitlines() if s.strip()]
    rooms=[r for r in rooms if r['house']==house['house'] and r['room_label'] in house['candidate_room_labels']]
    cameras=[];sources=[];pairs=[];counts=[]
    for r in rooms:
        m=r['metrics'];geometry=shape(m['floor_polygon']);floor_y=m['floor_y_m']
        _,points,clearance,adj,comps=sample_navigation(pf,habitat,geometry,floor_y,p)
        main=max(comps,key=len,default=[])
        ci=farthest_sample(points,[i for i in main if clearance[i]>=p['placement_camera_clearance_m']],p['placement_camera_samples'])
        si=farthest_sample(points,[i for i in main if clearance[i]>=p['placement_source_clearance_m']],p['placement_source_samples'])
        cam=points[ci]+[0,p['camera_height_m'],0];src=points[si]+[0,p['source_height_m'],0]
        counts.append({'room_label':r['room_label'],'cameras':len(cam),'sources':len(src),'native_navigable_grid_samples':len(points)})
        cameras.extend({'room_label':r['room_label'],'position_m':v.tolist()} for v in cam)
        sources.extend({'room_label':r['room_label'],'position_m':v.tolist()} for v in src)
        # Superset includes occluded and out-of-range pairs: rejecting an extra
        # pair is conservative and cannot hide a valid placement obstruction.
        for c in cam:
            for s in src:
                pairs.append({'room_label':r['room_label'],'listener_m':c.tolist(),'source_m':s.tolist()})
    for r in house['acoustic_rooms']:
        p0=r['placement'];cameras.append({'room_label':r['room_label'],'position_m':p0['camera_m']})
        for k in ('source_1_m','source_2_m'):
            pairs.append({'room_label':r['room_label'],'listener_m':p0['camera_m'],'source_m':p0[k]})
    return pf,nav_vertices,nav_triangles,cameras,sources,pairs,counts


def raw_scan_rays(path):
    """Load original GLB positions/indices without textures, filtering or GPU."""
    from avengine.acoustics.gltf import load_glb,_scene_node_instances,_decode_accessor
    from tools.acoustics.cpu_scan_rays import CPUScanRayIntersector
    glb=load_glb(path);d=glb.document
    vertices=[];faces=[];offset=0
    for node_index,world in _scene_node_instances(glb):
        node=d['nodes'][node_index]
        for prim in d['meshes'][node['mesh']]['primitives']:
            if prim.get('mode',4)!=4:raise ValueError('NON_TRIANGLE_RAW_PRIMITIVE')
            v=_decode_accessor(glb,prim['attributes']['POSITION'],expected_type='VEC3',allowed_component_types={5126})
            if 'indices' in prim:
                t=_decode_accessor(glb,prim['indices'],expected_type='SCALAR',allowed_component_types={5121,5123,5125}).astype(np.int64).reshape(-1,3)
            else:t=np.arange(len(v)).reshape(-1,3)
            v=np.asarray(v,float)@world[:3,:3].T+world[:3,3]
            v=v[:,[0,2,1]]*[1,1,-1]
            vertices.append(v);faces.append(t+offset);offset+=len(v)
    v=np.concatenate(vertices);t=np.concatenate(faces)
    intersector=CPUScanRayIntersector(SimpleNamespace(triangles=v[t]))
    intersector.receipt.update(raw_asset=str(path),raw_vertex_count=len(v),raw_triangle_count=len(t),
                             coordinate_transform='raw (x,y,z) -> Habitat (x,z,-y)',omitted_faces=0)
    return intersector


def opening_boxes(vertices, triangles, labels, voxel=.30):
    """Connected occupied voxels of door/window geometry, with conservative boxes.

    Category-wide bounds would span the entire house. Voxel connected components
    keep separate openings distinct, and expand by a voxel to cover unlabelled
    aperture space next to the annotated frame. Large joined components are
    retained conservatively rather than split according to repair outcomes.
    """
    boxes=[]
    for kind in ('door','window'):
        fi=np.flatnonzero([kind in x for x in labels])
        if not len(fi):continue
        vi=np.unique(np.asarray(triangles)[fi]);points=np.asarray(vertices)[vi]
        cells=np.floor(points/voxel).astype(np.int64)
        occupied=set(map(tuple,cells));seen=set()
        for start in sorted(occupied):
            if start in seen:continue
            stack=[start];seen.add(start);comp=[]
            while stack:
                q=stack.pop();comp.append(q)
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        for dz in (-1,0,1):
                            k=(q[0]+dx,q[1]+dy,q[2]+dz)
                            if k in occupied and k not in seen:seen.add(k);stack.append(k)
            a=np.asarray(comp);lo=a.min(0)*voxel-voxel;hi=(a.max(0)+1)*voxel+voxel
            boxes.append({'kind':kind,'bounds_m':[lo.tolist(),hi.tolist()],'occupied_voxels':len(comp)})
    return boxes


def rays_through_boxes(origins,directions,maximum,boxes):
    selected=np.zeros(len(origins),bool)
    for record in boxes:
        lo,hi=np.asarray(record['bounds_m'])
        near=np.zeros(len(origins));far=np.asarray(maximum).copy()
        for axis in range(3):
            d=directions[:,axis];parallel=np.abs(d)<1e-12
            with np.errstate(divide='ignore',invalid='ignore'):
                a=(lo[axis]-origins[:,axis])/d;b=(hi[axis]-origins[:,axis])/d
            near=np.maximum(near,np.where(parallel,-np.inf,np.minimum(a,b)))
            far=np.minimum(far,np.where(parallel,np.inf,np.maximum(a,b)))
            far=np.where(parallel&((origins[:,axis]<lo[axis])|(origins[:,axis]>hi[axis])),-np.inf,far)
        selected|=(far>=near)&(far>.01)&(near<maximum-.005)
    return selected


def visible_opening_rays(quad,cameras,boxes,raw_intersector,step=.1):
    grid=quad_grid(quad,step);violations=[];tested=0
    for cam in cameras:
        c=np.asarray(cam['position_m'],float);delta=grid-c;lengths=np.linalg.norm(delta,axis=1)
        dirs=delta/lengths[:,None];origins=np.repeat(c[None,:],len(grid),axis=0)
        mask=rays_through_boxes(origins,dirs,lengths,boxes)
        ids=np.flatnonzero(mask);tested+=len(ids)
        if not len(ids):continue
        loc,ri,_=raw_intersector.intersects_location(origins[ids],dirs[ids],multiple_hits=False)
        hits=np.full(len(ids),np.inf)
        if len(ri):hits[ri]=np.linalg.norm(loc-origins[ids][ri],axis=1)
        visible=np.flatnonzero(hits>=lengths[ids]-.03)
        for j in visible[:8]:
            k=ids[j];violations.append({'room_label':cam['room_label'],'camera_m':c.tolist(),
                'patch_point_m':grid[k].tolist(),'raw_first_hit_distance_m':None if not np.isfinite(hits[j]) else float(hits[j]),
                'patch_distance_m':float(lengths[k])})
    return {'tested_opening_rays':tested,'visible_opening_violation_count':len(violations),
            'examples':violations[:32],'camera_count':len(cameras),'quad_sample_count':len(grid),
            'grid_step_m':step,'all_orientations':True,'method':'all candidate cameras; door/window component boxes; original raw GLB CPU first-hit rays'}


def opening_cone_overlaps(quad, cameras, boxes):
    """Continuous camera-cone check, independent of visibility grid resolution.

    Any overlap is unresolved unless blocked everywhere by the raw mesh, which
    this bounded CPU trial does not attempt to prove. Thus *all* overlaps are
    withheld. The raw ray check provides evidence, while this conservative cone
    makes admission insensitive to a narrow aperture falling between samples.
    """
    from scipy.spatial import ConvexHull
    from avengine.acoustics.hole_patch import quad_frame
    import itertools
    q=np.asarray(quad,float);hits=[]
    qc,qu,qw,qn,qlu,qlw=quad_frame(q)
    for ci,cam in enumerate(cameras):
        c=np.asarray(cam['position_m'],float)
        for bi,record in enumerate(boxes):
            lo,hi=np.asarray(record['bounds_m'],float)
            if np.all(c>=lo)&np.all(c<=hi):
                hits.append({'camera_index':ci,'opening_box_index':bi});continue
            corners=np.array(list(itertools.product(*zip(lo,hi))))
            vectors=corners-c
            numerator=float((qc-c)@qn);denominator=vectors@qn
            if np.all(numerator*denominator<=0):continue
            if np.all(numerator*denominator>1e-10):
                projected=c+vectors*(numerator/denominator)[:,None]-qc
                pu,pw=projected@qu,projected@qw
                if pu.max()<0 or pu.min()>qlu or pw.max()<0 or pw.min()>qlw:continue
            minimum=float(np.min(np.linalg.norm(vectors,axis=1)))
            factor=max(2.,2*np.max(np.linalg.norm(q-c,axis=1))/max(minimum,1e-6))
            hull=ConvexHull(np.concatenate((c[None,:],c+factor*vectors)))
            poly=q.copy()
            for equation in hull.equations:
                n,offset=equation[:3],equation[3];clipped=[]
                for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                    da,db=a@n+offset,b@n+offset
                    if da<=1e-9:clipped.append(a)
                    if (da<=0)!=(db<=0):clipped.append(a+(b-a)*da/(da-db))
                poly=np.asarray(clipped)
                if not len(poly):break
            if len(poly)>=3:
                area=float(np.linalg.norm(np.sum(np.cross(poly,np.roll(poly,-1,axis=0)),axis=0))*.5)
                if area>1e-9:hits.append({'camera_index':ci,'opening_box_index':bi,'quad_overlap_area_m2':area})
    return hits
