"""Read-only semantic door geometry and fresh bounded-height floor measurements."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import shapely
from tools.rooms.room_selection.geometry import canonical,read_glb,short_side,floor_windows
from tools.rooms.room_screening.geometry import parse_semantic_annotations,GROUND_CATEGORIES
from tools.rooms.room_selection.measurements import load_scene
from avengine.acoustics.gltf import triangle_vertex_colours,_decode_accessor,_scene_node_instances
from avengine.acoustics.semantic import _linear_to_srgb_bytes



def raw_zero_ground_faces(document,ann,palette):
    """Detect zero ground triangles before the shared GLB extractor omits them."""
    result={};doc=document.document
    for node,world in _scene_node_instances(document):
        mesh=doc['meshes'][doc['nodes'][node]['mesh']]
        for primitive in mesh['primitives']:
            attrs=primitive['attributes']
            if 'COLOR_0' not in attrs:continue
            pos=_decode_accessor(document,attrs['POSITION'],expected_type='VEC3',allowed_component_types={5126})
            indices=_decode_accessor(document,primitive['indices'],expected_type='SCALAR',allowed_component_types={5121,5123,5125}).reshape(-1,3)
            verts=(world@np.column_stack([pos,np.ones(len(pos))]).T).T[:,:3]
            tri=verts[indices];bad=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0
            if not bad.any():continue
            acc=doc['accessors'][attrs['COLOR_0']];scale={5121:255.,5123:65535.,5126:1.}[acc['componentType']]
            raw=_decode_accessor(document,attrs['COLOR_0'],expected_type=acc['type'],allowed_component_types={5121,5123,5126},allow_normalized=True)
            rgb=_linear_to_srgb_bytes(np.asarray(raw,float)[:,:3]/scale).astype(np.int64)
            colours=rgb[indices[bad,0]];codes=(colours[:,0]<<16)|(colours[:,1]<<8)|colours[:,2]
            for code in codes:
                iid=palette.get(int(code))
                if iid is None:continue
                a=ann[iid]
                if a['category'] in GROUND_CATEGORIES:result[a['region_id']]=result.get(a['region_id'],0)+1
    return result


def structural_instances(scene_dir):
    path=Path(scene_dir);sid=path.name.split('-',1)[1]
    ann,palette,regions=parse_semantic_annotations(path/f'{sid}.semantic.txt')
    document,mesh=read_glb(path/f'{sid}.semantic.glb')
    linear,_=triangle_vertex_colours(document,mesh)
    rgb=_linear_to_srgb_bytes(linear).astype(np.int64)
    codes=(rgb[:,0]<<16)|(rgb[:,1]<<8)|rgb[:,2]
    unique,inverse,counts=np.unique(codes,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(counts)]
    keys=np.array([c for c,iid in palette.items() if iid is not None],dtype=np.int64)
    colours=np.column_stack([(keys>>16)&255,(keys>>8)&255,keys&255])
    selected=[];invalid_ground=raw_zero_ground_faces(document,ann,palette);door_annotation_count=sum('door' in a['category'] for a in ann.values())
    for index,code in enumerate(unique):
        iid=palette.get(int(code))
        if iid is None and int(code) not in palette and int(code)!=0 and len(keys):
            colour=np.array([(int(code)>>16)&255,(int(code)>>8)&255,int(code)&255]);ds=np.max(abs(colours-colour),axis=1);j=int(ds.argmin())
            if ds[j]<=2 and (ds==ds[j]).sum()==1:iid=palette[int(keys[j])]
        if iid is None:continue
        a=ann[iid];category=a['category']
        wanted='door' in category or any(t in category for t in ('stair','balcony','outdoor','patio','terrace','yard','grass'))
        if not wanted and category not in GROUND_CATEGORIES:continue
        ids=order[starts[index]:starts[index+1]]
        tri=canonical(mesh.vertices[mesh.triangles[ids]])
        if category in GROUND_CATEGORIES:
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);bad=np.linalg.norm(cross,axis=1)==0
            if bad.any():invalid_ground[a['region_id']]=invalid_ground.get(a['region_id'],0)+int(bad.sum())
        if not wanted:continue
        vertices=tri.reshape(-1,3);xz=vertices[:,[0,2]]
        vals,vecs=np.linalg.eigh(np.cov(xz.T)) if len(xz)>2 else (None,np.eye(2))
        axis=vecs[:,-1];bounds=[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()]
        selected.append(dict(instance_id=iid,region_id=a['region_id'],category=category,centre_xz_m=((xz.min(axis=0)+xz.max(axis=0))/2).tolist(),extent_xz_m=np.ptp(xz,axis=0).tolist(),axis_angle_deg=float(np.degrees(np.arctan2(axis[1],axis[0]))),height_range_m=[float(vertices[:,1].min()),float(vertices[:,1].max())],bounds_xyz_m=bounds,semantic_source=str(path/f'{sid}.semantic.txt'),triangles=tri))
    return selected,dict(door_annotation_count=door_annotation_count,door_mesh_instance_count=sum('door' in a['category'] for a in selected),invalid_ground_zero_area_faces_by_region=invalid_ground,ambiguous_palette_regions=list(k for k,v in regions.items() if v.get('ambiguous_colours')))


def measure_job(job,p):
    scene=load_scene(job['scene_directory'])
    rows=[]
    registered={r['region_id']:r for r in job['rows']}
    for rid in sorted(set(scene.regions)|set(scene.ground)):
        if rid<0:continue
        floors=floor_windows(scene.ground.get(rid,[]),p)
        rows.append({'house':job['house'],'region_id':rid,'room_label':f'R{rid}','floor_area_sum_m2':sum(f['geometry'].area for f in floors),'floors':[{'floor_id':f'F{i}','floor_y_m':f['floor_y_m'],'height_range_m':f['height_range_m'],'face_count':f['face_count'],'floor_area_m2':float(f['geometry'].area),'short_side_m':short_side(f['geometry']),'floor_polygon':shapely.geometry.mapping(f['geometry'])} for i,f in enumerate(floors)],'scene_directory':job['scene_directory'],'semantic_source':scene.diagnostics['semantic_glb'],'annotation_source':str(Path(scene.diagnostics['semantic_glb']).with_suffix('.txt')),'navmesh_source':job['navmesh'],'legacy_rooms_source':registered.get(rid,{}).get('legacy_rooms_source'),'ground_measurement':'room_screening.union_projected_polygons via room_selection.floor_windows; fresh read of original semantic GLB; no furniture subtraction','semantic_categories':sorted({a['category'] for a in scene.annotations.values() if a['region_id']==rid}),'ambiguous_semantic_palette':bool(scene.regions.get(rid,{}).get('ambiguous_colours'))})
    return {'house':job['house'],'status':'measured','rows':rows,'source':job['scene_directory'],'diagnostics':scene.diagnostics}
