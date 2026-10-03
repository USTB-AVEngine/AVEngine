"""Read-only MP3D identity adapter for shared room screening measurements.

The semantic PLY and .house are Z-up. Their only coordinate operation is the
same (x,z,-y) rotation used by acoustic compilation and Habitat navigation.
Native room boxes describe identity; measured floor faces own the room scope.
"""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Polygon
from avengine.acoustics.semantic import parse_mp3d_house_bytes, _parse_semantic_ply_bytes
from tools.rooms.room_screening.geometry import classify_category

def read_house(path):
    payload=Path(path).read_bytes();semantics=parse_mp3d_house_bytes(payload)
    objects={};regions={};levels={}
    for number,line in enumerate(payload.decode().splitlines(),1):
        f=line.split()
        if not f:continue
        if f[0]=='O':
            iid,rid,category=map(int,f[1:4])
            name=semantics.category_index_to_raw_label.get(category,'unknown').replace('_',' ').lower()
            value=dict(instance_id=iid,region_id=rid,category=name,role=classify_category(name))
            if iid in objects and objects[iid]!=value:raise ValueError(f'conflicting MP3D object at line {number}')
            objects[iid]=value
        elif f[0]=='R':
            if len(f)<20:raise ValueError(f'truncated MP3D region at line {number}')
            rid=int(f[1]);lo=np.asarray(f[9:12],float);hi=np.asarray(f[12:15],float)
            if not np.isfinite(np.r_[lo,hi]).all() or np.any(lo>hi):raise ValueError('invalid MP3D region box')
            if rid in regions:raise ValueError('duplicate MP3D region')
            regions[rid]=dict(level_index=int(f[2]),region_label=f[5],floor_y_m=float(lo[2]),height_range_m=[float(lo[2]),float(hi[2])],bbox_xz_m=[[float(lo[0]),float(-hi[1])],[float(hi[0]),float(-lo[1])]])
        elif f[0]=='L':levels[int(f[1])]=dict(floor_y_m=float(f[7]))
    if not regions:raise ValueError('no MP3D regions')
    if any(o['region_id']>=0 and o['region_id'] not in regions for o in objects.values()):raise ValueError('object references absent MP3D region')
    return objects,regions,levels

def load_mp3d_scene(scene_dir):
    scene_dir=Path(scene_dir);sid=scene_dir.name
    annotations,regions,levels=read_house(scene_dir/(sid+'.house'))
    path=scene_dir/(sid+'_semantic.ply')
    vertices,faces,ids=_parse_semantic_ply_bytes(path.read_bytes())
    xyz=np.stack((vertices[:,0],vertices[:,2],-vertices[:,1]),axis=-1)
    if not np.isfinite(xyz).all():raise ValueError('nonfinite MP3D semantic vertices')
    unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    order=np.argsort(inv,kind='stable');starts=np.r_[0,np.cumsum(counts)]
    ground=defaultdict(list);instances=[];unmapped=0
    for gi,iid in enumerate(unique):
        label=annotations.get(int(iid))
        if label is None:unmapped+=int(counts[gi]);continue
        if label['role'] not in {'ground','blocker','review'}:continue
        triangles=xyz[faces[order[starts[gi]:starts[gi+1]]]].astype(np.float64)
        if label['role']=='ground':
            if label['region_id']<0:continue
            for face in triangles:
                poly=Polygon(face[:,(0,2)])
                # Identical projection cutoff to the HM3D loader. Raw/native
                # triangles are never edited or dropped to force RLR loading.
                if not poly.is_empty and poly.area>1e-10:
                    ground[label['region_id']].append(dict(polygon=poly,ys=float(face[:,1].mean()),category=label['category']))
        else:instances.append(dict(**label,triangles=triangles,face_count=len(triangles)))
    return SimpleNamespace(ground=ground,instances=instances,annotations=annotations,regions=regions,levels=levels,
        diagnostics=dict(measurement_backend='room_screening.mp3d_geometry -> shared geometry',face_count=len(faces),unmapped_faces=unmapped,semantic_ply=str(path),annotation=str(scene_dir/(sid+'.house')),coordinate_transform='raw Z-up (x,y,z) -> Habitat (x,z,-y)',native_bbox_policy='identity metadata only; not measurement scope',semantic_bounds_habitat_m=[xyz.min(0).tolist(),xyz.max(0).tolist()]))
