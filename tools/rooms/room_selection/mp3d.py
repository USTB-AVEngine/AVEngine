"""MP3D source adapter; all measurement and decision algorithms stay shared.

.house and semantic PLY are read-only external inputs. Source Z-up coordinates
rotate to Habitat X,Y-up,Z as (x,z,-y), including native boxes and viewpoints.
Native region boxes are metadata, never measurement clipping boundaries.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Polygon, mapping
from tools.rooms.room_screening.geometry import classify_category, union_projected_polygons
from .geometry import canonical


def read_house(path):
    categories, objects, regions, surfaces, vertices, levels = {}, {}, {}, {}, defaultdict(list), {}
    views = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "C":
            categories[int(f[1])] = f[5].replace("_", " ").lower()
        elif f[0] == "O":
            objects[int(f[1])] = (int(f[2]), int(f[3]))
        elif f[0] == "R":
            lo, hi = np.asarray(f[9:12], float), np.asarray(f[12:15], float)
            regions[int(f[1])] = dict(
                level_index=int(f[2]), region_label=f[5],
                bbox_xz_m=[[float(lo[0]), float(-hi[1])], [float(hi[0]), float(-lo[1])]],
                floor_y_m=float(lo[2]), height_range_m=[float(lo[2]), float(hi[2])],
            )
        elif f[0] == "L":
            levels[int(f[1])] = dict(floor_y_m=float(f[7]))
        elif f[0] == "S" and f[4] == "F":
            surfaces[int(f[1])] = int(f[2])
        elif f[0] == "V":
            vertices[int(f[2])].append((float(f[4]), -float(f[5])))
        elif f[0] == "P":
            views[int(f[3])].append(dict(panorama_id=f[1], camera_m=[float(f[5]),float(f[7]),-float(f[6])], source="native MP3D acquisition position; not placement evidence"))
    for rid, region in regions.items():
        polys=[]
        for sid, srid in surfaces.items():
            if srid==rid and len(vertices[sid])>=3:
                poly=Polygon(vertices[sid])
                if not poly.is_valid: poly=poly.buffer(0)
                if not poly.is_empty: polys.append(poly)
        region["native_floor_polygon"] = mapping(union_projected_polygons(polys))
        region["native_viewpoints"] = views.get(rid, [])
    annotations={iid:dict(instance_id=iid,region_id=r,category=categories.get(c,"unknown"),role=classify_category(categories.get(c,"unknown"))) for iid,(r,c) in objects.items()}
    return annotations, regions, levels


def load_scene(scene_dir):
    """Build the exact ground/instance shape consumed by merged measurements.

    Projection degeneracy cutoff and classifier match room_screening's source
    loader. Floor-window clustering, polygon union, furniture footprint, native
    navmesh intersection, placement rays and split decisions remain unchanged.
    """
    from plyfile import PlyData
    scene_dir=Path(scene_dir); sid=scene_dir.name
    annotations,regions,levels=read_house(scene_dir/(sid+".house"))
    path=scene_dir/(sid+"_semantic.ply")
    ply=PlyData.read(str(path),known_list_len={"face":{"vertex_indices":3}})
    v=ply["vertex"].data
    xyz=canonical(np.column_stack([v["x"],v["y"],v["z"]]))
    faces=np.asarray(ply["face"].data["vertex_indices"],dtype=np.int32)
    ids=np.asarray(ply["face"].data["object_id"],dtype=np.int32)
    unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    order=np.argsort(inv,kind="stable"); starts=np.r_[0,np.cumsum(counts)]
    ground=defaultdict(list); instances=[]; unmapped=0
    for gi,iid in enumerate(unique):
        label=annotations.get(int(iid))
        if label is None:
            unmapped+=int(counts[gi]); continue
        if label["role"] not in {"ground","blocker","review"}: continue
        triangles=xyz[faces[order[starts[gi]:starts[gi+1]]]].astype(np.float64)
        if label["role"]=="ground":
            if label["region_id"]<0: continue
            for face in triangles:
                poly=Polygon(face[:,(0,2)])
                if not poly.is_empty and poly.area>1e-10:
                    ground[label["region_id"]].append(dict(polygon=poly,ys=float(face[:,1].mean()),category=label["category"]))
        else:
            instances.append(dict(**label,triangles=triangles,face_count=len(triangles)))
    return SimpleNamespace(ground=ground,instances=instances,annotations=annotations,regions=regions,levels=levels,
        diagnostics=dict(measurement_backend="MP3D PLY identity adapter -> shared room_screening geometry",face_count=len(faces),unmapped_faces=unmapped,semantic_ply=str(path),annotation=str(scene_dir/(sid+".house")),coordinate_transform="x,y,z -> x,z,-y",native_bbox_policy="metadata only; never clips measured floor"))
