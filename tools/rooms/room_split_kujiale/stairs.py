"""Remove only the actual stair triangles in each measured floor-height band."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import shape, mapping
from tools.rooms.room_split_kujiale.adapter import read, dump, transform_points
from tools.rooms.room_split_kujiale.usd_mesh import mesh_arrays
from tools.rooms.room_split_auto.pipeline import stair_partition


def prepare(root):
    from pxr import Usd, UsdGeom, Work
    Work.SetConcurrencyLimit(1)
    root=Path(root)
    for h in read(root/'input_plan_v1.json')['houses']:
        adapter=read(root/'scene_adapters_v1'/(h['house']+'.json'))
        markers=[m for m in adapter['markers'] if 'stair' in m['category'] or m['category'] in ('step','steps')]
        prepared=[]
        if markers:
            stage=Usd.Stage.Open(h['source_stage']);cache=UsdGeom.XformCache();unit=UsdGeom.GetStageMetersPerUnit(stage)
            for m in markers:
                arrays=[]
                for prim in Usd.PrimRange(stage.GetPrimAtPath(m['usd_instance'])):
                    if prim.GetTypeName()!='Mesh' or UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
                    v,f,_=mesh_arrays(prim,cache)
                    if len(f):arrays.append(transform_points(v*unit,h['matrix'])[f])
                if arrays:prepared.append(dict(m,triangles=np.concatenate(arrays)))
        rooms={}
        for r in h['rooms']:
            _,stair_parts,audit=stair_partition(shape(r['floor_polygon_xz_m']),r['floor_y_m'],prepared,.3)
            rooms[r['room_label']]=dict(stair_parts=[mapping(p) for p in stair_parts],audit=audit,
                   stair_area_m2=sum(p.area for p in stair_parts),
                   source='only explicit original USD stair-instance triangles in the original .3m floor window; flat neighbouring floor excluded')
        dump(root/'scene_stairs_v1'/(h['house']+'.json'),dict(house=h['house'],stair_instances=len(prepared),rooms=rooms))
        print('STAIRS',h['house'],len(prepared),sum(x['stair_area_m2'] for x in rooms.values()),flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',required=True);v=a.parse_args();prepare(v.root)
