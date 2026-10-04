import numpy as np
import pytest
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from tools.rooms.check_kujiale_native_capsule_contact import capsule_rays,query_capsule

pytestmark=pytest.mark.fast_unit


def plane(vertices):
    mesh=trimesh.Trimesh(vertices=vertices,faces=[[0,1,2],[0,2,3]],process=False)
    mesh.ray=RayMeshIntersector(mesh)
    return mesh


def probe():
    return {'witness_index':0,'role':'source_1','capsule':{'center_ue_cm':[0.,0.,75.],'radius_cm':20.,'half_height_cm':75.}}


def test_floor_contact_is_not_room_interpenetration():
    floor=plane([[-2,0,-2],[2,0,-2],[2,0,2],[-2,0,2]])
    result=query_capsule(floor,probe())
    assert result['status']=='pass' and result['interior_intersections']==[]


def test_static_wall_inside_native_capsule_is_rejected():
    wall=plane([[.10,0,-2],[.10,2,-2],[.10,2,2],[.10,0,2]])
    result=query_capsule(wall,probe())
    assert result['status']=='fail' and any(x['penetration_m']>.09 for x in result['interior_intersections'])


def test_ue_world_y_axis_is_mapped_to_canonical_depth():
    wall=plane([[-2,0,1.10],[2,0,1.10],[2,2,1.10],[-2,2,1.10]])
    native=probe();native['capsule']['center_ue_cm']=[0.,100.,75.]
    assert query_capsule(wall,native)['status']=='fail'


def test_invalid_capsule_dimensions_refused():
    with pytest.raises(ValueError):capsule_rays([0,0,0],.2,.1)
