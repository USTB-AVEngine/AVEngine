import importlib.util
from pathlib import Path
import numpy as np
import pytest
import trimesh
pytestmark=pytest.mark.fast_unit
spec=importlib.util.spec_from_file_location('cpu_scan_rays',Path(__file__).resolve().parents[1]/'tools/acoustics/cpu_scan_rays.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_bvh_matches_independent_trimesh_first_hits_and_keeps_zero_faces():
    mesh=trimesh.creation.box(extents=[4,3,5]);mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=np.vstack([mesh.faces,[0,0,1]]),process=False)
    rng=np.random.default_rng(20261003);origins=rng.uniform([-.8,-.8,-.8],[.8,.8,.8],size=(200,3));directions=rng.normal(size=(200,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    reference=trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    expected,ids,_=reference.intersects_location(origins,directions,multiple_hits=False)
    engine=m.CPUScanRayIntersector(mesh,leaf_size=2);actual,j,_=engine.intersects_location(origins,directions)
    assert np.array_equal(j,np.arange(len(origins)))
    by_ray=dict(zip(ids,expected,strict=True));np.testing.assert_allclose(actual,np.array([by_ray[i] for i in j]),atol=1e-10,rtol=1e-10)
    assert engine.receipt['omitted_faces']==0 and engine.receipt['input_faces']==len(mesh.faces)

def test_bvh_handles_parallel_miss_and_nearest_surface():
    mesh=trimesh.Trimesh(vertices=[[-2,-2,-2],[2,-2,-2],[0,2,-2],[-2,-2,-4],[2,-2,-4],[0,2,-4]],faces=[[0,1,2],[3,4,5]],process=False)
    engine=m.CPUScanRayIntersector(mesh,leaf_size=1)
    p,i,f=engine.intersects_location([[0,0,0],[0,0,0]],[[0,0,-1],[1,0,0]])
    assert i.tolist()==[0] and f.tolist()==[0];np.testing.assert_allclose(p,[[0,0,-2]])
