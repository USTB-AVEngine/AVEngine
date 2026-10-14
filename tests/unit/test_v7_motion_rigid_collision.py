import numpy as np
import pytest
from avengine.assets.roster_motion.episode import TriangleClearance, fcl_model
from avengine.assets.roster_motion.contact import yaw_matrix

fcl = pytest.importorskip('fcl')


def cube():
    import trimesh
    mesh = trimesh.creation.box(extents=[.1, .1, .1])
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def gate(obstacle):
    g = object.__new__(TriangleClearance)
    g.fcl=fcl;g.cache={};g.rigid_models={};g.floor_ray=None
    g.lo=np.empty((0,3));g.hi=np.empty((0,3));g.faces=np.empty((0,3),dtype=int)
    g.bystanders=[obstacle]
    return g


@pytest.mark.parametrize('gap', [.019, .021, .15])
def test_cached_original_triangles_keep_frozen_collision_distance_gate(gap):
    vertices, faces = cube()
    root=np.array([1., .2, -2.])
    obstacle=fcl_model(vertices+root+[.1+gap, 0., 0.], faces)
    fast=gate(obstacle);reference=gate(obstacle)
    for yaw in [0., .2, 0.]:
        world=vertices@yaw_matrix(yaw).T+root
        try:
            expected=reference.check_world(world,faces)
        except ValueError:
            with pytest.raises(ValueError):fast.check(vertices,faces,root,yaw)
        else:
            observed, actual_world=fast.check(vertices,faces,root,yaw)
            assert observed == pytest.approx(expected,abs=1e-10)
            assert np.array_equal(actual_world,world)
    assert len(fast.rigid_models)==1
