from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.roster_motion.episode import clear_native_foot_sound_paths

pytestmark = pytest.mark.fast_unit


def test_scan_obstacle_rejects_the_low_shoe_emitter_but_inserted_bodies_do_not():
    rays=[]
    hit=SimpleNamespace(object_id=-1)
    def cast(ray,distance):
        rays.append((ray,distance))
        return SimpleNamespace(hits=[hit])
    video=SimpleNamespace(
        hs=SimpleNamespace(geo=SimpleNamespace(Ray=lambda a,b:(a,b))),
        mn=SimpleNamespace(Vector3=lambda x:np.asarray(x)),
        sim=SimpleNamespace(cast_ray=cast))
    vertices=np.array([[0.,.025,0.],[0.,.2,0.]])
    feet={'L':{'in_contact':True,'sole_local_m':[0.,0.,0.]}}
    args=(video,[0.,1.3,0.],vertices,feet,[2.,0.,0.],0.)
    assert not clear_native_foot_sound_paths(*args,set())
    hit.object_id=123
    assert clear_native_foot_sound_paths(*args,{123})
    assert np.allclose(rays[0][0][1],np.array([2.,-1.275,0.])/np.linalg.norm([2.,-1.275,0.]))
    assert rays[0][1] == pytest.approx(np.linalg.norm([2.,-1.275,0.])-.05)
