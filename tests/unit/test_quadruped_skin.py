import numpy as np
import pytest
from avengine.assets.quadruped_skin import capsule_skin_weights,segment_distance

def fixture():
    names=["Body","Head","Tail"]
    heads=[[0,.53,.25],[0,.75,-.35],[0,.55,.40]]
    feet=np.array([[-.12,0,-.25],[.12,0,-.25],[-.12,0,.25],[.12,0,.25]])
    for side,p in zip(("FL","FR","HL","HR"),feet):
        for n,y,zoff in (("Hip",.48,0),("Knee",.26,.03),("Ankle",.065,0)):
            names.append(n+"_"+side);heads.append([p[0],y,p[2]+zoff])
    return names,np.array(heads),feet
def test_foot_and_muzzle_attach_to_own_bones():
    names,heads,feet=fixture()
    points=np.vstack([feet,[[0,.82,-.55]]])
    w=capsule_skin_weights(points,names,heads,feet,1.,[0,.55,0])
    for i,s in enumerate(("FL","FR","HL","HR")):
        assert w[i,names.index("Ankle_"+s)]==pytest.approx(1)
    assert w[-1,names.index("Head")]==pytest.approx(1)
def test_own_capsule_field_is_finite_normalized_and_fits_four_influences():
    names,heads,feet=fixture();rng=np.random.default_rng(104)
    v=rng.uniform([-.2,0,-.6],[.2,1.,.6],size=(12000,3))
    w=capsule_skin_weights(v,names,heads,feet,1.,[0,.55,0])
    assert np.isfinite(w).all() and np.min(w)>=0
    np.testing.assert_allclose(w.sum(1),1,atol=1e-12)
    assert np.max((w>1e-12).sum(1))<=4
def test_capsule_weight_transitions_are_continuous_along_a_leg():
    names,heads,feet=fixture()
    y=np.linspace(0,.7,20001)
    v=np.column_stack([np.full(len(y),feet[0,0]),y,np.full(len(y),feet[0,2])])
    w=capsule_skin_weights(v,names,heads,feet,1.,[0,.55,0])
    assert np.max(np.abs(np.diff(w,axis=0)))<.002
def test_capsule_scales_with_target_dimensions():
    names,heads,feet=fixture();v=np.array([[-.12,.2,-.25],[0,.82,-.55]])
    a=capsule_skin_weights(v,names,heads,feet,1.,[0,.55,0])
    b=capsule_skin_weights(v*.2,names,heads*.2,feet*.2,.2,[0,.11,0])
    np.testing.assert_allclose(a,b,atol=1e-12)
def test_invalid_target_geometry_is_rejected():
    names,heads,feet=fixture()
    with pytest.raises(ValueError):capsule_skin_weights([[0,np.nan,0]],names,heads,feet,1.,[0,.55,0])
def test_segment_distance_includes_end_caps():
    v=np.array([[0,1,0],[2,0,0],[-1,0,0]])
    np.testing.assert_allclose(segment_distance(v,np.zeros(3),np.array([1,0,0])),[1,1,1])
