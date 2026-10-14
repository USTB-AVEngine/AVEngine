"""Contact depth and area must ignore parts outside the touching footprint."""
import numpy as np
import pytest
from tools.assets.cat_contact_patches import clipped_internal_depths,clipped_penetrating_patches
pytestmark=pytest.mark.fast_unit

def small_crossing():
    xy=np.array([[-.02,-.02],[.03,-.02],[-.02,.03]])
    source=np.c_[xy,.2*xy[:,0]-.0004][None]
    target=np.array([[[0,0,0],[.005,0,0],[0,.005,0]]],float)
    return source,target

def test_internal_depth_uses_only_the_actual_contact_footprint():
    xy=np.array([[-.05,-.001],[.05,-.001],[0,.001]])
    source=np.c_[xy,.2*xy.sum(1)][None]
    target=np.array([[[-.001,-.05,0],[-.001,.05,0],[.001,0,0]]],float)
    signed=source[0,:,2]
    first=min(signed.max(),-signed.min())
    normal=np.cross(source[0,1]-source[0,0],source[0,2]-source[0,0]);normal/=np.linalg.norm(normal)
    signed=(target[0]-source[0,0])@normal
    second=min(signed.max(),-signed.min())
    assert min(first,second)>.009
    actual=clipped_internal_depths(source,target)[0]
    assert .0003<actual<.0005

def test_visible_area_is_the_clipped_fragment_of_a_large_triangle():
    source,target=small_crossing()
    area,depth=clipped_penetrating_patches(source,target,np.ones(1))
    assert area[0]==pytest.approx(8e-6*np.sqrt(1+.2**2),abs=1e-12)
    assert depth[0]==pytest.approx(.0004,abs=1e-12)
    assert area[0]<25e-6

def test_large_deep_contact_still_exceeds_unchanged_owner_limits():
    xy=np.array([[-.03,-.03],[.09,-.03],[-.03,.09]])
    source=np.c_[xy,.2*xy[:,0]-.003][None]
    target=np.array([[[0,0,0],[.03,0,0],[0,.03,0]]],float)
    area,depth=clipped_penetrating_patches(source,target,np.ones(1))
    assert area[0]>25e-6
    assert depth[0]==pytest.approx(.003,abs=1e-12)
    assert clipped_internal_depths(source,target)[0]>.002

def test_contact_measurements_are_invariant_under_rigid_scene_placement():
    source,target=small_crossing()
    ax=.41;ay=-.29
    rx=np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    ry=np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    rotation=rx@ry;shift=np.array([2.,-1.5,8.])
    area,depth=clipped_penetrating_patches(source,target,np.ones(1))
    actual=clipped_penetrating_patches(source@rotation.T+shift,target@rotation.T+shift,np.ones(1))
    assert np.allclose(actual[0],area,atol=1e-12)
    assert np.allclose(actual[1],depth,atol=1e-12)
