"""World-delta retargeting must not duplicate rotations inherited from a parent."""
import numpy as np
import pytest
from avengine.assets.animal_motion.retarget import local_from_world_delta
pytestmark = pytest.mark.fast_unit

def rz(deg):
    angle=np.deg2rad(deg);c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def test_world_rotation_does_not_repeat_the_parent_motion():
    rest=rz(90);parent_rest=rz(15);parent_pose=rz(30)@parent_rest
    delta=rz(30)
    local=local_from_world_delta(rest,parent_rest,parent_pose,delta)
    actual=parent_pose@np.linalg.inv(parent_rest)@rest@local
    assert np.allclose(actual,delta@rest,atol=1e-12)
    assert np.allclose(local,np.eye(3),atol=1e-12)
    legacy=np.linalg.inv(rest)@delta@rest
    assert not np.allclose(parent_pose@np.linalg.inv(parent_rest)@rest@legacy,delta@rest)

def test_three_joint_hierarchy_reaches_each_requested_world_rotation():
    parent_rest=parent_pose=np.eye(3)
    for rest,delta in ((rz(15),rz(20)),(rz(90),rz(35)),(rz(-40),rz(-15))):
        local=local_from_world_delta(rest,parent_rest,parent_pose,delta)
        actual=parent_pose@np.linalg.inv(parent_rest)@rest@local
        assert np.allclose(actual,delta@rest,atol=1e-12)
        parent_rest,parent_pose=rest,actual

def test_stationary_parent_keeps_the_single_bone_rotation_contract():
    rest=rz(80);delta=rz(-25)
    actual=local_from_world_delta(rest,np.eye(3),np.eye(3),delta)
    assert np.allclose(actual,np.linalg.inv(rest)@delta@rest,atol=1e-12)
