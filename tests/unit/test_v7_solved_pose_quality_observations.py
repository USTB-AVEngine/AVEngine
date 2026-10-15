from copy import deepcopy
import pytest
from avengine.assets.roster_motion.visual_observations import solved_pose_quality
pytestmark=pytest.mark.fast_unit
QUALITY={'foot_contact_limit_m':.012,'support_lock_limit_m':.015}


def frame(delta=0.,slide=0.,pelvis=0.):
    return {'pelvis_lowering_m':pelvis,'feet':{'L':{'in_contact':True,'floor_delta_m':delta,'support_lock_error_m':slide},
        'R':{'in_contact':True,'floor_delta_m':0.,'support_lock_error_m':0.}}}


@pytest.mark.parametrize('delta,slide,reach,pelvis',[(.013,0.,0.,0.),(0.,.016,0.,0.),(0.,0.,.002,0.),(0.,0.,0.,.081)])
def test_completed_native_pose_measurements_are_preserved_as_observations(delta,slide,reach,pelvis):
    pose=frame(delta,slide,pelvis);before=deepcopy(pose)
    assert solved_pose_quality(pose,reach,QUALITY,record_only=True)
    assert pose['feet']==before['feet']and pose['pelvis_lowering_m']==before['pelvis_lowering_m']
    assert pose['visual_quality_observations']
    with pytest.raises(ValueError):solved_pose_quality(deepcopy(before),reach,QUALITY,record_only=False)


def test_nonfinite_geometry_measurements_remain_strict():
    with pytest.raises(ValueError,match='nonfinite'):solved_pose_quality(frame(float('nan')),0.,QUALITY,record_only=True)


def test_good_pose_has_no_quality_warning():
    pose=frame();assert solved_pose_quality(pose,0.,QUALITY,record_only=True)==[]
    assert 'visual_quality_observations'not in pose
