import math
import pytest
from avengine.timeline.v7_natural_orientation import sample_initial_orientation,all_people_face_camera
pytestmark=pytest.mark.fast_unit

def test_unseen_scenes_have_reproducible_mixed_natural_directions():
    args=({'asset_id':'person'},17,[2.,0.,2.],[0.,1.3,0.],[0.,0.,0.])
    assert sample_initial_orientation(*args)==sample_initial_orientation(*args)
    samples=[sample_initial_orientation({'asset_id':'person'},seed,[2.,0.,2.],[0.,1.3,0.],[0.,0.,0.])for seed in range(30)]
    assert {s['mode']for s in samples}=={'random','room_center'}
    assert len({round(s['yaw_rad'],3)for s in samples})>20

def test_people_and_furniture_are_available_natural_targets():
    people=[{'position_m':[0.,0.,-2.],'yaw_rad':math.pi}]
    modes={sample_initial_orientation({'asset_id':'p'},seed,[0.,0.,0.],[3.,1.3,0.],[1.,0.,1.],people,[[2.,0.,2.]])['mode']for seed in range(100)}
    assert {'conversation','furniture','random','room_center'}<=modes

def test_all_frontal_group_is_rejected_but_single_person_is_not_forced_sideways():
    listener=[0.,1.3,0.];people=[{'position_m':[0.,0.,2.],'yaw_rad':0.},{'position_m':[0.,0.,4.],'yaw_rad':0.}]
    assert all_people_face_camera(people,listener)
    people[1]['yaw_rad']=math.pi/2
    assert not all_people_face_camera(people,listener)
    assert not all_people_face_camera(people[:1],listener)
