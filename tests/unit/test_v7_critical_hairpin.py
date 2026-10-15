import numpy as np
import pytest
from types import SimpleNamespace
from avengine.assets.roster_motion.critical_hairpin import hairpin_samples,find_critical_hairpin
pytestmark=pytest.mark.fast_unit

def test_short_straight_windows_have_a_smooth_turn_between_them():
    distance=np.linspace(0.,8.,2401);points,tangent,segments=hairpin_samples(distance,2.4,.65)
    assert np.max(np.linalg.norm(np.diff(points,axis=0),axis=1))<=np.diff(distance).max()+1e-9
    assert np.max(np.linalg.norm(np.diff(tangent,axis=0),axis=1))<.006
    assert np.allclose(tangent[segments==0],[1,0])
    assert np.allclose(tangent[segments==1],[-1,0])

def test_unseen_room_gets_sound_straight_and_two_seconds_silent_straight():
    room={'floor_polygon':{'type':'Polygon','coordinates':[[[-4,-4],[4,-4],[4,4],[-4,4],[-4,-4]]]},'measured_floor_y_m':0.}
    pf=SimpleNamespace(snap_point=lambda p:np.asarray(p),distance_to_closest_obstacle=lambda p,r:1.)
    route=find_critical_hairpin(room,pf,42,(.8,1.2),.8)
    assert route is not None and route['critical_hairpin']
    assert route['minimum_stop_s']-route['post_turn_straight_start_s']>=2.5-1e-9
    assert route['turn_speed_m_s']>=.8
    assert route['candidate_native_root_samples']==901
