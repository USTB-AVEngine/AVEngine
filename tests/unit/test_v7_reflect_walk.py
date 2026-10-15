import math
import numpy as np
import pytest
from shapely.geometry import box
from avengine.assets.roster_motion.reflect_walk import reflect_walk,schedule_emissions,SPECIES_SPEED_M_S
from avengine.assets.roster_motion.reflect_navmesh import native_free_predicate,reflected_distance_clock,find_reflected_route

@pytest.mark.parametrize('species',list(SPECIES_SPEED_M_S))
def test_native_species_walk_and_per_family_slots(species):
    lo,hi=SPECIES_SPEED_M_S[species]
    def free(p):return -3<p[0]<3 and -2.5<p[1]<2.5 and not(-.5<p[0]<.5 and -.4<p[1]<.4)
    successes=[]
    for seed in range(8):
        w=reflect_walk(free,[-2.,-1.5],seed*.7,(lo+hi)/2,min_speed=lo,duration_s=10.)
        if w is None:continue
        assert all(free(p) for p in w.positions)
        actual=np.linalg.norm(np.diff(w.positions,axis=0),axis=1)*30
        assert np.min(actual[actual>1e-8])>=lo-1e-10
        assert np.max(actual)<=hi+1e-10
        successes.append(schedule_emissions(w,[-2.,2.],utterance_s=(.8,1.2)))
    assert successes and any(slots for slots in successes)
    assert all(set(('qa06','qa15','qa16','qa17'))<=set(s) for slots in successes for s in slots)


def test_short_leg_can_supply_during_sound_without_post_sound():
    w=reflect_walk(lambda p:True,[0.,0.],0.,.9,min_speed=.8,duration_s=2.)
    slots=schedule_emissions(w,[-2.,0.],utterance_s=(1.2,1.2))
    assert slots and slots[0]['qa06'] and slots[0]['qa15']
    assert not slots[0]['qa16'] and not slots[0]['qa17']


def test_native_predicate_does_not_relax_snap_obstacle_or_camera():
    class PF:
        drift=0.;clearance=1.
        def snap_point(self,p):return np.asarray(p)+[self.drift,0,0]
        def distance_to_closest_obstacle(self,p,r):return self.clearance
    pf=PF();free=native_free_predicate(pf,box(-4,-4,4,4),0,[0,1.25,0],obstacles=[box(2,2,3,3)])
    assert free([2.,0.]);assert not free([1.,0.]);assert not free([2.5,2.5])
    pf.drift=.0021;assert not free([2.,0.])
    pf.drift=0.;pf.clearance=.349;assert not free([2.,0.])


def test_distance_driven_clock_follows_actual_turns():
    p=[[0,0],[.03,0],[.03,.03],[.03,.06]]
    d,v,c=reflected_distance_clock({'positions_xz_m':p},1.2,.2)
    np.testing.assert_allclose(d,[0,.03,.06,.09]);np.testing.assert_allclose(v,.9)
    np.testing.assert_allclose(c,.2+d/1.2)


def test_constructor_uses_fixed_input_camera_and_short_legs():
    class PF:
        def snap_point(self,p):return np.asarray(p)
        def distance_to_closest_obstacle(self,p,r):return min(3-abs(p[0]),2.5-abs(p[2]))
    room={'floor_polygon':box(-3,-2.5,3,2.5).__geo_interface__,'measured_floor_y_m':0.,'placement':{'camera_m':[0,1.25,1.5]}}
    route=find_reflected_route(room,PF(),42,species='cat',utterance_s=1.2,focus='M1')
    assert route['listener_m']==room['placement']['camera_m']
    assert route['candidate_count']>=1 and route['emission_slot']['qa06']
    assert len(route['positions_xz_m'])==901
    assert route['speed_range_m_s']==[.3,.8]
