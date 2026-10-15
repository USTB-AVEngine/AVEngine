import math
import numpy as np
import pytest
from shapely.geometry import box
pytestmark=pytest.mark.fast_unit
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


def test_fixed_camera_placement_can_observe_later_native_entry(monkeypatch):
    from types import SimpleNamespace
    from avengine.capture.v7_visible_native import NativeVisibleVideo
    import avengine.capture.v7_alpha_native as camera
    monkeypatch.setattr(camera,'forward_at',lambda scene,t:[0.,0.,-1.])
    actor={'declaration':{'asset_id':'walker'},'motion_episode':{'qualified':True},'current':0}
    applied=[]
    def apply(entry,scene,frame_index=None):entry['current']=frame_index;applied.append(frame_index)
    def pixels(scene,entries,forward):return {'walker':200 if actor['current']>=30 else 0}
    video=SimpleNamespace(_apply_entry=apply,_native_pair_pixels=pixels)
    maximum,evidence=NativeVisibleVideo._native_observed_pixels(video,{},[actor])
    assert maximum=={'walker':200}
    assert evidence[-1]['time_s']==2.
    assert actor['current']==0 and applied[-1]==0


def test_selected_reflection_leg_is_verified_by_original_motion_solver():
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row,verify_motion_during_native
    class PF:
        def snap_point(self,p):return np.asarray(p)
        def distance_to_closest_obstacle(self,p,r):return min(4-abs(p[0]),3-abs(p[2]))
    room={'floor_polygon':box(-4,-3,4,3).__geo_interface__,'measured_floor_y_m':0.,'placement':{'camera_m':[0.,1.25,2.]}}
    route=find_reflected_route(room,PF(),17,species='cat',utterance_s=1.2,focus='M1')
    slot=route['emission_slot'];offset=slot['offset_s']
    assets=[{'asset_id':k,'kind':'human','walk_allowed':True}for k in ('target','companion')]
    records=[{'sound_asset_id':k,'header':{'sample_rate_hz':16000,'sample_count':n},'screens':{'automatic_activity':{'activity_intervals_samples':[[0,n]]}}}for k,n in [('voice',19200),('other',16000)]]
    requests=[]
    for qa,branch,flag in [('QA-06','moving','qa06'),('QA-15',slot['branch'],'qa15'),('QA-16',None,'qa16'),('QA-17','yes','qa17')]:
        if slot[flag]:requests.append({'qa_id':qa,'branch':branch,'target_asset_id':'target','speed_m_s':route['speed_m_s'],'speed_range_m_s':[.3,.8],'minimum_motion_s':1.5,'reserve_tail_s':.5,'end_hold_s':10-min(9.8,slot['leg'][1]/30-offset-.1),'offsets_s':[offset],'wet_tail_planning_bound_s':None})
    row={'asset_ids':['target','companion'],'motion_quartet_template_revision':'test','motion_requests':{'target':{'action':'walk'}},'proposed_sound_asset_ids':{'target':'voice','companion':'other'},'motion_source_start_s':{'target':slot['start_s']-offset,'companion':0},'camera_yaw_deg':route['camera_yaw_deg'],'motion_during_condition_requests':requests}
    result=prepare_motion_during_row(row,assets,records)
    positions=np.column_stack((np.array(route['positions_xz_m'])[:,0],np.full(901,1.5),np.array(route['positions_xz_m'])[:,1]))
    scene={'listener_m':route['listener_m'],'sources':{'t':{'asset_id':'target','trajectory':[{'time_s':i/30.,'position_m':p.tolist()}for i,p in enumerate(positions)]},'c':{'asset_id':'companion','trajectory':[{'time_s':0.,'position_m':[2.,1.5,2.]},{'time_s':30.,'position_m':[2.,1.5,2.]}]}}}
    report=verify_motion_during_native(result,scene,{'native_motion_admissions':{'t':{'world_roots_m':(positions-[0,1.5,0]).tolist()}}})
    assert report['status']=='pass',report
