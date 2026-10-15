from types import SimpleNamespace
import numpy as np
import pytest
from avengine.timeline.v7_motion_quartet_template import authored_pairs,find_template_route,MotionTemplateUnavailable
from avengine.timeline.v7_phase2_sources import qualify_registered_walk_tts
pytestmark=pytest.mark.fast_unit

def test_complete_tts_pair_has_distinct_roles_and_physical_voice_references():
    def sound(name,role,duration,voice):
        return {'scenario_id':'ordinary unseen script','sound_asset_id':name,'role_id':role,
                'synthesis_readback':{'duration_s':duration},'voice_reference':{'speaker_id':voice}}
    manifest={'sounds':[sound('a','one',2.2,'v1'),sound('b','two',1.4,'v2'),sound('cut','two',4.,'v3')]}
    assert [(a['sound_asset_id'],b['sound_asset_id'])for a,b in authored_pairs(manifest)]==[('a','b')]

def test_unseen_open_room_gets_native_straight_windows_and_alternating_distance_trends():
    room={'floor_polygon':{'type':'Polygon','coordinates':[[[-6,-6],[6,-6],[6,6],[-6,6],[-6,-6]]]},'measured_floor_y_m':0.}
    pf=SimpleNamespace(snap_point=lambda point:np.asarray(point,float))
    result=find_template_route(room,pf,97)
    assert result['candidate_native_root_samples']==901
    assert [x['branch']for x in result['distance_branches']]==['nearer','farther','nearer']
    assert .8<=result['speed_m_s']<=1.2
    assert result['minimum_emission_radial_cosine']>=np.cos(np.deg2rad(30))
    assert result['straight_length_m']<result['speed_m_s']*6.9

def test_tts_walk_permission_does_not_replace_complete_native_admission():
    scene={'pending_registered_walk_tts':['a'],'sources':{'a':{'asset_id':'registered body'}}}
    with pytest.raises(ValueError,match='complete actual native'):qualify_registered_walk_tts(scene,{'native_motion_admissions':{}})


def test_turns_use_exact_distance_clock_and_do_not_interrupt_speech_or_post_query():
    from avengine.assets.roster_motion.timed_straight_clock import timed_straight_clock
    from avengine.assets.roster_motion.straight_window_routes import stadium_samples
    times=np.arange(901)/30.;distance,velocity,profile=timed_straight_clock(times,.9)
    _,_,segments=stadium_samples(distance+profile['phase_distance_offset_m'],profile['straight_length_m'],profile['turn_radius_m'])
    for offset in [0,10,20]:
        selected=(times>=offset+1)&(times<=offset+7.5)
        assert np.min(velocity[selected])>=.4
        emitting=(times>=offset+1)&(times<=offset+3.4)
        assert np.allclose(velocity[emitting],.9)
        assert len(set(segments[selected]))==1
    assert distance[0]==pytest.approx(0.)
    assert np.all(np.diff(distance)>0)
    assert velocity.min()>=.4

def test_native_narrow_room_uses_one_complete_straight_pass_with_all_four_conditions():
    room={'floor_polygon':{'type':'Polygon','coordinates':[[[-6,-.5],[6,-.5],[6,.5],[-6,.5],[-6,-.5]]]},'measured_floor_y_m':0.}
    pf=SimpleNamespace(snap_point=lambda point:np.asarray(point,float))
    route=find_template_route(room,pf,97)
    assert route['single_pass'] and route['all_four_families_retained']
    assert route['motion_template_windows_s']==[0.]
    assert route['speed_m_s']>=.8

def test_complete_native_gait_finish_preserves_emission_speed_and_planted_pause():
    from avengine.assets.roster_motion.navmesh_execution import human_distance_driven_clock
    times,moving,velocity,distance,clock,proof=human_distance_driven_clock(np.array([0.,2.]),[0.,.5,1.,1.5],.85,[[0.,30.]],speed_range=(.4,1.2),timed_route={'single_pass':True})
    assert not moving[times>=8].any()
    assert len(set(distance[times>=8]))==1
    assert min(abs(clock[-1]-.5*i)for i in range(30))<1e-10
    assert .8<=np.min(velocity[(times>=1)&(times<=3.4)])<=1.2
    assert proof['complete_step_source_phase_s']==pytest.approx(clock[-1])

def test_formal_catalog_does_not_require_an_invented_optional_voice_flag():
    from avengine.timeline.v7_motion_quartet_template import choose_template_assets
    catalog=[{'asset_id':str(i),'kind':'human','controlled_shirt':True,'standing_registered_qualified':True,'walk_allowed':True,'visual_gender':'female' if i%2 else 'male','color':str(i)}for i in range(8)]
    chosen=choose_template_assets(catalog,[{'voice_gender':'female'},{'voice_gender':'male'}],3)
    assert len(chosen)==4
