"""Two real motion targets, independent times, no resting turn or speed sliver."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import Point,box
from avengine.assets.roster_motion.human_event_route import find_human_seed_route
from avengine.timeline.v7_two_human_motion import paired_corridors,two_human_motion_template
from avengine.timeline.v7_export_scene_policy import measured_actor_metrics
pytestmark=pytest.mark.fast_unit


def geometry():
    poly=box(0.,0.,10.,9.)
    class Pathfinder:
        def snap_point(self,p):return np.asarray(p,float)
        def distance_to_closest_obstacle(self,p,maximum):return min(maximum,poly.exterior.distance(Point(p[0],p[2])))
    return {'floor_polygon':poly.__geo_interface__,'measured_floor_y_m':0.,'placement':{'camera_m':[.5,1.25,.5]}},Pathfinder()


def test_independent_opposing_walks_keep_constant_heading_and_all_crop_speeds():
    room,pf=geometry();seed=find_human_seed_route(room,pf,42,utterance_s=1.8,local_start_s=2.6,post_motion_s=.6)
    a,b,layout=next(paired_corridors(seed,room,pf,first_branch='nearer',second_duration_s=1.6))
    assert a['emission_slot']['start_s']==2.6 and b['emission_slot']['start_s']==17.6
    assert a['emission_slot']['branch']=='nearer'and b['emission_slot']['branch']=='farther'
    assert layout['minimum_pair_distance_m']>=1.2
    assert a['short_window_walk']['walk_end_s']==pytest.approx(6.)
    assert b['short_window_walk']['walk_end_s']==pytest.approx(21.)
    assert np.linalg.norm(np.asarray(a['positions_xz_m'])[-1]-np.asarray(a['positions_xz_m'])[0])<3.3
    assert np.ptp(a['travel_yaws_rad'])==0. and np.ptp(b['travel_yaws_rad'])==0.
    for offset in (0.,5.,10.,15.,20.):
        actors={str(i):{'species_id':'human','root_positions_m':np.column_stack((np.asarray(r['positions_xz_m'])[:,0],np.zeros(901),np.asarray(r['positions_xz_m'])[:,1]))[round(offset*30):round((offset+10)*30):2].tolist()}for i,r in enumerate((a,b))}
        assert all(r['status']=='pass'for r in measured_actor_metrics(actors,15.))


@pytest.mark.parametrize('pair_mode',['opposing','following'])
def test_full_two_person_template_certifies_both_targets_with_original_solver(monkeypatch,tmp_path,pair_mode):
    import soundfile as sf
    from avengine.timeline import v7_registered_sampling
    sounds=[];records={}
    for name,gender,role,duration in [('a','male','one',1.8),('b','female','two',1.6)]:
        path=tmp_path/(name+'.wav');samples=np.full(round(16000*duration),.1);sf.write(path,samples,16000)
        sounds.append({'scenario_id':'test-complete-script','sound_asset_id':name,'role_id':role,'voice_gender':gender,
                       'voice_reference':{'speaker_id':name},'synthesis_readback':{'duration_s':duration}})
        records[name]={'sound_asset_id':name,'normalized_path':str(path),'path':str(path),
                       'header':{'sample_count':len(samples),'sample_rate_hz':16000}}
    monkeypatch.setattr(v7_registered_sampling,'semantic_record',lambda sound,gender:deepcopy(records[sound['sound_asset_id']]))
    catalog=[{'asset_id':str(i),'kind':'human','entity_class':'articulated_human','species_id':'human',
              'controlled_shirt':True,'standing_registered_qualified':True,'walk_allowed':True,
              'visual_gender':'female'if i%2 else'male','color':str(i)}for i in range(8)]
    room,pf=geometry()
    from avengine.timeline import v7_two_human_motion as module
    original=module.paired_corridors;calls=[]
    def after_first(*args,**kwargs):
        calls.append(1)
        return iter(())if len(calls)==1 else original(*args,**kwargs)
    monkeypatch.setattr(module,'paired_corridors',after_first)
    row=two_human_motion_template({'scene_seed':42,'scene_id':'new','motion_source_type':'human_speech','human_motion_event_count':2,'human_motion_pair_mode':pair_mode},
        room,catalog,{'sounds':sounds},[],pf)
    assert len(calls)>1
    assert len(row['asset_ids'])==3 and len(row['motion_requests'])==2
    assert row['silent_asset_ids']==[row['motion_wrong_emitter_control']['asset_id']]
    assert row['motion_wrong_emitter_control']['minimum_planned_wrong_angle_deg']>=24.
    assert row['joint_motion_witness_certificate']['passed']
    assert len(row['joint_motion_witness_certificate']['checks'])==8
    from avengine.construction.joint.families.motion_human import scene_plan_for_row
    typed=scene_plan_for_row(row,room)
    assert len(typed.actors)==3 and len(typed.sounds)==6
    from avengine.timeline.v7_phase2_motion_audio import wrong_registered_pose
    emitters=typed.metadata['planned_emitter_positions_by_asset']
    proxy={'listener_m':list(typed.camera.position_m),'sources':{aid:{'native_visual_kind':'registered_drawable','trajectory':[{'time_s':i/30.,'position_m':p.tolist()}for i,p in enumerate(track)]}for aid,track in emitters.items()}}
    for aid in row['motion_requests']:
        for second in (0.,3.,7.5,18.,22.,30.):
            endpoint,point,separation=wrong_registered_pose(proxy,emitters[aid][round(second*30)],second)
            assert endpoint!=aid and endpoint in proxy['sources'] and separation>=20.
    assert row['motion_quartet_construction']['no_turns']
    assert len(row['human_motion_event_planned_verification'])==2
    assert all(all(v['families'].values())for v in row['human_motion_event_planned_verification'])
    for aid,request in row['motion_requests'].items():
        assert {p['target_asset_id']for p in request['motion_during_condition_plan_for_fit']['plans']}=={aid}
        assert request['speed_m_s']==request['motion_quartet_route']['speed_m_s']
    assert len(row['motion_during_condition_plan']['plans'])==8
    assert all(m['status']=='pass'for c in row['planned_crop_speed_checks']for m in c['metrics'])


def test_joint_partner_obligation_is_checked_before_selecting_a_corridor():
    from avengine.assets.roster_motion.straight_route import _find_straight_route
    room,pf=geometry();seen=[]
    def reject(candidate):
        seen.append(candidate['emission_slot']['start_s'])
        return False
    with pytest.raises(ValueError,match='joint_partner_constraint'):
        _find_straight_route(room,pf,9,utterance_s=1.8,families='all4',max_grid=20,headings=8,keep=1,
                             local_start_s=2.,candidate_validator=reject)
    assert seen


def test_joint_geometry_cpu_budget_is_explicit_instead_of_claiming_no_solution():
    import time
    from avengine.assets.roster_motion.human_event_route import find_human_seed_route
    room,pf=geometry()
    with pytest.raises(TimeoutError,match='CPU budget exhausted'):
        find_human_seed_route(room,pf,42,utterance_s=1.8,local_start_s=2.,cpu_deadline=time.process_time()-1.)


def test_failed_resting_turn_recipe_cannot_be_dispatched_as_admitted_production():
    from avengine.timeline.v7_motion_quartet_template import motion_quartet_template,MotionTemplateUnavailable
    with pytest.raises(MotionTemplateUnavailable,match='resting-turn replay is not admitted'):
        motion_quartet_template({'human_motion_event_count':3},{},[],{},[],object())



def test_following_people_reuse_one_narrow_corridor_with_independent_original_witnesses():
    from avengine.construction.joint.routes import minimum_track_distance
    room,pf=geometry()
    seed=find_human_seed_route(room,pf,42,utterance_s=1.8,local_start_s=2.6,post_motion_s=.6)
    a,b,layout=next(paired_corridors(seed,room,pf,first_branch='nearer',second_duration_s=1.6,pair_mode='following'))
    assert a['emission_slot']['branch']==b['emission_slot']['branch']=='nearer'
    p,q=np.asarray(a['positions_xz_m']),np.asarray(b['positions_xz_m'])
    direction=(p[-1]-p[0])/np.linalg.norm(p[-1]-p[0])
    assert np.linalg.norm(np.cross(direction,q[0]-p[0]))<1e-8
    assert minimum_track_distance(np.column_stack((p[:,0],np.zeros(901),p[:,1])),
        np.column_stack((q[:,0],np.zeros(901),q[:,1])))>=1.2
    assert layout['pair_mode']=='following'
    assert b['emission_slot']['start_s']>=a['short_window_walk']['walk_end_s']+10
