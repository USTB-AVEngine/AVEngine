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
    room,pf=geometry();seed=find_human_seed_route(room,pf,42,utterance_s=1.8,local_start_s=2.)
    a,b,layout=next(paired_corridors(seed,room,pf,first_branch='nearer',second_duration_s=1.6))
    assert a['emission_slot']['start_s']==2. and b['emission_slot']['start_s']==17.
    assert a['emission_slot']['branch']=='nearer'and b['emission_slot']['branch']=='farther'
    assert layout['minimum_pair_distance_m']>=1.2
    assert np.ptp(a['travel_yaws_rad'])==0. and np.ptp(b['travel_yaws_rad'])==0.
    for offset in (0.,5.,10.,15.,20.):
        actors={str(i):{'species_id':'human','root_positions_m':np.column_stack((np.asarray(r['positions_xz_m'])[:,0],np.zeros(901),np.asarray(r['positions_xz_m'])[:,1]))[round(offset*30):round((offset+10)*30):2].tolist()}for i,r in enumerate((a,b))}
        assert all(r['status']=='pass'for r in measured_actor_metrics(actors,15.))


def test_full_two_person_template_certifies_both_targets_with_original_solver(monkeypatch,tmp_path):
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
    row=two_human_motion_template({'scene_seed':42,'scene_id':'new','motion_source_type':'human_speech','human_motion_event_count':2},
        room,catalog,{'sounds':sounds},[],pf)
    assert len(row['asset_ids'])==len(row['motion_requests'])==2
    assert row['motion_quartet_construction']['no_turns']
    assert len(row['human_motion_event_planned_verification'])==2
    assert all(all(v['families'].values())for v in row['human_motion_event_planned_verification'])
    for aid,request in row['motion_requests'].items():
        assert {p['target_asset_id']for p in request['motion_during_condition_plan_for_fit']['plans']}=={aid}
    assert len(row['motion_during_condition_plan']['plans'])==8
    assert all(m['status']=='pass'for c in row['planned_crop_speed_checks']for m in c['metrics'])
