from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.qa.answerability import (MeshHandle, line_of_sight, listener_azimuth_deg,
    max_concurrent_entities, separation_stats, structural_baselines)
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms.furniture_layout import clock_config
from avengine.rooms.walkable_space import RasterWalkableSpace, NativeRouteWalkableSpace
from avengine.routes.raster_pathfinder import RasterPathfinder


def registry():
    return {'assets':[{'asset_id':f'human_{i}','revision':'v1','entity_class':'articulated_human',
        'identity':{'species_id':'human'},'display_label':f'person {i}',
        'realized_attributes':{'sex_or_gender_label':'male','top_color':color},
        'timeline':{'idle_action_id':'idle','walking_action_id':'walk','walk_phase_period_frames':30,
                    'local_anatomical_forward_axis':[1.,0.,0.]},
        'default_emitter_anchor_id':'mouth','emitter_anchors':[{'anchor_id':'mouth','offset_m':[0.,1.6,0.],
              'offset_space':'final_scaled_asset_root'}]} for i,color in enumerate(['blue','green','red','yellow'])]}


def request(**extra):
    return {'episode_id':'fixture','seed':111,'sampling_policy':cs.POLICY,'source_asset_ids':['human_0','human_1'],
        'profile':{'anchor_count':1,'separation_bin_deg':[15,60],'speech_motion':'all_still','event_relation':'sequential',
                   'reserve_tail_s':1.,'retry_budget_within_profile':30},**extra}


def sounds():
    return [{'sound_asset_id':f'speech_{i}','sound_class':'speech','gender':'M','transcript':f'utterance {i}',
        'sample_count':32000,'sample_rate_hz':16000,'audible_start_sample':800,'audible_end_sample_exclusive':31200,
        'active_duration_s':1.9,'path':f'/prepared/{i}.wav'} for i in range(4)]


def space():
    pf=RasterPathfinder(np.ones((32,32),dtype=bool),bounds_m=[[0,-1,0],[8,1,8]],floor_height_m=0.)
    return RasterWalkableSpace(pf,{'floor_height_m':0.,'resolution_m':.25,'authority':'fixture_retained_grid'})


def clock():
    return clock_config(frame_count=120,frame_rate_hz=15,sample_rate_hz=16000)


def test_fixed_profile_failure_keeps_denominator_and_histogram(monkeypatch):
    r=request();r['profile']['retry_budget_within_profile']=3;p=cs.resolve_condition_profile(r,registry());seen=[]
    def fail(*args,**kwargs):
        seen.append(deepcopy(args[2]));raise cs.CandidateFailure('routes','fixture_no_path')
    monkeypatch.setattr(cs,'sample_routes',fail)
    with pytest.raises(cs.ConditionedPlanningFailure) as error:
        cs.build_conditioned_plan(room={'room_id':'r'},request=r,source_registry=registry(),sounds=sounds(),space=space(),mesh=None,clock=clock(),condition_profile=p)
    assert error.value.result['attempts']==3
    assert error.value.result['failure_histogram']=={'routes:fixture_no_path':3}
    assert seen==[p,p,p]


def test_explicit_source_count_conflict_is_rejected():
    with pytest.raises(ValueError,match='agree with total_count'):
        cs.resolve_condition_profile(request(entities={'total_count':3}),registry())


def test_rigid_pair_not_excluded_by_unrequested_min_articulated_rule():
    reg=registry()
    for r in reg['assets']:r['entity_class']='rigid_object';r.pop('timeline')
    profile=cs.resolve_condition_profile(request(),reg)
    assert profile['source_classes']==['rigid_static_object']*2


def test_unknown_gender_never_pairs_biological_human():
    actor=cs.neutral_source_declaration(registry()['assets'][0],'source1')
    assert cs.sound_matches(actor,{'sound_class':'speech','gender':'M'})
    assert not cs.sound_matches(actor,{'sound_class':'speech','gender':'F'})
    assert not cs.sound_matches(actor,{'sound_class':'speech'})
    actor['entity_class']='rigid_object'
    assert cs.sound_matches(actor,{'sound_class':'speech','gender':'F'})


def test_half_open_legal_activity_start_ranges_include_sample_zero():
    c={'sample_rate_hz':10,'frame_rate_hz':2,'sample_count':30}
    s={'sample_count':10,'audible_start_sample':2,'audible_end_sample_exclusive':8}
    ranges=cs.legal_start_ranges([True,True,False,True,True,False],s,c,{'reserve_tail_s':0})
    assert ranges==[[0,2],[13,17]]


def test_sequential_uniform_starts_keep_feasible_suffix():
    events={'a':{'actor_id':'source1','audible_start_sample':0,'audible_end_sample_exclusive':10},
            'b':{'actor_id':'source2','audible_start_sample':0,'audible_end_sample_exclusive':10}}
    starts={'a':[[0,20]],'b':[[15,15]]};p={'event_relation':'sequential','min_gap_between_audible_windows_s':0}
    values=[cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p,np.random.default_rng(i)) for i in range(60)]
    assert {r['a'] for r in values}==set(range(6))
    assert all(r['b']==15 and r['a']+10<=r['b'] for r in values)


def test_overlap_uses_declared_duration_not_merely_order():
    events={'a':{'actor_id':'source1','audible_start_sample':1,'audible_end_sample_exclusive':8},
            'b':{'actor_id':'source2','audible_start_sample':2,'audible_end_sample_exclusive':7}}
    p={'event_relation':'overlap','minimum_overlap_s':3};starts={'a':[[0,20]],'b':[[2,30]]}
    for seed in range(30):
        r=cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p,np.random.default_rng(seed))
        assert min(r['a']+8,r['b']+7)-max(r['a']+1,r['b']+2)>=3
    p['minimum_overlap_s']=6
    assert cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p) is None


def test_repeat_keeps_original_playback_and_unique_event_times():
    events={'a':{'actor_id':'source1','audible_start_sample':0,'audible_end_sample_exclusive':3},
            'b':{'actor_id':'source2','audible_start_sample':0,'audible_end_sample_exclusive':3},
            'r':{'actor_id':'source1','audible_start_sample':0,'audible_end_sample_exclusive':3,'repeat_of':'a'}}
    p={'event_relation':'repeat','min_gap_between_audible_windows_s':1};starts={k:[[0,20]] for k in events}
    for seed in range(20):
        r=cs.schedule_legal_events(events,starts,{'sample_rate_hz':1},p,np.random.default_rng(seed))
        assert r['r']>=r['a']+4
        order=sorted(r,key=r.get)
        assert all(r[b]>=r[a]+4 for a,b in zip(order,order[1:]))


def test_native_routes_preserve_points_and_never_call_raster_solver():
    raster=space();bank=[{'route_id':'a','points_m':[[0,0,0],[0,0,1],[0,0,2]]},
                        {'route_id':'b','points_m':[[2,0,0],[2,0,1],[2,0,2]]}]
    native=NativeRouteWalkableSpace(raster.pathfinder,raster.metadata,bank,15.)
    paths,record=cs._native_routes(native,[True,True],5,15.,np.random.default_rng(3))
    assert set(record['selected_route_ids'])=={'a','b'}
    for route,rid in zip(paths,record['selected_route_ids']):
        original=np.asarray(next(b['points_m'] for b in bank if b['route_id']==rid))
        assert all(any(np.array_equal(p,x) for x in original) for p in route)
        assert any(np.array_equal(route[start:start+3],original) for start in range(3))
    with pytest.raises(ValueError,match='only permits retained'):
        native.shortest_path([0,0,0],[2,0,2])


def test_full_plan_same_seed_bytes_and_camera_membership(monkeypatch):
    kwargs={'room':{'room_id':'fixture'},'request':request(),'source_registry':registry(),'sounds':sounds(),
            'space':space(),'mesh':MeshHandle(np.zeros((0,3)),np.zeros((0,3),dtype=int)),'clock':clock()}
    a=cs.build_conditioned_plan(**kwargs);b=cs.build_conditioned_plan(**kwargs)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    assert a['visual_plan']['camera']['candidate_id'] in a['planned_conditions']['legal_candidate_ids']
    assert a['condition_profile']['anchor_count']==1
    assert 'achieved_conditions' not in a
    assert all(f['camera_state']==a['visual_plan']['camera'] for f in a['visual_plan']['frames'])
    assert all('translation_ue_cm' not in s for f in a['visual_plan']['frames'] for s in f['actor_states'])
    p=dict(kwargs);p['request']=request(seed=112);c=cs.build_conditioned_plan(**p)
    assert a['audio_events']!=c['audio_events']
    assert a['visual_plan']['camera']!=c['visual_plan']['camera']
    from avengine.capture.qa_plan_adapters import materialize_ue_episode_plan
    from avengine.rooms import qa_episode
    monkeypatch.setattr(qa_episode, 'source_declaration',
                        lambda _registry, asset_id, actor_id: {'asset_id': asset_id, 'actor_id': actor_id})
    native = materialize_ue_episode_plan(a, registry())
    assert [f['camera_state']['frame_index'] for f in native['visual_plan']['frames']] == list(range(clock()['frame_count']))
    assert all('frame_index' not in f['camera_state'] for f in a['visual_plan']['frames'])
    assert native['visual_plan']['camera']['resolution_hw'] == a['visual_plan']['camera']['resolution_hw']


def test_azimuth_matches_catalog_horizontal_projection():
    from avengine.qa.unified_catalog import _listener_azimuth
    for yaw in np.linspace(-180,180,17):
        angle=np.deg2rad(yaw);basis={'forward':[np.sin(angle),.1,-np.cos(angle)],'right':[np.cos(angle),0.,np.sin(angle)],'up':[0.,1.,0.]}
        pose={'position_m':[1.,2.,3.],'basis':basis};listener={'positions_m':[[1.,2.,3.]],'basis_m3':[basis]}
        for p in [[2,1,4],[0,0,0],[-2,8,4]]:
            assert listener_azimuth_deg(pose,p)==_listener_azimuth(p,listener,0)


def test_concurrency_sweep_deduplicates_entities_and_touching_boundaries():
    assert max_concurrent_entities([(0,10,'a'),(1,2,'b'),(8,9,'c')])==2
    assert max_concurrent_entities([(0,3,'a'),(1,5,'a'),(5,6,'b')])==1


def test_separation_tracks_nearest_switch_and_sustained_seconds():
    r=separation_stats([0]*4,{'offscreen':[30,30,60,60],'visible':[50,50,20,20]},[0,4],frame_rate_hz=2,thresholds_deg=[25])
    assert r['min']==20 and r['max']==30 and r['nearest_competitor_changed']
    assert r['sustained_s_above']['25.0']==1.


def test_los_keeps_missing_geometry_unmeasured_and_exact_triangle_block():
    mesh=MeshHandle([[0,0,0],[0,2,0],[0,0,2]],[[0,1,2]])
    assert line_of_sight(None,[-1,.5,.5],[1,.5,.5])=='unmeasured'
    assert line_of_sight(mesh,[-1,.5,.5],[1,.5,.5])=='blocked'
    assert line_of_sight(mesh,[-1,3,3],[1,3,3])=='clear'


def test_structure_reports_majority_shortcut_without_refusing():
    r=structural_baselines({'a':'blue','b':'blue','c':'red'},'c')
    assert r['unique_minority_hits']==1 and r['majority_hits']==0 and r['random_hits']==pytest.approx(1/3)
    assert structural_baselines({'a':None,'b':1},'b')['status']=='unmeasured'
