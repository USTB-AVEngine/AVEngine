from copy import deepcopy
import numpy as np
import pytest
from avengine.timeline.v7_scene_profile import load_profile_config,add_profile_requests,profile_request_summary,validate_profile_row,_distance_cdf
from avengine.timeline.v7_candidate_generation import seeded_scene
pytestmark=pytest.mark.fast_unit

def rows(count):
    return [{'schema':'v7-production-inputs-20261003-v3','scene_index':i,'scene_id':f'p{i}','scene_seed':i+777,'house':'h','split':'train','duration_s':30,
        'window_start_ticks':[40,160],'window_count':2,'seeds':{'camera':i+11,'pose':i+23,'trajectory':i+37,'event_schedule':i+41,'appearance':i+47}} for i in range(count)]

def test_prefix_shard_order_and_independent_appearance():
    source=rows(400);small=add_profile_requests(source);full=add_profile_requests(rows(500))
    assert full[:400]==small and source==rows(400)
    assert add_profile_requests(list(reversed(source)))==list(reversed(small))
    changed=deepcopy(source);changed[17]['seeds']['appearance']+=10000
    assert add_profile_requests(changed)[17]['scene_profile_request']==small[17]['scene_profile_request']

def test_coverage_zero_to_eight_and_silence_is_not_absence():
    result=add_profile_requests(rows(400));coverage=[r for r in result if r['scene_profile_request']['branch']=='coverage']
    assert len(coverage)==40
    assert {r['scene_quotas']['speakers'] for r in coverage}==set(range(9))
    assert {r['scene_profile_request']['visible_people_360_band_target'] for r in coverage}==set(range(9))
    for row in coverage:
        request=validate_profile_row(row)
        if request['speech_actor_pool_count']==0:
            assert request['visible_people_360_band_target']>=1 and not request['speech_activity_requests']
            assert row['scene_quotas']['starss13_focus_class'] not in (0,1)

def test_aggregate_distance_matches_bins_and_reported_quantiles():
    config=load_profile_config();p,d=_distance_cdf(config)
    for q,value in config['speech_distance_m']['quantiles'].items():assert np.interp(float(q),p,d)==pytest.approx(value)
    cumulative=np.cumsum(config['speech_distance_m']['histogram_proportions'])
    for edge,prob in zip(config['speech_distance_m']['histogram_edges'][1:6],cumulative[:5]):assert np.interp(edge,d,p)==pytest.approx(prob)

def test_sampler_distribution_and_direction_scope():
    sampled=add_profile_requests(rows(1500));summary=profile_request_summary(sampled);g=summary['branches']['aggregate_profile'];config=load_profile_config()
    for key,expected in config['people_count']['distribution'].items():assert abs(g['visible_people_360'].get(key,0)/g['parents']-expected)<.04
    for key in ('0','1','2'):assert abs(g['concurrent_speech_fraction'][key]-config['concurrent_speakers']['distribution'][key])<.025
    assert abs(g['distance_quantiles_m']['0.5']-1.87)<.08
    assert abs(g['horizontal_speech_out_fraction']-(1-85/360))<.04
    assert all(g['azimuth_octants'].get(str(i),0)>0 for i in range(8))
    for branch in ('resolved_proxy','unknown_independent'):assert all(summary['posture_by_branch'][branch][p]>0 for p in ('sitting','standing','walking','crouching'))

def test_invalid_requests_and_no_native_or_alpha_fallback():
    row=add_profile_requests(rows(1))[0]
    for schema in ['v7-production-inputs-20261004-v5','v7-S-alpha-static-recorded-v1']:
        incompatible=rows(1);incompatible[0]['schema']=schema
        with pytest.raises(ValueError,match='cannot be silently replaced'):add_profile_requests(incompatible)
    with pytest.raises(RuntimeError,match='frozen alpha/two-person fallback'):seeded_scene(row,{})
    bad=deepcopy(row);bad['scene_profile_request']['speech_activity_requests']=[{'actor_id':9,'start_tick':0,'end_tick':20}]
    with pytest.raises(ValueError,match='invalid/overlapping'):validate_profile_row(bad)
    bad=deepcopy(row);bad['scene_profile_request']['formal_media_qualified_questions']=1
    with pytest.raises(ValueError,match='not native truth'):validate_profile_row(bad)
