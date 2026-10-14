"""Hermetic QA sampling regressions, never substitutes for a native canary."""
from copy import deepcopy
from collections import Counter

import pytest

from avengine.qa import unified_catalog as authority
from avengine.qa.v7_families import export_items,deduplicate_questions
from avengine.timeline.v7_full_question_delivery import sample_pilot_comparison_frame


PILOT={'production_scenario':{'camera_policy':'pilot_binding_coverage'}}


def retained_canary_distance_facts():
    """Compact actual repair3 v0_a0 w00 mouth/visibility facts; no new geometry."""
    count=150
    listener=[-2.125,1.5115453898906708,.125]
    mouths=[[-3.699911608441282,1.5572681443907723,1.1422432707896855],
            [-4.2060901424505746,1.5572681443907723,.13797684067541557],
            [-4.702317191407296,1.5572681443907723,1.141171387115542]]
    roots=[[-3.695676326751709,.0015454313158988953,1.1295028924942017],
           [-4.195676326751709,.0015454313158988953,.12950292229652405],
           [-4.695676326751709,.0015454313158988953,1.1295028924942017]]
    intervals=[[(0,116,'visible_clear'),(116,142,'out_of_view'),(142,150,'visible_clear')],
               [(0,120,'visible_clear'),(120,146,'out_of_view'),(146,150,'visible_clear')],
               [(0,116,'visible_occluded'),(116,117,'visible_clear'),(117,144,'out_of_view'),(144,150,'visible_occluded')]]
    actors={};review={};visibility={};masks={}
    for k,color,mouth,root,runs in zip(['1','2','3'],['blue','yellow','green'],mouths,roots,intervals):
        actors[k]={'actor_id':k,'asset_id':'asset_'+k,'species_id':'human','display_label':color+' actor',
                   'appearance':{'field':'top_color','value':color,'label':color+' actor'},
                   'root_positions_m':[root[:] for _ in range(count)],
                   'emitter_positions_m':[mouth[:] for _ in range(count)],
                   'speed_mps':[0.]*count,'moving':[False]*count}
        review[k]={'status':'reviewed','value':color,'frame_refs':[6,21,36,51,66,81,96,111],
                   'review_method':'automatic_native_color_pixels'}
        visibility[k]={i:{'frame_index':i,'state':state} for start,end,state in runs for i in range(start,end)}
        masks[k]={'frames':[dict(visibility[k][i]) for i in range(count)]}
    return {'schema':authority.UNIFIED_FACT_SCHEMA,'episode_id':'retained_canary_distance_slice',
            'time':{'frame_count':count,'frame_rate_hz':15,'sample_rate_hz':16000,
                    'sample_count':160000,'duration_seconds':10.},
            'actors':actors,'events':[],'listener':{'status':'pass','positions_m':[listener[:] for _ in range(count)],
                                                'yaw_deg':[0.]*count},
            'sampling':{},'appearance_review':review,'visibility':visibility,
            'visibility_meta':{'source':'computed_modal_target_only_v1'},
            'pixel_visibility_truth':{'status':'computed_modal_target_only_v1','per_instance':masks}}


def separated_distance_fixture():
    """Hermetic sampling fixture with three distance layers, not native evidence."""
    raw=retained_canary_distance_facts();listener=raw['listener']['positions_m'][0]
    for actor,distance in zip(['1','2','3'],[1.0,1.7,2.4]):
        raw['actors'][actor]['emitter_positions_m']=[[listener[0]+distance,listener[1],listener[2]] for _ in range(150)]
    return raw


def distinct_distance_items(raw):
    result=authority.generate_unified_questions(raw,qa_ids=['QA-14'],seed='retained-distance-review',
                                              items_per_type=5,include_angle_followups=False)
    public,private=export_items(result['items'],'test_window',{})
    public,_,duplicates=deduplicate_questions(public,private)
    return [q for q in public if q['form']=='open'],duplicates


def test_retained_repair3_has_two_actual_distance_pairs_and_remains_blocked():
    raw=retained_canary_distance_facts();before=deepcopy(raw)
    pairs=Counter(tuple(x['actor_ids']) for x in authority._P8_CANDIDATES['QA-14'](raw))
    assert pairs=={('1','3'):122,('2','3'):121}
    assert authority._distance_at(raw,'2',30)-authority._distance_at(raw,'1',30)==pytest.approx(.2062072524141476)
    questions,duplicates=distinct_distance_items(raw)
    assert len(questions)==2 and duplicates
    audit=sample_pilot_comparison_frame(raw,PILOT)
    assert audit['status']=='blocked'
    assert audit['reason']=='no_three_native_distance_pairs_in_a_publishable_common_run'
    assert raw==before


def test_three_real_comparisons_use_one_frame_and_do_not_spend_budget_on_duplicates():
    raw=separated_distance_fixture();before=deepcopy(raw)
    audit=sample_pilot_comparison_frame(raw,PILOT)
    assert audit['status']=='pass'
    assert audit['common_run_frames']==[0,116]
    assert audit['common_run_public_s']==[0.,7.]
    assert raw['sampling']['query_frame_by_qa']['QA-14']==audit['query_frame']
    candidates=authority._P8_CANDIDATES['QA-14'](raw)
    assert len(candidates)==3
    assert {tuple(x['actor_ids']) for x in candidates}=={('1','2'),('1','3'),('2','3')}
    questions,duplicates=distinct_distance_items(raw)
    assert len(questions)==3
    assert not duplicates
    assert all(p['authority_status']=='pass' for p in audit['authority_comparison_proofs'])
    assert all(abs(list(p['distances_m'].values())[0]-list(p['distances_m'].values())[1])>=.5
               for p in audit['authority_comparison_proofs'])
    del raw['sampling']['query_frame_by_qa'];assert raw==before


def test_nonpilot_sampling_and_all_native_readbacks_are_unchanged():
    raw=separated_distance_fixture();before=deepcopy(raw)
    audit=sample_pilot_comparison_frame(raw,{'production_scenario':{'camera_policy':'uniform_static'}})
    assert audit['status']=='not_applicable'
    assert raw==before


def test_short_common_visibility_cannot_invent_a_publishable_second():
    raw=separated_distance_fixture()
    for actor in ['1','2','3']:
        for frame in range(8,150):
            raw['visibility'][actor][frame]['state']='out_of_view'
            raw['pixel_visibility_truth']['per_instance'][actor]['frames'][frame]['state']='out_of_view'
    before=deepcopy(raw)
    audit=sample_pilot_comparison_frame(raw,PILOT)
    assert audit['status']=='blocked'
    assert audit['publishable_common_run_count']==0
    assert raw==before


@pytest.mark.parametrize('missing',['mouth_track','listener_frame','native_mask_frame'])
def test_missing_native_evidence_does_not_fallback_to_root_or_sparse_masks(missing):
    raw=separated_distance_fixture()
    if missing=='mouth_track':del raw['actors']['2']['emitter_positions_m']
    elif missing=='listener_frame':raw['listener']['positions_m'].pop()
    else:raw['pixel_visibility_truth']['per_instance']['2']['frames'].pop()
    before=deepcopy(raw);audit=sample_pilot_comparison_frame(raw,PILOT)
    assert audit['status']=='blocked'
    assert raw==before
