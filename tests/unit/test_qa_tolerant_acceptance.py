from copy import deepcopy
from avengine.qa.unified_catalog import motion_state_during_audible_window, _tag_question_tolerance

POLICY={"policy_id":"test_noticeable_v1","motion":{"mode":"noticeable_motion",
    "min_moving_duration_s":0.8,"min_travel_m":0.2,"max_still_travel_m":0.05}}


def make_case(positions, enabled=True):
    facts={"sampling":{"acceptance_policy":deepcopy(POLICY)} if enabled else {},
           "time":{"frame_count":len(positions),"frame_rate_hz":10,"sample_rate_hz":16000},
           "actors":{"a":{"root_positions_m":positions,
                          "moving":[i<9 for i in range(len(positions))]}}}
    event={"event_id":"e","actor_id":"a","start_frame":0,"end_frame":len(positions)}
    return facts,event


def test_noticeable_motion_allows_a_natural_stop_and_uses_positions():
    positions=[[min(i,8)*0.1,0,0] for i in range(30)]
    facts,event=make_case(positions)
    result=motion_state_during_audible_window(facts,event)
    assert result["moving"] is True
    assert result["measurement"]["travel_m"]>=0.79
    assert result["measurement"]["moving_duration_s"]==0.8
    facts["actors"]["a"]["moving"]=[False]*30
    assert motion_state_during_audible_window(facts,event)["moving"] is True
    facts["sampling"]={}
    assert motion_state_during_audible_window(facts,event)["moving"] is False


def test_strict_default_still_refuses_a_changing_state():
    facts,event=make_case([[min(i,8)*0.1,0,0] for i in range(30)],enabled=False)
    assert motion_state_during_audible_window(facts,event)["moving"] is None


def test_small_jitter_still_and_ambiguous_travel_deferred():
    facts,event=make_case([[i*.001,0,0] for i in range(30)])
    assert motion_state_during_audible_window(facts,event)["moving"] is False
    facts["actors"]["a"]["root_positions_m"]=[[min(i,8)*.015,0,0] for i in range(30)]
    assert motion_state_during_audible_window(facts,event)["moving"] is None


def test_tolerant_ids_and_source_are_distinct_from_old_gold():
    facts,_=make_case([[0,0,0],[0,0,0]])
    item={"question_id":"old_id","evidence":{},"truth":{"value":"moving"}}
    result=_tag_question_tolerance(item,facts,"QA-06")
    assert result["question_id"]=="old_id__policy_test_noticeable_v1"
    assert result["truth"]["source"]=="native_measurements_with_configured_question_tolerance"
    assert result["truth"]["evidence"]["acceptance_policy"]==POLICY
