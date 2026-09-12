from avengine.rooms.conditioned_visibility import VisibilityRequirement, _judge_requirement


def judge(states, allow=False):
    series=[{"frame_index":i,"state":s} for i,s in enumerate(states)]
    policy={"accept_observed_branches":{"QA-09":["yes","no"]}} if allow else {}
    facts={"time":{"frame_count":len(states),"frame_rate_hz":15},
           "visibility":{"source1":{r["frame_index"]:r for r in series}},
           "sampling":{"acceptance_policy":policy}}
    requirement=VisibilityRequirement(kind="fully_occluded_without_return",
        subject="target",qa_id="QA-09",require_complete_coverage=False)
    return _judge_requirement(requirement,instance="source1",series=series,facts_view=facts,
        frame_count=len(states),occluder_registry=None,observed=sorted(set(states)))


def test_explicit_type_only_acceptance_keeps_the_measured_yes_answer():
    states=["visible_clear","fully_occluded","visible_clear"]
    assert judge(states)["status"]=="fail"
    row=judge(states,allow=True)
    assert row["status"]=="pass"
    assert row["measured"]["observed_answer"]=="yes"
    assert row["measured"]["planned_answer"]=="no"


def test_no_occlusion_is_still_a_failure():
    assert judge(["visible_clear"]*3,allow=True)["status"]=="fail"


def test_matching_no_answer_remains_valid():
    row=judge(["visible_clear","fully_occluded","fully_occluded"],allow=True)
    assert row["status"]=="pass"
    assert row["measured"]["observed_answer"]=="no"
