import pytest
from avengine.timeline.v7_acceptance_motion_planner import choose_motion_family,prepare_conditioned_motion_gap
pytestmark=pytest.mark.fast_unit

def state(counts):return {'family_counts':counts,'logical_questions':sum(counts.values())}

def test_lowest_independent_motion_gap_is_planned_before_a_covered_family():
    counts={'QA-06':3,'QA-15':3,'QA-16':0,'QA-17':3,'QA-13':20}
    assert choose_motion_family(state(counts),7)=='QA-16'
    counts['QA-16']=3;counts['QA-17']=0
    assert choose_motion_family(state(counts),7)=='QA-17'

def test_equal_missing_families_get_deterministic_scene_diversity():
    a={choose_motion_family(state({}),seed)for seed in range(100)}
    assert a=={'QA-06','QA-15','QA-16','QA-17','QA-13'}
    assert choose_motion_family(state({}),17)==choose_motion_family(state({}),17)

def test_retained_solver_plan_is_not_rewritten_on_resume():
    row={'conditioned_post_motion':{'qa_id':'QA-16'}}
    assert prepare_conditioned_motion_gap(row,{},[],[],{}, {},{})is row
