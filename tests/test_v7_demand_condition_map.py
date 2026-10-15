import pytest
from avengine.timeline.v7_demand_condition_map import requested_families
from avengine.timeline.v7_demand import FAMILY_SCENE_REQUIREMENTS
pytestmark=pytest.mark.fast_unit
def test_named_post_sound_requirements_resolve_without_losing_the_requested_family():
    demand={'condition_families':{},'family_scene_requirements':FAMILY_SCENE_REQUIREMENTS,
        'requested_conditions':['complete_native_wet_support','post_stop_moving_and_static_alternatives'],
        'required_independent_family_counts':{'QA-16':3}}
    assert {'QA-13','QA-16','QA-17'}<=requested_families(demand)
def test_descriptive_layout_notes_preserve_explicit_qa10_target_and_coarse_mapping():
    demand={'condition_families':{'authored':['QA-26']},
        'requested_conditions':['authored','two_measured_partial_occluders'],
        'required_independent_family_counts':{'QA-10':1}}
    assert requested_families(demand)=={'QA-10','QA-26'}
