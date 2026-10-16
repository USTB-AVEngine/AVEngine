import pytest
from avengine.timeline.v7_audit_actor_kinds import actor_kind
pytestmark=pytest.mark.fast_unit


def test_real_silent_or_offscreen_human_is_not_lost_with_a_numeric_label():
    assert actor_kind({'species_id':'human','display_label':'3'})=='human'


def test_native_species_overrides_a_misleading_display_name():
    assert actor_kind({'species_id':'dog','display_label':'the adult man'})=='dog'
    assert actor_kind({'species_id':'object','display_label':'speaker'})=='other'


def test_explicit_sliding_animal_keeps_the_owner_slide_speed_band():
    assert actor_kind({'species_id':'cat','source':{'motion_mode':'slide_no_animation'}})=='slide'
