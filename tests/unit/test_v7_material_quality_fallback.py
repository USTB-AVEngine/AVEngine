import pytest
from avengine.timeline.v7_material_quality_fallback import same_object_material_class
pytestmark=pytest.mark.fast_unit

def test_material_prior_requires_actual_object_function_and_attachment_class():
    target={'kind':'object','object_type':'keypad','attachment_surface':'wall','allowed_classes':['alarm_beep']}
    peer={'kind':'object','object_type':'co_alarm','attachment_surface':'wall','allowed_classes':['alarm_beep']}
    assert same_object_material_class(target,peer)
    assert not same_object_material_class(target,dict(peer,kind='human'))
    assert not same_object_material_class(target,dict(peer,attachment_surface='ceiling'))
    assert not same_object_material_class(target,dict(peer,allowed_classes=['music_playback']))
