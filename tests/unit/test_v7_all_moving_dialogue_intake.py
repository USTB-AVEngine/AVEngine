import pytest
from avengine.timeline.v7_registered_semantics import dialogue_evidence
from avengine.timeline.v7_visible_semantics import visible_dialogue_evidence
pytestmark=pytest.mark.fast_unit


def test_empty_standing_subset_does_not_qualify_a_walking_person_as_standing(monkeypatch):
    monkeypatch.setattr('avengine.timeline.v7_registered_dialogue_scope.registered_dialogue_view',lambda scene,visual:(visual,{}))
    monkeypatch.setattr('avengine.timeline.v7_registry_snapshot.resolve_registry',lambda *args:pytest.fail('empty standing subset reached static namespace'))
    visual={'frames':[{'actors':[{'actor_id':1,'asset_id':'moving_person','action_id':'walk','body_to_world_m':[],'loaded_artifact_sha256':'actual'}]}]}
    result=dialogue_evidence({}, {}, {}, visual, {}, 0., 'actual_native.json')
    assert result['native_dialogue_deferred']['standing_proof_fabricated']is False
    assert result['moving_TTS_content_binding_deferred_asset_ids']==['moving_person']
    assert 'native_dialogue_evidence'not in result


def test_visible_wrapper_preserves_the_explicit_deferred_scope(monkeypatch):
    expected={'native_dialogue_deferred':{'status':'not_applicable_no_static_authored_actor'}}
    monkeypatch.setattr('avengine.timeline.v7_visible_semantics.dialogue_evidence',lambda *args:expected)
    result=visible_dialogue_evidence({'events':[{'event_id':'e','source_id':1}]},{'records':[{'sound_asset_id':'s','source_type':'author_tts'}]},{'events':[{'event_id':'e','sound_asset_id':'s'}]},{'frames':[{'actors':[{'actor_id':1}]}]}, {},0.,'native.json')
    assert result is expected
