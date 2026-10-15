"""Claude 10-15: foot emitters only for a walker's own footstep sources."""
import pytest


def test_speaking_walker_has_no_foot_emitter_to_place():
    mod = pytest.importorskip('avengine.capture.v7_phase2_native')
    needs = mod.NativePhase2Video._phase2_needs_foot_emitters
    fake = type('Fake', (), {})()
    fake._phase2_scene = {'sources': {
        'voice1': {'asset_id': 'speech'},
        'voice2_foot_L': {'owner_endpoint_id': 'voice2', 'emitter_part': 'foot_L'},
        'voice2_foot_R': {'owner_endpoint_id': 'voice2', 'emitter_part': 'foot_R'}}}
    assert needs(fake, {'endpoint_id': 'voice1'}) is False
    assert needs(fake, {'endpoint_id': 'voice2'}) is True
    del fake._phase2_scene
    assert needs(fake, {'endpoint_id': 'voice1'}) is True        # unknown scene: unchanged behaviour
