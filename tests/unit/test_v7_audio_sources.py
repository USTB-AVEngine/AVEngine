"""The production source route cannot silently return to robotic dry inputs."""
from copy import deepcopy
import numpy as np
import pytest
import soundfile as sf
from avengine.timeline.v7_audio_sources import (
    AUDIO_SOURCE_REVISION, assemble_recorded_buses, filter_admission,
    rebind_receipt, source_blockers, validate_dry_receipt,
)

pytestmark = pytest.mark.fast_unit


def record(path):
    return {"sound_asset_id": "recorded_clip", "path": str(path),
            "source_origin": "public_library/original_42.wav", "source_type": "recorded",
            "strict_v7_eligible": True, "formal_status": "admitted",
            "screens": {"license": {"status": "pass", "original_id": "42", "license": "CC0-1.0",
                                    "source_uri": "https://example.org/original/42"},
                        "provenance": {"status": "pass"}, "deduplication": {"status": "pass", "nearest": {"dataset": "STARSS23"}},
                        "automatic_label": {"status": "pass"}}}


def fixture(tmp_path):
    path = tmp_path / 'hermetic_fixture.wav'
    sf.write(path, np.linspace(-.2, .2, 16000), 16000, subtype='FLOAT')
    r = record(path)
    scene = {'duration_s': 2., 'sample_rate_hz': 16000,
             'sources': {'voice1': {'dry_gain': .8}},
             'events': [{'event_id': 'e0', 'endpoint_id': 'voice1', 'start_tick': 2, 'end_tick': 7}]}
    program = {'schema': 'v7-recorded-dry-program-v1', 'records': [r],
               'events': [{'event_id': 'e0', 'sound_asset_id': 'recorded_clip', 'crop_start_sample': 1600}]}
    return scene, program


def test_recorded_waveform_timing_crop_and_provenance(tmp_path):
    scene, program = fixture(tmp_path)
    dry, receipt = assemble_recorded_buses(scene, program)
    original, _ = sf.read(program['records'][0]['path'])
    assert np.array_equal(dry['voice1'][3280:11120], original[1680:9520] * .8)
    assert not np.any(dry['voice1'][:3200]) and not np.any(dry['voice1'][11200:])
    assert receipt['events'][0]['original_id'] == '42'
    assert receipt['events'][0]['repeat_or_time_stretch'] is False
    validate_dry_receipt(receipt)


def test_espeak_admitted_upstream_is_still_excluded(tmp_path):
    r = record(tmp_path / 'tts.wav')
    r.update(source_type='author_tts', synthesis_readback={'engine': {'engine': 'eSpeak NG formant'}})
    admitted, rejected = filter_admission([r])
    assert not admitted and 'robotic_or_procedural_source_forbidden' in rejected[0]['reasons']


@pytest.mark.parametrize('origin', ['STARSS23/foo.wav', 'DCASE/foo.wav', 'TAU/foo.wav', 'SAVVY/foo.wav'])
def test_eval_source_is_rejected_but_dedup_reference_is_allowed(tmp_path, origin):
    r = record(tmp_path / 'clip.wav')
    assert source_blockers(r) == []
    r['source_origin'] = origin
    assert any(x.startswith('excluded_source_') for x in source_blockers(r))


def test_short_clip_and_missing_program_fail_without_synthesis(tmp_path):
    scene, program = fixture(tmp_path)
    with pytest.raises(ValueError, match='registered recorded_dry_program'):
        assemble_recorded_buses(scene, None)
    scene['events'][0]['end_tick'] = 15
    with pytest.raises(ValueError, match='too short'):
        assemble_recorded_buses(scene, program)


def test_vctk_release_license_does_not_promote_unadmitted_clips(tmp_path):
    r = record(tmp_path / 'clip.wav'); r['strict_v7_eligible'] = False
    assert 'upstream_automatic_admission_missing' in source_blockers(r)


def test_natural_tts_requires_actual_model_reference_and_output_terms(tmp_path):
    r = record(tmp_path / 'clip.wav')
    r.update(source_type='author_tts', script_author='project_author', transcript='Please bring a cup.')
    assert 'natural_tts_not_verified' in source_blockers(r)
    component = {'source_uri': 'https://example.org/model', 'revision': 'actual_revision', 'license': 'Apache-2.0'}
    r['tts_provenance'] = {'natural_voice': True, 'mode': 'unconditioned', 'code': component,
                           'weights': component, 'voice_id': 'voice_A',
                           'output_terms': {'usage_allowed': True, 'source_uri': 'https://example.org/terms', 'assessment': 'recorded'}}
    assert source_blockers(r) == []


def test_robotic_receipt_cannot_reenter_audio_stage(tmp_path):
    from avengine.timeline.v7_candidate_generation import audio_stage
    def forbidden_renderer(*args):
        raise AssertionError('source rejection must precede native rendering')
    with pytest.raises(ValueError, match='recorded dry receipt'):
        audio_stage({}, {}, {}, tmp_path/'cache', tmp_path, forbidden_renderer,
                    dry_override={'voice1': np.zeros(16000)}, dry_receipt={'engine': 'eSpeak'})


def test_counterfactual_receipt_swaps_only_endpoint_binding(tmp_path):
    scene, program = fixture(tmp_path)
    _, receipt = assemble_recorded_buses(scene, program)
    moved = rebind_receipt(receipt, {'voice1': 'voice2', 'voice2': 'voice1'})
    assert moved['events'][0]['endpoint_id'] == 'voice2'
    assert receipt['events'][0]['endpoint_id'] == 'voice1'
    assert moved['events'][0]['original_id'] == receipt['events'][0]['original_id']



def test_natural_tts_cannot_use_an_evaluation_recording_as_reference(tmp_path):
    r = record(tmp_path / 'clip.wav')
    r['tts_provenance'] = {'reference_audio': {'source_dataset': 'STARSS23', 'original_id': 'room23'}}
    assert 'excluded_source_STARSS23' in source_blockers(r)


def test_speech_source_cannot_take_an_appliance_clip_or_wrong_voice_label(tmp_path):
    scene, program = fixture(tmp_path)
    scene['sources']['voice1'].update(content_kind='speech', class_id=1)
    program['records'][0]['sound_class'] = 'alarm_beep'
    with pytest.raises(ValueError, match='speech dry clip'):
        assemble_recorded_buses(scene, program)
    program['records'][0].update(sound_class='speech_playback', gender='F')
    with pytest.raises(ValueError, match='matching voice metadata'):
        assemble_recorded_buses(scene, program)
    program['records'][0]['gender'] = 'M'
    dry, _ = assemble_recorded_buses(scene, program)
    assert np.any(dry['voice1'])


def test_quiet_world_needs_no_synthesized_placeholder():
    scene = {'duration_s': 1., 'sample_rate_hz': 16000, 'sources': {'silent_actor': {}}, 'events': []}
    program = {'schema': 'v7-recorded-dry-program-v1', 'records': [], 'events': []}
    dry, receipt = assemble_recorded_buses(scene, program)
    assert not np.any(dry['silent_actor'])
    validate_dry_receipt(receipt)
