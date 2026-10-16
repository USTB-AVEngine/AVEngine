"""Speech construction counterexamples; original QA direction code is authority."""
from copy import deepcopy
import itertools
import math
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.timeline.v7_speech_construction import full_clip_ticks, schedule_epochs
from avengine.construction.joint.families.speech import direction, SpeechPlugin
from avengine.construction.joint import BodySpec, CameraPose, FamilyRequest

pytestmark = pytest.mark.fast_unit


def sound(sid, seconds, rate=16000):
    return {'sound_asset_id': sid, 'scenario_id': 'scenario_'+sid.split('_')[0],
            'role_id': sid.split('_')[-1],
            'synthesis_readback': {'sample_count': round(seconds*rate), 'sample_rate_hz': rate}}


@pytest.mark.parametrize('lengths', [(1., 1.7), (2.49, 2.49), (2.51, 2.51), (4.9, .1), (6.2, 7.3), (10., 9.9)])
def test_whole_waveforms_have_shared_full_windows(lengths):
    epochs = [[sound(str(i)+'_'+str(j), d) for j, d in enumerate(lengths)] for i in range(3)]
    planned = schedule_epochs(epochs, ['a', 'b'])
    for i, pair in enumerate(epochs):
        events = planned[i*2:i*2+2]
        assert set(events[0]['complete_window_starts_s']) & set(events[1]['complete_window_starts_s'])
        for e, s in zip(events, pair):
            assert e['end_tick']-e['start_tick'] == math.ceil(s['synthesis_readback']['sample_count']/1600)
            assert 0 <= e['start_tick'] < e['end_tick'] <= 300
            assert all(w*10 <= e['start_tick'] and e['end_tick'] <= (w+10)*10
                       for w in e['complete_window_starts_s'])


def test_short_pairs_are_never_partial_in_any_intersecting_crop():
    epochs = [[sound(str(i)+'_a', 1.01), sound(str(i)+'_b', 2.2)] for i in range(3)]
    for e in schedule_epochs(epochs, ['a', 'b']):
        lo, hi = e['start_tick']/10, e['end_tick']/10
        for window in (0, 5, 10, 15, 20):
            if lo < window+10 and hi > window:
                assert window <= lo < hi <= window+10


def test_clock_ceil_preserves_last_sample():
    s = sound('a_0', 1.0000625)
    assert full_clip_ticks(s) == 11
    assert schedule_epochs([[s, sound('a_1', .5)]], ['a', 'b'])[0]['source_sample_count'] == 16001


def test_overlong_clip_cannot_be_relabelled_complete():
    with pytest.raises(ValueError, match='10s'):
        schedule_epochs([[sound('a_0', 10.01), sound('a_1', .5)]], ['a', 'b'])


@pytest.mark.parametrize('yaw', [0., 90., -120., 179.])
def test_direction_calls_original_QA_convention(yaw):
    camera = CameraPose((1, 1.2, 2), yaw)
    sector, _ = direction(camera, np.asarray(camera.position_m)+4*camera.forward+3*camera.right)
    assert sector == 'front-right'
    assert direction(camera, np.asarray(camera.position_m)+4*camera.forward)[0] == 'front'


def test_dead_zone_and_distinct_sectors_keep_original_answer_rule():
    camera = CameraPose((0, 1.2, 0), 0)
    p = lambda angle: [4*math.sin(math.radians(angle)), 1.6, -4*math.cos(math.radians(angle))]
    assert direction(camera, p(22.5))[0] is None
    assert direction(camera, p(28))[0] == 'front-right'
    assert direction(camera, p(-28))[0] == 'front-left'


def test_joint_roles_visit_second_camera_and_keep_distinct_QA_answers():
    from test_joint_core import context
    c = context()
    c.cameras = (CameraPose((0, 1.25, .5), 180), CameraPose((0, 1.25, .5), 0))
    assets = [{'asset_id': aid, 'kind': 'human', 'emitter_local_m': [0, 1.5, 0]} for aid in ('a', 'b')]
    schedule = schedule_epochs([[sound('s_0', 1), sound('s_1', 1)]], ['a', 'b'])
    plugin = SpeechPlugin(assets, {aid: BodySpec() for aid in ('a', 'b')}, schedule, ())
    request = FamilyRequest('QA-28', 'distinct_complete_speech', ('a', 'b'))
    p = next(plugin.candidates(c, request))
    assert p.camera.yaw_deg == 0
    assert len(set(p.metadata['requested_speech_sectors'].values())) == 2
    assert all(check.status == 'pass' for check in plugin.verify(c, request, p))
    assert np.linalg.norm((p.actors['a'].positions_m[0]-p.actors['b'].positions_m[0])[[0, 2]]) >= 1


def test_epoch_selection_preserves_physical_voices_and_real_capabilities():
    from avengine.timeline.v7_speech_construction import choose_speech_epochs
    people = [{'asset_id': a, 'visual_gender': 'female', 'kind': 'human'} for a in ('a', 'b')]
    sounds, scenarios = [], []
    for epoch in range(3):
        scenario = 'scenario_'+str(epoch)
        roles = []
        for role in range(3):
            s = sound(str(epoch)+'_'+str(role), 3.2)
            s.update(scenario_id=scenario, transcript=f'Original words {epoch} {role}.',
                semantic_answer={'answer': str(role)},
                voice_reference={'speaker_id': 'voice_'+str(role), 'gender': 'female'})
            sounds.append(s)
            roles.append({'role_id': s['role_id'], 'required_asset_capability':
                          'megaphone_speaker' if role == 2 else 'standing_human'})
        scenarios.append({'id': scenario, 'roles': roles})
    manifest = {'source_registry': 'fixture', 'dialogues': {'scenarios': scenarios}, 'sounds': sounds}
    epochs = choose_speech_epochs(manifest, people, {}, 10)
    assert len({group[0]['scenario_id'] for group in epochs}) == 3
    assert len({tuple(s['voice_reference']['speaker_id'] for s in group) for group in epochs}) == 1
    assert all(s['role_id'] != '2' for group in epochs for s in group)
    assert all(s['synthesis_readback']['sample_count'] == 51200 for group in epochs for s in group)


def test_compiler_binds_each_epoch_own_transcript_without_crop(monkeypatch):
    import avengine.timeline.v7_phase2_sources as phase2
    import avengine.timeline.v7_registered_sampling as registered
    from avengine.timeline.v7_speech_construction import build_speech_scene
    assets = [{'asset_id': a, 'endpoint_id': ep, 'visual_gender': 'female'}
              for a, ep in [('a', 'voice1'), ('b', 'voice2')]]
    sources = {a['endpoint_id']: {'actor_id': i+1, 'class_id': 0, 'mechanism': 'human_live_recorded'}
               for i, a in enumerate(assets)}
    monkeypatch.setattr(phase2, 'build_phase2_scene', lambda *args: (
        {}, {'sources': sources}, {'schema': 'original_program'}, assets))
    monkeypatch.setattr(registered, 'semantic_record', lambda s, gender: deepcopy(s))
    epochs = [[{**sound(str(i)+'_'+str(j), 1.1), 'transcript': f'Exact epoch {i} role {j}.'}
               for j in range(2)] for i in range(3)]
    schedule = schedule_epochs(epochs, ['a', 'b'])
    row = {'speech_schedule': schedule, 'automatic_construction': {}, 'independent_parent_scene_id': 'fixture'}
    _, scene, program, _ = build_speech_scene(row, {}, [], [], {'sounds': sum(epochs, [])})
    originals = {s['sound_asset_id']: s['transcript'] for group in epochs for s in group}
    actual = {s['sound_asset_id']: s['transcript'] for s in program['records']}
    assert actual == originals and len(actual) == len(scene['events']) == 6
    assert all(b['crop_start_sample'] == 0 for b in program['events'])
    bindings = {b['event_id']: b['sound_asset_id'] for b in program['events']}
    assert all(bindings[e['event_id']] == e['sound_asset_id'] for e in schedule)


def test_room_preferred_listener_height_is_not_discarded_for_old_crop():
    from avengine.construction.joint.families.speech import speech_camera_heights
    room = {'measured_floor_y_m': .04931496332089106,
            'preferred_listener_m': [1.125, 1.2993149633208911, 2.625],
            'placement': {'camera_m': [1.125, 1.705729067325592, 2.625]}}
    heights = speech_camera_heights(room)
    assert heights[0] == pytest.approx(1.25)
    assert heights[1] == pytest.approx(1.656414104004701)
    assert speech_camera_heights({**room, 'placement': {'camera_m': room['preferred_listener_m']}}) == heights[:1]
