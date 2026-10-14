from copy import deepcopy
import pytest
from avengine.timeline.v7_phase2_fallback import drop_failed_object
pytestmark=pytest.mark.fast_unit


def test_object_drop_keeps_original_and_all_remaining_sound_bindings():
    row={'scene_id':'parent','asset_ids':['human','bad','good'],
         'source_classes':{'human':['speech_playback'],'bad':['alarm_beep'],'good':['music_playback']},
         'proposed_sound_asset_ids':{'human':'english_recording','bad':'bad_alarm','good':'actual_music'},
         'motion_requests':{'human':{'action':'sit'}},'silent_asset_ids':[]}
    original=deepcopy(row)
    catalog=[{'asset_id':'human','kind':'human'},{'asset_id':'bad','kind':'object'},{'asset_id':'good','kind':'object'}]
    result=drop_failed_object(row,catalog,"native integration gate failed: ['source material never measured: bad']")
    assert row==original and result['asset_ids']==['human','good']
    assert result['source_classes']=={'human':['speech_playback'],'good':['music_playback']}
    assert result['proposed_sound_asset_ids']=={'human':'english_recording','good':'actual_music'}
    assert result['motion_requests']==original['motion_requests']
    assert result['element_degradations'][0]['asset_id']=='bad'


@pytest.mark.parametrize('reason',[
    'no strict native ceiling placement for human',
    'moving RIR at actual emitter exceeds unchanged 5deg geometry criterion',
    'independent original-family truth mismatch'])
def test_object_fallback_never_suppresses_human_rir_or_answer_failures(reason):
    row={'scene_id':'parent','asset_ids':['human','object']}
    catalog=[{'asset_id':'human','kind':'human'},{'asset_id':'object','kind':'object'}]
    assert drop_failed_object(row,catalog,reason) is None
