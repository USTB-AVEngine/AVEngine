from copy import deepcopy
import pytest
from avengine.timeline.v7_demand import assign_authored_tts
from avengine.timeline.v7_phase2_sources import schedule_authored_speech
pytestmark=pytest.mark.fast_unit


def test_authored_assignment_keeps_bodies_motion_silent_bystander_and_fixed_voice():
    row={'asset_ids':['walking','standing','silent'],'motion_requests':{'walking':{'action':'walk'}},
         'silent_asset_ids':['silent'],'source_classes':{},'phase2_recorded_speech_schedule_s':{'walking':[2.],'standing':[5.]}}
    catalog=[{'asset_id':a,'kind':'human','visual_gender':'female'}for a in row['asset_ids']]
    sound={'sound_asset_id':'line','scenario_id':'request','role_id':'one','transcript':'Bring a clean mug.',
           'voice_reference':{'speaker_id':'qualified_voice','gender':'female'},'synthesis_readback':{'duration_s':2.}}
    before=deepcopy(row);planned=assign_authored_tts(row,catalog,{'sounds':[sound]})
    assert row==before
    assert planned['asset_ids']==row['asset_ids'] and planned['motion_requests']==row['motion_requests']
    assert planned['semantic_human_asset_ids']==['standing']
    assert planned['phase2_authored_speech_schedule_s']['standing']==[.5,10.5,20.5]
    assert planned['phase2_recorded_speech_schedule_s']=={'walking':[2.]}


def test_whole_tts_schedule_creates_real_activity_in_every_ten_second_epoch():
    scene={'events':[{'event_id':'old','endpoint_id':'voice1','source_id':1,'sound_class':'speech_playback','start_tick':5,'end_tick':25}]}
    program={'events':[{'event_id':'old','sound_asset_id':'whole','crop_start_sample':0}],
             'records':[{'sound_asset_id':'whole','source_type':'author_tts','header':{'sample_count':32000}}]}
    assets=[{'asset_id':'standing','endpoint_id':'voice1','kind':'human'}]
    schedule_authored_speech(scene,program,assets,{'standing':[.5,10.5,20.5]})
    assert [(e['start_tick'],e['end_tick'])for e in scene['events']]==[(5,25),(105,125),(205,225)]
    assert all(b['sound_asset_id']=='whole'and b['crop_start_sample']==0 for b in program['events'])
    program['records'][0]['source_type']='recorded'
    with pytest.raises(ValueError,match='cannot relabel'):schedule_authored_speech(scene,program,assets,{'standing':[1.]})
