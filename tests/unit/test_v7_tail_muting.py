import numpy as np
import pytest
import soundfile as sf
from avengine.timeline.v7_tail_muting import REASON,tail_muted_segment,read_tail_muted_program
from avengine.timeline.v7_motion_source_roles import select_source_program
pytestmark=pytest.mark.fast_unit


def long_record(tmp_path,*,duration_s=6.,actual_onset_s=.6,annotated_onset_s=.4,sound_class='dog_bark'):
    samples=np.full(round(duration_s*16000),.25);samples[:round(actual_onset_s*16000)]=0.
    path=tmp_path/'admitted.wav';sf.write(path,samples,16000,subtype='FLOAT')
    return {'sound_asset_id':'long','sound_class':sound_class,'source_type':'recorded','duration_s':duration_s,
        'path':str(path),'header':{'sample_rate_hz':16000,'sample_count':len(samples)},
        'screens':{'automatic_activity':{'activity_intervals_samples':[[round(annotated_onset_s*16000),len(samples)]]},
            'license':{'original_id':'fixture','license':'test fixture','source_uri':'https://fixture.invalid/audio'}}}


def test_sixty_ms_fade_energy_falls_monotonically_and_all_later_samples_are_zero():
    profile={'tail_muted':True,'sample_count':3200,'mute_after_sample':2400,'fade_out_samples':960}
    y=tail_muted_segment(np.ones(4000),profile,output_sample_count=5000)
    energy=y[1440:2400]**2
    assert np.all(np.diff(energy)<=0.) and energy[0]==1. and energy[-1]==0.
    assert np.all(y[2400:]==0.)


def test_long_tail_fallback_remeasures_actual_waveform_and_preserves_original_file(tmp_path):
    record=long_record(tmp_path);path=tmp_path/'admitted.wav';before=path.read_bytes()
    selected=select_source_program([record],{'dog_bark'},(.9,1.2));profile=selected['program']
    assert profile['tail_muted']is True and profile['reason']==REASON
    assert profile['crop_start_sample']==round((.4-.1)*16000)
    assert profile['sample_count']==38400 and profile['fade_out_samples']==960
    _,measured=read_tail_muted_program(record,profile)
    assert measured==profile['processed_activity']
    assert measured['activity_intervals_samples'][0][0]==round((.6-.3)*16000)
    assert measured['activity_intervals_samples']!=record['screens']['automatic_activity']['activity_intervals_samples']
    assert measured['wet_tail_status']=='pending actual native rendered waveform measurement'
    assert path.read_bytes()==before


def test_whole_short_call_stays_preferred_and_a_shorter_motion_window_can_request_muting(tmp_path):
    record=long_record(tmp_path,duration_s=2.1,actual_onset_s=.1,annotated_onset_s=.1)
    complete=select_source_program([record],{'dog_bark'},(.9,1.2))
    assert complete['program']is None
    shortened=select_source_program([record],{'dog_bark'},(.9,1.2),maximum_program_s=1.6)
    assert shortened['program']['tail_muted']is True
    assert shortened['program']['sample_count']==25600


@pytest.mark.parametrize('source_type,sound_class',[('author_tts','speech_playback'),('recorded','speech_playback')])
def test_human_utterances_are_never_tail_muted(tmp_path,source_type,sound_class):
    record=long_record(tmp_path,sound_class=sound_class);record['source_type']=source_type
    profile={'tail_muted':True,'sample_count':38400}
    with pytest.raises(ValueError,match='human utterances must remain whole'):
        read_tail_muted_program(record,profile)
    with pytest.raises(ValueError,match='no complete admitted'):
        select_source_program([record],{sound_class},(.9,1.2))


def test_dry_renderer_uses_exact_faded_crop_and_reports_fresh_activity(tmp_path,monkeypatch):
    from avengine.timeline import v7_audio_sources as audio
    record=long_record(tmp_path);selected=select_source_program([record],{'dog_bark'},(.9,1.2));profile=selected['program']
    monkeypatch.setattr(audio,'filter_admission',lambda records:(records,[]))
    scene={'sample_rate_hz':16000,'duration_s':4.,'sources':{'animal':{'content_kind':'recorded_animal','dry_gain':1.}},
        'events':[{'event_id':'call','endpoint_id':'animal','start_tick':0,'end_tick':24}]}
    program={'schema':audio.PROGRAM_SCHEMA,'records':[record],'events':[{'event_id':'call','sound_asset_id':'long',
        'crop_start_sample':profile['crop_start_sample'],'tail_muting':profile}]}
    buses,receipt=audio.assemble_recorded_buses(scene,program)
    expected,measured=read_tail_muted_program(record,profile)
    np.testing.assert_array_equal(buses['animal'][:38400],expected)
    assert np.all(buses['animal'][38400:]==0.)
    event=receipt['events'][0]
    assert event['tail_muted'] and event['reason']==REASON
    assert event['processed_activity']==measured
    assert event['semantic_content_complete']is False


def test_original_motion_solver_uses_reprocessed_activity_not_old_profile_labels(tmp_path):
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
    record=long_record(tmp_path);profile=select_source_program([record],{'dog_bark'},(.9,1.2))['program']
    _,actual=read_tail_muted_program(record,profile)
    profile['processed_activity']={'activity_intervals_samples':[[1,2]]}
    other={'sound_asset_id':'other','header':{'sample_rate_hz':16000,'sample_count':16000},'screens':{'automatic_activity':{'activity_intervals_samples':[[0,16000]]}}}
    assets=[{'asset_id':'animal','kind':'animal','walk_allowed':True},{'asset_id':'person','kind':'human','walk_allowed':True}]
    row={'asset_ids':['animal','person'],'motion_requests':{'animal':{'action':'walk'}},'proposed_sound_asset_ids':{'animal':'long','person':'other'},
        'phase2_animal_program_samples':{'animal':profile},'motion_source_start_s':{'animal':.3,'person':0.},'camera_yaw_deg':0.,
        'motion_during_condition_requests':[{'qa_id':'QA-06','branch':'moving','target_asset_id':'animal','speed_m_s':.9,'speed_range_m_s':[.5,1.2]}]}
    result=prepare_motion_during_row(row,assets,[record,other])
    sound=result['motion_during_condition_plan']['plans'][0]['config']['sounds']['animal']
    assert sound['source_activity_intervals_samples']==actual['activity_intervals_samples']
    assert sound['audible_end_sample_exclusive']==actual['activity_intervals_samples'][-1][1]
