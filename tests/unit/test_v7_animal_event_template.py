from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from avengine.assets.animal_motion.event_template import animal_motion_events_template
from avengine.assets.roster_motion.straight_route import stand_walk_stand_distance

pytestmark = pytest.mark.fast_unit


@pytest.mark.parametrize('kind,species,mode,speed', [
    ('dog_walk','dog','walk_animation',.6), ('cat_walk','cat','walk_animation',.45),
    ('animal_slide','cat','slide_no_animation',.4)])
@pytest.mark.parametrize('programmed', [False, True])
def test_real_source_binding_and_every_event_pass_original_verifier(tmp_path, monkeypatch, kind, species, mode, speed, programmed):
    from avengine.timeline import v7_motion_quartet_template as base, v7_registered_sampling as sampling
    rate = 16000
    call_path = tmp_path/'call.wav'; tts_path = tmp_path/'speech.wav'
    sf.write(call_path,np.ones(36800)*.1,rate);sf.write(tts_path,np.ones(24000)*.1,rate)
    call={'sound_asset_id':'call','sound_class':'dog_bark' if species=='dog'else'cat_meow','normalized_path':str(call_path),
          'header':{'sample_rate_hz':rate,'sample_count':36800},'screens':{'automatic_activity':{'activity_intervals_samples':[[1600,35680]]}}}
    speech={'sound_asset_id':'tts','sound_class':'speech_playback','source_type':'author_tts','normalized_path':str(tts_path),
            'header':{'sample_rate_hz':rate,'sample_count':24000}}
    catalog=[{'asset_id':'animal','kind':'animal','species_id':species,'walk_allowed':mode=='walk_animation'},
             *[{'asset_id':aid,'kind':'human','species_id':'human','walk_allowed':True,'visual_gender':'male'}for aid in ['speaker','quiet1','quiet2']]]
    distance=stand_walk_stand_distance(.8,6.4,speed,.3)
    route={'reflect_walk':True,'positions_xz_m':np.column_stack((np.zeros(901),-6.+distance)).tolist(),
        'travel_yaws_rad':[np.pi/2]*901,'speed_m_s':speed,'speed_range_m_s':{'dog':[.5,1.2],'cat':[.3,.8],'slide':[.2,.5]}['slide'if mode=='slide_no_animation'else species],
        'listener_m':[0.,1.1,0.],'camera_yaw_deg':0.,'candidate_trials':[],
        'emission_slot':{'offset_s':0.,'start_s':1.3,'end_s':3.6,'post_until_s':5.9,'branch':'nearer',
            'radial_change_m':-2.3*speed,'leg':[24,192],'window_frames':[24,192],
            'qa06':True,'qa15':True,'qa16':True,'qa17':True},
        'short_window_walk':{'walk_start_s':.8,'walk_end_s':6.4,'silent_query_until_s':5.9,
            'emission_s':[1.3,3.6],'heading_rad':np.pi/2,'straight_line':True}}
    seed={'camera_yaw_deg':0.,'listener_m':[0.,1.1,0.],'asset_ids':['animal','speaker','quiet1','quiet2'],'silent_asset_ids':['quiet1','quiet2'],
          'source_classes':{'animal':[call['sound_class']],'speaker':['speech_playback']},
          'proposed_sound_asset_ids':{'animal':'call','speaker':'tts'},'semantic_human_asset_ids':['speaker'],
          'phase2_whole_source_recordings':{'animal':[1.3]},'scene_demand':{'required_asset_ids':['animal','speaker','quiet1','quiet2']},
          'motion_quartet_construction':{'route':route},'motion_requests':{'animal':{'speed_m_s':speed,'intentional_slide':mode=='slide_no_animation',
          'animal_motion_admission':{'mode':mode},'motion_quartet_route':route}}}
    if programmed:
        seed.pop('phase2_whole_source_recordings')
        seed['phase2_sound_program_schedule_s']={'animal':[1.3]}
        seed['phase2_animal_program_samples']={'animal':{'crop_start_sample':0,'sample_count':36800,
            'selected_activity_index':0,'required_activity_intervals_samples':[[1600,35680]]}}
    calls=[]
    def seed_template(row,*args,**kwargs):
        calls.append((row,kwargs));return deepcopy(seed)
    monkeypatch.setattr(base,'motion_quartet_template',seed_template)
    monkeypatch.setattr(sampling,'semantic_record',lambda *args:deepcopy(speech))
    manifest={'sounds':[{'sound_asset_id':'tts','synthesis_readback':{'duration_s':1.5}}]}
    row=animal_motion_events_template({'motion_source_type':kind},None,catalog,manifest,[call],None,source_type=kind)
    assert len(calls)==1 and calls[0][0]['animal_motion_event_count']==1
    assert row['asset_ids']==['animal','speaker'] and row['silent_asset_ids']==[]
    assert row['native_view_roles']['speaker']['view']=='outside'
    assert row['animal_motion_event_count']==3
    schedule=row['phase2_sound_program_schedule_s'if programmed else'phase2_whole_source_recordings']['animal']
    assert schedule==pytest.approx([1.3,11.3,21.3])
    assert row['source_classes']==seed['source_classes']
    if programmed:assert row['phase2_animal_program_samples']==seed['phase2_animal_program_samples']
    checks=row['animal_motion_event_planned_verification']
    assert len(checks)==3 and all(all(check['families'].values())for check in checks)
    assert all(check['qa16_integer_queries']for check in checks)
    assert all(check['status']=='pass'for check in row['animal_delivery_speed_verification'])
    assert len(row['motion_during_condition_plan']['plans'])==12
    assert len(row['motion_requests']['animal']['motion_quartet_route']['animal_stationary_turn_intervals_s'])==2
