import json
from copy import deepcopy
import numpy as np
import pytest
import soundfile as sf
from avengine.timeline.v7_audio_sources import source_blockers
from avengine.timeline.v7_environmental_sources import attach_environmental_program
from avengine.capture.kujiale_registered_environment import environment_bindings
from avengine.qa.kujiale_current_color import filter_current_scene_colors

def test_generated_audio_requires_all_existing_and_sao_specific_screens():
    row={'strict_v7_eligible':True,'formal_status':'admitted','source_type':'generated_audio',
        'generation':{'model_id':'stabilityai/stable-audio-open-1.0'},'owner_acceptance':'owner accepts listed risk',
        'allowed_splits':['train'],'split':'train','automatic_no_speech':{'status':'pass'},
        'automatic_activity_readback':{'status':'pass','active_duration_s':1.2},
        'reference_dedup':{'status':'pass','coverage_complete_for_listed_roots':True,'positive_controls_pass':True},
        'automatic_label':[{'status':'pass','confidence':.3,'margin':.1,'gate':{'status':'pass','threshold':.1917129977,'minimum_margin':.05}}],
        'screens':{n:{'status':'pass'} for n in ['license','provenance','deduplication','automatic_label']}}
    row['screens']['license'].update(license='Stability AI Community License',original_id='one',source_uri='upstream')
    assert source_blockers(row)==[]
    for field,value,reason in [('owner_acceptance',None,'owner_or_train'),('automatic_no_speech',{'status':'fail'},'no_speech'),
                                ('reference_dedup',{'status':'pass'},'complete_reference'),('split','dev','train_only')]:
        bad=deepcopy(row);bad[field]=value
        assert any(reason in x for x in source_blockers(bad))
    bad=deepcopy(row);bad['automatic_label'][0]['margin']=.049
    assert 'generated_audio_calibrated_label_gate_not_pass' in source_blockers(bad)
    bad=deepcopy(row);bad['screens']['license']['status']='fail'
    assert 'upstream_license_not_pass' in source_blockers(bad)

def test_recordings_use_physical_mouth_or_device_points_and_air_has_no_events(tmp_path):
    wav=tmp_path/'clip.wav';sf.write(wav,np.sin(np.arange(64000)*.1)*.1,16000)
    records=[]
    for cls in ['dog_bark','alarm_beep']:
        records.append({'sound_asset_id':cls,'sound_class':cls,'normalized_path':str(wav),'original_id':cls,
            'screens':{'license':{'source_uri':'origin','license':'approved'},'automatic_activity':{'activity_intervals_samples':[[0,64000]]}}})
    points=[[1,.4,2],[2,1.2,3]]
    bindings=[{'asset_id':cls,'sound_class':cls,'silent_decor':False,'expected_emitter_m':p} for cls,p in zip(['dog_bark','alarm_beep'],points)]
    bindings.append({'asset_id':'air','sound_class':None,'silent_decor':True,'expected_emitter_m':[1,2,3]})
    scene={'native_placement_candidates':[[99,99,99]],'sources':{},'events':[]}
    program={'records':[],'events':[]}
    attach_environmental_program(scene,program,{'environment_sound_classes':['dog_bark','alarm_beep'],'native_environment':bindings},records,np.random.default_rng(0),[3,17])
    assert [s['trajectory'][0]['position_m'] for s in scene['sources'].values()]==points
    assert all(s['native_visual_kind']=='registered_physical' for s in scene['sources'].values())
    assert [x['asset_id'] for x in scene['decorations']]==['air']
    assert len(scene['events'])==8
    assert len({e['start_tick'] for e in scene['events']})==8

def test_current_scene_geometry_is_pending_until_real_native_qualification(tmp_path):
    p=tmp_path/'book.json'
    row={'asset_id':'dog','ue_native_physical_status':'pass','silent_decor':False,'allowed_v7_classes_candidate':['dog_bark'],
        'ue_asset_paths':['/Game/Dog.Dog'],'ue_project':str(tmp_path/'own.uproject')}
    p.write_text(json.dumps({'assets':[row]}))
    plan={'asset_id':'dog','silent_decor':False,'sound_class':'dog_bark','status':'pass_geometry_requires_current_native_scene',
          'ue_asset_paths':['/Game/Dog.Dog'],'expected_emitter_m':[1,.4,2]}
    scene={'require_current_native_environment':True,'sources':{'pet':{'actor_id':1001,'native_asset_id':'dog',
           'trajectory':[{'position_m':[1,.4,2]}],'native_scene_plan':plan}}}
    result=environment_bindings(scene,'new_house',p,set())
    assert result['pet']['native']['status']=='pass_geometry_requires_current_native_scene'
    bad=deepcopy(scene);bad['sources']['pet']['native_scene_plan']['sound_class']='printer'
    with pytest.raises(ValueError,match='registered asset classes'):environment_bindings(bad,'new_house',p,set())
    bad=deepcopy(scene);bad.pop('require_current_native_environment')
    with pytest.raises(ValueError,match='native requalification'):environment_bindings(bad,'new_house',p,set())

def test_colour_mismatch_drops_only_the_question_without_relabelling_asset():
    sample={'status':'pass','label':'gray_silver','body_pixels':200,'family_shares':{'gray_silver':1.}}
    visual={'native_physical_environment':{'frames':[{'frame_index':i,'entities':[{'source_id':1001,'registered_color_label':'blue','color':sample}]} for i in [0,15]]}}
    raw={'actors':{'1001':{'display_label':'蓝色音箱'}},'appearance_review':{'actors':{}}}
    q=[{'question_id':'color','logical_question_id':'color','modality':'vision','question_zh':'蓝色音箱位于哪边？'},
       {'question_id':'sound','logical_question_id':'sound','modality':'audio','question_zh':'声音是什么？'}]
    gold=[{'question_id':x['question_id']} for x in q];before=deepcopy(visual)
    public,private,audit=filter_current_scene_colors(q,gold,{'native_bundle':raw},visual,0)
    assert [x['question_id'] for x in public]==['sound']
    assert audit['discarded_logical_questions']==1
    assert visual==before

def test_independent_class_truth_rejects_tampered_answer_and_ambiguous_onset():
    from copy import deepcopy
    from avengine.qa.kujiale_pilot_truth import validate_pilot_items
    audit={'v6_generation':{'candidate_attempts':[]},'v6_items':[]}
    proof={'question_seed':'fixture','protocol_scene':{'events':[]},
           'events':{'dog':{'native_endpoint_id':'environment1','sound_class':'dog_bark',
                           'activity_intervals_samples_global':[[14400,22400]]}}}
    evidence={'start_s':0.,'source_qa_evidence':proof}
    request={'question_id':'q','qa_id':'V7-CLASS34','protocol_panel':'v7_admitted_recording_classes','form':'open'}
    gold={'question_id':'q','canonical_answer':'dog_bark','answer':'dog_bark',
          'evidence':{'activity':{'endpoint':'environment1','start_tick':9,'end_tick':14},'domain':['dog_bark','cat_meow']}}
    assert validate_pilot_items([request],[gold],audit,evidence)['status']=='pass'
    wrong=deepcopy(gold);wrong['answer']='cat_meow'
    assert validate_pilot_items([request],[wrong],audit,evidence)['mismatches']
    proof['events']['cat']={'native_endpoint_id':'environment2','sound_class':'cat_meow',
                           'activity_intervals_samples_global':[[14400,22400]]}
    with pytest.raises(ValueError,match='ambiguous'):
        validate_pilot_items([request],[gold],audit,evidence)


def test_independent_speaker_count_unions_segments_and_checks_actual_mcq_value():
    from avengine.qa.kujiale_pilot_truth import validate_pilot_items
    audit={'v6_generation':{'candidate_attempts':[]},'v6_items':[]}
    proof={'question_seed':'fixture','events':{},'protocol_scene':{'events':[
        {'class_id':0,'source_id':1,'endpoint_id':'voice1','start_tick':5,'end_tick':10},
        {'class_id':0,'source_id':1,'endpoint_id':'voice1','start_tick':20,'end_tick':30},
        {'class_id':1,'source_id':2,'endpoint_id':'voice2','start_tick':50,'end_tick':60}]}}
    request={'question_id':'q','qa_id':'STARSS-SPEAKERS','protocol_panel':'STARSS23_v0.2','form':'mcq',
             'choices':[{'id':'1','text':'3'},{'id':'2','text':'2'},{'id':'3','text':'1'}]}
    gold={'question_id':'q','canonical_answer':'2','answer':'2'}
    report=validate_pilot_items([request],[gold],audit,{'start_s':0.,'source_qa_evidence':proof})
    assert report['checked_forms']==1 and report['status']=='pass'
    gold['answer']='1'
    assert validate_pilot_items([request],[gold],audit,{'start_s':0.,'source_qa_evidence':proof})['status']=='fail'

def test_disjoint_human_cannot_hide_a_physical_animal_from_production_camera():
    from types import SimpleNamespace
    from avengine.capture.kujiale_pilot_environment import human_clear_of_environment
    video=SimpleNamespace(scene={'listener_m':[0.,1.,0.]},environment_nodes={'animal':{
        'physical':True,'native_evidence':{'placement':{'native_world_bounds_m':[[1.,0.,1.],[2.,1.,2.]]}}}})
    assert not human_clear_of_environment(video,np.array([.2,0.,.2]),np.array([.7,2.,.7]))
    assert human_clear_of_environment(video,np.array([.2,0.,-2.]),np.array([.7,2.,-1.]))


def test_qualified_camera_offset_keeps_exact_native_pose_gate(monkeypatch):
    from types import SimpleNamespace
    from avengine.capture.v7_registered_ue import NativeRegisteredUEVideo
    from avengine.backends.spear_ue import research_runtime
    video=object.__new__(NativeRegisteredUEVideo)
    video.scene={'listener_m':[1.,2.,3.],'camera_yaw_knots':[{'time_s':0.,'yaw_deg':20.},{'time_s':30.,'yaw_deg':380.}]}
    video.current_camera_yaw_offset=37.
    video.camera=SimpleNamespace(K2_GetRootComponent=lambda:object())
    pose={'rotation_deg':[0.,0.,-33.],'location_cm':[100.,300.,200.]}
    monkeypatch.setattr(research_runtime,'read_scene_component_pose',lambda component:pose)
    assert video._camera_readback(0.)['yaw_deg']==57.
    pose['rotation_deg'][2]+=.001
    with pytest.raises(ValueError,match='camera readback'):video._camera_readback(0.)
    pose['rotation_deg'][2]= -33.;pose['location_cm'][0]+=.1
    with pytest.raises(ValueError,match='camera readback'):video._camera_readback(0.)


def test_native_rollout_requires_first_seal_and_stops_repeated_gate_failures():
    from avengine.timeline.kujiale_asset_pilot import NativePilotBatchGate
    gate=NativePilotBatchGate()
    assert gate.batch_limit==1
    gate.observe({'status':'formal_alpha_admitted'})
    with pytest.raises(RuntimeError,match='canary failed'):gate.require_next_batch()
    gate=NativePilotBatchGate()
    good={'status':'formal_alpha_admitted','independent_truth':{'status':'pass'}}
    gate.observe(good)
    assert gate.batch_limit==2
    bad={'status':'failed','error_type':'ValueError','reason':'native UE camera readback differs from the alpha pan'}
    gate.observe(bad);gate.require_next_batch()
    gate.observe(bad)
    with pytest.raises(RuntimeError,match='two consecutive'):gate.require_next_batch()
    gate.observe(good);gate.require_next_batch()


def test_formal_native_pilot_requires_actual_seal_and_independent_truth(tmp_path):
    from avengine.timeline.kujiale_asset_pilot import formal_native_parent_passed
    scene='canary';terminal=tmp_path/'evidence/parent_results'/f'{scene}.json';terminal.parent.mkdir(parents=True)
    terminal.write_text(json.dumps({'status':'formal_alpha_admitted'}))
    parent=tmp_path/'production/parents'/scene;parent.mkdir(parents=True)
    assert not formal_native_parent_passed(tmp_path,scene)
    (parent/'alpha_seal_completed.json').write_text('{}')
    proof=parent/'independent_truth_audit.json';proof.write_text(json.dumps({'status':'fail','mismatches':['tampered truth']}))
    assert not formal_native_parent_passed(tmp_path,scene)
    proof.write_text(json.dumps({'status':'pass','mismatches':[]}))
    assert formal_native_parent_passed(tmp_path,scene)


def test_binding_view_centres_horizontal_rays_and_rejects_unbindable_pairs():
    from avengine.capture.kujiale_pilot_environment import binding_camera_yaw,binding_pair_fits_view
    listener=[0.,1.5,0.]
    assert binding_camera_yaw(listener,[[1.,0.,-1.],[-2.,0.,-2.]])==pytest.approx(0.)
    assert binding_camera_yaw(listener,[[1.,-5.,-1.],[-2.,4.,-2.]])==pytest.approx(0.)
    assert binding_pair_fits_view(listener,[0.,0.,-2.],[1.,0.,-2.])
    assert not binding_pair_fits_view(listener,[0.,0.,-2.],[0.,0.,-4.])
    assert not binding_pair_fits_view(listener,[-2.,0.,0.],[2.,0.,0.])


def test_static_foot_reference_still_rejects_pose_or_root_changes():
    from avengine.capture.v7_registered_ue import NativeRegisteredUEVideo
    video=object.__new__(NativeRegisteredUEVideo)
    video.alpha_standing_refs={'voice1':{'position_m':[1.,0.,2.],'reference':{'status':'pass','gap_m':.003,'views':['front','rear']}}}
    actor={'endpoint_id':'voice1','position_m':[1.,0.,2.],'moving':False,'action_id':'registered_stand'}
    assert video._uncropped_foot_probe(actor)['gap_m']==.003
    for change in [{'moving':True},{'action_id':'walk'},{'position_m':[1.,.001,2.]}]:
        with pytest.raises(ValueError,match='support reference changed'):video._uncropped_foot_probe(dict(actor,**change))
