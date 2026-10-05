"""Round 5 joins and negative checks; all native receipts here are unit fixtures."""
from copy import deepcopy
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from avengine.qa.v7_recorded_audio import semantic_asr_comparison
from avengine.qa.v7_dialogue_binding import attach_dialogue_sources,worded_dialogues,audit_semantic_wording_exchange
from avengine.qa.v7_roster_intake import intake_roster
from avengine.qa.v7_content_binding import collect_content_binding
from avengine.qa.v7_validation import validate_content_items

pytestmark=pytest.mark.fast_unit

def load_fixture(name):
    path=Path(__file__).with_name(name)
    spec=importlib.util.spec_from_file_location('round5_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('reference,hypothesis,slot',[
 ('Let us meet at three twenty-five.','Let us meet at 3.25.','three twenty-five'),
 ('Let us meet at six forty.','Let us meet at 640.','six forty'),
 ('Let us meet at eleven fifty.','Let us meet at 11:50.','eleven fifty'),
 ('Please bring me a tea spoon.','Please bring me a teaspoon.','a tea spoon'),
 ('Please bring me a math workbook.','Please bring me a math work book.','a math workbook'),
 ('Let us meet at two forty.','Let\'s meet at 2.40.','two forty'),
])
def test_ASR_representation_equivalence_is_value_preserving(reference,hypothesis,slot):
    result=semantic_asr_comparison(reference,hypothesis,[slot])
    assert result['status']=='pass' and result['wer']==0

@pytest.mark.parametrize('hypothesis',[
 'Let us meet at 3.45.','Let us meet at 3250.','Let us meet at age 25.',
 'Let us meet at 3.60.','Let us meet at 2.25.'])
def test_clock_normalization_does_not_fix_wrong_or_ambiguous_values(hypothesis):
    result=semantic_asr_comparison('Let us meet at three twenty-five.',hypothesis,['three twenty-five'])
    assert result['status']=='blocked'

def test_critical_slot_is_a_complete_token_sequence_not_a_word_prefix():
    r=semantic_asr_comparison('Please bring me a clean mug.','Please bring me a clean mugged.',['a clean mug'])
    assert not r['all_critical_slots_correct'] and r['status']=='blocked'

def setup_native_join(tmp_path):
    _,registry,_=load_fixture('test_v7_roster_intake.py').package(tmp_path)
    roster=intake_roster(registry)
    receipt=tmp_path/'unit_native_receipt.json';receipt.write_text('{"unit_fixture":true}')
    actor=[]
    for i in (1,2):
        matrix=np.eye(4);matrix[0,3]=i
        actor.append({'native_actor_id':i,'asset_id':'red','pose':'stand','body_to_world_m':matrix.tolist(),
            'standing_pose_readback':{'status':'pass','source':str(receipt),'actor_id':str(i),'asset_id':'red',
              'source_registry_sha256':roster['source_registry_sha256']}})
    sounds=[];events=[];placements={};roles={}
    for i,noun in enumerate(('cup','book'),1):
        sid='script__role_'+str(i);eid='event_'+str(i);path=tmp_path/(sid+'.wav')
        # Numerical sound fixture, never a real speech or training admission.
        sf.write(path,np.zeros(32000),16000,subtype='FLOAT')
        text='Please bring me the '+noun+'.';sha=hashlib.sha256(path.read_bytes()).hexdigest()
        sounds.append({'sound_asset_id':sid,'scenario_id':'request','role_id':'role_'+str(i),
          'transcript':text,'voice_gender':'female',
          'voice_reference':{'speaker_id':'voice_'+str(i),'gender':'female'},
          'sha256':sha,'synthesis_readback':{'path':str(path),'sample_count':32000,'sample_rate_hz':16000},
          'semantic_answer':{'answer':noun,'label_en':noun,'label_zh':'杯子' if i==1 else '书'}})
        start=(i-1)*3;end=start+2
        events.append({'event_id':eid,'actor_id':str(i),'sound_asset_id':sid,'transcript':text,
                       'start_s':float(start),'end_s':float(end)})
        placements[eid]={'status':'pass','source':str(receipt),'event_id':eid,'actor_id':str(i),
            'window_start_s':start,'window_end_s':end,'native_emitter_world_m':[float(i),1.5,0],
            'emitter_point_kind':'mouth','source_sample_count':32000,'source_sample_rate_hz':16000,
            'source_audio_sha256':sha,'full_content_preserved':True,'trimmed_or_repeated':False}
        roles[eid]={'status':'pass','source':str(receipt),'actor_id':str(i),'scenario_id':'request','role':'role_'+str(i)}
    facts={'time':{'sample_rate_hz':48000,'sample_count':480000,'frame_count':2},'events':events,
      'actors':{str(i):{'asset_id':'red','root_positions_m':[[i,0,0]]*2,
                       'emitter_positions_m':[[i,1.5,0]]*2,
                       'source':{'root_readbacks':True,'emitter_readbacks':True}}
                for i in (1,2)}}
    manifest={'sounds':sounds,'dialogues':{'scenarios':[{'id':'request','question_en':'What does this person want?',
       'question_zh':'这个人想要什么？','slot':'object'}]}}
    return facts,manifest,actor,placements,roles,roster

def test_event_join_uses_registered_mouth_and_distinct_actual_roles(tmp_path):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert not deferred and len(attached['sounds'])==2
    assert len({s['voice_reference']['speaker_id'] for s in attached['sounds']})==2
    assert f['events'][0]['sound_asset_id']!=m['sounds'][0]['sound_asset_id']
    assert attached['sounds'][0]['native_placement_readback']['mouth_local_glb_m']==[0,1.5,0]
    assert '{appearance}' in attached['dialogues']['scenarios'][0]['question_zh']

@pytest.mark.parametrize('bad', ['mouth','gender','clock','pose','stale_registry','source_sha','role'])
def test_native_attachment_rejects_wrong_join_before_semantic_questions(tmp_path,bad):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    if bad=='mouth':p['event_1']['native_emitter_world_m'][1]+=.01
    if bad=='gender':m['sounds'][0]['voice_gender']='male'
    if bad=='clock':p['event_1']['window_end_s']=1.5
    if bad=='pose':a[0]['pose']='walk'
    if bad=='stale_registry':a[0]['standing_pose_readback']['source_registry_sha256']='wrong'
    if bad=='source_sha':p['event_1']['source_audio_sha256']='wrong'
    if bad=='role':r['event_1']['role']='role_2'
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert len(attached['sounds'])==1 and deferred

def test_one_asset_can_be_instanced_but_two_dialogue_roles_cannot_be_one_actor(tmp_path):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    f['events'][1]['actor_id']='1';p['event_2']['actor_id']='1';r['event_2']['actor_id']='1'
    p['event_2']['native_emitter_world_m']=[1,1.5,0]
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert not attached['sounds']
    assert any(d['code']=='different_dialogue_roles_share_one_physical_actor' for d in deferred)

def test_generic_QA26_author_stem_gets_observed_appearance_in_both_wordings():
    s={'scenarios':[{'id':'x','question_zh':'这个人请你拿什么？','question_en':'What does this person ask for?'}]}
    standard=worded_dialogues(s);alternate=worded_dialogues(s,'alternate')
    a=standard['scenarios'][0]['question_zh'];b=alternate['scenarios'][0]['question_zh']
    assert '{appearance}' in a and '{appearance}' in b and a!=b
    assert a.format(appearance='红衣人物')!=a.format(appearance='蓝衣人物')

def test_semantic_wording_exchange_preserves_unmodified_v6_truth():
    f,m=load_fixture('test_qa_semantic_questions.py').fixture()
    s=m['dialogues']['scenarios'][0];s['question_zh']='这个人想要什么？';s['question_en']='What does this person want?'
    r=audit_semantic_wording_exchange(f,m)
    assert r['status']=='pass' and r['standard_items']>=4 and not r['mismatches']

def test_content_panel_is_wired_and_its_answers_recompute_independently(tmp_path,monkeypatch):
    import avengine.qa.v7_content_binding as module
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    receipt=tmp_path/'unit_native_receipt.json'
    actors=[]
    for i,descriptor in enumerate(('红衣人物','蓝衣人物'),1):
        instance=a[i-1]
        actors.append({'actor_id':str(i),'asset_id':'red','pose':'stand',
          'descriptor_zh':descriptor,'visible':True,'capability':{'asset_kind':'standing_human'},
          'standing_pose_readback':instance['standing_pose_readback'],'body_to_world_m':instance['body_to_world_m']})
    emissions=[]
    for i,interval in enumerate(((0,2),(3,5)),1):
        emissions.append({'emission_id':'e'+str(i),'actor_id':str(i),'mechanism':'actor_live',
             'content_class':'speech','emitter_point_kind':'mouth','native_emitter_world_m':[i,1.5,0],
             'measured_activity_s':[interval]})
    scene={'window_s':[0,5],'actors':actors,'emissions':emissions}
    queries=[]
    for kind in ('same_class_actor','mixed_class_actor','offscreen_origin','live_or_playback','first_actor'):
        queries.append({'query':{'kind':kind,'time_s':1,'content_class':'speech',
          'subject_actor_id':'2','subject_descriptor_zh':'蓝衣人物'}})
    bundle={'scene':scene,'queries':queries,'source_registry':str(roster['source_registry'])}
    # Unit wiring test only: no fabricated media are labelled native-admitted.
    monkeypatch.setattr(module,'admission_blockers',lambda *args,**kwargs:[])
    public,private,audit=collect_content_binding(bundle,'unit',{'video':'neutral','audio':'neutral'})
    assert len(public)==10 and sum(audit['logical_counts_by_kind'].values())==5
    check=validate_content_items(public,private)
    assert check['passed_forms']==10 and not check['mismatches']
    private[0]['canonical_answer']='wrong'
    assert validate_content_items(public,private)['mismatches']
    scene['actors'][0]['pose']='walk'
    public,_,audit=collect_content_binding(bundle,'unit',{})
    assert not public and any(d['code']=='content_actor_not_standing' for d in audit['deferred'])

def test_custom_panel_selection_preserves_budget_and_does_not_crash_on_new_family():
    from avengine.qa.v7_families import select_questions
    p=[{'question_id':'x','logical_question_id':'x','qa_id':'V7-CONTENT-BIND',
        'protocol_panel':'v7_content_binding','modality':'audiovisual','form':'open','question_zh':'谁在发声？'}]
    g=[{'question_id':'x','logical_question_id':'x','canonical_answer':'红衣人物'}]
    a,b=select_questions(p,g,{'audiovisual':1})
    assert a==p and b==g

def test_falsified_attachment_vector_cannot_override_actual_native_FOA_frames(tmp_path):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    f['actors']['1']['emitter_positions_m'][0]=[1,1.55,0]
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert len(attached['sounds'])==1
    assert any(d['code']=='actual_native_frames_not_registered_standing_mouth' for d in deferred)

def test_registered_plan_does_not_override_a_different_loaded_body_resource(tmp_path):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    f['actors']['1']['asset_id']='different_actual_resource'
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert len(attached['sounds'])==1
    assert any(d['code']=='native_loaded_actor_resource_not_bound_to_formal_asset' for d in deferred)

def test_critical_noun_does_not_allow_a_new_negation_in_a_long_sentence():
    reference="The clean mugs are on the table and ready for the visitors."
    hypothesis="The clean mugs are not on the table and ready for the visitors."
    result=semantic_asr_comparison(reference,hypothesis,['clean mugs'])
    assert result['wer']<=.1 and result['status']=='blocked'
    assert not result['negation_counts_match']

def test_phone_message_cannot_claim_phone_action_from_an_ordinary_pose_plan(tmp_path):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    m['dialogues']['scenarios'][0]['roles']=[{'role_id':'role_1','required_asset_capability':'phone_call'}]
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert len(attached['sounds'])==1
    assert any(d['code']=='actual_registered_role_action_or_prop_missing' for d in deferred)

def test_real_megaphone_driver_keeps_registered_actor_identity(tmp_path):
    f,m,a,p,r,roster=setup_native_join(tmp_path)
    m['dialogues']['scenarios'][0]['roles']=[{'role_id':'role_1','required_asset_capability':'megaphone_speaker'}]
    r['event_1']['native_action_readback']={'status':'pass','source':r['event_1']['source'],
        'actor_id':'1','asset_kind':'megaphone_speaker','device_id':'megaphone1','native_driver_world_m':[1,1.6,0]}
    p['event_1'].update(emitter_point_kind='megaphone_driver',native_emitter_world_m=[1,1.6,0])
    f['actors']['1']['emitter_positions_m']=[[1,1.6,0]]*2
    attached,deferred=attach_dialogue_sources(f,m,a,p,r,roster=roster)
    assert not deferred and len(attached['sounds'])==2
    assert attached['sounds'][0]['dialogue_role_evidence']['actor_id']=='1'
    assert attached['sounds'][0]['native_placement_readback']['mouth_local_glb_m']==[0,1.5,0]

def test_megaphone_nonfinite_driver_is_rejected(tmp_path):
    facts,manifest,actors,placements,roles,roster=setup_native_join(tmp_path)
    manifest['dialogues']['scenarios'][0]['roles']=[{'role_id':'role_1','required_asset_capability':'megaphone_speaker'}]
    roles['event_1']['native_action_readback']={'status':'pass','source':roles['event_1']['source'],
        'actor_id':'1','asset_kind':'megaphone_speaker','device_id':'mega1',
        'native_driver_world_m':[float('nan'),0,0]}
    placements['event_1']['emitter_point_kind']='megaphone_driver'
    attached,deferred=attach_dialogue_sources(facts,manifest,actors,placements,roles,roster=roster)
    assert any(r.get('event_id')=='event_1' for r in deferred)
    assert all(s['role_id']!='role_1' for s in attached['sounds'])


def test_reconciliation_retains_raw_records_and_uses_existing_hypotheses(tmp_path):
    import hashlib,json
    import numpy as np
    import soundfile as sf
    from avengine.qa.v7_reconcile_semantic_assets import reconcile_batch
    raw=tmp_path/'raw';(raw/'audio').mkdir(parents=True);(raw/'records').mkdir()
    path=raw/'audio'/'s__r__attempt_1.wav'
    sf.write(path,np.zeros(16000,dtype=np.float32),16000,subtype='FLOAT')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    task={'sound_asset_id':'s__r','scenario_id':'s','role_id':'r',
        'transcript':'Bring the teaspoon.','semantic_answer':{'critical_semantic_slots':['teaspoon']},
        'voice_reference':{'speaker_id':'p100','gender':'female'},'synthesis_config':{},
        'standing_actor_plan':{'visual_gender':'female'}}
    receipt=dict(code_source='unit://cosy',model_license='Apache-2.0',model_id='unit/cosy',
                 model_revision='unit',code_revision='unit',model_license_path='unit')
    plan={'tasks':[task],'max_attempts':3,'model_receipt':receipt,
        'dialogues':{'scenarios':[{'id':'s','roles':[{'role_id':'r'}]}]},
        'script_source':'unit','script_source_sha256':'unit',
        'source_registry':'unit','source_registry_sha256':'unit'}
    plan_path=raw/'plan.json';plan_path.write_text(json.dumps(plan))
    proof={'status':'blocked','audio_sha256':sha,'reference_text':task['transcript'],
        'blind_decoded_text':'Bring the tea spoon.',
        'reference_text_or_answer_prompted_into_ASR':False}
    row={'sound_asset_id':'s__r','scenario_id':'s','role_id':'r','status':'dropped','terminal':True,
        'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'sound':None,
        'attempts':[{'attempt':1,'seed':1,'path':str(path),'status':'blocked','ASR':proof,
                     'reason':'initial_representation_rejection'}]}
    record=raw/'records'/'s__r.json';record.write_text(json.dumps(row));original=record.read_bytes()
    result=reconcile_batch(plan_path,raw,tmp_path/'reconciled',require_complete=True)
    assert result['passed_lines']==1 and result['rescued_initially_dropped_lines']==1
    assert result['total_actual_synthesis_attempts']==1
    assert result['new_synthesis_or_ASR_runs_during_reconciliation']==0
    assert record.read_bytes()==original and hashlib.sha256(path.read_bytes()).hexdigest()==sha
    sound=json.loads((tmp_path/'reconciled'/'semantic_manifest.json').read_text())['sounds'][0]
    assert sound['synthesis_readback']['source_sample_count'] is None
    assert sound['automatic_content_readback']['raw_comparison']['status']=='blocked'
    assert Path(sound['automatic_content_readback']['source']).is_file()

@pytest.mark.parametrize('reference,hypothesis,slot',[
 ('The wheelbarrow tyre is flat.','The wheelbarrow tire is flat.','wheelbarrow tyre'),
 ('For my meal I would choose plain yoghurt.','For my meal I would choose plain yogurt.','plain yoghurt'),
 ('For my meal I would choose a mushroom omelette.','For my meal I would choose a mushroom omelet.','mushroom omelette'),
 ('Let us meet at the noticeboard.','Let us meet at the notice board.','noticeboard')])
def test_standard_English_spelling_does_not_change_meaning(reference,hypothesis,slot):
    assert semantic_asr_comparison(reference,hypothesis,[slot])['status']=='pass'

@pytest.mark.parametrize('reference,hypothesis,slot',[
 ('How do I get to the coat rack by the window?','How do I get to the court rack by the window?','coat rack'),
 ('How do I get to the spare blankets in the hall cupboard?','How do I get to the spear blankets in the hall cupboard?','spare blankets')])
def test_spelling_equivalence_never_repairs_different_nouns(reference,hypothesis,slot):
    assert semantic_asr_comparison(reference,hypothesis,[slot])['status']=='blocked'

def test_dialogue_manifest_cannot_bypass_formal_actor_evidence_by_omitting_registry():
    from avengine.qa.v7_families import collect_questions
    raw=load_fixture('test_qa_unified_catalog.py')._fixture()
    manifest={'sounds':[],'dialogues':{'scenarios':[]}}
    _,_,audit=collect_questions(raw,'unit',{},semantic_manifest=manifest,items_per_family=1)
    assert any(d['code']=='registered_native_dialogue_evidence_missing' for d in audit['deferred'])
    assert all(audit['family_counts'][q]==0 for q in ('QA-26','QA-27','QA-28'))
