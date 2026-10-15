from collections import Counter
import numpy as np
import pytest
from avengine.timeline.v7_demand import demand_snapshot,default_targets,record_selected_parent,read_demand_state
from avengine.timeline.v7_wet_support import event_wet_support


def test_default_target_keeps_thirty_percent_vision():
    targets=default_targets()
    assert sum(targets.values())==pytest.approx(1)
    assert sum(targets[f] for f in ('QA-05','QA-07','QA-10','QA-11','QA-14','QA-20'))==pytest.approx(.3)


def test_feedback_prioritizes_underproduced_family_from_actual_counts():
    targets={'QA-06':.5,'QA-19':.5}
    s=demand_snapshot({'logical_questions':100,'family_counts':{'QA-19':100},'target_shares':targets},next_questions=14)
    assert s['family_deficits']['QA-06']==57
    assert s['family_deficits']['QA-19']==0
    assert s['selection_counts']['QA-06']<s['selection_counts']['QA-19']


def test_feedback_rejects_forms_or_candidates_counted_as_actual_items():
    with pytest.raises(ValueError):
        demand_snapshot({'logical_questions':14,'family_counts':{'QA-06':28}})


def test_completed_parent_resume_is_idempotent_and_failed_parent_is_ignored(tmp_path):
    p=tmp_path/'state.json'
    r={'status':'integration_pilot_admitted','scene_id':'parent',
       'windows':[{'logical_questions':14,'family_counts':{'QA-06':3,'QA-19':11}}]}
    record_selected_parent(p,r);record_selected_parent(p,r)
    assert read_demand_state(p)['logical_questions']==14
    record_selected_parent(p,{'status':'failed_not_formal','scene_id':'failed'})
    r['windows'][0]['family_counts']={'QA-06':7,'QA-19':7}
    record_selected_parent(p,r)
    assert read_demand_state(p)['family_counts']=={'QA-06':7,'QA-19':7}


def test_wet_silence_bound_contains_every_exact_convolution_sample():
    dry=np.zeros(100);dry[20:25]=[1,-2,3,1,2]
    kernels=np.zeros((4,20));kernels[:,4:9]=np.arange(20).reshape(4,5)+1
    result=event_wet_support(dry,[{'event_id':'e','output_start_sample':10,'output_end_sample_exclusive':40}],[kernels])['e']
    exact=np.stack([np.convolve(dry,h) for h in kernels])
    nonzero=np.flatnonzero(np.any(exact!=0,axis=0));lo,hi=result['support_samples_global']
    assert (lo,hi)==(int(nonzero[0]),int(nonzero[-1])+1)
    assert hi>25  # measured native tail is retained after the dry source stops
    assert result['silence_threshold_added'] is False


def test_moving_native_delay_spread_expands_silence_bound():
    dry=np.zeros(100);dry[20]=1
    h=np.zeros((4,50));h[:,4]=1;h[:,40]=.01
    r=event_wet_support(dry,[{'event_id':'e','output_start_sample':10,'output_end_sample_exclusive':30}],[h],delay_spread_samples=8)['e']
    assert r['support_samples_global']==[16,69]


def test_overlapping_endpoint_events_do_not_forge_per_event_silence():
    dry=np.ones(100);h=np.ones((4,20))
    events=[{'event_id':'a','output_start_sample':10,'output_end_sample_exclusive':40},
            {'event_id':'b','output_start_sample':30,'output_end_sample_exclusive':50}]
    assert event_wet_support(dry,events,[h])=={}


def test_all_33_families_have_scene_requirements_and_minimum_share():
    from avengine.timeline.v7_demand import FAMILY_SCENE_REQUIREMENTS
    from avengine.qa.v7_families import FAMILIES,PANELS
    expected=set(FAMILIES+PANELS+("V7-CLASS34","V7-CONTENT-BIND"))
    assert len(expected)==33
    assert set(default_targets())==set(FAMILY_SCENE_REQUIREMENTS)==expected
    assert min(default_targets().values())>=.01
    assert all(r["conditions"] for r in FAMILY_SCENE_REQUIREMENTS.values())


def test_proposed_clip_is_resolved_inside_admitted_pool():
    from avengine.timeline.v7_visible_sources import select_proposed_record
    pool=[{'sound_asset_id':'long'},{'sound_asset_id':'short'}]
    row={'proposed_sound_asset_ids':{'body':'short'}}
    assert select_proposed_record(pool,'body',row,np.random.default_rng(1))['sound_asset_id']=='short'
    row['proposed_sound_asset_ids']['body']='unadmitted'
    with pytest.raises(ValueError):select_proposed_record(pool,'body',row,np.random.default_rng(1))


def test_occluder_peer_failure_keeps_only_complete_native_target_stream():
    from avengine.timeline.v7_registered_audio import complete_native_occluder_targets
    rows=[{'target_instance_id':str(a),'frame_index':f,'occluder_instance_ids':['4'],
           'unknown_occluder_pixels':1 if a==2 and f==1 else 0} for a in (1,2,3) for f in range(3)]
    rows=[r for r in rows if not(r['target_instance_id']=='3'and r['frame_index']==2)]
    selected,ids=complete_native_occluder_targets({'frame_count':3,'occluder_evidence':{'frame_records':rows}})
    assert ids==['1']
    assert selected==[r for r in rows if r['target_instance_id']=='1']


def test_independence_ignores_crop_time_labels_but_keeps_target_and_answer():
    from avengine.timeline.v7_question_independence import summarize_independence,window_overlap
    q=[{'question_id':'a','logical_question_id':'a','qa_id':'QA-17','question_zh':'第8秒蓝色的人是否移动？'},
       {'question_id':'b','logical_question_id':'b','qa_id':'QA-17','question_zh':'第4秒蓝色的人是否移动？'}]
    gold=[{'question_id':x['question_id'],'canonical_answer':True,'evidence':{'actor_id':'1'}}for x in q]
    report,_=summarize_independence('parent',q,gold)
    assert report['raw_logical_questions']==2 and report['independent_questions']==1
    gold[1]['canonical_answer']=False
    assert summarize_independence('parent',q,gold)[0]['independent_questions']==2
    gold[1]['canonical_answer']=True;gold[1]['evidence']['actor_id']='2'
    assert summarize_independence('parent',q,gold)[0]['independent_questions']==2
    assert window_overlap([0,4,8,12,16,20])['adjacent_overlap_seconds']==[6]*5
    assert window_overlap([0,10,20])['maximum_adjacent_overlap_s']==0


def test_batched_original_floor_rays_keep_exact_supports(tmp_path):
    from types import SimpleNamespace
    from avengine.assets.v7_phase2_floor import OriginalVerticalStageRay
    path=tmp_path/'original.glb';path.write_bytes(b'original mesh identity')
    mesh=SimpleNamespace(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.],[1.,0.,1.]]),faces=np.array([[0,1,2],[1,3,2]]))
    points=np.array([[.1+(i%10)*.08,.1,.1+(i//10)*.05]for i in range(129)])
    batched=OriginalVerticalStageRay(path,mesh=mesh)
    reference=OriginalVerticalStageRay(path,mesh=mesh);reference.MAX_QUERY_POINTS=1000
    assert np.array_equal(batched.many(points),reference.many(points))
    assert batched.requested_points==reference.requested_points==129


def test_post_admission_parent_dedup_keeps_earliest_crop_and_sparse_counts(tmp_path):
    import json
    from avengine.timeline.v7_question_independence import deduplicate_parent_delivery
    source=tmp_path/'raw';root=tmp_path/'parent';root.mkdir()
    windows=[]
    for i,(start,answer) in enumerate([(0,True),(5,True),(10,False)]):
        wid=f'parent__v0_a0__w{i:02d}';dest=source/'windows'/wid;dest.mkdir(parents=True)
        for name in ('video.mp4','audio_foa.wav','audio_mono.wav'):(dest/name).write_bytes(b'unchanged admitted media')
        q={'question_id':wid+'__open','logical_question_id':wid,'qa_id':'QA-17','question_en':f'At {8-i} seconds, did the person move?'}
        g={'question_id':q['question_id'],'canonical_answer':answer,'evidence':{'actor_id':'1'}}
        (dest/'questions.json').write_text(json.dumps([q]));private=source/'private';private.mkdir(exist_ok=True)
        (private/f'{wid}_answers.json').write_text(json.dumps([g]));(private/f'{wid}_evidence.json').write_text(json.dumps({'source_qa_evidence':{'window_start_s':start}}))
        windows.append({'window_id':wid,'logical_questions':1,'family_counts':{'QA-17':1}})
    result=deduplicate_parent_delivery({'scene_id':'parent'},{'status':'integration_pilot_admitted','windows':windows,'pilot_logical_questions':3},root,source)
    assert result['raw_logical_questions']==3
    assert result['independent_logical_questions']==2 and result['dedup_dropped']==1
    assert [w['logical_questions']for w in result['windows']]==[1,0,1]
    assert result['original_raw_native_and_14_item_admission_preserved'] is True
    assert len(json.loads((source/'windows'/windows[1]['window_id']/'questions.json').read_text()))==1


def test_content_dedup_keeps_same_appearance_on_different_physical_actors():
    from avengine.timeline.v7_question_independence import independence_key
    q={'qa_id':'V7-CONTENT-BIND','question_en':'Who spoke first?'}
    gold=lambda actor:{'canonical_answer':'the person in the blue shirt','evidence':{'truth':{'actor_id':actor}}}
    first=independence_key('parent',q,gold('1'))[0]
    second=independence_key('parent',q,gold('2'))[0]
    assert first!=second
    assert first==independence_key('parent',q,gold('1'))[0]


@pytest.mark.fast_unit
def test_occlusion_proposals_require_a_real_foreground_position():
    from avengine.capture.v7_occlusion_demand import foreground_geometry
    listener=[0.,1.3,0.];target=[0.,0.,-3.]
    assert foreground_geometry(listener,target,[.2,0.,-1.5])['foreground']
    assert not foreground_geometry(listener,target,[0.,0.,-4.])['foreground']
    assert not foreground_geometry(listener,target,[1.,0.,-1.5])['foreground']
    assert not foreground_geometry(listener,target,[0.,0.,1.])['foreground']


@pytest.mark.fast_unit
def test_requested_occluder_requires_target_placement_first():
    from avengine.capture.v7_occlusion_demand import proposal_score
    scene={'listener_m':[0.,1.3,0.],'native_occlusion_relations':[
        {'target_asset_id':'rear','occluder_asset_id':'front'}]}
    with pytest.raises(ValueError,match='after its target'):
        proposal_score(scene,{'declaration':{'asset_id':'front'}},[0,0,-1.5],[])


@pytest.mark.fast_unit
def test_authored_content_demand_constructs_members_roles_and_separate_taxonomy():
    from avengine.timeline.v7_demand import plan_registered_content_demand
    row={'schema':'v7-recorded-registered-standing-v1','scene_seed':7,'members':['v0_a0'],
         'window_count':2,'questions_per_window':14,'production_scenario':{
             'speaker_count':2,'asset_ids':['blue','yellow','green'],
             'category_focus_class':'dog_bark','semantic_sound_ids':['role1','role2']}}
    assets=[{'asset_id':a,'kind':'human','visual_gender':'female'} for a in ['blue','yellow','green']]
    semantic={'sounds':[{'sound_asset_id':'role'+str(i),'scenario_id':'request',
                        'role_id':str(i),'voice_reference':{'gender':'female'}} for i in (1,2)]}
    planned=plan_registered_content_demand(row,{'logical_questions':0,'family_counts':{}},assets,semantic)
    assert len(planned['members'])==4 and planned['window_count']==5
    assert planned['production_scenario']['semantic_scenario_id']=='request'
    assert planned['production_scenario']['category_focus_class'] is None
    assert planned['scene_demand']['required_program_source_type']=='author_tts'
    assert row['members']==['v0_a0']  # No caller/template mutation.
    semantic['sounds'][1]['voice_reference']['gender']='male'
    with pytest.raises(ValueError,match='gender'):
        plan_registered_content_demand(row,{'logical_questions':0,'family_counts':{}},assets,semantic)


@pytest.mark.fast_unit
def test_admitted_parent_keeps_running_but_output_containment_remains(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from avengine.timeline import v7_storage_budget as budget
    monkeypatch.setenv(budget.ENV_KEY,json.dumps({'root':str(tmp_path/'own'),'filesystem':str(tmp_path),
        'minimum_free_bytes':100,'maximum_task_bytes':1,'admission_only':True}))
    monkeypatch.setattr(budget.shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    assert budget.check_write(tmp_path/'own'/'running',expected_bytes=1000)['resource_limits_enforced_at']=='parent_admission'
    with pytest.raises(budget.StorageBudgetExceeded,match='escaped'):
        budget.check_write(tmp_path/'another_task')


@pytest.mark.fast_unit
def test_distance_request_checks_the_speaking_span_not_only_the_whole_route():
    from avengine.assets.roster_motion.demand import matches_distance_request
    times=np.array([0.,1.,2.,3.]);points=np.array([[0,0,-1.],[0,0,-1.05],[0,0,-1.1],[0,0,-2.]])
    request={'distance_trend':'receding','distance_trend_windows_s':[[0.,2.]],'minimum_window_distance_change_m':.26}
    assert not matches_distance_request(points,times,[0,1.3,0],request)
    request['distance_trend_windows_s']=[[2.,3.]]
    assert matches_distance_request(points,times,[0,1.3,0],request)
    request['distance_trend']='approaching'
    assert not matches_distance_request(points,times,[0,1.3,0],request)


@pytest.mark.fast_unit
def test_new_whole_utterances_cannot_change_or_fake_the_recorded_voice():
    from avengine.timeline.v7_phase2_sources import _validate_recorded_speech_voice
    original={'speaker_id':'same_admitted_voice','gender':'F'}
    record={'speaker_id':original['speaker_id'],'gender':'F','source_type':'recorded',
            'transcript':'A different complete sentence.','sound_class':'speech_playback'}
    _validate_recorded_speech_voice(record,original)
    with pytest.raises(ValueError,match='same admitted'):
        _validate_recorded_speech_voice({**record,'speaker_id':'another_voice'},original)
    with pytest.raises(ValueError,match='same admitted'):
        _validate_recorded_speech_voice({**record,'source_type':'author_tts'},original)
