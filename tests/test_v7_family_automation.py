"""Unattended construction uses legal registries, not retained scene recipes."""
import copy,json,os,time
from pathlib import Path
import pytest
from avengine.timeline.v7_family_constructors import construct_family_scene,choose_authored
from avengine.timeline.v7_unattended_resources import ResourcePolicy,admission,runtime_resource_view
pytestmark=pytest.mark.fast_unit

@pytest.fixture
def inputs():
    rooms=[];houses={}
    for i in range(2):
        key='new_'+str(i);rid='room_'+str(i);polygon={'type':'Polygon','coordinates':[[[-5,-5],[5,-5],[5,5],[-5,5],[-5,-5]]]}
        rooms.append({'key':key+'::'+rid,'house':key,'room_id':rid,'family':'hm3d','categories':['standing_human'],'listener_m':None})
        houses[key]={'house':key,'split':'train','acoustic_status':'pass','checked_room_ids':[rid],
            'checked_rooms':[{'room_label':rid,'floor_polygon':polygon,'placement':{'camera_m':[i*10,1.5,0],
                'source_1_m':[i*10-1,1.3,-2],'source_2_m':[i*10+1,1.3,-2]}}]}
    assets=[{'asset_id':g+'_'+color,'kind':'human','controlled_shirt':True,'color':color,
        'visual_gender':g,'entity_class':'articulated_human','species_id':'human','walk_allowed':True,
        'allowed_classes':['speech_playback','laughter','coughing'],'attachment_surface':'floor',
        'dialogue_voice_binding_eligible':True}for color in ('blue','yellow','green')for g in ('female','male')]
    sounds=[]
    for scenario in range(3):
        for role in ('a','b'):
            for gender in ('female','male'):
                for voice in (0,1):
                    sid=f'{scenario}_{role}_{gender}_{voice}'
                    sounds.append({'sound_asset_id':sid,'scenario_id':str(scenario),'role_id':role,
                        'transcript':f'complete sentence {scenario} {role}',
                        'synthesis_readback':{'duration_s':2.,'sample_count':32000},
                        'voice_reference':{'speaker_id':gender+str(voice),'gender':gender}})
    contract={'schema':'v7-production-inputs-v1','test_houses':[],'rooms':rooms,'animal_motion':{},'policy':{}}
    return rooms,houses,assets,{'sounds':sounds},contract

@pytest.mark.parametrize('family',['QA-10','QA-28','V7-CLASS34','QA-01','QA-03','QA-19'])
def test_any_legal_room_retains_required_roles_and_real_source_kinds(inputs,family):
    rooms,houses,assets,semantic,contract=content_inputs(inputs)if family=='QA-28'else inputs
    for room in rooms:
        result=construct_family_scene(room,houses[room['house']],{},assets,[{'sound_class':'laughter'},{'sound_class':'coughing'}],semantic,contract,{},scene_id='new_scene',seed=43,family=family)
        row=result['row'];assert row['house']==room['house']and row['room_id']==room['room_id']
        assert len(row['asset_ids'])==3 and len(set(row['asset_ids']))==3
        assert len(row['silent_asset_ids'])==(0 if family=='QA-28'else 1)
        assert row['automatic_construction']['scene_template_files_read']==[]
        assert 'native_attachment_supports'not in row and 'sampler_template_source'not in row
        assert all(len(kinds)==1 for kinds in row['source_classes'].values())
        if family=='QA-10':
            assert len(row['native_occlusion_relations'])==1
            relation=row['native_occlusion_relations'][0]
            assert relation['occluder_asset_id']in row['silent_asset_ids']
            assert relation['target_asset_id']not in row['silent_asset_ids']
            assert relation['minimum_stable_seconds']==.5 and relation['minimum_occluded_fraction']==.25
        if family=='QA-28':
            for angle in row['semantic_azimuth_targets_deg'].values():
                margin=(angle+22.5)%45;assert min(margin,45-margin)>=15
        if family in {'QA-21','V7-CLASS34'}:assert row['category_focus_class']=='laughter'
        if family in {'QA-03','QA-19'}:assert row['semantic_start_offsets_s'][1]>=2.5


def test_content_quartet_has_only_whole_authored_sources_and_fixed_real_voices(inputs):
    rooms,houses,assets,semantic,contract=content_inputs(inputs);room=rooms[0]
    result=construct_family_scene(room,houses[room['house']],{},assets,[],semantic,contract,{},scene_id='quad',seed=18,family='V7-CONTENT-BIND')
    row=result['row'];assert result['profile']=='registered'
    assert row['members']==['v0_a0','v0_a1','v1_a0','v1_a1']
    assert len(row['production_scenario']['asset_ids'])==4
    lookup={r['sound_asset_id']:r for r in semantic['sounds']}
    epochs=row['production_scenario']['semantic_epoch_sound_ids']
    assert len({lookup[ids[0]]['scenario_id']for ids in epochs})==3
    voices=[[lookup[sid]['voice_reference']['speaker_id']for sid in ids]for ids in epochs]
    assert voices[0]==voices[1]==voices[2]and len(set(voices[0]))==3
    assert row['production_scenario']['environment_sound_classes']==[]



def test_original_room_legality_is_not_widened_to_make_a_constructor_fit(inputs):
    rooms,houses,assets,semantic,contract=inputs
    wrong=copy.deepcopy(rooms[0]);wrong['categories']=[]
    with pytest.raises(ValueError,match='three distinct registered'):
        construct_family_scene(wrong,houses[wrong['house']],{},assets,[],semantic,contract,{},scene_id='bad',seed=4,family='QA-10')


def test_bounded_retry_changes_positions_orientations_and_individuals_without_dropping_roles(inputs):
    from avengine.capture.v7_layout_candidates import order_support_candidates,facing_yaw
    rooms,houses,assets,semantic,contract=inputs;room=rooms[0]
    first=construct_family_scene(room,houses[room['house']],{},assets,[],semantic,contract,{},scene_id='a',seed=12,family='QA-10')['row']
    retry=construct_family_scene(room,houses[room['house']],{},assets,[],semantic,contract,{},scene_id='b',seed=12,family='QA-10',attempt=4)['row']
    assert len(first['asset_ids'])==len(retry['asset_ids'])==3
    candidates=list(range(80));scene={'automatic_construction':{'candidate_index':4}}
    ordered=order_support_candidates(scene,candidates);assert set(ordered)==set(candidates)and ordered!=candidates
    baseline=facing_yaw({},assets[0],[0,0,0],[1,0,1])
    changed=facing_yaw({'automatic_construction':{'candidate_index':1}},assets[0],[0,0,0],[1,0,1]);assert changed!=baseline


def test_resource_expiry_and_pressure_only_change_admission():
    policy=ResourcePolicy();expiry=policy.temporary_minimum_free_until_unix
    assert admission(policy,available=40_000_000_000,free=46_000_000_000,now=expiry-1)['status']=='admitted'
    assert admission(policy,available=40_000_000_000,free=46_000_000_000,now=expiry)['status']=='paused_before_new_job'
    assert admission(policy,available=29_000_000_000,free=160_000_000_000,now=expiry-1)['status']=='paused_before_new_job'
    assert admission(policy,available=40_000_000_000,free=160_000_000_000,own_pss=41_000_000_000)['status']=='paused_before_new_job'
    view=runtime_resource_view({},policy,list(range(16)));assert view['fresh_rir_workers']+1<=16


def test_passed_feedback_deduplicates_imports_and_never_counts_failed_or_requested_quota(tmp_path):
    from avengine.timeline.v7_automatic_feedback import from_passed_exports
    roots=[tmp_path/'a',tmp_path/'b']
    for root in roots:
        (root/'evidence/jobs').mkdir(parents=True);(root/'export/parent_rows').mkdir(parents=True)
        (root/'evidence/jobs/000.json').write_text(json.dumps({'status':'passed','house':'already_seen'}))
        (root/'export/parent_rows/000.jsonl').write_text(json.dumps({'independent_parent_scene_id':'p','logical_question_id':'q','qa_id':'V7-CLASS34','canonical_answer':'laughter'})+'\n')
        (root/'evidence/jobs/001.json').write_text(json.dumps({'status':'failed','house':'failed_seen','requested_questions':99}))
    state=from_passed_exports(roots);assert state['logical_questions']==1
    assert state['class34_positive_counts']=={'laughter':1}
    assert state['family_counts']=={'V7-CLASS34':1}


def test_global_entry_refuses_handwritten_scene_fields(tmp_path):
    from tools.timeline.generate_v7_demand_scenes import load_global
    config=tmp_path/'bad.json';config.write_text(json.dumps({'schema':'v7-global-family-runtime-v1','parents':[{'position':[1,2,3]}]}))
    with pytest.raises(ValueError,match='no scene configuration'):load_global(config)


def test_new_failed_copy_cleanup_preserves_old_files_all_receipts_and_sealed_checkpoints(tmp_path):
    from tools.timeline.generate_v7_demand_scenes import cleanup_new_failed
    native=tmp_path/'native/parents/failure';native.mkdir(parents=True)
    recent=native/'new.npy';old=native/'old.wav';recent.write_bytes(bytes(1_000_001));old.write_bytes(bytes(1_000_001))
    now=time.time();os.utime(old,(now-100,now-100));receipt=tmp_path/'automatic_result.json';receipt.write_text('{"status":"failed"}')
    log=tmp_path/'worker.log';log.write_text('explicit failure saved')
    result={'status':'failed','automatic_started_unix':now-1}
    cleanup_new_failed(tmp_path,result,now-2,log)
    assert recent.exists()and old.exists()and receipt.exists()and log.exists()
    audit=json.loads((tmp_path/'failed_large_copy_cleanup.json').read_text());assert audit['removed_bytes']==0 and audit['retained_bytes']==1_000_001
    recent.write_bytes(bytes(1_000_001));(native/'integration_seal_completed.json').write_text('{}')
    cleanup_new_failed(tmp_path,result,now-2,log);assert recent.exists()


def test_whole_source_clock_keeps_waveform_binding_and_separates_taxonomy_from_authored_speech(monkeypatch):
    from avengine.timeline.v7_family_clock import build_constructed_scene
    from avengine.timeline import v7_phase2_sources
    scene={'sources':{'voice1':{'mechanism':'human_live_recorded_laughter'},'voice2':{'mechanism':'human_live_recorded'},'voice3':{'mechanism':'visible_silent_bystander'}},
        'events':[{'event_id':'laugh0','endpoint_id':'voice1','start_tick':5,'end_tick':25},
                  {'event_id':'tts0','endpoint_id':'voice2','start_tick':0,'end_tick':20}]}
    program={'records':[{'sound_asset_id':'laugh','source_type':'recorded','header':{'sample_count':64000,'sample_rate_hz':16000}},
        {'sound_asset_id':'tts','source_type':'author_tts','header':{'sample_count':32000,'sample_rate_hz':16000}}],
        'events':[{'event_id':'laugh0','sound_asset_id':'laugh','crop_start_sample':1234,'crop_end_sample':33234},
                  {'event_id':'tts0','sound_asset_id':'tts','crop_start_sample':0,'crop_end_sample':32000}]}
    assets=[{'asset_id':'laugher','endpoint_id':'voice1'},{'asset_id':'author','endpoint_id':'voice2'},{'asset_id':'quiet','endpoint_id':'voice3'}]
    monkeypatch.setattr(v7_phase2_sources,'build_phase2_scene',lambda *a:(None,copy.deepcopy(scene),copy.deepcopy(program),assets))
    row={'scene_id':'candidate','independent_parent_scene_id':'parent','automatic_construction':{'candidate_index':0},
        'phase2_authored_speech_schedule_s':{'author':[0.,10.,20.]},'automatic_taxonomy_clock':{'asset_id':'laugher'}}
    _,actual,bindings,kept=build_constructed_scene(row,{},[],[],{})
    assert set(actual['sources'])==set(scene['sources'])and kept==assets
    author=[e for e in actual['events']if e['endpoint_id']=='voice2'];laugh=[e for e in actual['events']if e['endpoint_id']=='voice1']
    assert [(e['start_tick'],e['end_tick'])for e in author]==[(0,20),(100,120),(200,220)]
    assert [(e['start_tick'],e['end_tick'])for e in laugh]==[(25,45),(125,145),(225,245)]
    for b in bindings['events']:
        assert b['crop_start_sample']==(0 if b['sound_asset_id']=='tts'else 1234)
        assert b['crop_end_sample']==(32000 if b['sound_asset_id']=='tts'else 33234)
    assert actual['independent_parent_scene_id']=='parent'


def test_content_constructor_changes_real_body_gender_instead_of_reducing_complete_epochs(inputs):
    rooms,houses,assets,semantic,contract=content_inputs(inputs);room=rooms[0]
    restricted={'sounds':[s for s in semantic['sounds']if s['voice_reference']['gender']=='female'or s['scenario_id']=='0']}
    for a in assets:a['body_identity_id']=a['visual_gender']+'_registered_body'
    plan=construct_family_scene(room,houses[room['house']],{},assets,[],restricted,contract,{},scene_id='auto',seed=1,family='V7-CONTENT-BIND')
    ids=plan['row']['production_scenario']['asset_ids'];lookup={a['asset_id']:a for a in assets}
    assert all(lookup[aid]['visual_gender']=='female'for aid in ids[:2])
    assert len({lookup[aid]['body_identity_id']for aid in ids[:2]})==1
    assert len(plan['row']['production_scenario']['semantic_epoch_sound_ids'])==3



def test_public_new_room_entry_never_reads_an_agent_scene_configuration(tmp_path,monkeypatch,inputs):
    from tools.timeline import generate_v7_demand_scenes as entry
    from avengine.timeline import v7_family_clock
    rooms,houses,assets,semantic,contract=content_inputs(inputs);root=tmp_path/'unattended';hand=root/'inputs/planned/000.json'
    hand.parent.mkdir(parents=True);hand.write_text('{"native_attachment_supports":"forbidden old manual choice"}')
    config=tmp_path/'global.json';config.write_text(json.dumps({'schema':'v7-global-family-runtime-v1','authorized_output_root':str(tmp_path),'feedback_roots':[]}))
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        if path==hand:raise AssertionError('read a handwritten scene configuration')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    monkeypatch.setattr(entry,'data',lambda cfg:(contract,assets,{}, {'records':[]},semantic,houses))
    monkeypatch.setattr(entry,'commit',lambda:'f'*40)
    monkeypatch.setattr(entry,'clean_source',lambda:True)
    monkeypatch.setattr(v7_family_clock,'build_constructed_scene',lambda *args:({}, {'events':[],'sources':{}},{'records':[]},[]))
    monkeypatch.setattr(entry.os,'environ',dict(os.environ))
    monkeypatch.setattr(entry.sys,'argv',['automatic','--config',str(config),'--output',str(root),'--plan-only','--parents','2','--families','QA-10','QA-28'])
    entry.main();result=json.loads((root/'AUTOMATION_RESULT.json').read_text())
    assert result['native_passed_parents']==0 and result['handwritten_scene_configuration_inputs_read']==[]
    assert len(list((root/'proposals').glob('*.json')))==2


def test_frozen_global_registry_fingerprints_notice_in_place_changes(tmp_path):
    from tools.timeline.generate_v7_demand_scenes import global_fingerprints
    path=tmp_path/'assets.json';path.write_text('{"assets":[]}')
    cfg={'assets':str(path)};before=global_fingerprints(cfg)
    path.write_text('{"assets":["a new admitted body"]}')
    assert global_fingerprints(cfg)!=before


def test_production_write_guard_uses_the_same_exact_floor_and_never_aborts_admitted_work(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from avengine.timeline import v7_storage_budget as storage
    from avengine.timeline.v7_unattended_resources import configure_worker
    monkeypatch.setenv(storage.ENV_KEY,'')
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=40_000_000_000))
    policy=ResourcePolicy(temporary_minimum_free_until_unix=time.time()+3600)
    budget=configure_worker(tmp_path,policy,[0,1]);assert budget['admission_only']is True
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=39_000_000_000))
    # A different user's disk growth must not abort a parent already admitted.
    record=storage.check_write(tmp_path/'actual_new_native_stage',expected_bytes=8_000_000_000)
    assert record['resource_limits_enforced_at']=='parent_admission'
    assert record['free_bytes']==39_000_000_000


def test_automatic_resource_defaults_do_not_grant_a_dated_disk_exception():
    from avengine.timeline.v7_unattended_resources import ResourcePolicy,launch_disk_policy
    policy=ResourcePolicy()
    assert policy.disk_floor()==100_000_000_000 and launch_disk_policy(policy)=={}


def test_frozen_archive_detects_changed_code_and_has_a_stable_commit(tmp_path):
    import hashlib
    from avengine.timeline.v7_frozen_code import source_commit,verify_source,MARKER
    code=tmp_path/'worker.py';code.write_text('print(1)')
    (tmp_path/MARKER).write_text(json.dumps({'schema':'avengine-committed-source-snapshot-v1','commit':'a'*40,
        'files':{'worker.py':hashlib.sha256(code.read_bytes()).hexdigest()}}))
    assert source_commit(tmp_path)=='a'*40 and verify_source(tmp_path)
    code.write_text('print(2)');assert not verify_source(tmp_path)


def test_whole_recorded_triplet_keeps_one_voice_distinct_transcripts_and_complete_short_clips():
    from avengine.timeline.v7_family_constructors import whole_recorded_triplet
    records=[]
    for i,duration in enumerate((1.,1.1,1.2,2.2)):
        records.append({'dataset':'VCTK-Corpus 0.92','source_type':'recorded','gender':'F','legacy_library_split':'train',
            'speaker_id':'actual_voice','transcript':'complete spoken sentence '+str(i),'sound_asset_id':str(i),
            'duration_s':duration,'normalized_path':'read_only_full_source_'+str(i)+'.wav'})
    records.append({**records[0],'sound_asset_id':'duplicate_long','duration_s':2.3})
    selected=whole_recorded_triplet(records,'F',12)
    assert [r['duration_s']for r in selected]==[1.,1.1,1.2]
    assert len({r['transcript']for r in selected})==3 and len({r['speaker_id']for r in selected})==1
    assert all(any(r is original for original in records)for r in selected)
    assert all('crop_start_sample'not in r for r in selected)


def test_shared_task_cache_remains_in_scope_without_rechecking_an_admitted_parent(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from avengine.timeline import v7_storage_budget as storage
    from avengine.timeline.v7_unattended_resources import configure_worker,configure_parent_storage
    monkeypatch.setenv(storage.ENV_KEY,'')
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=80_000_000_000))
    root=tmp_path/'run';root.mkdir();policy=ResourcePolicy(temporary_minimum_free_until_unix=time.time()+3600)
    configure_worker(root,policy,[0,1])
    # A parent already admitted keeps going after another task consumes space.
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=39_000_000_000))
    attempt=root/'attempts/one';attempt.mkdir(parents=True)
    configured=configure_parent_storage({'storage_budget_root':str(root),'storage_admission_prechecked':True},attempt)
    assert configured['root']==str(root)
    assert storage.check_write(root/'cache/materials',expected_bytes=64_000_000)['resource_limits_enforced_at']=='parent_admission'
    with pytest.raises(storage.StorageBudgetExceeded,match='escaped'):
        storage.check_write(tmp_path/'outside_the_task',expected_bytes=0)


def test_taxonomy_rotation_constructs_a_new_capable_entity_and_keeps_every_human(inputs):
    rooms,houses,assets,semantic,contract=inputs;room=rooms[0];room['categories'].append('floor_object')
    phone={'asset_id':'registered_phone','kind':'object','entity_class':'device','attachment_surface':'floor',
        'allowed_classes':['telephone_bell_ringing']}
    catalog=[*assets,phone];records=[{'sound_class':'laughter'},{'sound_class':'telephone_bell_ringing'}]
    plan=construct_family_scene(room,houses[room['house']],{'class34_positive_counts':{'laughter':3}},catalog,records,
        semantic,contract,{},scene_id='new',seed=12,family='V7-CLASS34')
    row=plan['row'];assert len(row['asset_ids'])==4 and 'registered_phone'in row['asset_ids']
    assert len([aid for aid in row['asset_ids']if aid!='registered_phone'])==3
    assert row['category_focus_class']=='telephone_bell_ringing'
    assert len(row['silent_asset_ids'])==2 and len(row['semantic_human_asset_ids'])==1
    assert row['automatic_construction']['required_role_count']==4


def test_path_guard_reports_resolved_actual_path_and_declared_root(tmp_path,monkeypatch):
    from avengine.timeline import v7_storage_budget as storage
    root=tmp_path/'own_run';outside=tmp_path/'sibling_cache/materials'
    monkeypatch.setenv(storage.ENV_KEY,json.dumps({'root':str(root)}))
    with pytest.raises(storage.StorageBudgetExceeded)as caught:
        storage.check_write(outside,expected_bytes=64_000_000)
    message=str(caught.value)
    assert 'path='+str(outside.resolve())in message
    assert 'root='+str(root.resolve())in message
    assert not outside.exists()


def test_runtime_relocates_the_exact_engine_material_cache_key_into_the_declared_root(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from avengine.timeline import v7_storage_budget as storage
    from avengine.timeline.v7_unattended_resources import configure_worker
    root=tmp_path/'own_run';root.mkdir();outside=tmp_path/'shared_sibling_cache'
    inherited={'source_material_cache_directory':str(outside),'source_material_cache':'ignored_wrong_alias'}
    monkeypatch.setenv(storage.ENV_KEY,'')
    policy=ResourcePolicy(temporary_minimum_free_until_unix=time.time()+3600)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=80_000_000_000))
    configure_worker(root,policy,[0,1])
    view=runtime_resource_view(inherited,policy,[0,1],cache_root=root)
    assert view['source_material_cache_directory']==str(root/'cache/source_materials')
    assert 'source_material_cache'not in view
    assert inherited['source_material_cache_directory']==str(outside)
    assert storage.check_write(view['source_material_cache_directory'],expected_bytes=64_000_000)
    with pytest.raises(storage.StorageBudgetExceeded):storage.check_write(outside)


def test_repeated_cause_counter_ignores_attempt_paths_and_stops_on_second_failure():
    from tools.timeline.generate_v7_demand_scenes import failure_stop
    history={}
    first={'status':'failed','error_type':'RuntimeError','reason':'native mask missing at /task/parent000/mask.npy'}
    second={**first,'reason':'native mask missing at /task/parent001/mask.npy'}
    assert failure_stop(first,history)is None
    stopped=failure_stop(second,history)
    assert stopped['same_cause_occurrences']==2
    assert stopped['status']=='stopped_before_next_candidate'


@pytest.mark.parametrize('error_type,expected_dispatches',[('StorageBudgetExceeded',1),('RuntimeError',2)])
def test_public_coordinator_stops_the_entire_batch_instead_of_only_the_current_parent(tmp_path,monkeypatch,inputs,error_type,expected_dispatches):
    from tools.timeline import generate_v7_demand_scenes as entry
    from avengine.timeline import v7_unattended_resources as resources
    rooms,houses,assets,semantic,contract=inputs;root=tmp_path/'automatic'
    config=tmp_path/'global.json';config.write_text(json.dumps({'schema':'v7-global-family-runtime-v1',
        'authorized_output_root':str(tmp_path),'feedback_roots':[]}))
    monkeypatch.setattr(entry,'data',lambda cfg:(contract,assets,{}, {'records':[]},semantic,houses))
    monkeypatch.setattr(entry,'commit',lambda:'f'*40)
    monkeypatch.setattr(entry,'child_env',lambda cfg,root:dict(os.environ))
    monkeypatch.setattr(resources,'admission',lambda policy:{'status':'admitted'})
    monkeypatch.setattr(resources,'idle_cpu_order',lambda:list(range(16)))
    monkeypatch.setattr(entry.os,'environ',dict(os.environ))
    dispatched=[]
    class FailedWorker:
        def __init__(self,command,**kwargs):
            attempt=Path(command[command.index('--output')+1]);dispatched.append(attempt)
            self.pid=10000+len(dispatched);self.returncode=2
            entry.write(attempt/'automatic_result.json',{'status':'failed','error_type':error_type,
                'reason':'write escaped the declared own task output root: path=/old/cache; root=/new/task'if error_type=='StorageBudgetExceeded'else'original native layout rejected',
                'automatic_attempt_root':str(attempt),'scene_id':attempt.name,'automatic_started_unix':time.time()})
        def poll(self):return self.returncode
    monkeypatch.setattr(entry.subprocess,'Popen',FailedWorker)
    monkeypatch.setattr(entry.sys,'argv',['automatic','--config',str(config),'--output',str(root),
        '--parents','3','--maximum-candidates','1','--families','QA-10'])
    entry.main()
    assert len(dispatched)==expected_dispatches
    result=entry.read(root/'AUTOMATION_RESULT.json');stopped=entry.read(root/'AUTOMATION_STOP.json')
    assert result['stop_requested']is True and result['native_passed_parents']==0
    assert len(result['outcomes'])==expected_dispatches
    assert stopped['same_cause_occurrences']==expected_dispatches


@pytest.mark.parametrize('stop_during_admission',[False,True])
def test_graceful_stop_always_finalizes_receipts_and_never_dispatches_another_worker(tmp_path,monkeypatch,inputs,stop_during_admission):
    from tools.timeline import generate_v7_demand_scenes as entry
    from avengine.timeline import v7_unattended_resources as resources
    rooms,houses,assets,semantic,contract=inputs;root=tmp_path/'stopped'
    config=tmp_path/'global.json';config.write_text(json.dumps({'schema':'v7-global-family-runtime-v1',
        'authorized_output_root':str(tmp_path),'feedback_roots':[]}))
    monkeypatch.setattr(entry,'data',lambda cfg:(contract,assets,{}, {'records':[]},semantic,houses))
    monkeypatch.setattr(entry,'commit',lambda:'f'*40)
    monkeypatch.setattr(entry,'child_env',lambda cfg,root:dict(os.environ))
    monkeypatch.setattr(entry.os,'environ',dict(os.environ))
    monkeypatch.setattr(resources,'idle_cpu_order',lambda:list(range(16)))
    handlers={};dispatched=[]
    monkeypatch.setattr(entry.signal,'signal',lambda number,handler:handlers.update({number:handler}))
    monkeypatch.setattr(entry.time,'sleep',lambda duration:None)
    def admission(policy):
        if stop_during_admission:
            handlers[entry.signal.SIGTERM](entry.signal.SIGTERM,None)
            return {'status':'paused_before_new_job'}
        return {'status':'admitted'}
    monkeypatch.setattr(resources,'admission',admission)
    class FinishingWorker:
        def __init__(self,command,**kwargs):
            attempt=Path(command[command.index('--output')+1]);dispatched.append(attempt)
            self.pid=10001;self.returncode=2
            entry.write(attempt/'automatic_result.json',{'status':'failed','error_type':'RuntimeError',
                'reason':'native layout rejected','automatic_attempt_root':str(attempt),
                'scene_id':attempt.name,'automatic_started_unix':time.time()})
        def poll(self):
            handlers[entry.signal.SIGTERM](entry.signal.SIGTERM,None)
            return self.returncode
    monkeypatch.setattr(entry.subprocess,'Popen',FinishingWorker)
    monkeypatch.setattr(entry.sys,'argv',['automatic','--config',str(config),'--output',str(root),
        '--parents','3','--maximum-candidates','3','--families','QA-10'])
    entry.main();result=entry.read(root/'AUTOMATION_RESULT.json')
    assert result['stop_requested']is True
    assert len(dispatched)==(0 if stop_during_admission else 1)
    assert len(result['outcomes'])==len(dispatched)
    assert result['native_passed_parents']==0


def test_silent_human_visibility_proposal_uses_actual_body_without_changing_native_gates():
    import math
    from avengine.capture.v7_layout_candidates import silent_candidate_local_point
    vertices=[[-.3,0.,-.2],[.3,1.8,.2]];actor={'kind':'human','silent_decor':True}
    actual=silent_candidate_local_point({'automatic_construction':{}},actor,vertices)
    # Empty automatic metadata still denotes an ordinary legacy call.
    assert actual==[0.,.5,0.]
    actual=silent_candidate_local_point({'automatic_construction':{'candidate_index':0}},actor,vertices)
    assert actual==[0.,.9,0.]
    assert abs(math.degrees(math.atan2(actual[1]-1.5,1.5)))<24
    assert abs(math.degrees(math.atan2(.5-1.5,1.5)))>24
    assert silent_candidate_local_point({},actor,vertices)==[0.,.5,0.]
    assert silent_candidate_local_point({'automatic_construction':{'candidate_index':0}},{'silent_decor':False},vertices)==[0.,.5,0.]


def test_registered_quartet_selects_colors_supported_by_its_original_reviewer_without_banning_phase2_neutrals(inputs):
    from avengine.qa.v7_visual_admission import COLOR_HUE_DEGREES
    from avengine.timeline.v7_family_constructors import eligible_humans
    rooms,houses,assets,semantic,contract=content_inputs(inputs,add_people=False);room=rooms[0]
    for a in assets:a['body_identity_id']=a['visual_gender']+'_body'
    catalog=[*assets,*[{**a,'asset_id':a['visual_gender']+'_'+neutral,'color':neutral}
        for a in assets if a['color']=='blue'for neutral in ('black','white')]]
    lookup={a['asset_id']:a for a in catalog}
    assert {'black','white'}<={a['color']for a in eligible_humans(catalog,room,contract)}
    for seed in range(12):
        plan=construct_family_scene(room,houses[room['house']],{},catalog,[],semantic,contract,{},
            scene_id='new',seed=seed,family='V7-CONTENT-BIND')
        ids=plan['row']['production_scenario']['asset_ids']
        assert len(ids)==4 and len(plan['row']['members'])==4
        assert all(lookup[aid]['color']in COLOR_HUE_DEGREES for aid in ids[:2])



def test_placement_failure_signature_is_the_same_cause_for_different_registered_individuals():
    from tools.timeline.generate_v7_demand_scenes import failure_stop
    history={};first={'status':'failed','error_type':'ValueError',
        'reason':'no strict native floor placement for v7_female_adult_13_blue_v1'}
    assert failure_stop(first,history)is None
    stopped=failure_stop({**first,'reason':'no strict native floor placement for v7_male_adult_20_white_v1'},history)
    assert stopped['same_cause_occurrences']==2


def test_worker_failure_log_keeps_actual_path_declared_root_and_source_traceback(capsys):
    from tools.timeline.generate_v7_demand_scenes import log_attempt_failure
    log_attempt_failure({'status':'failed','error_type':'StorageBudgetExceeded',
        'reason':'write escaped the declared own task output root: path=/actual/cache; root=/declared/run',
        'traceback':'v7_storage_budget.py:74: check_write'})
    error=capsys.readouterr().err
    assert '/actual/cache'in error and '/declared/run'in error and 'v7_storage_budget.py:74'in error


def test_only_actual_footstep_sources_request_shoe_emitters():
    from avengine.capture.v7_phase2_native import requires_foot_emitters
    scene={'sources':{'voice1':{'sound_class':'speech_playback'},'voice2':{'silent_decor':True}}}
    assert not requires_foot_emitters(scene,'voice1')and not requires_foot_emitters(scene,'voice2')
    scene['sources']['voice3_foot_L']={'owner_endpoint_id':'voice3','emitter_part':'foot_L','sound_class':'footsteps'}
    assert requires_foot_emitters(scene,'voice3')and not requires_foot_emitters(scene,'voice1')


@pytest.mark.parametrize('has_footsteps',[False,True])
def test_speaking_walker_skips_unused_foot_gate_but_footstep_walker_keeps_unchanged_gates(monkeypatch,has_footsteps):
    import numpy as np
    from types import SimpleNamespace
    from avengine.capture.v7_phase2_native import NativePhase2Video
    from avengine.capture.v7_visible_native import NativeVisibleVideo
    from avengine.capture import v7_visible_mesh
    monkeypatch.setattr(NativeVisibleVideo,'_apply_motion_entry',lambda *args:{'body_to_world_m':np.eye(4).tolist()})
    # No original vertex satisfies the existing 50mm/2mm surface predicate.
    monkeypatch.setattr(v7_visible_mesh,'indexed_static_mesh',lambda path:SimpleNamespace(vertices=np.array([[0.,0.,0.]])))
    entry={'declaration':{'silent_decor':False},'phase2_required_foot_emitters':has_footsteps,
        'motion_episode':{'source_frames':[{'mouth_m':[0.,1.4,0.],'feet':{'L':{'sole_local_m':[0.,0.,0.]}}}],
            'pose_glbs':['registered_original_pose.glb']}}
    video=object.__new__(NativePhase2Video)
    if has_footsteps:
        # Still fails when no shoe vertex within 10 cm is >= 2 mm above the contact (owner 10-15 widened 5 -> 10 cm).
        with pytest.raises(ValueError,match='2mm not-buried gate'):video._apply_motion_entry(entry,0)
    else:
        row=video._apply_motion_entry(entry,0)
        assert row['emitter_local_m_frame']==[0.,1.4,0.]
        assert row['native_emitter_present']is True and row['attached_emitter_readbacks']==[]


@pytest.mark.parametrize('key',['recorded_program','program'])
def test_real_native_program_schema_drives_actual_authored_voice_rotation(key):
    from tools.timeline.generate_v7_demand_scenes import authored_voice_ids
    recipe={key:{'records':[{'source_type':'author_tts','voice_reference':{'speaker_id':'p225'}},
        {'source_type':'author_tts','voice_reference':{'speaker_id':'p234'}},
        {'source_type':'recorded','speaker_id':'p999'},
        {'source_type':'author_tts','voice_reference':{'speaker_id':'p225'}}]},
        'seed_row':{'requested_voice':'never_count_a_request'}}
    assert authored_voice_ids(recipe)==['p225','p234']


def test_owner_layout_keeps_three_people_one_metre_apart_distributed_and_with_varied_facing(inputs):
    import math
    rooms,houses,assets,semantic,contract=content_inputs(inputs);room=rooms[0]
    row=construct_family_scene(room,houses[room['house']],{},assets,[],semantic,contract,{},
        scene_id='new',seed=10,family='V7-CONTENT-BIND')['row']
    valid=row['measured_static_layout']['valid'];points=[v['position_m']for v in valid]
    assert len(points)==4 and len(row['members'])==4
    for i,p in enumerate(points):
        for q in points[i+1:]:assert math.hypot(p[0]-q[0],p[2]-q[2])>=1.
    assert len({round(v['body_yaw_deg'],2)for v in valid})>=2
    assert row['production_scenario']['camera_policy']=='hold'



def test_owner_spacing_checks_reserved_motion_track_and_human_animal_pairs():
    from avengine.capture.v7_social_layout import spacing_violation
    scene={'automatic_construction':{'family':'QA-16'},'listener_m':[0.,1.3,-4.]}
    human={'kind':'human','asset_id':'person'};animal={'kind':'animal','asset_id':'cat'}
    placed=[{'declaration':human,'motion_episode':{'world_roots_m':[[0,0,0],[1,0,0],[2,0,0]]}}]
    assert spacing_violation(scene,animal,[1,0,.8],placed)['required_m']==1.
    assert spacing_violation(scene,human,[1,0,1.1],placed)is None


def test_owner_direction_constructor_has_no_end_of_clip_camera_sweep(inputs):
    rooms,houses,assets,semantic,contract=content_inputs(inputs);room=rooms[0]
    row=construct_family_scene(room,houses[room['house']],{},assets,[],semantic,contract,{},
        scene_id='new',seed=10,family='QA-28')['row']
    assert len({k['yaw_deg']for k in row['camera_yaw_knots']})==1
    assert row['camera_yaw_knots'][-1]['time_s']==30.



def test_owner_invalidated_roots_never_supply_coverage_or_receive_new_jobs(tmp_path):
    from avengine.timeline.v7_automatic_feedback import from_passed_exports
    from avengine.timeline.v7_generation_scope import assert_new_generation_output
    old=tmp_path/'old';(old/'evidence/jobs').mkdir(parents=True)
    (old/'INVALID_20261015.md').write_text('owner: all old exports void')
    (old/'evidence/jobs/000.json').write_text('{"status":"passed","house":"old_house"}')
    assert from_passed_exports([old])['logical_questions']==0
    with pytest.raises(ValueError,match='owner-invalidated'):assert_new_generation_output(old/'another_attempt')
    assert_new_generation_output(tmp_path/'new_v2/attempt')


def test_generation_namespace_changes_both_scene_ids_and_deterministic_seeds(tmp_path):
    from avengine.timeline.v7_generation_scope import generation_scene_id
    from avengine.timeline.v7_family_constructors import stable_seed
    a=generation_scene_id(tmp_path/'old',tmp_path/'old/automatic_01',0)
    b=generation_scene_id(tmp_path/'new_v2',tmp_path/'new_v2/automatic_01',0)
    assert a!=b and stable_seed(a,'same_room')!=stable_seed(b,'same_room')


def test_owner_human_proposals_prefer_different_floor_regions_instead_of_one_front_row():
    import math
    from avengine.capture.v7_social_layout import social_proposal_score
    scene={'automatic_construction':{'family':'QA-01'}};a={'kind':'human','native_instance_id':1}
    target=[-2.8*math.sin(math.radians(28)),1.5,-2.8*math.cos(math.radians(28))]
    assert social_proposal_score(scene,a,target,[0,1.5,0],[0,0,-1])<social_proposal_score(scene,a,[0,1.5,-2.8],[0,1.5,0],[0,0,-1])


def test_layout_retry_changes_room_only_inside_the_original_legal_cell(inputs):
    from tools.timeline.generate_v7_demand_scenes import room_for_candidate
    rooms,*_=inputs;pairs=list(enumerate(rooms))
    assert room_for_candidate(rooms[0],pairs,0)[1]['room_id']==rooms[0]['room_id']
    assert room_for_candidate(rooms[0],pairs,1)[1]['room_id']==rooms[1]['room_id']
    incompatible={**rooms[1],'room_id':'wrong_family','family':'mp3d'}
    assert room_for_candidate(rooms[0],[pairs[0],(2,incompatible)],1)[1]['room_id']==rooms[0]['room_id']


def test_new_post_sound_construction_defaults_request_owner_legal_human_speed():
    import inspect
    from avengine.timeline.v7_conditioned_post_motion import solve_post_sound_window,plan_post_sound_motion
    for function in (solve_post_sound_window,plan_post_sound_motion):
        assert .8<=inspect.signature(function).parameters['speed_m_s'].default<=1.4


@pytest.mark.parametrize('offscreen_people',[1,2])
def test_offscreen_constructor_keeps_every_person_distinct_registered_shirts_and_whole_sounds(inputs,offscreen_people):
    from avengine.timeline.v7_offscreen_construction import construct_offscreen_scene
    rooms,houses,assets,semantic,contract=inputs;room=rooms[0]
    for color in ('white','black'):
        for gender in ('female','male'):
            assets.append({**copy.deepcopy(assets[0]),'asset_id':gender+'_'+color,'color':color,'visual_gender':gender})
    records=[{'sound_asset_id':f'whole_{gender}_{i}','sound_class':'speech_playback',
        'dataset':'VCTK-Corpus 0.92','legacy_library_split':'train','source_type':'recorded',
        'gender':gender,'speaker_id':f'external_{gender}_{i}','transcript':f'A complete fresh sentence {i}.',
        'duration_s':1.}for gender in ('M','F')for i in range(2)]
    row=construct_offscreen_scene(room,houses[room['house']],{},assets,records,semantic,contract,{},
        scene_id='new_offscreen',seed=51,offscreen_people=offscreen_people)['row']
    assert len(row['asset_ids'])==2+offscreen_people
    colors={a['asset_id']:a['color']for a in assets}
    assert len({colors[aid]for aid in row['asset_ids']})==2+offscreen_people
    assert not row['silent_asset_ids']
    assert len(row['native_view_roles'])==offscreen_people
    assert len({knot['yaw_deg']for knot in row['camera_yaw_knots']})==1
    assert row['offscreen_construction']['audio_only_points']==0
    for aid in row['native_view_roles']:
        assert row['source_classes'][aid]==['speech_playback']
        assert row['phase2_recorded_sound_ids'][aid]
        assert len(row['phase2_recorded_sound_ids'][aid])==len(row['phase2_recorded_speech_schedule_s'][aid])
    assert all(len(kinds)==1 for kinds in row['source_classes'].values())


def test_offscreen_speaks_first_when_the_actual_selected_answer_branch_is_missing(inputs):
    from avengine.timeline.v7_offscreen_construction import construct_offscreen_scene
    rooms,houses,assets,semantic,contract=inputs;room=rooms[0]
    for gender in ('female','male'):
        assets.append({**copy.deepcopy(assets[0]),'asset_id':gender+'_white','color':'white','visual_gender':gender})
    records=[{'sound_asset_id':'outside_'+g,'sound_class':'speech_playback','dataset':'VCTK-Corpus 0.92',
        'legacy_library_split':'train','source_type':'recorded','gender':g,'speaker_id':'external_'+g,
        'transcript':'This is a complete short sentence.','duration_s':1.}for g in ('M','F')]
    row=construct_offscreen_scene(room,houses[room['house']],{},assets,records,semantic,contract,{},
        scene_id='positive',seed=13,offscreen_people=1)['row']
    outside=row['offscreen_construction']['offscreen_asset_ids'][0]
    assert row['phase2_recorded_speech_schedule_s'][outside][0]==0.
    assert min(row['phase2_authored_speech_schedule_s'][aid][0]for aid in row['semantic_human_asset_ids'])>0.


def content_inputs(inputs,add_people=True):
    rooms,houses,assets,semantic,contract=copy.deepcopy(inputs)
    if add_people:
        for color in ('white','black'):
            for gender in ('female','male'):
                if not any(a['asset_id']==gender+'_'+color for a in assets):
                    assets.append({**copy.deepcopy(assets[0]),'asset_id':gender+'_'+color,'color':color,'visual_gender':gender})
    for scenario in ('0','1','2'):
        for gender in ('female','male'):
            semantic['sounds'].append({'sound_asset_id':scenario+'_c_'+gender,'scenario_id':scenario,'role_id':'c',
                'transcript':'A third complete sentence for '+scenario,'synthesis_readback':{'duration_s':2.,'sample_count':32000},
                'voice_reference':{'speaker_id':gender+'2','gender':gender}})
    return rooms,houses,assets,semantic,contract


def test_qa21_constructs_a_real_fixed_domain_emitter_and_retains_the_three_people(inputs):
    rooms,houses,assets,semantic,contract=inputs
    assets.append({'asset_id':'registered_alarm','kind':'object','entity_class':'static_object',
        'attachment_surface':'floor','object_type':'alarm_clock','color':'white','allowed_classes':['alarm_beep']})
    for room in rooms:
        room['categories'].append('floor_object')
        row=construct_family_scene(room,houses[room['house']],{},assets,
            [{'sound_class':'alarm_beep'},{'sound_class':'laughter'}],semantic,contract,{},
            scene_id='new_scene',seed=43,family='QA-21')['row']
        assert len(row['asset_ids'])==4 and 'registered_alarm'in row['asset_ids']
        assert row['source_classes']['registered_alarm']==['alarm_beep']
        assert row['category_focus_class']=='alarm_beep'
        assert len(row['silent_asset_ids'])==2
        assert row['scene_demand']['class34_construction']['required_answer_domain']
        assert row['native_view_roles']['registered_alarm']['view']=='outside'
        assert all(aid in row['asset_ids']for aid in row['automatic_taxonomy_clock']['real_offscreen_source_asset_ids'])
        assert len(row['silent_asset_ids'])==2


def test_qa21_keeps_request_when_cell_lacks_a_real_fixed_domain_source(inputs):
    rooms,houses,assets,semantic,contract=inputs
    with pytest.raises(ValueError,match='no registered source'):
        construct_family_scene(rooms[0],houses[rooms[0]['house']],{},assets,
            [{'sound_class':'laughter'}],semantic,contract,{},scene_id='new_scene',seed=43,family='QA-21')


def test_room_capacity_rank_uses_original_floor_and_retry_preserves_needed_categories(inputs):
    from avengine.timeline.v7_family_room_capacity import ranked_room_candidates
    from tools.timeline.generate_v7_demand_scenes import room_for_candidate
    rooms,houses,assets,semantic,contract=copy.deepcopy(inputs)
    houses[rooms[1]['house']]['checked_rooms'][0]['floor_polygon']={'type':'Polygon','coordinates':[[[-9,-9],[9,-9],[9,9],[-9,9],[-9,-9]]]}
    rooms[1]['categories'].append('floor_object')
    pairs=ranked_room_candidates(list(enumerate(rooms)),houses)
    assert pairs[0][1]['room_id']==rooms[1]['room_id']
    assert room_for_candidate(rooms[0],pairs,1,required_categories={'standing_human'})[1]['room_id']==rooms[1]['room_id']
    assert room_for_candidate(rooms[0],pairs,1,required_categories={'standing_human','floor_object'})[1]['room_id']==rooms[1]['room_id']


def test_native_failure_stopping_uses_concrete_cause_instead_of_asset_or_changing_counts():
    from tools.timeline.generate_v7_demand_scenes import failure_signature
    a={'status':'failed','error_type':'ValueError','reason':'no strict native floor placement for v7_a [placement_cause=attachment_rays]'}
    b={**a,'reason':'no strict native floor placement for v7_b [placement_cause=attachment_rays]'}
    c={**a,'reason':'no strict native floor placement for v7_c [placement_cause=owner_minimum_camera_distance_1_5m]'}
    assert failure_signature(a)==failure_signature(b)
    assert failure_signature(a)!=failure_signature(c)
