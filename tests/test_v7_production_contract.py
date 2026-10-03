from copy import deepcopy
from collections import Counter
import numpy as np
import pytest

pytestmark = pytest.mark.fast_unit

from avengine.timeline.v7_production_contract import (
    seed_rows,validate_seed_tables,derived_seed,question_allocation,validate_question_counts,
    audit_counterfactuals,audit_family_floor,TRAIN_HOUSES,DEV_HOUSES,
)
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy

def houses():
    return [{'house':h,'split':split,'acoustic_status':'pass','checked_room_ids':['R1','R4']} for split,ids in
            [('train',TRAIN_HOUSES),('development',DEV_HOUSES)] for h in ids]

def test_small_prefix_is_identical_and_unadmitted_full_houses_remain_pending():
    train=seed_rows('train',400,houses());dev=seed_rows('dev',100,houses());full=seed_rows('train',7000,houses())
    result=validate_seed_tables(train,dev,full)
    assert result['prefix_equal'] and result['full_train_unassigned_rows']==6600
    assert all(x['house'] is None and x['assignment_status']=='awaiting_admitted_house' for x in full[400:])
    assert not set(TRAIN_HOUSES).intersection(DEV_HOUSES)
    assert all(t%2==0 and 0<=t<=200 for x in train+dev for t in x['window_start_ticks'])
    changed=deepcopy(train);changed[10]['scene_seed']+=1
    with pytest.raises(ValueError,match='prefix'):validate_seed_tables(changed,dev,full)

def test_seed_purposes_and_splits_do_not_reuse_random_streams():
    seeds={derived_seed(s,i,p) for s in ('train','dev','test') for i in range(100)
           for p in ('scene','appearance','pose','audio','camera','window')}
    assert len(seeds)==1800

def test_house_gate_and_room_scope_cannot_be_bypassed():
    h=houses();h[0]['acoustic_status']='review'
    with pytest.raises(ValueError,match='pass'):seed_rows('train',1,h)
    h=houses();h[0]['checked_room_ids']=[]
    with pytest.raises(ValueError,match='scope'):seed_rows('train',1,h)

@pytest.mark.parametrize('count',[12,14])
def test_modality_budget_is_exact_over_twenty_windows(count):
    totals=Counter()
    for i in range(20):
        allocation=question_allocation(i,count);assert sum(allocation.values())==count;totals.update(allocation)
    assert dict(totals)=={'vision':6*count,'audio':5*count,'audiovisual':9*count}

def question_rows(count=14):
    public=[];private=[]
    for i in range(count):
        logical=f'logical_{i}'
        for form in ('open','mcq'):
            qid=f'{logical}_{form}'
            public.append({'question_id':qid,'logical_question_id':logical,'qa_id':'QA-14',
                'modality':'vision','question_zh':f'第{i}秒哪个人更近？','form':form})
            private.append({'question_id':qid,'canonical_answer':'blue'})
    return public,private

def test_forms_do_not_inflate_logical_item_count():
    q,a=question_rows();result=validate_question_counts(q,a)
    assert result['logical_questions']==14 and result['offered_forms']==28
    q,a=question_rows(8)
    with pytest.raises(ValueError,match='8 logical'):validate_question_counts(q,a)

def test_truth_leakage_and_form_disagreement_are_rejected():
    q,a=question_rows();q[0]['media']={'hidden':{'actor_id':1}}
    with pytest.raises(ValueError,match='truth'):validate_question_counts(q,a)
    q,a=question_rows();q[0]['question_zh']='a different question'
    with pytest.raises(ValueError,match='identity'):validate_question_counts(q,a)

def members():
    shared={'house':'h','room':'R1','materials':'m','camera':'c','background':'b','trajectory':'t','event_times':[1,4],'split':'train'}
    return [{**shared,'member':name,'audio_nuisance':audio,'visual_nuisance':visual,'emitter_association':binding}
            for name,audio,visual,binding in [('base',0,0,0),('sound',1,0,0),('picture',0,1,0),('binding',0,0,1)]]

def comparisons():
    return [{'left':'base','right':'sound','kind':'vision','comparison_key':'v'},
            {'left':'base','right':'picture','kind':'audio','comparison_key':'a'},
            {'left':'base','right':'binding','kind':'binding','comparison_key':'av'}]

def answers():
    return [{'member':m,'comparison_key':key,'canonical_answer':ans} for m,key,ans in
            [('base','v','2'),('sound','v','2'),('base','a','left'),('picture','a','left'),('base','av','blue'),('binding','av','yellow')]]

def test_counterfactual_checks_observed_answers_and_shared_scene_factors():
    assert audit_counterfactuals(members(),answers(),comparisons())['status']=='pass'
    a=answers();a[-1]['canonical_answer']='blue'
    assert audit_counterfactuals(members(),a,comparisons())['status']=='fail'
    m=members();m[1]['camera']='another camera'
    with pytest.raises(ValueError,match='shared'):audit_counterfactuals(m,answers(),comparisons())

def test_counterfactual_declared_intervention_is_insufficient_when_other_factors_change():
    m=members();m[1]['visual_nuisance']=1
    with pytest.raises(ValueError,match='outside'):audit_counterfactuals(m,answers(),comparisons())

def test_family_floor_counts_logical_questions_and_reports_empty_families():
    rows=[{'logical_question_id':str(i),'qa_id':'QA-14'} for i in range(14)]
    result=audit_family_floor(rows,{'QA-14':10,'QA-09':10})
    assert result['status']=='fail' and result['deficits']=={'QA-09':10}
    with pytest.raises(ValueError,match='deduplicated'):audit_family_floor(rows+rows,{})

def physical_test_rir():
    rng=np.random.default_rng(902);samples=32000;time=np.arange(samples)/16000
    w=rng.normal(size=samples)*np.exp(-time*9.0)*.01;w[:140]=0.;w[140]=.08
    direction=np.array([1.,0.,0.]);xyz=np.sqrt(3)*direction[:,None]*w
    return np.stack((w,xyz[1],xyz[2],xyz[0]))

def test_energy_failure_is_not_hidden_by_correct_foa_direction():
    h=physical_test_rir();kwargs={'source':np.array([3.,0.,0.]),'listener':np.zeros(3),'forward':np.array([1.,0.,0.])}
    equal=compare_rirs(h,h,**kwargs)
    assert qualify_energy([equal])['status']=='pass'
    gain=compare_rirs(h*2,h,**kwargs)
    assert gain['foa_error_deg']<1e-5 and gain['direct_energy_difference_db']>6
    assert qualify_energy([gain])['status']=='fail'
    assert qualify_energy([])['status']=='fail'

@pytest.mark.parametrize('split,count',[('train',400),('dev',100),('train',7000),('dev',1500),('test',1500)])
def test_planned_modality_counts_include_every_counterfactual_member(split,count):
    rows=seed_rows(split,count,houses());totals=Counter();questions=0
    for row in rows:
        for member_index in range(len(row['members'])):
            for w in range(row['window_count']):
                totals.update(question_allocation(row['question_quota_base']+member_index*row['window_count']+w,
                                                  row['questions_per_window']))
                questions+=row['questions_per_window']
    assert totals['vision']*100==30*questions
    assert totals['audio']*100==25*questions
    assert totals['audiovisual']*100==45*questions


def test_interrupted_stage_preserves_partial_outputs_and_recovers_without_overwrite(tmp_path):
    from avengine.timeline.v7_candidate_generation import stage
    from avengine.timeline.continuous_pilot import write_json
    calls=[]
    def interrupted(attempt):
        (attempt/'half_video.mp4').write_bytes(b'partial native stream')
        raise RuntimeError('simulated reboot before completion')
    with pytest.raises(RuntimeError,match='reboot'):stage(tmp_path,'video',{'seed':1},interrupted)
    assert not (tmp_path/'video_completed.json').exists()
    def complete(attempt):
        calls.append(attempt);(attempt/'video.mp4').write_bytes(b'complete media')
        write_json(attempt/'evidence.json',{'observed':True})
        return {'frames':150},['video.mp4','evidence.json']
    directory,result,resumed=stage(tmp_path,'video',{'seed':1},complete)
    assert not resumed and result['frames']==150 and len(calls)==1
    original=(directory/'video.mp4').stat().st_mtime_ns
    repeat=stage(tmp_path,'video',{'seed':1},complete)
    assert repeat[2] and len(calls)==1 and (directory/'video.mp4').stat().st_mtime_ns==original
    assert len(list(tmp_path.glob('video_attempt_*')))==2
    with pytest.raises(ValueError,match='recipe'):stage(tmp_path,'video',{'seed':2},complete)
    (directory/'evidence.json').unlink()  # test-owned fixture only
    with pytest.raises(ValueError,match='artifact'):stage(tmp_path,'video',{'seed':1},complete)


def test_seeded_candidate_refuses_exposure_and_unmeasured_rooms():
    from avengine.timeline.v7_candidate_generation import seeded_scene
    admission=houses()[0];row=seed_rows('train',1,houses())[0]
    bad=deepcopy(row);bad['room_id']='unchecked'
    with pytest.raises(ValueError,match='scope'):seeded_scene(bad,admission)
    bad=deepcopy(row);bad['assignment_status']='awaiting_admitted_house'
    with pytest.raises(ValueError,match='admitted'):seeded_scene(bad,admission)
    admission['split']='development';admission['known_exposure']=True
    bad=deepcopy(row);bad['split']='dev'
    with pytest.raises(ValueError,match='exposed'):seeded_scene(bad,admission)


def test_batch_budget_does_not_credit_planned_counterfactual_members():
    from avengine.timeline.v7_production_contract import audit_batch_budget
    q,a=question_rows()
    for i in range(0,len(q),2):
        for row in q[i:i+2]:row['modality']='vision' if i//2<4 else 'audio' if i//2<8 else 'audiovisual'
    result=audit_batch_budget([(q,a)],{'QA-14':1})
    assert result['verified_counterfactual_binding_questions']==0
    assert 'verified_counterfactual_binding_below_20_percent' in result['failure_reasons']
    with pytest.raises(ValueError,match='missing'):audit_batch_budget([(q,a)],{},['not_in_dataset'])
    with pytest.raises(ValueError,match='non-binding'):audit_batch_budget([(q,a)],{},['logical_0'])


def test_sealing_keeps_delivery_restartable_after_freeing_owned_intermediates(tmp_path):
    from avengine.timeline.continuous_pilot import write_json
    from avengine.timeline.v7_compaction import compact_parent,load_sealed
    root=tmp_path/'parents'/'scene';root.mkdir(parents=True)
    recipe={'seed_row':{'scene_id':'scene','members':['v0_a0']}}
    write_json(root/'input_recipe.json',recipe)
    for name in ('visual_v0','audio_v0_a0','windows_v0_a0'):
        attempt=root/(name+'_attempt_fixture');attempt.mkdir()
        if name.startswith('visual'):
            write_json(attempt/'native_visual_evidence.json',{'observed_mouth_trajectories':{'voice1':[1,2]}})
            files=['native_visual_evidence.json']
        elif name.startswith('audio'):
            write_json(attempt/'native_RLR_setup.json',{'physics':'unchanged'});files=['native_RLR_setup.json']
        else:
            window=attempt/'win';window.mkdir();(window/'video.mp4').write_bytes(b'encoded immutable')
            write_json(window/'evidence.json',{'native_visual_evidence':'old path','native_audio_evidence':'old path'})
            files=['win/video.mp4','win/evidence.json']
        write_json(root/(name+'_completed.json'),{'attempt':attempt.name,'required_files':files,'result':{}})
    cache=root/'rir_cache';cache.mkdir();(cache/'own_partial.npy').write_bytes(b'owned cache')
    external=tmp_path/'external_dataset';external.mkdir();(external/'keep').write_bytes(b'original')
    receipt={'members':[{'member':'v0_a0','windows':[{'window_id':'win'}]}],'formal_dataset_count':0}
    sealed=compact_parent(root,receipt)
    assert not cache.exists() and (external/'keep').read_bytes()==b'original'
    assert load_sealed(root,recipe)==sealed
    final=root/sealed['attempt'];assert (final/'win/video.mp4').read_bytes()==b'encoded immutable'
    assert 'native_visual_evidence' not in __import__('json').loads((final/'win/evidence.json').read_text())
    assert compact_parent(root,receipt)==sealed
    with pytest.raises(ValueError,match='recipe'):load_sealed(root,{'wrong':1})


def test_v3_streams_change_their_factor_without_recoupling_camera_or_events():
    from avengine.timeline.v7_candidate_generation import seeded_scene
    row=seed_rows('train',1,houses())[0]
    h=deepcopy(houses()[0]);h['checked_rooms']=[{'room_label':r,'placement':{
        'source_1_m':[2,1.3,4],'source_2_m':[-2,1.3,4],'camera_m':[0,1.6,0]}} for r in h['checked_room_ids']]
    _,base=seeded_scene(row,h);altered=deepcopy(row);altered['seeds']['camera']+=1
    _,camera=seeded_scene(altered,h)
    assert camera['camera_forward']!=base['camera_forward']
    assert camera['events']==base['events'] and camera['sources']==base['sources']
    altered=deepcopy(row);altered['seeds']['event_schedule']+=1;_,timed=seeded_scene(altered,h)
    assert timed['events']!=base['events'] and timed['camera_forward']==base['camera_forward']
    assert timed['sources']==base['sources']
    assert min(base['window_starts_s'])>=4 and max(base['window_starts_s'])<=20


def test_stage_recipe_resumes_with_json_integer_appearance_keys(tmp_path):
    from avengine.timeline.v7_candidate_generation import stage
    recipe={'appearance':{1:'blue',2:'yellow'},'bounds':(0,150)}
    def complete(attempt):
        (attempt/'evidence').write_text('native observation')
        return {'valid':True},['evidence']
    first=stage(tmp_path,'windows',recipe,complete)
    second=stage(tmp_path,'windows',recipe,complete)
    assert not first[2] and second[2] and first[0]==second[0]


@pytest.mark.parametrize('publish_panel',[False,True])
def test_optional_panel_floor_is_enforced_only_when_published(publish_panel):
    from avengine.timeline.v7_production_contract import family_minima_for_policy,audit_family_floor
    policy={'family_scope':['QA-01'],'family_minimum_logical_items':{'v7_S':{'train':100}},
        'additional_protocol_panel_minimum_when_published':{'STARSS-SPEAKERS':{'train':100}}}
    rows=[{'logical_question_id':f'q{i}','qa_id':'QA-01'} for i in range(100)]
    if publish_panel:rows.append({'logical_question_id':'panel','qa_id':'STARSS-SPEAKERS'})
    minima=family_minima_for_policy(policy,'v7_S','train',[r['qa_id'] for r in rows])
    audit=audit_family_floor(rows,minima)
    if publish_panel:
        assert audit['status']=='fail' and audit['deficits']=={'STARSS-SPEAKERS':99}
    else:assert audit['status']=='pass'
