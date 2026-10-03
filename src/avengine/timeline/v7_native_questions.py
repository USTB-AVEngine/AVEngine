"""Balanced native-pixel pilot candidates; formal family quotas are separate.

Uses v0.2 half-open activity segments. Every audiovisual item binds an
unambiguously identified audio activity to a visible native person. Missing
evidence causes rejection; forms, repeats and empty placeholders cannot fill
the 12--15 logical-item requirement.
"""
from __future__ import annotations
import math
import numpy as np
from avengine.timeline.continuous_scene import activity_segments,window_segments,generate_v02_questions
from avengine.timeline.v7_production_contract import question_allocation,validate_question_counts


def make_questions(scene,start_s,window_id,visual,appearances,media,*,quota_index=0,count=14):
    frames=visual['frames'];lo=round(start_s*15);end=lo+150
    if len(frames[lo:end])!=150:raise ValueError('window lacks 150 native frames')
    actors=[a['actor_id'] for a in frames[lo]['actors']]
    if len(actors)!=2 or set(actors)!=set(appearances):
        raise ValueError('this checked pilot requires two explicit appearance bindings')
    listener=np.asarray(scene['listener_m'])
    segments=window_segments(activity_segments(scene['events']),start_s)
    public,private,pool=[],[],{'vision':[],'audio':[],'audiovisual':[]}
    def emit(modality,family,key,text,answer,evidence,domain):
        pool[modality].append({'qa_id':family,'key':key,'question_zh':text,'canonical_answer':str(answer),
                               'evidence':evidence,'answer_domain':domain})
    def frame(t):return frames[lo+round(t*15)]
    def visible(t):return [x for x in frame(t)['actors'] if x['visible_pixels']>=20]
    # Visual questions never consult events or dry-audio identities.
    seen={x['actor_id'] for f in frames[lo:end] for x in f['actors'] if x['visible_pixels']>=20}
    emit('vision','QA-22','visible_count','这10秒画面中出现过多少个不同的人？',len(seen),
         {'kind':'native_semantic_pixels','frame_span':[lo,end],'visible_actor_ids':sorted(seen)},'nonnegative_integer')
    selected=[]
    for tick in range(0,50):
        t=tick/5;people=visible(t)
        if len(people)!=2:continue
        distances=[np.linalg.norm(np.asarray(x['position_m'])-listener) for x in people]
        if abs(distances[0]-distances[1])<.2 or any(abs(t-old)<1.8 for old in selected):continue
        near=people[int(np.argmin(distances))]['actor_id'];selected.append(t)
        emit('vision','QA-14',f'near_{tick}',f'画面第{t:.1f}秒，哪个人更靠近镜头？',appearances[near],
             {'kind':'native_roots_and_pixels','frame':lo+round(t*15),'distances_m':[float(x) for x in distances]},list(appearances.values()))
    # The v0.2 count and sound-anchor protocol is reused, including exclusions
    # for tied first-three anchors and direction dead zones.
    base,gold,audit=generate_v02_questions(scene,start_s,window_id,media)
    answers={x['question_id']:x for x in gold}
    for q in base:
        if q['form']!='open' or q['qa_id'] not in ('QA-23','STARSS-SPEAKERS','QA-04','QA-25'):continue
        emit('audio',q['qa_id'],q['question_id'].split('__')[-2],q['question_zh'],answers[q['question_id']]['canonical_answer'],
             {'kind':'v02_activity_and_native_mouth_trajectory','protocol_audit':audit},q['answer_domain'])
    for lo_tick,hi_tick,label in ((0,50,'前5秒'),(50,100,'后5秒')):
        n=sum(e['anchor_tick']<hi_tick and e['window_end_tick']>lo_tick for e in segments)
        emit('audio','QA-23',f'count_{lo_tick}_{hi_tick}',f'{label}内共有多少段声音活动（包括该半段开始前就在响的）？',n,
             {'kind':'v02_half_open_activity_intersection','interval_ticks':[lo_tick,hi_tick]},'nonnegative_integer')
    activity=np.zeros((2,100),bool)
    for e in segments:
        if e['class_id'] not in (0,1):continue
        activity[actors.index(e['source_id']),e['anchor_tick']:e['window_end_tick']]=True
    overlap=int(np.count_nonzero(activity.sum(axis=0)>=2))
    emit('audio','QA-05','overlap','片段内是否有两个人的话声重叠至少0.5秒？','yes' if overlap>=5 else 'no',
         {'kind':'v02_activity_intersection','overlap_ticks':overlap},['yes','no'])
    # Single-speaker timestamps avoid answers that remain "both" after a
    # counterfactual association swap. Native masks prove both choices exist.
    picked=[]
    for tick in range(2,98,2):
        if activity[:,tick].sum()!=1 or any(abs(tick-old)<4 for old in picked):continue
        t=tick/10;people=visible(t)
        if len(people)!=2:continue
        actor=actors[int(np.argmax(activity[:,tick]))]
        picked.append(tick)
        emit('audiovisual','QA-18',f'active_{tick}',f'在第{t:.1f}秒，正在说话的是哪个人？',appearances[actor],
             {'kind':'native_pixels_and_activity_binding','frame':lo+round(t*15),'active_actor_id':actor},list(appearances.values()))
        named=actors[len(picked)%2]
        emit('audiovisual','QA-01',f'speaking_{tick}',f'第{t:.1f}秒，{appearances[named]}是否正在讲话？',
             'yes' if actor==named else 'no',{'kind':'native_pixels_and_activity_binding','frame':lo+round(t*15),
             'target_actor_id':named,'active_actor_id':actor},['yes','no'])
    allocation=question_allocation(quota_index,count)
    for modality in ('vision','audio','audiovisual'):
        # A rejected candidate never silently becomes a fabricated question.
        unique=[];keys=set()
        for x in pool[modality]:
            key=(x['qa_id'],x['question_zh'])
            if key not in keys:unique.append(x);keys.add(key)
        if len(unique)<allocation[modality]:
            raise ValueError(f'{modality}: {len(unique)} candidates, need {allocation[modality]}')
        for item in unique[:allocation[modality]]:
            logical=f'{window_id}__{item["qa_id"]}__{item["key"]}'
            public.append({'question_id':logical+'__open','logical_question_id':logical,'qa_id':item['qa_id'],
                'modality':modality,'form':'open','question_zh':item['question_zh'],'answer_domain':item['answer_domain'],'media':media})
            private.append({'question_id':logical+'__open','canonical_answer':item['canonical_answer'],
                'answer':item['canonical_answer'],'evidence':item['evidence'],'comparison_key':item['key']})
    check=validate_question_counts(public,private,expected_allocation=allocation)
    return public,private,{'question_check':check,'protocol':audit,'candidate_pool_counts':{k:len(v) for k,v in pool.items()},
        'qualification':'native-pixel pilot candidates; full family floor and moving RIR energy must also pass'}
