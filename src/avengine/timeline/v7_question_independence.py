"""Audit independent questions within a parent, without changing any QA answer.

Clip-relative timestamps in public wording identify crop presentation, not a
new question type. Physical targets and canonical answers remain in the key.
The raw offered/selected item counts are retained separately for throughput.
"""
from __future__ import annotations
from collections import Counter
import json,re,math

INDEPENDENCE_REVISION='parent-family-target-wording-answer-20261015-v2'


def normalized_wording(q):
    text=q.get('question_en') or q.get('question_zh') or ''
    # Only time/frame presentation is normalized. Content, colors, taxonomy,
    # comparisons and other wording remain literal; no model is involved.
    number=r'-?\d+(?:\.\d+)?'
    text=re.sub(r'\[\s*'+number+r'\s*,\s*'+number+r'\s*[)\]]','[时间区间]',text)
    text=re.sub(r'第\s*'+number+r'\s*(?:至|到|[-–])\s*第?\s*'+number+r'\s*秒','指定时间区间',text)
    text=re.sub(r'第\s*'+number+r'\s*(?:秒|帧)','指定时刻',text)
    text=re.sub(number+r'\s*(?:至|到|[-–])\s*'+number+r'\s*秒','指定时间区间',text)
    text=re.sub(r'\b'+number+r'\s*(?:seconds?|secs?|frames?)\b','specified time',text,flags=re.I)
    return re.sub(r'\s+',' ',text).strip()


def _stable(value):
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('nonfinite canonical answer')
        return round(value,6)
    if isinstance(value,dict):return {k:_stable(v) for k,v in sorted(value.items())}
    if isinstance(value,list):return [_stable(v) for v in value]
    return value


def target_entities(q,gold=None):
    gold=gold or {};evidence=gold.get('evidence',{})
    preferred=('target_actor_id','actor_id','target_source_id','target_entity_id','native_actor_id')
    def find(node,depth=0):
        if depth>5 or not isinstance(node,dict):return []
        for key in preferred:
            if node.get(key) is not None:return [str(node[key])]
        for key in ('target_actor_ids','actor_ids','target_entity_ids'):
            if isinstance(node.get(key),list):return sorted(map(str,node[key]))
        for key in ('v6_item','item','target','anchor_event','target_binding','private_evidence','evidence','truth'):
            if key in node:
                found=find(node[key],depth+1)
                if found:return found
        return []
    query=evidence.get('query',{})
    if query.get('kind')=='offscreen_origin'and query.get('subject_actor_id')is not None:
        return (str(query['subject_actor_id']),),'private_physical_subject'
    found=find(evidence) or find(gold) or find(q.get('target_verification',{}))
    if found:return tuple(found),'private_physical_target'
    reference=(q.get('target_verification',{}).get('result',{}).get('question_id')
               or gold.get('authority_question_id') or gold.get('v6_question_id') or q.get('question_id',''))
    match=re.search(r'__target_([^_]+)',reference)
    if match:return (match.group(1),),'original_target_identifier'
    # The literal referent remains in the wording. An absent target ID never
    # gets a new per-window identity, which would falsely inflate independence.
    return ('wording_referent_or_episode_scope',),'conservative_wording_scope'


def independence_key(parent_id,q,gold=None):
    gold=gold or {}
    if 'canonical_answer' in gold:answer=gold['canonical_answer']
    elif 'canonical_answer' in q:answer=q['canonical_answer']
    else:raise ValueError('independence audit requires the actual canonical answer')
    targets,method=target_entities(q,gold)
    return (parent_id,q['qa_id'],targets,normalized_wording(q),json.dumps(_stable(answer),ensure_ascii=False,sort_keys=True)),method


def summarize_independence(parent_id,questions,answers=None):
    byid={g['question_id']:g for g in (answers or [])}
    logical={q['logical_question_id']:q for q in questions}
    unique={};methods=Counter();raw=Counter()
    for q in logical.values():
        key,method=independence_key(parent_id,q,byid.get(q['question_id']))
        methods[method]+=1;raw[q['qa_id']]+=1
        unique.setdefault(key,q)
    return {'revision':INDEPENDENCE_REVISION,'raw_logical_questions':len(logical),
            'independent_questions':len(unique),'family_counts':dict(Counter(q['qa_id'] for q in unique.values())),
            'raw_family_counts':dict(raw),'target_identity_methods':dict(methods)},unique


def window_overlap(starts,duration=10.):
    starts=sorted(float(s) for s in starts)
    adjacent=[max(0.,duration-(b-a)) for a,b in zip(starts,starts[1:])]
    return {'window_starts_s':starts,'window_duration_s':duration,'window_count':len(starts),
            'adjacent_overlap_seconds':adjacent,'maximum_adjacent_overlap_s':max(adjacent,default=0.),
            'pairwise_overlapping_pairs':sum(a<b<a+duration for i,a in enumerate(starts) for b in starts[i+1:]),
            'aligned_nonoverlap_starts_present_s':[s for s in starts if s in (0.,10.,20.)],
            'aligned_nonoverlap_starts_missing_s':[s for s in (0.,10.,20.) if s not in starts]}


def deduplicate_parent_delivery(row,result,root,directory):
    """Derive sparse export after the unchanged raw 14-item/native admission."""
    from pathlib import Path
    from copy import deepcopy
    import os
    from avengine.timeline.v7_candidate_generation import stage,read_json
    from avengine.timeline.continuous_pilot import write_json
    source=Path(directory);root=Path(root);windows=result['windows'];ordered=[]
    for win in windows:
        wid=win['window_id'];e=read_json(source/'private'/f'{wid}_evidence.json')
        start=e.get('source_qa_evidence',{}).get('window_start_s',e.get('start_s'))
        if start is None:
            wi=int(wid.rsplit('w',1)[1]);values=row.get('acceptance_window_starts_s') or [t/10 for t in row.get('window_start_ticks',[0,150])]
            start=values[wi]
        ordered.append((float(start),wid,win))
    seen=set();selected={};stats=[];dropped=[]
    for start,wid,old in sorted(ordered,key=lambda x:(x[0],x[1])):
        public=read_json(source/'windows'/wid/'questions.json');gold=read_json(source/'private'/f'{wid}_answers.json');byid={g['question_id']:g for g in gold}
        keep=[];answers=[];drop=[]
        for q in public:
            g=byid[q['question_id']];key,_=independence_key(row['scene_id'],q,g)
            if key in seen:drop.append(q['question_id']);continue
            seen.add(key);keep.append(q);answers.append(g)
        selected[wid]=(keep,answers)
        raw=len({q['logical_question_id']for q in public});n=len({q['logical_question_id']for q in keep})
        stats.append({**deepcopy(old),'window_start_s':start,'raw_logical_questions':raw,'logical_questions':n,
            'family_counts':dict(Counter(q['qa_id']for q in keep)),'dedup_dropped':raw-n,'dropped_question_ids':drop})
        dropped.extend(drop)
    actual={**deepcopy(result),'windows':stats,'raw_logical_questions':sum(w['raw_logical_questions']for w in stats),
            'independent_logical_questions':sum(w['logical_questions']for w in stats),
            'dedup_dropped':sum(w['dedup_dropped']for w in stats),'dedup_rule_version':INDEPENDENCE_REVISION,
            'raw_admitted_delivery_directory':str(source),'original_raw_native_and_14_item_admission_preserved':True}
    for key in ('pilot_logical_questions','alpha_formal_logical_questions'):
        if key in actual and actual[key]:actual[key]=actual['independent_logical_questions']
    def publish(attempt):
        required=[]
        for start,wid,old in ordered:
            dest=attempt/'windows'/wid;dest.mkdir(parents=True)
            for name in ('video.mp4','audio_foa.wav','audio_mono.wav'):
                os.link(source/'windows'/wid/name,dest/name);required.append(str((dest/name).relative_to(attempt)))
            q,g=selected[wid];write_json(dest/'questions.json',q);required.append(str((dest/'questions.json').relative_to(attempt)))
            for name,data in [(f'{wid}_answers.json',g),(f'{wid}_evidence.json',read_json(source/'private'/f'{wid}_evidence.json'))]:
                p=attempt/'private'/name;p.parent.mkdir(parents=True,exist_ok=True);write_json(p,data);required.append(str(p.relative_to(attempt)))
        write_json(attempt/'parent_dedup_manifest.json',actual);required.append('parent_dedup_manifest.json')
        return actual,required
    folder,saved,_=stage(root,'deduplicated_delivery_v2',{'source_directory':str(source),'rule_version':INDEPENDENCE_REVISION},publish)
    return {**saved,'delivery_directory':str(folder)}
