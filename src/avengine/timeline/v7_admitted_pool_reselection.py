"""Re-select English rows from fully admitted native pools, without rendering.

Visual and requested motion supplements are added. Every supplement already passed the full
original pool replay and is rechecked with the original QA and scoring code.
Quartet supplements need separate nuisance checks, so remain in their original
sealed selection here. Original seals and per-parent exports are immutable.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import datetime,fcntl,hashlib,json,os
from avengine.timeline.v7_question_independence import independence_key
from avengine.timeline.v7_post_dedup_selection import select_independent_rows,REVISION


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def _entry(scene,question,gold):
    key,_=independence_key(scene,question,gold)
    return {'row':question,'key':key}


def compatible_prior_candidates(path,pointer_path):
    """Reuse this fresh parent's already replayed pool; recheck current scope.

    Its native source is immutable and independently checked above. Earlier
    pre-policy cache versions are deliberately never offered to this helper.
    """
    path=Path(path)
    if not path.is_file():return None
    result=load(path)
    if result.get('source_seal_sha256')!=sha(pointer_path):return None
    if Path(result.get('source_seal','')).resolve()!=Path(pointer_path).resolve():return None
    from avengine.timeline.v7_english_export import validate_english_model_text
    for entry in result.get('entries',[]):validate_english_model_text(entry['row'])
    return result


def parent_entries(receipt,output,*,read_only_source_roots=()):
    from avengine.qa.v7_validation import validate_v6_items
    from avengine.qa.v7_english_surface import appearance_replacements,appearance_choice_replacements,reissue_item_english,relabel_english
    from avengine.timeline.v7_english_export import english_public_question,validate_english_model_text
    from avengine.timeline.v7_alpha_delivery import readable_training_answer
    from avengine.qa.unified_scoring import score_unified_item
    output=Path(output);pointer_path=Path(receipt['export']['source_seal'])
    allowed=[output.resolve(),*[Path(p).resolve()for p in read_only_source_roots]]
    def permitted(path):
        if not any(Path(path).resolve().is_relative_to(p)for p in allowed):
            raise ValueError('candidate evidence escaped authorized read-only roots')
    permitted(pointer_path)
    from avengine.timeline.v7_export_scene_policy import cached_parent_policy,REVISION as SCENE_POLICY
    scene_policy=cached_parent_policy(pointer_path,output/'evidence/scene_delivery_policy')
    if scene_policy['status']!='pass':raise ValueError('supplement parent violates current native scene delivery policy')
    pointer=load(pointer_path)
    if pointer['result']['status']not in {'integration_pilot_admitted','formal_alpha_admitted'}:
        raise ValueError('supplement source parent is not admitted')
    recipe=pointer['recipe']['input_recipe'];row=recipe.get('row',recipe.get('seed_row'))
    scene=row['scene_id'];root=pointer_path.parent;seal=root/pointer['attempt']
    cache=output/'evidence/post_dedup_candidates'/(str(receipt['slot'])+'_'+sha(pointer_path)+'_v6_owner_count_scope.json')
    if cache.exists():return load(cache)
    prior_path=output/'evidence/post_dedup_candidates'/(str(receipt['slot'])+'_'+sha(pointer_path)+'_v5_all_legal_families.json')
    prior=compatible_prior_candidates(prior_path,pointer_path)  # v5 already has fixed-observer, English and original full-pool replay.
    base=[json.loads(s)for s in(output/'export/parent_rows'/f"{receipt['slot']:03d}.jsonl").read_text().splitlines()if s.strip()]
    bywindow=defaultdict(list)
    for r in base:bywindow[r['window_id']].append(r)
    entries=list(prior['entries'])if prior else [];proofs=list(prior['proofs'])if prior else [];known_ids={e['row']['question_id']for e in entries}|{r['question_id']for r in base};supplemented=prior['new_visual_candidate_rows']if prior else 0;motion_supplemented=0
    for win in pointer['result']['windows']:
        wid=win['window_id'];gold={g['question_id']:g for g in load(seal/'private'/f'{wid}_answers.json')}
        if not prior:entries.extend(_entry(scene,r,gold[r['question_id']])for r in bywindow[wid])
        if len(row['members'])!=1:continue
        member=(win.get('member')or wid.rsplit('__',2)[-2]);marker=load(root/f'selected_windows_{member}_completed.json')
        pool=Path(marker['recipe']['pool_directory'])/wid;permitted(pool)
        qpath=pool/'questions.json';apath=pool/'candidate_audit.json';rpath=pool/'independent_v6_truth.json'
        if not rpath.exists():
            proofs.append({'window_id':wid,'supplements':0,'reason':'original complete pool replay file absent; no candidate added'});continue
        pub=load(qpath);original=load(rpath)
        if original.get('status')!='pass' or original.get('checked_forms')!=len(pub) or original.get('mismatches'):
            raise ValueError('original full-pool replay is incomplete or failed')
        candidates=[q for q in pub if q['form']=='open'and q['question_id']not in known_ids]
        if not candidates:continue
        audit=load(apath);all_gold={g['question_id']:g for g in load(pool/'private_answers.json')}
        evidence=load(pool/'evidence.json')
        from avengine.qa.v7_visible_truth import validate_additional_items
        replay_parts=[validate_v6_items(candidates,[all_gold[q['question_id']]for q in candidates],audit,seed=str(row['scene_seed'])),
            validate_additional_items(candidates,[all_gold[q['question_id']]for q in candidates],evidence['source_qa_evidence'])]
        replay={'checked_forms':sum(p['checked_forms']for p in replay_parts),'passed_forms':sum(p['passed_forms']for p in replay_parts),
            'mismatches':[i for p in replay_parts for i in p['mismatches']],'methods':[p.get('method')for p in replay_parts]}
        if replay['mismatches']or replay['checked_forms']!=len(candidates)or replay['passed_forms']!=len(candidates):
            raise ValueError('supplement original QA truth replay failed')
        replacements=appearance_replacements(recipe['assets'],audit['facts'])
        items={i['question_id']:i for i in audit['v6_items']};evidence=load(pool/'evidence.json')
        start=evidence.get('source_qa_evidence',{}).get('window_start_s',evidence.get('start_s'))
        if start is None:start=(row.get('acceptance_window_starts_s')or[t/10 for t in row['window_start_ticks']])[int(wid.rsplit('w',1)[1])]
        for q in candidates:
            q=deepcopy(q);g=all_gold[q['question_id']];authority={}
            if g.get('authority_question_id'):
                item=items[g['authority_question_id']]
                local={**replacements,**appearance_choice_replacements(item,recipe['assets'],audit['facts'])}
                authority={item['question_id']:reissue_item_english(item,local)}
                q['question_en']=relabel_english(q.get('question_en',''),local)
                for c in q.get('choices',[]):
                    if 'text'in c:c['text']=relabel_english(c['text'],local)
                bad=score_unified_item(authority[item['question_id']],'__invalid_trial_control_answer__',form=q['form'])
                if bad.get('status')=='scored'and bad.get('score')==1.:raise ValueError('supplement invalid-answer control was accepted')
            else:
                invalid=deepcopy(g);invalid.update(canonical_answer='__invalid_control__',answer='__invalid_control__')
                control=validate_additional_items([q],[invalid],evidence['source_qa_evidence'])
                if control['passed_forms']or not control['mismatches']:raise ValueError('native supplement invalid-answer control was accepted')
            answer,check=readable_training_answer(q,g,authority)
            english=english_public_question(q);validate_english_model_text({**english,'model_answer':answer})
            folder=Path(receipt.get('delivery_export_prefix',''))/'parents'/scene/'windows'/wid
            training={'question_id':q['question_id'],'logical_question_id':q['logical_question_id'],
                'independent_parent_scene_id':scene,'window_id':wid,'window_start_s':float(start),
                'split':row['split'],'house':row['house'],'qa_id':q['qa_id'],'protocol_panel':q['protocol_panel'],
                'modality':q['modality'],'question_en':english['question_en'],'choices':english.get('choices',[]),
                'video':str(folder/'video.mp4'),'audio_foa':str(folder/'audio_foa.wav'),'audio_mono':str(folder/'audio_mono.wav'),
                'foa_format':'ACN_WYZX_SN3D_camera','model_answer':answer,'canonical_answer':g['canonical_answer'],
                'private_evidence':str(Path(receipt.get('delivery_export_prefix',''))/'parents'/scene/'private'/f'{wid}_evidence.json'),
                'target_verification':check,'E_S_input_source':'deployment_prediction_not_supplied',
                'dataset_role':'acceptance_only','producer_admission_status':pointer['result']['status'],
                'post_dedup_supplement':{'revision':REVISION,'original_candidate_pool':str(pool),
                    'original_candidate_audit_sha256':sha(apath),'original_complete_truth_replay_sha256':sha(rpath),
                    'original_14_question_seal_preserved':str(pointer_path),'new_renders':0}}
            for name in('video','audio_foa','audio_mono'):
                if not(output/'export'/training[name]).is_file():raise ValueError('supplement media missing from admitted export')
            entries.append(_entry(scene,training,g));supplemented+=q['modality']=='vision';motion_supplemented+=q['qa_id']in {'QA-06','QA-15','QA-16','QA-17','QA-08','STARSS-FRUSTUM','QA-20','QA-24','QA-04','QA-13','QA-25','QA-28','QA-21','V7-CLASS34'}
        proofs.append({'window_id':wid,'original_replay':str(rpath),'supplements':len(candidates),
            'truth_replay':replay,'source_pool':str(pool),'CJK_and_ID_gate':'pass','invalid_answer_controls':'pass'})
    from avengine.timeline.v7_count_scope_review import review_qa22_scope
    count_reviews=[];eligible=[];scope_facts={}
    for entry in entries:
        training=entry['row']
        if training['qa_id']=='QA-22':
            wid=training['window_id']
            if wid not in scope_facts:
                member=wid.rsplit('__',2)[-2]
                marker=load(root/f'selected_windows_{member}_completed.json')
                audit_path=Path(marker['recipe']['pool_directory'])/wid/'candidate_audit.json';permitted(audit_path)
                scope_facts[wid]=load(audit_path)['facts']
            review=review_qa22_scope(training,scope_facts[wid]);count_reviews.append({'question_id':training['question_id'],**review})
            if review['status']=='answer_scope_requires_reconstruction':continue
        eligible.append(entry)
    entries=eligible
    result={'entries':entries,'owner_qa22_scope_reviews':count_reviews,'new_visual_candidate_rows':supplemented,'new_motion_candidate_rows':motion_supplemented,'source_seal':str(pointer_path),
        'source_seal_sha256':sha(pointer_path),'proofs':proofs,'scene_delivery_policy':scene_policy}
    save(cache,result);return result


def _collect_parent_job(arguments):
    receipt,output,roots=arguments
    return parent_entries(receipt,output,read_only_source_roots=roots)


def balanced_batch(receipts,output,*,read_only_source_roots=(),workers=1,family_minimums=None):
    entries=[];parents=[]
    if workers>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(4,workers),mp_context=multiprocessing.get_context('spawn'))as executor:
            results=list(executor.map(_collect_parent_job,[(r,output,read_only_source_roots)for r in receipts]))
    else:results=[parent_entries(r,output,read_only_source_roots=read_only_source_roots)for r in receipts]
    for receipt,result in zip(receipts,results):
        entries.extend(result['entries']);parents.append({'scene_id':receipt['scene_id'],
            'slot':receipt['slot'],'new_visual_candidate_rows':result['new_visual_candidate_rows'],'new_motion_candidate_rows':result.get('new_motion_candidate_rows',0),
            'source_seal':result['source_seal'],'proofs':result['proofs']})
    rows,report=select_independent_rows(entries,family_minimums=family_minimums)
    report.update(original_admitted_independent_rows=sum(r['independent_questions']for r in receipts),
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parents=parents,
        visual_supplements_actually_selected=sum('post_dedup_supplement'in r and r['modality']=='vision'for r in rows),motion_supplements_actually_selected=sum('post_dedup_supplement'in r and r['qa_id']in {'QA-06','QA-15','QA-16','QA-17'}for r in rows),
        newly_rendered_parents=0,original_native_seals_changed=False)
    save(Path(output)/'evidence/POST_DEDUP_SELECTION.json',report)
    return rows,report


def record_batch_selection(path,rows,*,source_usage=None):
    """Feedback uses the final independent export, including modality selection."""
    from avengine.timeline.v7_demand import read_demand_state,demand_snapshot
    path=Path(path);groups=defaultdict(list)
    for row in rows:groups[row['independent_parent_scene_id']].append(row)
    with path.with_suffix(path.suffix+'.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=read_demand_state(path);current_keys=set()
        for scene,selected in groups.items():
            key=next((k for k,v in state['parents'].items()if v.get('scene_id')==scene),scene)
            current_keys.add(key)
            old=state['parents'].get(key,{})
            state['parents'][key]={**old,'scene_id':scene,'logical_questions':len(selected),
                'family_counts':dict(Counter(r['qa_id']for r in selected)),
                'modality_counts':dict(Counter(r['modality']for r in selected)),
                'post_dedup_selection_revision':REVISION}
        state['parents']={k:v for k,v in state['parents'].items()if k in current_keys}
        from avengine.timeline.v7_answer_branch_demand import summarize
        state['answer_branch_distribution']=summarize(rows)
        state['answer_counts']={f:item['answers']for f,item in state['answer_branch_distribution']['families'].items()}
        if source_usage is not None:
            state.update(source_record_usage_counts=source_usage['recording_counts'],
                source_entity_usage_counts=source_usage['sounding_entity_counts'],source_class_usage_counts=source_usage['registered_class_counts'],
                authored_voice_counts=source_usage['authored_voice_counts'],recording_use_counts=source_usage['recording_counts'],
                entity_use_counts=source_usage['sounding_entity_counts'],source_inventory_requests=source_usage.get('next_inventory_requests',[]))
        counts=Counter();modalities=Counter()
        for parent in state['parents'].values():counts.update(parent['family_counts']);modalities.update(parent.get('modality_counts',{}))
        state.update(logical_questions=sum(counts.values()),family_counts=dict(counts),modality_counts=dict(modalities),
            post_dedup_selection_revision=REVISION)
        state['class34_positive_counts']=dict(Counter(str(r['canonical_answer'])for r in rows if r['qa_id']=='V7-CLASS34'))
        state['live_speaker_count_counts']=dict(Counter(str(r['canonical_answer'])for r in rows if r['qa_id']=='STARSS-SPEAKERS'))
        demand_snapshot(state);save(path,state)
