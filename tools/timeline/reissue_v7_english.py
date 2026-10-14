"""Reissue admitted question presentation and export in English; render nothing."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse,json,hashlib,shutil,re,time
from avengine.qa.v7_english_surface import appearance_replacements,relabel_english,reissue_item_english,protocol_question_en
from avengine.timeline.v7_english_export import english_public_question,validate_english_model_text,ENGLISH_EXPORT_REVISION
from avengine.timeline.v7_alpha_delivery import readable_training_answer
from avengine.timeline.v7_question_independence import independence_key,window_overlap,INDEPENDENCE_REVISION
from avengine.qa.unified_scoring import score_unified_item


def load(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10000)
    p.add_argument('--owner-half-grid',action='store_true',help='Select only already delivered 0/5/10/15/20 crops; missing crops remain not_run')
    a=p.parse_args()
    if a.output.exists():raise ValueError('English reissue requires a fresh task-owned output')
    a.output.mkdir(parents=True);summary_bytes=a.summary.read_bytes();summary=json.loads(summary_bytes)
    parents=[x for x in summary['parents'] if x.get('status')=='passed'][:a.limit]
    write(a.output/'source_summary_snapshot.json',summary)
    rows=[];parent_reports=[];errors=[];raw_family=Counter();independent_family=Counter();scorer_checks=0;seen=set();aligned_rows=[]
    for parent_info in parents:
        scene=parent_info['scene_id'];root=Path(parent_info['parent_directory']);pointer_path=Path(parent_info['seal_path']);pointer=load(pointer_path)
        if pointer['result']['status'] not in {'integration_pilot_admitted','formal_alpha_admitted'}:raise ValueError('only originally admitted parents may be reissued')
        seal=root/pointer['attempt'];recipe=pointer['recipe']['input_recipe'];seed=recipe.get('row',recipe.get('seed_row',{}));assets=recipe['assets']
        parent_raw=[];parent_unique=[];win_reports=[];before=len(rows);seen_before=set(seen);aligned_before=len(aligned_rows);raw_before=raw_family.copy();independent_before=independent_family.copy()
        try:
            ordered=[]
            for win in pointer['result']['windows']:
                wid=win['window_id'];e=load(seal/'private'/f'{wid}_evidence.json')
                start=e.get('source_qa_evidence',{}).get('window_start_s',e.get('start_s'))
                if start is None:
                    wi=int(wid.rsplit('w',1)[1]);start=(seed.get('acceptance_window_starts_s') or [x/10 for x in seed.get('window_start_ticks',[0,150])])[wi]
                if a.owner_half_grid and float(start) not in (0.,5.,10.,15.,20.):continue
                ordered.append((float(start),wid,e,win))
            for start,wid,e,win in sorted(ordered,key=lambda x:(x[0],x[1])):
                member=win.get('member',wid.rsplit('__',2)[-2]);selected=load(root/f'selected_windows_{member}_completed.json')
                pool=Path(selected['recipe']['pool_directory'])/wid;audit=load(pool/'candidate_audit.json')
                replacements=appearance_replacements(assets,audit.get('facts',{}))
                authority={i['question_id']:(reissue_item_english(i,replacements) if 'truth'in i else deepcopy(i)) for i in audit['v6_items']}
                qold=load(seal/'windows'/wid/'questions.json');gold_old=load(seal/'private'/f'{wid}_answers.json');byid={g['question_id']:g for g in gold_old};qnew=[];gnew=[];n_raw=0;n_kept=0;dropped=[]
                for q in qold:
                    g=deepcopy(byid[q['question_id']]);new=deepcopy(q)
                    if g.get('authority_question_id'):
                        item=authority[g['authority_question_id']];request=item.get('model_input',{}).get(q['form'],{})
                        new['question_en']=relabel_english(request.get('question_en') or q.get('question_en') or '',replacements)
                        if 'label'in item.get('truth',{}):g.setdefault('authority_truth',{})['label']=item['truth']['label']
                    elif q.get('protocol_panel')=='STARSS23_v0.2':
                        new['question_en']=protocol_question_en(q['qa_id'],g['evidence'])
                    else:new['question_en']=relabel_english(q.get('question_en')or'',replacements)
                    for choice in new.get('choices',[]):
                        if 'text'in choice:choice['text']=relabel_english(choice['text'],replacements)
                    new=english_public_question(new)
                    answer,check=readable_training_answer(new,g,authority)
                    new['model_answer']=answer;validate_english_model_text(new)
                    if g['canonical_answer']!=byid[q['question_id']]['canonical_answer']:raise ValueError('English reissue changed canonical answer')
                    if g.get('authority_question_id'):
                        negative=score_unified_item(authority[g['authority_question_id']],'__invalid_control_answer__',form=q['form'])
                        if negative.get('status')=='scored'and negative.get('score')==1:raise ValueError('unchanged scorer accepted invalid control')
                        scorer_checks+=1
                    key,method=independence_key(scene,new,g);raw_family[q['qa_id']]+=1;n_raw+=1
                    if start in (0.,10.,20.):aligned_rows.append((key,q['qa_id']))
                    if key in seen:dropped.append(q['question_id']);continue
                    seen.add(key);independent_family[q['qa_id']]+=1;n_kept+=1
                    folder=Path('parents')/scene/'windows'/wid
                    exported={k:new[k]for k in ['question_id','logical_question_id','qa_id','protocol_panel','modality','form','question_en']}
                    if new.get('choices'):exported['choices']=new['choices']
                    exported.update(independent_parent_scene_id=scene,window_id=wid,window_start_s=start,split=seed['split'],house=seed['house'],
                        video=str(folder/'video.mp4'),audio_foa=str(folder/'audio_foa.wav'),audio_mono=str(folder/'audio_mono.wav'),
                        model_answer=answer,canonical_answer=g['canonical_answer'],target_verification=check,
                        dataset_role='isolated_English_reissue',dedup_rule_version=INDEPENDENCE_REVISION,
                        English_export_revision=ENGLISH_EXPORT_REVISION,source_original_question_id=q['question_id'])
                    validate_english_model_text(exported);rows.append(exported);public_question=deepcopy(new);public_question.pop('model_answer',None);qnew.append(public_question);gnew.append(g)
                dest=a.output/'parents'/scene/'windows'/wid;dest.mkdir(parents=True)
                # Copy bytes from other tasks; no hardlink metadata or original
                # inode can be mutated through this isolated reissue.
                for name in ['video.mp4','audio_foa.wav','audio_mono.wav']:
                    shutil.copyfile(seal/'windows'/wid/name,dest/name)
                write(dest/'questions.json',qnew);write(a.output/'parents'/scene/'private'/f'{wid}_answers.json',gnew)
                used={g['authority_question_id']for g in gnew if g.get('authority_question_id')}
                write(a.output/'parents'/scene/'private'/f'{wid}_English_authority.json',{k:authority[k]for k in used})
                win_reports.append({'window_id':wid,'member':member,'start_s':start,'raw_export_questions':n_raw,'independent_questions':n_kept,'dedup_dropped':len(dropped),'dropped_question_ids':dropped})
            report={'scene_id':scene,'source_seal':str(pointer_path),'source_seal_sha256':sha(pointer_path),
                'raw_export_questions':sum(w['raw_export_questions']for w in win_reports),
                'independent_questions':sum(w['independent_questions']for w in win_reports),'dedup_dropped':sum(w['dedup_dropped']for w in win_reports),
                'windows':win_reports,'window_overlap_by_member':{m:window_overlap([w['start_s']for w in win_reports if w['member']==m])for m in sorted({w['member']for w in win_reports})}}
            parent_reports.append(report);write(a.output/'parents'/scene/'English_reissue_manifest.json',report)
            print(json.dumps({'scene_id':scene,'status':'pass','raw':report['raw_export_questions'],'independent':report['independent_questions']},ensure_ascii=False),flush=True)
        except Exception as exc:
            del rows[before:];del aligned_rows[aligned_before:];seen=seen_before;raw_family=raw_before;independent_family=independent_before
            errors.append({'scene_id':scene,'error_type':type(exc).__name__,'reason':str(exc)})
            print(json.dumps(errors[-1],ensure_ascii=False),flush=True)
    with (a.output/'train.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    model_bad=[r['question_id']for r in rows if re.search('[\u3400-\u9fff]',json.dumps({k:r.get(k)for k in ['question_en','choices','model_answer']},ensure_ascii=False))]
    aligned_unique={key:family for key,family in aligned_rows}
    report={'schema':'v7-English-reissue-and-independent-coverage-v1','source_summary':str(a.summary),'source_summary_sha256':hashlib.sha256(summary_bytes).hexdigest(),
        'raw_export_questions':sum(r['raw_export_questions']for r in parent_reports),'independent_questions':len(rows),'dedup_dropped':sum(r['dedup_dropped']for r in parent_reports),
        'dedup_rule_version':INDEPENDENCE_REVISION,'raw_family_counts':dict(raw_family),'independent_family_counts':dict(independent_family),
        'aligned_nonoverlap_raw_questions':len(aligned_rows),'aligned_nonoverlap_independent_questions':len(aligned_unique),
        'aligned_nonoverlap_family_counts':dict(Counter(aligned_unique.values())),
        'question_en_CJK_rows':len(model_bad),'missing_question_en_rows':sum(not r.get('question_en')for r in rows),
        'model_visible_CJK_rows':len(model_bad),'unchanged_scorer_checks':scorer_checks,'parents':parent_reports,'errors':errors,
        'status':'pass'if not errors and not model_bad else'partial_or_failed','new_native_or_GPU_renders':0,
        'window_policy':'owner-20261015-half-overlap-0-5-10-15-20' if a.owner_half_grid else 'historical-window-audit',
        'target_1500_independent_questions_met':len(rows)>=1500,'original_data_modified':False,'CCI_transfer':False}
    write(a.output/'PILOT_SUMMARY.json',report)
    print(json.dumps({k:v for k,v in report.items()if k not in ['parents','raw_family_counts','independent_family_counts']},ensure_ascii=False),flush=True)
    if errors or model_bad:raise SystemExit(2)


if __name__=='__main__':main()
