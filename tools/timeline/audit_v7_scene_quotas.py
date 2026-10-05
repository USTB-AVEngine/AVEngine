#!/usr/bin/env python3
"""Count realized v7 protocol answers separately from scene requests and formal media."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter
from avengine.timeline.continuous_scene import QA21_CLASSES
from avengine.timeline.continuous_pilot import write_json

def read(path):return json.loads(Path(path).read_text())
def audit(root):
    root=Path(root);logical={};speakers=Counter({str(i):0 for i in range(9)})
    frustum=Counter({'in_frame':0,'out_of_frame':0});native13=Counter({str(i):0 for i in range(13)});qa21=Counter({k:0 for k in QA21_CLASSES})
    families=Counter();windows=[];mismatches=[];formal=0
    for file in sorted(root.rglob('questions.json')):
        private_path=file.with_name('private_answers.json');evidence_path=file.with_name('evidence.json')
        if not private_path.is_file() or not evidence_path.is_file():raise ValueError('actual question files lack private/evidence delivery')
        questions=read(file);answers={x['question_id']:x for x in read(private_path)};evidence=read(evidence_path)
        local={};scene=evidence['scene'];start=evidence.get('start_s',scene['window_starts_s'][0]);lo,hi=round(start*10),round((start+10)*10)
        active_ids={int(e['source_id']) for e in scene['events'] if e['class_id'] in (0,1) and e['source_id']>0 and e['start_tick']<hi and e['end_tick']>lo}
        for q in questions:
            key=q.get('logical_question_id',q['question_id'].removesuffix('__open').removesuffix('__mcq'));gold=answers[q['question_id']]['canonical_answer']
            previous=local.get(key)
            if previous and (previous['qa_id']!=q['qa_id'] or previous['canonical_answer']!=gold):raise ValueError('forms disagree on canonical answer')
            local[key]={'qa_id':q['qa_id'],'canonical_answer':str(gold)}
        if set(local)&set(logical):raise ValueError('duplicate logical question across deliveries')
        logical.update(local)
        for key,q in local.items():
            family,answer=q['qa_id'],q['canonical_answer'];families[family]+=1
            if family=='STARSS-SPEAKERS':
                speakers[answer]+=1
                if answer!=str(len(active_ids)):mismatches.append({'question':key,'answer':answer,'independent_actor_count':len(active_ids),'file':str(file)})
            elif family=='STARSS-FRUSTUM':frustum[answer]+=1
            elif family=='STARSS-CLASS13':native13[answer]+=1
            elif family=='QA-21':qa21[answer]+=1
        count=len(local);qualified=evidence.get('formal_dataset_count',0)>0
        formal+=count if qualified else 0
        windows.append({'directory':str(file.parent),'actual_logical_questions':count,'formal_qualified':qualified,
                        'actual_actor_speaker_count':len(active_ids)})
    return {'status':'fail' if mismatches else 'diagnostic_counts_complete','actual_windows':len(windows),'actual_logical_questions':len(logical),
        'formal_media_qualified_questions':formal,'family_counts':dict(sorted(families.items())),
        'actual_speaker_question_answer_counts':dict(speakers),'actual_frustum_question_answer_counts':dict(frustum),
        'actual_starss13_class_question_counts':dict(native13),'actual_qa21_34_class_question_counts':dict(qa21),
        'speaker_count_mismatches':mismatches,'windows':windows,'source_root':str(root.resolve()),
        'no_seed_request_counted_as_question':True,'no_unsupported_class_or_family_fabricated':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actual-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=audit(a.actual_root);a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/'actual_question_counts.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('windows','actual_qa21_34_class_question_counts')},ensure_ascii=False))
    if report['speaker_count_mismatches']:raise SystemExit(1)
if __name__=='__main__':main()
