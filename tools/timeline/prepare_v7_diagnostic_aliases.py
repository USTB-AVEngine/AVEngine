#!/usr/bin/env python3
"""Give diagnostic media neutral IDs and quarantine incompatible public domains.

Source questions, answers and media stay unchanged. This is not a production
release: research admission, motion energy, 12--15 questions and quotas remain
blocked. Only valid published-domain question copies enter the alias panel.
"""
from __future__ import annotations
import argparse,json,os,re,uuid
from copy import deepcopy
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json

NAMESPACE=uuid.UUID('e041b025-efae-458a-b09b-80ac5c05b71f')

def read(path):return json.loads(Path(path).read_text())

def within_domain(question,answer):
    domain=question.get('answer_domain')
    if domain is None:return True
    if isinstance(domain,list):return str(answer) in [str(v) for v in domain]
    if domain=='nonnegative_integer':
        try:value=int(answer)
        except (TypeError,ValueError):return False
        return str(value)==str(answer) and value>=0
    match=re.fullmatch(r'integer\[(-?\d+),(-?\d+)\]',str(domain))
    if match:
        try:value=int(answer)
        except (TypeError,ValueError):return False
        return str(value)==str(answer) and int(match[1])<=value<=int(match[2])
    raise ValueError(f'unsupported public domain declaration: {domain}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actual-roots',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);origins=[];deferred=[];emitted=selected=0
    for root in a.actual_roots:
        for file in sorted(root.rglob('questions.json')):
            source=file.parent;questions=read(file);answers={v['question_id']:v for v in read(source/'private_answers.json')};evidence=read(source/'evidence.json')
            address=root.name+'/'+str(source.relative_to(root));wid='w_'+uuid.uuid5(NAMESPACE,address).hex;dest=a.output/wid;dest.mkdir()
            public=[];private=[];mapping={}
            for q in questions:
                emitted+=1;answer=answers[q['question_id']]
                if not within_domain(q,answer['canonical_answer']):
                    deferred.append({'source':str(file),'question_id':q['question_id'],'qa_id':q['qa_id'],
                        'canonical_answer':answer['canonical_answer'],'declared_domain':q['answer_domain'],
                        'reason':'outside_existing_public_domain; source truth untouched'})
                    continue
                logical=q.get('logical_question_id',q['question_id'].removesuffix('__open').removesuffix('__mcq'))
                lid='q_'+uuid.uuid5(NAMESPACE,wid+'/'+logical).hex;qid=lid+'__'+q['form']
                item=deepcopy(q);item.update(question_id=qid,logical_question_id=lid)
                gold=deepcopy(answer);gold['question_id']=qid
                assert gold['canonical_answer']==answer['canonical_answer']
                assert item['question_zh']==q['question_zh'] and item['answer_domain']==q['answer_domain']
                mapping[q['question_id']]=qid;public.append(item);private.append(gold);selected+=1
            for name in ('video.mp4','audio_foa.wav','audio_mono.wav'):
                path=source/name
                if not path.is_file():raise ValueError('native diagnostic media missing')
                os.link(path,dest/name)
            evidence=deepcopy(evidence);evidence['formal_dataset_count']=0
            evidence['neutral_alias_private_origin']=str(source);evidence['source_question_id_to_neutral']=mapping
            evidence['question_text_answers_and_domain_unchanged']=True
            write_json(dest/'questions.json',public);write_json(dest/'private_answers.json',private);write_json(dest/'evidence.json',evidence)
            origins.append({'window_id':wid,'private_origin':str(source),'emitted_source_questions':len(questions),'selected_public_domain_questions':len(public)})
    write_json(a.output/'private_alias_manifest.json',{'status':'diagnostic_only','source_windows':len(origins),
        'source_emitted_questions':emitted,'public_domain_consistent_question_copies':selected,
        'deferred_question_count':len(deferred),'deferred':deferred,'windows':origins,'formal_dataset_count':0,
        'no_source_media_or_truth_modified':True,'public_identifiers_do_not_include_answer_labels':True})
    print(json.dumps({'windows':len(origins),'source_emitted_questions':emitted,'public_domain_consistent':selected,'deferred':len(deferred),'formal_dataset_count':0}))
if __name__=='__main__':main()
