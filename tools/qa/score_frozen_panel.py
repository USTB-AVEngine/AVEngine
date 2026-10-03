#!/usr/bin/env python3
"""Sole answer-file reader. Seal or audit scoring truth without running a model."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def require(x,msg):
    if not x:raise ValueError(msg)
def check(members,public):
    groups=defaultdict(list);ids=set()
    for row in members:
        require(row['sample_id'] not in ids,'duplicate answer sample');ids.add(row['sample_id']);groups[row['group']].append(row)
        request=read(public/'groups'/row['sample_id']/'request.json')
        require(request['question']==row['question'],'private/public question differs')
        require(set(request)<= {'sample_id','question','qa_id','form','split','video','foa','binaural','mono','frame_indices','frame_times_s'},'private field in request')
    for group,rows in groups.items():
        require(len(rows)==4 and len({(r['video_index'],r['audio_index']) for r in rows})==4,'not one quartet')
        table={(r['video_index'],r['audio_index']):r['answer_label'] for r in rows}
        require(table[(0,0)]==table[(1,1)] and table[(0,1)]==table[(1,0)] and table[(0,0)]!=table[(0,1)],'answers not checkerboard')
        require(len({r['question'] for r in rows})==1,'question not fixed')
        require(all(r['physical_person_id']==f"{group}:source{r['audio_index']+1}" for r in rows),'audio person routing differs')
        certificate=read(public.parent/'scoring_private/certificates'/f'{group}.json')
        appearances=certificate['joint_solvability']['appearance']
        for row in rows:
            native_label=appearances[row['video_index']]['actors'][f"source{row['audio_index']+1}"]['value'].casefold()
            require(native_label==row['answer_label'],'answer disagrees with native pixel-reviewed clothing')
    return {'status':'pass','groups':len(groups),'members':len(members),'checkerboard_groups':len(groups),
            'answer_files_reader':'score_frozen_panel.py','model_inference':'not_run','model_evaluation':'not_run'}
def seal(root,expected):
    private=root/'scoring_private';public=root/'public';private.chmod(0o700)
    parts=sorted((private/'answers_parts').glob('*.json'));members=[]
    for path in parts:members.extend(read(path)['members'])
    audit=check(members,public);require(audit['groups']==expected,'not all requested groups admitted')
    write(private/'answers.json',{'nonce':secrets.token_hex(32),'members':members})
    requests=[read(public/'groups'/r['sample_id']/'request.json') for r in members];secrets.SystemRandom().shuffle(requests)
    with (public/'requests.jsonl').open('x',encoding='utf8') as f:
        for r in requests:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(private/'seal_audit.json',audit)
    files=sorted(p for p in private.rglob('*') if p.is_file())+[public/'requests.jsonl']
    with (public/'COMMITMENTS.sha256').open('x') as f:
        for path in files:f.write(f'{sha(path)}  {path.relative_to(root)}\n')
    for path in files:path.chmod(0o400 if path.is_relative_to(private) else 0o444)
    for path in public.glob('groups/*/*'):path.chmod(0o444)
    (public/'COMMITMENTS.sha256').chmod(0o444)
    return audit
def audit(root):
    private=root/'scoring_private';result=check(read(private/'answers.json')['members'],root/'public')
    for line in (root/'public/COMMITMENTS.sha256').read_text().splitlines():
        digest,relative=line.split('  ',1);require(sha(root/relative)==digest,'commitment changed')
    return result
def assemble(root,shards,expected):
    require(len(shards)>1 and len(set(shards))==len(shards),'distinct shard roots required')
    require(not root.exists(),'assembly output must be new')
    audits=[audit(s) for s in shards]
    require(sum(x['groups'] for x in audits)==expected,'shard total differs from expected groups')
    configs=[]
    for shard in shards:
        config_path=shard/'inputs/shard_config.json'
        if not config_path.exists():config_path=shard/'inputs/config.json'
        c=read(config_path);c.pop('output',None);configs.append(c)
    require(all(c==configs[0] for c in configs),'shards do not share one production configuration')
    root.mkdir(parents=True)
    for d in ('inputs','evidence','public/groups','scoring_private/certificates','scoring_private/answers_parts'):(root/d).mkdir(parents=True)
    (root/'scoring_private').chmod(0o700)
    write(root/'inputs/shard_config.json',{**configs[0],'output':str(root)})
    ids=set();groups=set();catalog=[]
    for shard,checked in zip(shards,audits):
        for path in sorted((shard/'scoring_private/certificates').glob('*.json')):
            require(path.stem not in groups,'parent group overlaps shards');groups.add(path.stem)
            shutil.copyfile(path,root/'scoring_private/certificates'/path.name)
        for path in sorted((shard/'scoring_private/answers_parts').glob('*.json')):
            shutil.copyfile(path,root/'scoring_private/answers_parts'/path.name)
        for folder in sorted((shard/'public/groups').iterdir()):
            require(folder.name not in ids,'transport member overlaps shards');ids.add(folder.name)
            destination=root/'public/groups'/folder.name;destination.mkdir()
            for path in folder.iterdir():
                require(path.is_file(),'unexpected nested public directory')
                try:os.link(path,destination/path.name)
                except OSError:shutil.copyfile(path,destination/path.name)
        catalog.append({'root':str(shard),'commitment_sha256':sha(shard/'public/COMMITMENTS.sha256'),'audit':checked})
    require(len(groups)==expected and len(ids)==4*expected,'assembly certificates/members incomplete')
    write(root/'scoring_private/shard_catalog.json',{'shards':catalog,'native_provenance':'original sealed shards retained; certificates keep their source native paths'})
    return seal(root,expected)
def score(root,predictions):
    result=audit(root);truth=read(root/'scoring_private/answers.json')['members']
    rows=[json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    values={r['sample_id']:r['prediction_label'] for r in rows}
    require(len(values)==len(rows) and set(values)=={r['sample_id'] for r in truth},'predictions need exactly one row per member')
    require(set(values.values())<={'blue','green','unknown'},'prediction label must be blue, green or unknown')
    by_group=defaultdict(list)
    for r in truth:by_group[r['group']].append(values[r['sample_id']]==r['answer_label'])
    correct=sum(sum(v) for v in by_group.values());all_correct=sum(all(v) for v in by_group.values())
    return {**result,'model_evaluation':'prediction_file_scored','correct_members':correct,'member_accuracy':correct/len(truth),
            'all_four_correct_groups':all_correct,'all_four_accuracy':all_correct/len(by_group),'predictions_sha256':sha(predictions)}
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['seal','audit','assemble','score']);p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-groups',type=int)
    p.add_argument('--shard-root',type=Path,action='append');p.add_argument('--predictions',type=Path)
    a=p.parse_args();require(a.action!='seal' or a.expected_groups is not None,'seal needs expected groups')
    if a.action=='seal':result=seal(a.root,a.expected_groups)
    elif a.action=='assemble':
        require(a.expected_groups is not None and a.shard_root,'assemble needs expected groups and shard roots')
        result=assemble(a.root.resolve(),[x.resolve() for x in a.shard_root],a.expected_groups)
    elif a.action=='score':
        require(a.predictions is not None,'score needs predictions');result=score(a.root,a.predictions)
    else:result=audit(a.root)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
