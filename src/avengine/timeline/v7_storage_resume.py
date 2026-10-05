"""Resume a space-stopped supplement without redispatching sealed/rejected parents."""
from __future__ import annotations
from pathlib import Path
import hashlib,json

def initialize_resume(profile,rows,work):
    path=Path(profile['resume_checkpoint']);root=Path(profile['root']).resolve();work=Path(work).resolve()
    if not path.resolve().is_relative_to(root)or work.parent!=root:raise ValueError('resume checkpoint escaped task')
    c=json.loads(path.read_text())
    if c.get('schema')!='v7-space-resume-r19-v1':raise ValueError('unknown resume checkpoint')
    if hashlib.sha256(Path(profile['seeds']).read_bytes()).hexdigest()!=c['seed_SHA256']:raise ValueError('resume seed authority changed')
    groups=[c[k]for k in ('sealed_ids','rejected_ids','interrupted_ids','unstarted_ids')]
    ids=[sid for g in groups for sid in g];actual=[r['scene_id']for r in rows]
    if len(ids)!=len(set(ids))or set(ids)!=set(actual):raise ValueError('resume partition does not cover the original seeds exactly')
    sealed,rejected,interrupted,unstarted=map(set,groups);done=[];failures=[];pending=[]
    for index,row in enumerate(rows):
        sid=row['scene_id'];parent=work/'parents'/sid;pointer=parent/'alpha_seal_completed.json';fail=list(parent.glob('failed_*.json'))
        if sid in rejected:
            if pointer.exists()or not fail:raise ValueError('retained rejection lacks a unique unsealed failure: '+sid)
            recipe=json.loads((parent/'input_recipe.json').read_text())
            if recipe['seed_row']!=row:raise ValueError('rejected seed recipe changed: '+sid)
            failures.append({'scene_id':sid,'status':'retained_previous_gate_rejection','redispatched':False,'failure_receipts':[str(p)for p in fail]});continue
        if pointer.exists():
            p=json.loads(pointer.read_text());recipe=json.loads((parent/'input_recipe.json').read_text());attempt=parent/p['attempt']
            if not attempt.resolve().is_relative_to(parent.resolve())or recipe['seed_row']!=row or p['recipe']!={'input_recipe':recipe}:raise ValueError('sealed resume recipe/scope mismatch: '+sid)
            r=p['result'];count=len(row['members'])*row['window_count']
            if r['status']!='formal_alpha_admitted'or r['alpha_formal_windows']!=count or r['alpha_formal_logical_questions']!=14*count:raise ValueError('sealed resume counts/status differ from original gate: '+sid)
            for rel in p['required_files']:
                f=attempt/rel
                if not f.resolve().is_relative_to(attempt.resolve())or not f.is_file():raise ValueError('sealed resume required file missing/escaped: '+sid)
            done.append({**r,'delivery_directory':str(attempt),'resumed_sealed_delivery':True});continue
        if sid in sealed:raise ValueError('previous formal seal missing: '+sid)
        if sid not in interrupted|unstarted:raise ValueError('unknown pending resume parent: '+sid)
        pending.append((index,row))
    return done,failures,pending
