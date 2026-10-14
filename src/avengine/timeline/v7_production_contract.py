"""Reproducible v7 inputs, question quotas and observed counterfactual checks.

Seed rows are independent parent scenes; members and crops never count as new
parents. The first 400 train rows use the already measured eight-house scope.
Remaining full-set seeds deliberately need a later admitted-house assignment.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Mapping

VERSION = 'v7-production-inputs-20261003-v3'
MODALITIES = ('vision', 'audio', 'audiovisual')
TRAIN_HOUSES = (
    'hm3d_train_00059_kJxT5qssH4H', 'hm3d_train_00096_6HRFAUDqpTb',
    'hm3d_train_00155_iLDo95ZbDJq', 'hm3d_train_00168_bHKTDQFJxTw',
    'hm3d_train_00272_kA2nG18hCAr', 'hm3d_train_00382_S7uMvxjBVZq',
    'hm3d_train_00712_HZ2iMMBsBQ9', 'hm3d_train_00746_RTV2n6fXB2w',
)
DEV_HOUSES = ('hm3d_val_00862_LT9Jq6dN3Ea', 'hm3d_val_00880_Nfvxx8J5NCo')


def derived_seed(split, index, purpose='scene'):
    payload = f'{VERSION}/{split}/{int(index)}/{purpose}'.encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big')


def question_allocation(index, count=14):
    """20-window blocks give exactly 30/25/45% with 14 logical questions."""
    if count not in (12,14):
        raise ValueError('the frozen schedules support 12 or 14 logical questions')
    if count == 12:
        visual=4 if index % 5 < 3 else 3
        return {'vision':visual,'audio':3,'audiovisual':12-visual-3}
    k = index % 20
    visual = 5 if k in (0, 5, 10, 15) else 4
    audio = 4 if k % 2 == 0 else 3
    return dict(zip(MODALITIES, (visual, audio, count-visual-audio), strict=True))


def seed_rows(split, count, admitted_houses):
    if split not in ('train', 'dev', 'test') or count < 0:
        raise ValueError('invalid split or scene count')
    houses = TRAIN_HOUSES if split == 'train' else DEV_HOUSES if split == 'dev' else ()
    fixed = len(houses)*50
    by_id = {h['house']: h for h in admitted_houses}
    if len(by_id) != len(admitted_houses):
        raise ValueError('duplicate physical house in input')
    result = []
    quota_base = 0
    for i in range(count):
        house = houses[i//50] if i < fixed else None
        if house is not None:
            admission = by_id.get(house)
            if not admission or {'development':'dev'}.get(admission['split'],admission['split']) != split or admission['acoustic_status'] != 'pass':
                raise ValueError(f'{house}: missing measured acoustic pass or split mismatch')
            rooms = admission['checked_room_ids']
            if not rooms:
                raise ValueError('admitted house has no measured placement scope')
            room = rooms[derived_seed(split,i,'room') % len(rooms)]
        else:
            room = None
        # One quarter of parents forms a 2x2 group. All members share nuisance
        # factors and remain in the same physical-house split.
        group = i % (4 if i < fixed else 5) == 0
        window_count=2 if i < fixed else 1
        window_seed=derived_seed(split,i,'window')
        starts=[40+2*(window_seed%8),140+2*((window_seed//8)%30)]
        result.append({
            'schema': VERSION, 'scene_id': f'v7_{split}_{i:06d}',
            'split': split, 'scene_index': i, 'scene_seed': derived_seed(split,i),
            'house': house, 'room_id': room,
            'assignment_status': 'admitted' if house else 'awaiting_admitted_house',
            'duration_s': 30, 'window_start_ticks': starts,
            'window_count':window_count, 'questions_per_window':14 if i < fixed else 12,
            'question_quota_base':quota_base,
            'seeds': {p: derived_seed(split,i,p) for p in
                      ('trajectory','camera','appearance','pose','audio','event_schedule','questions','window')},
            'members': ['v0_a0','v0_a1','v1_a0','v1_a1'] if group else ['v0_a0'],
            'counterfactual_plan': 'appearance_x_emitter_association' if group else 'single',
            'shared_factors': ['house','room','materials','camera','background','trajectory','event_times'],
            'allowed_member_changes': ['actor_appearance_assignment','dry_bus_to_emitter_assignment'] if group else [],
        })
        quota_base += window_count*(4 if group else 1)
    return result


def validate_seed_tables(small_train, small_dev, full_train):
    if len(small_train) != 400 or len(small_dev) != 100 or full_train[:400] != small_train:
        raise ValueError('v7-S requires 400/100 parents and exact full-train prefix')
    for rows, split, houses in ((small_train,'train',TRAIN_HOUSES),(small_dev,'dev',DEV_HOUSES)):
        if Counter(x['house'] for x in rows) != Counter({h:50 for h in houses}):
            raise ValueError('each v7-S house requires 50 parents')
        if any(x['split'] != split or x['assignment_status'] != 'admitted' for x in rows):
            raise ValueError('v7-S contains an unadmitted or wrongly split row')
    ids=[x['scene_id'] for x in full_train+small_dev]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate parent scene ID')
    seeds=[x['scene_seed'] for x in full_train+small_dev]
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate parent seed')
    return {'status':'pass','small_train_parents':400,'small_dev_parents':100,
            'train_houses':8,'dev_houses':2,'full_train_seed_rows':len(full_train),
            'prefix_equal':True,'all_seed_values_unique':True,
            'full_train_unassigned_rows':sum(x['house'] is None for x in full_train)}


def validate_question_counts(public_rows, private_rows, *, expected_allocation=None, allow_partial_window=False):
    """Count one logical item once; MCQ/open forms share logical_question_id."""
    answers={x['question_id']:x for x in private_rows}
    if len(answers) != len(private_rows):
        raise ValueError('duplicate private question ID')
    logical={}
    forbidden={'answer','canonical_answer','correct_choice','actor_id','source_id','teacher_answer'}
    def inspect(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError('public question exposes private truth')
            for child in value.values(): inspect(child)
        elif isinstance(value, list):
            for child in value: inspect(child)
    for row in public_rows:
        inspect(row)
        if row['question_id'] not in answers:
            raise ValueError('question has no private evidence answer')
        key=row['logical_question_id']
        semantic=(row['qa_id'],row['modality'],row['question_zh'])
        if key in logical and logical[key] != semantic:
            raise ValueError('forms disagree on logical question identity')
        logical[key]=semantic
    if set(answers) != {x['question_id'] for x in public_rows}:
        raise ValueError('private/public question domains differ')
    n=len(logical)
    minimum = 1 if allow_partial_window else 12
    if not minimum <= n <= 15:
        raise ValueError(f'window has {n} logical questions; required {minimum}..15')
    counts=Counter(x[1] for x in logical.values())
    if set(counts)-set(MODALITIES):
        raise ValueError('unknown modality requirement')
    if expected_allocation is not None and dict(counts) != expected_allocation:
        raise ValueError('realized modality allocation differs from seed schedule')
    return {'status':'pass','logical_questions':n,'offered_forms':len(public_rows),'modality_counts':dict(counts)}


def audit_counterfactuals(members, answer_rows, comparisons):
    """Compare observed private answers; merely declaring a group never passes.

    Each comparison names two actual members and a corresponding logical item.
    nuisance swaps require invariance; emitter-association swaps require change.
    """
    by_member={x['member']:x for x in members}
    if len(by_member)!=len(members) or len(members) not in (2,4):
        raise ValueError('counterfactual group must contain two or four unique members')
    shared=('house','room','materials','camera','background','trajectory','event_times')
    reference=members[0]
    for member in members:
        if any(member[k]!=reference[k] for k in shared):
            raise ValueError('counterfactual shared factor changed')
        if member['split']!=reference['split']:
            raise ValueError('counterfactual group crossed splits')
    answers={(x['member'],x['comparison_key']):x['canonical_answer'] for x in answer_rows}
    if len(answers)!=len(answer_rows):
        raise ValueError('duplicate counterfactual answer key')
    checked=[]
    for c in comparisons:
        left,right=by_member[c['left']],by_member[c['right']]
        changes={k for k in set(left)|set(right) if left.get(k)!=right.get(k)}-{'member'}
        required={'vision':({'audio_nuisance'},{'emitter_association'}),
                  'audio':({'visual_nuisance'},), 'binding':({'emitter_association'},)}.get(c['kind'])
        if required is None or changes not in required:
            raise ValueError('comparison changed factors outside its stated intervention')
        a=answers[(c['left'],c['comparison_key'])];b=answers[(c['right'],c['comparison_key'])]
        success=(a!=b) if c['kind']=='binding' else (a==b)
        checked.append({**c,'pass':success})
    kinds=Counter(x['kind'] for x in checked)
    if any(kinds[k]==0 for k in ('vision','audio','binding')):
        raise ValueError('group lacks observed invariance/change comparisons')
    return {'status':'pass' if all(x['pass'] for x in checked) else 'fail','checks':checked}


def audit_family_floor(logical_rows, minima):
    ids=[x['logical_question_id'] for x in logical_rows]
    if len(set(ids))!=len(ids):
        raise ValueError('family audit requires deduplicated logical questions')
    counts=Counter(x['qa_id'] for x in logical_rows)
    deficits={k:n-counts[k] for k,n in minima.items() if counts[k]<n}
    return {'status':'fail' if deficits else 'pass','counts':dict(counts),'deficits':deficits}



def family_minima_for_policy(policy, scale, split, published_families):
    """Apply every v6 floor and optional protocol-panel floors when present."""
    floor=policy['family_minimum_logical_items'][scale][split]
    minima={family:floor for family in policy['family_scope']}
    published=set(published_families)
    for family,by_split in policy.get('additional_protocol_panel_minimum_when_published',{}).items():
        if family in published and split in by_split:
            minima[family]=by_split[split]
    return minima


def audit_batch_budget(windows, minima, verified_binding_ids=()):
    """Realized question budget; planned group membership earns no credit.

    The calling media/answer group audit supplies IDs of observed changed
    binding answers. Unknown IDs and visual/audio-only IDs are rejected.
    """
    logical={};window_checks=[]
    for public,private in windows:
        window_checks.append(validate_question_counts(public,private))
        local={q['logical_question_id']:q for q in public}
        if set(local).intersection(logical):raise ValueError('logical question reused across windows')
        logical.update(local)
    rows=list(logical.values());n=len(rows);counts=Counter(q['modality'] for q in rows)
    binding=set(verified_binding_ids)
    if binding-set(logical) or any(logical[k]['modality']!='audiovisual' for k in binding):
        raise ValueError('binding audit refers to missing or non-binding questions')
    family=audit_family_floor(rows,minima)
    differences={m:counts[m]-fraction*n for m,fraction in zip(MODALITIES,(.30,.25,.45),strict=True)}
    reasons=[]
    if not n:reasons.append('empty_dataset')
    if max(map(abs,differences.values()),default=0)>1:reasons.append('modality_budget')
    if len(binding)<.20*n:reasons.append('verified_counterfactual_binding_below_20_percent')
    if family['status']!='pass':reasons.append('family_minimum_deficit')
    return {'status':'fail' if reasons else 'pass','logical_questions':n,'windows':len(windows),
            'modality_counts':dict(counts),'modality_rounding_difference':differences,
            'verified_counterfactual_binding_questions':len(binding),
            'counterfactual_binding_fraction':len(binding)/n if n else 0.,
            'family_audit':family,'failure_reasons':reasons}
