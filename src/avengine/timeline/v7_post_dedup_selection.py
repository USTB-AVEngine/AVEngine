"""Batch modality selection after physical-parent deduplication.

Every input is an already validated native candidate. This module only chooses
which independent rows to export; it cannot change answers or candidate gates.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json

REVISION='independent-parent-rounded-modality-30-25-45-family-floor-20261015-v2'
WEIGHTS={'vision':6,'audio':5,'audiovisual':9}


def rounded_modality_quotas(capacity,minimums=None):
    """Largest feasible row count with Hamilton rounding of 30/25/45.

    Each quota differs from its fractional target by less than one question.
    Native candidate capacity and existing family floors remain constraints;
    a batch need not contain a multiple of twenty questions.
    """
    minimums=minimums or {}
    upper=min(sum(capacity.values()),*((20*(capacity.get(m,0)+1)-1)//w for m,w in WEIGHTS.items()))
    lower=max(1,sum(minimums.values()))
    order=list(WEIGHTS)
    for total in range(upper,lower-1,-1):
        quotas={m:total*w//20 for m,w in WEIGHTS.items()}
        remaining=total-sum(quotas.values())
        for m in sorted(order,key=lambda m:(-(total*WEIGHTS[m]%20),order.index(m)))[:remaining]:quotas[m]+=1
        if all(minimums.get(m,0)<=quotas[m]<=capacity.get(m,0)for m in WEIGHTS):return quotas
    raise ValueError('modality budget cannot preserve the available family floor in a rounded 30/25/45 batch')


def select_independent_rows(entries, *, minimum_family=3, family_minimums=None):
    """Keep earliest crop, then rounded 30/25/45 with family and parent coverage.

    An entry contains the public row plus its private physical independence key.
    Keeping the key outside the public row prevents metadata from becoming text
    presented to a model. The source candidate and seal never change.
    """
    entries=sorted(entries,key=lambda e:(float(e['row']['window_start_s']),
        e['row']['window_id'],e['row']['question_id']))
    unique={}
    for entry in entries:
        key=json.dumps(entry['key'],ensure_ascii=False,sort_keys=True)
        previous=unique.get(key)
        if previous is None:unique[key]=entry
        elif previous['row']['canonical_answer']!=entry['row']['canonical_answer']:
            raise ValueError('same physical question has inconsistent canonical answers')
    from avengine.timeline.v7_answer_branch_demand import balance_available_branches
    pool,branch_capacity=balance_available_branches(list(unique.values()),minimum_family=minimum_family)
    capacity=Counter(e['row']['modality']for e in pool)
    if set(capacity)-set(WEIGHTS):raise ValueError('unknown model modality')
    available_families=Counter(e['row']['qa_id']for e in pool)
    floors={f:min(max(minimum_family,(family_minimums or {}).get(f,minimum_family)),n)for f,n in available_families.items()}
    family_modalities=defaultdict(set)
    for e in pool:family_modalities[e['row']['qa_id']].add(e['row']['modality'])
    fixed_floors=Counter()
    for family,modalities in family_modalities.items():
        if len(modalities)==1:fixed_floors[next(iter(modalities))]+=floors[family]
    quotas=rounded_modality_quotas(capacity,fixed_floors)
    from avengine.timeline.v7_answer_branch_demand import branch,selection_priority,summarize
    used=set();counts=Counter();families=Counter();parents=Counter();selected=[];branch_counts=Counter()
    byfamily=defaultdict(list);byparent=defaultdict(list)
    for i,e in enumerate(pool):
        byfamily[e['row']['qa_id']].append(i)
        byparent[e['row']['independent_parent_scene_id']].append(i)
    def take(i):
        if i in used:return False
        r=pool[i]['row'];m=r['modality']
        if counts[m]>=quotas[m]:return False
        used.add(i);selected.append(pool[i]);counts[m]+=1
        families[r['qa_id']]+=1;parents[r['independent_parent_scene_id']]+=1
        branch_counts[(r['qa_id'],branch(r))]+=1
        return True
    # Reserve scarce families before common families consume a modality budget.
    for family in sorted(byfamily,key=lambda f:(len(byfamily[f]),f)):
        while families[family]<floors[family]:
            choices=[i for i in byfamily[family]if i not in used and counts[pool[i]['row']['modality']]<quotas[pool[i]['row']['modality']]]
            if not choices:break
            take(min(choices,key=lambda i:(selection_priority(pool[i]['row'],families,branch_counts),i)))
        if families[family]<floors[family]:
            raise ValueError('modality budget cannot preserve the available family floor: '+family)
    # All admitted parents and houses remain represented in the balanced batch.
    for parent in sorted(byparent,key=lambda p:(len(byparent[p]),p)):
        if parents[parent]:continue
        candidates=[i for i in byparent[parent]if i not in used and counts[pool[i]['row']['modality']]<quotas[pool[i]['row']['modality']]]
        if not candidates:raise ValueError('modality budget cannot preserve native parent: '+parent)
        take(max(candidates,key=lambda i:(quotas[pool[i]['row']['modality']]-counts[pool[i]['row']['modality']],-i)))
    for modality in WEIGHTS:
        candidates=[i for i,e in enumerate(pool)if e['row']['modality']==modality and i not in used]
        while counts[modality]<quotas[modality]:
            i=min(candidates,key=lambda j:(families[pool[j]['row']['qa_id']],selection_priority(pool[j]['row'],families,branch_counts),parents[pool[j]['row']['independent_parent_scene_id']],j))
            candidates.remove(i);take(i)
    selected.sort(key=lambda e:(e['row']['independent_parent_scene_id'],float(e['row']['window_start_s']),e['row']['question_id']))
    report={'revision':REVISION,'input_rows':len(entries),'answer_branch_capacity':branch_capacity,'independent_candidate_rows':len(pool),
        'physical_duplicates_dropped':len(entries)-len(unique),'answer_branch_candidates_not_selected':len(unique)-len(pool),'capacity_by_modality':dict(capacity),
        'modality_quotas':quotas,'quota_rounding':'Hamilton; less than one question per modality',
        'selected_rows':len(selected),'selected_by_modality':dict(counts),
        'selected_family_counts':dict(families),'available_family_floors':floors,
        'represented_parents':len(parents),'target_proportions':{m:w/20 for m,w in WEIGHTS.items()},
        'independent_candidates_not_selected':len(pool)-len(selected),
        'answer_branch_distribution':summarize([e['row']for e in selected]),
        'source_candidates_and_answers_changed':False}
    return [e['row']for e in selected],report
