"""Batch modality selection after physical-parent deduplication.

Every input is an already validated native candidate. This module only chooses
which independent rows to export; it cannot change answers or candidate gates.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json

REVISION='independent-parent-modality-30-25-45-family-floor-20261015-v1'
WEIGHTS={'vision':6,'audio':5,'audiovisual':9}


def select_independent_rows(entries, *, minimum_family=3):
    """Keep earliest crop, then 6/5/9 per 20 with family and parent coverage.

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
    pool=list(unique.values())
    capacity=Counter(e['row']['modality']for e in pool)
    if set(capacity)-set(WEIGHTS):raise ValueError('unknown model modality')
    units=min(capacity[m]//n for m,n in WEIGHTS.items())
    if not units:raise ValueError('validated independent pool cannot fill a 30/25/45 batch')
    quotas={m:units*n for m,n in WEIGHTS.items()}
    available_families=Counter(e['row']['qa_id']for e in pool)
    floors={f:min(minimum_family,n)for f,n in available_families.items()}
    used=set();counts=Counter();families=Counter();parents=Counter();selected=[]
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
        return True
    # Reserve scarce families before common families consume a modality budget.
    for family in sorted(byfamily,key=lambda f:(len(byfamily[f]),f)):
        for i in byfamily[family]:
            if families[family]>=floors[family]:break
            take(i)
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
            i=min(candidates,key=lambda j:(families[pool[j]['row']['qa_id']],parents[pool[j]['row']['independent_parent_scene_id']],j))
            candidates.remove(i);take(i)
    selected.sort(key=lambda e:(e['row']['independent_parent_scene_id'],float(e['row']['window_start_s']),e['row']['question_id']))
    report={'revision':REVISION,'input_rows':len(entries),'independent_candidate_rows':len(pool),
        'physical_duplicates_dropped':len(entries)-len(pool),'capacity_by_modality':dict(capacity),
        'selected_rows':len(selected),'selected_by_modality':dict(counts),
        'selected_family_counts':dict(families),'available_family_floors':floors,
        'represented_parents':len(parents),'target_proportions':{m:w/20 for m,w in WEIGHTS.items()},
        'independent_candidates_not_selected':len(pool)-len(selected),
        'source_candidates_and_answers_changed':False}
    return [e['row']for e in selected],report
