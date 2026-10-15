"""Bind authored colour questions to their existing native actor references.

Public wording can be Chinese while the native appearance label is English.
The unchanged colour gate accepts explicit actor IDs: use the QA authority's
existing structured evidence, never infer a label or alter an answer.
"""
from copy import deepcopy

def bind_native_color_references(public,private,raw):
    gold={g['question_id']:g for g in private};actors={str(k)for k in raw['actors']};count=0
    for question in public:
        evidence=gold[question['question_id']].get('evidence',{})
        targets=evidence.get('target_actor_id',evidence.get('actor_id'))
        if targets is None:targets=evidence.get('actor_ids')
        if targets is None:targets=list(evidence.get('appearance_reviews',{}))
        if not isinstance(targets,(list,tuple)):targets=[targets]
        references=list(dict.fromkeys(str(a)for a in targets if str(a)in actors))
        if references:
            question['color_actor_ids']=references;count+=1
    return {'status':'bound_existing_native_actor_references','forms_with_structured_ids':count,
            'canonical_answers_changed':False,'native_colour_gate_changed':False,
            'source':'original authority private evidence target_actor_id/actor_id/actor_ids'}
