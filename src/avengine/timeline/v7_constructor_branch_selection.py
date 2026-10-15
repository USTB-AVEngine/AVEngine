"""Prefer the constructor's actual missing branch without changing any gold."""
from copy import deepcopy

POSITIVE={'QA-08':'out_of_view','STARSS-FRUSTUM':'out_of_frame','QA-20':'none_of_visible','QA-24':'out_of_view'}


def restrict_constructed_branch(public,private,demand):
    family=(demand or {}).get('target_family');request=(demand or {}).get('answer_branch_targets',{}).get(family,{})
    branch=request.get('requested_branch')
    if family not in POSITIVE or branch not in {'outside','inside'}:return public,{'applied':False}
    gold={g['question_id']:g for g in private};allowed=set();available=0
    for q in public:
        if q['qa_id']!=family or q.get('form','open')!='open':continue
        g=gold[q['question_id']];value=g.get('canonical_answer')
        if isinstance(value,dict):value=value.get('visibility_state',value.get('final_visibility_state',value.get('value')))
        value=str(value).lower().replace(' ','_')
        outside=value==POSITIVE[family]or family=='QA-20'and value in {'none','none_of_them'}
        available+=1
        if outside==(branch=='outside'):allowed.add(q['logical_question_id'])
    if not allowed:return public,{'applied':False,'family':family,'requested_branch':branch,'reason':'no actual legal requested branch; deficit remains unfulfilled'}
    kept=[q for q in public if q['qa_id']!=family or q['logical_question_id']in allowed]
    return kept,{'applied':True,'family':family,'requested_branch':branch,'legal_open_candidates_before':available,
        'requested_logical_candidates':len(allowed),'answers_changed':False,'truth_or_scorers_changed':False}
