"""Answer-branch debts from already parent-deduplicated model rows only."""
from collections import Counter,defaultdict
import json

REVISION='independent-answer-branch-demand-v1'
TARGETS={'QA-08':('out_of_view',.25,.35),'STARSS-FRUSTUM':('out_of_frame',.30,.40),
    'QA-20':('none_of_visible',.20,.30),'QA-24':('out_of_view',.15,1.)}


def branch(row):
    family=row['qa_id'];value=row.get('canonical_answer',row.get('model_answer'))
    if isinstance(value,dict):
        value=value.get('visibility_state',value.get('final_visibility_state',value.get('answer',value.get('value',value))))
    if isinstance(value,(dict,list)):value=json.dumps(value,sort_keys=True)
    value=str(value).strip().lower().replace(' ','_').replace('-','_')
    if family=='QA-20':return 'none_of_visible'if value in {'none_of_visible','none_of_them','none'}else'visible_person'
    return value


def summarize(rows):
    counts=defaultdict(Counter);seen=set()
    for row in rows:
        if row['question_id']in seen:raise ValueError('answer branch feedback needs unique independent rows')
        seen.add(row['question_id'])
        if row['qa_id']in TARGETS or row['qa_id']in {'QA-04','QA-13','QA-25','QA-28'}:counts[row['qa_id']][branch(row)]+=1
    result={}
    for family,(positive,lo,hi)in TARGETS.items():
        n=sum(counts[family].values());p=counts[family][positive]
        result[family]={'answers':dict(counts[family]),'total_independent_questions':n,'positive_answer':positive,'positive_questions':p,
            'positive_fraction':p/n if n else None,'target_fraction':[lo,hi],'target_met':bool(n and lo<=p/n<=hi)}
    return {'revision':REVISION,'denominator':'already parent-deduplicated exported model rows','families':result,
        'direction_answers':{f:dict(counts[f])for f in ['QA-04','QA-13','QA-25','QA-28']}}


def construction_request(state,next_questions=3):
    observed=state.get('answer_branch_distribution',{}).get('families',{});debts={}
    for family,(positive,lo,hi)in TARGETS.items():
        item=observed.get(family,{});n=item.get('total_independent_questions',0);p=item.get('positive_questions',0)
        goal=lo if hi==1 else(lo+hi)/2
        debts[family]=max(0.,goal*(n+max(1,next_questions))-p)
    count=2 if max(debts.values(),default=0)>=1 else 1
    return {'answer_branch_deficits':debts,'minimum_people':max(3,count+2),'maximum_people':5,
        'required_offscreen_sounding_people':count,'minimum_visible_people':2,
        'offscreen_entities_must_be_real_registered_bodies':True,'camera_must_remain_fixed':True,
        'preferred_offscreen_directions':['left_side','right_side','behind_left','behind_right'],
        'speaker_count_includes_offscreen':True,'visible_entity_count_requires_actual_pixels':True,
        'target_fractions':{f:[lo,hi]for f,(_,lo,hi)in TARGETS.items()}}


def selection_priority(row,family_counts,branch_counts):
    if row['qa_id']not in TARGETS:return 0.
    positive,lo,hi=TARGETS[row['qa_id']];goal=lo if hi==1 else(lo+hi)/2
    debt=goal*(family_counts[row['qa_id']]+1)-branch_counts[(row['qa_id'],positive)]
    return -debt if branch(row)==positive else debt
