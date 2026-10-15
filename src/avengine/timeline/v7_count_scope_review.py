"""Review the owner's visible-entity/all-sounding-people count scope.

The original QA-22 scorer counts sounding members of the visible set. Where
an offscreen source makes that differ from the owner's requested all-source
count, do not relabel its answer. Construct an unambiguous example instead.
"""

def review_qa22_scope(row,facts):
    if row.get('qa_id')!='QA-22':return {'status':'not_applicable'}
    answer=row.get('canonical_answer')
    if not isinstance(answer,(list,tuple))or len(answer)!=2:return {'status':'not_applicable_single_visual_count'}
    actual={str(e['actor_id'])for e in facts.get('events',[])if e.get('actor_id')is not None}
    passed=int(answer[1])==len(actual)
    return {'status':'pass'if passed else'answer_scope_requires_reconstruction',
        'original_canonical_answer':answer,'all_actual_sounding_actor_ids':sorted(actual),
        'owner_requested_sounding_count':len(actual),'original_answer_changed':False,
        'owner_instruction':'20261015 16:0x: count appeared entities visually and include offscreen sounding people'}
