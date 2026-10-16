"""Rule B retains only original passing family/event witnesses.

Failed original checks remain failed in provenance. They are never rewritten
as passing gates. Common physical/clock checks remain mandatory; only the
unavailable family/event promise is withdrawn. Final native QA decides output.
"""
from copy import deepcopy
from dataclasses import asdict


def witness_key(item):
    return str(item['config']['query']['qa_id']),float(item['offsets_s'][0])


def retain_family_witnesses(plan, checks, *, mover, families):
    row=plan.metadata['row'];items=row['motion_during_condition_plan']['plans']
    slots=plan.metadata['animal_route']['emission_slots'];keys={witness_key(item) for item in items}
    mapped={};withdraw=set();observations=[]
    for index,check in enumerate(checks):
        parts=check.name.split(':');key=None
        if check.name.startswith('animal:event:') and len(parts)==4:
            item=items[int(parts[2])];key=(parts[3],float(item['offsets_s'][0]))
        elif check.name.startswith('animal:muzzle_distance:'):
            key=('QA-15',float(slots[int(parts[-1])]['offset_s']))
        elif check.name.startswith('animal:integer_muzzle_query:'):
            key=('QA-16',float(slots[int(parts[-1])]['offset_s']))
        if key is not None:
            mapped[index]=key
            if check.status!='pass':withdraw.add(key)
        # An optional range optimisation is not an answer predicate.
        observation_only=check.name.startswith('animal:nearer_call_range:')
        if (key is not None and check.status!='pass') or observation_only:
            observations.append({'original_check':asdict(check),'witness':list(key) if key else None,
                'action':'withdraw_only_this_family_event' if key else 'record_optional_range_optimisation',
                'rule':'PHASE2 B/C','original_status_preserved':True})
    retained=[item for item in items if witness_key(item) not in withdraw]
    row['motion_during_condition_plan']['plans']=retained
    row['motion_requests'][mover]['motion_during_condition_plan_for_fit']=deepcopy(row['motion_during_condition_plan'])
    retained_keys={witness_key(item) for item in retained}
    row['motion_during_condition_requests']=[item for item in row['motion_during_condition_requests']
        if (item['qa_id'],float(item['offsets_s'][0])) in retained_keys]
    available=[family for family in families if any(qa==family for qa,offset in retained_keys)]
    row['scene_demand']['required_actual_candidate_families']=available
    row['scene_demand']['condition_families']['motion_quartet']=available
    row['motion_quartet_planned_verification']={family:family in available for family in families}
    evidence={'rule':'PHASE2 B','requested_witnesses':len(keys),'retained_witnesses':len(retained_keys),
        'retained_family_events':[{'family':qa,'offset_s':offset} for qa,offset in sorted(retained_keys)],
        'observations':observations,'final_native_QA_still_required':True}
    plan.metadata['family_witness_retention']=evidence
    row['family_witness_retention']=deepcopy(evidence)
    for index,check in enumerate(checks):
        if index in mapped and mapped[index] in withdraw:continue
        if check.name.startswith('animal:nearer_call_range:'):continue
        yield check


def prepare_available_conditions(row, catalog, records):
    """An unavailable original condition withdraws only that family/event.

    The unchanged shared compiler builds the retained set. Unexpected errors
    propagate, rather than being mislabeled as unavailable geometry.
    """
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
    try:return prepare_motion_during_row(row,catalog,records)
    except ValueError as error:
        if not str(error).startswith('conditioned_motion rejected:'):raise
    retained=[];observed=[]
    for item in row['motion_during_condition_requests']:
        one=deepcopy(row);one['motion_during_condition_requests']=[deepcopy(item)]
        try:prepare_motion_during_row(one,catalog,records)
        except ValueError as error:
            if not str(error).startswith('conditioned_motion rejected:'):raise
            observed.append({'family':item['qa_id'],'offsets_s':item['offsets_s'],
                'original_compiler_failure':str(error),'action':'withdraw_only_this_family_event','rule':'PHASE2 B'})
        else:retained.append(deepcopy(item))
    base=deepcopy(row);base['motion_during_condition_requests']=retained
    result=prepare_motion_during_row(base,catalog,records)
    if not retained:
        result['motion_during_condition_plan']={'status':'degraded_no_requested_motion_witness',
            'plans':[],'qa_predicates_changed':False,'native_admission_still_required':True}
    result['motion_condition_compilation_observations']=observed
    return result
