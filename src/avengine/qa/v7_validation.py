"""Independent per-item v6 authority replay for native v7 family candidates.

This validates truth and offered forms from each retained candidate, rather than
rerunning the candidate-selection pipeline. Separate v0.2 audits can use the
frozen AVModel implementation without adding that checkout as a dependency.
"""
from __future__ import annotations

from copy import deepcopy

from avengine.qa import unified_catalog as u
from avengine.qa.visibility_interpretation import prepare_facts_for_qa


def validate_v6_items(public, private, audit, *, seed='v7'):
    answers = {row['question_id']: row for row in private}
    attempts = {row['question_id']: row for row in audit['v6_generation']['candidate_attempts']
                if row.get('status') == 'pass' and row.get('question_id')}
    retained = {row['question_id']: row for row in audit['v6_items']}
    checked, mismatches = [], []
    cache = {}
    for request in public:
        if request.get('protocol_panel') != 'v6':
            continue
        gold = answers[request['question_id']]
        authority_id = gold['authority_question_id']
        item = retained[authority_id]
        family = request['qa_id']
        if authority_id not in cache:
            facts = u._restore_normalized_frame_keys(audit.get('family_fact_overrides', {}).get(family, audit['facts']))
            context = audit.get('item_sampling_contexts', {}).get(authority_id)
            if context is not None:
                if (family != 'QA-14' or not isinstance(context, dict)
                        or set(context) != {'profile', 'partition_index'}
                        or context['profile'] != 'native_fixed_qa14_query2s_v2'
                        or type(context['partition_index']) is not int
                        or not 0 <= context['partition_index'] < 5
                        or not authority_id.endswith(f"__native_fixed_query2s_{context['partition_index']:02d}")):
                    raise ValueError('unknown per-item native sampling context')
                from avengine.qa.v7_families import native_qa14_query_facts
                facts = native_qa14_query_facts(facts, context['partition_index'])
            if family in u.QA_IDS:
                prepared = prepare_facts_for_qa(deepcopy(facts), family)
                attempt = attempts[authority_id]
                candidates = u._P8_CANDIDATES[family](prepared)
                candidate = next(c for c in candidates if c['candidate_id'] == attempt['candidate_id'])
                expected = u._P8_EMITTERS[family](prepared, candidate, str(seed))
                cache[authority_id] = expected['forms']
            else:
                evidence = item['evidence']
                event = next(e for e in facts['events'] if e['event_id'] == evidence['event_id'])
                actor = event['actor_id']
                if actor != evidence['target_actor_id']:
                    raise ValueError('semantic target actor differs from the materialized event')
                sounds = {s['sound_asset_id']: s for s in audit['semantic_manifest']['sounds']}
                sound = sounds[event['sound_asset_id']]
                if event['transcript'] != sound['transcript']:
                    raise ValueError('semantic transcript differs from the synthesized utterance')
                if family == 'QA-26':
                    value = sound['semantic_answer']['answer']
                elif family == 'QA-27':
                    value = u._reviewed_appearances(facts)[actor]['value']
                elif family == 'QA-28':
                    event = next(e for e in facts['events'] if e['event_id'] == evidence['event_id'])
                    window = u._event_start_sector_window(facts, event)
                    if window is None:
                        raise ValueError('semantic direction has no legal authoritative sector')
                    value = window[1]
                else:
                    raise ValueError('unknown family')
                # Independently recompute the semantic truth; MCQ positions are
                # obtained from the actual authority option labels.
                expected_forms = deepcopy(item['forms'])
                expected_forms['open']['truth'] = value
                if 'mcq' in expected_forms:
                    # Current semantic collection offers open forms only when
                    # four observed distractors are unavailable. A future MCQ
                    # must retain a value-labelled authority option list.
                    options = item.get('options') or item['forms']['mcq'].get('options')
                    if options is None:
                        raise ValueError('semantic MCQ lacks retained value-labelled options for independent replay')
                    correct = next(i for i, option in enumerate(options) if option['value'] == value)
                    expected_forms['mcq']['gold'].update(value=value, correct_index=correct)
                cache[authority_id] = expected_forms
        expected = cache[authority_id][request['form']]
        canonical = expected['truth'] if request['form'] == 'open' else expected['gold']['value']
        answer = canonical if request['form'] == 'open' else chr(65 + expected['gold']['correct_index'])
        ok = canonical == gold['canonical_answer'] and answer == gold['answer']
        row = {'question_id': request['question_id'], 'qa_id': family,
               'canonical_match': canonical == gold['canonical_answer'],
               'form_answer_match': answer == gold['answer'], 'status': 'pass' if ok else 'fail'}
        checked.append(row)
        if not ok:
            mismatches.append(dict(row, expected_canonical=canonical, actual_canonical=gold['canonical_answer'],
                                   expected_answer=answer, actual_answer=gold['answer']))
    return {'method': 'independent candidate replay against unmodified v6 emitters; semantic truth recomputed from roles/appearance/sector',
            'checked_forms': len(checked), 'passed_forms': sum(x['status'] == 'pass' for x in checked),
            'mismatches': mismatches, 'items': checked,
            'status': 'pass' if checked and not mismatches else 'fail' if mismatches else 'not_run'}

def validate_content_items(public, private):
    """Independent table reduction for the separate content-binding panel.

    Does not rerun its emitter/selection functions. Native media qualification
    is checked by collection; this reference recomputes the offered answer.
    """
    by_id={r['question_id']:r for r in private}
    checked=[];mismatches=[]
    for request in public:
        if request.get('protocol_panel')!='v7_content_binding':continue
        gold=by_id[request['question_id']];e=gold['evidence'];scene=e['scene'];query=e['query']
        lo,hi=map(float,scene['window_s'])
        actor={str(a['actor_id']):a for a in scene['actors']}
        eligible=[]
        for source in scene['emissions']:
            if query.get('names_sound_class_in_question') is True and source['content_class']!=query['content_class']:continue
            eligible.append(source)
        if query['kind']=='first_actor':
            onset={}
            for source in eligible:
                if source['mechanism']!='actor_live':continue
                key=str(source['actor_id'])
                for a,b in source['measured_activity_s']:
                    if a<hi and b>lo:onset[key]=min(onset.get(key,float('inf')),max(a,lo))
            first=min(onset.values());keys=[k for k,v in onset.items() if v==first]
            if len(keys)!=1:raise ValueError('independent first-actor tie')
            expected=actor[keys[0]]['descriptor_zh']
        else:
            time=query['time_s'];active=set()
            for source in eligible:
                if any(a<=time<b for a,b in source['measured_activity_s']):
                    active.add(('actor',str(source['actor_id'])) if source['mechanism']=='actor_live'
                               else ('device',str(source['device_id'])))
            if len(active)!=1:raise ValueError('independent physical source is not unique')
            entity,key=next(iter(active))
            if query['kind']=='live_or_playback':expected='现场人物' if entity=='actor' else '音箱播放'
            elif query['kind']=='offscreen_origin':expected='是' if entity=='actor' and key==str(query['subject_actor_id']) else '否'
            else:expected=actor[key]['descriptor_zh']
        form_answer=expected
        if request['form']=='mcq':
            form_answer=next(c['id'] for c in request['choices'] if c['text']==expected)
        ok=expected==gold['canonical_answer'] and form_answer==gold['answer']
        row={'question_id':request['question_id'],'status':'pass' if ok else 'fail',
             'expected_canonical':expected,'expected_form_answer':form_answer}
        checked.append(row)
        if not ok:mismatches.append({**row,'actual_canonical':gold['canonical_answer'],'actual_form_answer':gold['answer']})
    return {'method':'independent half-open emission table / actor set / clipped onset reduction',
            'checked_forms':len(checked),'passed_forms':sum(r['status']=='pass' for r in checked),
            'mismatches':mismatches,'items':checked,'native_qualification_not_replaced':True}
