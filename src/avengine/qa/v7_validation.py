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
