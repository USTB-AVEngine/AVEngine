"""Full v6/semantic candidate collection for v7; budgets never invent evidence.

The authority modules remain unchanged. The only audio transport extension is
an explicit, measured native camera-relative FOA readback for v7: four channels
stay four channels in the private facts, and this is never a v6 stereo receipt.
Unqualified motion, assets and perception still prevent formal publication.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from avengine.qa import unified_catalog as u
from avengine.qa.semantic_questions import generate_semantic_questions
from avengine.qa.v7_protocol import generate_questions as generate_protocol_questions
from avengine.qa.v7_recorded_audio import natural_speech_blockers

FAMILIES = tuple(f'QA-{i:02d}' for i in range(1, 29))
PANELS = ('STARSS-SPEAKERS', 'STARSS-FRUSTUM', 'STARSS-CLASS13')


def normalize_v7_facts(raw, *, foa_readback=None):
    facts = (u._restore_normalized_frame_keys(raw) if raw.get('schema') == u.UNIFIED_FACT_SCHEMA
             else u.normalize_episode_bundle(raw))
    from avengine.qa.v7_nonvisual_entity_scope import validate_native_nonvisual_registry
    scope = validate_native_nonvisual_registry(raw)
    if scope is not None: facts['native_nonvisual_entity_scope'] = scope
    if foa_readback is None:
        return facts
    if foa_readback.get('format') != 'ACN_WYZX_SN3D_camera':
        raise ValueError('native v7 FOA needs the explicit camera ACN/SN3D format')
    path = Path(foa_readback['path'])
    rate = facts['time']['sample_rate_hz']
    count = facts['time']['sample_count']
    audio, observed_rate = sf.read(path, dtype='float64', always_2d=True)
    if observed_rate != rate or audio.shape != (count, 4) or not np.isfinite(audio).all():
        raise ValueError('native v7 FOA readback disagrees with the window clock')
    review = foa_readback.get('spatial_review') or {}
    if review.get('status') != 'pass' or not review.get('source'):
        raise ValueError('native FOA requires a retained spatial validation source')
    facts['audio'] = {**facts['audio'], 'status': 'pass', 'channel_count': 4,
                      'sample_rate_hz': rate, 'sample_count': count, 'path': str(path),
                      'proof': 'native_v7_FOA_file_readback',
                      'transport': 'v7_camera_FOA', 'format': foa_readback['format'],
                      'spatial_review': deepcopy(review), 'v6_stereo_receipt': False}
    intervals = (raw.get('audio_readback') or {}).get('wet_tail_intervals')
    if intervals is not None:
        checked = []
        for interval in intervals:
            lo, hi = interval.get('start_s'), interval.get('end_s')
            if (not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or
                    not np.isfinite([lo, hi]).all() or lo < 0 or hi <= lo or hi > count / rate + 1e-8):
                raise ValueError('invalid measured native wet interval')
            checked.append(dict(interval, source='native_v7_audio_readback'))
        facts['audio']['wet_tail_intervals'] = checked
    facts['v7_audio_transport_extension'] = 'native_camera_FOA_readback; v6 predicates unchanged'
    return facts


def _semantic_manifest(facts, manifest):
    """Authored words alone cannot certify speech truncated by a pilot loop."""
    if manifest is None:
        return None, [{'qa_id': family, 'code': 'authored_semantic_manifest_missing'}
                      for family in ('QA-26', 'QA-27', 'QA-28')]
    records = {r['sound_asset_id']: r for r in manifest.get('sounds', [])}
    eligible, deferred = [], []
    for event in facts['events']:
        sound = records.get(event.get('sound_asset_id'))
        if sound is None:
            continue
        proof = sound.get('synthesis_readback') or {}
        reasons = natural_speech_blockers(sound)
        path = Path(proof.get('path', ''))
        if not path.is_file():
            reasons.append('actual_synthesized_audio_missing')
        if proof.get('transcript') != sound.get('transcript') or event.get('transcript') != sound.get('transcript'):
            reasons.append('synthesized_transcript_mismatch')
        if proof.get('content_complete') is not True or proof.get('trimmed_or_repeated') is not False:
            reasons.append('speech_content_truncated_repeated_or_unverified')
        if sound.get('script_author') != 'project_author':
            reasons.append('author_script_provenance_missing')
        placement = sound.get('native_placement_readback') or {}
        role = sound.get('dialogue_role_evidence') or {}
        if (placement.get('status') != 'pass' or placement.get('event_id') != event['event_id'] or
                str(placement.get('actor_id')) != str(event.get('actor_id')) or
                placement.get('full_content_preserved') is not True or not placement.get('source')):
            reasons.append('complete_native_utterance_placement_missing')
        if (role.get('role') is None or str(role.get('actor_id')) != str(event.get('actor_id')) or
                role.get('scenario_id') != sound.get('scenario_id') or not role.get('source')):
            reasons.append('native_dialogue_role_evidence_missing')
        if reasons:
            deferred.extend({'qa_id': family, 'event_id': event['event_id'], 'code': reason}
                            for family in ('QA-26', 'QA-27', 'QA-28') for reason in reasons)
            continue
        audio, sr = sf.read(path, dtype='float64', always_2d=True)
        if (not audio.shape[0] or not np.isfinite(audio).all() or
                sr != proof.get('sample_rate_hz') or audio.shape[0] != proof.get('sample_count')):
            raise ValueError('semantic synthesis readback differs from the actual waveform')
        eligible.append(sound)
    if not eligible:
        deferred.extend({'qa_id': family, 'code': 'no_complete_matched_semantic_audio'}
                        for family in ('QA-26', 'QA-27', 'QA-28'))
    filtered = deepcopy(manifest)
    filtered['sounds'] = eligible
    return filtered, deferred


def _modality(item):
    modalities = set(item.get('required_modalities') or [])
    if not modalities:
        modalities = set(u.get_requirements(item['qa_id'])['required_modalities'])
    visual = bool(modalities & {'video', 'pixel_visibility', 'pixel_instance_visibility'})
    audio = bool(modalities & {'audio', 'binaural_audio', 'spatial_audio'})
    return 'audiovisual' if visual and audio else 'vision' if visual else 'audio'


def export_items(items, window_id, media):
    public, private = [], []
    for index, item in enumerate(items):
        # Enumeration IDs are independent of answer, actor and asset identifiers.
        logical = f'{window_id}__v6__{item["qa_id"]}__q{index:05d}'
        for form, request in item.get('model_input', {}).items():
            gold = item['forms'][form]
            qid = logical + '__' + form
            row = {'question_id': qid, 'logical_question_id': logical, 'qa_id': item['qa_id'],
                   'protocol_panel': 'v6', 'modality': _modality(item), 'form': form,
                   'question_zh': request['question_zh'], 'question_en': request.get('question_en', ''),
                   'media': deepcopy(media), 'answer_type': gold['answer_type']}
            if 'options' in request:
                row['choices'] = [{'id': option['option'], 'text': option['label_zh']}
                                  for option in request['options']]
            canonical = gold['truth'] if form == 'open' else gold['gold']['value']
            answer = canonical if form == 'open' else chr(65 + gold['gold']['correct_index'])
            public.append(row)
            private.append({'question_id': qid, 'logical_question_id': logical,
                            'canonical_answer': deepcopy(canonical), 'answer': deepcopy(answer),
                            'evidence': deepcopy(item['evidence']),
                            'authority_question_id': item['question_id'],
                            'comparison_key': item['qa_id'] + '__' + str(item.get('question_variant', '')) + '__' +
                                              str(item.get('evidence', {}).get('query_frame', '')) + '__' +
                                              str(item.get('evidence', {}).get('event_id', '')) + '__' + str(index),
                            'authority_truth': deepcopy(item['truth'])})
    return public, private


def deduplicate_questions(public, private):
    """Count identical published open questions once, retaining their forms."""
    gold = {x['question_id']: x for x in private}
    groups = defaultdict(list)
    for row in public:
        groups[row['logical_question_id']].append(row)
    signatures = {}
    retained = set()
    removed = []
    for logical, forms in groups.items():
        representative = next((r for r in forms if r['form'] == 'open'), forms[0])
        signature = (representative['protocol_panel'], representative['qa_id'],
                     representative['question_zh'])
        canonical = gold[representative['question_id']]['canonical_answer']
        if signature in signatures:
            first, expected = signatures[signature]
            if canonical != expected:
                raise ValueError('identical public question has conflicting truth')
            removed.append({'logical_question_id': logical, 'duplicate_of': first,
                            'qa_id': representative['qa_id'],
                            'reason': 'same_published_question_stem'})
        else:
            signatures[signature] = (logical, canonical)
            retained.add(logical)
    return ([r for r in public if r['logical_question_id'] in retained],
            [g for g in private if g['logical_question_id'] in retained], removed)


def collect_questions(raw, window_id, media, *, seed='v7', items_per_family=5,
                      foa_readback=None, semantic_manifest=None, protocol_scene=None,
                      window_start_s=0., content_binding_bundle=None, native_dialogue_evidence=None,
                      semantic_wording_variant='standard'):
    """Collect every family with the v6 implementation, plus distinct v0.2 panels."""
    facts = normalize_v7_facts(raw, foa_readback=foa_readback)
    result = u.generate_unified_questions(facts, qa_ids=list(u.QA_IDS), seed=str(seed),
                                          items_per_type=items_per_family, include_angle_followups=False)
    family_fact_overrides = {}
    fixed_domain = (facts.get('sampling') or {}).get('v7_sound_class_options')
    if fixed_domain is not None:
        from avengine.timeline.continuous_scene import QA21_CLASSES
        if list(fixed_domain) != list(QA21_CLASSES):
            raise ValueError('v7 QA-21 requires the registered fixed 34-class domain')
        # Invoke the authority's documented fixed-domain mode for QA-21 only.
        # Other families retain the original fact policy and all predicates.
        class_facts = deepcopy(facts)
        policy = class_facts.setdefault('sampling', {}).setdefault('acceptance_policy', {})
        policy.update(question_mode='ordinary_observation', sound_class_options=list(fixed_domain))
        class_result = u.generate_unified_questions(class_facts, qa_ids=['QA-21'], seed=str(seed),
                                                    items_per_type=items_per_family, include_angle_followups=False)
        family_fact_overrides['QA-21'] = class_facts
        result['items'] = [i for i in result['items'] if i['qa_id'] != 'QA-21'] + class_result['items']
        result['deferred'] = [i for i in result['deferred'] if i['qa_id'] != 'QA-21'] + class_result['deferred']
        result['candidate_attempts'] = [i for i in result['candidate_attempts'] if i['qa_id'] != 'QA-21'] + class_result['candidate_attempts']
        result['v7_QA21_fixed_domain_authority_run'] = class_result
    items = list(u.iter_unified_items(result))
    deferred = deepcopy(result['deferred'])
    prepared_manifest=semantic_manifest
    semantic_facts=deepcopy(facts)
    if semantic_manifest is not None and native_dialogue_evidence is not None:
        from avengine.qa.v7_dialogue_binding import attach_dialogue_sources
        try:
            prepared_manifest,attachment_deferred=attach_dialogue_sources(
                semantic_facts,semantic_manifest,native_dialogue_evidence.get('registered_actors',[]),
                native_dialogue_evidence.get('placements',{}),native_dialogue_evidence.get('roles',{}),
                wording_variant=semantic_wording_variant)
            deferred.extend(attachment_deferred)
        except (KeyError,TypeError,ValueError,OSError) as error:
            prepared_manifest={**semantic_manifest,'sounds':[]}
            deferred.extend({'qa_id':f,'code':'registered_native_dialogue_input_invalid','detail':str(error)}
                            for f in ('QA-26','QA-27','QA-28'))
    elif semantic_manifest is not None:
        # A batch author plan cannot bypass actual standing/native attachment.
        prepared_manifest={**semantic_manifest,'sounds':[]}
        deferred.extend({'qa_id':f,'code':'registered_native_dialogue_evidence_missing'}
                        for f in ('QA-26','QA-27','QA-28'))
    manifest, semantic_deferred = _semantic_manifest(semantic_facts, prepared_manifest)
    deferred.extend(semantic_deferred)
    if manifest is not None:
        semantic = generate_semantic_questions(semantic_facts, manifest, variant_id=window_id, seed=str(seed))
        family_fact_overrides.update({f:semantic_facts for f in ('QA-26','QA-27','QA-28')})
        items.extend(semantic['items'])
        deferred.extend(semantic['deferred'])
    public, private = export_items(items, window_id, media)
    protocol = {'status': 'not_run', 'reason': '100ms_protocol_scene_missing'}
    if protocol_scene is not None:
        p, g, protocol = generate_protocol_questions(protocol_scene, window_start_s, window_id, media)
        public.extend(p); private.extend(g)
    content={'status':'not_run','deferred':[]}
    if content_binding_bundle is not None:
        from avengine.qa.v7_content_binding import collect_content_binding
        cp,cg,content=collect_content_binding(content_binding_bundle,window_id,media,seed=seed)
        public.extend(cp);private.extend(cg);deferred.extend(content['deferred'])
    public, private, duplicates = deduplicate_questions(public, private)
    counts = Counter()
    seen = set()
    for row in public:
        logical = row['logical_question_id']
        if logical not in seen:
            counts[row['qa_id']] += 1
            seen.add(logical)
    return public, private, {'family_counts': {f: counts[f] for f in (*FAMILIES, *PANELS, 'V7-CONTENT-BIND')},
                             'duplicate_question_groups': duplicates, 'deferred': deferred, 'v6_generation': result,
                             'v6_items': items, 'facts': facts, 'semantic_manifest': manifest, 'content_binding':content, 'family_fact_overrides': family_fact_overrides, 'protocol': protocol,
                             'formal_dataset_count': 0,
                             'qualification': 'research candidates; native and asset admission remain separate'}


def select_questions(public, private, allocation, *, quota_index=0, family_counts=None):
    """Rotate families before repeated queries; retain exact modality allocations."""
    public, private, _ = deduplicate_questions(public, private)
    gold = {x['question_id']: x for x in private}
    pool = defaultdict(list)
    seen = set()
    for row in public:
        key = row['logical_question_id']
        if key in seen:
            continue
        seen.add(key)
        pool[row['modality']].append(row)
    selection_families=FAMILIES+PANELS+tuple(f for f in ('V7-CONTENT-BIND',) if any(q['qa_id']==f for q in public))
    priority = {f: (i - quota_index) % len(selection_families)
                for i, f in enumerate(selection_families)}
    counts = Counter(family_counts or {})
    selected = []
    for modality, need in allocation.items():
        candidates = pool[modality]
        if len(candidates) < need:
            raise ValueError(f'{modality}: {len(candidates)} legal candidates, need {need}; inspect family deferrals')
        for _ in range(need):
            row = min(candidates, key=lambda q: (counts[q['qa_id']], priority[q['qa_id']], q['logical_question_id']))
            candidates.remove(row); selected.append(row); counts[row['qa_id']] += 1
    return selected, [gold[row['question_id']] for row in selected]


def coverage_projection(seed_rows, observed_counts, policy, *, split, windows_sampled):
    """Empirical extrapolation, never a promise that missing scenes are legal."""
    windows = sum(row['window_count'] * len(row['members']) for row in seed_rows if row['split'] == split)
    floors = {family: policy['family_minimum_logical_items']['v7_S'][split] for family in FAMILIES}
    for panel in PANELS:
        floors[panel] = policy['additional_protocol_panel_minimum_when_published'][panel][split]
    rows = []
    for family in (*FAMILIES, *PANELS):
        estimate = observed_counts.get(family, 0) * windows / windows_sampled if windows_sampled else None
        rows.append({'qa_id': family, 'split': split, 'minimum': floors[family],
                     'sample_count': observed_counts.get(family, 0), 'planned_member_windows': windows,
                     'estimated_questions': estimate, 'estimated_deficit': max(0., floors[family] - estimate) if estimate is not None else None,
                     'formal_admitted_questions': 0,
                     'estimation_status': 'sampled_scene_extrapolation_unverified' if estimate is not None else 'not_estimable'})
    return rows
