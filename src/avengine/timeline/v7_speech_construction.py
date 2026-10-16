"""Whole authored speech clocks and exact per-epoch physical source programs.

Only construction changes here. Native source readbacks and QA remain authority.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import itertools
import math

WINDOWS = (0., 5., 10., 15., 20.)


def full_clip_ticks(sound):
    proof = sound['synthesis_readback']
    count, rate = int(proof['sample_count']), int(proof['sample_rate_hz'])
    if count <= 0 or rate <= 0:
        raise ValueError('complete speech needs its actual positive source clock')
    return (count * 10 + rate - 1) // rate


def schedule_epochs(epochs, asset_ids):
    """Pack complete files into common windows, preferring no partial 5s crops.

    Long valid utterances remain supported. If a pair cannot fit sequentially,
    both complete sources share a window; overlap is recorded, never a new gate.
    """
    result = []
    for epoch_index, sounds in enumerate(epochs):
        if len(sounds) != len(asset_ids):
            raise ValueError('one exact authored sound per physical speaker required')
        base = epoch_index * 100
        lengths = [full_clip_ticks(s) for s in sounds]
        if any(t > 100 for t in lengths) or base + 100 > 300:
            raise ValueError('complete authored file has no 10s scene window')
        total = sum(lengths)
        cell = 50 if total < 50 else 100
        offset = base + (100 - cell)
        overlap = total > cell
        slack = max(0, cell-total)
        cursor = offset + slack//(len(sounds)+1)
        for index, (aid, sound, ticks) in enumerate(zip(asset_ids, sounds, lengths, strict=True)):
            lo = (base + (100-ticks)//2) if overlap else cursor
            hi = lo + ticks
            witnesses = [w for w in WINDOWS if round(w*10) <= lo and hi <= round((w+10)*10)]
            if not witnesses:
                raise ValueError('complete source clock has no full window witness')
            result.append({'event_id': f'speech_epoch_{epoch_index}_role_{index}',
                'asset_id': aid, 'sound_asset_id': sound['sound_asset_id'],
                'scenario_id': sound['scenario_id'], 'role_id': sound['role_id'],
                'start_tick': lo, 'end_tick': hi, 'complete_window_starts_s': witnesses,
                'source_sample_count': sound['synthesis_readback']['sample_count'],
                'source_sample_rate_hz': sound['synthesis_readback']['sample_rate_hz'],
                'overlap_observation': overlap})
            if not overlap:
                cursor = hi + slack//(len(sounds)+1)
    return result


def choose_speech_epochs(manifest, people, state, seed, *, count=3):
    """Match real capabilities and fixed voices before solving geometry."""
    from avengine.timeline.v7_family_constructors import stable_seed, ConstructionUnavailable
    from avengine.timeline.v7_production_inputs import bind_semantic_registry
    from avengine.qa.v7_question_wording import contains_internal_wording
    enriched = bind_semantic_registry(manifest, people)
    roles = {(s['id'], r['role_id']): r for s in enriched['dialogues']['scenarios'] for r in s['roles']}
    scenarios = defaultdict(list)
    for sound in manifest['sounds']:
        if full_clip_ticks(sound) > 100 or contains_internal_wording(sound.get('transcript', '')):
            continue
        scenarios[sound['scenario_id']].append(sound)
    voices = defaultdict(dict)
    for scenario, sounds in sorted(scenarios.items()):
        for pair in itertools.permutations(sounds, len(people)):
            if len({s['role_id'] for s in pair}) != len(people):
                continue
            if len({s['semantic_answer']['answer'] for s in pair}) != len(people):
                continue
            current = tuple(s['voice_reference']['speaker_id'] for s in pair)
            if len(set(current)) != len(current):
                continue
            compatible = True
            for person, sound in zip(people, pair, strict=True):
                gender = person['visual_gender']
                capability = roles[(scenario, sound['role_id'])].get('required_asset_capability', 'standing_human')
                actual = person.get('capability', {}).get('asset_kind', 'standing_human')
                if (sound['voice_reference']['gender'].lower() not in {gender, gender[:1]}
                        or capability not in {'standing_human', actual}):
                    compatible = False
                    break
            if compatible:
                voices[current].setdefault(scenario, pair)
    if not voices:
        raise ConstructionUnavailable('no complete authored roles match the real bodies and distinct fixed voices')
    usage = state.get('authored_voice_counts', {})
    selected, choices = min(voices.items(), key=lambda item: (
        -min(count, len(item[1])), sum(usage.get(v, 0) for v in item[0]),
        stable_seed(seed, 'speech_voices', *item[0])))
    source_use = state.get('source_record_usage_counts', {})
    ordered = sorted(choices.values(), key=lambda pair: (
        sum(source_use.get(s['sound_asset_id'], 0) for s in pair),
        sum(full_clip_ticks(s) for s in pair) > 50,
        stable_seed(seed, 'speech_scenario', *[s['sound_asset_id'] for s in pair])))
    # Rank actual QA-26 query/answer keys, not scenario IDs: two different
    # scenarios can use the same stem and generic canonical meaning_1 code.
    from avengine.timeline.v7_question_independence import normalized_wording
    scenarios_by_id = {s['id']: s for s in enriched['dialogues']['scenarios']}
    seen_queries = [set() for _ in people]
    seen_meanings = [set() for _ in people]
    def signatures(pair):
        return [(normalized_wording({'question_en': scenarios_by_id[s['scenario_id']].get(
                    'question_en', s['scenario_id'])}), str(s['semantic_answer']['answer']),
                 str(s['semantic_answer'].get('label_en', s['semantic_answer']['answer'])))
                for s in pair]
    chosen, remaining = [], list(enumerate(ordered))
    for _ in range(count):
        candidates = remaining or list(enumerate(ordered))
        def score(candidate):
            index, pair = candidate
            keys = signatures(pair)
            return (-sum((stem, answer) not in seen_queries[j] for j, (stem, answer, meaning) in enumerate(keys)),
                    -sum(meaning not in seen_meanings[j] for j, (stem, answer, meaning) in enumerate(keys)), index)
        index, pair = min(candidates, key=score)
        chosen.append(pair)
        remaining = [item for item in remaining if item[0] != index]
        for j, (stem, answer, meaning) in enumerate(signatures(pair)):
            seen_queries[j].add((stem, answer)); seen_meanings[j].add(meaning)
    # Insufficient diversity is observed, never a new source/scene rejection.
    return deepcopy(chosen)


def build_speech_scene(row, house, records, catalog, semantic_manifest):
    """Compile the solved epoch binding, retaining actual complete file samples."""
    from avengine.timeline.v7_phase2_sources import build_phase2_scene
    from avengine.timeline.v7_registered_sampling import semantic_record
    working, scene, program, assets = build_phase2_scene(row, house, records, catalog, semantic_manifest)
    by_asset = {a['asset_id']: a for a in assets}
    sounds = {s['sound_asset_id']: s for s in semantic_manifest['sounds']}
    selected, events, bindings = {}, [], []
    for planned in row['speech_schedule']:
        asset = by_asset[planned['asset_id']]
        sound = sounds[planned['sound_asset_id']]
        record = selected.setdefault(sound['sound_asset_id'], semantic_record(
            sound, 'F' if asset['visual_gender'] == 'female' else 'M'))
        if (record['transcript'] != sound['transcript']
                or planned['end_tick']-planned['start_tick'] != full_clip_ticks(sound)):
            raise ValueError('speech constructor changed the original transcript or full file clock')
        source = scene['sources'][asset['endpoint_id']]
        events.append({'event_id': planned['event_id'], 'endpoint_id': asset['endpoint_id'],
            'source_id': source['actor_id'], 'class_id': source['class_id'],
            'sound_class': 'speech_playback', 'start_tick': planned['start_tick'],
            'end_tick': planned['end_tick'], 'mechanism': source['mechanism']})
        bindings.append({'event_id': planned['event_id'], 'sound_asset_id': sound['sound_asset_id'],
                         'crop_start_sample': 0})
    scene['events'] = sorted(events, key=lambda e: (e['start_tick'], e['event_id']))
    order = {e['event_id']: i for i, e in enumerate(scene['events'])}
    program['records'] = list(selected.values())
    program['events'] = sorted(bindings, key=lambda b: order[b['event_id']])
    scene['speech_schedule'] = deepcopy(row['speech_schedule'])
    scene['automatic_construction'] = deepcopy(row['automatic_construction'])
    scene['independent_parent_scene_id'] = row['independent_parent_scene_id']
    return working, scene, program, assets
