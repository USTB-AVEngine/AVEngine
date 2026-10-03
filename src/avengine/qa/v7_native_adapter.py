"""Convert actual v7 native readbacks to v6 evidence without filling missing truth.

Modal visible pixels alone cannot classify partial versus complete visibility.
The producer supplies measured modal/target-only truth and appearance reviews.
Likewise a planned event duration is never substituted for dry-source activity.
"""
from __future__ import annotations

from copy import deepcopy
import math

from avengine.timeline.continuous_scene import QA21_CLASSES
from avengine.qa.v7_families import collect_questions, select_questions
from avengine.timeline.v7_production_contract import question_allocation, validate_question_counts


def _crop_pixel_truth(truth, first, count, actor_ids):
    if not truth or truth.get('status') != 'computed_modal_target_only_v1':
        return None
    cropped = deepcopy(truth)
    cropped['per_instance'] = {}
    for actor in actor_ids:
        entry = truth.get('per_instance', {}).get(str(actor))
        if entry is None:
            continue
        rows = []
        for row in entry.get('frames', []):
            frame = row.get('frame_index')
            if isinstance(frame, int) and first <= frame < first + count:
                rows.append(dict(row, frame_index=frame - first))
        cropped['per_instance'][str(actor)] = {**deepcopy(entry), 'frames': rows}
    return cropped


def native_bundle(scene, start_s, window_id, visual, evidence):
    fps = visual['renderer']['fps']
    if fps != scene['video_fps'] or fps != 15:
        raise ValueError('native frame clock must be 15 Hz')
    first = round(start_s * fps)
    if abs(first / fps - start_s) > 1e-8:
        raise ValueError('window start must also be an observed native frame')
    frames = visual['frames'][first:first + 150]
    if len(frames) != 150:
        raise ValueError('window needs 150 native frames')
    ids = [row['actor_id'] for row in frames[0]['actors']]
    if len(ids) != len(set(ids)):
        raise ValueError('native actor IDs are not unique')
    actors, actor_readbacks, emitters = {}, {}, {}
    review = {'actors': {}}
    appearances = evidence.get('appearance_review', {}).get('actors', {})
    for actor in ids:
        key = str(actor)
        rows = []
        for frame in frames:
            matches = [x for x in frame['actors'] if x['actor_id'] == actor]
            if len(matches) != 1:
                raise ValueError('native physical actor identity changed inside the window')
            rows.append(matches[0])
        supplied = deepcopy(appearances.get(key) or {})
        value = supplied.get('value')
        actors[key] = {'asset_id': rows[0]['asset_id'], 'species_id': 'human',
                       'display_label': supplied.get('label_zh', key)}
        if value:
            actors[key]['appearance'] = {'field': 'top_color', 'value': value,
                                         'label_zh': supplied.get('label_zh', value),
                                         'label_en': supplied.get('label_en', value)}
        actor_readbacks[key] = [dict(row, frame_index=i) for i, row in enumerate(rows)]
        emitters[key] = [{'frame_index': i, 'position_m': row['emitter_position_m']}
                         for i, row in enumerate(rows)]
        if supplied.get('status') == 'reviewed':
            supplied['frame_refs'] = [frame - first for frame in supplied.get('frame_refs', [])
                                      if first <= frame < first + 150]
            if supplied['frame_refs']:
                review['actors'][key] = supplied
    sr = scene['sample_rate_hz']
    lo, hi = round(start_s * sr), round((start_s + 10) * sr)
    events, activity = [], {}
    source_evidence = evidence.get('events', {})
    for index, event in enumerate(scene['events']):
        start, end = event['start_tick'] * (sr // 10), event['end_tick'] * (sr // 10)
        if start >= hi or end <= lo:
            continue
        event_id = event.get('event_id', f'event_{index:04d}')
        proof = source_evidence.get(event_id, {})
        # Only explicit source-class provenance can introduce a v6 sound label.
        # Live formant speech is not silently relabelled speech_playback.
        row = {'event_id': event_id, 'actor_id': str(event['source_id']),
               'sound_asset_id': proof.get('sound_asset_id', event_id),
               'start_sample': max(start, lo) - lo,
               'end_sample_exclusive': min(end, hi) - lo,
               'source_endpoint_id': event['endpoint_id']}
        for name in ('sound_class', 'transcript', 'event_unit', 'event_segmentation', 'source_qc'):
            if name in proof:
                row[name] = deepcopy(proof[name])
        events.append(row)
        if proof.get('activity_status') == 'observed' and 'activity_intervals_samples_global' in proof:
            activity[event_id] = [{'start_sample': max(a, lo) - lo, 'end_sample_exclusive': min(b, hi) - lo}
                                  for a, b in proof['activity_intervals_samples_global'] if a < hi and b > lo]
    yaw = math.degrees(math.atan2(scene['camera_forward'][0], -scene['camera_forward'][2]))
    pose = {'position_m': scene['listener_m'], 'yaw_deg': yaw}
    raw = {'episode_id': evidence.get('question_episode_id', window_id),
           'native_window_id': window_id,
           'clock': {'frame_count': 150, 'frame_rate_hz': fps, 'sample_rate_hz': sr, 'sample_count': sr * 10, 'time_base_hz': 48000},
           'actors': actors, 'frame_readbacks': {'actors': actor_readbacks, 'emitters': emitters,
                                                'camera': [dict(pose, frame_index=i) for i in range(150)]},
           'listener': pose, 'appearance_review': review,
           'audio_program': {'events': events}, 'source_activity_intervals_samples': activity,
           'sampling': {'v7_sound_class_options': list(QA21_CLASSES)}}
    truth = _crop_pixel_truth(evidence.get('pixel_visibility_truth'), first, 150, ids)
    if truth is not None:
        raw['pixel_visibility_truth'] = truth
    for key in ('occluder_registry', 'occluder_evidence', 'camera_calibration'):
        if key in evidence:
            raw[key] = deepcopy(evidence[key])
    raw['audio_readback'] = {'wet_tail_intervals': [
        {'event_id': event_id, 'start_s': max(a - start_s, 0.), 'end_s': min(b - start_s, 10.)}
        for event_id, proof in source_evidence.items()
        for a, b in proof.get('wet_intervals_global_s', []) if a < start_s + 10 and b > start_s]}
    return raw


def make_full_questions(scene, start_s, window_id, visual, media, evidence, *,
                        quota_index=0, count=14, family_counts=None, collect_all=False):
    raw = native_bundle(scene, start_s, window_id, visual, evidence)
    public, private, audit = collect_questions(
        raw, window_id, media, seed=str(evidence.get('question_seed', 'v7')),
        foa_readback=evidence.get('foa_readback'), semantic_manifest=evidence.get('semantic_manifest'),
        protocol_scene=evidence.get('protocol_scene'), window_start_s=start_s)
    # Keep private evidence out of the producer's public question metadata.
    audit['native_bundle'] = raw
    if collect_all:
        return public, private, audit
    allocation = question_allocation(quota_index, count)
    questions, answers = select_questions(public, private, allocation,
                                           quota_index=quota_index, family_counts=family_counts)
    check = validate_question_counts(questions, answers, expected_allocation=allocation)
    return questions, answers, {'question_check': check,
                               'candidate_pool_counts': audit['family_counts'],
                               'deferred': audit['deferred'], 'protocol': audit['protocol'],
                               'qualification': audit['qualification'], 'formal_dataset_count': 0}
