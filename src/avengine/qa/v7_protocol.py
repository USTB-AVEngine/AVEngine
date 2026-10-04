"""Native simulation questions for the frozen STARSS23 v0.2 panel.

The old v6 families retain their own predicates. This opt-in panel uses 100 ms
half-open annotations, physical actor identities and a complete fixed domain.
No target is substituted when a first-three onset is tied. Geometry is read at
an observed frame, or at identical neighbouring frames, never guessed across
unobserved motion. Public requests contain no evidence or teacher answers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math

import numpy as np

from avengine.timeline.continuous_scene import QA21_CLASSES, DIRECTIONS

CLASS13 = (
    'female speech', 'male speech', 'clapping', 'telephone', 'laughter',
    'domestic sounds', 'footsteps', 'door', 'music', 'musical instrument',
    'water tap', 'bell', 'knock',
)
ACTOR_CLASSES = {0, 1, 2, 4, 6}
SPEECH_CLASSES = {0, 1}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    return value


def annotations(events):
    """Union adjacent bins per (class,source); keep duplicate/conflict evidence."""
    tracks, endpoints = defaultdict(set), defaultdict(set)
    per_frame = defaultdict(list)
    for event in events:
        lo = _integer(event['start_tick'], 'start_tick')
        hi = _integer(event['end_tick'], 'end_tick')
        cls = _integer(event['class_id'], 'class_id')
        source = _integer(event['source_id'], 'source_id')
        if not 0 <= cls < len(CLASS13) or lo < 0 or hi <= lo:
            raise ValueError('invalid class or half-open activity interval')
        mechanism = event.get('mechanism', event.get('emission_mechanism'))
        if cls in SPEECH_CLASSES and mechanism is not None and ('playback' in mechanism or mechanism == 'speaker'):
            raise ValueError('playback speech cannot use a live-speaker class')
        key = (cls, source)
        endpoints[key].add(event['endpoint_id'])
        for tick in range(lo, hi):
            tracks[key].add(tick)
            per_frame[tick].append(key)
    segments = []
    for (cls, source), ticks in tracks.items():
        ordered = sorted(ticks)
        start = last = ordered[0]
        for tick in ordered[1:] + [ordered[-1] + 2]:
            if tick != last + 1:
                endpoint = next(iter(endpoints[(cls, source)])) if len(endpoints[(cls, source)]) == 1 else None
                segments.append({'class_id': cls, 'source_id': source,
                                 'start_tick': start, 'end_tick': last + 1,
                                 'endpoint_id': endpoint})
                start = tick
            last = tick
    return sorted(segments, key=lambda s: (s['start_tick'], s['class_id'], s['source_id'])), per_frame


def window_annotations(events, start_s, duration_s=10.):
    lo, hi = round(start_s * 10), round((start_s + duration_s) * 10)
    if abs(lo / 10 - start_s) > 1e-9 or abs(hi / 10 - start_s - duration_s) > 1e-9:
        raise ValueError('window boundaries must be on the 100 ms grid')
    segments, per_frame = annotations(events)
    clipped = [dict(s, anchor_tick=max(s['start_tick'], lo) - lo,
                    window_end_tick=min(s['end_tick'], hi) - lo,
                    carry_in=s['start_tick'] < lo)
               for s in segments if s['start_tick'] < hi and s['end_tick'] > lo]
    clipped.sort(key=lambda s: (s['anchor_tick'], s['class_id'], s['source_id']))
    frames = [per_frame.get(t, []) for t in range(lo, hi)]
    duplicate = any(len(rows) != len(set(rows)) for rows in frames)
    speech_duplicate = any(len([x for x in rows if x[0] in SPEECH_CLASSES]) !=
                           len({x for x in rows if x[0] in SPEECH_CLASSES}) for rows in frames)
    conflict = any(any(len({cls for cls, src in rows if src == source and cls in SPEECH_CLASSES}) > 1
                       for source in {x[1] for x in rows}) for rows in frames)
    return clipped, {'duplicate_track': duplicate, 'speech_duplicate': speech_duplicate,
                     'speech_class_conflict': conflict}


def _position(track, time_s):
    """Do not interpolate a moving native mouth between retained frames."""
    times = np.asarray([row['time_s'] for row in track], dtype=float)
    values = np.asarray([row['position_m'] for row in track], dtype=float)
    if (not len(times) or values.shape != (len(times), 3) or
            not np.isfinite(values).all() or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0)):
        raise ValueError('invalid native trajectory')
    exact = np.flatnonzero(np.abs(times - time_s) <= 1e-8)
    if len(exact) == 1:
        return values[exact[0]]
    index = int(np.searchsorted(times, time_s))
    if 0 < index < len(times) and np.array_equal(values[index - 1], values[index]):
        return values[index]
    raise ValueError('moving_geometry_at_unobserved_subframe')


def _basis(scene):
    forward = np.asarray(scene['camera_forward'], dtype=float)
    listener = np.asarray(scene['listener_m'], dtype=float)
    if forward.shape != (3,) or listener.shape != (3,) or not np.isfinite(forward).all() or not np.isfinite(listener).all():
        raise ValueError('invalid camera/listener')
    forward[1] = 0
    if np.linalg.norm(forward) < 1e-12:
        raise ValueError('camera has no horizontal direction')
    forward /= np.linalg.norm(forward)
    return listener, forward, np.cross(forward, [0., 1., 0.])


def geometry(scene, segment, time_s):
    listener, forward, right = _basis(scene)
    source = scene['sources'].get(segment['endpoint_id'])
    if source is None or source.get('actor_id') != segment['source_id']:
        raise ValueError('physical_endpoint_identity_missing_or_conflicting')
    delta = _position(source['trajectory'], time_s) - listener
    z = float(delta @ forward)
    angle = (math.degrees(math.atan2(-float(delta @ right), z)) + 180) % 360 - 180
    width, height = scene.get('resolution_wh', [1280, 720])
    hfov = float(scene.get('hfov_deg', 85.))
    focal = width / (2 * math.tan(math.radians(hfov / 2)))
    px = (width - 1) / 2 + focal * float(delta @ right) / z if z > 0 else math.inf
    py = (height - 1) / 2 - focal * float(delta[1]) / z if z > 0 else math.inf
    frustum = 'in_frame' if z > 0 and 0 <= px < width and 0 <= py < height else 'out_of_frame'
    return angle, frustum


def generate_questions(scene, start_s, window_id, media, *, forms=('open', 'mcq'), wording_variant='standard'):
    if wording_variant not in {'standard', 'alternate'}:
        raise ValueError('unknown answer-independent wording variant')
    segments, flags = window_annotations(scene['events'], start_s)
    onset_counts = Counter(s['anchor_tick'] for s in segments)
    public, private, excluded = [], [], []

    def emit(family, key, text, value, domain, evidence, mcq=True):
        if wording_variant == 'alternate':
            replacements = {
                'QA-23': [('这10秒内共有多少个声音活动连续段？', '这10秒的声音活动可以分成多少个连续段？')],
                'STARSS-SPEAKERS': [('这10秒任意时刻发出现场话声的不同说话人有多少位？', '在这10秒里，实际说过话的不同现场说话人共有几位？')],
                'QA-04': [('在起始活动帧的水平方向是什么？', '刚开始活动时位于哪个水平八向？')],
                'QA-25': [('的水平角度是多少？', '相对相机朝向的水平角为多少？')],
                'QA-21': [('属于哪个声音类别？', '按完整声音分类词表应归入哪一类？')],
                'STARSS-CLASS13': [('属于STARSS的哪一类声音？', '在STARSS固定13类中应归入哪一类声音？')],
                'STARSS-FRUSTUM': [('起始活动帧的方向是否位于画面几何视锥内？', '刚开始活动时其方向是否落在画面几何视锥范围内？')],
            }
            for before, after in replacements.get(family, ()):
                text = text.replace(before, after)
        logical = f'{window_id}__v02__{family}__{key}'
        allowed = forms if mcq else tuple(f for f in forms if f == 'open')
        for form in allowed:
            if form not in {'open', 'mcq'}:
                raise ValueError('unsupported question form')
            qid = logical + '__' + form
            row = {'question_id': qid, 'logical_question_id': logical,
                   'qa_id': family, 'protocol_panel': 'STARSS23_v0.2',
                   'modality': 'audio', 'form': form, 'question_zh': text,
                   'answer_domain': deepcopy(domain), 'media': deepcopy(media)}
            answer = str(value)
            if form == 'mcq':
                order = sorted(domain, key=lambda v: hashlib.sha256((qid + '|' + v).encode()).digest())
                row['choices'] = [{'id': str(i + 1), 'text': v} for i, v in enumerate(order)]
                answer = str(order.index(str(value)) + 1)
            public.append(row)
            private.append({'question_id': qid, 'logical_question_id': logical,
                            'canonical_answer': str(value), 'answer': answer,
                            'comparison_key': 'v02__' + family + '__' + key,
                            'evidence': dict(evidence, protocol='STARSS23_v0.2')})

    reliable = not flags['duplicate_track'] and all(s['class_id'] in ACTOR_CLASSES and s['source_id'] > 0 for s in segments)
    if reliable:
        count = len(segments)
        emit('QA-23', 'count', '这10秒内共有多少个声音活动连续段？片头已在响的段计一次，缺一个100毫秒标注帧即另段；不是人数或动作次数。', count,
             [str(i) for i in range(21)] if count <= 20 else 'nonnegative_integer',
             {'count_kind': 'activity_segments', 'segments': segments}, count <= 20)
    else:
        excluded.append({'qa_id': 'QA-23', 'reason': 'non_actor_unreliable_or_duplicate_track'})
    speech = [s for s in segments if s['class_id'] in SPEECH_CLASSES]
    if all(s['source_id'] > 0 for s in speech) and not flags['speech_class_conflict'] and not flags['speech_duplicate']:
        actors = sorted({s['source_id'] for s in speech})
        count = len(actors)
        emit('STARSS-SPEAKERS', 'count', '这10秒任意时刻发出现场话声的不同说话人有多少位？同一人重复讲话只计一位，片头已在讲话者计入；播放讲话不计。', count,
             [str(i) for i in range(9)] if count <= 8 else 'nonnegative_integer',
             {'count_kind': 'distinct_speech_actors', 'actor_ids': actors}, count <= 8)
    else:
        excluded.append({'qa_id': 'STARSS-SPEAKERS', 'reason': 'speech_identity_conflict_or_duplicate'})
    for rank, segment in enumerate(segments[:3], 1):
        if (flags['duplicate_track'] or onset_counts[segment['anchor_tick']] != 1 or
                segment['source_id'] <= 0 or segment['endpoint_id'] is None):
            excluded.append({'candidate': rank, 'reason': 'tied_anchor_duplicate_or_unreliable_identity'})
            continue
        anchor = segment['anchor_tick'] / 10
        description = ('片段开始时已在响的那个声音' if segment['anchor_tick'] == 0 else
                       f'约第{anchor:.1f}秒开始的那个声音')
        evidence = dict(segment, anchor_s=anchor, window_start_s=start_s, candidate_rank=rank)
        mapped = {8: 'music_playback', 10: 'water_tap_faucet'}.get(segment['class_id'])
        if mapped:
            emit('QA-21', f'target{rank}', f'{description}属于哪个声音类别？', mapped, list(QA21_CLASSES), evidence)
        else:
            excluded.append({'qa_id': 'QA-21', 'candidate': rank, 'reason': 'no_strict_v6_class_mapping'})
        emit('STARSS-CLASS13', f'target{rank}', f'{description}属于STARSS的哪一类声音？', CLASS13[segment['class_id']], list(CLASS13), evidence)
        try:
            angle, frustum = geometry(scene, segment, start_s + anchor)
        except ValueError as error:
            excluded.append({'candidate': rank, 'reason': str(error), 'qa_ids': ['QA-04', 'STARSS-FRUSTUM']})
        else:
            sector = math.floor((angle + 22.5) / 45) % 8
            offset = abs((angle + 22.5) % 45 - 22.5)
            if offset < 17.5:
                emit('QA-04', f'target{rank}', f'{description}在起始活动帧的水平方向是什么？相机朝向为正前。', DIRECTIONS[sector], list(DIRECTIONS), dict(evidence, azimuth_deg=angle, sector_dead_zone_deg=5))
            else:
                excluded.append({'qa_id': 'QA-04', 'candidate': rank, 'reason': 'within5deg_sector_boundary'})
            emit('STARSS-FRUSTUM', f'target{rank}', f'{description}起始活动帧的方向是否位于画面几何视锥内？不问遮挡。', frustum, ['in_frame', 'out_of_frame'], dict(evidence, azimuth_deg=angle))
        query = math.ceil(segment["anchor_tick"] / 10)
        if query * 10 >= segment['window_end_tick']:
            excluded.append({'qa_id': 'QA-25', 'candidate': rank, 'reason': 'segment_ends_before_authoritative_integer_query'})
            continue
        try:
            angle, _ = geometry(scene, segment, start_s + query)
        except ValueError as error:
            excluded.append({'qa_id': 'QA-25', 'candidate': rank, 'reason': str(error)})
            continue
        integer = (int(math.floor(angle + .5)) + 180) % 360 - 180
        emit('QA-25', f'target{rank}', f'{description}在第{query}秒的水平角度是多少？正前0°，左正右负，输出[-180,179]整数。', integer, 'integer[-180,179]',
             dict(evidence, query_s=query, azimuth_deg=angle, full_credit_deg=15, half_credit_deg=30), False)
    return public, private, {'protocol': 'STARSS23_v0.2', 'segments': segments,
                             'flags': flags, 'exclusions': excluded}
