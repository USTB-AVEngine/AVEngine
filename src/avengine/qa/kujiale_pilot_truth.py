"""Independent truth audit for sealed Kujiale asset pilot questions.

The v6 families replay their unmodified authority candidates. The new panels
use a separate activity-table and camera-geometry reduction, without calling
their question emitters or the production selection code.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

import numpy as np
from avengine.qa.v7_validation import validate_v6_items, validate_content_items

CLASS13 = (
    'female speech', 'male speech', 'clapping', 'telephone', 'laughter',
    'domestic sounds', 'footsteps', 'door', 'music', 'musical instrument',
    'water tap', 'bell', 'knock',
)
DIRECTIONS = ('front', 'front_left', 'left', 'back_left', 'back', 'back_right', 'right', 'front_right')


def segments_from_ticks(tracks, first_tick, last_tick):
    result = []
    for key, ticks in tracks.items():
        kept = sorted(t for t in ticks if first_tick <= t < last_tick)
        if not kept:
            continue
        lo = previous = kept[0]
        for tick in kept[1:] + [last_tick + 1]:
            if tick != previous + 1:
                result.append((key, lo - first_tick, previous + 1 - first_tick))
                lo = tick
            previous = tick
    return sorted(result, key=lambda x: (x[1], tuple((type(v).__name__, str(v)) for v in x[0])))


def activity_reference(proof, start):
    """Union original measured, actually emitted sample intervals on the 100ms grid."""
    tracks = defaultdict(set)
    for event in proof['events'].values():
        endpoint = event['native_endpoint_id']
        for lo, hi in event['activity_intervals_samples_global']:
            if not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
                raise ValueError('invalid emitted activity samples')
            tracks[(event['sound_class'], endpoint)].update(range(lo // 1600, math.ceil(hi / 1600)))
    return segments_from_ticks(tracks, round(start * 10), round((start + 10) * 10))


def protocol_reference(proof, start):
    tracks = defaultdict(set)
    for event in proof['protocol_scene']['events']:
        tracks[(event['class_id'], event['source_id'], event['endpoint_id'])].update(
            range(event['start_tick'], event['end_tick']))
    return segments_from_ticks(tracks, round(start * 10), round((start + 10) * 10))


def camera_geometry(protocol, endpoint, time):
    """Read a retained point; independently project using the native camera basis."""
    track = protocol['sources'][endpoint]['trajectory']
    positions = [np.asarray(r['position_m'], float) for r in track]
    times = [float(r['time_s']) for r in track]
    exact = [i for i, t in enumerate(times) if abs(t - time) < 1e-8]
    if len(exact) == 1:
        point = positions[exact[0]]
    else:
        i = int(np.searchsorted(times, time))
        if not 0 < i < len(times) or not np.array_equal(positions[i - 1], positions[i]):
            raise ValueError('independent native geometry lacks a measured frame')
        point = positions[i]
    forward = np.asarray(protocol['camera_forward'], float).copy()
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    delta = point - np.asarray(protocol['listener_m'], float)
    angle = (math.degrees(math.atan2(-float(delta @ right), float(delta @ forward))) + 180) % 360 - 180
    full_forward = np.asarray(protocol['camera_forward'], float)
    full_forward /= np.linalg.norm(full_forward)
    right = np.asarray(protocol.get('camera_right', np.cross(full_forward, [0., 1., 0.])), float)
    right /= np.linalg.norm(right)
    up = np.asarray(protocol.get('camera_up', np.cross(right, full_forward)), float)
    up /= np.linalg.norm(up)
    width, height = protocol.get('resolution_wh', [1280, 720])
    calibration = protocol.get('camera_calibration', {})
    focal_x = calibration.get('fx', width / (2 * math.tan(math.radians(protocol.get('hfov_deg', 85.)) / 2)))
    focal_y = calibration.get('fy', focal_x)
    shift = .5 if calibration.get('coordinate_convention', 'zero_based_pixel_centers') == 'zero_based_pixel_centers' else 0.
    cx = calibration.get('cx', (width - 1) / 2) + shift
    cy = calibration.get('cy', (height - 1) / 2) + shift
    depth = float(delta @ full_forward)
    inside = depth > 1e-9 and 0 <= cx + focal_x * float(delta @ right) / depth < width and 0 <= cy - focal_y * float(delta @ up) / depth < height
    return angle, 'in_frame' if inside else 'out_of_frame'


def validate_pilot_items(public, private, audit, evidence):
    answers = {row['question_id']: row for row in private}
    checked = []
    for report in (validate_v6_items(public, private, audit, seed=evidence['source_qa_evidence'].get('question_seed', 'v7')),
                   validate_content_items(public, private)):
        checked.extend(report['items'])
    proof = evidence['source_qa_evidence']
    start = float(evidence['start_s'])
    protocol = proof['protocol_scene']
    segments = protocol_reference(proof, start)
    activities = activity_reference(proof, start)
    starts = defaultdict(set)
    for (sound, endpoint), lo, hi in activities:
        starts[lo].add(endpoint)
    for request in public:
        panel = request.get('protocol_panel')
        if panel in ('v6', 'v7_content_binding'):
            continue
        gold = answers[request['question_id']]
        family = request['qa_id']
        if panel == 'v7_admitted_recording_classes':
            target = gold['evidence']['activity']
            matches = [(sound, endpoint) for (sound, endpoint), lo, hi in activities
                       if endpoint == target['endpoint'] and lo == target['start_tick'] and hi == target['end_tick']]
            if len(matches) != 1 or len(starts[target['start_tick']]) != 1:
                raise ValueError('independent admitted-class onset is ambiguous')
            expected = matches[0][0]
            if expected not in gold['evidence']['domain']:
                raise ValueError('independent class is outside the admitted domain')
        elif panel == 'STARSS23_v0.2':
            if family == 'STARSS-SPEAKERS':
                expected = str(len({key[1] for key, _, _ in segments if key[0] in (0, 1)}))
            elif family == 'QA-23':
                expected = str(len(segments))
            else:
                rank = int(gold['evidence']['candidate_rank'])
                (cls, source, endpoint), onset, end = segments[rank - 1]
                ev = gold['evidence']
                if (cls, source, endpoint, onset, end) != (ev['class_id'], ev['source_id'], ev['endpoint_id'], ev['anchor_tick'], ev['window_end_tick']):
                    raise ValueError('independent STARSS target differs')
                if family == 'STARSS-CLASS13':
                    expected = CLASS13[cls]
                elif family == 'QA-21':
                    expected = {8: 'music_playback', 10: 'water_tap_faucet'}[cls]
                else:
                    query = float(ev['query_s']) if family == 'QA-25' else onset / 10
                    angle, frustum = camera_geometry(protocol, endpoint, start + query)
                    if family == 'QA-25':
                        expected = str((math.floor(angle + .5) + 180) % 360 - 180)
                    elif family == 'QA-04':
                        # Keep the published domain labels; the independent
                        # reference calculates the sector index itself.
                        expected = request['answer_domain'][math.floor((angle + 22.5) / 45) % 8]
                    elif family == 'STARSS-FRUSTUM':
                        expected = frustum
                    else:
                        raise ValueError('unhandled independent protocol family: ' + family)
        else:
            raise ValueError('unhandled independent question panel: ' + str(panel))
        expected_answer = expected
        if request['form'] == 'mcq':
            choices = [c['id'] for c in request['choices'] if c['text'] == expected]
            if len(choices) != 1:
                raise ValueError('independent MCQ value is not uniquely offered')
            expected_answer = choices[0]
        match = expected == gold['canonical_answer'] and expected_answer == gold['answer']
        checked.append({'question_id': request['question_id'], 'qa_id': family,
                        'status': 'pass' if match else 'fail', 'expected_canonical': expected,
                        'expected_answer': expected_answer, 'canonical_match': expected == gold['canonical_answer'],
                        'form_answer_match': expected_answer == gold['answer']})
    identities = [r['question_id'] for r in checked]
    if len(identities) != len(public) or set(identities) != {r['question_id'] for r in public}:
        raise ValueError('independent audit did not cover every retained form exactly once')
    return {'status': 'pass' if all(r['status'] == 'pass' for r in checked) else 'fail',
            'checked_forms': len(checked), 'passed_forms': sum(r['status'] == 'pass' for r in checked),
            'mismatches': [r for r in checked if r['status'] != 'pass'], 'items': checked,
            'method': 'unmodified v6 authority candidate replay; separate 100ms activity union, physical actor set, native camera projection and MCQ value lookup'}


def audit_sealed_parent(parent, delivery):
    """Audit each retained form after immutable native delivery has been made."""
    from pathlib import Path
    import json
    from avengine.timeline.continuous_pilot import write_json
    parent, delivery = Path(parent), Path(delivery)
    if parent not in delivery.parents:
        raise ValueError('independent pilot audit escaped its parent directory')
    reports = []
    for folder in sorted((delivery / 'windows').iterdir()):
        window_id = folder.name
        public = json.loads((folder / 'questions.json').read_text())
        private = json.loads((delivery / 'private' / (window_id + '_answers.json')).read_text())
        evidence = json.loads((delivery / 'private' / (window_id + '_evidence.json')).read_text())
        pointer = json.loads((parent / ('windows_' + evidence['member'] + '_completed.json')).read_text())
        audit = json.loads((parent / pointer['attempt'] / window_id / 'candidate_audit.json').read_text())
        report = validate_pilot_items(public, private, audit, evidence)
        reports.append(dict(report, window_id=window_id))
    report = {'status': 'pass' if reports and all(r['status'] == 'pass' for r in reports) else 'fail',
              'checked_forms': sum(r['checked_forms'] for r in reports),
              'passed_forms': sum(r['passed_forms'] for r in reports),
              'mismatches': [x for r in reports for x in r['mismatches']],
              'windows': reports, 'delivery_directory': str(delivery)}
    path = parent / 'independent_truth_audit.json'
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError('completed independent truth report differs')
    else:
        write_json(path, report)
    if report['status'] != 'pass':
        raise ValueError('independent retained question truth does not match')
    return report
