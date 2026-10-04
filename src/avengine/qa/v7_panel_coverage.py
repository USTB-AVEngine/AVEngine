"""Actor-bound v0.2 panel inputs and answer-independent coverage planning.

All identifiers and activity supervision returned here are private. A render
slot, utterance, voice asset, visible body or direction is never an identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import numpy as np


def canonical_speech_scene(scene):
    """Validate an optional native schedule and bind live events to actors."""
    result = deepcopy(scene)
    schedule = scene.get('actor_speech_schedule')
    if schedule is None:
        problems = []
        for event in scene.get('events', []):
            if event.get('class_id') not in {0, 1}:
                continue
            source = scene.get('sources', {}).get(event.get('endpoint_id'), {})
            if (not isinstance(event.get('source_id'), int) or isinstance(event.get('source_id'), bool)
                    or event['source_id'] <= 0 or source.get('actor_id') != event['source_id']):
                problems.append('physical_endpoint_identity_missing_or_conflicting')
            if any(m and m not in {'human_live', 'human_live_formant', 'human_live_tts', 'human_live_recorded'}
                   for m in (event.get('mechanism'), source.get('mechanism'))):
                problems.append('speech_endpoint_is_not_a_live_actor')
        return result, {'mode': 'legacy_physical_actor_ids',
                        'identity_complete': not problems, 'problems': sorted(set(problems))}
    if schedule.get('schema') != 'v7_actor_speech_schedule_v1':
        raise ValueError('unsupported actor speech schedule')
    if schedule.get('activity_complete') is not True:
        raise ValueError('speech activity is incomplete; absence cannot establish zero speakers')
    by_uid, native_ids, video_ids = {}, set(), set()
    for actor in schedule.get('actors', []):
        uid, native = actor.get('actor_uid'), actor.get('native_actor_id')
        endpoint = actor.get('native_endpoint_id')
        if (not isinstance(uid, str) or not uid or uid in by_uid
                or isinstance(native, bool) or not isinstance(native, int) or native <= 0
                or native in native_ids or not actor.get('video_track_id') or actor['video_track_id'] in video_ids
                or not endpoint or not actor.get('native_readback')):
            raise ValueError('actor registry needs unique physical identities and native/video bindings')
        if scene.get('sources', {}).get(endpoint, {}).get('actor_id') != native:
            raise ValueError('registered actor differs from native mouth endpoint')
        by_uid[uid] = actor
        native_ids.add(native)
        video_ids.add(actor['video_track_id'])
    bindings = {}
    for binding in schedule.get('event_bindings', []):
        eid = binding.get('event_id')
        if not eid or eid in bindings:
            raise ValueError('speech event binding is missing or duplicated')
        bindings[eid] = binding
    speech_ids = set()
    supervision = []
    for event in result.get('events', []):
        if event.get('class_id') not in {0, 1}:
            continue
        eid = event.get('event_id')
        if not eid or eid in speech_ids or eid not in bindings:
            raise ValueError('each live speech event requires one unique explicit binding')
        speech_ids.add(eid)
        binding = bindings[eid]
        actor = by_uid.get(binding.get('actor_uid'))
        if actor is None:
            raise ValueError('speech event names an unregistered actor')
        native, endpoint = actor['native_actor_id'], actor['native_endpoint_id']
        if (binding.get('source_slot_id') != event['source_id']
                or binding.get('endpoint_id') != event['endpoint_id']
                or binding.get('class_id') != event['class_id']
                or binding.get('native_actor_id') != native):
            raise ValueError('speech binding conflicts with emitted event')
        mechanism = binding.get('emission_mechanism')
        if mechanism not in {'human_live', 'human_live_formant', 'human_live_tts', 'human_live_recorded'}:
            raise ValueError('a playback asset is not a live-speaking actor')
        if (binding.get('activity_observed') is not True or not binding.get('activity_receipt')
                or not binding.get('sound_asset_id')):
            raise ValueError('speech requires observed dry activity and an identified source asset')
        profile = scene.get('sources', {}).get(event['endpoint_id'], {})
        if any(m and m not in {'human_live', 'human_live_formant', 'human_live_tts', 'human_live_recorded'}
               for m in (event.get('mechanism'), profile.get('mechanism'))):
            raise ValueError('speech binding contradicts the actual playback mechanism')
        if profile.get('actor_id') != native:
            raise ValueError('render slot endpoint is not bound to the claimed physical actor')
        # Aliased rendering endpoints must be the same observed native mouth.
        if event['endpoint_id'] != endpoint and profile.get('trajectory') != scene['sources'][endpoint].get('trajectory'):
            raise ValueError('two render slots disagree on their physical native mouth')
        event.update(source_id=native, endpoint_id=endpoint, mechanism=mechanism)
        supervision.append({'event_id': eid, 'actor_uid': actor['actor_uid'],
                            'native_actor_id': native, 'video_track_id': actor['video_track_id'],
                            'native_endpoint_id': endpoint, 'start_tick': event['start_tick'],
                            'end_tick': event['end_tick'], 'sound_asset_id': binding['sound_asset_id']})
    if set(bindings) != speech_ids:
        raise ValueError('speech binding has an unmatched event')
    return result, {'mode': 'native_actor_schedule', 'identity_complete': True,
                    'registered_actor_count': len(by_uid), 'problems': [],
                    'private_event_actor_bindings': supervision}


def speaker_state_sidecar(scene, start_s=0., duration_s=10.):
    """Private 10 Hz speaking state for every registered actor, including silent."""
    canonical, proof = canonical_speech_scene(scene)
    if not proof['identity_complete']:
        raise ValueError('cannot supervise speaking state without physical identity')
    registry = canonical.get('actor_speech_schedule', {}).get('actors', [])
    ids = sorted({a['native_actor_id'] for a in registry} if registry else
                 {s['actor_id'] for s in canonical.get('sources', {}).values()
                  if isinstance(s.get('actor_id'), int) and s['actor_id'] > 0
                  and s.get('mechanism', '').startswith('human_live')})
    lo, hi = round(start_s * 10), round((start_s + duration_s) * 10)
    states = []
    for tick in range(lo, hi):
        active = {e['source_id'] for e in canonical.get('events', [])
                  if e['class_id'] in {0, 1} and e['start_tick'] <= tick < e['end_tick']}
        states.append({'tick': tick - lo, 'actors': [
            {'native_actor_id': actor, 'is_speaking': actor in active} for actor in ids]})
    return {'schema': 'v7_private_speaking_state_v1', 'private_training_supervision_only': True,
            'identity': proof, 'frames_10hz': states}


def boundary_projection(scene, delta):
    """Project a native point using pixel-boundary coordinates and half-open bounds."""
    delta = np.asarray(delta, dtype=float)
    forward = np.asarray(scene['camera_forward'], dtype=float).copy()
    if delta.shape != (3,) or forward.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(forward).all():
        raise ValueError('invalid native projection geometry')
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        raise ValueError('invalid native camera direction')
    forward /= norm
    right = np.asarray(scene.get('camera_right', np.cross(forward, [0., 1., 0.])), dtype=float)
    if np.linalg.norm(right) < 1e-12:
        raise ValueError('camera right basis missing at vertical view')
    right /= np.linalg.norm(right)
    up = np.asarray(scene.get('camera_up', np.cross(right, forward)), dtype=float)
    up /= np.linalg.norm(up)
    if (abs(float(right @ forward)) > 1e-6 or abs(float(up @ forward)) > 1e-6
            or abs(float(right @ up)) > 1e-6 or not np.isfinite(up).all()):
        raise ValueError('camera basis is not orthonormal')
    width, height = scene.get('resolution_wh', [1280, 720])
    if width <= 0 or height <= 0:
        raise ValueError('invalid image extent')
    calibration = scene.get('camera_calibration', {})
    hfov = float(scene.get('hfov_deg', 85.))
    fx = float(calibration.get('fx', width / (2 * math.tan(math.radians(hfov / 2)))))
    fy = float(calibration.get('fy', fx))
    cx = float(calibration.get('cx', (width - 1) / 2))
    cy = float(calibration.get('cy', (height - 1) / 2))
    convention = calibration.get('coordinate_convention', 'zero_based_pixel_centers')
    if (convention not in {'zero_based_pixel_centers', 'pixel_boundaries'}
            or not all(math.isfinite(v) for v in (fx, fy, cx, cy, float(width), float(height)))
            or fx <= 0 or fy <= 0):
        raise ValueError('invalid intrinsic convention or focal length')
    shift = .5 if convention == 'zero_based_pixel_centers' else 0.
    z = float(delta @ forward)
    if z <= 1e-9:
        return {'frustum': 'out_of_frame', 'depth_m': z,
                'pixel_boundary_xy': None, 'pixel_center_xy': None,
                'resolution_wh': [width, height], 'coordinate_convention': 'pixel_boundaries'}
    u = cx + fx * float(delta @ right) / z + shift
    v = cy - fy * float(delta @ up) / z + shift
    inside = 0 <= u < width and 0 <= v < height
    return {'frustum': 'in_frame' if inside else 'out_of_frame', 'depth_m': z,
            'pixel_boundary_xy': [u, v], 'pixel_center_xy': [u - .5, v - .5],
            'resolution_wh': [width, height], 'coordinate_convention': 'pixel_boundaries',
            'intrinsics_boundary': {'fx': fx, 'fy': fy, 'cx': cx + shift, 'cy': cy + shift},
            'bounds': '[0,width) x [0,height)', 'frustum_dead_zone_px': 0}


def panel_coverage(records, minimum_worlds=2):
    """Count logical questions once; independent worlds exclude camera/count variants."""
    speakers, frustum, worlds = Counter(), Counter(), defaultdict(set)
    angular = Counter()
    seen = set()
    for row in records:
        logical = row['logical_question_id']
        if logical in seen:
            continue
        seen.add(logical)
        if row['qa_id'] == 'STARSS-SPEAKERS':
            count = int(row['canonical_answer'])
            speakers[count] += 1
            worlds[count].add(row['world_id'])
        elif row['qa_id'] == 'STARSS-FRUSTUM':
            answer = row['canonical_answer']
            frustum[answer] += 1
            if 'world_azimuth_deg' in row:
                sector = math.floor((row['world_azimuth_deg'] + 22.5) / 45) % 8
                angular[answer, sector] += 1
    return {'speakers_exact': [{'speakers': n, 'logical_questions': speakers[n],
                               'independent_worlds': len(worlds[n]),
                               'world_ids': sorted(worlds[n]),
                               'meets_world_floor': len(worlds[n]) >= minimum_worlds} for n in range(9)],
            'speakers_5_to_8': sum(speakers[n] for n in range(5, 9)),
            'frustum': {label: frustum[label] for label in ['in_frame', 'out_of_frame']},
            'frustum_world_azimuth_cells': [
                {'answer': label, 'world_sector': n, 'logical_questions': angular[label, n]}
                for label in ['in_frame', 'out_of_frame'] for n in range(8)],
            'counts_are_logical_questions_not_open_mcq_or_paraphrase_exposures': True}


def select_balanced_panels(records, speaker_per_count, frustum_per_cell, *, split=None, minimum_worlds=2):
    """Round-robin worlds, with actual counterfactual pairs selected atomically."""
    selected, rejected, buckets = [], [], defaultdict(list)
    pair_members = defaultdict(list)
    seen = {}
    def key(row):
        if row['qa_id'] == 'STARSS-SPEAKERS':
            return ('speakers', int(row['canonical_answer']))
        if row['qa_id'] == 'STARSS-FRUSTUM' and 'world_azimuth_deg' in row:
            sector = math.floor((row['world_azimuth_deg'] + 22.5) / 45) % 8
            return ('frustum', row['canonical_answer'], sector)
        return None
    for row in records:
        if split is not None and row.get('split') != split:
            continue
        k = key(row)
        if k is None:
            continue
        logical = row['logical_question_id']
        if logical in seen:
            old = seen[logical]
            if old['canonical_answer'] != row['canonical_answer'] or old['world_id'] != row['world_id']:
                raise ValueError('same logical question has conflicting private coverage metadata')
            continue
        seen[logical] = row
        if row.get('pair_id'):
            if k[0] != 'frustum':
                raise ValueError('FRUSTUM pair metadata attached to another panel')
            pair_members[row['pair_id']].append(row)
        else:
            buckets[k].append(row)
    strata = ([('speakers', n) for n in range(9)] +
              [('frustum', label, n) for label in ['in_frame', 'out_of_frame'] for n in range(8)])
    targets = {k: speaker_per_count if k[0] == 'speakers' else frustum_per_cell for k in strata}
    counts = Counter()
    pairs_by_world = defaultdict(list)
    for pid, members in sorted(pair_members.items()):
        validate_counterfactual_pair(members)
        pairs_by_world[members[0]['world_id']].append(members)
    while pairs_by_world:
        for world in sorted(list(pairs_by_world)):
            members = pairs_by_world[world].pop(0)
            demand = Counter(key(r) for r in members)
            if all(counts[k] + n <= targets[k] for k, n in demand.items()):
                selected.extend(members)
                counts.update(demand)
            else:
                rejected.extend(members)
            if not pairs_by_world[world]:
                del pairs_by_world[world]
    for k in strata:
        by_world = defaultdict(list)
        for row in sorted(buckets[k], key=lambda r: r['logical_question_id']):
            by_world[row['world_id']].append(row)
        while by_world:
            for world in sorted(list(by_world)):
                row = by_world[world].pop(0)
                if counts[k] < targets[k]:
                    selected.append(row)
                    counts[k] += 1
                else:
                    rejected.append(row)
                if not by_world[world]:
                    del by_world[world]
    deficits = [{'stratum': list(k), 'target': targets[k], 'selected': counts[k],
                 'missing': targets[k] - counts[k]} for k in strata if counts[k] < targets[k]]
    selected_worlds = defaultdict(set)
    for row in selected:
        if row['qa_id'] == 'STARSS-SPEAKERS':
            selected_worlds[int(row['canonical_answer'])].add(row['world_id'])
    deficits.extend({'stratum': ['speakers_worlds', n], 'target': minimum_worlds,
                     'selected': len(selected_worlds[n]), 'missing': minimum_worlds-len(selected_worlds[n])}
                    for n in range(9) if len(selected_worlds[n]) < minimum_worlds)
    return {'selected': selected, 'not_selected': rejected, 'deficits': deficits,
            'counterfactual_pair_atomicity': True, 'status': 'blocked' if deficits else 'pass',
            'may_publish_as_balanced_panel': not deficits, 'split': split}


def validate_counterfactual_pair(members):
    """Check private native receipts; keep both members or retain neither."""
    if len(members) != 2:
        raise ValueError('counterfactual pair needs exactly two members')
    a, b = members
    for field in ('pair_id', 'world_id', 'split', 'source_actor_uid',
                  'dry_audio_identity', 'activity_identity', 'question_stem_id'):
        if not a.get(field) or a.get(field) != b.get(field):
            raise ValueError('counterfactual pair changes invariant: ' + field)
    if {a.get('canonical_answer'), b.get('canonical_answer')} != {'in_frame', 'out_of_frame'}:
        raise ValueError('counterfactual answers must flip')
    mode = a.get('intervention')
    if mode != b.get('intervention') or mode not in {'camera_orientation_only', 'source_position_only'}:
        raise ValueError('counterfactual intervention is unspecified')
    for row in members:
        if not row.get('native_media_receipt') or not row.get('native_projection_receipt'):
            raise ValueError('counterfactual member lacks actual native media/projection receipt')
    if mode == 'camera_orientation_only':
        for field in ('source_trajectory_identity', 'listener_position_identity', 'world_foa_identity'):
            if not a.get(field) or a.get(field) != b.get(field):
                raise ValueError('camera-only pair changes ' + field)
        if a.get('camera_orientation') == b.get('camera_orientation'):
            raise ValueError('camera-only pair has no intervention')
    else:
        for field in ('camera_orientation', 'listener_position_identity'):
            if a.get(field) is None or a.get(field) != b.get(field):
                raise ValueError('source-only pair changes ' + field)
        if (a.get('source_trajectory_identity') == b.get('source_trajectory_identity')
                or any(not r.get('rir_recomputed_receipt') for r in members)):
            raise ValueError('source-only pair needs changed position and actual recomputed RIR')
    return {'pair_id': a['pair_id'], 'status': 'pass', 'intervention': mode,
            'logical_question_ids': [r['logical_question_id'] for r in members]}
