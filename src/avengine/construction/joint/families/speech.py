"""QA-26/27/28: jointly solve real speakers, fixed camera and whole speech.

Native truth remains in qa/. No fixed speaker distance or point bearing is used.
"""
from __future__ import annotations
from copy import copy, deepcopy
from dataclasses import asdict
import itertools
import math
from pathlib import Path
import numpy as np

from avengine.construction.joint import CheckResult

REVISION = 'speech-joint-whole-epochs-v1'
FAMILIES = ('QA-26', 'QA-27', 'QA-28')


def direction(camera, point):
    # Call the actual QA azimuth implementation: its basis is image-right positive.
    from avengine.qa.unified_catalog import _listener_azimuth, _azimuth_sector
    listener = {'positions_m': [list(camera.position_m)], 'yaw_deg': [float(camera.yaw_deg)]}
    angle = _listener_azimuth(list(map(float, point)), listener, 0)
    return _azimuth_sector(angle), angle


def emitter_world(actor, asset):
    from avengine.construction.joint.native import native_yaw
    from avengine.capture.v7_visible_attachment import rotation
    return actor.positions_m[0] + rotation(native_yaw(actor, asset)) @ np.asarray(asset['emitter_local_m'])


class SpeechPlugin:
    def __init__(self, assets, bodies, schedule, sounds, *, max_candidates=512):
        self.assets = {a['asset_id']: a for a in assets}
        self.bodies, self.schedule, self.sounds = bodies, schedule, tuple(sounds)
        self.max_candidates = max_candidates

    def candidates(self, context, request):
        from avengine.construction.joint import enumerate_static_layouts, static_actor
        from avengine.construction.joint.geometry import framing
        from avengine.qa.unified_catalog import _DIRECTION_SECTORS
        ids = tuple(self.assets)
        roles = dict(request.options.get('roles', {aid: 'inside' for aid in ids}))
        # Every legal camera participates. Domains intersect the original sector
        # rules before the core's finite layout beam, rather than filtering luck.
        streams = []
        for camera in context.cameras:
            available, mouth_sectors = {}, {}
            for aid in ids:
                offset = emitter_world(static_actor(aid, [0., 0., 0.], self.bodies[aid]), self.assets[aid])
                sectors = np.asarray([direction(camera, p+offset)[0] for p in context.points_m], object)
                mouth_sectors[aid] = sectors
                frame_states = np.asarray([framing(camera, p, self.bodies[aid])[0]
                                           for p in context.points_m])
                available[aid] = [name for name, _, _ in _DIRECTION_SECTORS
                    if np.any((sectors == name) & (frame_states == roles[aid]))]
            for sectors in itertools.product(*(available[aid] for aid in ids)):
                if len(set(sectors)) != len(sectors):
                    continue
                masks = {aid: np.asarray(mouth_sectors[aid] == sector, bool)
                         for aid, sector in zip(ids, sectors, strict=True)}
                view = copy(context)
                view.cameras = (camera,)
                iterator = enumerate_static_layouts(view, self.bodies, roles,
                    max_candidates=self.max_candidates, point_masks=masks)
                streams.append((iterator, dict(zip(ids, sectors, strict=True))))
        # Fairly visit camera/answer domains; a crowded first camera cannot use
        # the whole construction budget without trying another legal pose.
        yielded = 0
        while streams:
            remaining = []
            for iterator, sectors in streams:
                try:
                    plan = next(iterator)
                except StopIteration:
                    continue
                remaining.append((iterator, sectors))
                plan.sounds = self.sounds
                plan.metadata.update(speech_schedule=deepcopy(self.schedule),
                    requested_speech_sectors=sectors, roles=roles)
                yield plan
                yielded += 1
                if yielded >= self.max_candidates:
                    return
            streams = remaining

    def verify(self, context, request, plan):
        sectors, angles = {}, {}
        for aid, actor in plan.actors.items():
            sectors[aid], angles[aid] = direction(plan.camera, emitter_world(actor, self.assets[aid]))
        promised = plan.metadata['requested_speech_sectors']
        yield CheckResult('speech:distinct_original_QA_sectors',
            'pass' if all(sectors.values()) and len(set(sectors.values())) == len(sectors)
                and sectors == promised else 'fail',
            'actual mouth offset must retain the original non-dead-zone answer domains',
            evidence={'sectors': sectors, 'azimuth_deg': angles, 'requested': promised})
        valid = all(any(w*10 <= e['start_tick'] < e['end_tick'] <= (w+10)*10
                        for w in e['complete_window_starts_s']) for e in self.schedule)
        yield CheckResult('speech:complete_utterance_windows', 'pass' if valid else 'fail',
                          evidence={'schedule': self.schedule})


def build_speech_row(room, house, state, catalog, semantic, contract, plan,
                     people, epochs, schedule, *, scene_id, seed, family):
    from avengine.timeline.v7_family_constructors import _request
    from avengine.timeline.v7_production_inputs import validate_recipe
    ids = [a['asset_id'] for a in people]
    yaw = float(plan.camera.yaw_deg)
    row = {'schema': 'v7-visible-registered-pilot-phase2-v1',
        'scene_id': scene_id, 'independent_parent_scene_id': scene_id,
        'scene_seed': seed, 'house': room['house'], 'room_id': room['room_id'],
        'split': 'train', 'family': room['family'], 'listener_m': list(plan.camera.position_m),
        'camera_yaw_deg': yaw, 'camera_yaw_knots': [{'time_s': t, 'yaw_deg': yaw} for t in (0., 30.)],
        'asset_ids': ids, 'silent_asset_ids': [], 'source_classes': {aid: ['speech_playback'] for aid in ids},
        'motion_requests': {}, 'members': ['v0_a0'], 'trial_only': True,
        'joint_prepared_scene_required': True, 'acceptance_allow_partial_families': True,
        'window_count': 5, 'acceptance_window_starts_s': [0., 5., 10., 15., 20.],
        'questions_per_window': 14, 'question_quota_base': 0,
        'semantic_scenario_id': epochs[0][0]['scenario_id'],
        'semantic_sound_ids': [s['sound_asset_id'] for s in epochs[0]],
        'semantic_human_asset_ids': ids,
        'semantic_start_offsets_s': [e['start_tick']/10 for e in schedule[:len(ids)]],
        'speech_schedule': deepcopy(schedule),
        'speech_epoch_sound_ids': [[s['sound_asset_id'] for s in group] for group in epochs],
        'native_view_roles': {aid: {'view': 'inside', 'static': True,
            'required_visible_sometime': True} for aid in ids},
        'owner_social_layout': {'minimum_human_center_distance_m': 1.,
            'minimum_human_camera_distance_m': 1.5,
            'human_orientation': 'varied registered initial headings'},
        'automatic_construction': {'revision': REVISION, 'family': family,
            'room_key': room['key'], 'candidate_index': 0, 'required_role_count': len(ids),
            'scene_template_files_read': [], 'requested_branch': 'distinct_complete_speech',
            'speech_sectors': plan.metadata['requested_speech_sectors']}}
    _request(row, state, family)
    row['scene_demand']['required_independent_family_counts'] = {f: 1 for f in FAMILIES}
    row['scene_demand']['speech_target_families'] = list(FAMILIES)
    validate_recipe(row, catalog, contract)
    return row


def speech_camera_heights(room, bodies=None):
    """Retain the room's preferred listener height as a joint candidate.

    Legacy placement can have a different height which crops every body.
    Both supplied heights remain proposals; the original core certifies them.
    """
    points = (room.get('preferred_listener_m'), room.get('placement', {}).get('camera_m'))
    heights = tuple(dict.fromkeys(float(p[1])-float(room['measured_floor_y_m'])
                                 for p in points if p is not None))
    if bodies:
        # Derive an additional framing proposal from the actual expanded body
        # envelopes; no fixed camera height or unrelated distance is imposed.
        torso_height = max(body.height_m/2 + body.margin_m for body in bodies.values())
        heights = tuple(dict.fromkeys((*heights, torso_height)))
    return heights or None


def construct_speech_scene(room, house, state, catalog, records, semantic, contract, runtime,
                           *, scene_id, seed, family='QA-28', attempt=0):
    from avengine.construction.joint import (
        BodySpec, FamilyRequest, SoundEvent, build_room_context, solve, PreparedCandidate)
    from avengine.construction.joint.native import prepare_static_native
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable, eligible_humans, stable_seed
    from avengine.timeline.v7_speech_construction import (
        choose_speech_epochs, schedule_epochs, build_speech_scene)
    from avengine.timeline.v7_registered_sampling import semantic_record
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from avengine.qa.unified_catalog import _appearance_family
    if family not in FAMILIES:
        raise ConstructionUnavailable('speech plugin only owns QA-26/27/28')
    pool = sorted(eligible_humans(catalog, room, contract), key=lambda a: (
        state.get('source_entity_usage_counts', {}).get(a['asset_id'], 0),
        stable_seed(seed, 'speech_body', a['asset_id'])))
    if not pool:
        raise ConstructionUnavailable('no legal registered speaking body')
    first = pool[0]
    alternatives = [a for a in pool if _appearance_family(a['color']) != _appearance_family(first['color'])]
    if not alternatives:
        raise ConstructionUnavailable('speech appearance questions need two distinct visible appearances')
    people = [first, alternatives[0]]
    epochs = choose_speech_epochs(semantic, people, state, seed)
    ids = [a['asset_id'] for a in people]
    schedule = schedule_epochs(epochs, ids)
    lookup = {s['sound_asset_id']: s for group in epochs for s in group}
    bodies = {}
    for asset in people:
        vertices = np.asarray(indexed_static_mesh(asset['habitat_glb']).vertices, float)
        bodies[asset['asset_id']] = BodySpec('human',
            radius_m=float(np.linalg.norm(vertices[:, [0, 2]], axis=1).max()),
            height_m=float(np.ptp(vertices[:, 1])))
    sounds = []
    for event in schedule:
        asset = people[ids.index(event['asset_id'])]
        record = semantic_record(lookup[event['sound_asset_id']],
            'F' if asset['visual_gender'] == 'female' else 'M')
        intervals = record['screens']['automatic_activity']['activity_intervals_samples']
        for lo, hi in intervals:
            start = event['start_tick']/10 + lo/16000
            end = event['start_tick']/10 + hi/16000
            sounds.append(SoundEvent(event['asset_id'], event['sound_asset_id'],
                                      'speech_playback', start, end, min(30., end+.5)))
    roles = {aid: 'inside' for aid in ids}
    request = FamilyRequest(family, 'distinct_complete_speech', tuple(ids),
                            {'require_native_preflight': True, 'roles': roles})
    cache = runtime.get('joint_cache_directory')
    if not cache:
        raise ConstructionUnavailable('speech needs a caller-owned joint_cache_directory')
    cache = Path(cache)/'speech'
    actual = next(r for r in house['checked_rooms'] if r['room_label'] == room['room_id'])
    def prepare(context, request, plan):
        row = build_speech_row(room, house, state, catalog, semantic, contract, plan,
            people, epochs, schedule, scene_id=scene_id, seed=seed, family=family)
        def family_check(video, scene, entries):
            from avengine.construction.joint import CameraPose
            sensor = video.sim._sensors['rgb']._sensor_object
            matrix = np.asarray(sensor.node.absolute_transformation(), float)
            camera = CameraPose(tuple(matrix[:3, 3]),
                math.degrees(math.atan2(-matrix[0, 2], matrix[2, 2])),
                width=video.width, height=video.height)
            observed = {}
            for entry in entries:
                asset = entry['declaration'];aid = asset['asset_id']
                transform = np.asarray(entry['object'].root_scene_node.absolute_transformation(), float)
                point = (transform @ np.r_[asset['emitter_local_m'], 1.])[:3]
                observed[aid] = direction(camera, point)[0]
            yield CheckResult('speech:native_original_sectors',
                'pass' if observed == plan.metadata['requested_speech_sectors'] else 'fail',
                evidence={'native_mouth_sectors': observed,
                    'requested': plan.metadata['requested_speech_sectors']})
        return prepare_static_native(context, request, plan, row_factory=lambda _: row,
            house=house, catalog=catalog, records=records, semantic=semantic, runtime=runtime,
            output_dir=cache/'native_preflight'/scene_id, family_check=family_check,
            roles=roles, minimum_visible_pixels={aid: 100 for aid in ids},
            appearance_ids=tuple(ids), scene_factory=build_speech_scene)
    height_candidates = speech_camera_heights(actual, bodies)
    with build_room_context(actual, house=house, runtime=runtime, cache_dir=cache/'geometry',
            camera_heights_m=height_candidates,
            camera_limit=24*len(height_candidates or (None,)), width=1280, height=720) as context:
        solved = solve(context, request, SpeechPlugin(people, bodies, schedule, sounds),
            prepare_and_verify=prepare, budget_s=float(runtime.get('speech_planning_budget_s', 180.)),
            max_candidates=int(runtime.get('speech_max_candidates', 512)),
            max_native_preparations=int(runtime.get('speech_max_native_preparations', 8)))
    from avengine.timeline.continuous_pilot import write_json
    (cache/'planning_results').mkdir(parents=True, exist_ok=True)
    write_json(cache/'planning_results'/(scene_id+'.json'), {
        'status': solved.status, 'reasons': list(solved.reasons), 'diagnostics': solved.diagnostics,
        'certificate': asdict(solved.certificate) if solved.certificate else None})
    if solved.status != 'certified':
        raise ConstructionUnavailable('speech joint construction '+solved.status+': '+str(solved.reasons))
    payload = solved.prepared.payload if isinstance(solved.prepared, PreparedCandidate) else solved.prepared
    payload.row['automatic_construction']['planning_certificate'] = asdict(solved.certificate)
    payload.row['automatic_construction']['joint_solver'] = {'status': solved.status,
        'diagnostics': solved.diagnostics, 'native_execution_must_not_replan': True}
    return {'profile': 'phase2', 'row': payload.row, 'prepared_native': payload}
