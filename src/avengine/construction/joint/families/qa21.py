"""Construct QA-21 from a chosen class and its real, uniquely visible emitter.

Geometry belongs to joint_core. Native masks and the original QA authority
remain the only truth source. These functions never assign visibility truth.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np

REVISION = 'qa21-target-first-20261016-v1'
MINIMUM_TARGET_PIXELS = 300
FRAME_MARGIN_PX = 8.


def emission_mechanism(asset, sound_class):
    """Use the same actual source binding as the existing phase-2 producer."""
    if sound_class not in asset.get('allowed_classes', ()):
        raise ValueError('requested class is not a registered emitter capability')
    kind = asset.get('kind')
    if kind == 'animal':
        species = asset.get('species_id')
        if (species, sound_class) not in {('cat', 'cat_meow'), ('dog', 'dog_bark')}:
            raise ValueError('QA21 animal class does not describe its actual species')
        return 'animal_live_recorded'
    if asset.get('entity_class') == 'loudspeaker':
        if sound_class not in {'speech_playback', 'music_playback'}:
            raise ValueError('a loudspeaker is not an intrinsic appliance sound source')
        return 'loudspeaker_playback'
    if kind == 'object' and not asset.get('silent_decor'):
        return 'device_recorded_event'
    raise ValueError('no actual registered QA21 emission mechanism')


def target_options(room, catalog, records, contract, state, *, requested_class=None,
                   floor_only=True):
    """Enumerate admitted class/body pairs before choosing any camera or route."""
    from avengine.timeline.continuous_scene import QA21_CLASSES
    from avengine.timeline.v7_production_inputs import asset_category
    from avengine.qa.unified_catalog import _sound_class_phrases
    available = {r['sound_class'] for r in records if r.get('sound_class')}
    excluded = contract.get('policy', {}).get('temporarily_excluded_asset_ids', {})
    result = []
    for asset in catalog:
        if asset['asset_id'] in excluded or asset.get('silent_decor'):
            continue
        if not asset.get('label_color') or asset.get('standing_registered_qualified') is not True:
            continue
        if floor_only and asset.get('attachment_surface') != 'floor':
            continue
        if asset.get('kind') not in {'animal', 'object'}:
            continue
        if asset_category(asset, {}, room['family']) not in room['categories']:
            continue
        for cls in sorted(set(asset.get('allowed_classes', ())) & available & set(QA21_CLASSES)):
            if requested_class is not None and cls != requested_class:
                continue
            try:
                mechanism = emission_mechanism(asset, cls)
                label = _sound_class_phrases(cls)[0]
            except ValueError:
                continue
            result.append((asset, cls, {'actual_asset_id': asset['asset_id'],
                'selected_class': cls, 'actual_mechanism': mechanism,
                'attachment_surface': asset['attachment_surface'],
                'recording_count': sum(r.get('sound_class') == cls for r in records),
                'display_label': label, 'expected_reviewed_appearance': asset['label_color'],
                'registered_fixed_domain_size': len(QA21_CLASSES)}))
    positives = state.get('qa21_positive_counts', state.get('class34_positive_counts', {}))
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in positives.values()):
        raise ValueError('positive class feedback must be nonnegative delivered counts')
    result.sort(key=lambda t: (positives.get(t[1], 0),
        state.get('source_entity_usage_counts', {}).get(t[0]['asset_id'], 0), t[1], t[0]['asset_id']))
    return result


def body_spec(asset):
    """Build the core envelope from the original admitted mesh, with no recolour."""
    from avengine.construction.joint import BodySpec
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    mesh = indexed_static_mesh(asset['habitat_glb'])
    vertices = np.asarray(mesh.vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('original emitter mesh has no finite geometry')
    radius = float(np.max(np.linalg.norm(vertices[:, [0, 2]], axis=1)))
    height = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    species = asset.get('species_id') or ('human' if asset.get('kind') == 'human' else 'slide')
    return BodySpec(species=species, radius_m=max(.04, radius), height_m=max(.05, height), margin_m=.06)


class Qa21Plugin:
    def __init__(self, bodies, roles, target_id, sound_class, *, sounds=(), maximum=256, targets=None, assets=None):
        self.bodies, self.roles = bodies, roles
        self.target_id, self.sound_class = target_id, sound_class
        self.sounds, self.maximum = tuple(sounds), maximum
        self.targets = targets or {target_id: sound_class}
        self.assets = assets or {}
        self._mesh_raycasters = {}

    def candidates(self, context, request):
        from avengine.construction.joint import enumerate_static_layouts
        for plan in enumerate_static_layouts(context, self.bodies, self.roles, max_candidates=self.maximum):
            yield replace(plan, sounds=self.sounds, metadata={**plan.metadata,
                'qa21_target_id': self.target_id, 'qa21_sound_class': self.sound_class})

    def exact_prediction(self, context, plan, target_id):
        from avengine.construction.joint.geometry import MeshRaycaster
        from avengine.construction.joint.mesh_prediction import MeshInstance, predict_mesh_visibility
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        instances = {}
        for aid, actor in plan.actors.items():
            # A whole-body outside envelope cannot occlude an image ray; its
            # human yaw remains the original natural-orientation policy.
            if self.roles.get(aid) == 'outside':continue
            asset = self.assets[aid]
            if aid not in self._mesh_raycasters:
                self._mesh_raycasters[aid] = MeshRaycaster(indexed_static_mesh(asset['habitat_glb']),
                    evidence={'asset_id': aid, 'source': 'original admitted static mesh'})
            root = np.asarray(actor.positions_m, float)[0]
            delta = np.asarray(plan.camera.position_m)-root
            yaw = math.atan2(-delta[0], -delta[2]) if asset.get('kind') in {'human', 'performer', 'animal'} else math.atan2(-delta[2], delta[0])
            c, s = math.cos(yaw), math.sin(yaw)
            transform = np.eye(4);transform[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
            transform[:3, 3] = root
            instances[aid] = MeshInstance(aid, self._mesh_raycasters[aid], transform)
        return predict_mesh_visibility(plan.camera, instances[target_id], scene_depth=context.depth(plan.camera),
            occluders=tuple(instance for aid, instance in instances.items() if aid != target_id))

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        expected_branch = '|'.join(self.targets.values())
        if request.branch != expected_branch or tuple(request.target_ids) != tuple(self.targets):
            yield CheckResult('qa21_requested_classes', 'fail', 'plan changed requested classes or bodies')
            return
        from itertools import combinations
        for (aid, actor), (bid, other) in combinations(plan.actors.items(), 2):
            distance = float(np.linalg.norm((np.asarray(actor.positions_m)[0]-np.asarray(other.positions_m)[0])[[0, 2]]))
            required = max(1.1, actor.body.radius_m+other.body.radius_m+.1)
            yield CheckResult('qa21_spacing_'+aid+'_'+bid, 'pass' if distance >= required else 'fail',
                margin=distance-required, evidence={'actual_planned_m': distance, 'required_m': required})
        for target_id, cls in self.targets.items():
            actor = plan.actors.get(target_id)
            if actor is None:
                yield CheckResult('qa21_target_'+target_id, 'fail', 'classified real body is absent')
                continue
            positions = np.asarray(actor.positions_m)
            if np.max(np.linalg.norm(positions-positions[0], axis=1)) > 1.e-8:
                yield CheckResult('qa21_static_body_'+target_id, 'fail', 'static target moved')
                continue
            occluders = tuple((other_id, np.asarray(other.positions_m)[0], other.body)
                for other_id, other in plan.actors.items() if other_id != target_id)
            prediction = (self.exact_prediction(context, plan, target_id) if self.assets
                else context.predict(plan.camera, positions[0], actor.body, occluders=occluders))
            lower = float(prediction.visible_pixels_lower); edge = float(prediction.edge_margin_px)
            status = 'pass' if self.assets and prediction.state in {'visible_clear', 'visible_occluded'} and lower >= MINIMUM_TARGET_PIXELS and edge >= FRAME_MARGIN_PX else 'fail'
            yield CheckResult('qa21_visible_'+target_id, status,
                margin=min(lower-MINIMUM_TARGET_PIXELS, edge-FRAME_MARGIN_PX),
                evidence={'state': prediction.state, 'visible_pixels_lower': lower,
                    'edge_margin_px': edge, 'authority': 'planning_geometry_only',
                    'prediction_confidence': getattr(prediction, 'confidence', 'unknown'),
                    'mesh_prepared': bool(self.assets),
                    'original_native_pixels_still_required': True})
            classes = {event.sound_class for event in plan.sounds if event.entity_id == target_id}
            yield CheckResult('qa21_one_class_'+target_id, 'pass' if classes == {cls} else 'fail',
                evidence={'target_classes': sorted(classes)})


def certify_qa21_observation(facts, target_actor_id, sound_class):
    """Diagnostic only: call the unchanged authority on actual normalized facts."""
    from avengine.qa import unified_catalog as authority
    result = authority.generate_unified_questions(facts, qa_ids=['QA-21'], seed='qa21-preflight',
        items_per_type=max(5, len(facts.get('actors', {}))), include_angle_followups=False)
    for item in result['items']:
        truth = item.get('truth', {})
        evidence = item.get('evidence') or truth.get('evidence', {})
        if str(evidence.get('target_actor_id')) != str(target_actor_id):
            continue
        value = truth.get('value')
        return {'status': 'pass' if value == sound_class else 'fail',
            'target_actor_id': str(target_actor_id), 'canonical_class': value,
            'expected_class': sound_class, 'authority': 'unchanged_qa21'}
    return {'status': 'fail', 'reason': 'unchanged authority has no candidate for the planned target',
        'expected_target_actor_id': str(target_actor_id), 'expected_class': sound_class,
        'deferred': result.get('deferred', [])}


def native_geometry_candidate_check(video, scene, entries, pixel_counts):
    """Optional native placement hook, before video/RIR cost; rejects edge rests.

    Integrator calls this after actual drawables have been applied and native
    semantic pixels measured. It checks all already placed bodies again, so a
    later companion cannot hide the target. A rejected site is reconstructed.
    """
    construction = scene.get('automatic_construction', {})
    if construction.get('family') != 'QA-21' or construction.get('revision') != REVISION:
        return {'status': 'not_applicable'}
    from avengine.capture.v7_offscreen_layout import vertices_outside_camera
    target_ids = set(construction.get('qa21_target_asset_ids') or [construction['qa21_target_asset_id']])
    sensor = video.sim._sensors['rgb']._sensor_object
    projection = np.asarray(sensor.render_camera.projection_matrix, float)
    world_to_camera = np.linalg.inv(np.asarray(sensor.node.absolute_transformation(), float))
    width = int(construction.get('image_width', 1280)); height = int(construction.get('image_height', 720))
    evidence = []
    for entry in entries:
        asset = entry['declaration']; aid = asset['asset_id']
        transform = np.asarray(entry['applied_root_input'], float)
        local = np.asarray(entry['original_mesh'].vertices, float)
        vertices = local@transform[:3, :3].T+transform[:3, 3]
        clip = np.column_stack([vertices, np.ones(len(vertices))])@world_to_camera.T@projection.T
        counts = int(pixel_counts.get(aid, 0))
        wholly_outside = vertices_outside_camera(vertices, world_to_camera, projection)
        if wholly_outside and counts == 0 and aid not in target_ids:
            evidence.append({'asset_id': aid, 'full_body': 'outside', 'native_pixels': counts})
            continue
        if np.any(clip[:, 3] <= 0):
            return {'status': 'fail', 'reason': 'qa21_rest_body_crosses_camera_plane', 'asset_id': aid}
        ndc = clip[:, :2]/clip[:, 3, None]
        margin = float(np.min((1.-np.abs(ndc))*np.array([width/2., height/2.])))
        needed = MINIMUM_TARGET_PIXELS if aid in target_ids else 100
        if margin < FRAME_MARGIN_PX or counts < needed:
            return {'status': 'fail', 'reason': 'qa21_rest_body_is_clipped_or_insufficient_actual_pixels',
                'asset_id': aid, 'native_pixels': counts, 'required_pixels': needed, 'edge_margin_px': margin}
        if asset.get('kind') in {'human', 'performer'}:
            distance = float(np.linalg.norm((transform[:3, 3]-np.asarray(scene['listener_m']))[[0, 2]]))
            if distance < 1.5:
                return {'status': 'fail', 'reason': 'qa21_person_camera_construction_clearance', 'asset_id': aid, 'distance_m': distance}
        evidence.append({'asset_id': aid, 'full_body': 'inside', 'native_pixels': counts, 'edge_margin_px': margin})
    from itertools import combinations
    for first, second in combinations(entries, 2):
        distance = float(np.linalg.norm((np.asarray(first['applied_root_input'])[:3, 3]-np.asarray(second['applied_root_input'])[:3, 3])[[0, 2]]))
        if distance < 1.:
            return {'status': 'fail', 'reason': 'qa21_actual_static_body_spacing',
                'asset_ids': [first['declaration']['asset_id'], second['declaration']['asset_id']], 'distance_m': distance}
    return {'status': 'pass', 'authority': 'actual_native_mesh_projection_and_pixels', 'bodies': evidence,
        'target_placed': sorted({e['asset_id'] for e in evidence} & target_ids), 'qa_truth_changed': False}


def select_target_set(options, count=3):
    """Different species/devices; no repeated class or same-looking emitter kind."""
    def identities(option):
        asset, cls, _ = option
        kind = asset.get('species_id') if asset.get('kind') == 'animal' else asset.get('entity_class')
        tokens = {c for c in ('black', 'white', 'brown', 'blue', 'green', 'yellow', 'gray', 'silver')
                  if c in str(asset.get('label_color') or asset.get('registered_label_color') or '').lower()}
        return kind, cls, tokens
    def search(start, selected, kinds, classes, colours, needed):
        if len(selected) == needed:return selected
        for i in range(start, len(options)):
            kind, cls, tokens = identities(options[i])
            if kind in kinds or cls in classes or tokens & colours:continue
            result = search(i+1, [*selected, options[i]], kinds | {kind}, classes | {cls}, colours | tokens, needed)
            if result:return result
        return None
    for needed in range(count, 0, -1):
        result = search(0, [], set(), set(), set(), needed)
        if result:return result
    return []


def apply_emission_schedule(row, scene, program, assets):
    """Compile disjoint real bindings before rendering; never modify dry records.

    The integrator calls this from build_constructed_scene when the row carries
    qa21_taxonomy_schedule. The input scene/program may be reused by its caller.
    """
    requested = row.get('qa21_taxonomy_schedule')
    if not requested or scene.get('qa21_schedule_revision') == REVISION:
        return scene, program
    scene = deepcopy(scene);program = deepcopy(program)
    by_asset = {a['asset_id']: a for a in assets}
    records = {r['sound_asset_id']: r for r in program['records']}
    bindings = {b['event_id']: b for b in program['events']}
    authored = [r for r in program['records'] if r.get('source_type') == 'author_tts']
    if not authored:raise ValueError('QA21 schedule requires its complete actual authored companion')
    offset = max(math.ceil(r['header']['sample_count'] / r['header']['sample_rate_hz']*10) for r in authored)+5
    templates = []
    for aid in requested['target_asset_ids']:
        if aid not in by_asset:raise ValueError('QA21 planned emitter has no actual source body')
        ep = by_asset[aid]['endpoint_id'];events = [e for e in scene['events'] if e['endpoint_id'] == ep]
        expected = row['source_classes'][aid]
        if len(expected) != 1 or not events or {e.get('sound_class') for e in events} != set(expected):
            raise ValueError('QA21 target must retain exactly one observed registered class')
        event = events[0];binding = bindings[event['event_id']];record = records[binding['sound_asset_id']]
        if record.get('source_type') == 'author_tts' or record['sound_class'] != expected[0]:
            raise ValueError('QA21 target cannot relabel an authored or different sound class')
        duration = event['end_tick']-event['start_tick']
        if offset+duration > 95:
            raise ValueError('complete QA21 source bindings and companion do not fit the requested window')
        templates.append((aid, ep, event, binding, offset, duration))
        offset += duration+2
    endpoints = {ep for _, ep, *_ in templates}
    removed = {e['event_id'] for e in scene['events'] if e['endpoint_id'] in endpoints}
    scene['events'] = [e for e in scene['events'] if e['event_id'] not in removed]
    program['events'] = [b for b in program['events'] if b['event_id'] not in removed]
    evidence = []
    for aid, ep, template, binding, offset, duration in templates:
        for rank, epoch in enumerate((0, 100, 200)):
            eid = 'qa21_taxonomy_'+ep+'_'+str(rank)
            scene['events'].append({**template, 'event_id': eid, 'start_tick': epoch+offset, 'end_tick': epoch+offset+duration})
            program['events'].append({**binding, 'event_id': eid})
        evidence.append({'asset_id': aid, 'sound_class': template['sound_class'],
            'sound_asset_id': binding['sound_asset_id'], 'crop_start_sample': binding.get('crop_start_sample', 0),
            'offset_tick': offset, 'duration_ticks': duration})
    scene['events'].sort(key=lambda e: (e['start_tick'], e['end_tick'], e['event_id']))
    order = {e['event_id']: i for i, e in enumerate(scene['events'])}
    program['events'].sort(key=lambda e: order[e['event_id']])
    scene['qa21_schedule_revision'] = REVISION
    scene['qa21_compiled_emissions'] = evidence
    return scene, program


def compiled_sound_events(row, house, records, catalog, semantic):
    """Use the real compiler's actual record IDs, crops and 100ms clock."""
    from avengine.construction.joint import SoundEvent
    from avengine.timeline.v7_family_clock import build_constructed_scene
    _, scene, program, assets = build_constructed_scene(row, house, records, catalog, semantic)
    scene, program = apply_emission_schedule(row, scene, program, assets)
    bindings = {b['event_id']: b for b in program['events']}
    sounds = []
    pinned = {}
    for event in scene['events']:
        source = scene['sources'][event['endpoint_id']]
        binding = bindings[event['event_id']]
        aid = source['asset_id']; pinned[aid] = binding['sound_asset_id']
        sounds.append(SoundEvent(aid, binding['sound_asset_id'], event['sound_class'],
            event['start_tick']/10., event['end_tick']/10., wet_end_s=min(30., event['end_tick']/10.+.5)))
    return tuple(sounds), pinned, scene.get('qa21_compiled_emissions', [])


def prepare_qa21_sources(room, house, state, catalog, records, semantic, contract, runtime,
                         *, scene_id, seed, family='QA-21', attempt=0):
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable, eligible_humans, choose_authored, _request, room_context, stable_seed
    from avengine.timeline.v7_production_inputs import validate_recipe
    if family != 'QA-21':raise ValueError('QA21 plugin cannot construct another family')
    requested = runtime.get('qa21_requested_class') or state.get('qa21_requested_class')
    maximum = 1 if requested else int(runtime.get('qa21_targets_per_parent', 3))
    if not 1 <= maximum <= 3:raise ValueError('QA21 supports one to three distinct planned targets')
    options = target_options(room, catalog, records, contract, state, requested_class=requested)
    selected = select_target_set(options, maximum)
    if not selected:raise ConstructionUnavailable('no admitted floor emitter can make the requested QA21 class in this legal room')
    # Only the human required by this question plan is introduced. Its whole
    # authored speech is retained and its entire body stays genuinely outside.
    humans = eligible_humans(catalog, room, contract)
    if not humans:raise ConstructionUnavailable('QA21 room needs one real admitted companion')
    person = min(humans, key=lambda a: (state.get('source_entity_usage_counts', {}).get(a['asset_id'], 0), stable_seed(seed, a['asset_id'])))
    short_semantic = {**semantic, 'sounds': [r for r in semantic['sounds'] if r['synthesis_readback']['duration_s'] <= 2.]}
    authored = choose_authored(short_semantic, [person], state, seed, attempt=attempt)[0][0]
    companion = person['asset_id'];target_ids = [a['asset_id'] for a, _, _ in selected]
    targets = {a['asset_id']: cls for a, cls, _ in selected}
    ctx = room_context(room, house)
    row = {'schema': 'v7-visible-registered-pilot-phase2-v1', 'scene_id': scene_id,
        'independent_parent_scene_id': scene_id, 'scene_seed': seed, 'house': room['house'],
        'room_id': room['room_id'], 'split': 'train', 'family': room['family'],
        'listener_m': ctx['listener_m'], 'camera_yaw_deg': ctx['camera_yaw_deg'],
        'camera_yaw_knots': [{'time_s': 0., 'yaw_deg': ctx['camera_yaw_deg']}, {'time_s': 30., 'yaw_deg': ctx['camera_yaw_deg']}],
        'asset_ids': [companion, *target_ids], 'silent_asset_ids': [],
        'source_classes': {companion: ['speech_playback'], **{aid: [cls] for aid, cls in targets.items()}},
        'motion_requests': {}, 'members': ['v0_a0'], 'trial_only': True,
        'acceptance_allow_partial_families': True, 'window_count': 5, 'questions_per_window': 14,
        'question_quota_base': 0, 'acceptance_window_starts_s': [0., 5., 10., 15., 20.],
        'automatic_construction': {'revision': REVISION, 'family': family, 'candidate_index': attempt,
            'room_key': room['key'], 'required_role_count': 1+len(selected), 'qa21_target_asset_ids': target_ids,
            'qa21_target_classes': targets, 'image_width': 1280, 'image_height': 720, 'scene_template_files_read': []},
        'owner_social_layout': {'minimum_human_center_distance_m': 1., 'minimum_human_animal_distance_m': 1.,
            'minimum_human_camera_distance_m': 1.5, 'human_orientation': 'varied natural standing'},
        'semantic_scenario_id': authored['scenario_id'], 'semantic_sound_ids': [authored['sound_asset_id']],
        'semantic_human_asset_ids': [companion], 'semantic_start_offsets_s': [0.],
        'phase2_authored_speech_schedule_s': {companion: [0., 10., 20.]},
        'category_focus_class': next(iter(targets.values())),
        'qa21_taxonomy_schedule': {'target_asset_ids': target_ids, 'real_offscreen_companion_asset_id': companion},
        'native_view_roles': {**{aid: {'view': 'inside', 'static': True, 'required_visible_sometime': True} for aid in target_ids},
            companion: {'view': 'outside', 'static': True, 'minimum_abs_azimuth_deg': 42.5, 'real_geometry_and_native_projection_required': True}}}
    _request(row, state, family)
    row['scene_demand']['class34_construction'] = [proof for _, _, proof in selected]
    row['scene_demand']['required_independent_family_counts'] = {family: len(selected)}
    row['scene_demand']['required_asset_ids'] = list(row['asset_ids'])
    sounds, pinned, emissions = compiled_sound_events(row, house, records, catalog, semantic)
    row['proposed_sound_asset_ids'] = pinned
    row['scene_demand']['qa21_real_compiled_emissions'] = emissions
    validate_recipe(row, catalog, contract)
    return row, selected, person, sounds, ctx


def construct_qa21_scene(room, house, state, catalog, records, semantic, contract, runtime,
                         *, scene_id, seed, family='QA-21', attempt=0):
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable, _request
    from avengine.timeline.v7_production_inputs import validate_recipe
    from avengine.construction.joint import build_room_context, FamilyRequest, solve
    row, selected, person, sounds, ctx = prepare_qa21_sources(room, house, state, catalog,
        records, semantic, contract, runtime, scene_id=scene_id, seed=seed, family=family, attempt=attempt)
    target_ids = row['automatic_construction']['qa21_target_asset_ids']
    targets = row['automatic_construction']['qa21_target_classes']
    companion = person['asset_id']
    bodies = {a['asset_id']: body_spec(a) for a in [*(a for a, _, _ in selected), person]}
    roles = {**{aid: 'inside' for aid in target_ids}, companion: 'outside'}
    context = build_room_context(ctx['original_room'], house=house, runtime=runtime,
        cache_dir=runtime.get('qa21_joint_cache_dir'), camera_limit=24, width=1280, height=720)
    try:
        request = FamilyRequest('QA-21', '|'.join(targets.values()), tuple(target_ids))
        plugin = Qa21Plugin(bodies, roles, target_ids[0], targets[target_ids[0]], sounds=sounds, targets=targets, assets={a['asset_id']: a for a in [*(a for a, _, _ in selected), person]})
        solved = solve(context, request, plugin, budget_s=120., max_candidates=256)
    finally:
        close = getattr(context, 'close', None)
        if close:close()
    status = solved.status if hasattr(solved, 'status') else solved.get('status')
    if status != 'certified':raise ConstructionUnavailable('QA21 core did not certify the chosen classes/bodies: '+str(solved))
    plan = solved.plan if hasattr(solved, 'plan') else solved['plan'];camera = plan.camera
    row.update(listener_m=list(camera.position_m), camera_yaw_deg=camera.yaw_deg,
        camera_yaw_knots=[{'time_s': 0., 'yaw_deg': camera.yaw_deg}, {'time_s': 30., 'yaw_deg': camera.yaw_deg}],
        native_placement_preferences_m={aid: np.asarray(actor.positions_m)[0].tolist() for aid, actor in plan.actors.items()})
    from dataclasses import asdict
    import json
    certificate = solved.certificate if hasattr(solved, 'certificate') else solved['certificate']
    row['scene_demand']['qa21_planning_certificate'] = json.loads(json.dumps(asdict(certificate),
        default=lambda value: value.tolist() if hasattr(value, 'tolist') else value.item()))
    row['scene_demand']['qa21_certification_authority'] = 'planning geometry and actual compiled source clock; original native masks and QA unchanged'
    validate_recipe(row, catalog, contract)
    return {'profile': 'phase2', 'row': row}


def native_appearance_check(video, scene, entries):
    """Run the existing native ROI reviewer on this real placement, without
    retaining duplicate frame-zero measurements in the later video evidence.
    """
    from avengine.assets.v7_visible_color import rebuild_appearance_review
    from avengine.qa.unified_catalog import _appearance_candidates, _Deferred
    construction = scene['automatic_construction']
    target_ids = set(construction.get('qa21_target_asset_ids') or [construction['qa21_target_asset_id']])
    target_entries = [e for e in entries if e['declaration']['asset_id'] in target_ids]
    if not target_entries:
        return {'status': 'pass', 'reviewed_target_ids': [], 'reason': 'targets_not_placed_yet'}
    # The original implementation owns ROI selection, thresholds and labels.
    # Its two append-only measurement lists are temporary here; full rendering
    # independently collects its normal frame stream afterwards.
    original_actors = video.actors
    original_colors, original_materials = video.color_checks, video.material_checks
    video.actors = list(entries);video.color_checks = [];video.material_checks = []
    try:
        observed = video.sim.get_sensor_observations()
        video._check_source_frame(np.asarray(observed['rgb'])[..., :3], np.asarray(observed['semantic']), 0)
        checks = deepcopy(video.color_checks)
    finally:
        video.actors = original_actors
        video.color_checks = original_colors;video.material_checks = original_materials
    assets = [e['declaration'] for e in entries]
    minimum = (getattr(video, 'animal_color_thresholds', None) or {}).get('min_pixels', 200)
    document, eligibility = rebuild_appearance_review(checks, assets, minimum)
    reviews = document['actors'];actors = {};reviewed_ids = []
    for entry in target_entries:
        asset = entry['declaration'];actor_id = str(asset['native_instance_id'])
        review = reviews.get(actor_id)
        if review is None:
            return {'status': 'fail', 'reason': 'qa21_native_appearance_not_reviewable',
                'asset_id': asset['asset_id'], 'eligibility': eligibility,
                'actual_native_color_checks': [r for r in checks if r['asset_id'] == asset['asset_id']]}
        actors[actor_id] = {'appearance': {'field': review['field'], 'value': review['value'],
            'label': review.get('label_en', review.get('label_zh'))}}
        reviewed_ids.append(asset['asset_id'])
    facts = {'actors': actors, 'appearance_review': reviews, 'time': {'frame_count': 1}}
    try:
        _appearance_candidates(facts)
    except _Deferred as exc:
        return {'status': 'fail', 'reason': 'qa21_native_appearances_not_unique', 'detail': str(exc),
            'native_appearance_review': document}
    return {'status': 'pass', 'authority': 'unchanged native ROI reviewer and appearance_candidates',
        'reviewed_target_ids': reviewed_ids, 'native_appearance_review': document,
        'native_frame_index': 0, 'final_native_stream_independently_required': True}


def native_candidate_check(video, scene, entries, pixel_counts):
    geometry = native_geometry_candidate_check(video, scene, entries, pixel_counts)
    if geometry['status'] != 'pass':return geometry
    appearance = native_appearance_check(video, scene, entries)
    if appearance['status'] != 'pass':return {**appearance, 'geometry_preflight': geometry}
    return {**geometry, 'appearance_preflight': appearance}
