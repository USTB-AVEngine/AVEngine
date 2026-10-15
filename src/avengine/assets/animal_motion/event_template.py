"""Bind real animal recordings to separately certified repeated motion bouts."""
from copy import deepcopy
import numpy as np

FAMILIES = ('QA-06', 'QA-15', 'QA-16', 'QA-17')
REVISION = 'animal-three-certified-source-bouts-v1'


def animal_motion_events_template(base_row, room, catalog, semantic_manifest, source_records,
                                 pathfinder, *, attempt=0, contract=None, source_type=None):
    from avengine.assets.roster_motion.human_event_route import repeat_human_bouts
    from avengine.timeline.v7_motion_quartet_template import motion_quartet_template, MotionTemplateUnavailable
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row, verify_planned_trajectories
    from avengine.timeline.v7_registered_sampling import semantic_record
    from .route_solver import delivery_speed_reports
    import soundfile as sf

    kind = source_type or base_row.get('motion_source_type')
    if kind not in {'dog_walk', 'cat_walk', 'animal_slide'}:
        raise ValueError('repeated animal template needs an admitted animal source')
    seed_row = deepcopy(base_row)
    seed_row.update(animal_motion_event_count=1, human_motion_event_count=1)
    row = motion_quartet_template(seed_row, room, catalog, semantic_manifest, source_records,
        pathfinder, attempt=attempt, contract=contract, source_type=kind)
    mover = next(iter(row['motion_requests']))
    seed_route = row['motion_requests'][mover]['motion_quartet_route']
    if not all(seed_route['emission_slot'].get(key) for key in ('qa06', 'qa15', 'qa16', 'qa17')):
        row['animal_motion_event_count'] = 1
        row['animal_multi_event_gap'] = 'seed corridor supports only the recorded partial-family subset'
        return row
    route = repeat_human_bouts(seed_route)
    route.update(animal_motion_event_count=3,
        animal_stationary_turn_intervals_s=[turn['interval_s'] for turn in route['human_rest_turns']],
        construction='three_independent_animal_corridor_bouts')
    species = 'slide' if kind == 'animal_slide' else ('dog' if kind == 'dog_walk' else 'cat')
    speed_reports = delivery_speed_reports(route, species)
    if not all(item['status'] == 'pass' for item in speed_reports):
        raise MotionTemplateUnavailable('repeated animal crop speed check failed: ' + str(speed_reports))
    companion = next(aid for aid in row['source_classes'] if aid != mover)
    explicitly_required = set(base_row.get('scene_demand', {}).get('required_asset_ids', []))
    retained = {mover, companion} | explicitly_required
    removed = [aid for aid in row['asset_ids'] if aid not in retained]
    row['asset_ids'] = [aid for aid in row['asset_ids'] if aid in retained]
    row['silent_asset_ids'] = [aid for aid in row.get('silent_asset_ids', []) if aid in retained]
    row['animal_unneeded_silent_bystanders_omitted'] = removed
    row.setdefault('native_view_roles', {})[companion] = {
        'view': 'outside', 'minimum_abs_azimuth_deg': 65.,
        'reason': 'the animal motion question needs the animal; companion remains a real off-camera speaker'}
    sound_by_id = {s['sound_asset_id']: s for s in semantic_manifest['sounds']}
    companion_sound = sound_by_id[row['proposed_sound_asset_ids'][companion]]
    companion_duration = float(companion_sound['synthesis_readback']['duration_s'])
    response_start = min(7.5, 9.8 - companion_duration)
    if response_start <= route['human_event_windows'][0]['silent_query_until_s'] + .2:
        raise MotionTemplateUnavailable('whole companion speech would overlap the animal silent query')
    slots = route['emission_slots']
    starts = [slot['start_s'] for slot in slots]
    schedule_keys = [key for key in ('phase2_sound_program_schedule_s', 'phase2_whole_source_recordings')
                     if mover in row.get(key, {})]
    if len(schedule_keys) != 1:
        raise MotionTemplateUnavailable('animal bout must retain exactly one real registered sound program')
    row[schedule_keys[0]][mover] = starts
    row['phase2_source_event_schedule_s'] = {companion: [s['offset_s'] + response_start for s in slots]}
    row['motion_source_start_s'] = {mover: starts[0], companion: response_start}
    row['semantic_start_offsets_s'] = [response_start for _ in row['semantic_human_asset_ids']]
    row.update(motion_quartet_template_revision=REVISION, animal_motion_event_count=3)
    requests = []
    for slot, window in zip(slots, route['human_event_windows']):
        end_local = window['walk_end_s'] - slot['offset_s'] - .1
        for qa, branch in [('QA-06', 'moving'), ('QA-15', slot['branch']), ('QA-16', None), ('QA-17', 'yes')]:
            requests.append({'qa_id': qa, 'branch': branch, 'target_asset_id': mover,
                'speed_m_s': route['speed_m_s'], 'speed_range_m_s': route['speed_range_m_s'],
                'minimum_motion_s': 1.5, 'end_hold_s': 10. - end_local, 'offsets_s': [slot['offset_s']],
                'reserve_tail_s': .5, 'wet_tail_planning_bound_s': None, 'minimum_distance_change_m': .5})
    row['motion_during_condition_requests'] = requests
    row.pop('motion_during_condition_plan', None)
    request = row['motion_requests'][mover]
    request['motion_quartet_route'] = route
    for key in ('motion_during_condition_plan_for_fit', 'straight_motion_windows_s',
                'sounding_straight_windows_s', 'distance_trend', 'distance_trend_intervals_s'):
        request.pop(key, None)
    by_asset = {a['asset_id']: a for a in catalog}
    voice = 'F' if by_asset[companion]['visual_gender'] == 'female' else 'M'
    companion_record = semantic_record(companion_sound, voice)
    samples, rate = sf.read(companion_record['normalized_path'], dtype='float64')
    nonzero = np.flatnonzero(samples != 0.)
    if rate != 16000 or samples.ndim != 1 or not len(nonzero):
        raise MotionTemplateUnavailable('whole companion speech must be non-silent 16kHz mono')
    companion_record.setdefault('screens', {})['automatic_activity'] = {
        'activity_intervals_samples': [[int(nonzero[0]), int(nonzero[-1] + 1)]],
        'method': 'exact nonzero envelope; whole original speech retained'}
    row = prepare_motion_during_row(row, catalog, list(source_records) + [companion_record])
    plan = row['motion_during_condition_plan']
    listener = np.asarray(route['listener_m'], float)
    points = np.asarray(route['positions_xz_m'], float)
    tracks = {aid: np.tile(listener + [25., 0., 25. + index * 5.], (2, 1))
              for index, aid in enumerate(row['asset_ids'])}
    tracks[mover] = np.column_stack((points[:, 0], np.full(len(points), listener[1] + .3), points[:, 1]))
    radial = np.linalg.norm(tracks[mover] - listener, axis=1)
    times = np.arange(len(points)) / 30.
    verdicts = []
    for index, slot in enumerate(slots):
        event_plan = {**plan, 'plans': [p for p in plan['plans'] if p['offsets_s'] == [slot['offset_s']]]}
        verdict = verify_planned_trajectories(event_plan, tracks, listener.tolist())
        sound = event_plan['plans'][0]['config']['sounds'][mover]
        audible_end = slot['start_s'] + sound['audible_end_sample_exclusive'] / sound['sample_rate_hz']
        anchor = float(np.interp(audible_end, times, radial))
        stop = min(route['human_event_windows'][index]['walk_end_s'] - .3,
                   slot['offset_s'] + response_start - .2)
        integer_queries = []
        for second in range(int(np.ceil(audible_end + .65)), int(np.floor(stop)) + 1):
            delta = float(np.interp(second, times, radial)) - anchor
            if abs(delta) >= .3:
                integer_queries.append({'time_s': second, 'distance_delta_m': delta})
        if not integer_queries:
            raise MotionTemplateUnavailable('animal event has no whole-second QA16 query with 0.3m margin')
        verdicts.append({'event_index': index, 'offset_s': slot['offset_s'], 'families': verdict,
                         'qa16_integer_queries': integer_queries})
        if set(verdict) != set(FAMILIES) or not all(verdict.values()):
            raise MotionTemplateUnavailable('animal event failed original verifier: ' + str(verdicts[-1]))
    request = row['motion_requests'][mover]
    request['motion_during_condition_plan_for_fit'] = deepcopy(plan)
    request['intervals_s'] = [[0., 30.]]
    for key in ('straight_motion_windows_s', 'distance_trend', 'distance_trend_intervals_s'):
        request.pop(key, None)
    row['motion_quartet_planned_verification'] = {qa: True for qa in FAMILIES}
    row['animal_motion_event_planned_verification'] = verdicts
    row['animal_delivery_speed_verification'] = speed_reports
    row['scene_demand'].update(constructor=REVISION, required_asset_ids=list(row['asset_ids']),
        condition_families={'motion_quartet': list(FAMILIES)}, required_actual_candidate_families=list(FAMILIES))
    row['motion_quartet_construction'].update(revision=REVISION, route=route,
        independent_emissions=3, source_events_spread_over_parent=True,
        post_sound_query_clear_of_later_response=True)
    return row
