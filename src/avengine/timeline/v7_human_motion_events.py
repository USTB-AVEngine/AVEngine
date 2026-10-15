"""Schedule and certify every independent human motion event before rendering."""
from copy import deepcopy

import numpy as np

FAMILIES = ('QA-06', 'QA-15', 'QA-16', 'QA-17')
REVISION = 'human-three-certified-speech-bouts-v1'


def human_motion_events_template(base_row, room, catalog, semantic_manifest, source_records,
                                pathfinder, *, attempt=0, contract=None):
    from avengine.assets.roster_motion.human_event_route import repeat_human_bouts
    from avengine.timeline.v7_motion_quartet_template import motion_quartet_template, MotionTemplateUnavailable
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row, verify_planned_trajectories
    from avengine.timeline.v7_registered_sampling import semantic_record
    import soundfile as sf

    seed_row = deepcopy(base_row)
    seed_row.update(human_motion_event_count=1, human_motion_require_all_families=True,
                    motion_template_focus='M1')
    row = motion_quartet_template(seed_row, room, catalog, semantic_manifest, source_records,
                                  pathfinder, attempt=attempt, contract=contract, source_type='human_speech')
    mover = next(iter(row['motion_requests']))
    route = repeat_human_bouts(row['motion_requests'][mover]['motion_quartet_route'])
    ids = row['semantic_human_asset_ids']
    companion = next(aid for aid in ids if aid != mover)
    slots = route['emission_slots']
    local_start = slots[0]['start_s']
    # A short response is outside every anchor's post-sound certification
    # interval.  Both real people retain their complete, single-kind speech.
    sounds = {s['sound_asset_id']: s for s in semantic_manifest['sounds']}
    companion_duration = float(sounds[row['proposed_sound_asset_ids'][companion]]['synthesis_readback']['duration_s'])
    response_start = min(7.5, 9.8 - companion_duration)
    if response_start <= route['human_event_windows'][0]['silent_query_until_s'] + .2:
        raise MotionTemplateUnavailable('complete response would intrude into the certified silent query')
    row['phase2_source_event_schedule_s'] = {
        mover: [slot['start_s'] for slot in slots],
        companion: [slot['offset_s'] + response_start for slot in slots]}
    row['motion_source_start_s'] = {mover: local_start, companion: response_start}
    row['semantic_start_offsets_s'] = [local_start if aid == mover else response_start for aid in ids]
    row['motion_quartet_template_revision'] = REVISION
    row['human_motion_event_count'] = 3
    requests = []
    for slot, window in zip(slots, route['human_event_windows']):
        end_local = window['walk_end_s'] - slot['offset_s'] - .1
        for qa, branch in [('QA-06', 'moving'), ('QA-15', slot['branch']), ('QA-16', None), ('QA-17', 'yes')]:
            requests.append({'qa_id': qa, 'branch': branch, 'target_asset_id': mover,
                'speed_m_s': route['speed_m_s'], 'speed_range_m_s': route['speed_range_m_s'],
                'minimum_motion_s': 1.5, 'end_hold_s': 10. - end_local,
                'offsets_s': [slot['offset_s']], 'reserve_tail_s': .5,
                'wet_tail_planning_bound_s': None, 'minimum_distance_change_m': .5})
    row['motion_during_condition_requests'] = requests
    row.pop('motion_during_condition_plan', None)
    request = row['motion_requests'][mover]
    for key in ('motion_during_condition_plan_for_fit', 'straight_motion_windows_s',
                'sounding_straight_windows_s', 'distance_trend', 'distance_trend_intervals_s'):
        request.pop(key, None)
    request['motion_quartet_route'] = route
    measured = []
    audible_ends = {}
    byasset = {a['asset_id']: a for a in catalog}
    for aid in ids:
        sound = sounds[row['proposed_sound_asset_ids'][aid]]
        record = semantic_record(sound, 'F' if byasset[aid]['visual_gender'] == 'female' else 'M')
        samples, rate = sf.read(record['normalized_path'], dtype='float64')
        nonzero = np.flatnonzero(samples != 0.)
        if rate != 16000 or samples.ndim != 1 or not len(nonzero):
            raise MotionTemplateUnavailable('whole speech must be a non-silent 16kHz mono source')
        record.setdefault('screens', {})['automatic_activity'] = {
            'activity_intervals_samples': [[int(nonzero[0]), int(nonzero[-1] + 1)]],
            'method': 'exact nonzero waveform envelope; complete speech support'}
        audible_ends[aid] = float(nonzero[-1] + 1) / 16000.
        measured.append(record)
    row = prepare_motion_during_row(row, catalog, list(source_records) + measured)
    listener = np.asarray(route['listener_m'], float)
    track = np.asarray(route['positions_xz_m'], float)
    tracks = {aid: np.tile(listener + [25., 0., 25. + i * 5.], (2, 1))
              for i, aid in enumerate(row['asset_ids'])}
    tracks[mover] = np.column_stack((track[:, 0], np.full(len(track), listener[1] + .3), track[:, 1]))
    plan = row['motion_during_condition_plan']
    verdicts = []
    for index, slot in enumerate(slots):
        event_plan = {**plan, 'plans': [p for p in plan['plans'] if p['offsets_s'] == [slot['offset_s']]]}
        verdict = verify_planned_trajectories(event_plan, tracks, listener.tolist())
        # QA16 is published at integer seconds in the current production policy.
        # Leave extra distance and tail clearance rather than relying on an
        # arbitrary frame that cannot be asked in the final question.
        end_s = slot['start_s'] + audible_ends[mover]
        times = np.arange(len(track)) / 30.
        radial = np.linalg.norm(tracks[mover] - listener, axis=1)
        anchor = float(np.interp(end_s, times, radial))
        stop = min(route['human_event_windows'][index]['walk_end_s'] - .3,
                   slot['offset_s'] + response_start - .2)
        query_points = []
        for second in range(int(np.ceil(end_s + .65)), int(np.floor(stop)) + 1):
            delta = float(np.interp(second, times, radial)) - anchor
            if abs(delta) >= .3:
                query_points.append({'time_s': second, 'distance_delta_m': delta})
        if not query_points:
            raise MotionTemplateUnavailable('no whole-second QA16 query with 0.3m construction margin')
        verdicts.append({'event_index': index, 'offset_s': slot['offset_s'], 'families': verdict,
                         'qa16_integer_queries': query_points})
        if set(verdict) != set(FAMILIES) or not all(verdict.values()):
            raise MotionTemplateUnavailable('independent human event failed original verifier: ' + str(verdicts[-1]))
    request = row['motion_requests'][mover]
    request['motion_during_condition_plan_for_fit'] = deepcopy(plan)
    request['intervals_s'] = [[0., 30.]]
    for key in ('straight_motion_windows_s', 'distance_trend', 'distance_trend_intervals_s'):
        request.pop(key, None)
    row['motion_quartet_planned_verification'] = {qa: True for qa in FAMILIES}
    row['human_motion_event_planned_verification'] = verdicts
    row['scene_demand'].update(constructor=REVISION,
        condition_families={'motion_quartet': list(FAMILIES)}, required_actual_candidate_families=list(FAMILIES))
    row['motion_quartet_construction'].update(revision=REVISION, route=route,
        independent_emissions=3, source_events_spread_over_parent=True,
        post_sound_query_clear_of_later_response=True)
    return row
