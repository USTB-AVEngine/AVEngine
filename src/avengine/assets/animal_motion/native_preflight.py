"""Actual animal mask witnesses before complete video and fresh native RIRs.

This consumes the final native actors and admitted gait. Predictions never
replace masks or write QA truth. The full independent stream still runs later.
"""
import numpy as np


def _native_body_framing(video, entry):
    from avengine.construction.joint.native import native_rest_framing
    return native_rest_framing(video, entry)


def native_animal_preflight(video, scene, entries, *, rest_framing_fn=None):
    from avengine.capture.continuous_native_visibility import NativeVisibilityStream
    requests = scene.get('production_scenario', {}).get('motion_requests', {})
    targets = [(entry, requests[entry['endpoint_id']]['motion_quartet_route']) for entry in entries
        if requests.get(entry['endpoint_id'], {}).get('motion_quartet_route', {}).get('animal_native_preflight_required')]
    if not targets:
        return {'status': 'not_applicable', 'source_family': 'motion_animal'}
    if {id(entry) for entry in entries} != {id(entry) for entry in video.actors}:
        raise ValueError('animal native preflight requires every final scene actor')
    if any(not entry.get('motion_episode') for entry, route in targets):
        raise ValueError('animal native preflight needs the completed actual native gait')
    framing_fn = rest_framing_fn or _native_body_framing
    indices = {0, 449}
    for entry, route in targets:
        for slot in route.get('emission_slots', [route['emission_slot']]):
            for value in ((slot['start_s'] + slot['end_s']) / 2, slot['end_s'] + .55, slot.get('post_until_s')):
                if value is not None:
                    indices.add(max(0, min(449, round(float(value) * 15))))
        for turn in route.get('human_rest_turns', []):
            indices.add(max(0, min(449, round(sum(turn['interval_s']) / 2 * 15))))
    identities = [{'endpoint_id': entry['endpoint_id'], 'actor_id': entry['declaration']['native_instance_id']}
                  for entry in entries]
    stream = NativeVisibilityStream(video)
    measured = []
    rest_checks = []
    try:
        for index in sorted(indices):
            for actor in entries:
                if actor.get('motion_episode'):
                    video._apply_motion_entry(actor, index * 2)
            video._native_pair_pixels(scene, entries)
            modal = np.asarray(video.sim.get_sensor_observations()['semantic']).copy()
            truth = stream.capture(index, modal, identities)
            targets_at_frame = {}
            for entry, route in targets:
                instance = str(entry['declaration']['native_instance_id'])
                targets_at_frame[entry['declaration']['asset_id']] = truth['per_instance'][instance]['frames'][0]
                roots = np.asarray(entry['motion_episode']['world_roots_m'], float)
                k = index * 2
                still = np.linalg.norm((roots[min(900, k + 1)] - roots[max(0, k - 1)])[[0, 2]]) < 1e-8
                if still:
                    rest_checks.append({'frame_index': index, 'asset_id': entry['declaration']['asset_id'],
                                        **framing_fn(video, entry)})
            measured.append({'frame_index': index, 'time_s': index / 15., 'targets': targets_at_frame})
    finally:
        for actor in entries:
            if actor.get('motion_episode'):
                video._apply_motion_entry(actor, 0)
        video._native_pair_pixels(scene, entries)
        stream.close()
    reasons = []
    summaries = {}
    for entry, route in targets:
        asset = entry['declaration']['asset_id']
        peak = max((int(row['targets'][asset]['visible_pixels']) for row in measured), default=0)
        summaries[asset] = {'peak_actual_visible_pixels': peak, 'actual_probe_frames': len(measured)}
        if peak < 100:
            reasons.append(asset + ':actual_native_body_never_reaches_100_pixels_in_requested_probes')
        target_rests = [record for record in rest_checks if record['asset_id'] == asset]
        if any(record['state'] == 'edge' for record in target_rests):
            reasons.append(asset + ':actual_stationary_body_crosses_image_edge')
        if route.get('require_rest_poses_in_frame') and any(record['state'] != 'inside' for record in target_rests):
            reasons.append(asset + ':constructed_inside_rest_is_not_wholly_inside_native_frame')
    return {'status': 'fail' if reasons else 'pass', 'source_family': 'motion_animal', 'reasons': reasons,
        'evidence_kind': 'same_native_final_scene_modal_and_target_only_witnesses',
        'native_pixel_truth_verified': True, 'targets': summaries, 'measured_frames': measured,
        'actual_native_rest_framing': rest_checks, 'native_control_restoration': stream.receipt(),
        'final_complete_stream_still_required': True, 'QA_truth_or_scorer_modified': False}


def animal_native_checks(video, scene, entries, *, plan, row):
    """Family checks for the core's prepared payload, without moving a pin."""
    from avengine.construction.joint import CheckResult
    from avengine.timeline.v7_motion_during_conditions import verify_motion_during_native
    by_asset={entry['declaration']['asset_id']:entry for entry in entries}
    if set(by_asset)!=set(plan.actors):
        return (CheckResult('animal:native_entities','fail','native_registered_entities_differ_from_plan'),)
    checks=[];admissions={}
    for asset,actor in plan.actors.items():
        entry=by_asset[asset]
        if entry.get('motion_episode'):
            record=entry['motion_episode'];reads=record.get('native_30Hz_readbacks',[])
            if len(reads)!=901:
                checks.append(CheckResult(asset+':native_clock','fail','missing_complete_901_native_body_readbacks'))
                continue
            actual=np.asarray([np.asarray(sample['body_to_world_m'],float)[:3,3]for sample in reads])
            error=float(np.linalg.norm(actual-np.asarray(actor.positions_m),axis=1).max())
            checks.append(CheckResult(asset+':exact_solved_body_clock','pass'if error<=.002 else'fail',
                'actual_native_roots_against_solved_roots',evidence={'maximum_error_m':error,'frames':len(reads)}))
            if error<=.002:record['native_pose_readback_status']='pass_actual_901_native_roots'
            admissions[entry['endpoint_id']]=record
        else:
            root=np.asarray(entry['object'].root_scene_node.absolute_transformation(),float)[:3,3]
            error=float(np.linalg.norm(root-np.asarray(actor.positions_m)[0]))
            checks.append(CheckResult(asset+':exact_static_root','pass'if error<=.002 else'fail',
                'actual_native_root_against_solved_root',evidence={'error_m':error}))
    if any(check.status!='pass'for check in checks):return tuple(checks)
    verdict=verify_motion_during_native(row,scene,{'native_motion_admissions':admissions})
    checks.append(CheckResult('animal:original_native_motion_verifier',
        'pass'if verdict and verdict['status']=='pass'else'fail','unchanged_original_motion_verifier',
        evidence={'report':verdict}))
    if plan.metadata.get('negative_companion_motion_request'):
        from avengine.assets.animal_motion.negative_witness import native_negative_companion_checks
        mover=next(aid for aid,actor in plan.actors.items()if actor.body.species!='human')
        companion=next(aid for aid,actor in plan.actors.items()if actor.body.species=='human')
        start=plan.metadata['third_static_call_s']
        end=plan.metadata['negative_companion_motion_request']['intervals_s'][-1][1]-.8
        checks.extend(native_negative_companion_checks(video,entries,plan,mover,companion,start,end-start))
        if any(check.status!='pass'for check in checks):return tuple(checks)
    pixel=native_animal_preflight(video,scene,entries)
    checks.append(CheckResult('animal:actual_native_pixels_and_rests',
        'pass'if pixel['status']=='pass'else'fail','same_final_native_scene_preflight',evidence=pixel))
    return tuple(checks)

