"""Owner's 2026-10-14 09:3x SGT motion-only policy; legacy checks are unchanged.

Only support paws have a positive 12 mm floor-gap limit. Per-step sliding is
material-point displacement from that step's first support point; its speed is
reported without an extra speed gate. Cross-region added areas are summed in
each frame. Same-region overlap and UV seam diagnostics cannot reject a case.
Audio is deliberately absent from this admission policy.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.checks import FLOOR_PENETRATION_M
from avengine.assets.animal_motion.profiles import PAWS

from avengine.assets.animal_motion.motion_only_settings import (
    POLICY_NAME, VALIDATION_POLICY_NAMES, AUDIO_STATUS, validation_policy,
)


def direction_checks(frames):
    t = np.asarray([f['time_s'] for f in frames], float)
    def coordinates(primary, alternate):
        return np.asarray([f[primary] if primary in f else f[alternate] for f in frames], float)
    roots = coordinates('root_world_m', 'root')
    heads = coordinates('head_world_m', 'head')
    tails = coordinates('tail_world_m', 'tail')
    if len(t) < 2 or any(x.shape != (len(t), 3) for x in (roots, heads, tails)):
        raise ValueError('aligned_native_direction_frames_required')
    if not all(np.isfinite(x).all() for x in (t, roots, heads, tails)) or np.any(np.diff(t) <= 0):
        raise ValueError('finite_increasing_native_clock_required')
    velocity = np.diff(roots[:, [0, 2]], axis=0) / np.diff(t)[:, None]
    speeds = np.linalg.norm(velocity, axis=1)
    angles = []
    failures = []
    for interval in np.flatnonzero(speeds > .005):
        for frame in (interval, interval + 1):
            axis = (heads[frame] - tails[frame])[[0, 2]]
            length = np.linalg.norm(axis)
            if length < 1e-6:
                raise ValueError('degenerate_head_tail_axis')
            angle = float(np.degrees(np.arccos(np.clip(axis @ velocity[interval] / (length * speeds[interval]), -1, 1))))
            angles.append(angle)
            if angle > 20. + 1e-9:
                failures.append({'frame': int(frame), 'interval': int(interval), 'angle_deg': angle})
    if not angles:
        failures.append({'reason': 'no_actual_moving_interval'})
    return {'status': 'fail' if failures else 'pass', 'maximum_angle_deg': max(angles, default=None),
            'checked_moving_intervals': int((speeds > .005).sum()), 'threshold_deg': 20., 'failures': failures}


def contact_checks(frames, mode):
    failures, missing = [], []
    anchors = {}
    phases = {side: set() for side in PAWS}
    max_gap = max_slip = 0.
    floors = []
    for i, frame in enumerate(frames):
        feet = frame.get('feet', {})
        if set(feet) != set(PAWS):
            missing.append(f'{i}:four_paw_measurements_missing')
            continue
        if not any(foot.get('in_contact') for foot in feet.values()):
            failures.append(f'{i}:no_support_paw')
        for side, foot in feet.items():
            delta = float(foot['floor_delta_m'])
            if not math.isfinite(delta):
                failures.append(f'{i}:{side}:nonfinite_floor_gap')
                continue
            if delta < -FLOOR_PENETRATION_M:
                failures.append(f'{i}:{side}:paw_penetration')
            contact = bool(foot['in_contact'])
            phases[side].add(contact)
            if contact:
                max_gap = max(max_gap, delta)
                if delta > .012 + 1e-12:
                    failures.append(f'{i}:{side}:support_gap_over_12mm')
                if mode == 'walk_animation':
                    segment = foot.get('support_segment')
                    if segment is None:
                        missing.append(f'{i}:{side}:support_step_identity_missing')
                        continue
                    point = np.asarray(foot['support_world_m'], float)
                    if point.shape != (3,) or not np.isfinite(point).all():
                        failures.append(f'{i}:{side}:nonfinite_support_point')
                        continue
                    key = (side, segment)
                    anchors.setdefault(key, point)
                    drift = float(np.linalg.norm(point - anchors[key]))
                    max_slip = max(max_slip, drift)
                    if drift > .015 + 1e-12:
                        failures.append(f'{i}:{side}:step_slide_over_15mm')
        if 'body_floor_min_m' not in frame:
            missing.append(f'{i}:full_skin_floor_measurement_missing')
        else:
            floor = float(frame['body_floor_min_m'])
            floors.append(floor)
            if not math.isfinite(floor) or floor < -FLOOR_PENETRATION_M:
                failures.append(f'{i}:full_skin_penetration')
    if mode == 'walk_animation':
        for side, observed in phases.items():
            if observed != {False, True}:
                missing.append(side + ':complete_support_and_swing_not_observed')
    return {'status': 'fail' if failures else 'unverified' if missing else 'pass',
            'failures': failures, 'missing_evidence': missing, 'maximum_support_gap_m': max_gap,
            'maximum_support_step_slide_m': max_slip, 'minimum_full_skin_floor_m': min(floors, default=None),
            'swing_gap_has_no_12mm_gate': True, 'plant_speed_has_no_gate': True,
            'slide_exempt_from_gait_claim': mode == 'slide_no_animation'}


def surface_checks(surface, frame_count):
    if surface is None:
        return {'status': 'unverified', 'missing_evidence': ['surface_and_stretch_measurements_missing'], 'failures': []}
    if surface.get('schema') == 'rigid_slide_shape_invariance_v1':
        if surface.get('verified') and surface.get('frame_count') == frame_count:
            return {'status': 'pass', 'failures': [], 'maximum_cross_region_new_area_m2': 0.,
                    'maximum_cross_region_new_depth_m': 0., 'maximum_edge_growth': 1.,
                    'basis': 'one unchanged actual pose under rigid root transforms'}
        return {'status': 'unverified', 'missing_evidence': ['rigid_slide_shape_invariance_missing'], 'failures': []}
    rows = surface.get('frames', [])
    measured = min(len(rows), frame_count)
    complete = measured == frame_count and [row.get('frame') for row in rows[:frame_count]] == list(range(frame_count))
    if [row.get('frame') for row in rows[:measured]] != list(range(measured)):
        return {'status': 'unverified', 'missing_evidence': ['surface_frame_identity_mismatch'], 'failures': []}
    failures = []
    area_max = depth_max = 0.
    growth_max = 1.
    for row in rows[:frame_count]:
        if 'regional_penetration_measurements' not in row or 'max_edge_growth' not in row:
            return {'status': 'unverified', 'missing_evidence': ['per_frame_surface_or_stretch_measurements_missing'], 'failures': failures}
        cross = [v for v in row['regional_penetration_measurements'] if v['region_pair'][0] != v['region_pair'][1]]
        area = sum(float(v['new_area_m2']) for v in cross)
        depth = max((float(v['new_maximum_depth_m']) for v in cross), default=0.)
        growth = float(row['max_edge_growth'])
        if not all(math.isfinite(v) and v >= 0 for v in (area, depth, growth)) or row.get('nonfinite_vertices', 0):
            failures.append({'frame': row['frame'], 'reason': 'nonfinite_surface_measurement'})
        if area > 400e-6 + 1e-12:
            failures.append({'frame': row['frame'], 'reason': 'cross_region_total_area_over_400mm2', 'value_m2': area})
        if depth > .010 + 1e-9:
            failures.append({'frame': row['frame'], 'reason': 'cross_region_depth_over_10mm', 'value_m': depth})
        if growth > 2. + 1e-6:
            failures.append({'frame': row['frame'], 'reason': 'edge_growth_over_2', 'value': growth})
        area_max, depth_max, growth_max = max(area_max, area), max(depth_max, depth), max(growth_max, growth)
    return {'status': 'fail' if failures else 'pass' if complete else 'unverified', 'failures': failures,
            'missing_evidence': [] if complete else ['complete_surface_frame_coverage_missing'],
            'evaluated_frame_count': measured, 'required_frame_count': frame_count, 'maximum_cross_region_new_area_m2': area_max,
            'maximum_cross_region_new_depth_m': depth_max, 'maximum_edge_growth': growth_max,
            'same_region_intersections_ignored': True, 'uv_seam_is_diagnostic_only': True,
            'historical_surface_status': surface.get('status')}


def physical_direction_controls(native, evidence=None):
    candidates = []
    if native.get('native_direction_negative_frames'):
        for angle, frames in native['native_direction_negative_frames'].items():
            candidates.append({'body_rotation_error_deg': float(angle), 'frames': frames,
                               'physically_rotated_native_drawable': all(f.get('physically_rotated_native_drawable') for f in frames)})
    if evidence:
        candidates.extend(c for c in evidence.get('controls', []) if c.get('asset_id', native['asset_id']) == native['asset_id'])
    results = []
    for angle in (180, 90):
        matching = [c for c in candidates if abs(c.get('body_rotation_error_deg', -1) - angle) < 1e-6 and c.get('physically_rotated_native_drawable')]
        if not matching:
            return {'status': 'unverified', 'missing_evidence': [f'physical_{angle}deg_control_missing'], 'controls': results}
        control = matching[-1]
        check = direction_checks(control['frames'])
        results.append({'applied_rotation_deg': angle, 'check': check, 'physically_rotated_native_drawable': True})
    return {'status': 'pass' if all(x['check']['status'] == 'fail' for x in results) else 'fail', 'controls': results}


def rigid_slide_surface_proof(native):
    """One invariant rigid pose can use one file or identical bounded buffers."""
    import hashlib
    frames = native.get('frames', [])
    paths = {f.get('pose_glb') for f in frames}
    verified = native.get('motion_mode') == 'slide_no_animation' and bool(frames) and None not in paths
    hashes = {}
    if verified:
        for value in paths:
            path = Path(value)
            if not path.is_file():
                verified = False
                break
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        verified = verified and len(set(hashes.values())) == 1
    if verified and len(paths) > 1:
        # A reused slot is not an archived walking frame. Rigid slide receipts
        # must separately retain invariant local attachment and native bounds.
        for key in ('head_m', 'tail_m', 'mouth_m', 'mouth_bone_matrix'):
            if any(key not in f for f in frames):
                verified = False
                break
            values = np.asarray([f[key] for f in frames], float)
            verified = verified and np.isfinite(values).all() and np.allclose(values, values[0], atol=1e-10, rtol=0)
        bounds = [f.get('native_geometry_bounds_m') for f in frames]
        if any(v is None for v in bounds):
            verified = False
        else:
            values = np.asarray(bounds, float)
            verified = verified and np.isfinite(values).all() and np.allclose(values, values[0], atol=2e-5, rtol=0)
    return {'schema': 'rigid_slide_shape_invariance_v1', 'verified': bool(verified),
            'frame_count': len(frames), 'pose_paths': sorted(str(p) for p in paths),
            'pose_sha256': hashes,
            'basis': 'identical rigid mesh bytes, invariant local attachments and native bounds; original overlaps are the baseline'}


def native_context_checks(native):
    """Preserve existing measured room/attachment validity and full clock coverage."""
    frames = native.get('frames', [])
    missing, failures = [], []
    if native.get('status') in ('failed', 'pending') or native.get('frame_count') != len(frames):
        missing.append('native_replay_not_declared_complete')
    times = np.asarray([f['time_s'] for f in frames], float)
    if len(times) < 2 or not np.allclose(times, np.arange(len(times)) / 30, atol=1e-10, rtol=0):
        missing.append('complete_30Hz_native_clock_missing')
    for i, frame in enumerate(frames):
        limits = (('native_emitter_readback_error_m', .0001, 'mouth_attachment'),
                  ('navmesh_horizontal_error_m', .002, 'native_navigation'))
        for key, limit, label in limits:
            if key not in frame:missing.append(f'{i}:{label}_measurement_missing')
            elif not math.isfinite(float(frame[key])) or frame[key] > limit:failures.append(f'{i}:{label}')
        if 'minimum_nonfloor_clearance_m' not in frame:
            missing.append(f'{i}:scene_collision_measurement_missing')
        elif not math.isfinite(float(frame['minimum_nonfloor_clearance_m'])) or frame['minimum_nonfloor_clearance_m'] <= .02:
            failures.append(f'{i}:nonfloor_collision')
        if not frame.get('complete_silhouette_in_frame'):
            missing.append(f'{i}:full_native_silhouette_not_recorded')
    return {'status': 'fail' if failures else 'unverified' if missing else 'pass',
            'failures': failures, 'missing_evidence': missing,
            'scope': 'unchanged actual room collision, mouth attachment and navigation evidence; not an audio-energy gate'}


def qualify_motion(native, surface=None, controls=None, *, mode=None):
    mode = mode or native.get('motion_mode', 'walk_animation')
    frames = native.get('frames', [])
    if mode not in ('walk_animation', 'slide_no_animation') or len(frames) < 2:
        return {'status': 'unverified', 'failures': [], 'missing_evidence': ['complete_native_motion_missing'],
                'validation_policy': validation_policy(), 'audio_status': AUDIO_STATUS}
    parts = {'native_context': native_context_checks(native), 'direction': direction_checks(frames), 'contact': contact_checks(frames, mode),
             'surface': surface_checks(surface, len(frames)), 'physical_controls': physical_direction_controls(native, controls)}
    failed = [name for name, value in parts.items() if value['status'] == 'fail']
    missing = [name for name, value in parts.items() if value['status'] == 'unverified']
    return {'status': 'fail' if failed else 'unverified' if missing else 'pass',
            'failures': failed, 'missing_evidence': missing, 'checks': parts,
            'validation_policy': validation_policy(), 'admission_scope': 'motion_only',
            'audio_status': AUDIO_STATUS, 'audio_deferred_to_production': True,
            'historical_strict_checks': native.get('checks', {}).get('status'), 'frame_count': len(frames)}
