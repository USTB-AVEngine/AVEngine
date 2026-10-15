"""Direct construction of the executed stand -> walk -> stand straight line for moving sources (Claude 2026-10-15).

owner 10-15: "所有题目都应该是'刻意'去造的", "缺什么补什么".  The executed walk is one straight line with smooth
start/stop ramps (reflect_navmesh._short_window_walk: a planted foot cannot follow reflected corners).  Instead of
hoping that a random reflected walk contains a long enough straight leg (10-15: 3,611 "no whole-source sounding
leg" rejections), search the room directly: free grid points x headings, keep lines whose whole walk -- ramps,
lead, emission, silent query interval, lead -- stays free and whose emission geometry serves the requested motion
families.  Nothing here decides a truth; the native stage re-places the line and every judge reads the readback.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import shape

from .reflect_walk import SPECIES_SPEED_M_S

FPS = 30
FRAMES = 901
WET_TAIL_S = .5
POST_S = 2.0          # >= 2 s silent moving after the wet tail (QA-16/QA-17 construction request)
RADIAL_COS = math.cos(math.radians(30))


def stand_walk_stand_distance(w0, w1, speed, ramp):
    """Arc length per 30 Hz frame: stand, smoothstep ramp up, constant speed, ramp down, stand."""
    t = np.arange(FRAMES) / FPS
    edge = np.clip(np.minimum(t - w0, w1 - t) / ramp, 0, 1)
    v = np.where((t >= w0) & (t <= w1), speed * edge * edge * (3 - 2 * edge), 0.)
    return np.r_[0., np.cumsum((v[1:] + v[:-1]) / 2 / FPS)]


def _radial_flags(positions, listener, k, e, q, heading):
    d = np.array([math.cos(heading), math.sin(heading)])
    r = np.linalg.norm(positions[k:q + 1] - listener, axis=1)
    emit = r[:e - k + 1]
    toward = listener - positions[k]
    cosv = abs(float(d @ toward)) / (float(np.linalg.norm(toward)) + 1e-12)
    net = float(emit[-1] - emit[0])
    rising = bool(np.all(np.diff(emit) >= -.005)); falling = bool(np.all(np.diff(emit) <= .005))
    qa15 = cosv >= RADIAL_COS and abs(net) >= .5 and (rising or falling)
    silent = r[e - k + int(round(WET_TAIL_S * FPS)):]
    same = (net < 0 and np.all(np.diff(silent) <= .005)) or (net > 0 and np.all(np.diff(silent) >= -.005))
    qa16 = bool(qa15 and same and abs(float(silent[-1] - silent[0])) >= .2)
    return bool(qa15), qa16, net, cosv


def find_straight_route(room, pathfinder, seed, attempt=0, *, species='human', utterance_s=1.5, focus='M1',
                        speed_range=None, is_free=None, grid_m=.3, headings=24, keep=12, max_grid=600):
    """All four motion families when the room is deep enough, else the focus pair (owner 10-15: build what fits).

    'all4': radial line that keeps walking >= 2 s after the wet tail (QA-06/15/16/17);
    'M1'  : any direction, keeps walking >= 2 s after the wet tail (QA-06/17);
    'M2'  : radial during the sound, stops shortly after it (QA-06/15) -- a much shorter line.
    """
    errors = []
    for families in ('all4', 'M2' if focus == 'M2' else 'M1'):
        try:
            return _find_straight_route(room, pathfinder, seed, attempt, species=species, utterance_s=utterance_s,
                                        focus=focus, families=families, speed_range=speed_range, is_free=is_free,
                                        grid_m=grid_m, headings=headings, keep=keep, max_grid=max_grid)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError(' | '.join(errors))


def _find_straight_route(room, pathfinder, seed, attempt=0, *, species='human', utterance_s=1.5, focus='M1',
                         families='all4', speed_range=None, is_free=None, grid_m=.3, headings=24, keep=12, max_grid=600):
    from .reflect_navmesh import native_free_predicate, SHORT_WALK_LEAD_S, SHORT_WALK_RAMP_S
    rng = np.random.default_rng(int(seed) + int(attempt) * 101 + 17)
    polygon = shape(room['floor_polygon']); floor = float(room['measured_floor_y_m'])
    listener = np.asarray(room['placement']['camera_m'], float); L = listener[[0, 2]]
    if is_free is None:
        is_free = native_free_predicate(pathfinder, polygon, floor, listener)
    lo, hi = SPECIES_SPEED_M_S[species]
    if speed_range is None:
        speed_range = (lo + .02, hi - .02)
    if focus == 'M1':
        speed_range = (.82, .9) if species == 'human' else (lo + .02, hi - .02)
    speed = float(rng.uniform(max(lo + .01, speed_range[0]), min(hi - .01, speed_range[1])))
    duration = math.ceil(float(utterance_s) * 10 - 1e-9) / 10
    offset = float(rng.choice([0., 5., 10., 15., 20.]))
    start = offset + round(float(rng.uniform(1., 2.)), 1)
    end = start + duration
    post = families in ('all4', 'M1')
    tail = end + WET_TAIL_S + POST_S if post else end + .6
    if tail > offset + 9.8:
        raise ValueError('straight route: emission plus silent query interval does not fit one 10 s crop')
    w0, w1 = start - SHORT_WALK_LEAD_S, tail + SHORT_WALK_LEAD_S
    s = stand_walk_stand_distance(w0, w1, speed, SHORT_WALK_RAMP_S)
    k, e, q = (int(round(x * FPS)) for x in (start, end, tail))
    total = float(s[-1])
    samples = np.r_[np.arange(0., total, .1), total][::-1]
    x0, z0, x1, z1 = polygon.bounds
    grid = [np.array([x, z]) for x in np.arange(x0, x1 + 1e-9, grid_m) for z in np.arange(z0, z1 + 1e-9, grid_m)]
    rng.shuffle(grid); grid = grid[:max_grid]
    found = []; tested = 0; rejected = {'start_not_free': 0, 'radial_slot': 0, 'line_not_free': 0}
    for p in grid:
        if not is_free(p):
            rejected['start_not_free'] += 1; continue
        base = float(rng.uniform(0, 2 * math.pi / headings))
        for j in range(headings):
            h = base + 2 * math.pi * j / headings
            d = np.array([math.cos(h), math.sin(h)])
            origin = p - s[k] * d
            if families in ('all4', 'M2'):
                positions = origin + np.outer(s, d)
                qa15, qa16, _, _ = _radial_flags(positions, L, k, e, q, h)
                if not qa15 or (families == 'all4' and not qa16):
                    rejected['radial_slot'] += 1; continue
            tested += 1
            if not all(is_free(origin + c * d) for c in samples):
                rejected['line_not_free'] += 1; continue
            found.append((origin, h)); break
        if len(found) >= keep:
            break
    if not found:
        raise ValueError('straight route: no free straight %.2f m line for %s %s; tested=%d rejected=%s'
                         % (total, species, families, tested, rejected))
    origin, heading = found[int(rng.integers(len(found)))]
    d = np.array([math.cos(heading), math.sin(heading)])
    positions = origin + np.outer(s, d)
    qa15, qa16, net, cosv = _radial_flags(positions, L, k, e, q, heading)
    if not post:
        qa16 = False
    frames = (int(round(w0 * FPS)), int(round(w1 * FPS)))      # the whole walk is one straight leg
    slot = {'start_s': start, 'end_s': end, 'leg': frames, 'window_frames': frames,
            'radial_change_m': net, 'branch': 'nearer' if net < 0 else 'farther', 'radial_cos': cosv,
            'post_until_s': tail if post else None, 'qa06': True, 'qa15': qa15, 'qa16': qa16, 'qa17': post,
            'constructed_families': families,
            'kind': 'radial_emission' if qa15 else 'moving_emission', 'offset_s': offset}
    window = {'walk_start_s': w0, 'walk_end_s': w1, 'ramp_s': SHORT_WALK_RAMP_S, 'speed_m_s': speed, 'length_m': total,
              'emission_s': [start, end], 'silent_query_until_s': tail, 'heading_rad': heading, 'straight_line': True,
              'reason': 'owner 10-15: construct deliberately; one free straight stand-walk-stand line found directly'}
    target = np.mean(positions, axis=0) - L
    return {'reflect_walk': True, 'positions_xz_m': positions.tolist(), 'travel_yaws_rad': [heading] * FRAMES,
            'short_window_walk': window, 'speed_m_s': speed, 'speed_range_m_s': [lo, hi], 'listener_m': listener.tolist(),
            'camera_yaw_deg': math.degrees(math.atan2(target[0], -target[1])), 'species': species, 'template': focus,
            'emission_slot': slot, 'candidate_count': tested,
            'candidate_trials': [{'construction': 'direct_straight_line', 'found': len(found), 'rejected': rejected}],
            'selected_attempt': 'straight-%d-%d' % (int(seed), int(attempt)), 'candidate_native_root_samples': FRAMES,
            'construction': 'direct_straight_line_search',
            'geometry_status': 'native NavMesh proposal only; original pose/contact/collision/readback gates pending'}
