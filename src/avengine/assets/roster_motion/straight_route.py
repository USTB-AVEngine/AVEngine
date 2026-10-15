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
from shapely.geometry import shape, LineString

from .reflect_walk import SPECIES_SPEED_M_S

FPS = 30
FRAMES = 901
WET_TAIL_S = .5
POST_S = 1.6          # >= 1.5 s silent moving after the wet tail (QA-16/QA-17 construction request) + 0.1 s
# Construction heuristic only: QA-15's judge reads a monotonic >= 0.5 m distance change, not an angle (owner 10-15).
RADIAL_COS = math.cos(math.radians(60))
CORNER_FRACTIONS = (.3, .5, .7)
CORNER_TURNS_DEG = (60., 90., 120., -60., -90., -120.)
CORNER_RADII_M = (.8, 1.2)      # flat-floor replays 10-15: radius 0.6-2 m and a 180 deg turn all passed the gait gates


NATIVE_HFOV_DEG = 85.          # continuous_native_video / continuous_ue_video render hfov
BODY_HALF_WIDTH_M = .3         # shoulders and arms
EDGE_MARGIN_DEG = 3.
MAX_YAW_SHIFT_DEG = 35.


def _bearing_deg(point, listener_xz):
    d = np.asarray(point, float) - np.asarray(listener_xz, float)
    return math.degrees(math.atan2(d[0], -d[1]))


def rest_framing(point, listener_xz, yaw_deg, hfov_deg=NATIVE_HFOV_DEG):
    """'inside' / 'outside' when the whole standing body is clearly in or out of the frame, else 'edge'."""
    d = float(np.linalg.norm(np.asarray(point, float) - np.asarray(listener_xz, float)))
    w = math.degrees(math.atan2(BODY_HALF_WIDTH_M, max(d, 1e-6)))
    off = abs((_bearing_deg(point, listener_xz) - yaw_deg + 180.) % 360. - 180.)
    if off <= hfov_deg / 2 - w - EDGE_MARGIN_DEG:
        return 'inside'
    if off >= hfov_deg / 2 + w + EDGE_MARGIN_DEG:
        return 'outside'
    return 'edge'


def frame_rest_poses(rest_points, listener_xz, nominal_yaw_deg):
    """Camera yaw near the nominal one where no standing pose sits on the frame edge (owner 10-16 review).

    hm3d 00022: the walker stood ~10 s at the left edge 1.5 m from the camera, only a shoulder visible.  Yaws within
    MAX_YAW_SHIFT_DEG of the nominal (which looks at the walk) are tried nearest first; more poses fully inside is
    better.  Returns (yaw, record); the nominal yaw is kept and recorded when no yaw clears every pose.
    """
    best = None
    for shift in sorted(np.arange(-MAX_YAW_SHIFT_DEG, MAX_YAW_SHIFT_DEG + 1e-9, 1.), key=abs):
        yaw = nominal_yaw_deg + float(shift)
        states = [rest_framing(p, listener_xz, yaw) for p in rest_points]
        if 'edge' in states:
            continue
        key = (states.count('inside'), -abs(shift))
        if best is None or key > best[0]:
            best = (key, yaw, states)
    if best is None:
        states = [rest_framing(p, listener_xz, nominal_yaw_deg) for p in rest_points]
        return nominal_yaw_deg, {'status': 'unresolved_kept_nominal', 'nominal_yaw_deg': nominal_yaw_deg, 'states': states}
    return best[1], {'status': 'pass', 'nominal_yaw_deg': nominal_yaw_deg, 'yaw_shift_deg': best[1] - nominal_yaw_deg,
                     'states': best[2], 'hfov_deg': NATIVE_HFOV_DEG, 'body_half_width_m': BODY_HALF_WIDTH_M}


def stand_walk_stand_distance(w0, w1, speed, ramp):
    """Arc length per 30 Hz frame: stand, smoothstep ramp up, constant speed, ramp down, stand."""
    t = np.arange(FRAMES) / FPS
    edge = np.clip(np.minimum(t - w0, w1 - t) / ramp, 0, 1)
    v = np.where((t >= w0) & (t <= w1), speed * edge * edge * (3 - 2 * edge), 0.)
    return np.r_[0., np.cumsum((v[1:] + v[:-1]) / 2 / FPS)]


def _shape_xy(sig, origin, heading, corner):
    """Positions and headings at arc lengths `sig` for a straight line or a line with one rounded corner.

    corner = (sigma_c, radius, turn_rad): straight until sigma_c, an arc of `radius` turning `turn_rad`, then straight.
    The yaw rate on the arc is speed / radius (0.8 m at 0.82 m/s = 59 deg/s).
    """
    sig = np.asarray(sig, float); d0 = np.array([math.cos(heading), math.sin(heading)])
    if corner is None:
        return origin + np.outer(sig, d0), np.full(len(sig), float(heading))
    sc, radius, turn = corner; arc = abs(turn) * radius; sign = 1. if turn > 0 else -1.
    pc = origin + sc * d0; centre = pc + radius * sign * np.array([-d0[1], d0[0]]); v = pc - centre
    out = np.empty((len(sig), 2)); head = np.empty(len(sig))
    m1 = sig <= sc; out[m1] = origin + np.outer(sig[m1], d0); head[m1] = heading
    m2 = (sig > sc) & (sig <= sc + arc); a = sign * (sig[m2] - sc) / radius; c, sn = np.cos(a), np.sin(a)
    out[m2] = centre + np.column_stack((c * v[0] - sn * v[1], sn * v[0] + c * v[1])); head[m2] = heading + a
    m3 = sig > sc + arc; d1 = np.array([math.cos(heading + turn), math.sin(heading + turn)])
    pe = centre + np.array([math.cos(turn) * v[0] - math.sin(turn) * v[1], math.sin(turn) * v[0] + math.cos(turn) * v[1]])
    out[m3] = pe + np.outer(sig[m3] - sc - arc, d1); head[m3] = heading + turn
    return out, head


def _radial_flags(positions, listener, k, e, q, heading=None):
    d = positions[min(len(positions) - 1, k + 1)] - positions[k]
    d = d / (float(np.linalg.norm(d)) + 1e-12)
    r = np.linalg.norm(positions[k:q + 1] - listener, axis=1)
    emit = r[:e - k + 1]
    # The audible part of a recording is shorter than its file (leading/trailing silence): hm3d 00272 planned 0.5 m
    # over the file but measured 0.415 m over the audible window.  Require the 0.5 m inside the middle 60 %.
    trim = int(round(.2 * (e - k)))
    inner = emit[trim:len(emit) - trim] if len(emit) - 2 * trim >= 2 else emit
    toward = listener - positions[k]
    cosv = abs(float(d @ toward)) / (float(np.linalg.norm(toward)) + 1e-12)
    net = float(inner[-1] - inner[0])
    rising = bool(np.all(np.diff(emit) >= -.005)); falling = bool(np.all(np.diff(emit) <= .005))
    qa15 = abs(net) >= .5 and (rising or falling)          # the judge's criterion; cos is recorded only
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
    for index, candidate in enumerate(_rooms_with_cameras(room, pathfinder, species, relocate=is_free is None)):
        pair = 'M2' if focus == 'M2' else 'M1'
        for families, corners, use_map in (('all4', False, True), (pair, False, True), (pair, True, True),
                                           ('all4', False, False), (pair, False, False), (pair, True, False)):
            try:
                route = _find_straight_route(candidate, pathfinder, seed, attempt, species=species, utterance_s=utterance_s,
                                             focus=focus, families=families, speed_range=speed_range, is_free=is_free,
                                             grid_m=grid_m, headings=headings, keep=keep, max_grid=max_grid, corners=corners,
                                             use_map=use_map)
            except ValueError as error:
                errors.append(('camera %d: ' % index) + str(error)); continue
            if index:
                route['camera_relocation'] = {
                    'reason': 'owner 10-15: build the scene the question needs; the room camera left no free line',
                    'room_camera_m': list(room['placement']['camera_m']), 'constructed_camera_m': route['listener_m'],
                    'candidate_index': index}
            return route
    raise ValueError(' | '.join(errors[:4]) + (' | ... %d more' % (len(errors) - 4) if len(errors) > 4 else ''))


def _rooms_with_cameras(room, pathfinder, species, relocate=True, limit=12):
    """The room's own camera first, then fixed cameras 0.5 m inside the room edge, farthest from the centre first.

    A camera in the middle of the floor (e.g. hm3d 00022 R8) leaves no 3 m line outside the 1.5 m person margin.
    Candidates stand on the human walkable map (not inside furniture) and on the NavMesh; height is kept.
    """
    yield room
    if not relocate:
        return
    from copy import deepcopy
    from shapely.geometry import MultiPolygon
    polygon = shape(room['floor_polygon']); floor = float(room['measured_floor_y_m'])
    camera = np.asarray(room['placement']['camera_m'], float); height = float(camera[1])
    inner = polygon.buffer(-.5)
    if inner.is_empty:
        return
    parts = list(inner.geoms) if isinstance(inner, MultiPolygon) else [inner]
    ring = max(parts, key=lambda g: g.area).exterior; centre = np.asarray(max(parts, key=lambda g: g.area).centroid.coords[0])
    walkable_map = None
    try:
        from avengine.rooms.walkable_map import load_room_map
        walkable_map = load_room_map(room.get('room_label', room.get('room_id')), path=room.get('walkable_map_path'),
                                     floor_reference=floor)
    except (ValueError, FileNotFoundError):
        walkable_map = None
    if walkable_map is not None and float(walkable_map.layers['human'].sum()) * walkable_map.resolution ** 2 < .25 * inner.area:
        walkable_map = None          # implausibly small cached map: rely on the NavMesh test for camera candidates
    points = []
    for d in np.arange(0., ring.length, .5):
        p = np.asarray(ring.interpolate(d).coords[0], float)
        if walkable_map is not None and not walkable_map.contains(p, 'human'):
            continue
        q = np.array([p[0], floor, p[1]]); s = np.asarray(pathfinder.snap_point(q), float)
        if not np.isfinite(s).all() or np.linalg.norm((s - q)[[0, 2]]) > .02:
            continue
        points.append(p)
    points.sort(key=lambda p: -float(np.linalg.norm(p - centre)))
    for p in points[:limit]:
        moved = deepcopy(room); moved['placement'] = dict(room['placement'], camera_m=[float(p[0]), height, float(p[1])])
        yield moved


def _find_straight_route(room, pathfinder, seed, attempt=0, *, species='human', utterance_s=1.5, focus='M1',
                         families='all4', speed_range=None, is_free=None, grid_m=.3, headings=24, keep=12, max_grid=600,
                         corners=False, use_map=True, minimum_camera_distance_m=None, require_reverse_radial=False, require_rest_in_frame=False, local_start_s=None):
    from .reflect_navmesh import native_free_predicate, SHORT_WALK_LEAD_S, SHORT_WALK_RAMP_S
    rng = np.random.default_rng(int(seed) + int(attempt) * 101 + 17)
    polygon = shape(room['floor_polygon']); floor = float(room['measured_floor_y_m'])
    listener = np.asarray(room['placement']['camera_m'], float); L = listener[[0, 2]]
    walkable_map = None
    if is_free is None:
        # The per-room static walkable map (furniture, rugs, steps, other floors excluded per species radius) is
        # consulted first, exactly as the reflected search does; native point tests still run at every sample.
        # 10-15 measured: the cached maps are far too conservative in most rooms (518/777 HM3D and 159/178 MP3D rooms
        # have < 2 m2 human-walkable area, e.g. mp3d JeFG25 R18 0.8 m2 of an 18 m2 floor). The map is a construction
        # aid, not a gate: when it leaves no line, retry without it; the native full-pose gates still judge.
        if use_map:
            from avengine.rooms.walkable_map import load_room_map
            walkable_map = load_room_map(room.get('room_label', room.get('room_id')), path=room.get('walkable_map_path'),
                                         floor_reference=floor)
        is_free = native_free_predicate(pathfinder, polygon, floor, listener, walkable_map=walkable_map, species=species, camera_margin=minimum_camera_distance_m)
    segment_is_free = getattr(is_free, 'segment_is_free', None)
    coverage_polygon=polygon.buffer(.002)
    lo, hi = SPECIES_SPEED_M_S[species]
    if speed_range is None:
        speed_range = (lo + .02, hi - .02)
    if focus == 'M1':
        speed_range = (.82, .9) if species == 'human' else (lo + .02, hi - .02)
    slow, fast = max(lo + .01, speed_range[0]), min(hi - .01, speed_range[1])
    duration = math.ceil(float(utterance_s) * 10 - 1e-9) / 10
    # The lower quarter of the owner band: realistic indoor pace and a shorter line for the same timing.
    speed = float(rng.uniform(slow, slow + .25 * max(0., fast - slow)))
    if families in ('all4', 'M2'):
        # QA-15 needs >= 0.5 m of distance change inside the middle 60 % of the sound: slow species (cat, slide)
        # walk faster within their own band when the sound is short (never above the band).
        speed = float(min(fast, max(speed, .55 / (.6 * duration * .9))))
    offset = float(rng.choice([0., 5., 10., 15., 20.]))
    start = offset + (float(local_start_s) if local_start_s is not None else round(float(rng.uniform(1., 2.)), 1))
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
    from .reflect_navmesh import CAMERA_MARGIN_M
    camera_margin = max(CAMERA_MARGIN_M.get(species, 1.), float(minimum_camera_distance_m or 0.))
    x0, z0, x1, z1 = polygon.bounds
    grid = [np.array([x, z]) for x in np.arange(x0, x1 + 1e-9, grid_m) for z in np.arange(z0, z1 + 1e-9, grid_m)]
    rng.shuffle(grid); grid = grid[:max_grid if not corners else min(max_grid, 220)]
    shapes = [None] if not corners else [(f * total, r, math.radians(t)) for f in CORNER_FRACTIONS
                                         for r in CORNER_RADII_M for t in CORNER_TURNS_DEG]
    n_heads = headings if not corners else max(8, headings // 2)
    found = []; tested = 0; rejected = {'start_not_free': 0, 'camera': 0, 'walkable_map': 0, 'radial_slot': 0, 'line_not_free': 0}
    for p in grid:
        if not is_free(p):
            rejected['start_not_free'] += 1; continue
        base = float(rng.uniform(0, 2 * math.pi / n_heads))
        for j in range(n_heads):
            h = base + 2 * math.pi * j / n_heads
            for corner in shapes:
                pts, _ = _shape_xy(samples, p, h, corner)
                if np.any(np.linalg.norm(pts - L, axis=1) < camera_margin):
                    rejected['camera'] += 1; continue
                if walkable_map is not None and not walkable_map.contains_many(pts, species).all():
                    rejected['walkable_map'] += 1; continue
                positions, _ = _shape_xy(s, p, h, corner)
                if families in ('all4', 'M2'):
                    qa15, qa16, _, _ = _radial_flags(positions, L, k, e, q)
                    if not qa15 or (families == 'all4' and not qa16):
                        rejected['radial_slot'] += 1; continue
                    if require_reverse_radial:
                        reverse_qa15, reverse_qa16, _, _ = _radial_flags(positions[0] + positions[-1] - positions, L, k, e, q)
                        if not reverse_qa15 or (families == 'all4' and not reverse_qa16):
                            rejected['radial_slot'] += 1; continue
                if require_rest_in_frame:
                    target=np.mean(positions,axis=0)-L
                    _,framing=frame_rest_poses([positions[0],positions[-1]],L,math.degrees(math.atan2(target[0],-target[1])))
                    if framing['states']!=['inside','inside']:
                        rejected['rest_pose_framing']=rejected.get('rest_pose_framing',0)+1;continue
                tested += 1
                if not all(is_free(x) for x in pts):
                    rejected['line_not_free'] += 1; continue
                if not coverage_polygon.covers(LineString(positions)):
                    rejected['line_not_free'] += 1; continue
                if not all(is_free(x) for x in np.unique(positions,axis=0)):
                    rejected['line_not_free'] += 1; continue
                if segment_is_free is not None and not all(segment_is_free(a,b) for a,b in zip(positions[:-1],positions[1:]) if np.linalg.norm(b-a)>1e-10):
                    rejected['line_not_free'] += 1; continue
                found.append((p, h, corner)); break
            if found and found[-1][0] is p:
                break
        if len(found) >= keep:
            break
    if not found:
        raise ValueError('straight route: no free %s%.2f m line for %s %s; tested=%d rejected=%s'
                         % ('cornered ' if corners else 'straight ', total, species, families, tested, rejected))
    origin, heading, corner = found[int(rng.integers(len(found)))]
    positions, yaws = _shape_xy(s, origin, heading, corner)
    qa15, qa16, net, cosv = _radial_flags(positions, L, k, e, q)
    if families == 'M1':
        # A route not built as radial must not advertise the distance-trend families by chance: UwV83 claimed
        # QA-16 from a near-radial plan and failed the post-sound trend on the render.
        qa15 = qa16 = False
    if not post:
        qa16 = False
    frames = (int(round(w0 * FPS)), int(round(w1 * FPS)))      # the whole walk is one straight leg
    slot = {'start_s': start, 'end_s': end, 'leg': frames, 'window_frames': frames,
            'radial_change_m': net, 'branch': 'nearer' if net < 0 else 'farther', 'radial_cos': cosv,
            'post_until_s': tail if post else None, 'qa06': True, 'qa15': qa15, 'qa16': qa16, 'qa17': post,
            'constructed_families': families,
            'kind': 'radial_emission' if qa15 else 'moving_emission', 'offset_s': offset}
    window = {'walk_start_s': w0, 'walk_end_s': w1, 'ramp_s': SHORT_WALK_RAMP_S, 'speed_m_s': speed, 'length_m': total,
              'emission_s': [start, end], 'silent_query_until_s': tail, 'heading_rad': heading, 'straight_line': corner is None,
              'shape': None if corner is None else {'corner_at_m': corner[0], 'radius_m': corner[1], 'turn_deg': math.degrees(corner[2]),
                                                   'maximum_yaw_rate_deg_s': math.degrees(speed / corner[1])},
              'reason': 'owner 10-15: construct deliberately; one free straight stand-walk-stand line found directly'}
    target = np.mean(positions, axis=0) - L
    camera_yaw, framing = frame_rest_poses([positions[0], positions[-1]], L, math.degrees(math.atan2(target[0], -target[1])))
    return {'reflect_walk': True, 'positions_xz_m': positions.tolist(), 'travel_yaws_rad': np.unwrap(yaws).tolist(),
            'rest_pose_framing': framing,
            'short_window_walk': window,
            'walkable_map_path': getattr(walkable_map, 'path', None),
            'walkable_map_version': walkable_map.metadata['version'] if walkable_map is not None else None,
            'walkable_map_room_id': walkable_map.metadata['room_id'] if walkable_map is not None else None,
            'walkable_map_bypassed': walkable_map is None,
            'speed_m_s': speed, 'speed_range_m_s': [lo, hi], 'listener_m': listener.tolist(),
            'camera_yaw_deg': camera_yaw, 'species': species, 'template': focus,
            'emission_slot': slot, 'candidate_count': tested,
            'candidate_trials': [{'construction': 'direct_straight_line', 'found': len(found), 'rejected': rejected}],
            'selected_attempt': 'straight-%d-%d' % (int(seed), int(attempt)), 'candidate_native_root_samples': FRAMES,
            'construction': 'direct_straight_line_search',
            'geometry_status': 'native NavMesh proposal only; original pose/contact/collision/readback gates pending'}
