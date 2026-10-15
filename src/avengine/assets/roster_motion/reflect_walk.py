"""Reflecting random walks for v7 moving sources, and emission scheduling on their straight legs (Claude, 2026-10-15).

owner 10-15: "能走的路径相当的多……遇到障碍就反射，相当多走法，反正我觉得不应该一道题都出不来".

Pure geometry: the caller supplies `is_free(xz) -> bool` (walkable, clear of furniture/walls/other bodies by the
caller's own margin, and away from the camera) and the listener position.  A source walks straight at a constant
real speed; when the point a short look-ahead in front stops being free, it turns smoothly (slowing down) towards the
free direction closest to the mirror reflection of its heading.  The straight legs between turns are then used to
schedule emissions so that each motion question family has the conditions its unchanged judge reads:

  QA-06  sounding entirely while moving (any direction)          -> 'moving_emission'
  QA-17  keeps moving, silent, for >= post_s after the wet tail   -> same leg, continues straight after the sound
  QA-15  net radial change >= min_radial_m while sounding,         -> 'radial_emission' (|cos| >= radial_cos)
         monotonic
  QA-16  silent interval after the wet tail with the same          -> 'radial_emission' whose leg continues >= post_s
         monotonic distance trend

Nothing here decides a truth; the judge still reads the rendered per-frame readback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Walk:
    positions: np.ndarray          # (N, 2) xz per frame
    yaws: np.ndarray               # (N,) heading of travel (rad, atan2(dz, dx))
    speeds: np.ndarray             # (N,) m/s
    legs: list = field(default_factory=list)   # [(start_frame, end_frame)] straight constant-speed legs
    fps: int = 30


def _segment_free(is_free, start, end):
    check = getattr(is_free, 'segment_is_free', None)
    return bool(check(start, end)) if check is not None else bool(is_free(end))


def _free_dir(is_free, p, heading, lookahead, n=32):
    """Free headings (rad) at p, i.e. directions whose look-ahead point is free."""
    out = []
    for k in range(n):
        a = heading + 2 * math.pi * k / n
        q = p + lookahead * np.array([math.cos(a), math.sin(a)])
        if _segment_free(is_free, p, q) and is_free(p + 0.5 * lookahead * np.array([math.cos(a), math.sin(a)])):
            out.append(a)
    return out


def _free_run(is_free, p, angle, step=0.1, max_m=6.0):
    """How far one can walk straight from p along angle before leaving the free space."""
    d = np.array([math.cos(angle), math.sin(angle)])
    r = 0.0
    while r + step <= max_m and _segment_free(is_free, p, p + (r + step) * d):
        r += step
    return r


def _pick_heading(is_free, p, frees, mirror, *, angle_weight=0.35):
    """Among free headings prefer the longest straight run, biased towards the mirror reflection (Claude 10-15)."""
    def score(a):
        diff = abs(math.atan2(math.sin(a - mirror), math.cos(a - mirror)))
        return _free_run(is_free, p, a) - angle_weight * diff
    return max(frees, key=score)


def _mirror_heading(is_free, p, heading, lookahead):
    """Approximate the obstacle normal by probing, then mirror the heading about it."""
    blocked = []
    for k in range(-8, 9):
        a = heading + k * math.pi / 16
        q = p + lookahead * np.array([math.cos(a), math.sin(a)])
        if not _segment_free(is_free, p, q):
            blocked.append(a)
    if not blocked:
        return heading
    # normal points away from the mean blocked direction
    mb = math.atan2(np.mean(np.sin(blocked)), np.mean(np.cos(blocked)))
    n = np.array([-math.cos(mb), -math.sin(mb)])
    d = np.array([math.cos(heading), math.sin(heading)])
    r = d - 2 * float(np.dot(d, n)) * n
    return math.atan2(r[1], r[0])


SPECIES_SPEED_M_S = {'human': (0.8, 1.4), 'dog': (0.5, 1.2), 'cat': (0.3, 0.8), 'slide': (0.2, 0.5)}  # owner 10-15


def reflect_walk(is_free, start_xz, heading, speed, *, duration_s=30.0, fps=30, lookahead=0.45,
                 turn_time_s=0.6, turn_speed_factor=0.5, rng=None, pause_prob=0.0, min_speed=None,
                 prefer_long_legs=True):
    """Walk straight at `speed`; turn smoothly to the free heading nearest the mirror direction when blocked.

    Returns None if the start is not free or the walker gets boxed in (no free heading at all).
    Pass min_speed (the species minimum, see SPECIES_SPEED_M_S) so turns never drop below the owner's range.
    """
    rng = rng or np.random.default_rng(0)
    p = np.asarray(start_xz, float)
    if not is_free(p):
        return None
    if prefer_long_legs:
        cands = [heading + k * math.pi / 8 for k in range(16)]
        heading = _pick_heading(is_free, p, cands, heading, angle_weight=0.15)
    n = int(round(duration_s * fps)) + 1
    dt = 1.0 / fps
    pos = np.zeros((n, 2)); yaw = np.zeros(n); spd = np.zeros(n)
    h = float(heading)
    turning = 0
    target = h
    leg_start = 0
    legs = []
    for i in range(n):
        pos[i] = p; yaw[i] = h
        ahead = p + lookahead * np.array([math.cos(h), math.sin(h)])
        if turning == 0 and not _segment_free(is_free, p, ahead):
            frees = _free_dir(is_free, p, h, lookahead)
            if not frees:
                return None
            mirror = _mirror_heading(is_free, p, h, lookahead)
            target = _pick_heading(is_free, p, frees, mirror) if prefer_long_legs else \
                min(frees, key=lambda a: abs(math.atan2(math.sin(a - mirror), math.cos(a - mirror))))
            turning = max(1, int(round(turn_time_s * fps)))
            if i - 1 > leg_start:
                legs.append((leg_start, i - 1))
        if turning:
            d = math.atan2(math.sin(target - h), math.cos(target - h))
            h += d / turning
            turning -= 1
            v = max(speed * turn_speed_factor, min_speed if min_speed is not None else 0.0)
            if turning == 0:
                leg_start = i + 1
        else:
            v = speed
        spd[i] = v
        nxt = p + v * dt * np.array([math.cos(h), math.sin(h)])
        if not _segment_free(is_free, p, nxt):
            # never step into a blocked point: hold position this frame and force a turn next frame
            v = 0.0
            spd[i] = 0.0
            if turning == 0:
                turning = 1
                frees = _free_dir(is_free, p, h, lookahead)
                if not frees:
                    return None
                target = min(frees, key=lambda a: abs(math.atan2(math.sin(a - h), math.cos(a - h))))
                if i - 1 > leg_start:
                    legs.append((leg_start, i - 1))
            continue
        p = nxt
    if n - 1 > leg_start:
        legs.append((leg_start, n - 1))
    return Walk(positions=pos, yaws=yaw, speeds=spd, legs=[l for l in legs if l[1] > l[0]], fps=fps)


def schedule_emissions(walk, listener_xz, *, utterance_s=(1.2, 1.8), post_s=1.2, wet_tail_s=0.5,
                       min_radial_m=0.5, radial_cos=math.cos(math.radians(30)), margin_s=0.15):
    """Emission slots on straight legs, each marked with the motion families it can serve on its own.

    qa06: the whole utterance lies on one straight moving leg.
    qa15: additionally radial (|cos| >= radial_cos), monotonic, |net radial change| >= min_radial_m.
    qa17: the leg continues straight (still moving, silent) for wet_tail_s + post_s after the utterance.
    qa16: qa17 and the distance keeps the same monotonic trend through that silent interval, with a radial change.
    A short leg can still serve QA-06/QA-15 even if it is too short for QA-16/QA-17.
    """
    fps = walk.fps
    L = np.asarray(listener_xz, float)
    slots = []
    for a, b in walk.legs:
        dur = (b - a) / fps
        for u in (utterance_s[1], utterance_s[0]):
            if dur < u + 2 * margin_s:
                continue
            s0 = a + int(round(margin_s * fps))
            s1 = s0 + int(round(u * fps))
            if s1 > b:
                continue
            r = np.linalg.norm(walk.positions[s0:b + 1] - L, axis=1)
            d = walk.positions[min(b, s0 + 1)] - walk.positions[s0]
            to_l = L - walk.positions[s0]
            cosv = abs(float(np.dot(d, to_l) / (np.linalg.norm(d) * np.linalg.norm(to_l) + 1e-9)))
            r_emit = r[:s1 - s0 + 1]
            net = float(r_emit[-1] - r_emit[0])
            mono_emit = bool(np.all(np.diff(r_emit) <= 0.005)) or bool(np.all(np.diff(r_emit) >= -0.005))
            qa15 = cosv >= radial_cos and abs(net) >= min_radial_m and mono_emit
            post_end = s1 + int(round((wet_tail_s + post_s) * fps))
            qa17 = post_end <= b
            qa16 = False
            if qa17:
                r_all = r[:post_end - s0 + 1]
                q = r[s1 - s0 + int(round(wet_tail_s * fps)):post_end - s0 + 1]
                same = (net < 0 and np.all(np.diff(q) <= 0.005)) or (net > 0 and np.all(np.diff(q) >= -0.005))
                qa16 = bool(same) and abs(float(q[-1] - q[0])) >= 0.2 and cosv >= radial_cos
            slots.append({'start_s': s0 / fps, 'end_s': s1 / fps, 'leg': (a, b), 'radial_change_m': net,
                          'branch': 'nearer' if net < 0 else 'farther', 'radial_cos': cosv,
                          'post_until_s': post_end / fps if qa17 else None,
                          'qa06': True, 'qa15': bool(qa15), 'qa16': bool(qa16), 'qa17': bool(qa17),
                          'kind': 'radial_emission' if qa15 else 'moving_emission'})
            break
    return slots


def plan_mover(is_free, listener_xz, speed, *, seeds=64, duration_s=30.0, rng=None, start_candidates=None,
               need_radial=True, species=None, **kw):
    """Try several starts/headings; return (walk, slots) with at least one usable slot (radial if requested)."""
    rng = rng or np.random.default_rng(0)
    best = None
    for k in range(seeds):
        if start_candidates is not None and len(start_candidates):
            start = np.asarray(start_candidates[k % len(start_candidates)], float)
        else:
            return None
        heading = float(rng.uniform(-math.pi, math.pi))
        lo = SPECIES_SPEED_M_S.get(species, (None, None))[0] if species else None
        w = reflect_walk(is_free, start, heading, speed, duration_s=duration_s, rng=rng, min_speed=lo)
        if w is None:
            continue
        slots = schedule_emissions(w, listener_xz, **kw)
        radial = [s for s in slots if s['kind'] == 'radial_emission']
        if slots and (radial or not need_radial):
            return w, slots
        if slots and best is None:
            best = (w, slots)
    return best

def schedule_moving_windows(walk, *, utterance_s, post_s=1.5, wet_tail_s=0.5, lead_s=0.7, step_s=0.5):
    """QA-06/QA-17 slots that may span reflections: the source keeps moving (speed > 0) for the whole utterance and
    for wet_tail_s + post_s after it.  Neither family's judge needs a straight path, only a stable motion state.
    """
    fps = walk.fps
    n = len(walk.positions)
    slots = []
    t = lead_s
    total = float(utterance_s) + wet_tail_s + post_s
    while t + total + lead_s <= (n - 1) / fps:
        i0 = int(round(t * fps)); i1 = int(round((t + total) * fps))
        if np.all(walk.speeds[i0:i1 + 1] > 0):
            slots.append({'start_s': t, 'end_s': t + float(utterance_s), 'leg': None,
                          'window_frames': (max(0, i0 - int(round(lead_s * fps))), min(n - 1, i1 + int(round(lead_s * fps)))),
                          'post_until_s': t + total, 'radial_change_m': 0.0, 'branch': None, 'radial_cos': None,
                          'qa06': True, 'qa15': False, 'qa16': False, 'qa17': True, 'kind': 'moving_emission'})
        t += step_s
    return slots
