"""CPU continuous-scene algorithms: source-time RIR crossfade and v0.2 QA.

This opt-in research path does not change the existing Episode producer. Full
convolution tails are retained until after scene assembly and window cropping.
Public requests and private simulation answers are returned separately.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

import numpy as np
from scipy import signal

from avengine.qa.azimuth_publication import publish_azimuth_deg

SAMPLE_RATE = 16000
FPS = 15
ANNOTATION_HZ = 10
ACTOR_CLASSES = {0, 1, 2, 4, 6}
DIRECTIONS = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")
QA21_CLASSES = ("air_conditioning", "alarm_beep", "alarm_bell", "alarm_clock", "bathtub_filling_washing", "blender", "busy_signal", "cat_meow", "cellphone_vibration_alert", "chime", "clock_tick", "crackle", "ding_dong", "dog_bark", "doorbell", "doorbell_chime", "drip", "fire", "fire_alarm", "gurgling", "microwave_beep", "microwave_hum", "music_playback", "phone_ring", "printer", "ringtone", "sink_filling_washing", "smoke_alarm", "speech_playback", "telephone", "telephone_bell_ringing", "telephone_dialing_dtmf", "toilet_flush", "water_tap_faucet")


def position_at(trajectory, time_s):
    times = np.asarray([x["time_s"] for x in trajectory], dtype=float)
    points = np.asarray([x["position_m"] for x in trajectory], dtype=float)
    if len(times) < 1 or points.shape != (len(times), 3) or not np.isfinite(points).all():
        raise ValueError("trajectory must contain finite xyz positions")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("trajectory times must increase")
    return np.array([np.interp(time_s, times, points[:, i]) for i in range(3)])


def camera_basis(forward):
    f = np.asarray(forward, float).copy()
    f[1] = 0
    if np.linalg.norm(f) < 1e-12:
        raise ValueError("camera forward has no horizontal component")
    f /= np.linalg.norm(f)
    right = np.cross(f, [0., 1., 0.])
    return f, right, np.array([0., 1., 0.])


def published_bearing(position, listener, forward):
    f, r, _ = camera_basis(forward)
    delta = np.asarray(position) - listener
    return publish_azimuth_deg(math.degrees(math.atan2(float(delta @ r), float(delta @ f))))


def angular_distance(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if np.linalg.norm(a) < 1e-15 or np.linalg.norm(b) < 1e-15:
        raise ValueError("direction has zero magnitude")
    return float(np.degrees(np.arccos(np.clip((a @ b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def plan_rir_keyframes(trajectory, duration_s, listener, *, max_seconds=.25,
                       max_distance_m=.5, max_angle_deg=None):
    """Update at the earlier bound; stopped positions reuse their exact RIR.

    Optional angle subdivision is geometric protection, not a substitute for
    independent native RIR validation. Every motion corner remains a keyframe.
    """
    if duration_s <= 0 or max_seconds <= 0 or max_distance_m <= 0:
        raise ValueError("duration and update bounds must be positive")
    if max_angle_deg is not None and max_angle_deg <= 0:
        raise ValueError("angular bound must be positive")
    last = (round(duration_s * SAMPLE_RATE) - 1) / SAMPLE_RATE
    knots = sorted(set([0., last] + [float(x["time_s"]) for x in trajectory if 0 < x["time_s"] < last]))
    keys = [0.]
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        a, b = position_at(trajectory, left), position_at(trajectory, right)
        distance = float(np.linalg.norm(b - a))
        steps = 1 if distance < 1e-12 else max(1, math.ceil((right-left)/max_seconds), math.ceil(distance/max_distance_m))
        candidates = np.linspace(left, right, steps+1).tolist()
        if max_angle_deg is not None:
            refined = [candidates[0]]
            def visit(t0, t1):
                p0, p1 = position_at(trajectory, t0)-listener, position_at(trajectory, t1)-listener
                if angular_distance(p0, p1) > max_angle_deg and t1-t0 > 1/SAMPLE_RATE:
                    mid = (t0+t1)/2
                    visit(t0, mid); visit(mid, t1)
                else:
                    refined.append(t1)
            for t0, t1 in zip(candidates[:-1], candidates[1:], strict=True):
                visit(t0, t1)
            candidates = refined
        keys.extend(candidates[1:])
    samples = sorted(set(int(round(t * SAMPLE_RATE)) for t in keys))
    return [{"sample": k, "time_s": k/SAMPLE_RATE,
             "position_m": position_at(trajectory, k/SAMPLE_RATE).tolist()} for k in samples]


def partitioned_convolve(dry, filters, key_samples, *, fade="linear"):
    """Stream the v6 source-time partition without a K x duration allocation.

    Each weighted dry support is linearly convolved with the complete RIR;
    overlap-add happens on a full-tail output. Earlier emissions therefore
    continue after a stop and across window boundaries. Static filters use one
    convolution. Linear and v6 raised-cosine weights both sum to one.
    """
    x = np.asarray(dry, dtype=float)
    keys = np.asarray(key_samples, dtype=np.int64)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("dry must be a finite nonempty mono bus")
    if len(keys) != len(filters) or keys[0] != 0 or np.any(np.diff(keys) <= 0) or keys[-1] >= len(x):
        raise ValueError("one filter per increasing key, starting at zero")
    if fade not in {"linear", "raised_cosine"}:
        raise ValueError("unsupported crossfade")
    hs = [np.asarray(h, dtype=float) for h in filters]
    if any(h.ndim != 2 or h.shape[0] != 4 or h.shape[1] < 1 or not np.isfinite(h).all() for h in hs):
        raise ValueError("filters must be finite ACN W,Y,Z,X")
    full = np.zeros((4, len(x)+max(h.shape[1] for h in hs)-1), dtype=float)
    for i, h in enumerate(hs):
        lo = int(keys[i-1]) if i else 0
        hi = int(keys[i+1])+1 if i+1 < len(keys) else len(x)
        indices = np.arange(lo, hi)
        weight = np.ones(hi-lo)
        if i:
            mask = indices <= keys[i]
            u = (indices[mask]-keys[i-1])/(keys[i]-keys[i-1])
            weight[mask] = u if fade == "linear" else .5*(1-np.cos(np.pi*u))
        if i+1 < len(keys):
            mask = indices >= keys[i]
            u = (indices[mask]-keys[i])/(keys[i+1]-keys[i])
            weight[mask] = 1-u if fade == "linear" else .5*(1+np.cos(np.pi*u))
        weighted = x[lo:hi]*weight
        nonzero = np.flatnonzero(weighted != 0)
        if not len(nonzero):
            continue
        begin, end = int(nonzero[0]), int(nonzero[-1])+1
        for c in range(4):
            y = signal.fftconvolve(weighted[begin:end], h[c], mode="full")
            full[c, lo+begin:lo+begin+len(y)] += y
    return full


def interpolated_filter(filters, key_samples, sample, *, fade="linear"):
    if len(filters) == 1 or sample <= key_samples[0]:
        return np.asarray(filters[0], float)
    if sample >= key_samples[-1]:
        return np.asarray(filters[-1], float)
    i = int(np.searchsorted(key_samples, sample, side="right"))-1
    u = (sample-key_samples[i])/(key_samples[i+1]-key_samples[i])
    if fade == "raised_cosine":
        u = .5*(1-math.cos(math.pi*u))
    elif fade != "linear":
        raise ValueError("unsupported crossfade")
    a, b = np.asarray(filters[i], float), np.asarray(filters[i+1], float)
    n = max(a.shape[1], b.shape[1])
    out = np.zeros((4, n))
    out[:, :a.shape[1]] += (1-u)*a
    out[:, :b.shape[1]] += u*b
    return out


def raw_world_to_camera_sn3d(samples, forward):
    """Convert raw v6 RLR ACN N3D world to camera ACN SN3D once."""
    a = np.asarray(samples, float)
    if a.ndim != 2 or a.shape[0] != 4:
        raise ValueError("expected raw ACN W,Y,Z,X")
    f, r, up = camera_basis(forward)
    xyz = a[[3, 1, 2]] / math.sqrt(3)
    camera_xyz = np.stack((f, -r, up)) @ xyz
    return np.stack((a[0], camera_xyz[1], camera_xyz[2], camera_xyz[0]))


def activity_segments(events):
    """Full-scene 100 ms activity, joining adjacent frames only."""
    frames: dict[tuple[int, int], set[int]] = {}
    identity: dict[tuple[int, int], dict] = {}
    for event in events:
        lo = event["start_tick"]; hi = event["end_tick"]
        if not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi <= lo:
            raise ValueError("activity bounds must be increasing 100 ms integer ticks")
        key = (int(event["class_id"]), int(event["source_id"]))
        target = frames.setdefault(key, set())
        if target.intersection(range(lo, hi)):
            raise ValueError("duplicate same-class same-source activity")
        target.update(range(lo, hi)); identity[key] = event
    result = []
    for (cls, source), active in frames.items():
        ordered = sorted(active); start = previous = ordered[0]
        for tick in ordered[1:] + [ordered[-1]+2]:
            if tick != previous+1:
                result.append({"class_id": cls, "source_id": source,
                               "start_tick": start, "end_tick": previous+1,
                               "endpoint_id": identity[(cls, source)]["endpoint_id"]})
                start = tick
            previous = tick
    return sorted(result, key=lambda e: (e["start_tick"], e["class_id"], e["source_id"]))


def window_segments(segments, start_s, duration_s=10.):
    lo, hi = round(start_s*10), round((start_s+duration_s)*10)
    if abs(lo/10-start_s) > 1e-9:
        raise ValueError("window start must be on the 100 ms grid")
    clipped = []
    for e in segments:
        if e["start_tick"] < hi and e["end_tick"] > lo:
            clipped.append({**e, "anchor_tick": max(e["start_tick"], lo)-lo,
                            "window_end_tick": min(e["end_tick"], hi)-lo,
                            "carry_in": e["start_tick"] < lo})
    return sorted(clipped, key=lambda e: (e["anchor_tick"], e["class_id"], e["source_id"]))


def window_coverage(scene, start_s):
    segments = window_segments(activity_segments(scene["events"]), start_s)
    actor_events = Counter(e["source_id"] for e in segments if e["class_id"] in {0, 1})
    active = np.zeros(100, int)
    for e in segments:
        active[e["anchor_tick"]:e["window_end_tick"]] += 1
    moving = False
    stop_resume = False
    for e in segments:
        track = scene["sources"][e["endpoint_id"]]["trajectory"]
        moving |= any(np.linalg.norm(position_at(track, start_s+t/10+.1)-position_at(track, start_s+t/10)) > 1e-5
                      for t in range(e["anchor_tick"], e["window_end_tick"]))
        for a, b in zip(track[:-1], track[1:], strict=True):
            if b["time_s"]-a["time_s"] >= .5 and np.linalg.norm(np.array(a["position_m"])-b["position_m"]) < 1e-9:
                stop_resume |= start_s < b["time_s"] < start_s+10 and any(
                    q["source_id"] == e["source_id"] and q["start_tick"] >= b["time_s"]*10 for q in segments)
    return {"activity_segments": len(segments), "carry_in": any(e["carry_in"] for e in segments),
            "continuous_ge4s": any((e["window_end_tick"]-e["anchor_tick"])/10 >= 4 for e in segments),
            "same_actor_repeated": max(actor_events.values(), default=0) >= 2,
            "moving_while_active": bool(moving), "stop_then_voice_resume": bool(stop_resume),
            "two_source_overlap_ge05s": np.count_nonzero(active >= 2) >= 5,
            "overlap_seconds": float(np.count_nonzero(active >= 2)/10),
            "more_than6": len(segments) > 6}


def generate_v02_questions(scene, start_s, window_id, media):
    """The v0.2 pilot panel, with fixed public domains and no gold in requests."""
    segments = window_segments(activity_segments(scene["events"]), start_s)
    public, private, excluded = [], [], []
    anchors = Counter(e["anchor_tick"] for e in segments)
    listener = np.asarray(scene["listener_m"])
    f, r, up = camera_basis(scene["camera_forward"])

    def emit(qa_id, suffix, question, value, domain, mcq=True):
        qid = f"v7pilot__{window_id}__{qa_id}__{suffix}"
        forms = ("open", "mcq") if mcq else ("open",)
        for form in forms:
            item_id = qid+"__"+form
            row = {"question_id": item_id, "qa_id": qa_id, "form": form,
                   "question_zh": question, "media": dict(media), "answer_domain": domain}
            answer = str(value)
            if form == "mcq":
                order = sorted(domain, key=lambda v: hashlib.sha256((item_id+"|"+v).encode()).digest())
                row["choices"] = [{"id": str(i+1), "text": v} for i, v in enumerate(order)]
                answer = str(order.index(str(value))+1)
            public.append(row)
            private.append({"question_id": item_id, "canonical_answer": str(value), "answer": answer})

    reliable = all(e["class_id"] in ACTOR_CLASSES and e["source_id"] > 0 for e in segments)
    if reliable:
        n = len(segments)
        emit("QA-23", "count", "与本片段相交的100毫秒标注活动连续段有多少个？片头已在响的段计一次，缺一帧即另段；不是物理声源数或动作次数。", n,
             [str(i) for i in range(21)] if n <= 20 else "nonnegative_integer", n <= 20)
    else:
        excluded.append({"qa_id": "QA-23", "reason": "non_actor_or_unreliable_source_identity"})
    speech = [e for e in segments if e["class_id"] in {0, 1}]
    conflict = any(a["source_id"] == b["source_id"] and a["class_id"] != b["class_id"]
                   and max(a["start_tick"], b["start_tick"]) < min(a["end_tick"], b["end_tick"])
                   for i, a in enumerate(speech) for b in speech[i+1:])
    if all(e["source_id"] > 0 for e in speech) and not conflict:
        n = len({e["source_id"] for e in speech})
        emit("STARSS-SPEAKERS", "count", "本片段任意时刻出声的不同说话人有多少位？同一人重复讲话只计一位，片头已在讲话者计入。", n,
             [str(i) for i in range(9)] if n <= 8 else "nonnegative_integer", n <= 8)
    else:
        excluded.append({"qa_id": "STARSS-SPEAKERS", "reason": "speech_identity_conflict"})
    for rank, e in enumerate(segments[:3]):
        if anchors[e["anchor_tick"]] != 1 or e["source_id"] <= 0:
            excluded.append({"candidate": rank+1, "reason": "tied_anchor_or_unreliable_source"}); continue
        anchor = e["anchor_tick"]/10
        description = "片段开始时已经在响的那个声音" if anchor == 0 else f"大约第{anchor:.1f}秒开始的那个声音"
        track = scene["sources"][e["endpoint_id"]]["trajectory"]
        point = position_at(track, start_s+anchor)
        angle = published_bearing(point, listener, f)
        sector = int(math.floor((angle+22.5)/45)) % 8
        boundary_distance = abs(((angle+22.5) % 45)-22.5)
        if boundary_distance <= 17.5:
            emit("QA-04", f"target{rank+1}", f"{description}在起始活动帧的水平方向是什么？以相机朝向为正前。", DIRECTIONS[sector], list(DIRECTIONS))
        else:
            excluded.append({"qa_id": "QA-04", "candidate": rank+1, "reason": "within5deg_sector_boundary"})
        query = math.floor(anchor)+1
        if query*10 < e["window_end_tick"]:
            value = published_bearing(position_at(track, start_s+query), listener, f)
            integer = (int(math.floor(value+.5))+180) % 360-180
            emit("QA-25", f"target{rank+1}", f"{description}在第{query}秒的水平角度是多少？正前0°，左正右负，输出[-180,179]整数。", integer, "integer[-180,179]", False)
        else:
            excluded.append({"qa_id": "QA-25", "candidate": rank+1, "reason": "segment_ends_before_next_integer_second"})
        delta = point-listener
        z = float(delta @ f)
        focal = 1280/(2*math.tan(math.radians(85/2)))
        px = 640+focal*float(delta @ r)/z if z > 0 else math.inf
        py = 360-focal*float(delta @ up)/z if z > 0 else math.inf
        value = "in_frame" if z > 0 and 0 <= px < 1280 and 0 <= py < 720 else "out_of_frame"
        emit("STARSS-FRUSTUM", f"target{rank+1}", f"{description}在起始活动帧的方向是否位于画面几何视锥内？不问遮挡。只输出in_frame或out_of_frame。", value, ["in_frame", "out_of_frame"])
        mapped = {8: "music_playback", 10: "water_tap_faucet"}.get(e["class_id"])
        if mapped:
            emit("QA-21", f"target{rank+1}", f"{description}属于哪个声音类别？", mapped, list(QA21_CLASSES))
        else:
            excluded.append({"qa_id": "QA-21", "candidate": rank+1, "reason": "no_strict_v6_class_mapping"})
    from avengine.qa.answer_format import normalize_export
    public, private = normalize_export(public, private)
    return public, private, {"segments": segments, "exclusions": excluded,
                             "coverage": window_coverage(scene, start_s), "protocol": "STARSS23_v0.2"}
