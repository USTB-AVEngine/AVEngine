"""Small finite-domain scheduling of measured source activity and silent queries.

No optimizer dependency or random restarts: constrained events are assigned first,
conflicts backtrack only event starts. Exhausted node/time budgets are not no-solution
proofs. Times are represented on the native 48kHz tick clock.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math
import time
from .types import SoundEvent


@dataclass(frozen=True)
class SoundRequirement:
    entity_id: str
    sound_id: str
    sound_class: str
    activity_intervals_s: tuple[tuple[float, float], ...]
    start_domain_s: tuple[float, float]
    allowed_activity_windows_s: tuple[tuple[float, float], ...] = ()
    wet_tail_s: float = .5
    exclusive_group: str | None = None


@dataclass
class ScheduleResult:
    status: str  # solved / unavailable / budget_exhausted
    events: tuple[SoundEvent, ...] = ()
    starts_s: tuple[float, ...] = ()
    reasons: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def schedule_sound_events(requirements, *, duration_s=30., silent_intervals_s=(),
                          start_grid_hz=30, clock_hz=48000, minimum_gap_s=0.,
                          max_nodes=10000, budget_s=5.):
    if (not math.isfinite(duration_s) or duration_s <= 0 or start_grid_hz <= 0 or clock_hz <= 0
            or clock_hz % start_grid_hz or minimum_gap_s < 0 or max_nodes < 1 or budget_s <= 0):
        raise ValueError("invalid sound scheduling clock/budget")
    begun = time.monotonic()
    step, horizon, gap = clock_hz//start_grid_hz, round(duration_s*clock_hz), round(minimum_gap_s*clock_hz)
    def ticks(interval):
        a, b = interval
        if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b:
            raise ValueError("activity and query intervals must be finite nonempty nonnegative intervals")
        return round(a*clock_hz), round(b*clock_hz)
    silence = tuple(ticks(v) for v in silent_intervals_s)
    def conflict(a, b, padding=0):
        return a[0] < b[1]+padding and b[0] < a[1]+padding
    domains, activities = [], []
    for req in requirements:
        if not req.entity_id or not req.sound_id or not req.sound_class or not 0 <= req.wet_tail_s <= .5:
            raise ValueError("sound binding and bounded wet tail are required")
        active = tuple(ticks(v) for v in req.activity_intervals_s)
        if not active or any(a[1] > b[0] for a, b in zip(active[:-1], active[1:])):
            raise ValueError("measured activity must be sorted, disjoint and nonempty")
        domain = req.start_domain_s
        if len(domain) != 2 or not all(math.isfinite(v) for v in domain) or not 0 <= domain[0] <= domain[1]:
            raise ValueError("invalid event start domain")
        allowed = tuple(ticks(v) for v in req.allowed_activity_windows_s)
        tail = round(req.wet_tail_s*clock_hz)
        choices = []
        first = math.ceil(domain[0]*clock_hz/step)*step
        last = math.floor(domain[1]*clock_hz/step)*step
        for start in range(first, last+1, step):
            shifted = tuple((start+a,start+b) for a,b in active)
            wet = tuple((a,b+tail) for a,b in shifted)
            if wet[-1][1] > horizon:
                continue
            if allowed and any(not any(w[0] <= a and b <= w[1] for w in allowed) for a,b in shifted):
                continue
            if any(conflict(w,q) for w in wet for q in silence):
                continue
            choices.append((start, wet))
        if not choices:
            return ScheduleResult("unavailable", reasons=("empty_start_domain:"+req.entity_id,),
                                  diagnostics={"requirement_index":len(domains), "clock_hz":clock_hz})
        activities.append(active);domains.append(choices)
    order = sorted(range(len(requirements)),key=lambda i:(len(domains[i]),i))
    selected = {};nodes=0;exhausted=False
    def search(k):
        nonlocal nodes,exhausted
        if k == len(order):
            return True
        i=order[k];req=requirements[i]
        for start,wet in domains[i]:
            if nodes >= max_nodes or time.monotonic()-begun >= budget_s:
                exhausted=True
                return False
            nodes+=1
            bad=False
            for j,(_,other_wet) in selected.items():
                other=requirements[j]
                exclusive=(req.entity_id==other.entity_id or
                           (req.exclusive_group is not None and req.exclusive_group==other.exclusive_group))
                if exclusive and any(conflict(a,b,gap)for a in wet for b in other_wet):
                    bad=True;break
            if bad:
                continue
            selected[i]=(start,wet)
            if search(k+1):
                return True
            del selected[i]
            if exhausted:
                return False
        return False
    ok=search(0)
    diagnostics={"nodes":nodes,"elapsed_s":time.monotonic()-begun,"clock_hz":clock_hz,
                 "start_grid_hz":start_grid_hz,"domain_sizes":[len(d)for d in domains],
                 "measured_audio_rendered":False}
    if not ok:
        return ScheduleResult("budget_exhausted"if exhausted else"unavailable",
                              reasons=("event_start_conflicts",),diagnostics=diagnostics)
    events=[];starts=[]
    for i,req in enumerate(requirements):
        start=selected[i][0];starts.append(start/clock_hz)
        for a,b in activities[i]:
            events.append(SoundEvent(req.entity_id,req.sound_id,req.sound_class,
                (start+a)/clock_hz,(start+b)/clock_hz,(start+b)/clock_hz+req.wet_tail_s))
    return ScheduleResult("solved",tuple(events),tuple(starts),diagnostics=diagnostics)
