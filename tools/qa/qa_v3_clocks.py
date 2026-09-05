#!/usr/bin/env python3
"""Derive QA-v3 event and binding clocks from params.

Binding frames, event starts and last-frame selectors follow the live
CLIP_SECONDS / VIDEO_FPS / FRAME_COUNT / SAMPLE_RATE_HZ values.  A 75-frame
reference is only used when a profile writes absolute frames against an
explicit clock_reference.  Callers that omit params keep the historical
5 s / 15 fps / 16 kHz numbers so existing unit tests stay exact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


REFERENCE_FRAME_COUNT = 75
REFERENCE_VIDEO_FPS = 15.0
REFERENCE_SAMPLE_RATE_HZ = 16000
REFERENCE_CLIP_SECONDS = 5.0
DEFAULT_EVENT_FRACTIONS = (0.1, 0.3, 0.5, 0.7)
CARD11_REFERENCE_BINDING_FRAME = 30
CARD11_REFERENCE_EVENT_START_SAMPLE = 30000
CARD11_REFERENCE_EVENT_DURATION_SAMPLES = 4800


class ClockError(ValueError):
    """A clock field cannot be turned into a live sample or frame index."""


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClockError(f"{owner} must be a positive integer")
    return value


def _positive_number(value: Any, *, owner: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ClockError(f"{owner} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ClockError(f"{owner} must be a finite positive number")
    return number


def frame_count(params: Mapping[str, Any] | None, *, default: int = REFERENCE_FRAME_COUNT) -> int:
    if not params or params.get("FRAME_COUNT") is None:
        return default
    return _positive_int(params["FRAME_COUNT"], owner="FRAME_COUNT")


def video_fps(params: Mapping[str, Any] | None, *, default: float = REFERENCE_VIDEO_FPS) -> float:
    if not params or params.get("VIDEO_FPS") is None:
        return default
    return _positive_number(params["VIDEO_FPS"], owner="VIDEO_FPS")


def sample_rate_hz(
    params: Mapping[str, Any] | None, *, default: int = REFERENCE_SAMPLE_RATE_HZ
) -> int:
    if not params or params.get("SAMPLE_RATE_HZ") is None:
        return default
    return _positive_int(params["SAMPLE_RATE_HZ"], owner="SAMPLE_RATE_HZ")


def clip_seconds(
    params: Mapping[str, Any] | None, *, default: float = REFERENCE_CLIP_SECONDS
) -> float:
    if not params or params.get("CLIP_SECONDS") is None:
        count = frame_count(params, default=REFERENCE_FRAME_COUNT)
        fps = video_fps(params, default=REFERENCE_VIDEO_FPS)
        return count / fps if params else default
    return _positive_number(params["CLIP_SECONDS"], owner="CLIP_SECONDS")


def last_frame_index(params: Mapping[str, Any] | None, *, default: int = 74) -> int:
    if not params or params.get("FRAME_COUNT") is None:
        return default
    return frame_count(params) - 1


def event_start_samples(
    params: Mapping[str, Any] | None,
    *,
    count: int = 4,
    fractions: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Sequential event starts as integer samples on the live audio clock.

    Default fractions reproduce 8000/24000/40000/56000 on a 5 s / 16 kHz clip.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ClockError("event start count must be a positive integer")
    if params and params.get("EVENT_START_FRACTIONS") is not None:
        fractions = params["EVENT_START_FRACTIONS"]
    if fractions is None:
        if count <= len(DEFAULT_EVENT_FRACTIONS):
            fractions = DEFAULT_EVENT_FRACTIONS[:count]
        else:
            fractions = tuple((index + 0.5) / count for index in range(count))
    if not isinstance(fractions, Sequence) or isinstance(fractions, (str, bytes)):
        raise ClockError("EVENT_START_FRACTIONS must be a list of fractions")
    if len(fractions) < count:
        raise ClockError("EVENT_START_FRACTIONS is shorter than the event count")
    rate = sample_rate_hz(params)
    duration = clip_seconds(params)
    starts: list[int] = []
    for index, raw in enumerate(fractions[:count]):
        fraction = _positive_number(raw, owner=f"EVENT_START_FRACTIONS[{index}]")
        if fraction >= 1.0:
            raise ClockError(f"EVENT_START_FRACTIONS[{index}] must be inside the clip")
        starts.append(int(round(fraction * duration * rate)))
    if len(set(starts)) != len(starts):
        raise ClockError("event starts must be unique after rounding onto the sample clock")
    return tuple(starts)


def _reference_frame_count(profile: Mapping[str, Any] | None) -> int | None:
    if not profile:
        return None
    raw = profile.get("clock_reference")
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("frame_count")
    if value is None:
        return None
    return _positive_int(value, owner="clock_reference.frame_count")


def scaled_binding_frames(
    profile: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
    *,
    default: Sequence[int] = (12, 40),
) -> tuple[int, ...]:
    """Return live binding frames, scaling written frames when the clock changes."""

    source = default
    if profile and profile.get("binding_frames") is not None:
        source = profile["binding_frames"]
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ClockError("binding_frames must be a list of integers")
    live = frame_count(params)
    reference = _reference_frame_count(profile)
    if reference is None:
        reference = REFERENCE_FRAME_COUNT if params and params.get("FRAME_COUNT") is not None else live
    frames: list[int] = []
    for index, raw in enumerate(source):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ClockError(f"binding_frames[{index}] must be an integer")
        if raw < 0:
            frame = live + raw if raw == -1 else live + raw
        elif reference == live:
            frame = raw
        else:
            frame = int(round(raw * (live - 1) / (reference - 1)))
        if frame < 0 or frame >= live:
            raise ClockError(
                f"binding frame {frame} is outside FRAME_COUNT={live}"
            )
        frames.append(frame)
    return tuple(frames)


def card11_binding_frame(
    profile: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> int:
    frames = scaled_binding_frames(
        profile or {"binding_frames": [CARD11_REFERENCE_BINDING_FRAME]},
        params,
        default=(CARD11_REFERENCE_BINDING_FRAME,),
    )
    return frames[0]


def card11_event_start_sample(
    params: Mapping[str, Any] | None = None,
    *,
    binding_frame: int | None = None,
    duration_samples: int = CARD11_REFERENCE_EVENT_DURATION_SAMPLES,
) -> int:
    """Place a short event so it still covers the live binding frame."""

    if params is None and binding_frame in (None, CARD11_REFERENCE_BINDING_FRAME):
        return CARD11_REFERENCE_EVENT_START_SAMPLE
    if isinstance(duration_samples, bool) or not isinstance(duration_samples, int) or duration_samples <= 0:
        raise ClockError("card11 event duration must be a positive integer")
    frame = CARD11_REFERENCE_BINDING_FRAME if binding_frame is None else binding_frame
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ClockError("card11 binding frame must be a non-negative integer")
    rate = sample_rate_hz(params)
    fps = video_fps(params)
    samples_per_frame = rate / fps
    start = int(frame * samples_per_frame - duration_samples / 4)
    sample_count = int(round(clip_seconds(params) * rate))
    start = max(0, min(start, sample_count - duration_samples))
    first_frame = int(start // samples_per_frame)
    last_frame_exclusive = int(-(-(start + duration_samples) // samples_per_frame))
    if not first_frame <= frame < last_frame_exclusive:
        start = int(frame * samples_per_frame)
        start = max(0, min(start, sample_count - duration_samples))
    return start
