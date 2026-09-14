"""Count how many independent sound events one dry recording contains.

QA-23 asks how many independent sound events begin during a clip, and its
truth is the number of playback events the AudioProgram schedules.  That
answer is only correct if every scheduled recording is itself *one* sounding
event: a recording that holds two barks a second and a half apart makes the
scheduled count an undercount, and nobody can tell from the program alone.
The catalog therefore refuses to ask QA-23 until every non-speech event
carries a segmentation record, and until 2026-09-14 no pool wrote one.

This module measures that record from the audio, through the detector the
sound-segment stage already uses, and states plainly which recordings it can
certify and which it cannot.

Three things are worth saying about the method up front.

*One gate for every class.*  The detector policy here is forced to a single
family for every recording: full band, a peak-relative gate 25 dB under the
clip's own loudest window, the 20 ms / 10 ms analysis grid, and the 40 ms
dip-merge that the rest of the project already uses.  Nothing about the gate
is chosen per class, so "two sounding stretches 1.2 s apart" means the same
thing for a dog bark, a kettle and a doorbell.  The declared class travels
with the record as a label only.

*One threshold, and it is policy.*  Runs separated by no more than
``gap_max_s`` are the same event; a longer silence starts a new one.  The
number is not derivable from audio - it is where we decide a listener stops
hearing continuation and starts hearing repetition - so it is an explicit
parameter, recorded on every record, and the census that chose the default is
in the report beside this change.

*A recording that does not certify stays uncertified.*  There is no per-class
exception and no fallback that turns a two-event recording into a pass.  An
episode that plays such a recording keeps deferring QA-23, which is the
honest outcome, not a regression.

The certification is automatic: it is an acoustic measurement, not a person
listening.  Every record says so, in ``certification`` and in
``human_review``, so a later human pass can overwrite it without guessing
what the machine claimed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from avengine.assets.sound_events import MIN_EVENT_S
from avengine.dataset.sound_segments import (
    SoundSegmentError,
    measure_activity,
    pause_structure,
    read_source_pcm,
    segment_policy_for_class,
)

SCHEMA = "avengine_event_segmentation_v1"
SIDECAR_SCHEMA = "avengine_event_segmentation_sidecar_v1"
BACKFILL_SCHEMA = "avengine_event_segmentation_backfill_v1"
METHOD = "uniform_peak_relative_window_rms_onset_grouping_v1"
CERTIFICATION = "automatic"
CERTIFIED_BY = "acoustic_onset_grouping_not_human_listened"

# The family whose policy is full band and peak-relative, i.e. the
# "never cut the middle" treatment sound_segments applies to a class it does
# not recognise.  Forcing it for every class is what makes the measurement
# comparable across classes instead of meaning something different in each.
UNIFORM_ACTIVITY_FAMILY = "unknown"

# Two sounding stretches this far apart or closer are one event.
#
# Chosen from the census of the 564 non-speech recordings in the 2026-09-10
# pool (/data/jzy/tmp/claude_fix_20260914/qa23/probes/segmentation_census.json).
# Their longest internal silence is not spread evenly: 314 of 564 are under
# 0.1 s and the tail thins out quickly past a second.  Read as a curve, each
# extra 0.25 s of threshold certifies 21 to 37 more recordings while the
# threshold is under 1 s, and only 6 to 8 more after it - so 1 s is where the
# number stops buying much.  Certified counts are 477 at 0.5 s, 527 at 1.0 s,
# 541 at 1.5 s and 553 at 2.0 s; the report carries the class breakdown, and
# no class is emptied at any of those values.
DEFAULT_GAP_MAX_S = 1.0

# Three onsets is the fewest that can show a spacing is repeated rather than
# coincidental; the regularity test itself is pause_structure's, unchanged.
MIN_PULSE_TRAIN_ONSETS = 3

STATUS_PASS = "pass"
STATUS_MULTI = "multi_event_candidate"
STATUS_SILENT = "no_measurable_onset"

RULE_SINGLE_RUN = "single_sounding_run"
RULE_GAPS_WITHIN_MAX = "all_internal_silences_within_gap_max"
RULE_PULSE_TRAIN = "regular_pulse_train_is_one_repeating_sound"
RULE_GAP_EXCEEDS_MAX = "internal_silence_exceeds_gap_max"
RULE_NO_ONSET = "no_window_above_the_gate"

CLAIM_BOUNDARY = (
    "Event count measured from the dry recording by an acoustic onset "
    "grouping. No person listened to it, and it describes the recording, not "
    "the rendered mixture."
)

# The classes QA-23 exempts from the segmentation requirement, plus the
# presence of a transcript.  Kept here so a backfill does not stamp an
# event-count claim onto material the question never counts; a unit test pins
# it against the catalog's own test.
SPEECH_CLASSES = frozenset(
    {"speech", "speech_playback", "utterance", "human_speech"}
)


class EventSegmentationError(ValueError):
    """Raised when a recording or a pool cannot be segmented as asked."""


def is_speech_like(row: Mapping[str, Any]) -> bool:
    """Whether QA-23 would treat this row as speech and skip the requirement."""

    if not isinstance(row, Mapping):
        return False
    transcript = row.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        return True
    if transcript not in (None, "") and bool(transcript):
        return True
    sound_class = str(row.get("sound_class") or "").casefold()
    return sound_class in SPEECH_CLASSES


def uniform_policy(
    sound_class: str | None = None,
    *,
    relative_peak_db: float | None = None,
    window_s: float | None = None,
    hop_s: float | None = None,
    gap_merge_s: float | None = None,
) -> dict[str, Any]:
    """The one detector policy every recording is measured through.

    Built from ``segment_policy_for_class`` so the window, hop, dip merge and
    gate definition stay the ones the sound-segment stage already uses, with
    the family forced and the one remaining per-class number
    (``min_active_duration_s``) pinned to the project-wide minimum.  The only
    field that still differs between two calls is the class label.
    """

    overrides: dict[str, Any] = {
        "activity_family": UNIFORM_ACTIVITY_FAMILY,
        "min_active_duration_s": float(MIN_EVENT_S),
    }
    if relative_peak_db is not None:
        overrides["relative_peak_db"] = float(relative_peak_db)
    if window_s is not None:
        overrides["window_s"] = float(window_s)
    if hop_s is not None:
        overrides["hop_s"] = float(hop_s)
    if gap_merge_s is not None:
        overrides["silence_run_merge_s"] = float(gap_merge_s)
    policy = segment_policy_for_class(sound_class, overrides=overrides)
    # segment_policy_for_class labels the source of min_active_duration_s
    # before it applies overrides, so the label would otherwise credit the
    # per-class activity profile for a number this function pinned.
    policy["min_active_duration_source"] = "uniform_event_segmentation_min_event_s"
    policy["uniform_policy"] = METHOD
    return policy


def _group_at(
    intervals: Sequence[Sequence[int]], gap_samples: int
) -> list[list[int]]:
    """Join sounding stretches whose silence between them is short enough."""

    grouped: list[list[int]] = []
    for start, end in intervals:
        if grouped and int(start) - grouped[-1][1] <= gap_samples:
            grouped[-1][1] = max(grouped[-1][1], int(end))
        else:
            grouped.append([int(start), int(end)])
    return grouped


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance) / mean)


def classify_activity(
    activity: Mapping[str, Any],
    *,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
    min_pulse_train_onsets: int = MIN_PULSE_TRAIN_ONSETS,
) -> dict[str, Any]:
    """Turn one activity measurement into an event-count verdict.

    The measurement is grouped at ``gap_max_s`` first, and everything after
    that - the count, the regularity test, the reported temporal form - is
    computed on the groups, so the verdict and the number it reports describe
    the same objects.
    """

    rate = int(activity["source_rate_hz"])
    length = int(activity["source_sample_count"])
    intervals = [list(pair) for pair in activity["source_activity_intervals_samples"]]
    gap_samples = int(round(float(gap_max_s) * rate))
    groups = _group_at(intervals, gap_samples)
    structure = pause_structure(groups, 0, length, rate) if groups else None
    onsets = [group[0] / rate for group in groups]
    periods = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    period_cv = _coefficient_of_variation(periods)
    regular = (
        period_cv is not None
        and len(groups) >= int(min_pulse_train_onsets)
        and period_cv < 0.25
    )
    silences = [
        (groups[i + 1][0] - groups[i][1]) / rate for i in range(len(groups) - 1)
    ]

    if not groups:
        status, rule, event_count = STATUS_SILENT, RULE_NO_ONSET, 0
    elif len(groups) == 1:
        run_count = len(intervals)
        rule = RULE_SINGLE_RUN if run_count == 1 else RULE_GAPS_WITHIN_MAX
        status, event_count = STATUS_PASS, 1
    elif regular:
        status, rule, event_count = STATUS_PASS, RULE_PULSE_TRAIN, 1
    else:
        status, rule, event_count = STATUS_MULTI, RULE_GAP_EXCEEDS_MAX, len(groups)

    return {
        "status": status,
        "rule": rule,
        "event_count": event_count,
        "measured_group_count": len(groups),
        "sounding_run_count": len(intervals),
        "group_onsets_s": [round(value, 4) for value in onsets],
        "group_silences_s": [round(value, 4) for value in silences],
        "max_group_silence_s": float(max(silences)) if silences else 0.0,
        "group_onset_period_s": [round(value, 4) for value in periods],
        "group_onset_period_coefficient_of_variation": period_cv,
        "group_onset_spacing_is_regular": bool(regular),
        "grouped_temporal_form": (
            structure["temporal_form"] if structure else "silent"
        ),
        "grouped_intervals_samples": groups,
    }


def segmentation_record(
    activity: Mapping[str, Any],
    *,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
    min_pulse_train_onsets: int = MIN_PULSE_TRAIN_ONSETS,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The record a QA event carries, from one activity measurement."""

    verdict = classify_activity(
        activity,
        gap_max_s=gap_max_s,
        min_pulse_train_onsets=min_pulse_train_onsets,
    )
    certified = verdict["status"] == STATUS_PASS
    policy = activity.get("policy") or {}
    gate = activity.get("gate") or {}
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": verdict["status"],
        "event_count": verdict["event_count"],
        "certification": CERTIFICATION if certified else None,
        "certified_by": CERTIFIED_BY if certified else None,
        "human_certified": False,
        "human_review": {"status": "not_reviewed", "reviewer": None, "count": 0},
        "method": METHOD,
        "rule": verdict["rule"],
        "parameters": {
            "gap_max_s": float(gap_max_s),
            "min_pulse_train_onsets": int(min_pulse_train_onsets),
            "pulse_train_period_cv_max": 0.25,
            "relative_peak_db": policy.get("relative_peak_db"),
            "gate": policy.get("gate"),
            "band": policy.get("band"),
            "window_s": policy.get("window_s"),
            "hop_s": policy.get("hop_s"),
            "silence_run_merge_s": policy.get("silence_run_merge_s"),
            "absolute_active_floor_dbfs": policy.get("absolute_active_floor_dbfs"),
            "activity_family": policy.get("activity_family"),
            "activity_family_source": policy.get("activity_family_source"),
            "declared_sound_class": policy.get("sound_class"),
            "policy_is_uniform_across_classes": True,
        },
        "measurement": {
            "source_rate_hz": activity.get("source_rate_hz"),
            "source_sample_count": activity.get("source_sample_count"),
            "source_duration_s": activity.get("source_duration_s"),
            "sounding_run_count": verdict["sounding_run_count"],
            "measured_group_count": verdict["measured_group_count"],
            "group_onsets_s": verdict["group_onsets_s"],
            "group_silences_s": verdict["group_silences_s"],
            "max_group_silence_s": verdict["max_group_silence_s"],
            "max_internal_silence_s": activity.get("max_internal_silence_s"),
            "group_onset_period_s": verdict["group_onset_period_s"],
            "group_onset_period_coefficient_of_variation": verdict[
                "group_onset_period_coefficient_of_variation"
            ],
            "group_onset_spacing_is_regular": verdict["group_onset_spacing_is_regular"],
            "grouped_temporal_form": verdict["grouped_temporal_form"],
            "measured_temporal_form": (activity.get("structure") or {}).get(
                "temporal_form"
            ),
            "activity_coverage": activity.get("activity_coverage"),
            "active_duration_s": activity.get("active_duration_s"),
            "leading_inactive_s": activity.get("leading_inactive_s"),
            "trailing_inactive_s": activity.get("trailing_inactive_s"),
            "whole_clip_peak_dbfs": activity.get("whole_clip_peak_dbfs"),
            "whole_clip_rms_dbfs": activity.get("whole_clip_rms_dbfs"),
            "gate_threshold_dbfs": gate.get("threshold_dbfs"),
            "gate_peak_frame_dbfs": gate.get("peak_frame_dbfs"),
            "gate_contrast_db": gate.get("gate_contrast_db"),
            "grouped_intervals_samples": verdict["grouped_intervals_samples"],
        },
        # The whole point of the record: unlike the detector extents already in
        # the pool, this one *is* an event-count claim.
        "activity_is_qa_event_count": True,
        "crop_invariant": bool(certified),
        "crop_invariance_note": (
            "A recording certified as one event stays one event under any "
            "sub-crop: a crop cannot introduce an onset the whole recording "
            "does not have."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if source is not None:
        record["source"] = dict(source)
    return record


def segment_recording(
    samples: np.ndarray | Sequence[float],
    rate: int,
    *,
    sound_class: str | None = None,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
    min_pulse_train_onsets: int = MIN_PULSE_TRAIN_ONSETS,
    policy: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one mono recording and return its segmentation record."""

    active_policy = dict(policy) if policy is not None else uniform_policy(sound_class)
    activity = measure_activity(samples, rate, policy=active_policy)
    return segmentation_record(
        activity,
        gap_max_s=gap_max_s,
        min_pulse_train_onsets=min_pulse_train_onsets,
        source=source,
    )


def segment_audio_file(
    path: str | Path,
    *,
    sound_class: str | None = None,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
    min_pulse_train_onsets: int = MIN_PULSE_TRAIN_ONSETS,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one WAV read-only and return its segmentation record."""

    audio_path = Path(path).expanduser()
    samples, rate = read_source_pcm(audio_path)
    return segment_recording(
        samples,
        rate,
        sound_class=sound_class,
        gap_max_s=gap_max_s,
        min_pulse_train_onsets=min_pulse_train_onsets,
        policy=policy,
        source={"path": str(audio_path.resolve()), "read": "read_only"},
    )


def _pool_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        rows = payload.get("sounds")
        if rows is None:
            rows = payload.get("clips")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise EventSegmentationError(
            "sound pool must hold a list under 'sounds' (or 'clips')"
        )
    return rows


def backfill_pool(
    payload: Mapping[str, Any],
    *,
    pool_path: str | Path | None = None,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
    min_pulse_train_onsets: int = MIN_PULSE_TRAIN_ONSETS,
    include_speech: bool = False,
    overwrite_existing: bool = False,
    policy_overrides: Mapping[str, Any] | None = None,
    on_row: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure every eligible row and return (new pool payload, sidecar).

    The input payload is never modified: the returned pool is a deep copy with
    one ``event_segmentation`` field added per measured row and one
    ``event_segmentation_backfill`` block describing how they were made.
    """

    overrides = dict(policy_overrides or {})
    rows = _pool_rows(payload)
    out_payload = deepcopy(dict(payload)) if isinstance(payload, Mapping) else None
    out_rows = _pool_rows(out_payload) if out_payload is not None else None
    root = Path(pool_path).resolve().parent if pool_path else None

    by_id: dict[str, Any] = {}
    counts = {
        "rows": len(rows),
        "measured": 0,
        "skipped_speech": 0,
        "skipped_existing": 0,
        "failed": 0,
    }
    by_status: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EventSegmentationError("sound pool rows must be objects")
        sound_id = row.get("sound_asset_id") or row.get("prepared_audio_id")
        if not isinstance(sound_id, str) or not sound_id:
            raise EventSegmentationError(
                f"pool row {index} has no readable sound_asset_id"
            )
        if not include_speech and is_speech_like(row):
            counts["skipped_speech"] += 1
            continue
        if not overwrite_existing and row.get("event_segmentation") is not None:
            counts["skipped_existing"] += 1
            continue
        raw_path = row.get("path") or row.get("prepared")
        if not isinstance(raw_path, str) or not raw_path:
            raise EventSegmentationError(f"pool row {sound_id!r} has no audio path")
        audio_path = Path(raw_path).expanduser()
        if not audio_path.is_absolute() and root is not None:
            audio_path = root / audio_path
        sound_class = row.get("sound_class") or row.get("event_class")
        try:
            policy = uniform_policy(sound_class, **overrides)
            record = segment_audio_file(
                audio_path,
                sound_class=sound_class,
                gap_max_s=gap_max_s,
                min_pulse_train_onsets=min_pulse_train_onsets,
                policy=policy,
            )
        except (SoundSegmentError, OSError, ValueError) as error:
            counts["failed"] += 1
            failures.append(
                {
                    "sound_asset_id": sound_id,
                    "path": str(audio_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        record["sound_asset_id"] = sound_id
        record["sound_class"] = sound_class
        counts["measured"] += 1
        status = str(record["status"])
        by_status[status] = by_status.get(status, 0) + 1
        by_rule[str(record["rule"])] = by_rule.get(str(record["rule"]), 0) + 1
        class_row = by_class.setdefault(
            str(sound_class or "unlabelled"),
            {"measured": 0, "certified": 0, "not_certified": 0},
        )
        class_row["measured"] += 1
        if status == STATUS_PASS:
            class_row["certified"] += 1
        else:
            class_row["not_certified"] += 1
        by_id[sound_id] = record
        if out_rows is not None:
            out_rows[index]["event_segmentation"] = deepcopy(record)
        if on_row is not None:
            on_row(index, sound_id, record)

    parameters = {
        "gap_max_s": float(gap_max_s),
        "min_pulse_train_onsets": int(min_pulse_train_onsets),
        "include_speech": bool(include_speech),
        "policy_overrides": overrides,
        "uniform_activity_family": UNIFORM_ACTIVITY_FAMILY,
        "method": METHOD,
    }
    statistics = {
        "counts": counts,
        "by_status": by_status,
        "by_rule": by_rule,
        "by_sound_class": by_class,
        "certified": by_status.get(STATUS_PASS, 0),
        "not_certified": counts["measured"] - by_status.get(STATUS_PASS, 0),
        "human_reviewed": 0,
        "failures": failures,
    }
    backfill_block = {
        "schema": BACKFILL_SCHEMA,
        "method": METHOD,
        "certification": CERTIFICATION,
        "certified_by": CERTIFIED_BY,
        "human_certified": False,
        "source_pool": str(Path(pool_path).resolve()) if pool_path else None,
        "parameters": parameters,
        "statistics": statistics,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if out_payload is not None:
        out_payload["event_segmentation_backfill"] = backfill_block
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "method": METHOD,
        "source_pool": backfill_block["source_pool"],
        "parameters": parameters,
        "statistics": statistics,
        "claim_boundary": CLAIM_BOUNDARY,
        "by_sound_asset_id": by_id,
    }
    return out_payload if out_payload is not None else {"sounds": rows}, sidecar


__all__ = [
    "BACKFILL_SCHEMA",
    "CERTIFICATION",
    "CERTIFIED_BY",
    "CLAIM_BOUNDARY",
    "DEFAULT_GAP_MAX_S",
    "EventSegmentationError",
    "METHOD",
    "MIN_PULSE_TRAIN_ONSETS",
    "RULE_GAPS_WITHIN_MAX",
    "RULE_GAP_EXCEEDS_MAX",
    "RULE_NO_ONSET",
    "RULE_PULSE_TRAIN",
    "RULE_SINGLE_RUN",
    "SCHEMA",
    "SIDECAR_SCHEMA",
    "SPEECH_CLASSES",
    "STATUS_MULTI",
    "STATUS_PASS",
    "STATUS_SILENT",
    "UNIFORM_ACTIVITY_FAMILY",
    "backfill_pool",
    "classify_activity",
    "is_speech_like",
    "segment_audio_file",
    "segment_recording",
    "segmentation_record",
    "uniform_policy",
]
