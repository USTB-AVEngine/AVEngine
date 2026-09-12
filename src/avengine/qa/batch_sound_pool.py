"""Join existing P7 speech and registered event PCM for offline batch assignment."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf

from avengine.assets.sound_events import SoundEventError, extract_sound_events
from avengine.dataset.source_capabilities import (
    assets_accepting_sound_class, normalize_sound_class_config,
    sound_class_asset_index, verify_segment_selection,
)
from avengine.rooms.conditioned_sampler import (
    load_conditioned_sound_pool, source_normalization_metadata,
)


_SEGMENT_RELATIVE_FIELDS = ("relative_path", "relative", "source_origin", "source_path")


def _segment_row_key(item: Mapping[str, Any]) -> str | None:
    for field in _SEGMENT_RELATIVE_FIELDS:
        if item.get(field):
            return str(item[field])
    return None


def _segment_selection_index(
    value: Any,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    """Read declared segment selections and P25's own read-back status for each.

    A ``prepare_segments`` payload is accepted whole: its ``segments`` are the rows and
    its ``verifications`` carry the status P25 got when it read each written file back.
    That status is the audio evidence; the structural check here does not replace it.
    """
    if value is None:
        return {}, {}
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    readback: dict[str, str] = {}
    if isinstance(value, Mapping) and "segments" in value:
        by_segment_id = {}
        for entry in value.get("verifications") or []:
            if isinstance(entry, Mapping) and entry.get("segment_id"):
                by_segment_id[str(entry["segment_id"])] = str(entry.get("status"))
        rows = [row for row in value["segments"] if isinstance(row, Mapping)]
        index: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            key = _segment_row_key(row)
            if key is None:
                continue
            index[key] = row
            segment_id = row.get("segment_id") or row.get("prepared_audio_id")
            if segment_id and str(segment_id) in by_segment_id:
                readback[key] = by_segment_id[str(segment_id)]
        return index, readback
    if isinstance(value, Mapping) and "selections" in value:
        value = value["selections"]
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()
                if isinstance(item, Mapping)}, {}
    if isinstance(value, list):
        index = {}
        for item in value:
            if not isinstance(item, Mapping):
                continue
            key = _segment_row_key(item)
            if key is not None:
                index[key] = item
        return index, {}
    raise ValueError("segment_selections must be a mapping, list or path")


# P25 marks a written segment it re-measured successfully; anything else is not
# audio evidence and must not authorize a truncation.
_ACCEPTED_READBACK_STATUSES = frozenset({"qualified", "pass"})


def _authorized_segment(
    entries: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, Any]],
    readback: Mapping[str, str] | None = None,
    expected_processing: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the verified selection that explains this event's truncation.

    An authorized selection must name the same recording, carry its own sample bounds
    and activity measurement, and pass structural verification.  Anything else — a
    manifest that merely says ``truncated`` with no provenance, a selection for another
    clip, or bounds that disagree with the event — stays an unexplained truncation and
    keeps the existing rejection.
    """
    for entry in entries:
        declared = entry.get("segment_selection")
        relative = entry.get("source")
        candidate = declared if isinstance(declared, Mapping) else selections.get(
            str(relative))
        if not isinstance(candidate, Mapping):
            continue
        status = (readback or {}).get(str(relative))
        if status is not None and status not in _ACCEPTED_READBACK_STATUSES:
            continue
        verdict = verify_segment_selection(
            candidate, clip={"relative": relative},
            expected_processing=expected_processing)
        if not verdict["verified"]:
            continue
        start, end = entry.get("start_sample"), entry.get("end_sample_exclusive")
        if isinstance(start, int) and isinstance(end, int):
            if (verdict["source_crop_start_sample"] != start
                    or verdict["source_crop_end_sample_exclusive"] != end):
                continue
        return {"relative": str(relative),
                "crop_authorization": verdict["crop_authorization"],
                "source_crop_start_sample": verdict["source_crop_start_sample"],
                "source_crop_end_sample_exclusive": verdict[
                    "source_crop_end_sample_exclusive"],
                "activity_coverage": verdict["activity_coverage"],
                "segment_readback_status": status,
                "coordinate_bounds": verdict["coordinate_bounds"],
                "unverified_bounds": verdict["unverified_bounds"],
                "processing": "authorized_segment_selection",
                "claim_boundary": (
                    "structural verification plus P25's own read-back status; not a "
                    "listening test and not an audibility certificate"),
                }
    return None


def build_batch_sound_pool(spec: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    """Read original PCM and reuse its existing event measurement algorithm.

    No PCM is cropped, normalized, generated or overwritten. Detector extents
    include the existing event guard and remain a research measurement, not a
    human source-activity or acoustic-answerability certificate.
    """
    prepared_path = Path(spec["prepared_speech_manifest"]).resolve()
    event_registry_path = Path(spec["sound_event_registry"]).resolve()
    manifest_path = Path(spec["sound_event_manifest"]).resolve()
    prepared = json.loads(prepared_path.read_text())
    event_registry = json.loads(event_registry_path.read_text())
    event_manifest = json.loads(manifest_path.read_text())
    # Both semantic mappings stay required here: a spec missing one must fail rather
    # than silently resolve an empty allowlist.  One shared definition of "which assets
    # accept this sound class" is then inverted once, instead of rescanning the whole
    # registry for every clip in the library.
    sound_class_config = normalize_sound_class_config({
        "object_sound_classes": spec["object_sound_classes"],
        "species_sound_classes": spec["species_sound_classes"],
        "speech_playback_categories": spec.get("speech_playback_categories"),
        "human_nonverbal_sound_classes": spec.get("human_nonverbal_sound_classes"),
    })
    species_classes = sound_class_config["species_sound_classes"]
    accepting_index = sound_class_asset_index(registry, sound_class_config)
    # ``max_clip_s`` is a declared filter, not an episode property.  The historical
    # five-second value is kept whenever a spec states it, so an existing batch still
    # reproduces its pool exactly; when a spec states nothing, no length filter is
    # applied, because a recording longer than five seconds is now material for a
    # segment selection rather than something to drop before it is measured.
    declared_max = spec.get("max_clip_s")
    max_duration = None if declared_max is None else float(declared_max)
    if max_duration is not None and (
            not np.isfinite(max_duration) or max_duration <= 0):
        raise ValueError("max_clip_s must be positive and finite")
    segment_selections, segment_readback = _segment_selection_index(
        spec.get("segment_selections"))
    expected_segment_processing = spec.get("segment_processing")
    if expected_segment_processing is not None and not isinstance(
            expected_segment_processing, Mapping):
        raise ValueError("segment_processing must be an object of requested parameters")
    metadata = defaultdict(list)
    for row in event_manifest["clips"]:
        if row.get("status") == "event":
            metadata[row["sound_asset_id"]].append(row)
    speech = load_conditioned_sound_pool(prepared, source_path=prepared_path)
    speech_allowed = assets_accepting_sound_class(
        registry, "speech_playback", sound_class_config, index=accepting_index)
    sounds, rejected = [], []
    for raw in speech:
        sound = deepcopy(raw)
        sound.update(sound_class="speech_playback", compatible_asset_ids=speech_allowed,
                     compatible_object_categories=spec.get("speech_playback_categories", ["audio_playback"]),
                     sound_identity_id="speaker:" + sound["speaker_id"],
                     sound_identity_keys=["speaker:" + sound["speaker_id"],
                                          "source:" + sound["source_pcm_path"]],
                     activity_measurement="P7_prepared_speech_band_activity",
                     source_metadata_manifest=str(prepared_path))
        sounds.append(sound)
    for raw in event_registry["sound_assets"]:
        sound_id = raw["sound_asset_id"]
        sound_class = raw["semantic_sound_class"]
        reason = None
        entries = metadata.get(sound_id, [])
        # One chain, first match wins, exactly as before: a speech clip is consumed by
        # the prepared set whether or not it is also truncated or long.
        authorized = _authorized_segment(
            entries, segment_selections, segment_readback, expected_segment_processing)
        if sound_class == "speech_playback":
            reason = "speech_consumes_existing_P7_prepared_set"
        elif not entries:
            reason = "missing_original_event_lineage"
        elif any(entry.get("truncated") for entry in entries) and authorized is None:
            reason = "registered_event_is_truncated"
        elif (max_duration is not None
              and raw["dry_audio"]["sample_count"] / raw["dry_audio"]["sample_rate_hz"]
              > max_duration):
            reason = "registered_event_exceeds_explicit_clip_budget"
        allowed = assets_accepting_sound_class(
            registry, sound_class, sound_class_config, index=accepting_index)
        if reason is None and not allowed:
            reason = "no_explicit_semantic_asset_mapping"
        if reason is not None:
            rejected.append({"sound_asset_id": sound_id, "sound_class": sound_class, "reason": reason})
            continue
        uri = urlparse(raw["dry_audio"]["uri"])
        if uri.scheme != "file":
            rejected.append({"sound_asset_id": sound_id, "sound_class": sound_class,
                             "reason": "nonlocal_registered_pcm"})
            continue
        path = Path(unquote(uri.path)).resolve()
        samples, rate = sf.read(path, dtype="float64", always_2d=True)
        header = raw["dry_audio"]
        if samples.shape != (header["sample_count"], 1) or rate != header["sample_rate_hz"]:
            raise ValueError(f"registered PCM header differs: {sound_id}")
        if not np.isfinite(samples).all():
            raise ValueError(f"nonfinite source PCM: {sound_id}")
        try:
            measurements = extract_sound_events(samples[:, 0], rate, event_class=sound_class)
        except SoundEventError as exc:
            rejected.append({"sound_asset_id": sound_id, "sound_class": sound_class,
                             "reason": "existing_event_measurement_failed", "detail": str(exc)})
            continue
        if any(measurement.truncated for measurement in measurements) and authorized is None:
            rejected.append({"sound_asset_id": sound_id, "sound_class": sound_class,
                             "reason": "existing_event_measurement_would_truncate"})
            continue
        intervals = [[event.start_sample, event.end_sample_exclusive] for event in measurements]
        # Union overlapping guard intervals; detector islands are not independent
        # QA-23 events and are never relabelled as event counts here.
        union = []
        for start, end in sorted(intervals):
            if union and start <= union[-1][1]:
                union[-1][1] = max(end, union[-1][1])
            else:
                union.append([start, end])
        duration = sum(end - start for start, end in union) / rate
        species = next((name for name, classes in species_classes.items() if sound_class in classes), None)
        if species is not None and duration < float(spec.get("minimum_animal_activity_s", 0.5)):
            rejected.append({"sound_asset_id": sound_id, "sound_class": sound_class,
                             "reason": "animal_activity_below_declared_placeholder_minimum",
                             "measured_activity_s": duration})
            continue
        origins = sorted({str((Path(event_manifest["library_root"]) / entry["source"]).resolve())
                          for entry in entries})
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        source_normalization = source_normalization_metadata(
            raw,
            related=entries,
            measured_peak_dbfs=(float(20.0 * np.log10(peak)) if peak > 0.0 else None),
            measured_peak_source="registered_event_pcm",
            source_overrides={
                "pcm_path": str(path),
                "source_origin": origins[0],
                "source_origin_aliases": origins,
                "event_manifest_path": str(manifest_path),
                "event_registry_path": str(event_registry_path),
            },
        )
        sounds.append({"sound_asset_id": sound_id, "path": str(path), "sound_class": sound_class,
                       "event_class": sound_class, "species_id": species, "source_origin": origins[0],
                       "source_origin_aliases": origins, "sound_identity_id": "source:" + origins[0],
                       "sound_identity_keys": ["source:" + origin for origin in origins] + ["registered_pcm:" + sound_id],
                       "compatible_asset_ids": sorted(allowed), "sample_count": len(samples),
                       "sample_rate_hz": rate, "active_duration_s": duration,
                       "source_activity_intervals_samples": union,
                       "audible_start_sample": union[0][0], "audible_end_sample_exclusive": union[-1][1],
                       "activity_measurement": "existing_avengine_sound_events_detector_extent_with_guard",
                       "activity_calibration": "placeholder", "activity_guard_included": True,
                       "activity_is_qa_event_count": False, "linear_gain": 1.0,
                       **({"segment_selection": deepcopy(authorized)}
                          if authorized is not None else {}),
                       "normalization_applied": False, "source_normalization": source_normalization,
                       "source_metadata_manifest": str(manifest_path),
                       "source_event_registry": str(event_registry_path),
                       "human_review": {"status": "not_measured"}})
    ids = [sound["sound_asset_id"] for sound in sounds]
    if len(set(ids)) != len(ids):
        raise ValueError("joined sound pool has duplicate IDs")
    return {"schema": "avengine_qa_batch_sound_pool_v1", "status": "research_candidate",
            "sounds": sounds, "rejected": rejected,
            "source_spec": deepcopy(dict(spec)),
            "source_counts": {"P7_prepared": len(speech), "registered_events": len(event_registry["sound_assets"])},
            "clip_budget": {
                "max_clip_s": max_duration,
                "mode": ("declared_filter" if max_duration is not None
                         else "no_declared_length_filter"),
                "note": ("a declared max_clip_s reproduces the historical pool; with "
                         "none declared, length alone excludes nothing and a long "
                         "recording reaches segment selection instead"),
            },
            "segment_selection": {
                "declared_count": len(segment_selections),
                "with_p25_readback_status": len(segment_readback),
                "readback_statuses": dict(Counter(segment_readback.values())),
                "requested_processing": deepcopy(dict(expected_segment_processing))
                if expected_segment_processing else None,
                "authorized_count": sum(
                    1 for sound in sounds if sound.get("segment_selection")),
                "owner": "P25",
            },
            "counts": {"usable": len(sounds), "rejected_registered_events": len(rejected),
                       "by_sound_class": dict(Counter(sound["sound_class"] for sound in sounds)),
                       "rejections_by_reason": dict(Counter(row["reason"] for row in rejected))},
            "claim_boundary": "Source PCM remains unchanged. Existing detector extents and minimums are placeholders; no human calibration or event-count certification."}
