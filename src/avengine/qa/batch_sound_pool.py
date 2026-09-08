"""Join existing P7 speech and registered event PCM for offline batch assignment."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf

from avengine.assets.sound_events import SoundEventError, extract_sound_events
from avengine.rooms.conditioned_sampler import (
    load_conditioned_sound_pool, source_normalization_metadata,
)


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
    object_classes = spec["object_sound_classes"]
    species_classes = spec["species_sound_classes"]
    max_duration = float(spec.get("max_clip_s", 5.0))
    if not np.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("max_clip_s must be positive and finite")
    metadata = defaultdict(list)
    for row in event_manifest["clips"]:
        if row.get("status") == "event":
            metadata[row["sound_asset_id"]].append(row)
    speech = load_conditioned_sound_pool(prepared, source_path=prepared_path)
    speech_allowed = [
        asset["asset_id"] for asset in registry["assets"]
        if asset["entity_class"] == "articulated_human"
        or (asset["entity_class"] in {"rigid_object", "rigid_static_object"}
            and asset["identity"].get("category") in spec.get("speech_playback_categories", ["audio_playback"]))
    ]
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
        if sound_class == "speech_playback":
            reason = "speech_consumes_existing_P7_prepared_set"
        elif not entries:
            reason = "missing_original_event_lineage"
        elif any(entry.get("truncated") for entry in entries):
            reason = "registered_event_is_truncated"
        elif raw["dry_audio"]["sample_count"] / raw["dry_audio"]["sample_rate_hz"] > max_duration:
            reason = "registered_event_exceeds_explicit_clip_budget"
        allowed = []
        for asset in registry["assets"]:
            if asset["entity_class"] == "articulated_animal":
                classes = species_classes.get(asset["identity"].get("species_id"), [])
            elif asset["entity_class"] in {"rigid_object", "rigid_static_object"}:
                classes = object_classes.get(asset["identity"].get("object_type"), [])
            else:
                classes = spec.get("human_nonverbal_sound_classes", [])
            if sound_class in classes:
                allowed.append(asset["asset_id"])
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
        if any(measurement.truncated for measurement in measurements):
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
            "counts": {"usable": len(sounds), "rejected_registered_events": len(rejected),
                       "by_sound_class": dict(Counter(sound["sound_class"] for sound in sounds)),
                       "rejections_by_reason": dict(Counter(row["reason"] for row in rejected))},
            "claim_boundary": "Source PCM remains unchanged. Existing detector extents and minimums are placeholders; no human calibration or event-count certification."}
