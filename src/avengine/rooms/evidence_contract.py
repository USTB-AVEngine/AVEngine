"""Format and cross-file checks for existing native QA evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.capture.neutral_readback import validate_clock

FILES = ("pixel_visibility_truth.json", "native_pixel_masks_depth_authority_v1.npz",
         "appearance_review.json", "actor_occluders.json", "research_report.json")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence JSON must be an object: {path}")
    return value


def validate_evidence_contract(files: Mapping[str, Any], *,
                               clock: Mapping[str, Any] | None = None,
                               require_complete: bool = True) -> dict:
    """Validate structure/identity; do not certify visibility, sound or appearance."""
    paths = {}
    for name in FILES:
        if not isinstance(files.get(name), (str, Path)):
            raise ValueError(f"EvidenceContract missing {name}")
        path = Path(files[name]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"EvidenceContract file unavailable: {name}: {path}")
        paths[name] = path
    truth = _json(paths["pixel_visibility_truth.json"])
    if truth.get("schema") != "avengine_qa_pixel_visibility_truth_v1":
        raise ValueError("unsupported pixel visibility truth schema")
    frame_indices = truth.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in frame_indices):
        raise ValueError("pixel truth requires explicit nonnegative frame indices")
    if frame_indices != sorted(set(frame_indices)):
        raise ValueError("pixel truth frame indices must be unique and ordered")
    poses = truth.get("camera_pose_ids")
    if not isinstance(poses, list) or len(poses) != len(frame_indices) or len(set(poses)) != len(poses):
        raise ValueError("pixel truth camera pose identities are incomplete or duplicated")
    instances = truth.get("per_instance")
    if not isinstance(instances, Mapping) or not instances:
        raise ValueError("pixel truth requires per_instance records")
    semantic = [v.get("semantic_id") for v in instances.values()]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in semantic) or len(set(semantic)) != len(semantic):
        raise ValueError("source semantic IDs must be positive and distinct")
    resolution = truth.get("resolution_hw")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("pixel truth resolution_hw is required")
    if clock is not None:
        clock = validate_clock(clock)
        if max(frame_indices) >= clock["frame_count"]:
            raise ValueError("pixel truth exceeds episode clock")
        if require_complete and frame_indices != list(range(int(clock["frame_count"]))):
            raise ValueError("pixel truth has missing episode frames")
    for actor, record in instances.items():
        frames = record.get("frames")
        if not isinstance(frames, list) or [f.get("frame_index") for f in frames] != frame_indices:
            raise ValueError(f"pixel truth frame identity differs for {actor}")
        for frame in frames:
            if frame.get("state") not in {"visible_clear", "visible_occluded", "fully_occluded",
                                          "out_of_view", "unknown"}:
                raise ValueError(f"invalid pixel visibility state for {actor}")
    with np.load(paths["native_pixel_masks_depth_authority_v1.npz"], allow_pickle=False) as arrays:
        # Existing UE consumers use this key. The roadmap's shorter spelling is
        # accepted as an explicit alias; conflicting aliases are never accepted.
        key = "depth_derived_modal_semantic" if "depth_derived_modal_semantic" in arrays else "modal"
        if key not in arrays:
            raise ValueError("native masks lack modal semantic IDs")
        modal = arrays[key]
        if modal.ndim != 3 or list(modal.shape[1:]) != resolution or max(frame_indices) >= len(modal):
            raise ValueError("modal shape differs from pixel truth")
        if not np.issubdtype(modal.dtype, np.integer):
            raise ValueError("modal semantic IDs must be integers")
        if clock and require_complete and len(modal) != clock["frame_count"]:
            raise ValueError("modal frame count differs from episode clock")
        if key != "modal" and "modal" in arrays and not np.array_equal(arrays["modal"], modal):
            raise ValueError("modal semantic aliases disagree")
        for actor in instances:
            name = f"target_only_{actor}"
            if name not in arrays or arrays[name].shape != modal.shape:
                raise ValueError(f"missing or mismatched target-only masks: {actor}")
    appearance = _json(paths["appearance_review.json"])
    if not isinstance(appearance.get("actors"), Mapping):
        raise ValueError("appearance_review requires actors")
    if not set(appearance["actors"]).issubset(instances):
        raise ValueError("appearance_review contains an unknown actor")
    for actor, review in appearance["actors"].items():
        if review.get("status") in {"pass", "reviewed", "astra_reviewed"}:
            refs = review.get("frame_refs", review.get("reviewed_frames"))
            if not review.get("value") or not isinstance(refs, list) or not refs:
                raise ValueError(f"reviewed appearance lacks a value or frames: {actor}")
            if any(i not in frame_indices for i in refs):
                raise ValueError(f"appearance review refers to unavailable frame: {actor}")
    occluders = _json(paths["actor_occluders.json"])
    if not isinstance(occluders.get("frame_records"), list):
        raise ValueError("actor_occluders requires frame_records")
    if Path(occluders.get("masks_path", "")).resolve() != paths["native_pixel_masks_depth_authority_v1.npz"]:
        raise ValueError("occluders reference different native masks")
    for record in occluders["frame_records"]:
        if record.get("frame_index") not in frame_indices or record.get("target_instance_id") not in instances:
            raise ValueError("occluder frame or target identity is unavailable")
    audio = _json(paths["research_report.json"])
    audio_clock = validate_clock(audio.get("clock"))
    if clock and any(audio_clock[key] != clock[key] for key in
                     ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count",
                      "time_base_hz", "ticks_per_frame")):
        raise ValueError("audio clock differs from capture plan")
    if not isinstance(audio.get("events"), list) or not audio.get("mixture_path"):
        raise ValueError("audio report requires mixture_path and events")
    if not Path(audio["mixture_path"]).is_file():
        raise ValueError("audio mixture is missing")
    event_ids = set()
    for event in audio["events"]:
        if not event.get("event_id") or event["event_id"] in event_ids:
            raise ValueError("audio event IDs must be present and unique")
        event_ids.add(event["event_id"])
        if event.get("actor_id") not in instances:
            raise ValueError("audio event refers to an unknown actor")
        if not event.get("output_stem") or not Path(event["output_stem"]).is_file():
            raise ValueError("audio event stem is missing")
        interval = event.get("wet_tail_interval")
        if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or
                                     interval[0] < 0 or interval[1] < interval[0]):
            raise ValueError("invalid wet tail interval")
        if "wet_tail_interval" not in event:
            raise ValueError("audio event lacks wet tail readback")
    return {"status": "pass", "kind": "format_and_consistency_only",
            "frame_count": len(frame_indices), "actor_ids": list(instances),
            "files": {name: str(path) for name, path in paths.items()},
            "formal_certification": False}
