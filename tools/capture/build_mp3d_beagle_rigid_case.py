#!/usr/bin/env python3
"""Prepare one fresh beagle plus rigid-object Habitat capture case.

This is a bounded input adapter for the P4 native canary. It copies one
existing planned beagle track, truncates it to the requested clock, and adds a
floor-resting rigid asset using the explicit Habitat binding resolver.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.assets.habitat_static_assets import (
    HabitatStaticAssetError,
    load_habitat_asset_bindings,
)
from avengine.assets.mp3d_region_actor_tracks import (
    MP3DRegionActorTrackError,
    materialize_habitat_rigid_track,
)
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    _resolve_visual_clock,
)


class P4CaseError(ValueError):
    """The P4 canary input cannot be prepared safely."""


def _read_json(path: Path, owner: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise P4CaseError(f"{owner} must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P4CaseError(f"cannot read {owner}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise P4CaseError(f"{owner} must be an object")
    return deepcopy(dict(value))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )


def _clock(source: Mapping[str, Any], frame_count: int) -> dict[str, Any]:
    try:
        resolved = _resolve_visual_clock(
            frame_count=frame_count,
            frame_rate_hz=source["frame_rate_hz"],
            ticks_per_frame=source["ticks_per_frame"],
            time_base_hz=source["time_base_hz"],
        )
    except (CurrentMP3DDynamicAudioError, KeyError, TypeError, ValueError) as exc:
        raise P4CaseError(f"source clock cannot be resized: {exc}") from exc
    return dict(resolved)


def _track_for_slot(case_path: Path, case: Mapping[str, Any], slot: str) -> dict[str, Any]:
    records = case.get("actor_tracks")
    if not isinstance(records, list):
        raise P4CaseError("source case has no actor_tracks")
    matches = [
        item
        for item in records
        if isinstance(item, Mapping) and item.get("source_slot_id") == slot
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("track_path"), str):
        raise P4CaseError(f"source case has no unique track for {slot!r}")
    track_path = (case_path.parent / matches[0]["track_path"]).resolve()
    try:
        track_path.relative_to(case_path.parent)
    except ValueError as exc:
        raise P4CaseError("source track escapes case directory") from exc
    return _read_json(track_path, f"source track {slot}")


def _source_for_endpoint(request: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    sources = request.get("sources")
    if not isinstance(sources, list):
        raise P4CaseError("M1 request has no sources")
    matches = [
        item
        for item in sources
        if isinstance(item, Mapping) and item.get("source_id") == endpoint
    ]
    if len(matches) != 1:
        raise P4CaseError(f"M1 request has no unique source {endpoint!r}")
    return deepcopy(dict(matches[0]))


def build_case(
    *,
    source_case_manifest: Path,
    source_m1_request: Path,
    output: Path,
    asset_id: str,
    runtime_registry: Path | None,
    external_index: Path | None,
    binding_delta: Path | None,
    frame_count: int,
    floor_height_m: float,
    speaker_emitter_position: Sequence[float],
    speaker_semantic_id: int,
) -> dict[str, Any]:
    if frame_count < 2:
        raise P4CaseError("frame_count must be at least two")
    if len(speaker_emitter_position) != 3:
        raise P4CaseError("speaker_emitter_position must contain three values")
    source_case_path = source_case_manifest.expanduser().resolve()
    source_case = _read_json(source_case_path, "source case")
    source_request = _read_json(source_m1_request, "source M1 request")
    source_records = source_case.get("actor_tracks")
    if not isinstance(source_records, list) or not source_records:
        raise P4CaseError("source case has no actor tracks")
    source_track = _track_for_slot(source_case_path, source_case, "source1")
    source_clock = source_track.get("clock")
    if not isinstance(source_clock, Mapping):
        raise P4CaseError("source track has no clock")
    clock = _clock(source_clock, frame_count)
    source_track["clock"] = dict(clock)
    frames = source_track.get("frames")
    if not isinstance(frames, list) or len(frames) < frame_count:
        raise P4CaseError("source track has fewer frames than requested")
    source_track["frames"] = frames[:frame_count]

    try:
        binding = load_habitat_asset_bindings(
            [asset_id],
            runtime_registry_path=runtime_registry,
            external_index_path=external_index,
            binding_delta_path=binding_delta,
        )[asset_id]
    except HabitatStaticAssetError as exc:
        raise P4CaseError(str(exc)) from exc
    if binding.normalized_entity_class != "rigid_object":
        raise P4CaseError(f"{asset_id!r} is not a rigid Habitat source")

    beagle_endpoint = str(source_track["source_endpoint_id"])
    beagle_source = _source_for_endpoint(source_request, beagle_endpoint)
    speaker_endpoint = "speaker_muzzle"
    source_request["sources"] = [
        beagle_source,
        {
            "source_id": speaker_endpoint,
            "world_from_source": {
                "translation_m": [float(value) for value in speaker_emitter_position],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    ]
    try:
        rigid_track = materialize_habitat_rigid_track(
            actor={
                "actor_id": "speaker_0",
                "source_slot_id": "source2",
                "source_endpoint_id": speaker_endpoint,
                "semantic_id": int(speaker_semantic_id),
                "asset_id": asset_id,
                "asset_revision": binding.revision or "sound_source_assets_v1",
                "entity_class": "rigid_object",
            },
            m1_request=source_request,
            clock=clock,
            habitat_binding=binding.to_dict(),
            floor_height_m=floor_height_m,
        )
    except (MP3DRegionActorTrackError, TypeError, ValueError) as exc:
        raise P4CaseError(f"cannot materialize rigid P4 track: {exc}") from exc

    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise P4CaseError(f"refusing to replace P4 case output: {output}")
    output.mkdir(parents=True)
    _write_json(output / "m1_capture_request.json", source_request)
    _write_json(output / "tracks/source1.json", source_track)
    _write_json(output / "tracks/source2.json", rigid_track)
    actor_records = [
        {
            "actor_id": source_track["actor_id"],
            "source_slot_id": "source1",
            "source_endpoint_id": beagle_endpoint,
            "semantic_id": int(source_track["semantic_id"]),
            "track_path": "tracks/source1.json",
        },
        {
            "actor_id": rigid_track["actor_id"],
            "source_slot_id": "source2",
            "source_endpoint_id": speaker_endpoint,
            "semantic_id": int(speaker_semantic_id),
            "entity_class": "rigid_object",
            "track_path": "tracks/source2.json",
        },
    ]
    case = {
        "schema": source_case.get("schema"),
        "artifact_role": "planned_habitat_actor_apply_case",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "native_observed": False,
        "region": deepcopy(source_case.get("region", {"house_id": "17DRP5sb8fy"})),
        "route_family_id": source_case.get("route_family_id"),
        "motion_case": source_case.get("motion_case"),
        "clock": dict(clock),
        "m1_request_path": str((output / "m1_capture_request.json").resolve()),
        "planned_timeline_path": source_case.get("planned_timeline_path"),
        "actor_tracks": actor_records,
        "native_pending": {
            "capture": None,
            "emitter_readback": None,
            "support_contact": None,
            "collision": None,
            "object_id": None,
            "rlr": None,
        },
        "audio_consumption": (
            "requires observed native emitter trajectories; static emitter is "
            "read back from the rigid binding offset"
        ),
    }
    _write_json(output / "case_manifest.json", case)
    _write_json(
        output / "prepare_receipt.json",
        {
            "schema": "avengine_p4_beagle_rigid_case_v1",
            "status": "pass",
            "clock": clock,
            "asset_binding": binding.to_dict(),
            "source_case": str(source_case_path),
            "source_m1_request": str(source_m1_request.expanduser().resolve()),
            "case_manifest": "case_manifest.json",
            "m1_request": "m1_capture_request.json",
            "tracks": ["tracks/source1.json", "tracks/source2.json"],
        },
    )
    return case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case-manifest", required=True, type=Path)
    parser.add_argument("--source-m1-request", required=True, type=Path)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--runtime-registry", type=Path)
    parser.add_argument("--external-index", type=Path)
    parser.add_argument("--binding-delta", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--floor-height-m", type=float, default=0.072447)
    parser.add_argument(
        "--speaker-emitter-position",
        type=float,
        nargs=3,
        default=(-7.6, 0.173368, -3.1),
    )
    parser.add_argument("--speaker-semantic-id", type=int, default=333)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        case = build_case(
            source_case_manifest=args.source_case_manifest,
            source_m1_request=args.source_m1_request,
            output=args.output,
            asset_id=args.asset_id,
            runtime_registry=args.runtime_registry,
            external_index=args.external_index,
            binding_delta=args.binding_delta,
            frame_count=args.frame_count,
            floor_height_m=args.floor_height_m,
            speaker_emitter_position=args.speaker_emitter_position,
            speaker_semantic_id=args.speaker_semantic_id,
        )
    except (P4CaseError, HabitatStaticAssetError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "case_manifest": str((args.output / "case_manifest.json").resolve()),
                "frame_count": case["clock"]["frame_count"],
                "actor_count": len(case["actor_tracks"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
