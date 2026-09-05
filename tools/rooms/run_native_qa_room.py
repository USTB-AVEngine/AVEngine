#!/usr/bin/env python3
"""Materialize one native Apartment QA plan without launching UE.

The output is a fresh room-pool entry plus a common question-driven plan.
Actual SPEAR capture is intentionally a separate command written into the
output so GPU allocation and UE launch remain explicit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.rooms.native_qa_room import (  # noqa: E402
    DEFAULT_FRAME_COUNT,
    DEFAULT_FRAME_RATE_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SEED,
    DEFAULT_SOURCE_ASSET_IDS,
    discover_native_apartment_resources,
    build_native_apartment_layout,
    build_native_apartment_qa_plan,
    load_sound_pool,
    native_apartment_room_entry,
    write_json,
)
from avengine.runtime_profiles import load_source_asset_runtime_registry  # noqa: E402



def _materialize_manifest(
    output: Path,
    resources: Any,
    layout: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    room_dir = output / "room_pool" / "apartment_0000"
    room_dir.mkdir(parents=True, exist_ok=True)
    objects_path = write_json(room_dir / "objects.json", {"objects": layout["objects"]})
    manifest = {
        "schema": "avengine_native_qa_room_manifest_v1",
        "room_id": resources.room_id,
        "scene_id": resources.scene_id,
        "backend_route": "spear_unreal",
        "status": "research_candidate",
        "scene": {"scene_id": resources.scene_id, "map_path": resources.map_path},
        "geometry": layout["geometry"],
        "assets": {
            "visual_geometry": str(resources.surface_glb),
            "collision_geometry": str(resources.surface_glb),
            "objects": str(objects_path),
        },
        "native_execution": {
            "map_or_stage": resources.map_path,
            "navmesh": str(resources.navmesh),
            "route_bank": str(resources.route_bank),
        },
        "provenance": layout["native_source"],
        "coordinate_contract": layout["coordinate_contract"],
        "claim_boundary": (
            "native Apartment surface/object audit and native route bank are "
            "planning inputs; UE capture and pixel/audio evidence remain later gates"
        ),
    }
    manifest_path = write_json(room_dir / "room_manifest.json", manifest)
    entry = native_apartment_room_entry(resources, manifest_path=manifest_path)
    entry["floor_height_m"] = float(layout["native_floor_height_m"])
    return manifest_path, entry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_snapshot(
    source_stage: Path | None,
    target_stage: Path | None,
) -> dict[str, Any] | None:
    """Record the private stage copy without modifying either stage."""

    if source_stage is None and target_stage is None:
        return None
    if source_stage is None or target_stage is None:
        raise ValueError("stage-source and stage-target must be supplied together")
    source = source_stage.expanduser().resolve()
    target = target_stage.expanduser().resolve()
    source_content = source / "Content/SPEAR"
    target_content = target / "Content/SPEAR"
    if not source_content.is_dir() or not target_content.is_dir():
        raise FileNotFoundError("both stage Content/SPEAR directories are required")
    source_files = sorted(path for path in source_content.rglob("*") if path.is_file())
    target_files = sorted(path for path in target_content.rglob("*") if path.is_file())
    source_rel = {path.relative_to(source_content): path for path in source_files}
    target_rel = {path.relative_to(target_content): path for path in target_files}
    if set(source_rel) != set(target_rel):
        raise ValueError("private stage Content/SPEAR file set differs from read-only source")
    mismatches = [
        str(relative)
        for relative in sorted(source_rel)
        if source_rel[relative].stat().st_size != target_rel[relative].stat().st_size
        or _sha256(source_rel[relative]) != _sha256(target_rel[relative])
    ]
    map_rel = Path("Scenes/apartment_0000/Maps/apartment_0000.umap")
    source_map = source_content / map_rel
    target_map = target_content / map_rel
    if not source_map.is_file() or not target_map.is_file():
        raise FileNotFoundError("private stage apartment_0000 map is missing")
    return {
        "schema": "avengine_native_qa_stage_copy_readback_v1",
        "source_stage": str(source),
        "target_stage": str(target),
        "source_content_spear_bytes": sum(path.stat().st_size for path in source_files),
        "target_content_spear_bytes": sum(path.stat().st_size for path in target_files),
        "source_content_spear_file_count": len(source_files),
        "target_content_spear_file_count": len(target_files),
        "source_map": str(source_map),
        "target_map": str(target_map),
        "source_map_bytes": source_map.stat().st_size,
        "target_map_bytes": target_map.stat().st_size,
        "source_map_sha256": _sha256(source_map),
        "target_map_sha256": _sha256(target_map),
        "content_spear_checksum_status": "pass" if not mismatches else "fail",
        "content_spear_mismatches": mismatches,
        "copy_policy": "source_read_only_target_private_no_saved_cooked_copy",
    }


def _capture_command(
    *,
    plan_dir: Path,
    output_dir: Path,
    runtime_python: Path,
    uproject: Path,
    unreal_editor: Path,
    spear_ext_dir: Path,
    graphics_adapter: int,
    rpc_port: int,
    streaming_warmup_frames: int,
    ddc_profile: str,
    ddc_directory: Path,
) -> list[str]:
    return [
        str(runtime_python),
        str(REPOSITORY / "tools/rooms/run_spear_residential_episode.py"),
        "--episode-root", str(plan_dir),
        "--output", str(output_dir),
        "--uproject", str(uproject),
        "--unreal-editor", str(unreal_editor),
        "--spear-ext-dir", str(spear_ext_dir),
        "--graphics-adapter", str(graphics_adapter),
        "--rpc-port", str(rpc_port),
        "--expected-stage-actor-count", "0",
        "--ddc-profile", ddc_profile,
        "--ddc-directory", str(ddc_directory),
        "--width", "1280",
        "--height", "720",
        "--streaming-warmup-frames", str(streaming_warmup_frames),
        "--native-multimodal",
        "--visual-only-research",
        "--keep-frames",
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import soundfile  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "soundfile is required for real sound-pool metadata; add the "
            "task-owned native_python_addons directory to PYTHONPATH"
        ) from exc
    required_runtime = (
        "source_root",
        "route_bank",
        "sound_pool",
        "runtime_python",
        "uproject",
        "unreal_editor",
        "spear_ext_dir",
        "ddc_directory",
        "graphics_adapter",
        "rpc_port",
    )
    missing = [name for name in required_runtime if getattr(args, name, None) is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"explicit runtime/input arguments are required: {flags}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    resources = discover_native_apartment_resources(
        repository=REPOSITORY,
        source_root=args.source_root,
        route_bank=args.route_bank,
        room_profile_path=args.room_profile,
    )
    stage_readback = _stage_snapshot(args.stage_source, args.stage_target)
    layout = build_native_apartment_layout(resources)
    manifest_path, room_entry = _materialize_manifest(output, resources, layout)
    room_pool = {"schema": "avengine_qa_room_catalog_v1", "rooms": [room_entry]}
    room_pool_path = write_json(output / "room_pool.json", room_pool)
    source_registry = load_source_asset_runtime_registry(args.source_registry)
    sounds = load_sound_pool(args.sound_pool)
    plan, plan_layout, pf, routes = build_native_apartment_qa_plan(
        resources=resources,
        source_registry=source_registry,
        sounds=sounds,
        episode_id=args.episode_id,
        source_asset_ids=tuple(args.source_asset_ids),
        qa_ids=args.qa_ids,
        seed=args.seed,
        frame_count=args.frame_count,
        frame_rate_hz=args.frame_rate_hz,
        sample_rate_hz=args.sample_rate_hz,
        camera_motion=args.camera_motion,
        audio_mode=args.audio_mode,
        start_hold_frames=args.start_hold_frames,
    )
    plan["resources"].update(room_entry)
    plan["resources"]["manifest"] = str(manifest_path)
    plan["resources"]["native_room_adapter"] = "avengine_native_spear_apartment_qa_room_v1"
    if stage_readback is not None:
        plan["native_stage_readback"] = deepcopy(stage_readback)
    plan_dir = output / "plan"
    plan_dir.mkdir()
    plan_path = write_json(plan_dir / "episode_plan.json", plan)
    write_json(plan_dir / "room_layout.json", plan_layout)
    write_json(plan_dir / "voice_bindings.json", {"bindings": plan["voice_bindings"]})
    write_json(plan_dir / "audio_events.json", {"events": plan["audio_events"]})
    nav = plan["room_capabilities"]["evidence_refs"]["navigation"]
    np.savez_compressed(
        plan_dir / "navigation.npz",
        binary_navmesh=pf.get_topdown_view(nav["resolution_m"], nav["floor_height_m"]),
        bounds_m=pf.get_bounds(),
    )
    write_json(output / "source_pool.json", {"sounds": sounds})
    if stage_readback is not None:
        write_json(output / "stage_readback.json", stage_readback)
    request = {
        "schema": "avengine_native_qa_room_request_v1",
        "episode_id": args.episode_id,
        "seed": args.seed,
        "room_id": resources.room_id,
        "room_catalog": str(room_pool_path),
        "room_pool": str(room_pool_path),
        "source_registry": str(Path(args.source_registry).expanduser().resolve()),
        "source_asset_ids": list(args.source_asset_ids),
        "sound_pool": str(Path(args.sound_pool).expanduser().resolve()),
        "resolved_sound_pool": str((output / "source_pool.json").resolve()),
        "qa_ids": list(plan["request"]["qa_ids"]),
        "activity": "walking",
        "camera_motion": args.camera_motion,
        "audio_mode": args.audio_mode,
        "frame_count": args.frame_count,
        "frame_rate_hz": args.frame_rate_hz,
        "sample_rate_hz": args.sample_rate_hz,
        "runtime": {
            "uproject": str(args.uproject),
            "unreal_editor": str(args.unreal_editor),
            "spear_ext_dir": str(args.spear_ext_dir),
            "graphics_adapter": args.graphics_adapter,
            "rpc_port": args.rpc_port,
            "streaming_warmup_frames": args.streaming_warmup_frames,
            "ddc_profile": args.ddc_profile,
            "ddc_directory": str(args.ddc_directory),
        },
        "native_input_root": str(resources.source_root),
        "native_room_profile": (
            str(resources.room_profile_path)
            if resources.room_profile_path is not None
            else None
        ),
        "native_stage_source": (
            str(args.stage_source.expanduser().resolve())
            if args.stage_source is not None
            else None
        ),
        "native_stage_target": (
            str(args.stage_target.expanduser().resolve())
            if args.stage_target is not None
            else None
        ),
    }
    request_path = write_json(output / "request.json", request)
    capture_dir = output / "capture"
    capture = _capture_command(
        plan_dir=plan_dir,
        output_dir=capture_dir,
        runtime_python=args.runtime_python,
        uproject=args.uproject,
        unreal_editor=args.unreal_editor,
        spear_ext_dir=args.spear_ext_dir,
        graphics_adapter=args.graphics_adapter,
        rpc_port=args.rpc_port,
        streaming_warmup_frames=args.streaming_warmup_frames,
        ddc_profile=args.ddc_profile,
        ddc_directory=args.ddc_directory,
    )
    capture_path = write_json(output / "capture_command.json", {"capture": capture})
    result = {
        "schema": "avengine_native_qa_room_plan_result_v1",
        "status": "research_candidate",
        "episode_id": args.episode_id,
        "room_id": resources.room_id,
        "plan": str(plan_path),
        "room_pool": str(room_pool_path),
        "request": str(request_path),
        "capture_command": str(capture_path),
        "native_execution": "not_run",
        "native_input_root": str(resources.source_root),
        "native_room_profile": (
            str(resources.room_profile_path)
            if resources.room_profile_path is not None
            else None
        ),
        "stage_readback": str(output / "stage_readback.json") if stage_readback is not None else None,
        "route_selection": plan["activity_plan"],
        "camera_selection": plan["camera_condition_sampling"],
        "claim_boundary": (
            "CPU plan uses native Apartment mesh/object audit and native UE route bank; "
            "native RGB/depth/animation readback requires the emitted UE command"
        ),
    }
    write_json(output / "planning_result.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    def env_path(name: str) -> Path | None:
        value = os.environ.get(name)
        return Path(value) if value else None

    def env_int(name: str) -> int | None:
        value = os.environ.get(name)
        return int(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", default="qa_native_apartment_walk_001")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--frame-rate-hz", type=int, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument(
        "--source-root", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_SOURCE_ROOT"),
    )
    parser.add_argument(
        "--route-bank", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_ROUTE_BANK"),
    )
    parser.add_argument(
        "--room-profile", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_ROOM_PROFILE"),
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    parser.add_argument(
        "--sound-pool", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_SOUND_POOL"),
    )
    parser.add_argument("--source-asset-ids", nargs=2, default=list(DEFAULT_SOURCE_ASSET_IDS))
    parser.add_argument("--qa-ids", nargs="+", default=None)
    parser.add_argument("--camera-motion", choices=("static", "pan", "follow_group"), default="follow_group")
    parser.add_argument("--audio-mode", choices=("sequential", "overlap", "repeat"), default="sequential")
    parser.add_argument(
        "--start-hold-frames", type=int,
        default=env_int("AVENGINE_NATIVE_APARTMENT_START_HOLD_FRAMES") or 0,
    )
    parser.add_argument(
        "--runtime-python", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_RUNTIME_PYTHON"),
    )
    parser.add_argument(
        "--uproject", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_UPROJECT"),
    )
    parser.add_argument(
        "--unreal-editor", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_UNREAL_EDITOR"),
    )
    parser.add_argument(
        "--spear-ext-dir", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_SPEAR_EXT_DIR"),
    )
    parser.add_argument(
        "--ddc-profile",
        default=os.environ.get(
            "AVENGINE_NATIVE_APARTMENT_DDC_PROFILE",
            "InstalledNoZenLocalFallback",
        ),
    )
    parser.add_argument(
        "--ddc-directory", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_DDC_DIRECTORY"),
    )
    parser.add_argument(
        "--stage-source", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_STAGE_SOURCE"),
    )
    parser.add_argument(
        "--stage-target", type=Path,
        default=env_path("AVENGINE_NATIVE_APARTMENT_STAGE_TARGET"),
    )
    parser.add_argument(
        "--graphics-adapter", type=int,
        default=env_int("AVENGINE_NATIVE_APARTMENT_GRAPHICS_ADAPTER"),
    )
    parser.add_argument(
        "--rpc-port", type=int,
        default=env_int("AVENGINE_NATIVE_APARTMENT_RPC_PORT"),
    )
    parser.add_argument(
        "--streaming-warmup-frames", type=int,
        default=env_int("AVENGINE_NATIVE_APARTMENT_STREAMING_WARMUP_FRAMES") or 180,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = build(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
