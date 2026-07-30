#!/usr/bin/env python3
"""Batch Habitat-native RGB rendering for registry-selected rooms.

This is the Habitat-native counterpart of the SPEAR/UE batch runner: it wraps
the reviewed single-episode capture path in a batching shell with fixed
disjoint sharding, resumable execution and an artifact-level readback
contract. The first supported input layout is ``m5_1-mixed-route`` (one route
manifest per episode, human plus Beagle actors), matching the retained M5.1
MP3D visual gate. Rooms are selected through the room runtime profile
registry; only ``habitat_native`` profiles are accepted.

Every episode's retained gate evidence must independently read back as
``pass``; a completed episode is only skipped on resume after that readback
succeeds again. The batch manifest never claims dataset admission and stays
``research_candidate``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import (  # noqa: E402
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m1.contracts import validate_room_manifest  # noqa: E402
from avengine.m6.rooms import load_room_registry  # noqa: E402
from avengine.runtime_profiles import (  # noqa: E402
    load_room_runtime_profile_registry,
)

BATCH_SCHEMA = "avengine_m7_habitat_room_batch_v1"
SUPPORTED_LAYOUT = "m5_1-mixed-route"
GATE_EVIDENCE_NAME = "mp3d_gate_evidence.json"
DEFAULT_ROOM_REGISTRY = REPOSITORY / "examples/m6/rooms/room_registry.json"


class HabitatRoomBatchError(RuntimeError):
    pass


def _parse_episodes(raw_values: list[str]) -> list[tuple[str, Path]]:
    episodes: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in raw_values:
        key, separator, value = raw.partition("=")
        if separator != "=" or not key or not value:
            raise HabitatRoomBatchError(
                f"--episode must look like episode_id=route_manifest.json: {raw!r}"
            )
        if key in seen:
            raise HabitatRoomBatchError(f"duplicate episode id {key!r}")
        seen.add(key)
        episodes.append((key, Path(value).resolve()))
    return episodes


def _select_profile(registry_path: Path, profile_id: str) -> Mapping[str, Any]:
    registry = load_room_runtime_profile_registry(registry_path)
    for profile in registry["profiles"]:
        if profile["profile_id"] == profile_id:
            if profile["backend_id"] != "habitat_native":
                raise HabitatRoomBatchError(
                    f"profile {profile_id!r} uses backend "
                    f"{profile['backend_id']!r}; this runner only accepts "
                    "habitat_native profiles"
                )
            if SUPPORTED_LAYOUT not in profile["supported_input_layouts"]:
                raise HabitatRoomBatchError(
                    f"profile {profile_id!r} does not declare input layout "
                    f"{SUPPORTED_LAYOUT!r}"
                )
            return profile
    raise HabitatRoomBatchError(f"room profile {profile_id!r} not found")


def _resolve_acoustic_selection_binding(
    *,
    room_profile: Mapping[str, Any],
    room_registry_path: Path,
    acoustic_profile_registry_path: Path | None,
    simulation_profile: str,
) -> dict[str, Any]:
    """Bind the room's acoustic identity without executing audio in this runner."""

    from avengine.acoustic_profiles import (  # noqa: PLC0415
        load_acoustic_profile_registry,
        load_default_acoustic_profile_registry,
        resolve_acoustic_profile,
    )
    from avengine.m6.rooms import load_room_registry  # noqa: PLC0415

    acoustic_registry = (
        load_acoustic_profile_registry(acoustic_profile_registry_path)
        if acoustic_profile_registry_path is not None
        else load_default_acoustic_profile_registry()
    )
    room_registry_path = room_registry_path.resolve()
    room_registry = load_room_registry(room_registry_path)
    room_ref = room_profile["room_ref"]
    selection = resolve_acoustic_profile(
        acoustic_registry,
        room_registry,
        room_ref,
        repository_root=REPOSITORY,
        verify_paths=False,
    )
    simulation_request = Path(selection.simulation_path(simulation_profile)).resolve()
    return {
        "schema": "avengine_m7_acoustic_selection_binding_v1",
        "room_ref": dict(room_ref),
        "simulation_profile": simulation_profile,
        "acoustic_package_use": "bound_for_downstream_rir_not_consumed_here",
        "selected_simulation_request": {
            "path": str(simulation_request),
            "byte_size": simulation_request.stat().st_size,
            "sha256": sha256_file(simulation_request),
        },
        "acoustic_profile_selection": selection.receipt(simulation_profile),
    }


def _verify_room_manifest_binding(
    *,
    room_profile: Mapping[str, Any],
    room_registry_path: Path,
    room_manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed when the visual manifest differs from the selected room."""

    room_ref = room_profile["room_ref"]
    registry_path = room_registry_path.resolve()
    manifest_path = room_manifest_path.resolve()
    try:
        registry = load_room_registry(registry_path)
    except (OSError, ValueError) as error:
        raise HabitatRoomBatchError(
            f"room registry is not valid: {registry_path}: {error}"
        ) from error
    if registry.get("registry_id") != room_ref.get("registry_id"):
        raise HabitatRoomBatchError(
            "room profile room_ref.registry_id does not match the room registry"
        )
    records = [
        record
        for record in registry["records"]
        if record["room_id"] == room_ref.get("room_id")
        and record["revision"] == room_ref.get("revision")
    ]
    if len(records) != 1:
        raise HabitatRoomBatchError(
            "room profile room_ref does not resolve exactly one room registry record"
        )
    record = records[0]

    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError) as error:
        raise HabitatRoomBatchError(
            f"room manifest is not readable JSON: {manifest_path}: {error}"
        ) from error
    manifest_errors = validate_room_manifest(manifest)
    if manifest_errors:
        raise HabitatRoomBatchError(
            "room manifest contract failed: " + "; ".join(manifest_errors)
        )
    if manifest["room_id"] != room_ref["room_id"]:
        raise HabitatRoomBatchError(
            "room manifest room_id does not match the exact room profile room_ref"
        )

    actual_sha256 = sha256_file(manifest_path)
    declared_resources = [
        resource
        for resource in record["resources"]
        if resource["resource_type"] == "room_manifest"
    ]
    if len(declared_resources) != 1:
        raise HabitatRoomBatchError(
            "exact room registry record must declare exactly one sha256-bound "
            "room_manifest resource"
        )
    declared_resource = declared_resources[0]
    expected_sha256 = declared_resource.get("sha256")
    if not isinstance(expected_sha256, str):
        raise HabitatRoomBatchError(
            "declared room_manifest resource has no sha256 for fail-closed "
            "visual room verification"
        )
    if actual_sha256 != expected_sha256:
        raise HabitatRoomBatchError(
            "room manifest sha256 does not match the exact room registry record: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    registry_hash_verification = {
        "status": "pass",
        "resource_id": declared_resource["resource_id"],
        "declared_sha256": expected_sha256,
        "observed_sha256": actual_sha256,
    }

    return {
        "schema": "avengine_m7_room_manifest_binding_v1",
        "status": "pass",
        "room_ref": dict(room_ref),
        "room_manifest": {
            "path": str(manifest_path),
            "byte_size": manifest_path.stat().st_size,
            "sha256": actual_sha256,
            "room_id": manifest["room_id"],
        },
        "checks": {
            "manifest_contract": "pass",
            "room_ref_registry_record": "pass",
            "room_id_matches_room_ref": "pass",
            "registry_declared_hash": registry_hash_verification,
        },
    }


def _episode_readback(episode_dir: Path) -> dict[str, Any]:
    """Independently re-verify one completed episode's retained evidence."""

    evidence_path = episode_dir / GATE_EVIDENCE_NAME
    if not evidence_path.is_file():
        raise HabitatRoomBatchError(f"missing gate evidence: {evidence_path}")
    evidence = load_json(evidence_path)
    if evidence.get("status") != "pass":
        raise HabitatRoomBatchError(
            f"gate evidence status is {evidence.get('status')!r}: {evidence_path}"
        )
    gate_statuses = {
        str(gate.get("gate_id")): str(gate.get("status"))
        for gate in evidence.get("gates", [])
    }
    failing = sorted(
        gate_id for gate_id, status in gate_statuses.items() if status != "pass"
    )
    if failing:
        raise HabitatRoomBatchError(
            f"gates not passing in {evidence_path}: {failing}"
        )
    return {
        "gate_evidence": file_record(evidence_path, relative_to=episode_dir),
        "gate_count": len(gate_statuses),
        "frame_count": evidence.get("frame_count"),
        "frame_rate_hz": evidence.get("frame_rate_hz"),
        "route_id": evidence.get("route_id"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-runtime-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/room_runtime_profiles.json",
    )
    parser.add_argument("--room-profile", required=True)
    parser.add_argument(
        "--room-registry",
        type=Path,
        default=DEFAULT_ROOM_REGISTRY,
    )
    parser.add_argument(
        "--acoustic-profile-registry",
        type=Path,
        help="Override the installed/default acoustic profile registry",
    )
    parser.add_argument(
        "--simulation-profile",
        choices=("production", "reference"),
        default="production",
        help="Acoustic simulation request bound into the batch manifest",
    )
    parser.add_argument("--input-layout", default=SUPPORTED_LAYOUT)
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="episode_id=route_manifest.json; repeat once per episode",
    )
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--human-runtime-glb", type=Path, required=True)
    parser.add_argument("--beagle-manifest", type=Path, required=True)
    parser.add_argument("--beagle-m2-request", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-render episodes even when retained evidence reads back as pass",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.input_layout != SUPPORTED_LAYOUT:
        raise HabitatRoomBatchError(
            f"unsupported input layout {args.input_layout!r}; "
            f"this runner currently implements {SUPPORTED_LAYOUT!r}"
        )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise HabitatRoomBatchError(
            "sharding requires shard_count >= 1 and 0 <= shard_index < shard_count"
        )

    registry_path = args.room_runtime_registry.resolve()
    profile = _select_profile(registry_path, args.room_profile)
    room_manifest_path = args.room_manifest.resolve()
    room_manifest_binding = _verify_room_manifest_binding(
        room_profile=profile,
        room_registry_path=args.room_registry,
        room_manifest_path=room_manifest_path,
    )
    acoustic_selection = _resolve_acoustic_selection_binding(
        room_profile=profile,
        room_registry_path=args.room_registry,
        acoustic_profile_registry_path=args.acoustic_profile_registry,
        simulation_profile=args.simulation_profile,
    )
    episodes = _parse_episodes(args.episode)
    selected = [
        (ordinal, episode_id, route_path)
        for ordinal, (episode_id, route_path) in enumerate(episodes)
        if ordinal % args.shard_count == args.shard_index
    ]

    output = args.output.resolve()
    episodes_root = output / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)

    # Imported lazily so --help and sharding dry checks stay usable without
    # the native Habitat runtime installed.
    from avengine.m5_1.mp3d_capture import capture_mp3d_route  # noqa: PLC0415

    entries: list[dict[str, Any]] = []
    render_contract = dict(profile["render"])
    for ordinal, episode_id, route_path in selected:
        episode_dir = episodes_root / episode_id
        resumed = False
        if episode_dir.exists() and not args.no_resume:
            try:
                readback = _episode_readback(episode_dir)
                resumed = True
            except HabitatRoomBatchError:
                raise HabitatRoomBatchError(
                    f"episode {episode_id!r} exists but does not read back as "
                    "pass; move it aside or rerun with --no-resume"
                ) from None
        if not resumed:
            if episode_dir.exists():
                raise HabitatRoomBatchError(
                    f"--no-resume refuses to overwrite existing {episode_dir}"
                )
            started = time.monotonic()
            capture_mp3d_route(
                route_manifest_path=route_path,
                room_manifest_path=room_manifest_path,
                m1_request_path=args.m1_request,
                human_runtime_glb_path=args.human_runtime_glb,
                beagle_animal_manifest_path=args.beagle_manifest,
                beagle_m2_request_path=args.beagle_m2_request,
                output_dir=episode_dir,
                runtime_root=args.runtime_root,
            )
            wall_seconds = time.monotonic() - started
            readback = _episode_readback(episode_dir)
            readback["capture_wall_seconds"] = round(wall_seconds, 3)
        divergent = readback.get("frame_count") != render_contract["frame_count"]
        entries.append(
            {
                "episode_id": episode_id,
                "ordinal": ordinal,
                "route_manifest": str(route_path),
                "route_manifest_sha256": sha256_file(route_path),
                "resumed": resumed,
                "readback": readback,
                "frame_count_matches_profile_contract": not divergent,
                "episode_role": (
                    "dataset_contract_candidate" if not divergent else "review_only"
                ),
            }
        )
        print(
            f"HABITAT_BATCH_EPISODE_OK id={episode_id} resumed={resumed} "
            f"frames={readback.get('frame_count')} gates={readback.get('gate_count')}"
        )

    manifest = {
        "schema": BATCH_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Batch shell readback over reviewed single-episode Habitat "
            "captures; no room, asset or dataset admission is granted"
        ),
        "input_layout": SUPPORTED_LAYOUT,
        "room_profile": {
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "profile_id": profile["profile_id"],
            "revision": profile["revision"],
            "backend_id": profile["backend_id"],
            "render_contract": render_contract,
        },
        "room_manifest_binding": room_manifest_binding,
        "acoustic_selection": acoustic_selection,
        "shard": {"count": args.shard_count, "index": args.shard_index},
        "episode_total": len(episodes),
        "episode_selected": len(selected),
        "episodes": entries,
    }
    write_json(output / "batch_manifest.json", manifest)
    print(
        f"HABITAT_BATCH_OK output={output} selected={len(selected)} "
        f"of {len(episodes)} episodes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
