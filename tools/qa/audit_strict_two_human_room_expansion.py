#!/usr/bin/env python3
# HISTORICAL TOOL (single-repo closure, 2026-08-21): this script built or
# validates retained strict-two-human evidence recorded against the
# pre-closure transition environment (sibling Habitat fork, sound-spaces,
# SPEAR-lead-b, and multi-repo SPEAR checkouts). The hard-coded absolute
# paths below are a frozen historical record, not current inputs. The current
# production chain runs on the installed runtime prefix and external data
# roots under /data/avengine_external; do not use this tool for new work.
"""Audit two additional cooked SPEAR maps for the strict M/F/C room closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_MANIFEST = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
    "lead_b_siamese_post_approval_v1/packaged_runtime_v1/"
    "Standalone-Development/Linux/Manifest_UFSFiles_Linux.txt"
)
DEFAULT_RUNTIME_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
DEFAULT_ROOM_PROFILES = REPOSITORY / "examples/runtime/room_runtime_profiles.json"
DEFAULT_ACOUSTIC_REGISTRY = REPOSITORY / "examples/runtime/acoustic_profiles.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_fragment(path: str) -> str:
    return path.split(".", 1)[0].lower().replace("/game/", "spearsim/content/")


def _identity_assets(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aliases = {
        "M": "legacy_human",
        "F": "strict_two_human_female_adult",
        "C": "strict_two_human_construction_male",
    }
    by_id = {item["asset_id"]: item for item in registry["assets"]}
    result: dict[str, dict[str, Any]] = {}
    for key, alias in aliases.items():
        ref = registry["aliases"][alias]
        asset = by_id[ref["asset_id"]]
        unreal = asset["runtime_backends"]["spear_unreal"]
        result[key] = {
            "identity_id": asset["asset_id"],
            "revision": asset["revision"],
            "required_objects": [
                unreal["blueprint_class_path"],
                unreal["idle_animation"],
                unreal["walking_animation"],
            ],
        }
    return result


def audit(
    *,
    package_manifest: Path,
    runtime_registry: Path,
    room_profiles: Path,
    acoustic_registry: Path,
    output: Path,
) -> Path:
    package_text = package_manifest.read_text(encoding="utf-8").lower()
    identities = _identity_assets(_load(runtime_registry))
    identity_rows: dict[str, Any] = {}
    for key, record in identities.items():
        missing = [
            path
            for path in record["required_objects"]
            if _package_fragment(path) not in package_text
        ]
        identity_rows[key] = {
            "identity_id": record["identity_id"],
            "revision": record["revision"],
            "required_object_count": len(record["required_objects"]),
            "cooked_object_count": len(record["required_objects"]) - len(missing),
            "missing_objects": missing,
            "status": "pass" if not missing else "fail",
        }
    all_identities_cooked = all(item["status"] == "pass" for item in identity_rows.values())
    profile_ids = {item["profile_id"] for item in _load(room_profiles)["profiles"]}
    acoustic_text = json.dumps(_load(acoustic_registry), sort_keys=True)
    candidates = [
        {
            "candidate_id": "spear_debug_0000",
            "map_path": "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000",
            "package_fragment": "spearsim/content/spear/scenes/debug_0000/maps/debug_0000.umap",
        },
        {
            "candidate_id": "spear_debug_0001",
            "map_path": "/Game/SPEAR/Scenes/debug_0001/Maps/debug_0001",
            "package_fragment": "spearsim/content/spear/scenes/debug_0001/maps/debug_0001.umap",
        },
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        map_cooked = candidate["package_fragment"] in package_text
        runtime_profile_present = candidate["candidate_id"] in profile_ids
        acoustic_profile_present = candidate["candidate_id"] in acoustic_text
        rows.append(
            {
                **candidate,
                "map_cooked_in_current_package": map_cooked,
                "same_M_F_C_runtime_objects_cooked": all_identities_cooked,
                "capture_adapter_static_capability": {
                    "status": "code_path_present_pending_native_smoke",
                    "evidence": "tools/qa/capture_spear_native_pixel_episode.py takes suite.native_map and records RGB, metric depth, two target-only passes, and live asset readback.",
                },
                "room_runtime_profile_present": runtime_profile_present,
                "native_floor_or_navmesh_evidence_present": False,
                "exact_acoustic_profile_present": acoustic_profile_present,
                "exact_rir_closure_present": False,
                "residential_room_claim": False,
                "ready_for_strict_full75": False,
                "minimum_work_to_ready": [
                    "native smoke-load the cooked map on physical GPU1 and inventory persistent level geometry",
                    "define a real room_id/runtime profile; do not reuse the map name as a room claim without bounds",
                    "capture or derive authoritative floor/navmesh and collision-safe camera/actor placements",
                    "export or reconstruct the exact acoustic surface geometry with reviewed material mapping",
                    "register the room acoustic profile and build two exact source RIR jobs per Episode",
                    "run one sparse M/F/C two-human RGB/depth/two-target-only/live-readback canary",
                    "run one complete 75-frame canary and human visual/audio review",
                ],
            }
        )
    result = {
        "schema": "avengine_native_strict_two_human_room_expansion_audit_v1",
        "status": "additional_maps_packaged_but_not_ready_rooms",
        "claim_boundary": "debug_0000 and debug_0001 are real cooked UE maps, not residential rooms and not strict full75-ready until every listed visual, placement, and exact-acoustic gate closes.",
        "current_ready_room_count": 1,
        "final_required_ready_room_count": 3,
        "current_ready_room_id": "legacy_ue_apartment_0000_v1",
        "identity_package_readiness": identity_rows,
        "additional_map_candidates": rows,
        "additional_ready_room_count": 0,
        "final_multi_room_100_authorized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "room_expansion_audit.json"
    _write(result_path, result)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", type=Path, default=DEFAULT_PACKAGE_MANIFEST)
    parser.add_argument("--runtime-registry", type=Path, default=DEFAULT_RUNTIME_REGISTRY)
    parser.add_argument("--room-profiles", type=Path, default=DEFAULT_ROOM_PROFILES)
    parser.add_argument("--acoustic-registry", type=Path, default=DEFAULT_ACOUSTIC_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        package_manifest=args.package_manifest.resolve(),
        runtime_registry=args.runtime_registry.resolve(),
        room_profiles=args.room_profiles.resolve(),
        acoustic_registry=args.acoustic_registry.resolve(),
        output=args.output.resolve(),
    )
    print(f"STRICT_TWO_HUMAN_ROOM_EXPANSION_AUDIT_OK report={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
