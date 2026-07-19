#!/usr/bin/env python3
"""Build the fixed SPEAR Apartment S0--S5 M6.x review bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from avengine.m6x.canary import run_fixed_apartment_canary


REPOSITORY = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=REPOSITORY.parent / "habitat-sim-AVEngine",
    )
    parser.add_argument("--human-runtime-glb", required=True, type=Path)
    parser.add_argument("--beagle-audio", required=True, type=Path)
    parser.add_argument(
        "--review-visual-profile",
        type=Path,
        default=(
            REPOSITORY
            / "examples/m6x/fixed_apartment/review_visual_profile.json"
        ),
    )
    parser.add_argument(
        "--exterior-proxy-glb",
        type=Path,
        default=(
            REPOSITORY
            / "tmp/m6x/assets/approaching_storm_4k_exterior_v2/approaching_storm_4k_exterior.glb"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="Optional existing 270-frame M6.x master capture to reuse.",
    )
    parser.add_argument(
        "--acoustics-dir",
        type=Path,
        help="Optional completed M6.x master RIR directory to reuse.",
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default=Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"),
    )
    parser.add_argument(
        "--acoustic-package-manifest",
        type=Path,
        default=(
            REPOSITORY / "tmp/m3/root_ue_package_current_20260718_02/manifest.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_fixed_apartment_canary(
        config_root=args.config_root,
        room_manifest_path=(
            REPOSITORY / "tmp/m1/legacy_apartment_package/room_manifest.json"
        ),
        m1_request_path=(
            REPOSITORY
            / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json"
        ),
        room_registry_path=REPOSITORY / "examples/m6/rooms/room_registry.json",
        entity_registry_path=(
            REPOSITORY / "examples/m6/registries/entity_assets_v1.json"
        ),
        endpoint_registry_path=(
            REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
        ),
        sound_registry_path=(
            REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
        ),
        capture_provider_assets={
            "human_runtime_glb_path": args.human_runtime_glb,
            "animal_manifest_path": (
                REPOSITORY
                / "tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json"
            ),
            "animal_request_path": (
                REPOSITORY
                / "tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json"
            ),
            "review_visual_profile_path": args.review_visual_profile,
            "exterior_proxy_glb_path": args.exterior_proxy_glb,
        },
        external_sound_asset_paths={
            "dog_beagle_v2_scheduled_dry": args.beagle_audio,
        },
        acoustic_package_manifest_path=args.acoustic_package_manifest,
        m4_request_path=(
            REPOSITORY / "examples/m4/blender_custom/multi_source_canary_request.json"
        ),
        hrtf_file_path=args.hrtf,
        review_visual_profile_path=args.review_visual_profile,
        exterior_proxy_glb_path=args.exterior_proxy_glb,
        output_dir=args.output,
        runtime_root=args.runtime_root,
        capture_dir=args.capture_dir,
        acoustics_dir=args.acoustics_dir,
    )
    print(result.review_index)
    for path in result.videos:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
