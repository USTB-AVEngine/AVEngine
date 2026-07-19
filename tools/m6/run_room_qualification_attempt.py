#!/usr/bin/env python3
"""Run or verify the read-only M6 representative-room qualification attempt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from avengine.m6.room_attempts import (
    run_room_qualification_attempt,
    verify_room_qualification_attempt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="publish one immutable attempt bundle")
    run.add_argument("--repository-root", type=Path, default=Path.cwd())
    run.add_argument(
        "--registry",
        type=Path,
        default=Path("examples/m6/rooms/room_registry.json"),
    )
    run.add_argument(
        "--corrupted-fixture",
        type=Path,
        default=Path("tests/fixtures/m6/corrupted_acoustic_package/fixture.json"),
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--attempt-id", default="m6_representative_rooms_current_attempt_v1"
    )
    run.add_argument("--custom-package-manifest", type=Path)
    run.add_argument("--custom-m5-evidence", type=Path)
    run.add_argument("--legacy-package-manifest", type=Path)
    run.add_argument("--legacy-delivery-evidence", type=Path)
    run.add_argument("--mp3d-raw-package-manifest", type=Path)
    run.add_argument("--mp3d-derived-package-manifest", type=Path)
    run.add_argument("--replicacad-root", type=Path)
    run.add_argument("--legacy-export-root", type=Path)
    run.add_argument("--legacy-package-root", type=Path)
    run.add_argument("--habitat-runtime-root", type=Path)
    run.add_argument("--mp3d-proxy-root", type=Path)

    verify = subparsers.add_parser("verify", help="rehash and validate a bundle")
    verify.add_argument("manifest", type=Path)
    return parser


def _environment(arguments: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    overrides = {
        "AVENGINE_REPLICACAD_ROOT": arguments.replicacad_root,
        "AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT": arguments.legacy_export_root,
        "AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT": arguments.legacy_package_root,
        "AVENGINE_HABITAT_RUNTIME_ROOT": arguments.habitat_runtime_root,
        "AVENGINE_MP3D_PROXY_V2_ROOT": arguments.mp3d_proxy_root,
    }
    for name, value in overrides.items():
        if value is not None:
            environment[name] = str(value.resolve(strict=True))
    return environment


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "verify":
        status, checks = verify_room_qualification_attempt(arguments.manifest)
        print(json.dumps({"status": status, "checks": checks}, indent=2))
        return 0 if status == "pass" else 1

    root = arguments.repository_root.resolve(strict=True)
    manifest = run_room_qualification_attempt(
        registry_path=arguments.registry,
        corrupted_fixture_path=arguments.corrupted_fixture,
        output_directory=arguments.output,
        repository_root=root,
        environment=_environment(arguments),
        custom_package_manifest=arguments.custom_package_manifest,
        custom_m5_evidence=arguments.custom_m5_evidence,
        legacy_package_manifest=arguments.legacy_package_manifest,
        legacy_delivery_evidence=arguments.legacy_delivery_evidence,
        mp3d_raw_package_manifest=arguments.mp3d_raw_package_manifest,
        mp3d_derived_package_manifest=arguments.mp3d_derived_package_manifest,
        attempt_id=arguments.attempt_id,
    )
    status, checks = verify_room_qualification_attempt(manifest)
    print(
        json.dumps(
            {"status": status, "manifest": str(manifest), "checks": checks},
            indent=2,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
