#!/usr/bin/env python3
"""Compile a Kujiale USD room into an RLR-loadable research acoustic package.

Three stages in one order: extract the composed USD stage into an auditable
NPZ snapshot (the only step that needs Pixar USD), compile the snapshot's
material and object identities through the shared residential rules into the
standard M3 package, then derive the RLR-loadable variant with QA-degenerate
faces removed. The derivation step is not optional politeness: kujiale_0020
carries 342 zero-area triangles out of 1.89 million, and the native runtime
refuses the upload outright until they are gone.

The receipt records both manifests. Downstream RIR simulation loads the
derived one; the underived package remains the auditable source of what the
USD actually said.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", required=True, type=Path)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--transform-profile", required=True)
    parser.add_argument(
        "--interior-origin",
        nargs=3,
        action="append",
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--material-rules", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists (fresh/no-clobber): {output}")
    output.mkdir(parents=True)

    snapshot_dir = output / "snapshot"
    extract_argv = [
        sys.executable,
        str(REPOSITORY / "tools/acoustics/extract_usd_acoustic_snapshot.py"),
        "--source", str(args.source_usd),
        "--output", str(snapshot_dir),
        "--room-id", args.room_id,
        "--transform-profile", args.transform_profile,
        "--source-revision", args.source_revision,
        "--dataset-id", args.dataset_id,
        "--source-license", args.source_license,
    ]
    for origin in args.interior_origin:
        extract_argv += ["--interior-origin", *[str(value) for value in origin]]
    print("=== extract:", " ".join(extract_argv), flush=True)
    completed = subprocess.run(extract_argv, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"extraction failed with exit code {completed.returncode}")

    from avengine.acoustics.compiler import (
        compile_usd_snapshot_semantic_research_scene,
    )
    from avengine.acoustics.research_cleanup import (
        derive_rlr_compatible_research_package,
    )

    print("=== compile", flush=True)
    manifest_path, report_path = compile_usd_snapshot_semantic_research_scene(
        room_manifest=snapshot_dir / "room_manifest.json",
        material_rules=args.material_rules,
        output=output / "package",
        seed=args.seed,
    )
    print("=== derive rlr-loadable", flush=True)
    derived_manifest = derive_rlr_compatible_research_package(
        manifest_path, output / "package_rlr"
    )

    cleanup = json.loads(
        (
            derived_manifest.parent
            / json.loads(derived_manifest.read_text(encoding="utf-8"))["qa"][
                "compiler_source_to_package_parity"
            ]["path"]
        ).read_text(encoding="utf-8")
    )["research_cleanup"]

    receipt = {
        "schema": "avengine_kujiale_acoustic_package_receipt_v1",
        "room_id": args.room_id,
        "source_usd": str(args.source_usd.resolve()),
        "snapshot": str(snapshot_dir / "scene_snapshot.npz"),
        "package_manifest": str(manifest_path),
        "coverage_report": str(report_path),
        "rlr_manifest": str(derived_manifest),
        "removed_degenerate_triangles": cleanup["removed_triangle_count"],
        "derived_triangle_count": cleanup["derived_triangle_count"],
        "simulation_note": (
            "simulate with the rlr_manifest; the underived package is the "
            "auditable record of the source geometry"
        ),
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"receipt": str(output / "receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
