#!/usr/bin/env python3
"""Compile one room's semantic mesh into an M3/RLR research acoustic package.

One entry point for every semantic source the compiler understands, chosen by
what the room manifest actually declares rather than by the caller's memory.
A ``semantic_surface_mesh`` ending in ``.glb`` is HM3D's painted vertex
colours keyed by a ``.semantic.txt``; one ending in ``.ply`` is Matterport3D's
``object_id`` column paired with a ``.house`` descriptor. The compiled package
layout is identical either way, which is the point: downstream RIR simulation
should not know which dataset a room came from.

Declared asset paths may reference ``${AVENGINE_HM3D_ROOT}`` or
``${AVENGINE_MP3D_ROOT}``; the two ``--*-root`` flags define those variables
for this process, so a Studio task does not depend on the server's shell
environment being configured just so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.acoustics.compiler import (  # noqa: E402
    AcousticSceneCompileError,
    compile_hm3d_semantic_research_scene,
    compile_mp3d_semantic_research_scene,
)


def declared_semantic_suffix(room_manifest: Path) -> str:
    """Classify the room by its declared semantic asset, without resolving it.

    Detection reads the declared string rather than the resolved file because
    resolution needs the environment variables to be right, and a wrong
    environment should fail later with the resolver's own message, not here
    with a misleading "unknown dataset".
    """

    room = json.loads(room_manifest.read_text(encoding="utf-8"))
    for asset in room.get("assets", []):
        if asset.get("role") == "semantic_surface_mesh":
            declared = str(asset.get("path", ""))
            if declared.endswith(".glb"):
                return "hm3d"
            if declared.endswith(".ply"):
                return "mp3d"
            raise SystemExit(
                f"semantic_surface_mesh {declared!r} is neither a painted "
                ".glb (HM3D) nor a .ply (MP3D); no compiler claims it"
            )
    raise SystemExit(f"{room_manifest} declares no semantic_surface_mesh asset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--material-rules", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--package-id")
    parser.add_argument(
        "--hm3d-root", type=Path, help="defines ${AVENGINE_HM3D_ROOT} for this run"
    )
    parser.add_argument(
        "--mp3d-root", type=Path, help="defines ${AVENGINE_MP3D_ROOT} for this run"
    )
    args = parser.parse_args()

    if args.hm3d_root is not None:
        os.environ["AVENGINE_HM3D_ROOT"] = str(args.hm3d_root.resolve())
    if args.mp3d_root is not None:
        os.environ["AVENGINE_MP3D_ROOT"] = str(args.mp3d_root.resolve())

    source_kind = declared_semantic_suffix(args.room_manifest.resolve())
    compile_scene = (
        compile_hm3d_semantic_research_scene
        if source_kind == "hm3d"
        else compile_mp3d_semantic_research_scene
    )
    try:
        manifest_path, report_path = compile_scene(
            room_manifest=args.room_manifest,
            material_rules=args.material_rules,
            output=args.output,
            seed=args.seed,
            package_id=args.package_id,
        )
    except AcousticSceneCompileError as error:
        raise SystemExit(f"compilation failed: {error}") from error

    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report.get("category_triangle_counts", {})
    total = sum(counts.values()) or 1
    unannotated = counts.get("unannotated", 0)

    # The sideways-package incident in one lesson: the QA that would have
    # caught it was written into the package and never read, because the
    # research escape hatch silences geometry failures wholesale. So the
    # verdicts are surfaced here, in the one place the operator always looks.
    # geometry_status=fail is EXPECTED for scan meshes (open seams fail the
    # production watertight bar); the discriminating number is the leakage
    # escape fraction - probes outside the geometry push it toward one.
    qa_dir = manifest_path.parent / "qa"
    geometry_status = None
    worst_escape = None
    geometry_qa = qa_dir / "geometry_report.json"
    if geometry_qa.is_file():
        geometry_status = json.loads(geometry_qa.read_text(encoding="utf-8")).get(
            "status"
        )
    leakage_qa = qa_dir / "ray_leakage.json"
    if leakage_qa.is_file():
        fractions: list[float] = []

        def _collect(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "escape_fraction" and isinstance(value, (int, float)):
                        fractions.append(float(value))
                    else:
                        _collect(value)
            elif isinstance(node, list):
                for item in node:
                    _collect(item)

        _collect(json.loads(leakage_qa.read_text(encoding="utf-8")))
        worst_escape = max(fractions) if fractions else None

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "qa_geometry_status": geometry_status,
                "qa_worst_probe_escape_fraction": worst_escape,
                "qa_note": (
                    "scan meshes fail the watertight bar by design; an escape "
                    "fraction near 1 means the leakage probes are outside the "
                    "geometry - check the frame before anything else"
                ),
                "coverage_report": str(report_path),
                "source_kind": source_kind,
                "room_id": report.get("room_id"),
                "compiled_triangle_count": report.get("compiled_triangle_count"),
                "surface_count": report.get("surface_count"),
                "categories_on_default_material": report.get(
                    "unknown_semantic_category_count"
                ),
                "unannotated_triangle_share": round(unannotated / total, 4),
                "semantic_source_defects": report.get(
                    "semantic_source_defects", []
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
