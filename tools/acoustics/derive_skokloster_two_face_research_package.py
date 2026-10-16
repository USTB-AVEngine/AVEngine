#!/usr/bin/env python3
"""Derive the bounded Skokloster package by removing exactly two QA faces.

This deliberately does *not* claim native RLR compatibility.  It reuses the
atomic AVEngine research-package writer while narrowing its filter to the M3
geometry-QA area rule.  A separate native RLR load/simulation test is the
fail-closed admission gate; remaining native-RLR-small faces are not silently
removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.acoustics import research_cleanup

EXPECTED_REMOVED_TRIANGLES = [251199, 288544]
POLICY = "skokloster_remove_exactly_two_geometry_qa_degenerate_faces_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    # Narrow the existing writer's selection to the geometry-QA area rule.
    # The writer still preserves all provenance, array records, object ranges,
    # atomic publication, and package self-validation.  Zero here means any
    # positive cross product remains in the package for the independent native
    # RLR gate to accept or reject.
    research_cleanup.CLEANUP_POLICY = POLICY
    research_cleanup.DERIVED_PACKAGE_SUFFIX = "two_qa_faces_removed_v1"
    research_cleanup.RLR_MIN_CROSS_NORM_SQUARED = 0.0
    manifest_path = research_cleanup.derive_rlr_compatible_research_package(
        args.source_manifest, args.output_dir
    )
    manifest = load_json(manifest_path)
    parity_path = (
        manifest_path.parent
        / manifest["qa"]["compiler_source_to_package_parity"]["path"]
    )
    cleanup = load_json(parity_path)["research_cleanup"]
    _require(cleanup["policy"] == POLICY, "cleanup policy drift")
    _require(
        cleanup["removed_triangle_indices"] == EXPECTED_REMOVED_TRIANGLES,
        "cleanup did not remove exactly the two reviewed Skokloster faces",
    )
    _require(cleanup["removed_triangle_count"] == 2, "cleanup count drift")
    _require(cleanup["removed_vertex_count"] == 0, "cleanup removed vertices")
    _require(
        cleanup["source_triangle_count"] - cleanup["derived_triangle_count"] == 2,
        "derived triangle count drift",
    )
    print(
        json.dumps(
            {
                "status": "two_face_package_derived",
                "manifest": str(manifest_path),
                "policy": cleanup["policy"],
                "removed_triangle_indices": cleanup["removed_triangle_indices"],
                "derived_triangle_count": cleanup["derived_triangle_count"],
                "native_rlr_admission_claim": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
