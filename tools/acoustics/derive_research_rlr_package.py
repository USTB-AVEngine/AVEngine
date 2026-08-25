#!/usr/bin/env python3
"""Derive an RLR-loadable research package by removing QA-degenerate faces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.acoustics.research_cleanup import derive_rlr_compatible_research_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = derive_rlr_compatible_research_package(
        args.source_manifest, args.output_dir
    )
    manifest = load_json(manifest_path)
    cleanup = load_json(
        manifest_path.parent
        / manifest["qa"]["compiler_source_to_package_parity"]["path"]
    )["research_cleanup"]
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "package_mode": manifest["package_mode"],
                "material_semantics": manifest["materials"]["material_semantics"],
                "removed_triangle_count": cleanup["removed_triangle_count"],
                "removed_vertex_count": cleanup["removed_vertex_count"],
                "derived_triangle_count": cleanup["derived_triangle_count"],
                "record_content_sha256": cleanup["record_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
