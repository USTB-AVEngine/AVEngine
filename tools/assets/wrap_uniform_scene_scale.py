#!/usr/bin/env python3
"""Wrap every root of one GLB scene in an explicit uniform-scale node.

Use this only as the first half of a physical-unit normalization: the output
must immediately pass through ``bake_uniform_skin_scale.py`` so Habitat never
receives an external scale.  Binary payload data are unchanged by this step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.assets.glb import load_glb, parse_glb
from avengine.assets.glb_write import build_glb


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def wrap(source_path: Path, output_path: Path, factor: float) -> dict[str, Any]:
    if not math.isfinite(factor) or factor <= 0.0 or math.isclose(factor, 1.0):
        raise ValueError("factor must be finite, positive, and non-unit")
    source = load_glb(source_path)
    document = copy.deepcopy(source.json)
    scenes = document.get("scenes")
    nodes = document.get("nodes")
    scene_index = document.get("scene", 0)
    if (
        not isinstance(scenes, list)
        or len(scenes) != 1
        or not isinstance(nodes, list)
        or not nodes
        or scene_index != 0
        or not isinstance(scenes[0], dict)
    ):
        raise ValueError("input must contain exactly one default scene")
    roots = scenes[0].get("nodes")
    if (
        not isinstance(roots, list)
        or not roots
        or len(set(roots)) != len(roots)
        or any(
            not isinstance(index, int) or not 0 <= index < len(nodes) for index in roots
        )
    ):
        raise ValueError("default-scene root list is invalid")
    parented = {
        child
        for node in nodes
        if isinstance(node, dict)
        for child in node.get("children", [])
    }
    if any(root in parented for root in roots):
        raise ValueError("a declared scene root already has a parent")
    wrapper_index = len(nodes)
    nodes.append(
        {
            "name": "AVEnginePhysicalUnitScale",
            "children": list(roots),
            "scale": [factor, factor, factor],
        }
    )
    scenes[0]["nodes"] = [wrapper_index]
    payload = build_glb(document, source.binary)
    readback = parse_glb(payload)
    if readback.binary != source.binary:
        raise ValueError("binary payload changed")
    if readback.json["scenes"][0]["nodes"] != [wrapper_index]:
        raise ValueError("wrapper scene root failed readback")
    _write_exclusive(output_path, payload)
    return {
        "schema": "avengine_m2_uniform_scene_scale_wrapper_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_path),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "factor": factor,
        "wrapper_node_index": wrapper_index,
        "wrapped_scene_root_indices": roots,
        "binary_sha256": hashlib.sha256(source.binary).hexdigest(),
        "required_followup": "bake_uniform_skin_scale.py",
        "notes": [
            "This wrapper only declares the intended physical-unit similarity transform.",
            "The wrapped GLB is not a Habitat candidate until the scale is baked and independently verified.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source = args.input.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    if len({source, output, report}) != 3:
        parser.error("input, output, and report paths must differ")
    output_created = False
    try:
        value = wrap(source, output, args.factor)
        output_created = True
        value["report_content_sha256"] = canonical_json_sha256(value)
        _write_exclusive(
            report,
            (
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode(),
        )
    except (OSError, ValueError) as exc:
        message = str(exc)
        if output_created:
            try:
                output.unlink()
            except OSError as cleanup_exc:
                message += f"; failed to clean newly created output: {cleanup_exc}"
        parser.error(message)
    print(
        json.dumps(
            {
                "status": "pass",
                "output_sha256": sha256_file(output),
                "report_sha256": sha256_file(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
