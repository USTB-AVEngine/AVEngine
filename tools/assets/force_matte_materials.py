#!/usr/bin/env python3
"""Force a complete opaque matte-dielectric GLB material policy.

This bounded research compiler is for assets whose packed texture encodes a
visibly metallic/low-roughness surface despite factor normalization.  It
changes material JSON only and retains the embedded texture bytes for lineage.
When specular highlights are already baked into the base-color texture, the
explicit solid-coat mode can suppress that texture for diagnostic rendering.
The output is still research-only, but ``status=pass`` now means the complete
material JSON policy below passed readback; no second normalizer is required to
clear emission, transparency, unsupported extensions, or specular controls.
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


def _write(path: Path, payload: bytes) -> None:
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


def _base_color(value: Any, *, owner: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in value
        )
    ):
        raise ValueError(f"{owner} must contain four finite factors in [0, 1]")
    return [float(item) for item in value]


def _without_material_controls(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in document.items()
        if key not in {"materials", "extensionsUsed"}
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--remove-base-color-texture",
        action="store_true",
        help="Suppress a base-color texture with baked lighting artifacts",
    )
    parser.add_argument(
        "--base-color-factor",
        type=float,
        nargs=4,
        metavar=("R", "G", "B", "A"),
        help="Explicit linear RGBA factor; requires --remove-base-color-texture",
    )
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source, output, report_path = (
        args.input.resolve(),
        args.output.resolve(),
        args.report.resolve(),
    )
    if len({source, output, report_path}) != 3:
        parser.error("input, output, and report must differ")
    if args.base_color_factor is not None and not args.remove_base_color_texture:
        parser.error("--base-color-factor requires --remove-base-color-texture")
    if args.base_color_factor is not None and not all(
        0.0 <= value <= 1.0 for value in args.base_color_factor
    ):
        parser.error("--base-color-factor components must be in [0, 1]")
    output_created = False
    try:
        parsed = load_glb(source)
        document = copy.deepcopy(parsed.json)
        extensions_used = document.get("extensionsUsed", [])
        if (
            not isinstance(extensions_used, list)
            or any(not isinstance(item, str) for item in extensions_used)
            or len(set(extensions_used)) != len(extensions_used)
        ):
            raise ValueError("extensionsUsed must be a duplicate-free string array")
        if "KHR_materials_specular" not in extensions_used:
            document["extensionsUsed"] = [
                *extensions_used,
                "KHR_materials_specular",
            ]
        materials = document.get("materials")
        if not isinstance(materials, list) or not materials:
            raise ValueError("input contains no materials")
        changes = []
        for index, material in enumerate(materials):
            if not isinstance(material, dict):
                raise ValueError("material must be an object")
            extensions = material.get("extensions", {})
            if not isinstance(extensions, dict):
                raise ValueError("material.extensions must be an object")
            unsupported = set(extensions) - {"KHR_materials_specular"}
            if unsupported:
                raise ValueError(
                    f"unsupported material extensions: {sorted(unsupported)}"
                )
            pbr = material.get("pbrMetallicRoughness")
            if not isinstance(pbr, dict):
                pbr = {}
                material["pbrMetallicRoughness"] = pbr
            removed = copy.deepcopy(pbr.pop("metallicRoughnessTexture", None))
            removed_base_color = None
            if args.remove_base_color_texture:
                removed_base_color = copy.deepcopy(pbr.pop("baseColorTexture", None))
            if args.base_color_factor is not None:
                pbr["baseColorFactor"] = list(args.base_color_factor)
            base_color = _base_color(
                pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]),
                owner="baseColorFactor",
            )
            base_color[3] = 1.0
            pbr["baseColorFactor"] = base_color
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 1.0
            removed_emissive_texture = copy.deepcopy(
                material.pop("emissiveTexture", None)
            )
            material["emissiveFactor"] = [0.0, 0.0, 0.0]
            material["alphaMode"] = "OPAQUE"
            previous_extensions = copy.deepcopy(extensions)
            material["extensions"] = {
                "KHR_materials_specular": {
                    "specularFactor": 0.0,
                    "specularColorFactor": [1.0, 1.0, 1.0],
                }
            }
            changes.append(
                {
                    "material_index": index,
                    "removed_metallic_roughness_texture": removed,
                    "removed_base_color_texture": removed_base_color,
                    "base_color_factor": copy.deepcopy(base_color),
                    "metallic_factor": 0.0,
                    "roughness_factor": 1.0,
                    "alpha_mode": "OPAQUE",
                    "emissive_factor": [0.0, 0.0, 0.0],
                    "removed_emissive_texture": removed_emissive_texture,
                    "previous_extensions": previous_extensions,
                    "specular_factor": 0.0,
                }
            )
        payload = build_glb(document, parsed.binary)
        readback = parse_glb(payload)
        before_other = _without_material_controls(parsed.json)
        after_other = _without_material_controls(readback.json)
        if parsed.binary != readback.binary or before_other != after_other:
            raise ValueError("GLB content outside material controls changed")
        expected_extensions_used = (
            extensions_used
            if "KHR_materials_specular" in extensions_used
            else [*extensions_used, "KHR_materials_specular"]
        )
        if (
            readback.json.get("materials") != materials
            or readback.json.get("extensionsUsed") != expected_extensions_used
        ):
            raise ValueError("material control readback differs")
        report = {
            "schema": "avengine_m2_force_matte_materials_v2",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "material_policy_complete": True,
            "source": {
                "path": str(source),
                "sha256": parsed.sha256,
                "byte_size": parsed.byte_length,
            },
            "output": {
                "path": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            },
            "changes": changes,
            "policy": {
                "alpha_mode": "OPAQUE",
                "base_color_alpha": 1.0,
                "metallic_factor": 0.0,
                "roughness_factor": 1.0,
                "metallic_roughness_texture": "removed",
                "emissive_factor": [0.0, 0.0, 0.0],
                "emissive_texture": "removed",
                "specular_factor": 0.0,
                "allowed_material_extensions": ["KHR_materials_specular"],
            },
            "invariants": {
                "binary_unchanged": True,
                "binary_sha256": hashlib.sha256(parsed.binary).hexdigest(),
                "non_material_json_unchanged": True,
                "non_material_json_sha256": canonical_json_sha256(before_other),
            },
            "notes": [
                "Suppressed texture bytes remain embedded but are no longer referenced by the material.",
                "A retained base-color texture may still contain baked lighting; rendered human review remains mandatory.",
                "This visual material repair does not qualify the mesh, motion, anchors, contacts, or license.",
            ],
        }
        report["report_content_sha256"] = canonical_json_sha256(report)
        _write(output, payload)
        output_created = True
        _write(
            report_path,
            (
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
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
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
