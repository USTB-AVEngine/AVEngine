"""Record research-only emitter anchors from a reviewed GLB bounding box.

This does not admit an asset. Mouth/muzzle or chest anchors are bbox fractions
in the asset root after the class-level import transform, matching the published
sound-source offset_space final_scaled_asset_root convention.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


SCHEMA = "avengine_generated_asset_emitter_anchors_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-class", choices=("quadruped", "biped"), required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_argv()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"missing input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=False)
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for vertex in obj.data.vertices:
            points.append(obj.matrix_world @ vertex.co)
    if not points:
        raise SystemExit("no mesh vertices")
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    minimum = Vector((min(xs), min(ys), min(zs)))
    maximum = Vector((max(xs), max(ys), max(zs)))
    size = maximum - minimum

    def at_fraction(fx, fy, fz):
        return [
            float(minimum.x + size.x * fx),
            float(minimum.y + size.y * fy),
            float(minimum.z + size.z * fz),
        ]

    if args.body_class == "quadruped":
        # +X forward, +Z up in Blender after glTF import; muzzle near front-high.
        anchors = [
            {"anchor_id": "body", "anchor_type": "body", "offset_m": at_fraction(0.45, 0.50, 0.45)},
            {"anchor_id": "head", "anchor_type": "head", "offset_m": at_fraction(0.82, 0.50, 0.72)},
            {"anchor_id": "muzzle", "anchor_type": "muzzle", "offset_m": at_fraction(0.95, 0.50, 0.55)},
        ]
        default_id = "muzzle"
    else:
        anchors = [
            {"anchor_id": "body", "anchor_type": "body", "offset_m": at_fraction(0.50, 0.50, 0.50)},
            {"anchor_id": "head", "anchor_type": "head", "offset_m": at_fraction(0.50, 0.50, 0.90)},
            {"anchor_id": "chest", "anchor_type": "chest", "offset_m": at_fraction(0.50, 0.35, 0.70)},
        ]
        default_id = "head"
    for item in anchors:
        item["offset_space"] = "final_scaled_asset_root"
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_dataset_registration_authorized": False,
        "status": "research_bbox_fraction_pending_visual_review",
        "body_class": args.body_class,
        "default_emitter_anchor_id": default_id,
        "bounds_min": [float(minimum.x), float(minimum.y), float(minimum.z)],
        "bounds_max": [float(maximum.x), float(maximum.y), float(maximum.z)],
        "emitter_anchors": anchors,
        "input": str(source),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"EMITTER_ANCHORS_OK default={default_id} output={output}", flush=True)


if __name__ == "__main__":
    main()
