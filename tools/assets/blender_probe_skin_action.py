"""Add a short in-place bone rotation so a SkinTokens rig can play one action."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import bpy


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_argv()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace {output}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("no armature")
    armature = armatures[0]
    bpy.context.view_layer.objects.active = armature
    bone_names = [b.name for b in armature.pose.bones]
    preferred = [n for n in bone_names if any(k in n.lower() for k in ("spine", "chest", "torso", "hip", "pelvis"))]
    bone_name = preferred[0] if preferred else bone_names[0]
    bone = armature.pose.bones[bone_name]
    armature.animation_data_create()
    action = bpy.data.actions.new("SkinProbe")
    armature.animation_data.action = action
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.frames
    for index in range(args.frames):
        frame = index + 1
        angle = math.sin(2 * math.pi * index / max(args.frames - 1, 1)) * math.radians(12)
        scene.frame_set(frame)
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (angle, 0.0, 0.0)
        bone.keyframe_insert(data_path="rotation_euler", frame=frame)
    bpy.ops.object.select_all(action="SELECT")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB", use_selection=True, export_animations=True)
    print(f"SKIN_PROBE_OK bone={bone_name} output={output}", flush=True)


if __name__ == "__main__":
    main()
