"""Render front/side/back stills, a turntable, and a short action clip.

Numeric metrics are not the gate. This is the visual-review surface for a
Pixal3D mesh or a SkinTokens rig after the class-level import transform.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transform-profile", type=Path, default=None)
    parser.add_argument("--body-class", choices=("quadruped", "biped"), default="quadruped")
    parser.add_argument(
        "--apply-body-class-root",
        action="store_true",
        help="apply profile extra_root; omit when the GLB was already oriented",
    )
    parser.add_argument("--action", default="")
    parser.add_argument("--turntable-frames", type=int, default=36)
    parser.add_argument("--action-frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(argv)


def _bounds():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    skinned = [
        obj
        for obj in meshes
        if any(mod.type == "ARMATURE" for mod in obj.modifiers)
    ]
    chosen = skinned or [
        obj
        for obj in meshes
        if obj.name.lower() not in {"icosphere", "sphere"}
        and len(obj.data.vertices) > 32
    ] or meshes
    points = []
    for obj in chosen:
        for vertex in obj.data.vertices:
            points.append(obj.matrix_world @ vertex.co)
    if not points:
        raise RuntimeError("no mesh vertices to frame")
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    minv = Vector((min(xs), min(ys), min(zs)))
    maxv = Vector((max(xs), max(ys), max(zs)))
    return minv, maxv, (minv + maxv) * 0.5, max(maxv - minv)


def _look_at(camera, center: Vector, offset: Vector) -> None:
    camera.location = center + offset
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _setup_world(center: Vector, span: float) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    cam_data = bpy.data.cameras.new("review")
    cam_data.lens = 50
    camera = bpy.data.objects.new("review", cam_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    light_data = bpy.data.lights.new("sun", type="SUN")
    light_data.energy = 4.0
    light = bpy.data.objects.new("sun", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((span, -span, span))
    light.rotation_euler = (math.radians(50), 0, math.radians(40))
    fill_data = bpy.data.lights.new("fill", type="SUN")
    fill_data.energy = 1.2
    fill = bpy.data.objects.new("fill", fill_data)
    scene.collection.objects.link(fill)
    fill.rotation_euler = (math.radians(20), 0, math.radians(-120))
    return camera


def _render(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _playable_actions():
    """Collect clip names from bpy.data.actions and NLA strips after glTF import."""

    names = []
    seen = set()
    for action in list(bpy.data.actions):
        if action.name not in seen:
            seen.add(action.name)
            names.append(action.name)
    for obj in bpy.context.scene.objects:
        animation = getattr(obj, "animation_data", None)
        if animation is None:
            continue
        if animation.action is not None and animation.action.name not in seen:
            seen.add(animation.action.name)
            names.append(animation.action.name)
        for track in list(getattr(animation, "nla_tracks", []) or []):
            for strip in list(getattr(track, "strips", []) or []):
                action = getattr(strip, "action", None)
                if action is not None and action.name not in seen:
                    seen.add(action.name)
                    names.append(action.name)
    return names


def main() -> None:
    args = parse_argv()
    source = args.input.resolve()
    output = args.output_dir.resolve()
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"missing input: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution

    import_kwargs = {"import_pack_images": False}
    extra_root = None
    if args.transform_profile is not None:
        tools_dir = Path(__file__).resolve().parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from pixal3d_transform_profile import (
            blender_import_kwargs,
            body_class_root_matrix,
            load_profile,
        )
        profile = load_profile(args.transform_profile)
        import_kwargs = blender_import_kwargs(profile)
        extra_root = (
            body_class_root_matrix(profile, args.body_class)
            if args.apply_body_class_root
            else None
        )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), **import_kwargs)
    if extra_root is not None:
        from mathutils import Matrix
        extra = Matrix([tuple(row) for row in extra_root])
        for root in [obj for obj in bpy.context.scene.objects if obj.parent is None]:
            root.matrix_world = extra @ root.matrix_world

    minv, maxv, center, span = _bounds()
    camera = _setup_world(center, span)
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    distance = span * 2.4
    zoff = span * 0.12
    if args.body_class == "quadruped":
        # Asset forward is +X after the class-level extra_root.
        views = {
            "front": Vector((distance, 0.0, zoff)),
            "side": Vector((0.0, -distance, zoff)),
            "back": Vector((-distance, 0.0, zoff)),
        }
    else:
        # Biped canonical front is -Y after extra_root.
        views = {
            "front": Vector((0.0, -distance, zoff)),
            "side": Vector((distance, 0.0, zoff)),
            "back": Vector((0.0, distance, zoff)),
        }
    for name, offset in views.items():
        _look_at(camera, center, offset)
        _render(output / f"{name}.png")

    turntable = output / "turntable"
    turntable.mkdir()
    for index in range(args.turntable_frames):
        angle = 2.0 * math.pi * index / args.turntable_frames
        offset = Vector((math.sin(angle) * distance, -math.cos(angle) * distance, span * 0.12))
        _look_at(camera, center, offset)
        _render(turntable / f"{index:03d}.png")

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    action_name = args.action
    actions = _playable_actions()
    if not action_name and actions:
        preferred = [item for item in actions if item.lower() in {"walking", "walk", "standing_idle", "idle"}]
        action_name = preferred[0] if preferred else actions[0]
    played = None
    clip = output / "action"
    if armatures and action_name:
        armature = armatures[0]
        action = bpy.data.actions.get(action_name)
        if action is None:
            for name in actions:
                action = bpy.data.actions.get(name)
                if action is not None:
                    action_name = name
                    break
        if action is not None:
            armature.animation_data_create()
            armature.animation_data.action = action
            scene = bpy.context.scene
            start = int(action.frame_range[0])
            end = int(action.frame_range[1])
            count = max(2, args.action_frames)
            clip.mkdir()
            _look_at(camera, center, Vector((distance * 0.75, -distance, span * 0.18)))
            for index in range(count):
                frame = start + (end - start) * index / max(count - 1, 1)
                scene.frame_set(int(round(frame)))
                _render(clip / f"{index:03d}.png")
            played = action.name
        elif armatures:
            played = None

    import json as _json
    manifest = {
        "schema": "avengine_generated_asset_visual_review_v1",
        "front": str(output / "front.png") if (output / "front.png").is_file() else None,
        "side": str(output / "side.png") if (output / "side.png").is_file() else None,
        "back": str(output / "back.png") if (output / "back.png").is_file() else None,
        "turntable_dir": str(output / "turntable") if any((output / "turntable").glob("*.png")) else None,
        "action_dir": str(clip) if clip.is_dir() and any(clip.glob("*.png")) else None,
        "action_name": played,
    }
    (output / "review_manifest.json").write_text(
        _json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "GENERATED_ASSET_REVIEW_RENDER_OK "
        f"front={output / 'front.png'} action={played or 'none'} "
        f"bounds=({minv.x:.3f},{minv.y:.3f},{minv.z:.3f})-({maxv.x:.3f},{maxv.y:.3f},{maxv.z:.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
