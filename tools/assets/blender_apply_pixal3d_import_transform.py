"""Apply the class-level Pixal3D Blender import/root transform after glTF import.

This is the recovered TokenRig/Pixal3D orientation hook: one profile-selected
root matrix for every scene root, never a per-asset-id exception. Quadruped
extra_root is identity; biped extra_root is the historical 180 deg world-Z yaw.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix


SCHEMA = "avengine_pixal3d_import_transform_application_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transform-profile", type=Path, required=True)
    parser.add_argument("--body-class", choices=("quadruped", "biped"), required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_argv()
    source = args.input.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"missing input: {source}")
    for path, label in ((output, "output"), (manifest, "manifest")):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing to replace {label}: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from pixal3d_transform_profile import (
        blender_import_kwargs,
        body_class_root_matrix,
        body_class_target_front_axis,
        load_profile,
    )

    profile = load_profile(args.transform_profile)
    extra_root = body_class_root_matrix(profile, args.body_class)
    import_kwargs = blender_import_kwargs(profile)
    target_front = body_class_target_front_axis(profile, args.body_class)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), **import_kwargs)
    extra = Matrix([tuple(row) for row in extra_root])
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    if not roots:
        raise RuntimeError("imported scene has no root objects")
    for root in roots:
        root.matrix_world = extra @ root.matrix_world

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_all_vertex_colors=True,
        export_vertex_color="ACTIVE",
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_dataset_registration_authorized": False,
        "body_class": args.body_class,
        "target_front_axis": target_front,
        "transform_profile": str(Path(profile["_path"])),
        "extra_root_transform_4x4_row_major": extra_root,
        "blender_import": import_kwargs,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "PIXAL3D_IMPORT_TRANSFORM_OK "
        f"body_class={args.body_class} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
