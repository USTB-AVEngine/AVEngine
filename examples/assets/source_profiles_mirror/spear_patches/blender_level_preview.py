"""Apply a measured upright correction to a finalized static asset.

This is a preview, not the fix: it exists so the levelling decision can be
looked at rather than argued about. It rotates the asset so its own measured up
lands on world up, re-grounds it, and writes a sibling GLB. Nothing in the
published chain calls it.
"""

import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
source = argv[0]
output = argv[1]
up_gltf = [float(value) for value in argv[2:5]]

# glTF is y-up, Blender is z-up: (x, y, z) -> (x, -z, y).
up_blender = Vector((up_gltf[0], -up_gltf[2], up_gltf[1])).normalized()

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source)

roots = [item for item in bpy.context.scene.objects if item.parent is None]
rotation = up_blender.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix().to_4x4()
for root in roots:
    root.matrix_world = rotation @ root.matrix_world
bpy.context.view_layer.update()

meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
lowest = min(
    (item.matrix_world @ Vector(corner)).z
    for item in meshes
    for corner in item.bound_box
)
for root in roots:
    root.matrix_world = Matrix.Translation((0.0, 0.0, -lowest)) @ root.matrix_world
bpy.context.view_layer.update()

bpy.ops.export_scene.gltf(filepath=output, export_format="GLB")
print(f"LEVELLED {output}")
