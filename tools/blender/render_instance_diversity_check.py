"""Render several finished assets in one frame at room distance.

The instance-diversity argument rests on a claim that an albedo atlas cannot
settle: that two instances of the same product differ enough to be told apart
in a rendered frame from across a room, where the object is small, lit by room
light and seen at an angle. This is how that claim gets checked.

It is a review aid, not part of the publication chain. Nothing it writes is an
asset, and it reads published GLBs without modifying them.

Two details matter and are easy to get wrong:

  * lighting. The static review renderer is a deliberately hot studio setup and
    washes finishes out completely - a black-ash cabinet renders silver there.
    This uses plain even room light and the Standard view transform, which is
    what makes the finishes readable.
  * placement. Assets are positioned in world space rather than by nudging
    rotation_euler, because an asset carrying an upright correction already has
    a rotation on its node, so its local Z is not world up.

Run it with a JSON spec:

    blender -b --python <this file> -- '<spec json>' out.png

The spec names the camera, the lights, optional platforms to stand things on,
and the assets with their world positions and yaw in degrees. A finalized
static asset faces +X, so a camera looking from -Y wants yaw -90.
"""

import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
spec = json.loads(argv[0])
output = argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = spec.get("width", 1280)
scene.render.resolution_y = spec.get("height", 720)
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = spec.get("exposure", 0.0)

world = bpy.data.worlds.new("room")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.36, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = spec.get("ambient", 0.6)

# A floor, so the objects read as standing in a room rather than floating.
bpy.ops.mesh.primitive_plane_add(size=24.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.active_object
material = bpy.data.materials.new("floor")
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0.30,
    0.28,
    0.26,
    1.0,
)
material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.85
floor.data.materials.append(material)

for index, (x, y, z, energy) in enumerate(spec.get("lights", [])):
    light_data = bpy.data.lights.new(f"light_{index}", type="AREA")
    light_data.energy = energy
    light_data.size = 2.5
    light = bpy.data.objects.new(f"light_{index}", light_data)
    light.location = (x, y, z)
    light.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(light)

for index, box in enumerate(spec.get("platforms", [])):
    # A console or shelf, so the assets stand on something and the elevation
    # story - a speaker at 0.7 m against a cat on the floor - is visible.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    platform = bpy.context.active_object
    # primitive_cube_add(size=1.0) already spans -0.5..0.5, so the scale is the
    # extent itself and halving it again would make every platform half size.
    platform.scale = tuple(box["size"])
    platform.location = (
        box["position"][0],
        box["position"][1],
        box["position"][2] + box["size"][2] / 2.0,
    )
    shade = bpy.data.materials.new(f"platform_{index}")
    shade.use_nodes = True
    shade.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = tuple(
        box.get("colour", (0.22, 0.19, 0.17, 1.0))
    )
    shade.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.7
    platform.data.materials.append(shade)

for item in spec["assets"]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=item["glb"])
    added = [obj for obj in bpy.context.scene.objects if obj not in before]
    roots = [obj for obj in added if obj.parent is None]
    # World-space, not rotation_euler.rotate_axis: an asset that carries an
    # upright correction already has a rotation on its node, so its local Z is
    # not world up and rotating about it turns the object somewhere else.
    placement = Matrix.Translation(Vector(item["position"])) @ Matrix.Rotation(
        math.radians(item.get("yaw_deg", 0.0)), 4, "Z"
    )
    for root in roots:
        root.matrix_world = placement @ root.matrix_world

camera_data = bpy.data.cameras.new("camera")
camera_data.lens = spec.get("focal_mm", 35.0)
camera = bpy.data.objects.new("camera", camera_data)
camera.location = Vector(spec["camera"])
scene.collection.objects.link(camera)
scene.camera = camera
target = Vector(spec["look_at"])
direction = target - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

scene.render.filepath = output
bpy.ops.render.render(write_still=True)
print(f"ROOM_SCALE_RENDER {output}")
