"""Render a walk cycle with soft, shadow-free lighting and the asset's own materials."""
import bpy, sys, math, os

argv = sys.argv[sys.argv.index("--")+1:]
src, out_dir, action_name, frames, zoom = argv[0], argv[1], argv[2], int(argv[3]), float(argv[4])
os.makedirs(out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
body = max(meshes, key=lambda x: len(x.data.vertices))
for o in meshes:
    if o is not body:
        o.hide_render = True

arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
act = None
if arm is not None:
    act = next((a for a in bpy.data.actions if action_name.lower() in a.name.lower()), None)
    if act is not None:
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = act

world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.30, 0.32, 0.36, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.4
bpy.context.scene.world = world

for rot, energy in (((math.radians(55), 0, math.radians(35)), 3.0),
                    ((math.radians(65), 0, math.radians(-120)), 1.6),
                    ((math.radians(115), 0, math.radians(180)), 1.0)):
    ld = bpy.data.lights.new("l", type="SUN")
    ld.energy = energy
    ld.use_shadow = False
    lo = bpy.data.objects.new("l", ld)
    bpy.context.collection.objects.link(lo)
    lo.rotation_euler = rot

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = 640
sc.render.resolution_y = 640
cam_data = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cam_data)
bpy.context.collection.objects.link(cam)
sc.camera = cam

start, end = (act.frame_range if act is not None else (0.0, 1.0))
picks = [int(start + (end - start) * i / frames) for i in range(frames)]
first = True
for n, f in enumerate(picks):
    sc.frame_set(f)
    bpy.context.view_layer.update()
    if first:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg)
        me = ev.to_mesh()
        pts = [body.matrix_world @ me.vertices[i].co for i in range(0, len(me.vertices), 97)]
        ev.to_mesh_clear()
        minv = [min(p[i] for p in pts) for i in range(3)]
        maxv = [max(p[i] for p in pts) for i in range(3)]
        c = [(minv[i] + maxv[i]) / 2 for i in range(3)]
        span = max(maxv[i] - minv[i] for i in range(3))
        cam.location = (c[0], c[1] - span * zoom, c[2])
        cam.rotation_euler = (math.radians(90), 0, 0)
        first = False
    sc.render.filepath = os.path.join(out_dir, "frame_%04d.png" % n)
    bpy.ops.render.render(write_still=True)
print("CLEANWALK_OK", out_dir)
