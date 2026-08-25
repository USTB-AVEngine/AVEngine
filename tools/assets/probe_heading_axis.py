import bpy, sys, math, os
argv = sys.argv[sys.argv.index("--")+1:]
src, out = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.data.objects if o.type == "MESH"]
xs = []
for o in objs:
    vs = o.data.vertices
    for i in range(0, len(vs), 503):
        xs.append(o.matrix_world @ vs[i].co)
minx = min(p.x for p in xs); maxx = max(p.x for p in xs)
miny = min(p.y for p in xs); maxy = max(p.y for p in xs)
minz = min(p.z for p in xs); maxz = max(p.z for p in xs)
cx=(minx+maxx)/2; cy=(miny+maxy)/2; cz=(minz+maxz)/2
span=max(maxx-minx, maxy-miny, maxz-minz)
print(f"BBOX x[{minx:.3f},{maxx:.3f}] y[{miny:.3f},{maxy:.3f}] z[{minz:.3f},{maxz:.3f}]")
cam_data = bpy.data.cameras.new("probe"); cam = bpy.data.objects.new("probe", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (cx + span*2.2, cy, cz + span*0.15)
cam.rotation_euler = (math.radians(90), 0, math.radians(90))
bpy.context.scene.camera = cam
light_data = bpy.data.lights.new("sun", type="SUN"); light_data.energy = 4
light = bpy.data.objects.new("sun", light_data); bpy.context.collection.objects.link(light)
light.rotation_euler = (math.radians(50), 0, math.radians(40))
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = 384; sc.render.resolution_y = 384
sc.render.film_transparent = False
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("PROBE_OK", out)
