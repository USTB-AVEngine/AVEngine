"""Measure how much of the surface a pose stretches.

The grey patches are not holes: no interior is visible. They are triangles the
skinning pulls into long thin slivers, so the texture smears across them. This
measures that directly by comparing every face area at rest with the same face
in the posed frame, and reporting the share of posed surface that grew past a
threshold.
"""
import bpy, sys, json

argv = sys.argv[sys.argv.index("--")+1:]
src, out_json, action_name, ratio = argv[0], argv[1], argv[2], float(argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)

def face_areas():
    dg = bpy.context.evaluated_depsgraph_get()
    areas = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = o.matrix_world
        for p in me.polygons:
            vs = [mw @ me.vertices[i].co for i in p.vertices]
            if len(vs) == 3:
                a = (vs[1] - vs[0]).cross(vs[2] - vs[0]).length * 0.5
            else:
                a = p.area
            areas.append(a)
        ev.to_mesh_clear()
    return areas

arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm is not None and arm.animation_data:
    arm.animation_data.action = None
bpy.context.view_layer.update()
rest = face_areas()

posed = rest
if arm is not None and action_name != "none":
    act = next((a for a in bpy.data.actions if action_name.lower() in a.name.lower()), None)
    if act is not None:
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = act
        s, e = act.frame_range
        bpy.context.scene.frame_set(int(s + (e - s) * ratio))
        bpy.context.view_layer.update()
        posed = face_areas()

n = min(len(rest), len(posed))
total_posed = sum(posed[:n]) or 1.0
buckets = {"gt2": 0.0, "gt4": 0.0, "gt10": 0.0}
worst = 0.0
for i in range(n):
    r0 = rest[i]
    r1 = posed[i]
    if r0 <= 1e-12:
        continue
    g = r1 / r0
    worst = max(worst, g)
    if g > 2:
        buckets["gt2"] += r1
    if g > 4:
        buckets["gt4"] += r1
    if g > 10:
        buckets["gt10"] += r1
res = {
    "input": src,
    "faces": n,
    "share_area_stretched_over_2x": round(buckets["gt2"] / total_posed, 5),
    "share_area_stretched_over_4x": round(buckets["gt4"] / total_posed, 5),
    "share_area_stretched_over_10x": round(buckets["gt10"] / total_posed, 5),
    "max_growth": round(worst, 1),
}
with open(out_json, "w") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1)
print("STRETCH", json.dumps(res, ensure_ascii=False))
