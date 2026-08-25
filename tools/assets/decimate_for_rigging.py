"""Reduce a generated mesh to a runtime-sane face count before rigging.

The reconstruction arrives near a million triangles. At that density one bone's
influence spans a handful of triangles, so any weight step folds the surface
over itself during a walk. Collapsing to a normal skeletal-mesh budget makes
each bone cover a broad, smooth region, and it is the density the engine wants
anyway. UVs ride along with the collapse.
"""
import bpy, sys, json, bmesh

argv = sys.argv[sys.argv.index("--")+1:]
src, out, report, target_faces = argv[0], argv[1], argv[2], int(argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)

info = {"input": src, "target_faces": target_faces, "meshes": []}
for o in [x for x in bpy.data.objects if x.type == "MESH"]:
    bm = bmesh.new(); bm.from_mesh(o.data)
    before_v, before_f = len(bm.verts), len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.to_mesh(o.data); o.data.update(); bm.free()

    # A collapse ratio is applied to triangles, and triangulating an ngon mesh
    # yields more faces than the polygon count predicts, so aim, measure, repeat.
    bpy.context.view_layer.objects.active = o
    ratio = 1.0
    for _ in range(5):
        current = len(o.data.polygons)
        if current <= target_faces * 1.05:
            break
        ratio = min(1.0, target_faces / current)
        mod = o.modifiers.new("dec", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)

    bm = bmesh.new(); bm.from_mesh(o.data)
    after_v, after_f = len(bm.verts), len(bm.faces)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bm.free()
    info["meshes"].append({"object": o.name, "verts": [before_v, after_v],
                           "faces": [before_f, after_f], "ratio": round(ratio, 5),
                           "boundary_after": boundary,
                           "has_uv": bool(o.data.uv_layers)})
    print(f"DECIMATE {o.name} verts {before_v}->{after_v} faces {before_f}->{after_f} "
          f"ratio={ratio:.4f} boundary={boundary} uv={bool(o.data.uv_layers)}")

bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=False)
with open(report, "w") as fh:
    json.dump(info, fh, ensure_ascii=False, indent=1)
print("DECIMATE_OK", out)
