"""Compare mesh structure after welding, without a glTF round trip in between.

Exporting to glTF splits vertices at every uv and normal seam, so counting
islands or boundary edges on a re-imported file measures the file format, not
the surface. Welding inside one session and counting there measures the surface.
"""
import bpy, sys, json, bmesh, math

argv = sys.argv[sys.argv.index("--")+1:]
src, out_json = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)

o = max((x for x in bpy.data.objects if x.type == "MESH"), key=lambda x: len(x.data.vertices))
bm = bmesh.new()
bm.from_mesh(o.data)
raw = {"verts": len(bm.verts), "faces": len(bm.faces),
       "boundary": sum(1 for e in bm.edges if len(e.link_faces) == 1)}

coords = [v.co for v in bm.verts]
mn = [min(c[i] for c in coords) for i in range(3)]
mx = [max(c[i] for c in coords) for i in range(3)]
diag = math.dist(mn, mx)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=diag * 1e-4)
bm.verts.ensure_lookup_table()

welded = {"verts": len(bm.verts), "faces": len(bm.faces),
          "boundary": sum(1 for e in bm.edges if len(e.link_faces) == 1),
          "nonmanifold": sum(1 for e in bm.edges if len(e.link_faces) > 2)}

seen = set()
islands = []
for v in bm.verts:
    if v in seen:
        continue
    stack = [v]
    seen.add(v)
    n = 0
    while stack:
        cur = stack.pop()
        n += 1
        for e in cur.link_edges:
            other = e.other_vert(cur)
            if other not in seen:
                seen.add(other)
                stack.append(other)
    islands.append(n)
islands.sort(reverse=True)
res = {"input": src, "diagonal": round(diag, 4), "raw": raw, "welded": welded,
       "island_count": len(islands), "largest_islands": islands[:5],
       "share_in_largest": round(islands[0] / max(1, welded["verts"]), 4)}
bm.free()
with open(out_json, "w") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1)
print("STRUCT", json.dumps(res, ensure_ascii=False))
