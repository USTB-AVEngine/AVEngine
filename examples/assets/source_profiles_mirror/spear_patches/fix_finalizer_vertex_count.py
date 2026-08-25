#!/usr/bin/env python3
"""Protect a vertex count that survives the glTF round trip.

The finalizer exports the transformed scene, imports it back and refuses any
change to a list of protected counts.  Raw vertex count is not one of the
things that survives that round trip: the exporter splits vertices at UV and
normal seams, so a real bookshelf-speaker mesh came back 37353 vertices
against 37293 with an identical 60000 triangles.  The check was reading a
container detail as if it were geometry.

Welded vertex count is invariant under seam splitting and still catches
geometry being added or removed, so it is protected instead.  Raw vertex count
stays in the readback because it is informative, just not a gate.
"""

from __future__ import annotations

from pathlib import Path

TOOL = Path(
    "/data/jzy/code/SPEAR-lead-b/tools/blender_finalize_generated_static_object.py"
)

IMPORT_OLD = "import bpy\nfrom mathutils import Matrix, Vector\n"
IMPORT_NEW = "import bmesh\nimport bpy\nfrom mathutils import Matrix, Vector\n"

SUMMARY_OLD = '''def scene_summary(meshes: list[Any]) -> dict[str, Any]:
    return {
        "mesh_count": len(meshes),
'''
SUMMARY_NEW = '''def welded_vertex_count(meshes: list[Any], distance: float = 1.0e-6) -> int:
    """Vertices after welding by position, which the glTF round trip preserves.

    Exporting to glTF splits a vertex wherever the UV or the normal is
    discontinuous, so the raw count changes across an export and import even
    though no geometry moved.  Welding first removes that difference while
    still counting real geometry.
    """

    total = 0
    for mesh in meshes:
        working = bmesh.new()
        try:
            working.from_mesh(mesh.data)
            bmesh.ops.remove_doubles(
                working, verts=working.verts, dist=distance
            )
            total += len(working.verts)
        finally:
            working.free()
    return total


def scene_summary(meshes: list[Any]) -> dict[str, Any]:
    return {
        "mesh_count": len(meshes),
'''

VERTEX_OLD = '''        "vertex_count": sum(len(mesh.data.vertices) for mesh in meshes),
'''
VERTEX_NEW = '''        # Reported but deliberately not protected: see welded_vertex_count.
        "vertex_count": sum(len(mesh.data.vertices) for mesh in meshes),
        "welded_vertex_count": welded_vertex_count(meshes),
'''

PROTECTED_OLD = '''        "animation_count",
        "vertex_count",
        "face_count",
'''
PROTECTED_NEW = '''        "animation_count",
        "welded_vertex_count",
        "face_count",
'''

text = TOOL.read_text(encoding="utf-8")
for old, new in (
    (IMPORT_OLD, IMPORT_NEW),
    (SUMMARY_OLD, SUMMARY_NEW),
    (VERTEX_OLD, VERTEX_NEW),
    (PROTECTED_OLD, PROTECTED_NEW),
):
    if text.count(old) != 1:
        raise SystemExit(f"anchor matched {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
TOOL.write_text(text, encoding="utf-8")
print("patched", TOOL.name)
