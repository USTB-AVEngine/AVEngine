#!/usr/bin/env python3
"""Bound the mesh, not the box around the mesh.

scene_bounds transformed the eight corners of each object's local bounding box.
Under the yaw the finalizer used to apply that is exact - a rotation about Z
leaves the z extent alone - so the height it scaled by and the lowest point it
grounded were both right. Under the upright correction it is not: the axis
aligned box around a rotated box is strictly larger than the box around the
rotated points, and a bookshelf cabinet came out 25.7 cm against a 33 cm target
because the height it divided by was the inflated one.

Reading the vertices directly is exact under any rotation. It also removes the
same latent error from grounding, which was translating by an over-estimated
minimum for exactly the same reason.
"""

from __future__ import annotations

from pathlib import Path

TOOL = Path(
    "/data/jzy/code/SPEAR-lead-b/tools/blender_finalize_generated_static_object.py"
)

OLD = '''def scene_bounds(meshes: list[Any]) -> tuple[Vector, Vector]:
    corners = [
        mesh.matrix_world @ Vector(corner)
        for mesh in meshes
        for corner in mesh.bound_box
    ]
    minimum = Vector(
        tuple(min(point[axis] for point in corners) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(point[axis] for point in corners) for axis in range(3))
    )
'''

NEW = '''def scene_bounds(meshes: list[Any]) -> tuple[Vector, Vector]:
    """World bounds of the vertices themselves.

    Not of ``bound_box``: that is the object's local axis-aligned box, and the
    world box around a rotated box is strictly larger than the world box around
    the rotated points. The difference is zero for a rotation about Z, which is
    all this tool used to apply, and large once it also stands an object up.
    """

    corners = [
        mesh.matrix_world @ vertex.co
        for mesh in meshes
        for vertex in mesh.data.vertices
    ]
    if not corners:
        raise contract.EmitterContractError("static finalization scene has no vertices")
    minimum = Vector(
        tuple(min(point[axis] for point in corners) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(point[axis] for point in corners) for axis in range(3))
    )
'''

text = TOOL.read_text(encoding="utf-8")
if text.count(OLD) != 1:
    raise SystemExit(f"anchor matched {text.count(OLD)} times")
TOOL.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("scene_bounds now measures the vertices")
