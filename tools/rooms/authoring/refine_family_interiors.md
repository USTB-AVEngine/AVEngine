# Family interior detail authoring

`refine_family_interiors.py` opens one polished A/C Blender input and writes a fresh,
editable static asset package. It reuses `refine_room_details.py` geometry helpers
and the established `polish_furnished_room.py` GLB/USD/semantic exporters. It does
not own production cameras, actor placement, acoustics, motion or QA.

Run on the server with Blender 4.5, a fresh output directory, and the inspected
CC0 material directory:

```sh
blender -b -t 6 --python-exit-code 1 \
  --python tools/rooms/authoring/refine_family_interiors.py -- \
  --input-root /path/to/polished_v3_final/room_a \
  --output-root /path/to/fresh/room_a \
  --texture-root /path/to/detail_materials --room room_a --samples 24
```

Use `--room room_c` for C. Output contains Blender, evaluated GLB/USD and textures,
full static mesh bounds, source seating/anchors, material provenance, and four
review-only camera poses with world forward vectors and horizontal FOV. Images
are CPU Cycles authoring inspection, never final AVEngine pixels.

The pass keeps source seat centers, tops and fronts. It replaces soft furniture
with smooth crowned surfaces; adds seams, wood frames and grounded sofa feet;
replaces dining back slabs with bent wood and joinery; models thin-wall cups,
bowls and shallow plates; and positions tableware within actual table edges.
Books and folded cloth remain separately editable. Picture panels are mounted to
real wall segments rather than overlapping or hanging in front of windows.

Wood uses the verified walnut scan with a baked color adjustment, floor oak uses
a 1.2 m tile, and woven fabric uses a 0.3 m tile. Plaster uses a clean constant
base color with restrained scanned roughness/normal; its mottled scan color was
rejected after image inspection. No unsupported procedural shader is introduced.

September 5 inspection: A has 311 static mesh records and 8 unchanged seat points;
C has 405 static mesh records and 13 unchanged seat points. Geometry-derived seat
heights agree with preserved 0.53/0.57 m support heights. All GLB/USD geometry
references resolve. Overall, sofa, dining/tableware views were inspected; A also
has a separate clay inspection render. A is reference-informed by AEA loc3, not
an exact reconstruction. C is authored. The first rejected A attempt is retained
separately. Native SPEAR/UE import/render verification remains with integration;
these assets carry `research_candidate`, not admission claims.
