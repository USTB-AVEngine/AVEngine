# Cat retry using the admitted dog rigging procedure

The derivative tools were selectively adapted from the committed dog branch
`e43873c8`, which includes `91d82852`, `9676c57d`, `db6b8a47` and
`f0b9c66d`. The selected donor remains the independently licensed Quaternius
Cat.blend, CC0-1.0, not the dog's mesh, joint positions or skin weights.
Cat.blend has the same deform-bone chains; geometry establishes all four sides.

Use `build_static_quadruped_rig.py` with explicit targets, template, Blender,
asset and output paths. The retry uses `--weights blender_envelope` and
`--motion-amplitude .03`. The upper body, head and tail retain their own fitted
Idle pose. Continuous Voronoi boundaries extinguish limb influence smoothly.
The submitted gait still advances 48 mm per 1.6 s at 0.03 m/s, lifting paws
8 mm. Only source angles and fitted target weights differ; no gate changes.

`check_quadruped_source_preservation.py` independently verifies source vertices,
normals, triangles, UVs, PBR values and decoded texture pixels.
`check_static_quadruped_rig.py` evaluates 145 CPU and Blender frames.
Visible cross-region added area is summed over every region pair and limited
to 25 mm² per frame; inward material penetration is limited to 2 mm, including
same-region intersections. Counts are diagnostic. Stretch is at most 1.5;
new UV seam separation is at most 10 micrometres.

`validate_static_quadruped_native_bounded.py` evaluates 145 actual Habitat frames,
resets the simulator before reusing eight pose buffers, checks mesh bounds
against native readback, and runs actual 180° and 90° controls for every frame.
The original stage, floor, navigation, visibility, clearance, emitter and
audio gates remain authoritative. Native geometry pass is not joint audio
admission; fresh moving RIR qualification is required before changing the
final classification. Only the measured room and route are qualified.

The private controller preserves previous sliding receipts and enforces a
900-second per-asset attempt and a two-hour aggregate retry window. Timeboxed
or incomplete attempts remain unverified, never falsely admitted.
