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

Existing target-owned cat bindings can also be rechecked with these surface
measurements. TailRoot is accepted as the cat spelling of the Tail anchor;
--registration-request supplies anatomical labels in that binding coordinate
system. This changes bone-name compatibility, not the penetration thresholds.

The optional Rig.enable_skin_cache precomputes fixed linear contributions from
all skin sets. Default Rig.skin behavior remains unchanged. Comparison against
the original pose evaluator includes integer-normalized, repeated, four/eight
influences, bind transforms and static parts. Actual cat vertices and normals
are independently compared before this opt-in is used for native validation.

For existing cat bindings, normalize_owned_cat_rig_scale verifies every
rest triangle against the registered source under the already recorded rigid
support-plane placement, checks decoded texture pixels and semantic PBR
defaults, and applies only a uniform source-coordinate correction to a new
derivative. It then fixes upper Body/Head/Tail at fitted Idle while retaining
the previously Blender-baked limb world transforms, following the dog method.
AuthoredWalk replays that actual baked Walking clip directly; it does not
apply a second IK solve. Actual physical speed, stride and lift include the
registered source scale and are recorded explicitly. Both CPU and native
runners retain every existing frame, contact, stretch, visibility, floor,
clearance, emitter, 180/90 and acoustic gate. A full native pass is still
required; no source mesh, texture or registration is repaired.

Actual-skin stream mode passes the exact CPU float32 vertices that native
rigid_glb displays to the same Blender BVH and inward regional gates. It
avoids glTF reimport differences in multiple skin sets. Bind and all gait
frames retain original topology, edges, seam and shard checks; stopping at
the first violating frame cannot produce a passing receipt.

The optional flat_slope capsule boundary keeps a C1 partition with zero
endpoint slope and limits its peak derivative to 1/0.9, compared with 1.5
for cubic smoothstep. It changes target weight fitting, not the source
action or any admission threshold. Default smoothstep behavior is unchanged.
Actual paw minima are recorded from each source around its registered
paw landmark and used only in owned derivative requests.

Acoustic --interval-batch-size changes scheduling only. Every midpoint still
uses three separately executed raw native references and all original energy,
direction and +6 dB gates. The default remains two intervals. The private
controller fuses guard and dispatch to stay within eight processes with five
single-thread audio children; complete-tree PSS and disk limits still apply.
