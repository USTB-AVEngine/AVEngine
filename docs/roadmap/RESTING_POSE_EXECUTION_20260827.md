# Resting-pose work-order execution record

Date: 2026-08-27. Branch: `cc-static-sound-sources`.

This record closes Tasks A and C and records the complete Task B evidence up to
the final shared-tree switch.  The switch itself remains pending explicit owner
approval because it changes the path consumed by downstream jobs; no switch had
occurred when this record was written.

## 1. Task A: attachment-aware measurement

Implemented in:

- `ca18556 assets(static): measure pose from the attachment surface`
- `6b30091 assets(static): sync resting pose into the index`

The tool now reads `placement.attachment_surface` and keeps the calibrated
floor path unchanged: lowest face vertex selects the bottom slice and winding
is ignored before normals are directed downward.  Wall assets use the outer
back selected from `acceptance.front_axis`; ceiling assets use the top.  Missing
placement is explicitly recorded as an assumed floor measurement.

The concrete mounting-plane existence failure is the wall-connected bottle
trap: its small curved pipe end can produce a plausible averaged normal even
though no back plate exists.  Types and ordinary tests cannot distinguish that
connection from a flush wall mount at runtime because both legitimately declare
`attachment_surface=wall`.  The tool therefore records candidate area,
projected coverage and planarity and returns `no_mounting_plane_found` instead
of inventing an angle.  This did not add a formal-registration gate or frozen
baseline.

Preflight on the eight wall and one ceiling assets:

| Asset | Surface | Area share | Projected share | RMS fraction | Result |
|---|---:|---:|---:|---:|---|
| air_conditioner/wall_split | wall | 0.035779 | 0.158683 | 0.017640 | level 1.52 deg |
| air_conditioner/window_unit, original | wall | 0.011305 | 0.153138 | 0.007066 | leaning 18.98 deg |
| landline_phone/wall_mounted | wall | 0.004764 | 0.029715 | 0.008853 | level 1.01 deg |
| doorbell/video | wall | 0.010510 | 0.045083 | 0.007461 | acceptable 3.30 deg |
| doorbell/chime_box | wall | 0.010901 | 0.073665 | 0.017793 | acceptable 4.89 deg |
| microwave/over_range | wall | 0.014201 | 0.112342 | 0.005615 | acceptable 4.54 deg |
| floor_drain/bottle_trap | wall-connected | 0.002390 | 0.016676 | 0.006665 | no mounting plane |
| smoke_detector/ceiling_disc | ceiling | 0.021166 | 0.110070 | 0.012472 | level 0.38 deg |
| smoke_detector/wall_square | wall | 0.021611 | 0.183867 | 0.018239 | acceptable 3.01 deg |

`--apply` now writes the same complete acceptance fields into all 40 leaf
`asset.json` files and the embedded records in the 44-row `index.json`.  The
four animal records remain untouched.  Evidence and the reversible pre-apply
archive are under:

`tmp/resting_pose_task_a_20260826_v1/`

Relevant results:

- `asset_json_before_apply.tar`
- `index_before_pose_sync.json`
- `resting_pose_after_index_sync.json`
- `promotion_staging_raw_measurement.json`
- `television_splayed_feet_exemption.json`

The full unit suite after Task A was:

`3209 passed, 65 skipped, 71 subtests passed in 432.29s`.

The final full unit suite after index synchronization and all tracked method
revisions was:

`3235 passed, 65 skipped, 71 subtests passed in 290.03s`.

## 2. Task C: wall-connected bottle trap

`plumbing_fixture/floor_drain/exposed_bottle_trap_silver` intentionally remains
`attachment_surface=wall`: its outlet connects toward the plumbing wall.  It is
not a flush wall-mounted object and has no back plate.  Its correct result is
`no_mounting_plane_found`, not `floor` and not a fabricated wall angle.  No gate
or frozen contract was added.

## 3. Task B: final disposition of the nine measured candidates

Eight paths have fresh `research_v2` staging assets.  One original television
is retained under the explicit design exemption allowed by the work order.

| Path | Source of final geometry | Final raw pose | Disposition |
|---|---|---:|---|
| climate_control/air_conditioner/portable_floor_white | fresh r1 generation | level 2.86 deg | replace with v2 |
| household_clock/alarm_clock/digital_cube_black | fresh r1 generation | level 0.88 deg | replace with v2 |
| plumbing_fixture/sink_with_tap/counter_vanity_white | fresh r1 generation | level 0.15 deg | replace with v2 |
| kitchen_appliance/blender/jug_blender_black | fresh r2 generation + proxy correction | level 2.95 deg | replace with v2 |
| plumbing_fixture/bathtub/built_in_alcove_white | fresh r2 generation + 21.140 deg proxy correction | level 1.23 deg | replace with v2 |
| plumbing_fixture/sink_with_tap/pedestal_basin_white | fresh r2 generation + 14.037 deg proxy correction | level 0.70 deg | replace with v2 |
| climate_control/air_conditioner/window_unit_white | original reviewed geometry + 21.118 deg dual-authority correction | fitted back tilt 2.56 deg; `no_mounting_plane_found` because RMS fraction 0.021733 | replace with v2, retain unknown-plane clue |
| plumbing_fixture/toilet/elevated_tank_exposed_pipe_white | original reviewed c1e1 geometry + 9.744 deg dual-authority correction | level 1.20 deg | replace with v2 |
| audio_playback/television/flat_panel_16_9_two_splayed_feet | original reviewed geometry | raw foot result 29.61 deg | retain under design exemption |

The television exemption preserves the raw `geometry.resting_pose=leaning` and
`base_normal_tilt_deg=29.61`.  Its upper display-cabinet plane was independently
fit above five height cuts and stayed between 3.31 and 5.11 degrees, inside the
owner-approved acceptable band.  Selected base-face angles had a weighted
median of 34.95 degrees because the two feet are intentionally outward-splayed.
The leaf and index therefore record raw `leaning`, acceptance `acceptable`, and
`resting_pose_disposition=accepted_design_exemption`.

All replacement marker reviews were checked after +X finalization.  Corrections
made during review included moving the counter-sink marker from the basin rim
to the aerator, the blender marker onto the motor vents, the bathtub marker
onto the filler outlet, and the window marker onto the upper discharge louvers.

## 4. Preserved failed batches and method revisions

All outputs are fresh/no-clobber and remain under SPEAR `tmp/`:

- `static_sound_resting_pose_redo_20260826_r1`
- `static_sound_resting_pose_redo_20260827_r2`
- `static_sound_resting_pose_redo_20260827_r3_method`
- `static_sound_resting_pose_redo_20260827_r4_method`
- `static_sound_resting_pose_redo_20260827_r5_tv_method`
- `static_sound_resting_pose_redo_20260827_r6_tv`
- `static_sound_resting_pose_redo_20260827_r7_window_method`
- `static_sound_resting_pose_redo_20260827_r8_window`
- `static_sound_resting_pose_redo_20260827_r9_toilet`
- `static_sound_resting_pose_redo_20260827_r10_window_method`
- `static_sound_resting_pose_redo_20260827_r11_window`
- `static_sound_resting_pose_redo_20260827_r12_toilet_method`
- `static_sound_resting_pose_redo_20260827_r13_toilet`

Rejected 2D and 3D decisions were authenticated by the existing review tools.
No rejected item was sent through a later gate merely to obtain an asset.

Tracked method revisions and provenance:

- `0b6784c`: elevated toilet single-pipe clear-gap method
- `07d41d8`: television thin-back/level-camera method
- `c1e805a`: window-unit flat-rear method
- `ef8c36a`: window-unit squat-chassis method
- `e7df016`: toilet top-button/no-small-protrusion method

The SPEAR worktree Git pointer is historically broken because its common Git
directory no longer exists.  Runtime profile copies were therefore changed
only after the byte-identical tracked AVEngine mirror and provenance were
validated and committed.  AVEngine remains the source authority.

## 5. Promotion staging and pending shared-tree switch

The fully assembled promotion tree is:

`/data/avengine_external/assets/.sound_source_assets_v1_resting_pose_20260827_v1.staging`

Audit result:

`PROMOTION_AUDIT_OK assets=44 static=40 animals=4 replacements=8 formal=0 tv_exemption=accepted bottle=no_mounting window=no_mounting`

Direct measurement of all 40 GLBs produced:

- 24 level
- 13 acceptable
- 1 raw leaning: the accepted splayed-foot television
- 2 `no_mounting_plane_found`: the wall-connected bottle trap and corrected
  window unit whose fitted tilt is level but whose rear roughness remains just
  above the mounting-plane existence threshold

The intended reversible switch is:

- current tree ->
  `/data/avengine_external/assets/sound_source_assets_v1_before_resting_pose_20260827_v1`
- audited staging ->
  `/data/avengine_external/assets/sound_source_assets_v1`

Both directories are on filesystem device 2065.  The operation must roll back
the first rename if the second rename fails.  It was not executed because the
high-impact publication approval requires explicit owner confirmation.

All staged and current records remain `formal_dataset_registration_authorized=false`;
formal episode count remains zero.  No kettle or animal deformation gate was
introduced.
