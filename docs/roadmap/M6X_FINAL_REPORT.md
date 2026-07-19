# M6.x Fixed SPEAR Apartment Source-Logic Canary — Final Report

Status: **pass as a bounded fixed-room source-logic research canary**.

This result is deliberately narrower than room admission or dataset release.
It proves that the current AVEngine interfaces can produce controlled,
source-resolved episodes in the existing Habitat-compatible SPEAR
`apartment_0000` without editing or rearranging its furniture.

## What is feasible

- One frozen `RoomCapsule` drives all scenarios with one co-located and
  co-oriented `view0`/`listener0` rig.
- Registered articulated-human, articulated-animal and rigid-object endpoints
  are resolved through the entity, endpoint and sound registries.
- `AudioProgram` schedules zero, one or multiple named endpoints; each sound
  asset declares the AudioProgram event modes in which it may be used.
- S0--S5 execute routing, front/rear counterfactuals, visible silence, moving
  intermittent sound, overlapping sound and live Habitat LOS/NLOS contrast.
- Every variant retains Timeline v2, the existing M5.1 source/pair/clip flags,
  independent binaural stems, an exact stem-sum mixture and source-pair RIR
  evidence.
- Audio is 360 degrees and is not switched by camera FOV. Each variant has a
  clean video and a diagnostic main-view + Topdown video.
- Placement is intentionally source-center-only. The center gate and Topdown
  consume the same runtime obstacle snapshot. Apartment furniture is baked
  into the stage/navmesh; ReplicaCAD additionally exposes 113 live rigid
  collision OBBs.

The reviewed Apartment bundle contains eight variants and sixteen videos. S1
keeps decoded RGB identical while swapping front/rear routing; S2's silent
negative is exactly zero; S3 moves the active endpoint by about 0.585 m; S4
contains 0.9 s of real two-source overlap; and S5 uses live raycast LOS/NLOS.

## Claim boundary and remaining work

No S0--S5 source-logic scenario is blocked. The following broader claims remain
unqualified and are not hidden by the scenario-level acoustic `pass`:

- Acoustic materials are `research_placeholder`, not measured physical truth.
- The selected M3 Apartment package reports
  `compiler_source_to_package_parity=fail`, `geometry_report=fail`,
  `material_coverage=pass` and `ray_leakage=not_run`; its physical
  `qualification_claim` is therefore false. Native RLR output is valid for this
  bounded routing/spatialization canary, not for formal room admission.
- The center-point rule does not guarantee that a rendered human or animal body
  never visually brushes furniture.
- The fixed human/Beagle visual capture is a replaceable capture adapter. The
  RoomCapsule, source endpoints, AudioPrograms, timelines and acoustic renderer
  are data-driven, but a new articulated asset family still needs a compatible
  capture adapter.
- This task does not qualify arbitrary rooms, generate natural-language QA,
  furnish rooms automatically or admit a training dataset.

## Later room-authoring transition

The fixed-room experiment shows that source identity, zero/one/multi-source
programs, 360-degree binaural routing, synchronization, stems and diagnostics
are no longer the dominant uncertainty. A later automatic-room-authoring
milestone is justified when the project chooses a room-level acoustic
qualification policy and demonstrates the same contracts through at least one
additional qualified RoomProvider. Furniture generation should remain a
separate scaling milestone rather than being mixed into this canary.

## Review and use

When this report is copied into a generated bundle, open the sibling
`REVIEW_INDEX.html` for direct links to all media and evidence. The repository
`README.md` records the current local closeout path together with the exact
Conda commands, prerequisites, retained-capture fast path and ReplicaCAD
obstacle-review command.

Current verification: `1409 passed, 1 skipped`. The skip is the optional old-M6
retained-evidence readback and is not an M6.x failure.
