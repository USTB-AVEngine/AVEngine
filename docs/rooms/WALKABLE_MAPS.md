# Reusable native walkable room maps

`tools/rooms/build_walkable_maps.py` builds `ROOM_ID/walkable_map_v1.npz`
under a caller-selected output root. It uses the original Habitat stage,
its measured native basis, declared NavMesh and zero artificial stage collision
margin. No renderer or GPU is created. HM3D and MP3D are supported; Kujiale
requires a separate UE measured-floor/collision adapter and is not approximated
by a Habitat scan.

Set `AVENGINE_WALKABLE_MAP_ROOT` to this output root in both production planning
and acceptance/native execution. Set `AVENGINE_REQUIRE_WALKABLE_MAP=1` when maps
are mandatory. A configured missing, stale or wrong-room map fails closed.
Older unconfigured callers keep their proposal-only NavMesh behavior. Planned
routes retain the exact map path and room ID so native fitting uses the same map.
Changes need to be merged into the motion checkout before those workers use them.
The motion batch CLI also accepts `--walkable-map-root PATH`, persisted as
`walkable_map_root` in its manifest. The shared acceptance worker reads this
field before construction and native admission, so both stages use one cache.
A hand-authored acceptance manifest can set the same field.

Example (with the runtime's CPU environment and this worktree's `src` first on
`PYTHONPATH`):

```bash
nice -n 10 python tools/rooms/build_walkable_maps.py \
  --houses inputs/houses.json --runtime inputs/runtime.json \
  --priority-manifest inputs/acceptance_plan.json \
  --priority-manifest inputs/motion_batch.json \
  --priority-only --workers 4 --output-root tmp/walkable_maps
export AVENGINE_WALKABLE_MAP_ROOT="$PWD/tmp/walkable_maps"
export AVENGINE_REQUIRE_WALKABLE_MAP=1
```

The command checks its import path, requires committed producer code, limits
worker count, monitors its own process tree's PSS and pauses its own workers
below 30 GB available RAM or at 39 GB PSS. Start it under `nice -n 10` with
OMP/OpenBLAS/MKL/NumExpr thread counts set to one. Workers group rooms by house
and reuse native geometry in memory. Reports list each room's time, cell counts,
failures and exact PNG paths; `PROGRESS.md` gets a line per completed room.

NPZ arrays use `[z, x]` order, metres in native world coordinates. Fields are
`version`, `origin_xz_m` (lower cell corner), `resolution_m` (0.05),
`floor_reference_y_m`, `floor_y_m`, `source_code_commit`, `metadata_json`, and
boolean `walkable_human` (0.35 m), `walkable_dog` (0.25 m), `walkable_cat` and
`walkable_slide` (both 0.20 m). Diagnostic arrays retain `inside_room`,
`native_snap`, `support` and `static_clearance_m`. Unknown and outside cells are
blocked. File size/mtime and source paths invalidate changed stage/NavMesh
inputs; room polygon and floor identity are compared before cache reuse.
Unchanged caches are reused across code commits. Cache creation is atomic and
never replaces an existing map.

The v2 construction policy (owner 2026-10-15) reuses PhysicalFloor's
`OriginalVerticalStageRay` and the same native `sim.cast_ray` origin (+20 cm)
and length (85 cm). Center/corner supports use a +/-3 cm reference band, with
at most 4 cm probe span or 3x3 neighbour height change. Native center readback
uses the map band directly: calling PhysicalFloor's unchanged 15 mm final gate
here would silently undo this relaxation. Native/mesh differences remain
recorded as diagnostics. No native admission code is changed.

Faces within 8 cm of the measured reference and normal cosine >0.9 are ground.
Only faces with a vertex above 10 cm are obstacles; faces are clipped to the
10 cm--1.8 m body slab before XZ projection. Vertical and degenerate wall
triangles remain obstacles. Clearance uses the actual species radius, without
the old extra static margin, cell-diagonal margin or duplicate NavMesh body
inflation. The center must still lie inside the polygon and snap horizontally
within 2 mm. Unknown/outside support counts as unsupported.

Each actual-radius disk needs >=85% supported cells and a supported center.
This replaces complete binary erosion (which also added ~7 cm to the radius).
The diagnostic `support_fraction_human/dog/cat` arrays retain these ratios.
Independent radius coverage ratios need not be nested; metadata
`footprint_policy=independent-radius-coverage-v2` allows this explicitly while
cat and slide remain identical. Existing v1 erosion caches keep their nesting
validation. The NPZ container/version/name remain v1 for compatibility;
`algorithm_revision=permissive-support-3cm-step4cm-85pct-v2` identifies the policy.
Use a new output root to regenerate rather than silently reusing v1 maps.

The map is a permissive construction aid. The original per-pose skin, foot,
floor, reach, spacing, native snap, collision, camera, audio and question gates
remain authoritative and unchanged. A green map cell is not pose admission.
Native route fitting adds current qualified actors' expanded world bounding
boxes and the original >=1.5 m listener distance. Reflection probes and movement
segments use grid supercover, so they turn at blocked map cells and cannot jump
through a thin obstacle between endpoint samples.

PNG backgrounds are actual original-stage triangle top views (height shaded),
with green walkable and red blocked cells, separately shown for three radii.
They are geometry diagnostics; they are not textured camera renders.
