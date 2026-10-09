# HM3D oversized semantic region splitting on CPU

`tools/rooms/split_large_hm3d.py` implements the October 9 room-splitting
experiment. It consumes explicit external output/selection roots; it never
writes datasets, `rooms.json`, original checkpoints or the training list.
Outputs are created without replacement. The branch starts at `b2d625f`.

The source region is the registered HM3D semantic `region_id`. Its area is the
unrounded horizontal projection union of semantic ground faces, using the
existing `room_selection.floor_windows` and
`room_screening.geometry.union_projected_polygons`. Furniture does not subtract
area. Original regions at or below 50 square metres remain unchanged even if a
new diagnostic fails. Only oversized sources create new room candidates.

## Geometry and validation

1. The existing bounded 0.3 m floor-height windows separate levels. Directly
   annotated stair projections are separated and discarded; no stair room is
   admitted. Ambiguous semantic palettes or zero-area semantic ground faces
   make the source unresolved.
2. Door names and instance positions come from the original semantic text and
   GLB color mapping. A finite boundary chord near the door's main direction
   must split the floor. Every attempted door records its provenance and
   reliability; failures fall back to the floor contour cue.
3. The structural floor boundary is sampled every 0.10 m for a Voronoi medial
   axis. Candidate vertices and ridge midpoints locate narrow openings. A
   boundary-to-boundary chord must have the selected width and leave at least
   6 square metres on each side. Chord directions are sampled every 3 degrees;
   the continuous shortest chord is **not globally certified**. This numerical
   contour approximation, scan holes and floor materials can cause wrong cuts.
4. Remaining blocks above 50 square metres use native `PathFinder` navigation,
   continuous navmesh intersections, and 0.25 m grid samples. All floor area is
   conserved through exact clipped cell unions. Navigation cell weights are
   clipped continuous nav areas, rather than nominal grid-cell areas.
5. Raw scan GLB triangles provide CPU BVH occlusion rays at frozen camera and
   source heights of 1.5 and 1.2 m. `cpu_rays.py` retains the implementation from
   repository commit `82196bb`, `tools/acoustics/cpu_scan_rays.py`; no raw face
   is omitted. No acoustic sensor or RLR propagation is invoked.
6. A finite-camera mixed-integer assignment minimizes room count with 6–50 m²
   areas, the selected visibility fraction, and selected maximum 3D distance.
   At most 64 farthest-sampled camera candidates and a 20-second solve budget
   bound each solve. A second solve fixes the active cameras and room count and
   minimizes area-weighted squared distance with meaningful coefficients. It
   removes degenerate checkerboard ownership without weakening the original
   constraints. Compactness optimality is local to the fixed camera set. A
   solution attaining `ceil(area/50)` proves the global
   minimum by the area lower bound. Other solutions explicitly report that
   global minimum is **unverified**, even if the candidate-set MILP is optimal.
7. Wide interfaces above 1.6 m merge only when continuously supported by the
   native navmesh, protected semantic-door interfaces are absent, the union
   area is at most 50 m², and CPU visibility has a witness.
8. Final diagnostics include the 6–50 m² area, 2.4 m short side, medial body
   width/corridor proxy, localized semantic outdoor evidence, scan-black
   fraction and the existing frozen `room_selection.navigation.placement`
   witness. The existing sample budgets, clearances, field of view, source
   distances and ray tolerances remain unchanged. Missing evidence is unresolved.

## Read-only images and CPU software rendering

Saved orthographic projection/view matrices determine pixel alignment. Complete
same-floor frames are reused; a CPU world-coordinate mosaic can combine other
existing views. Missing source-frame coverage cannot certify the scan-black
fraction. Geometry-only drawings never enter black measurements.

When cached frames are incomplete, `render-overheads` uses the existing Habitat
runtime through Mesa llvmpipe. Compile `software_egl_only.c` to the explicit
artifact root with `cc -shared -fPIC -O2 ... -ldl -l:libEGL.so.1`. Set its path
in process-local `LD_PRELOAD`, hide CUDA, select only the Mesa EGL vendor,
set `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`,
`MESA_LOADER_DRIVER_OVERRIDE=llvmpipe`, `EGL_PLATFORM=surfaceless`,
`MESA_SHADER_CACHE_DISABLE=true`, and `LP_NUM_THREADS=1`.

The filter exposes exactly one `EGL_MESA_device_software` device. Enumeration is
checked before any Simulator is created. `gpu_device_id=-1` bypasses CUDA
selection, and `GL_RENDERER` must contain llvmpipe before an observation is
requested. Hardware EGL contexts are never a fallback. No shared library,
environment, driver or global configuration is modified.

Orthographic cameras are 30 m above the measured floor, clip the raw scene at
floor +1.8 m with near 28.2 m, and stop at floor −0.3 m with far 30.3 m. Actual
native matrices and renderer identity are saved. RGB channels below the
existing frozen threshold are a scan-quality proxy; dark real texture can be
misclassified. Images are produced for every relevant floor of at least 6 m²
and each source's largest layer; other small layers retain explicit diagnostics.

## External artifact layout and commands

The explicit output root contains `reference_smy_readonly_copy`, the fixed
`house_hash_split_v1.json`, fresh `inventory_v1`, and `PROGRESS_zh.md` before
`prepare`. The reference JSON paths must point inside that read-only copy.
The input selection root supplies `inputs.json`, `jobs`, `rooms_registry.jsonl`
and `media_reference`. Habitat, Magnum and optional native SDK dependency paths
come from environment variables, with no private server defaults in code.

```bash
python tools/rooms/split_large_hm3d.py prepare --output "$OUTPUT" \
  --selection-base "$SELECTION" --training-csv "$TRAINING_CSV"
python tools/rooms/split_large_hm3d.py run --output "$OUTPUT" \
  --scope tune --run-name tune_v2 --workers 8
python tools/rooms/split_large_hm3d.py select --output "$OUTPUT" --run-name tune_v2
python tools/rooms/split_large_hm3d.py render-overheads --output "$OUTPUT" --workers 4
python tools/rooms/split_large_hm3d.py run --output "$OUTPUT" \
  --scope holdout --run-name holdout_v1 --workers 8
python tools/rooms/split_large_hm3d.py evaluate --output "$OUTPUT" \
  --scope holdout --run-name holdout_v1
python tools/rooms/split_large_hm3d.py run --output "$OUTPUT" \
  --scope all --run-name all_v1 --workers 16 --reuse-reference-run reference_v1
python tools/rooms/split_large_hm3d.py review --output "$OUTPUT" --run-name all_v1
python tools/rooms/split_large_hm3d.py validate --output "$OUTPUT" --run-name all_v1
```

`NUMPY_MADVISE_HUGEPAGE=0`, single-thread BLAS/Numba, and an external `TMPDIR`
are required for the CPU launcher. A splitter parent has a 12 GiB address-space
limit; up to 16 workers each have 6 GiB (aggregate at most 108 GiB). A renderer
parent has 6 GiB and up to four workers each have 8 GiB. Those maxima cannot run
simultaneously at full concurrency; the launcher sequences them. Core dumps are
disabled in owned jobs. The limits are process-local.

## Comparison and honest limitations

Houses, rather than rooms, are split by a pre-recorded stable SHA256 ordering.
The 27 parameter combinations vary only allowed width, coverage and distance.
The combined reference delivery can be reused by `--reuse-reference-run` in the
all-source batch, so held-out geometry is not executed again. Tune scores select
once; ties prefer 0.6–1.6 m, 80%, 6 m. The holdout evaluation
is claimed once before scoring and cannot be repeated by the CLI.

Manual boxes crop the same-height source semantic floor, following existing
`room_selection.analysis.manual_iou`; they are not rectangle or navigation
areas. Hungarian pairing is one-to-one, extra auto blocks are reported, and
missing/unresolved targets score zero. Only the nine original-box dagger
references are excluded. Source macro and reference micro means are separate.

In this inventory all 33 hand-cut sources are at or below 50 m², as are all 181
V20 source IDs. They are therefore untouched under the frozen preservation rule.
The comparison scores cannot identify parameters or validate splitting on the
50 genuinely oversized sources. `scope_area_reconciliation_v1.csv` and
`source_lineage_audit_v1.json` record this mismatch; the implementation does not
expand a source into neighboring regions to manufacture eligibility.

`delivery_<run>` exports source JSON, one JSON per new block, unchanged
source diagnostics, summary CSV/JSON, a Chinese report, and one standalone flat
HTML file. Every image is embedded base64 but has no initial `src`; a viewport
intersection observer with zero root margin installs the source. Layers are
shown separately, accepted/discarded/unresolved blocks have different colors,
and manual rectangles are purple. Native browser rendering remains a separate
check; parser validation alone cannot prove browser behavior.

Run the focused `tests/test_room_split_auto.py` and the repository `fast_unit`
suite before handoff. Acoustic acceptance is intentionally not run.
