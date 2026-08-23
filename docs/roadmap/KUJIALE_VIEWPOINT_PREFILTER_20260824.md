# Kujiale autonomous viewpoint CPU prefilter (20260824)

## Scope

This checkpoint is the KJ1 planning slice only. It does not use Studio and did
not run UE, SPEAR, GPU capture, native pixels, M3, RLR, QuestionSpec, or a
mini-pilot. All outputs remain `research_only`, `episode_counted=false`, formal
dataset count 0, and explicitly carry pending UE/native-pixel certification.

The implementation uses each retained InteriorAgent room polygon plus its
5 cm descendant-mesh footprint raster to:

- enumerate fully feasible straight chords in the shared natural walking
  distance band from `examples/qa_v2/motion_profiles_v1.json`;
- sample camera X/Y, three planning heights, and deterministic UE yaws;
- measure 2D raster LOS, pinhole-HFOV feasible-area coverage, and per-view
  endpoint/midpoint/anchor visibility;
- remove near-duplicate pose/yaw/coverage candidates and choose 4-8 viewpoints
  through marginal corridor/visible-area coverage plus pose diversity;
- produce per-view directed corridor subsets and analytically verify continuous
  two-moving-source separation of at least 0.60 m with opposite image sides at
  the first question anchor; and
- draw a labelled Topdown that says UE/SPEAR and native pixels are pending.

No hash, new schema, frozen baseline, or release/admission gate was added.

## Code

- `src/avengine/optional_backends/kujiale_viewpoints.py`
- `tools/m6z/build_kujiale_viewpoint_bank.py`
- `tools/m6x/compile_kujiale_feasibility_bank.py`: exposes its existing
  metadata-to-runtime-obstacle adapter and no longer hard-codes the living-room
  label in Topdown.
- `src/avengine/optional_backends/interioragent_kujiale.py`: the existing visual
  canary profile now accepts 4-8 views.
- `tools/m6z/run_spear_kujiale_canary.py`: montage layout supports 4-8 views,
  including odd counts. This remains a visual canary, not an Episode qualifier.

## Real retained-data checks

### Living room: CPU prefilter complete

Pixar USD extraction ran fresh against the current full-home derived stage:

- metadata:
  `/data/jzy/tmp/kujiale_living_metadata_20260824T0110Z.json`
- source scope: `livingroom_491`
- extracted objects: 62 elevated + 18 ground blockers
- extraction: 14.79 s, maximum RSS 192,164 KiB

The final 8-view prefilter is:

`/data/jzy/tmp/kujiale_living_viewpoint_prefilter_8view_20260824T0340Z`

Observed output:

- 44.55 m2 common source-center feasible area;
- 256 fully feasible natural-speed straight corridors;
- 8 selected camera poses over heights 1.05/1.50/1.85 m;
- per-view eligible corridor counts: 114, 52, 18, 37, 26, 33, 14, 19;
- per-view compatible both-moving pair counts after continuous 0.60 m
  separation and opposite-side checks: 6906, 1104, 97, 517, 157, 664, 143,
  189;
- build: 5.34 s, maximum RSS 70,916 KiB.

I inspected `viewpoint_topdown.png`. All V1-V8 markers lie on the green retained
feasible raster, the direction arrows use the expected room X/Y axes, blockers
remain black, and the legend exposes each pose, corridor/pair count, and
`pixels pending`. This is a CPU Topdown review only; I did not see or claim any
new UE frame.

### Kitchen: honestly rejected for straight-chord production

Two fresh runs against the retained current kitchen metadata both failed before
viewpoint selection. The final failure evidence is:

`/data/jzy/tmp/kujiale_kitchen_viewpoint_prefilter_20260824T0305Z/FAILED.json`

The human/dog motion profiles require a 3.0-3.9 m five-second straight chord.
The kitchen polygon plus descendant-footprint raster contains no fully feasible
chord in that interval. The planner did not lower natural speed, ignore an
obstacle, or relabel the retained back-and-forth 0.5 m route as a straight
corridor. Kitchen is therefore not KJ1 mini-pilot-ready under the work-order's
straight-chord rule. A later slice must either author a separately allowed
multi-waypoint natural-speed policy or select another room; that decision is
not hidden inside this prefilter.

### Performance correction

The first implementation enumerated every pair among 727 living-room sample
points, then tested every line pixel. Two fresh attempts were stopped after the
quadratic candidate stage proved unsuitable; no UE/GPU process was involved.
The final implementation instead uses a deterministic 0.5 m origin lattice,
15-degree rays, and three lengths from the natural-distance band. Every line is
checked by conservative grid-supercover traversal, including both side cells
at an exact raster-corner crossing. Viewpoint
selection screens at most 64 camera positions, 256 area samples, and 64
balanced corridors, then re-evaluates every corridor only for the selected
views. The final per-view subsets are not truncated by screening.

## Verification

- 37 focused/adjacent unit tests passed in 10.90 s (maximum RSS 127,916 KiB),
  covering natural distance, full-segment obstacle rejection,
  deterministic 4-view selection, low/high candidates, pending certification,
  4-8 profile bounds, 0.60 m compatible pairs, odd-view montage layout, and
  fresh/no-clobber output;
- Ruff and `py_compile` pass for the changed files;
- `git diff --check` passes;
- no Studio interaction and no UE/GPU/native render occurred.

## Remaining production work

KJ1 does not certify a viewpoint. The next visual slice must build the fresh
current Kujiale actor stage, render 2-3 known-position UE frames per chosen
view, verify camera/FOV/actor readbacks and native masks, personally inspect the
montage, then run the four-point audiovisual/question mini-pilot. M3/RLR and
room-registry work remain separate slices.
