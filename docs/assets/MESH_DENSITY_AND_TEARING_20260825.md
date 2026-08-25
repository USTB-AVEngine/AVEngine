# Mesh density, tearing, and how to generalize across animals (2026-08-25)

A generated quadruped tears open during a walk cycle. This records what the
tearing actually is, what fixes it for every breed rather than one, and the two
readings that were wrong along the way. Every number here was measured on this
batch, not estimated.

## What the tearing is

Not holes: rendering back faces in magenta shows zero interior pixels on all
four animals. Not a texture fault: the raw mesh and the rigged mesh both render
clean at rest, and dilating the atlas into its unpainted half (51.8 percent of
it) changed nothing. Not scrambled normals: the angle between shading normals
and posed geometry is the same at rest and in the pose (8.56 versus 8.35 degrees
mean).

It is skinning distortion. At the density the reconstruction ships, one bone's
influence spans a handful of triangles, so a weight step folds the surface over
itself. The fold's front face points at the camera, which is why a back-face
test finds nothing.

## The reconstruction was never the problem

Rendering the untouched 938k-face mesh shows a complete, convincing cat: head,
ears, whiskers, eyes, fur relief. The crumpled-paper look, and the standard
Burmese's destroyed head, were produced entirely by our own reduction. An
earlier reading of this batch concluded that a mesh carrying thousands of
non-manifold edges was beyond saving; that was wrong. Its topology is dirty and
its geometry is excellent.

Render the source before diagnosing a reduction
(`tools/assets/render_turntable_review.py`). It is one minute and it decides
whether to fix the generator or the pipeline.

## Why a face-count target does not generalize

A collapse decimator cannot touch a non-manifold edge. With seven thousand of
them the whole budget is spent on the regions that *are* collapsible, which are
the clean detail-dense ones. Counting faces in thirds along the reviewed forward
direction, normalised so 1.0 means "lost the same share as the mesh overall",
shows exactly which end of the animal paid:

| preparation, standard Burmese | faces reached | boundary | non-manifold | head third | rear third |
| --- | --- | --- | --- | --- | --- |
| weld, then collapse | 116,478 | 2,275 | 7,102 | **0.53** | 1.36 |
| weld, delete interior flaps, collapse | 99,609 | 15 | 4,534 | 0.46* | 1.71* |
| weld, collapse with a curvature vertex group | 114,507 | 2,198 | 7,082 | 0.52* | 1.63* |
| **voxel remesh at diagonal/800, collapse** | **79,925** | **1** | **6** | **1.18** | **0.81** |

Rows marked * were measured before the census learned the forward direction, so
their figures are the worst and best octant rather than the head and rear thirds;
the ordering they establish holds either way.

Two things this settles. Deleting interior flaps helps the surface (2,247
boundary edges down to 15, so the real holes close) but not the reduction: the
head still starves and the target is still unreachable. And curvature-weighted
collapse does nothing at all here (0.52 against 0.51) — the blocker is topology,
not detail, so protecting detail protects nothing.

The census has to know which end is the head, and this is not a refinement. An
axis-blind version scored 0.51 on the mesh whose face collapsed into flat facets
and 0.507 on a Jack Russell that had merely thinned its tail — the same number
for a ruined asset and a fine one. Split along the reviewed forward direction,
the ruined mesh reads 0.53 at the head where the three good ones read 1.18, 1.23
and 1.39, and the threshold has somewhere to sit. The tool takes
`--front-yaw-deg`, which the chain already carries for the heading step.

## What generalizes: remesh first, then reduce

Voxel remeshing replaces the surface with one that has no non-manifold edges and
no boundary edges, after which the face target is reachable exactly and the loss
spreads evenly. Resolution is not a detail — it dominates fidelity, and coarse
remeshing is worse than not remeshing at all (0.0217 against 0.0058). At
diagonal/800 it is better than every alternative on every axis at once.

The remesh discards the uv layout, so the original colour is baked back onto a
fresh unwrap. That is the whole cost, and it is about a minute per asset.

`tools/assets/retopologize_for_rigging.py` does all of it in one Blender session
and reports the numbers above, so each asset certifies itself:

```
blender -b --python tools/assets/retopologize_for_rigging.py -- \
  --input raw.glb --output retopo.glb --report retopo.json \
  --target-faces 80000 --voxel-divisor 800
```

Both meshes have to stay in one session — the bake needs the dense original as
its source, and reloading it from a file that no longer carries the texture is
the failure this avoids. Every count is taken after welding, because a glTF file
splits vertices at each uv and normal seam and a boundary or island count read
off a freshly imported file measures the file format instead of the surface.

`tools/assets/gate_retopology.py` reads that report and decides, so a mesh that
lost its head is rejected before it consumes a rigging slot rather than after.
It gates on head-third survival at 0.7, on boundary and non-manifold edges in
single figures, and on the face target actually being reached. The boundary
allowance is not zero because welding the measurement copy leaves the occasional
stray edge; a genuine hole arrives in the hundreds.

Fidelity is reported and gated only against catastrophe, because it turned out
not to carry an accept decision: among three assets that all look right it spans
0.0019 to 0.0108. Two things were wrong with it before that was clear. It
sampled every source vertex including detached debris the remesh correctly drops,
so it read 0.0119 on a Jack Russell whose debris share is 1.31 percent and 0.0017
on a Burmese whose share is 0.92 percent — the percentile crossed into the debris
exactly when the debris passed one percent. Sampling the source's largest island
only fixes the measurement, and it still spans fivefold, which is what demoted
it.

`tools/assets/run_generated_animal_chain.sh` is the whole chain with that gate in
place: prepare, gate, rig, orient, level, transfer the donor walk, measure and
render. It refuses to write into an existing workdir, and it sweeps the remesh
resolution rather than fixing one, because diagonal/800 was settled on a single
cat and one asset is not a population. The constant in this pipeline is the gate,
not the resolution.

## Measured across four breeds

Same recipe, four different animals, all reaching exactly the requested budget
with a manifold surface, then rigged and walked through the same chain. This
batch ran at diagonal/700, before diagonal/800 was settled as the better
resolution; source counts are welded, sorted by how dirty the input was:

| asset | source non-manifold | remesh faces | final faces | boundary | area stretched >2x | >4x | >10x |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard Burmese | 7,082 | 418,774 | 80,000 | 0 | 0.333% | 0.089% | 0.012% |
| dark Burmese | 4,171 | 425,790 | 80,000 | 0 | 0.416% | 0.117% | 0.024% |
| Siamese | 1,986 | 690,890 | 80,000 | 0 | 1.038% | 0.339% | 0.070% |
| Jack Russell | 1,052 | 383,184 | 80,000 | 0 | 0.391% | 0.136% | 0.013% |

Against the decimate-only chain on the same animals, the standard Burmese
improves fourfold (1.355% to 0.333% of area stretched past 2x) and the dark
Burmese too (1.616% to 0.416%), and both now look like the animal instead of
crumpled paper.

## Density still costs, even on clean topology

The recipe removes the topology blocker; it does not make density free. The
Jack Russell and Siamese were already reducible, and pushing them from 25k to
80k costs tearing:

| asset | 25k decimate-only | 80k retopologised |
| --- | --- | --- |
| Jack Russell, >2x | 0.702% | 0.391% |
| Jack Russell, >10x | 0.011% | 0.013% |
| Siamese, >2x | 0.937% | 1.038% |
| Siamese, >10x | 0.011% | 0.070% |

So the budget is a judgement, not a constant: 80k buys detail the reviewer can
see and costs tearing the reviewer can also see. On a mesh whose topology is
already clean, prefer the lower end.

Note which animal is which in the table above. The Siamese has the *second
cleanest* topology of the four (1,986 non-manifold edges) and the worst tearing
at 80k, while the standard Burmese has the dirtiest (7,082) and the best. Once
the remesh has removed topology as a variable, what is left ranking these assets
is something else entirely.

## Surface relief is a separate axis from topology

The Siamese was the one asset the remesh alone did not fix, and the reason is
neither topology nor the bake. Its raw surface carries fur-scale relief: at
diagonal/700 it remeshes into 690,890 faces where the other three land at
383k-426k for the same bounding box, so nearly twice the surface area is
micro-structure. That relief is sub-pixel at a million faces and reads as fur.
At the rigging budget it survives as irregular faceting, which on a pale coat
reads as dark stipple over the whole body.

Both other explanations were tested and ruled out. Rendering the retopologised
Siamese untextured shows the pitted surface directly, so it is geometry rather
than colour. Tightening the bake ray length from 5 to 2 percent of the diagonal
changed nothing, so it is not a ray punching through a thin wall either.

Smoothing the remesh before the reduction removes it: 24 passes at factor 0.5
takes the Siamese from heavy stipple to faint traces on the flank, at a small
fidelity cost (p99 0.0017 to 0.0023) and a wider octant spread (0.79-1.38 to
0.73-2.04). The tool derives the pass count from the measured relief ratio —
remeshed faces over the face target — and reports both, so the decision is
visible in the report rather than buried in a flag.

The ratio is resolution-dependent: the same mesh reads 5.2 at diagonal/700 and
13.4 at diagonal/800, because a finer grid keeps finding more relief to resolve.
Compare ratios only within one resolution, and re-tune the threshold if the
resolution changes.

## Where the non-manifold edges come from

Binned along the body axis, 60 percent of the Burmese's non-manifold edges are
in the rear third, peaking at the tail. The pose guide holds the tail up and
close to the body, and reconstructing a thin structure that nearly touches the
torso produces self-intersections. The shared negative prompt forbids a tail
touching or overlapping a *leg*; it says nothing about the body or the back.
That is a one-line extension worth making before the next generation run — and
now a cheaper one to skip, since the remesh no longer cares.

## Review rendering

The blotches read very differently under different lighting: the same posed
frame is clean under the static review renderer and blotchy under the animation
renderer. Soft, shadow-free lighting shows the asset honestly
(`tools/assets/render_walk_review.py`, `render_turntable_review.py`). Judge by
image, not by metric: worst-face-growth in particular is one outlier triangle
and ranks assets wrongly — the retopologised Burmese scores 162 against the 25k
Siamese's 24 and looks better than it in motion. Read the stretched-area shares
instead, and then look at the picture.

## Heading

`blender_estimate_generated_animal_forward.py` was wrong on one of four animals,
and its confidence was anti-correlated with correctness: the failure scored 0.82
and the lowest-confidence pass, 0.17, was right. The failure mode is a 180
degree head-tail swap. Render `tools/assets/probe_heading_axis.py` after
normalization and look at it -- a face means forward, a tail means backwards.
The contract already demands a reviewed yaw and review evidence for this step.
