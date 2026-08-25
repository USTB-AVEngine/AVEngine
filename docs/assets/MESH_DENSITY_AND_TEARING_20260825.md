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
It gates on head-third survival at 0.7, on faceting at 0.30, on boundary and
non-manifold edges in single figures, and on the face target actually being
reached. The boundary
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

Reproduced with the landed tool at the default 80k and diagonal/800:

| breed | verdict | faceting | head third | relief passes | fidelity p99 |
| --- | --- | --- | --- | --- | --- |
| Jack Russell | accept | 4.3% | 1.23 | 1 | 0.0104 |
| dark Burmese | accept | 10.6% | 1.39 | 4 | 0.0108 |
| standard Burmese | accept | 23.3% | 1.18 | 29 | 0.0019 |
| Siamese | reject | 38.3% | 0.66 | 24 | 0.0021 |

Three of four pass at the default and the fourth is the relief case above. The
rigged results below were produced earlier at diagonal/700, before the gate
existed:

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

## The budget is squeezed from both sides

The recipe removes the topology blocker; it does not make the face budget free
to pick. Pushing an already-reducible mesh from 25k to 80k costs tearing:

| asset | 25k decimate-only | 80k retopologised |
| --- | --- | --- |
| Jack Russell, >2x | 0.702% | 0.391% |
| Jack Russell, >10x | 0.011% | 0.013% |
| Siamese, >2x | 0.937% | 1.038% |
| Siamese, >10x | 0.011% | 0.070% |

And cutting it too far starves the head, which is the opposite pressure and the
more surprising one. A head needs an absolute triangle count, not a share, and a
large smooth flank will take the budget first. The Siamese at diagonal/800:

| Siamese budget | faces reached | head third | verdict |
| --- | --- | --- | --- |
| 50k | 65,902 | 0.593 | reject |
| 80k | 79,983 | 0.658 | reject |
| 120k | 119,983 | 0.705 | accept |

Smoothing is not what moved it: at 80k the head reads 0.68, 0.67 and 0.66 for 6,
12 and 24 relief passes. And the budget is not the only knob either — the same
Siamese passes at 80k once the resolution drops to diagonal/700, which the sweep
found on its second attempt.

That is the whole argument for sweeping. Neither knob generalized from the animal
it was chosen on: 80k at diagonal/800 suits three of these four breeds and fails
the fourth, which passes at 80k/700. Two settings, one gate, and the gate is what
travels between animals.

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
fidelity cost (p99 0.0017 to 0.0023). The tool derives the pass count from the
measured relief ratio — remeshed faces over the face target — and reports both,
so the decision is visible in the report rather than buried in a flag. Note the
ratio moves with the budget as well as the resolution, so a smaller budget asks
for more smoothing: the Siamese derives 8 passes at 120k and 52 at 50k.

The relief ratio drives that derived pass count, but it is the wrong shape for a
threshold: it moves with the remesh resolution and the face budget, so the same
mesh reads 5.2 at diagonal/700 and 13.4 at diagonal/800 for reasons that have
nothing to do with how the surface looks. Measuring the faceting directly does
not have that problem, and it ranks assets the way the eye does:

| asset | relief passes | mean dihedral | edges past 30 degrees | reads as |
| --- | --- | --- | --- | --- |
| Jack Russell | 0 | 8.6 | 4.3% | clean |
| dark Burmese | 29 | 13.5 | 10.6% | clean |
| standard Burmese | 29 | 27.9 | 23.3% | clean |
| Siamese at diagonal/800 | 24 | 45.5 | 38.3% | faint speckle |
| Siamese at diagonal/700 | 0 | 56.9 | 48.5% | heavy speckle |

Everything that reads clean sits under 23 percent and everything speckled over
38, so the gate takes 0.30 and the remedy it names is more relief smoothing. The
ratio stays in the report because it drives the pass count; the faceting share
is what decides.

For the Siamese, though, no setting in the space we searched satisfies both
criteria at once, and the two remedies pull against each other:

| Siamese attempt | faceting | head third | non-manifold |
| --- | --- | --- | --- |
| 80k, 24 relief passes | 0.383 | 0.658 | 2 |
| 80k, 48 passes | 0.318 | 0.651 | 0 |
| 80k, 96 passes | 0.265 | 0.639 | 56 |
| 80k, 160 passes | 0.245 | 0.622 | 147 |
| 120k, 8 passes | 0.373 | 0.705 | - |
| 120k at diagonal/700, 0 passes | 0.481 | 0.844 | - |

Smoothing buys a smoother coat by starving the head and reintroducing
non-manifold edges; more faces save the head and make the faceting worse. A
volume-preserving laplacian smooth is the principled way out of that trade and
is a silent no-op at this mesh size — 24, 60 and 120 iterations returned
byte-identical readings — so the option was removed rather than shipped as a
setting that does nothing.

That puts the Siamese where the evidence actually points: its fur-scale relief is
a property of the reconstruction, and the fix belongs upstream in generation
rather than in the reduction. It stays a documented reject, which is the honest
state for it.

Welding before this measurement is not optional either, and this is the third
place in this pipeline where that mattered. A uv unwrap seams almost every
triangle and glTF splits a vertex at every seam, so the first version of this
measurement found five usable edges in 80,000 faces and reported a mean angle of
0.01 degrees for every asset alike. Any per-edge reading taken on a freshly
imported file describes the file format.

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
