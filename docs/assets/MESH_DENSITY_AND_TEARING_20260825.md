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

## What generalizes: a ladder, not a recipe

Voxel remeshing replaces the surface with one that has no non-manifold edges and
no boundary edges, after which the face target is reachable exactly and the loss
spreads evenly. That is what rescues a mesh the plain route ruins. But it is not
the right treatment for every animal, and applying it unconditionally makes some
assets worse: on one cat the plain weld-and-collapse to 25k reads 0.011 percent
of area stretched past ten times during the walk where the same source
retopologised to 80k reads 0.123, and rendered from the same camera under the
same light the 25k version is visibly the cleaner of the two.

So the pipeline tries the cheapest preparation first and escalates only when a
measurement says to. `tools/assets/run_generated_animal_chain.sh` walks a ladder,
preparing, gating, rigging, animating and gating again at each rung, and stops at
the first that passes both:

| rung | preparation |
| --- | --- |
| 1 | plain weld and collapse to 25k |
| 2 | voxel remesh at diagonal/800, collapse to 80k |
| 3 | voxel remesh at diagonal/700, collapse to 80k |
| 4 | voxel remesh at diagonal/800, collapse to 120k |

On these four breeds two stop at rung 1 and two need rung 2. Evidence from the
rungs that failed is kept rather than deleted.

`tools/assets/retopologize_for_rigging.py` performs a rung and reports what it
did, with `--skip-remesh` selecting the plain route. Both meshes stay in one
Blender session, because the colour bake needs the dense original as its source
and reloading it from a file that no longer carries the texture is the failure
this avoids. Every count is taken after welding, because a glTF file splits
vertices at each uv and normal seam and a boundary or island count read off a
freshly imported file measures the file format instead of the surface.

## The two gates, and why they gate what they do

**Before rigging**, `tools/assets/gate_retopology.py` judges only damage the
reduction itself did, because that is all it can judge route-independently:
head-third survival at 0.7, and whether the face target was reached at all. Both
readings work across routes and both catch a real failure - a starved head, or a
reduction blocked by non-manifold edges collapse cannot touch.

Three readings that look like criteria are reported and *not* judged, each for a
measured reason:

- **Faceting** punishes a low face count. The coarsest asset here scored worst on
  it, 0.657, and looked best; a denser one scored 0.383 and looked speckled.
  Face count and dihedral angle are not independent, so the share of edges past a
  fixed angle cannot say "good enough".
- **Boundary and non-manifold edges** are driven to zero by a remesh, so there is
  nothing to check, and inherited untouched by the plain route, where they
  describe the reconstruction rather than a defect this step introduced. Two
  assets that look right carry 1,052 and 1,986 non-manifold edges.
- **Fidelity** spans fivefold across assets that all look right.

**After rigging**, `tools/assets/gate_rigged_asset.py` decides whether the result
is good, from how the surface deforms during the walk. The share of area whose
faces grow past ten times separates every asset measured so far, across both
failure modes:

| asset | share past 10x | verdict | how it looked |
| --- | --- | --- | --- |
| Siamese, plain 25k | 0.011% | accept | clean |
| Jack Russell, plain 25k | 0.011% | accept | clean |
| standard Burmese, remesh 80k | 0.012% | accept | clean |
| Jack Russell, remesh 80k | 0.013% | accept | clean |
| dark Burmese, remesh 80k | 0.024% | accept | clean |
| standard Burmese, plain 113k | 0.070% | reject | head destroyed |
| Siamese, remesh 80k | 0.123% | reject | speckled coat |

Threshold 0.03 percent sits in that gap. Worst-face growth is reported and never
gated: it is one outlier triangle and it ranks assets wrongly, scoring 162 on an
asset that looks better than one scoring 24.

Rigging is what makes this reading available, and it costs about eighty seconds -
cheap enough to spend on a rung that might be the answer, which is what makes the
ladder practical.

## Measured across four breeds

Each breed at the rung the ladder settles on, with the readings both gates use:

| breed | rung | faces | head third | share past 10x | verdict |
| --- | --- | --- | --- | --- | --- |
| Jack Russell | plain 25k | 25,000 | 1.46 | 0.011% | accept |
| Siamese | plain 25k | 25,000 | 1.28 | 0.011% | accept |
| standard Burmese | remesh 80k | 79,925 | 1.18 | 0.012% | accept |
| dark Burmese | remesh 80k | 79,976 | 1.39 | 0.024% | accept |

And why each one rejects the rung it does not use:

| breed | rejected rung | reading that failed |
| --- | --- | --- |
| standard Burmese | plain 25k | head third 0.54, and the collapse stalled at 111,937 faces |
| dark Burmese | plain 25k | collapse stalled at 61,147 faces |
| Siamese | remesh 80k | head third 0.66, and 0.123% of area folds past 10x |
| Jack Russell | remesh 80k | nothing - it also passes, at 0.013%; rung 1 simply comes first |

The two Burmese are the meshes carrying 7,082 and 4,171 non-manifold edges, and
they are exactly the two that need the remesh. The two the plain route handles
carry 1,052 and 1,986. Against the plain route at the density it could actually
reach, the standard Burmese improves fourfold on area stretched past 2x (1.355%
to 0.333%) and stops looking like crumpled paper.

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

## Surface area is a separate axis from topology

The Siamese is the one asset the remesh does not fix, and none of the three
obvious explanations survived measurement. It is not the reconstruction: its raw
mesh is the *second smoothest* of the four, 12.1 percent of edges past thirty
degrees against the standard Burmese's 33.6, and rendering it untouched shows a
clean cat. It is not the generated reference image, which is the smoothest and
palest of the four with less visible fur striping than either Burmese. It is not
the bake: rendering the retopologised mesh untextured shows the pitted surface
directly, and tightening the bake ray from 5 to 2 percent of the diagonal
changed nothing.

What is different is surface area. At diagonal/700 the Siamese remeshes into
690,890 faces where the other three land at 383k-426k for the same bounding box,
so nearly twice the area is gentle undulation. Its undulations are shallow —
that is why the raw mesh reads smooth — but there are a great many of them, and
a fixed triangle budget spread over twice the area gives each triangle twice the
curvature to cross. That is the faceting. The two Burmese, whose raw surfaces are
much rougher, come out *smoother* through this pipeline (33.6 to 23.3 percent,
21.4 to 10.6) while the Siamese goes the other way (12.1 to 38.3).

Both remedies work, and the derived pass count made them cancel. More faces
raises head survival; more smoothing lowers faceting. But the pass count is
derived from remeshed faces over the budget, so raising the budget lowers the
smoothing by about the same factor:

| Siamese, diagonal/800 | derived passes | faceting | head third |
| --- | --- | --- | --- |
| 80k | 24 | 0.383 | 0.658 |
| 120k | 8 | 0.373 | 0.705 |

Given both at once, the two readings pass: 120k faces at diagonal/700 with 48
passes forced reads faceting 0.286 and head survival 0.823. It fails anyway, on
topology, and that is where this stops. Enough smoothing to bring the faceting
down squeezes the two walls of a thin tail or ear into each other, and the
self-intersection count tracks the total displacement rather than the step size:

| Siamese, 120k at diagonal/700 | faceting | head third | non-manifold |
| --- | --- | --- | --- |
| 48 passes at factor 0.5 | 0.286 | 0.823 | 348 |
| 96 passes at factor 0.25 | 0.285 | 0.821 | 313 |
| 160 passes at factor 0.15 | 0.285 | 0.822 | 311 |
| 320 passes at factor 0.15 | 0.274 | 0.786 | 1,069 |

Two ways out were tried and both fail. Remeshing again after the smoothing does
not repair the topology and re-imposes the voxel stair-stepping the smoothing had
just removed, taking faceting from 0.286 back to 0.434. A volume-preserving
laplacian smooth is the principled alternative and is a silent no-op at this mesh
size — 24, 60 and 120 iterations returned byte-identical readings — so that
option was removed rather than shipped as a setting that does nothing.

So the Siamese stays a documented reject, and the conclusion points upstream: an
asset that needs this much smoothing needs a reconstruction with less surface
area, not a better reduction. The remesh face count at a fixed resolution is the
cheap way to see it coming, well before anything is rigged or rendered.
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
