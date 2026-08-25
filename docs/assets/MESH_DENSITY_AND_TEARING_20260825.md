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
spreads evenly. That is what rescues a mesh the plain route ruins. It is also the
wrong treatment for other animals, and the inversion is total rather than
marginal:

| breed | plain 25k | plain 80k | remesh 80k |
| --- | --- | --- | --- |
| Siamese | **0.24%** | 3.90% | 2.85% |
| Jack Russell | 1.27% | **0.47%** | 0.98% |
| dark Burmese | never reaches 25k | 7.39% | **1.52%** |
| standard Burmese | never reaches 25k | never reaches 80k | **0.66%** |

Shards at the worst frame of the walk; bold is the best option for that breed.
The same setting is a breed's best and another's worst. The two cats carry 1,986
and 4,171 non-manifold edges against the dog's 1,052, and at 25k the aggressive
collapse averages the damaged regions away while at 80k the damage survives into
the rigged mesh - which is also why the two dirtiest meshes cannot reach a low
budget at all.

So no fixed recipe can work, and the thing that travels between animals is the
gate, not the setting. `tools/assets/run_generated_animal_chain.sh` walks a
ladder, preparing, gating, rigging, animating and gating again at each rung:

| rung | preparation |
| --- | --- |
| 1 | plain weld and collapse to 25k |
| 2 | plain weld and collapse to 80k |
| 3 | voxel remesh at diagonal/800, collapse to 80k |
| 4 | voxel remesh at diagonal/700, collapse to 80k |
| 5 | voxel remesh at diagonal/800, collapse to 120k |

`--pick first`, the default, stops at the first rung that passes both gates.
`--pick best` runs the whole ladder and keeps the least-torn result, which costs
five rungs and is worth it for an asset going into production - on the Jack
Russell it is the difference between 1.27 and 0.47 percent, both acceptable.
Failed rungs keep their evidence.

`tools/assets/retopologize_for_rigging.py` performs a rung and reports what it
did, with `--skip-remesh` selecting the plain route. Both meshes stay in one
Blender session, because the colour bake needs the dense original as its source
and reloading it from a file that no longer carries the texture is the failure
this avoids. Every count is taken after welding, because a glTF file splits
vertices at each uv and normal seam and a boundary or island count read off a
freshly imported file measures the file format instead of the surface.

## The two gates, and why they gate what they do

The two failure modes do not appear in the same number, so one gate cannot catch
both. A standard Burmese whose face collapsed into flat facets tears *less* than
a Jack Russell that looks fine - 1.07 percent of posed area in shards against
1.27 - so no tearing threshold will ever catch it. It is caught before rigging
instead, by how the triangle budget was distributed.

**Before rigging**, `tools/assets/gate_retopology.py` judges only damage the
reduction itself did, because that is all it can judge route-independently:
head-third survival at 0.7, and whether the face target was reached at all.

Three readings that look like criteria are reported and *not* judged, each for a
measured reason:

- **Faceting** punishes a low face count. The coarsest version in this batch
  scored worst on it, 0.657, and is the one the owner called good; a denser one
  scored 0.383 and reads as speckled. Face count and dihedral angle are not
  independent.
- **Boundary and non-manifold edges** are driven to zero by a remesh, so there is
  nothing to check, and inherited untouched by the plain route, where they
  describe the reconstruction rather than a defect this step introduced. Two
  versions the owner accepted carry 1,052 and 1,986 non-manifold edges.
- **Fidelity** spans fivefold across versions that all look right.

**After rigging**, `tools/assets/gate_rigged_asset.py` decides whether the result
looks right, from `measure_walk_deformation.py`. Two earlier versions of that
measurement were wrong in ways worth recording. One sampled a single pose at 35
percent through the action and understated the worst frame by ten to thirteen
times. The other weighted everything by area, while the artifact that dominates
what a viewer sees is a shard - a triangle stretched into a long thin sliver that
fans open at an armpit or a hip and carries almost no area at all. Shards are now
found by longest-edge growth, over every sampled frame, and the worst frame
decides.

Calibrated against owner judgement on rigged versions at ordinary viewing
distance:

| version | shards at worst frame | owner |
| --- | --- | --- |
| Siamese, plain 25k | 0.24% | good |
| Jack Russell, plain 80k | 0.47% | - |
| standard Burmese, remesh 80k/700 | 0.72% | - |
| Jack Russell, plain 25k | 1.27% | acceptable |
| dark Burmese, remesh 80k/700 | 1.92% | acceptable |
| dark Burmese, remesh 80k/800 | 2.25% | - |
| dark Burmese, remesh 120k | 2.60% | - |
| dark Burmese, plain 80k | 3.15% | - |
| Siamese, remesh 80k | 3.71% | rejected |
| dark Burmese, plain 100k | 3.89% | - |

Any threshold between 1.92 and 3.71 fits those labels, so the rig's own variance
decides where inside that band to sit. Rigging the *same* prepared mesh four
times moves this reading about ten percent (2.16 to 2.63 on one preparation), and
the same rung of the same ladder measured 1.92 and 2.27 on two runs. The
threshold is 2.5: thirty percent of clearance above the highest accepted version,
thirty-three below the rejected one, which is more than the noise.

Two readings that looked like they belonged in the gate do not.

The **ten-times area share** drew the same line across fifteen versions and
looked like independent confirmation, until the variance test moved it from 0.54
to 0.97 percent on identical input - a 1.8x spread against the shard share's
1.2x. It was measuring the draw, and it had already produced one false reject on
a route the owner accepted. Worst edge growth is worse still, 11.5 to 25.6 on the
same input.

**Where the shards sit** should matter and does not help. Owner judgement is
explicit that tearing under the belly "is not a big problem" while the same
amount on a flank gets an asset rejected, so the measurement separates
downward-facing low faces from the rest. But on these labels the visible-only
reading separates *worse* than the total: 1.56x between accepted and rejected
against 1.94x. Both numbers are reported and the total decides. This was a
hypothesis with a good reason behind it that the measurement did not support.

The scale matters and is part of the calibration. Magnified four times, *every*
version in this batch shows tearing somewhere, usually at an armpit or a hip. A
threshold argued from magnified stills would reject everything and mean nothing.
These thresholds describe ordinary viewing distance, which is the scale the
assets are used at; recalibrate if the delivery scale changes.

## Measured across four breeds

Where the ladder settles, and what each breed rejects on the way:

| breed | settles on | shards | rejects | on what |
| --- | --- | --- | --- | --- |
| Siamese | rung 1, plain 25k | 0.24% | rungs 2-4 | 3.90% and 2.85% of shards, and head 0.66 at diagonal/800 |
| Jack Russell | rung 1, plain 25k | 1.27% | - | passes rung 1; rung 2 would be better at 0.47% |
| standard Burmese | rung 3, remesh 80k | 0.66% | rungs 1-2 | collapse stalls at 111,937 faces, head 0.54 |
| dark Burmese | rung 3, remesh 80k | 1.52% | rungs 1-2 | collapse stalls at 61,147 faces; then 7.39% of shards |

The two breeds that need the remesh are exactly the two carrying 7,082 and 4,171
non-manifold edges. The two the plain route handles carry 1,052 and 1,986.

## The face budget is bounded from both sides

Neither bound is a constant, and they pull in opposite directions.

From below, a head needs an absolute triangle count rather than a share, and a
large flank takes the budget first. At diagonal/800 the Siamese head keeps 0.59
of its fair share at 50k, 0.66 at 80k and 0.71 at 120k - the pre-rig gate rejects
the first two.

From above, density carries the reconstruction's damage into the rigged mesh
instead of averaging it away. On the plain route the two cats go from 0.24 and
"cannot reach" at 25k to 3.90 and 7.39 percent of shards at 80k.

Which bound binds depends on the breed, and both are measured rather than
assumed. That is the whole reason the ladder exists.

## Where the speckle came from, and why it is not an upstream problem

The Siamese at 80k retopologised reads as dark speckle over a pale coat, and
three explanations did not survive measurement. It is not the reconstruction: its
raw mesh is the *second smoothest* of the four, 12.1 percent of edges past thirty
degrees against the standard Burmese's 33.6, and renders clean untouched. It is
not the generated reference image, which is the smoothest and palest of the four
with less visible fur striping than either Burmese. It is not the bake: rendering
the retopologised mesh untextured shows the pitted surface directly, and
tightening the bake ray from 5 to 2 percent of the diagonal changed nothing, as
did a 4096 atlas and four times the gutter dilation.

What is different is surface area. At diagonal/700 the Siamese remeshes into
690,890 faces where the other three land at 383k-426k for the same bounding box,
so nearly twice the area is gentle undulation - shallow, which is why the raw mesh
reads smooth, but a great deal of it. A fixed triangle budget spread over twice
the area gives each triangle twice the curvature to cross, and that is the
faceting. The two Burmese, whose raw surfaces are much rougher, come out
*smoother* through the remesh route (33.6 to 23.3 percent, 21.4 to 10.6) while
the Siamese goes the other way, 12.1 to 38.3.

Every remedy inside the remesh route fails. More faces raises head survival and
makes faceting worse. More smoothing lowers faceting and squeezes the two walls
of a thin tail into each other - the self-intersection count tracks total
displacement rather than step size, 311 non-manifold edges at 160 passes of
factor 0.15 against 348 at 48 passes of 0.5. Remeshing again afterwards does not
repair that and re-imposes the voxel stair-stepping the smoothing had just
removed, taking faceting from 0.286 back to 0.434. A volume-preserving laplacian
smooth is the principled alternative and is a silent no-op at this mesh size - 24,
60 and 120 iterations returned byte-identical readings - so that option was
removed rather than shipped as a setting that does nothing.

The conclusion is not that the asset needs a better reconstruction. It is that
this breed does not want the remesh route at all: the plain collapse to 25k gives
the least-torn asset in the entire batch, 0.24 percent. An earlier version of this
document concluded the opposite and recommended fixing the generator. That was
wrong, and it was wrong because the recipe was being defended instead of measured
against the alternative.

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
