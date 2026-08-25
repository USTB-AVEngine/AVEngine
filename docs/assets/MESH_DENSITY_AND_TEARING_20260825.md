# Mesh density, tearing, and how to generalize across animals (2026-08-25)

A generated quadruped tears open during a walk cycle. This records what the
tearing actually is, what fixes it, and why a single face-count target does not
generalize. Every number here was measured on this batch, not estimated.

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

## What fixes it, and the trap

Reducing density before rigging is the only lever that moved the needle:

| Jack Russell | area stretched >4x | >10x | worst face growth |
| --- | --- | --- | --- |
| 926k faces as generated | 0.248% | 0.041% | 286x |
| joint weight smoothing, default | 0.196% | 0.030% | 463x |
| joint weight smoothing, aggressive | 0.160% | 0.022% | 443x |
| decimated to 60k | 0.128% | 0.020% | 81x |
| decimated to 25k | 0.191% | 0.011% | 25x |
| decimated to 50k | 0.176% | 0.023% | 33x |
| decimated to 80k | 0.165% | 0.015% | 49x |

Weight smoothing after decimation makes it worse, not better: at low density the
smoothing over-blends and returns spikes. Pick one, not both.

**The trap:** the same decimation ruined a different animal. The standard
Burmese kept its head at full density and lost it entirely at 112k -- the face
collapsed into flat facets while the tail region did not reduce at all.

## Why, and the rule that generalizes

Welding each mesh inside one Blender session (a glTF round trip splits vertices
at every uv seam, so boundary and island counts measured after re-import
describe the file format, not the surface):

| welded | non-manifold edges | real holes | islands |
| --- | --- | --- | --- |
| Jack Russell | 1,052 | 405 | 74 |
| standard Burmese | 7,082 | 2,247 | 317 |

A collapse decimator cannot touch a non-manifold edge. With seven thousand of
them the whole reduction budget is spent on the regions that *are* collapsible,
which are the clean, detail-dense ones: the head. The tail, where 60 percent of
this mesh's non-manifold edges sit, does not move, so the target is never
reached either.

So the rule is not a face count. It is:

1. weld and measure the topology (`tools/assets/measure_mesh_topology.py`);
2. if non-manifold edges are low, decimate to 25k-50k and rig -- deformation
   comes out clean;
3. if they are high, decimation will eat the detail you care about. Repair the
   topology first; do not trade the head for a face count.

For this batch that threshold sits somewhere between 1k and 7k non-manifold
edges. Widen the sample before hard-coding a number.

## Where the non-manifold edges come from

Binned along the body axis, 60 percent of the Burmese's non-manifold edges are
in the rear third, peaking at the tail. The pose guide holds the tail up and
close to the body, and reconstructing a thin structure that nearly touches the
torso produces self-intersections. The shared negative prompt forbids a tail
touching or overlapping a *leg*; it says nothing about the body or the back.
That is a one-line extension worth making before the next generation run.

## Review rendering

The blotches read very differently under different lighting: the same posed
frame is clean under the static review renderer and blotchy under the animation
renderer. Soft, shadow-free lighting shows the asset honestly
(`tools/assets/render_walk_review.py`, `render_turntable_review.py`). Judge by
image, not by metric: the 25k asset scores a worse worst-face-growth than the
60k one on some runs and still looks clearly better, and the same density rigged
twice scored 25 and 126.

## Heading

`blender_estimate_generated_animal_forward.py` was wrong on one of four animals,
and its confidence was anti-correlated with correctness: the failure scored 0.82
and the lowest-confidence pass, 0.17, was right. The failure mode is a 180
degree head-tail swap. Render `tools/assets/probe_heading_axis.py` after
normalization and look at it -- a face means forward, a tail means backwards.
The contract already demands a reviewed yaw and review evidence for this step.
