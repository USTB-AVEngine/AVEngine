# MP3D Habitat-native viewpoint prefilter

## Checkpoint 2026-08-24: CPU candidate authoring slice

This slice adds a deterministic, CPU-only MP3D viewpoint candidate selector and
a fresh/no-clobber builder. It follows the existing current-MP3D route's
coordinate, camera-yaw and conservative analytic-frustum logic. It does not use
Studio, Unreal, Blender, or Skokloster.

Inputs are intentionally ordinary retained artifacts rather than a new frozen
contract:

- one validated Habitat-native MP3D room manifest and base M1 request;
- one retained current-route explanation containing the actor paths and its
  prior single-island native PathFinder evidence;
- one ordinary JSON file containing camera-floor samples exported from the
  room's declared Habitat navmesh, with the room id and a plain authority note.

The selector derives route focus anchors, three camera-height variants and yaw
from those inputs. It scores analytic navmesh-sample visibility, per-actor route
coverage, visible path segments and lateral separation. Greedy selection rejects
near duplicates only when position, circular yaw, coverage signature and path
signature are all near. Each selected view keeps the exact visible frame ranges
and segment indices for every actor and receives an independent formal `view0`
M1 camera/listener request.

The builder writes one sibling staging directory and publishes it with the
repository's existing Linux `renameat2(RENAME_NOREPLACE)` helper. The output is
always marked:

- `research_only=true`;
- `episode_counted=false`;
- formal release not run;
- candidate-floor native readback pending;
- native Habitat RGB/depth/semantic pixels pending;
- first-person certification pending;
- four-point mini-pilot pending.

No native/GPU dry build was run in this checkpoint. The current retained route
stores a feasibility summary but not the complete camera-floor sample bank, and
all GPUs were occupied. Inventing floor coordinates from component bounding
boxes would not be authoritative. A later native-authority export must create a
fresh sample file; then the CPU builder can run without starting Habitat.

Focused verification on the server covers deterministic output under reordered
floor samples and actor-map order, 4/6/8 selection, near-duplicate suppression,
tiny-authority failure, explicit per-view route subsets, room-authority binding,
formal co-located `view0` requests, pending claim fields and fresh/no-clobber
publication.

This checkpoint is not completion of the MP3D work-order item. Still required
for every retained view are declared-navmesh readback, Habitat-native Topdown,
RGB/depth/semantic first-person frames, agent visual inspection and the fresh
four-point visual/audio/question/gate mini-pilot. No viewpoint from this slice
is certified or eligible for production merely because the CPU prefilter passes.

## Follow-up: live floor-sample exporter

`tools/m5/export_mp3d_camera_floor_samples.py` now implements the missing native
export path with explicit installed-runtime and external-root arguments. It
loads the validated room's declared navmesh in a real Simulator and records
per-sample PathFinder snap/island/clearance evidence. See
`docs/roadmap/MP3D_FLOOR_SAMPLE_EXPORT_20260824.md`. The code and CPU mocks do
not themselves claim a native run; native execution remains pending a safe
device/process preflight.
