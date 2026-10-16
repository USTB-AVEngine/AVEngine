# ADR-0009: Single-View Multimodal Sensor Rig

- Status: Accepted
- Date: 2026-07-16

## Context

The current task contract has one task-facing camera viewpoint and one
listener, while audio emitters are independently named and positioned.
Habitat represents RGB, depth and semantic observations as separate sensor
specifications, so the phrase "multiple sensors" can be misread as multiple
camera viewpoints. Earlier legacy experiments also used four fixed views, and
top-down navigation maps are useful for QA, but neither is the official MVP
observation contract.

## Decision

Use one logical sensor rig, `camera_rig_0`, with exactly one formal dataset
`view_id`, `view0`, for M1, M2, M5 and the initial M6 MVP.

- The rig contains RGB, depth and semantic sensor specifications with the same
  world position, orientation and projection calibration. They are three
  modalities of one viewpoint, not three independently posed task cameras or
  three formal views. Habitat still implements them as three sensor specs.
- `world_from_rig` is that formal camera viewpoint, not an agent foot point.
  The M1 MVP therefore fixes `rig_from_sensor` and `rig_from_listener` to the
  identity transform; navigation test points are declared separately.
- All three observations are captured from one evaluated simulator state
  without a simulation or timeline step between modalities.
- The single listener, `listener0`, has the same world transform as
  `camera_rig_0` in the MVP profile.
- M1 treats `listener0` only as a transform anchor. It does not instantiate an
  AudioSensor, call RLR, render audio or claim an RIR; named multi-source RLR
  propagation is the M4 decision gate.
- There are at least two audio sources. Their names are unique and stable, and
  their world transforms are pairwise distinct. A source transform is never
  inferred from sensor count or overloaded as a camera/listener transform.
- Top-down navigation maps and other diagnostic artifacts are QA-only. They
  must be labeled as diagnostics and must not receive a formal `view_id`, enter
  the authoritative timeline, count as dataset observations or become
  training/benchmark inputs.
- Timeline v2 remains unchanged and retains an array of `view_ids` for future
  extensibility. The current semantic validator and MVP profiles impose the
  stricter rule `video.view_ids == ["view0"]` and require exactly the matching
  `view_pose_hashes` entry.

## Alternatives considered

- Four task-facing cameras at fixed cardinal headings.
- RGB-only capture from one camera.
- Treating a top-down navigation QA artifact as a second official dataset
  view.
- Allowing the listener pose to drift independently from the camera rig in the
  first MVP.

## Consequences

"Multimodal" and "multi-sensor" do not mean "multi-view" in current AVEngine
claims. RGB, depth and semantic payload hashes may differ by modality, while
their extrinsic state must agree. Camera/listener motion is coupled in the MVP;
named sources remain independently movable. Legacy four-view artifacts can be
retained only as migration evidence or optional diagnostics.

## Validation plan

For every formal capture, assert one `view_id`; matched RGB/depth/semantic
resolution, field of view and extrinsics; unchanged simulator/timeline state
across the three reads; equality of listener and rig transform hashes; and
unique source IDs with pairwise-distinct, round-trippable world transforms.
Verify that QA navigation artifacts are absent from timelines, admitted
observation manifests and benchmark inputs. For M1, also verify that no
AudioSensor or RLR execution is claimed; that capability belongs to M4.

## Reversal criteria

A future research requirement for viewpoint diversity may introduce a new
versioned capture profile and ADR with its own invariance and calibration
tests. Do not add formal views silently merely because timeline v2 can encode
more than one string.
