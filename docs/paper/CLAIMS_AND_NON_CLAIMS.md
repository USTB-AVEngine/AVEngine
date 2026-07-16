# Claims and Non-Claims

## Claims supported through M1

- The project has selected Habitat-Sim as its visual, scene, sensor, physics,
  and articulation foundation and RLR as its geometric acoustic foundation.
- Exact Habitat and RLR commits are pinned in `runtime.lock.yaml`.
- Main-engine and runtime-fork ownership are separated and documented.
- An integer timeline schema and M0-M7 evidence-gated roadmap exist.
- Legacy implementations have been inventoried and classified for migration.
- One formal `view0` can capture co-located RGB/depth/semantic observations
  and a co-located listener pose anchor without advancing state in three room
  types: official Habitat, Blender custom and audited legacy UE real surfaces.
- At least two independently named source transforms round-trip in each M1
  room. This is a pose/identity result, not a multi-source RLR result.
- M1 artifacts, raw arrays, runtime identity, geometry provenance and a second
  process rerun are checked by an executable evidence verifier.

These are repository/design and bounded M1 canary claims. They are not animal,
audio-propagation, dataset-performance or benchmark results.

## Planned claims requiring later evidence

- deterministic 75-pose articulated playback using M1's same-state,
  single-view co-located RGB/depth/semantic contract;
- real-surface acoustic packages with verified per-triangle materials;
- one-context named multi-source RLR with one camera-co-located MVP listener
  and per-source/listener-pair IRs/stems;
- exact 48 kHz tick, 15 fps frame, and 16 kHz audio synchronization;
- visually identical counterfactual pairs and identity-preserving labels;
- repeatable end-to-end QA, provenance, rejection, and admission;
- benchmark quality, generalization, and ablation outcomes.

Each remains `not_run` until its named later canary executes. A code path
existing is not equivalent to its scientific claim passing.

## Non-claims

AVEngine does not claim that it:

- creates a new visual renderer or geometric acoustic ray tracer;
- implements a simulator from scratch;
- is the first system to expose multiple RLR sources/listeners;
- provides multiple formal camera viewpoints in the first MVP; its top-down
  images are QA-only, and its RGB/depth/semantic sensors are modalities of one
  `view0` rather than separate views;
- adds articulated objects to Habitat as a new general capability;
- proves visual materials are acoustic ground truth;
- supports complete dynamic-body acoustics, visible mouth articulation, or
  stable animation for arbitrary generated meshes;
- has completed articulated, acoustic, timeline, dataset or benchmark gates
  M2-M7, or released a dataset;
- makes every dependency, model, asset, or output MIT or commercially cleared;
- is endorsed by Meta, Tencent, Epic, or any other upstream provider.
