# Claims and Non-Claims

## Claims supported through M2

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
- One exact Rocketbox Beagle package passed automatic motion/deformation QA,
  hash-bound user visual review, MIT source-use review and a four-paw
  world-contact/root-cadence gate.
- That package is `canary_qualified` for bounded research-canary use and
  completed 75 explicit Idle/Walk states in a clean Habitat run on the same
  single-view co-located RGB/depth/semantic contract without advancing world
  time.

These are repository/design and bounded M1/M2 canary claims. M2 is one fixed
dog asset, not arbitrary animal generation or formal dataset admission. No
audio-propagation, dataset-performance or benchmark result is claimed.

## Planned claims requiring later evidence

- real-surface acoustic packages with verified per-triangle materials;
- one-context named multi-source RLR with one camera-co-located MVP listener
  and per-source/listener-pair IRs/stems;
- exact 48 kHz tick, 15 fps frame, and 16 kHz audio synchronization;
- visually identical counterfactual pairs and identity-preserving labels;
- repeatable end-to-end QA, provenance, rejection, and admission;
- benchmark quality, generalization, and ablation outcomes.

Each remaining item stays `not_run` until its named later canary executes. A
code path existing is not equivalent to its scientific claim passing.

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
- has completed acoustic, timeline, dataset or benchmark gates M3-M7, or
  released a dataset;
- makes every dependency, model, asset, or output MIT or commercially cleared;
- is endorsed by Meta, Tencent, Epic, or any other upstream provider.
