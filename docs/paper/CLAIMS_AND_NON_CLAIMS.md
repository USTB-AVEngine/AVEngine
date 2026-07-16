# Claims and Non-Claims

## Claims supported at M0

- The project has selected Habitat-Sim as its visual, scene, sensor, physics,
  and articulation foundation and RLR as its geometric acoustic foundation.
- Exact Habitat and RLR commits are pinned in `runtime.lock.yaml`.
- Main-engine and runtime-fork ownership are separated and documented.
- An integer timeline schema and M0-M7 evidence-gated roadmap exist.
- Legacy implementations have been inventoried and classified for migration.

These are repository/design claims. They are not runtime-quality or dataset
performance results.

## Planned claims requiring later evidence

- deterministic 75-pose articulated playback and same-state multi-view capture;
- real-surface acoustic packages with verified per-triangle materials;
- one-context named multi-source/listener RLR and per-pair IR/stems;
- exact 48 kHz tick, 15 fps frame, and 16 kHz audio synchronization;
- visually identical counterfactual pairs and identity-preserving labels;
- repeatable end-to-end QA, provenance, rejection, and admission;
- benchmark quality, generalization, and ablation outcomes.

Each remains `not_run` until the named canary executes. A code path existing is
not equivalent to its scientific claim passing.

## Non-claims

AVEngine does not claim that it:

- creates a new visual renderer or geometric acoustic ray tracer;
- implements a simulator from scratch;
- is the first system to expose multiple RLR sources/listeners;
- adds articulated objects to Habitat as a new general capability;
- proves visual materials are acoustic ground truth;
- supports complete dynamic-body acoustics, visible mouth articulation, or
  stable animation for arbitrary generated meshes;
- has completed the Habitat-native runtime, any M1-M7 canary, or a released
  dataset at M0;
- makes every dependency, model, asset, or output MIT or commercially cleared;
- is endorsed by Meta, Tencent, Epic, or any other upstream provider.
