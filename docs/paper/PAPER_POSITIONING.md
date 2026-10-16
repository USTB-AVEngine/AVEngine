# Paper Positioning

AVEngine is a Habitat-native audiovisual **dataset engine**, not a simulator
built from scratch. Habitat-Sim supplies scene representation, rendering,
sensors, navigation, physics, and articulated-object infrastructure. RLR Audio
Propagation supplies the geometric acoustic propagation foundation associated
with SoundSpaces 2.0.

The intended AVEngine contribution is the contract that connects audited
non-human articulated assets, explicit acoustic scenes, deterministic runtime
state, identity-preserving sound events, exact frame/sample timing,
counterfactual episode generation, QA, provenance, and dataset admission.

At M0 these are architecture and roadmap commitments, not completed empirical
results. The paper may change them to completed-tense claims only after the
corresponding milestone canaries pass on frozen commits and their evidence is
retained.

## Intended research object

The target task is Dynamic Articulated Source Attribution: identify which
visible articulated actor emitted which acoustic event while actor motion,
camera view, listener pose, room response, and distractor sources are
controlled. The core counterfactual holds visual frames fixed and changes only
the declared vocalizing actor/event/audio variables.

## Reuse, extension, and original integration

- **Reused:** Habitat rendering/scene/sensor/articulation foundations and RLR
  geometric propagation; each retains upstream history, licenses, and citations.
- **Planned runtime extensions:** exact baked non-human pose playback, explicit
  acoustic-scene ingestion with verified triangle materials, and a modern RLR C
  API adapter exposing named source-listener pair results.
- **Planned AVEngine layer:** audited package compilers, integer timeline,
  identity/event/anchor contracts, deterministic counterfactuals, QA,
  provenance, registry, and admission.

The project does not require novelty claims about a new rasterizer, physics
engine, ray tracer, or the existence of multi-source primitives in RLR.
