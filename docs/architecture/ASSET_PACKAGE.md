# Canonical Animal Asset Package

## Authority model

Production assets are template-authoritative:

- Template: topology, vertex order, UVs, semantic skeleton, weights,
  collision proxies and declared action family.
- Generated guide: requested shape and PBR appearance proposal only.

Unknown generated topology directly bound to a borrowed rig remains an
experimental route and cannot enter the production registry automatically.

## Package layout

```text
animal_asset/
  asset_manifest.json
  provenance_manifest.json
  visual.glb
  skeleton.json
  skinning_manifest.json
  emitter_anchors.json
  collision_proxy.glb
  actions/
    action_manifest.json
    idle.npz
    walk.npz
  contacts/contact_phases.json
  textures/
  qa/
    static_geometry.json
    deformation.json
    animation.json
```

## Required identity and revision fields

- `asset_id`, `template_id`, `body_plan_id`, `morphotype_id`.
- Topology, UV, mesh and texture hashes.
- Skeleton, weights, collision and action revisions.
- Semantic emitter anchors for head/muzzle, paws and body.
- Guide provider/model/revision, inputs, seeds and commands.
- Source and derivative licenses and allowed-use classification.
- QA status and the evidence artifacts supporting admission.

## Compilation stages

```text
reference/attributes
-> guide generation and static QA
-> body-plan and morphotype classification
-> audited template selection or OOD rejection
-> constrained template fitting
-> PBR transfer to fixed template UV
-> baked template-native actions and contact phases
-> deformation/contact review
-> package hashing and registry admission
```

## Runtime contract

M2 Habitat canaries receive only a `canary_qualified` package and exact
root/joint poses evaluated at declared timeline ticks (initially the 75 video
frame PTS values). Dataset production accepts only `approved_for_dataset`,
which is unavailable before central M6 admission.
It does not perform online arbitrary-mesh rigging or general retargeting. A
runtime loader must expose semantic bone anchors, contacts and a canonical pose
hash without advancing the official episode clock during sensor capture.

## Hard rejection examples

- Missing limbs, floor/background fusion or severe bridges.
- Cross-limb forbidden weights or joints outside semantic limb volumes.
- Triangle flips, self-intersections, limb crossings or joint-limit failures.
- Paw penetration, contact-phase sliding or implausible hovering.
- Template out-of-distribution fit forced past the allowed morphotype domain.
- Missing provenance, unclear redistribution rights or unexecuted required QA.
