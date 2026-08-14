# Legacy Deprecation Plan

Status: historical M0--M7 sequence. ADR-0010 supersedes its repository and
production-backend assumptions. It remains useful for interpreting old
milestone records, but it is not the active single-repository execution plan;
see `SINGLE_REPOSITORY_EQUIVALENCE_PLAN.md`.

Deprecation is milestone-gated. Old evidence is preserved at immutable Git
commits; an entrypoint stops being production-supported before it is archived.

## M0 — freeze and classify

- Freeze exact legacy AVEngine and SPEAR SHAs.
- Stop presenting SPEAR/gpuRIR scripts as the primary architecture.
- Prohibit `promote_source_asset.py` from granting new-system approval.
- Quarantine raw generated-topology routes and all untracked legacy artifacts.
- Treat all old approval states as pending revalidation.

## M1 — Habitat visual and room canary

- Make Habitat the default visual/room canary.
- Keep UE rendering only as an optional comparison backend.
- Restrict apartment AABBs to debug/placement broad phase and begin real-surface
  export.

## M2 — articulated dog runtime

- Replace absolute Quaternius/species maps with one canonical Dog AssetPackage.
- Replace rigid-object dogs and freely running UE actions with exact baked poses.
- Remove legacy UE import/render success from M2 canary qualification criteria.

## M3 — acoustic scene and materials

- Ban AABB apartment acoustics in production.
- Remove adapters that generate but do not upload per-triangle material indices.
- Require AcousticScenePackage as the sole production acoustic input.

## M4 — multi-source RLR

- Replace sequential `setAudioSourceTransform()` sweeps with one modern RLR
  context containing named sources/listeners and per-pair IRs.
- Stop using `run_audio_pass_rlr.py` and `run_habitat_all.py` as primary paths.

## M5 — timeline and counterfactual episode

- Require all render, pose, audio, event, and mux consumers to read timeline v2.
- Remove fixed 1,067-sample slicing, free animation clocks, view-outer capture,
  root-height emitters, and mux-only duration assumptions.
- Use semantic bone anchors and verify 75 frames/80,000 samples by readback.

## M6 — dataset MVP

- Replace legacy runners with one stable `avengine` CLI and unified registries.
- Make legacy registries read-only after explicit importers are tested.
- Centralize QA aggregation, structured rejection, and Dataset Admission.

## M7 — benchmark and release

- Decide whether UE and gpuRIR remain maintained comparison backends.
- Archive unused raw-topology, human/Rocketbox, and old review-server paths.
- Retain only what a frozen benchmark, released dataset, or paper artifact needs.

No milestone authorizes deletion of the legacy repositories or ignored local
evidence. Destructive cleanup requires a separate, explicit decision after
release artifacts and provenance are independently verified.
