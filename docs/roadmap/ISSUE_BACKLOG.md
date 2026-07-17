# Issue Backlog

Each issue below is independently reviewable. `not_run` is an acceptable test
state only when the issue records the missing environment or prerequisite.

## M0-01: Repository governance and exact runtime lock

- Problem: repository identity, upstream base and dependencies were implicit.
- Scope: governance files, remotes, SHAs, submodules and lock validation.
- Non-goals: runtime feature changes.
- Dependencies: none.
- Deliverables: `UPSTREAM.md`, `MODIFICATIONS.md`, notices, lock and version files.
- Acceptance: recorded SHAs match Git; both worktrees are clean; lock parses.
- Not-run condition: none for Git checks; builds remain separate status entries.
- Documentation: repository boundaries and reproducibility guide.

## M0-02: Legacy inventory and migration policy

- Problem: AVEngine-owned code is mixed with ignored SPEAR and generated outputs.
- Scope: entrypoints, callers, I/O, ownership and keep/migrate/retire decisions.
- Non-goals: moving implementation code.
- Dependencies: M0-01.
- Deliverables: inventory, migration matrix and deprecation plan.
- Acceptance: every primary legacy path has a replacement and removal milestone.
- Not-run condition: media/GPU evidence unavailable is recorded, not inferred.
- Documentation: `docs/migration/*`.

## M0-03: Clean build and baseline report

- Problem: existing static tests do not prove a reproducible Habitat runtime.
- Scope: environment capture, clean configure/build attempt and minimal upstream tests.
- Non-goals: fixing unrelated upstream failures.
- Dependencies: M0-01.
- Deliverables: build log, environment manifest and verification table.
- Acceptance: result is reproducible and classified pass/fail/blocked/not_run.
- Not-run condition: missing compiler/GPU/data is identified with exact requirement.
- Documentation: runtime build/reproducibility guide.

## M1-01: Three-room visual loading canary

- Status: `pass`; see `M1_STATUS.md` and `M1_EXECUTION.md`.

- Problem: Habitat visual/room suitability is not yet demonstrated.
- Scope: Habitat room, Blender custom room and real-surface legacy apartment export.
- Non-goals: production acoustic propagation.
- Dependencies: M0-03.
- Deliverables: scene packages, load scripts and one-view RGB/depth/semantic evidence.
- Acceptance: repeatable loads, correct openings/connectivity, exactly one
  formal `view0` and recorded modality/scene hashes; top-down QA images do not
  count as dataset observations.
- Not-run condition: unavailable third-party scene is replaced only by an allowed sample.
- Documentation: room manifest examples and quality comparison.

## M1-02: Coordinate and one-state single-view multimodal contract

- Status: `pass`; see ADR-0009 and `M1_STATUS.md`.

- Problem: the single camera rig, listener and independent source transforms
  need one convention and state, without confusing sensor modalities for views.
- Scope: units, axes, one `camera_rig_0`, co-located RGB/depth/semantic
  calibration, one co-located `listener0`, independently named sources and
  capture without timeline advancement.
- Non-goals: dog skeletal playback.
- Dependencies: M1-01.
- Deliverables: transform contract, single-view multimodal capture API spike
  and parity report.
- Acceptance: `view_ids == ["view0"]`; all three sensors observe the same
  world state with matching extrinsics; listener and rig transforms match;
  named source transforms round-trip independently; diagnostic cameras remain
  outside formal manifests.
- Not-run condition: headless rendering unavailable is reported explicitly.
- Documentation: architecture and canary instructions.

## M2-01: Canonical dog package compiler

- Status: `pass` for the bounded v7/r5 research canary; see `M2_STATUS.md`.

- Problem: the reviewed Beagle needed a fail-closed path from research
  candidate to a hash-bound qualified runtime package.
- Scope: one audited template package with Walk/Idle, anchors, contacts and QA.
- Non-goals: arbitrary generated topology or five morphotypes.
- Dependencies: M0-02.
- Deliverables: package schema/example, offline compiler wrapper, and a bounded
  `canary_qualified` decision.
- Acceptance: all required hashes/revisions exist and asset QA passes for the
  M2 canary; dataset approval remains unavailable before M6.
- Not-run condition: Blender/GPU checks remain `not_run`, never auto-approved.
- Documentation: asset-package guide and provenance record.

## M2-02: Deterministic baked dog pose playback

- Status: `pass` for the clean 75-state single-view formal canary; see
  `M2_STATUS.md` and `M2_EXECUTION.md`.

- Problem: Habitat must execute exact animal poses without free-running animation.
- Scope: load M2-01 asset, apply 75 root/joint states, expose anchors/contacts/hash.
- Non-goals: online retarget or facial animation.
- Dependencies: M1-02, M2-01.
- Deliverables: runtime adapter, tests and single-view RGB/depth/semantic canary.
- Acceptance: the same-frame canonical pose and formal `view0` pose hash match
  across the co-located modality sensors without timeline advancement
  (rendered payload hashes are expected to differ by modality), and complete
  Walk/Idle QA passes.
- Not-run condition: unsupported skinned import is a recorded blocker with evidence.
- Documentation: runtime articulated-animal guide.

## M3-01: Acoustic Scene Package compiler and ingestion

- Status: implementation complete. Authoritative gate outcome:
  `M3_STATUS.md`; replay procedure: `M3_EXECUTION.md`.
- Problem: legacy AABB/implicit material paths cannot prove acoustic parity.
- Scope: explicit geometry/material package, strict source replay, modern RLR
  upload path, exact API receipts and post-ingestion geometry readback.
- Non-goals: dynamic body acoustics.
- Dependencies: M1-01.
- Deliverables: schemas, compiler, loader, provenance snapshots, debug mesh,
  modern runtime adapter and independent verifier.
- Acceptance: source GLB/mapping/database replay reproduces arrays, object
  partitions and resolved materials; every triangle is assigned without
  fallback; API receipts match object IDs/counts and per-material triangle
  counts; native geometry readback matches the canonical package.
- Not-run condition: RLR build unavailable is separated from compiler tests.
- Documentation: package format, material semantics, mapping confidence and
  ingestion evidence policy.

The post-ingestion OBJ has no recoverable per-face material-ID array. It verifies
geometry and resolved material blocks only; per-triangle identity is verified by
the hash-bound source replay and exact upload receipts. RLR remains the reused
propagation algorithm, while AVEngine owns explicit compilation and evidence.

## M3-02: Geometry leakage and material activation canary

- Status: implementation complete. Authoritative gate outcome:
  `M3_STATUS.md`; replay procedure: `M3_EXECUTION.md`.
- Problem: successful upload does not prove geometry or material effects.
- Scope: ray leakage, openings and high/low absorption comparison.
- Non-goals: perceptual room-quality benchmark.
- Dependencies: M3-01.
- Deliverables: metrics, artifacts and repeatability report.
- Acceptance: no unintended default material; CPU/native opening and control
  rays agree; direct arrival passes; synthetic high/low EDT, DRR and
  late-energy effects have the declared direction and exceed run variance.
- Not-run condition: acoustic runtime unavailable is `blocked`, not `pass`.
- Documentation: thresholds and canary commands.

The controlled custom-room coefficients are deliberate synthetic extremes for
activation testing, not physical material truth. MP3D and UE visual-slot
proposals remain `research_candidate` diagnostics and cannot satisfy this
formal controlled gate or room admission.

## M4-01: Modern RLR multi-source adapter

- Status: bounded `pass`; the retained formal evidence and focused
  native/Python tests close the software/source-pose gate. See `M4_STATUS.md`
  and `M4_EXECUTION.md`.
- Problem addressed: the stock/legacy Habitat AudioSensor path uses a
  deprecated single-source wrapper.
- Scope: explicit context lifecycle, canonical named source/listener
  realization, stable-ID updates, native endpoint receipts and all-pair owned
  IR access.
- Non-goals: a new propagation algorithm.
- Dependencies: M3-01.
- Deliverables: isolated fork adapter and C++/Python API tests.
- Acceptance: at least two named sources and exactly the one formal MVP
  listener are enumerated; native ID/index/pose/radius/orientation/layout/
  channel/HRTF receipts agree with the request; listener/camera-rig transforms
  agree; and every source-listener pair output has a valid owned shape and
  explicit layout metadata.
- Not-run condition: ABI/build failure is recorded with exact versions.
- Documentation: audio extension and attribution boundaries.

## M4-02: Source identity, order and stem invariance

- Status: bounded `pass`; the retained evidence passed the declared and
  independently recomputed order/stem/spatial/lifecycle checks.
- Problem addressed: multiple coordinates alone do not preserve dataset
  identity.
- Scope: actor/event/anchor mapping, exact caller-order invariance, per-source
  FOA/binaural stems and mixtures, reset/temporal policy, spatial probes,
  explicit HRTF binding and performance measurement.
- Non-goals: mixture model training.
- Dependencies: M4-01.
- Deliverables: identity manifest, invariance tests and performance report.
- Acceptance: source reordering preserves mapped full-indirect IRs exactly;
  stems reconstruct from dry audio and pair IRs; canonical mixtures equal the
  retained stem sum; raw FOA is `[W,Y,Z,X]` ACN/N3D `avengine_world`; explicit
  MIT KEMAR binaural output passes left/right probes; reset reproduces the
  initial temporal frame; and the one/multi-source performance report is
  complete.
- Not-run condition: missing native temporal/reset, HRTF, spatial-probe or
  independent artifact verification remains explicitly `not_run`/`blocked`.
- Documentation: source lifecycle and deterministic settings.

The checked-in M4 identity canary is bound to static formal M1 source poses.
Its M2 event-time dynamic-anchor evidence remains `not_run`, so neither issue
grants asset/dataset admission. M4 retains WAVs only. Exact timeline assembly,
counterfactuals and binaural video mux/readback are M5-01/M5-02 work.

## M5-01: Timeline builder, validator and fixed-state capture

- Problem: JSON Schema alone does not enforce cross-field synchronization.
- Scope: exact PTS/sample boundaries, references, single formal-view pose
  hashes and events.
- Non-goals: new timeline v3.
- Dependencies: M2-02, M4-02.
- Deliverables: builder, semantic validator and exact output verifier.
- Acceptance: 75/80,000/240,000 counts, exactly `view_ids == ["view0"]` with
  one matching `view_pose_hashes` entry per frame, and all cross-field
  invariants pass.
- Not-run condition: mux/codec checks absent are separately marked.
- Documentation: timeline examples and validation errors.

## M5-02: Anti-shortcut counterfactual pair

- Problem: source attribution needs controlled visual invariance.
- Scope: swap vocalizing actor while freezing the single formal view and its
  RGB/depth/semantic visual state.
- Non-goals: mouth animation.
- Dependencies: M5-01.
- Deliverables: paired episodes, visual hash proof and audio lineage.
- Acceptance: visuals match exactly and only declared variables differ.
- Not-run condition: either episode failing an M5 hard gate or failing to reach
  `canary_qualified` blocks the pair; formal dataset admission remains M6 work.
- Documentation: frozen/changed variable manifest.

## M6-01: Registry, QA aggregator and CLI

- Problem: files can currently exist without structured admission/rejection.
- Scope: versioned registries, request CLI, QA aggregation and provenance.
- Non-goals: web-scale orchestration.
- Dependencies: M2-01, M3-01, M5-01.
- Deliverables: stable CLI and registry APIs.
- Acceptance: failed gates resolve to structured rejection; `not_run` cannot approve.
- Not-run condition: human review remains pending rather than inferred.
- Documentation: CLI, status vocabulary and migration guide.

## M6-02: End-to-end Dog dataset canary

- Problem: component canaries do not prove a reproducible sample.
- Scope: two actor instances of one canonical Dog asset in a custom room, with
  exactly one formal `view0`, one co-located listener, at least two named
  sources and one counterfactual group, end to end.
- Non-goals: scale or throughput optimization.
- Dependencies: M6-01, M5-02.
- Deliverables: admitted sample, full manifests and deterministic rerun report.
- Acceptance: same request/seed reproduces compatible hashes and all hard gates pass.
- Not-run condition: any required component is reported at its real state.
- Documentation: end-to-end guide and known limitations.

## M7-01: Dynamic Articulated Source Attribution benchmark

- Problem: the dataset contribution needs a measurable downstream task.
- Scope: splits, loader, metrics and visual/audio/audio-visual baselines.
- Non-goals: claiming state-of-the-art before frozen evaluation.
- Dependencies: M6-02 and sufficient admitted samples.
- Deliverables: task API, baselines and ablations.
- Acceptance: identity-safe splits and reproducible metric outputs.
- Not-run condition: insufficient admitted data blocks training claims.
- Documentation: benchmark card and split provenance.

## M7-02: Research release and paper package

- Problem: code, data, licenses and claims must tell the same story.
- Scope: release manifests, notices, citations, paper tables and artifact audit.
- Non-goals: redistributing assets without rights.
- Dependencies: M7-01.
- Deliverables: release candidate, dataset card, citation files and paper artifacts.
- Acceptance: reused/extended/original statements and licenses pass final audit.
- Not-run condition: unresolved rights block affected artifacts from release.
- Documentation: final README, limitations and reproduction guide.
