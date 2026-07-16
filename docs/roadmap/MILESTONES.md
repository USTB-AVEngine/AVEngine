# AVEngine Milestones

Milestones are sequential evidence gates. Later milestones may be designed in
parallel, but they cannot claim completion before their dependencies pass.

## M0: Repository and Baseline

Deliverables: two-repository governance, exact upstream/submodule lock,
architecture and ADRs, legacy migration matrix, attribution/citations, build
instructions, issue backlog and baseline status table.

Exit criteria:

- Both repositories have explicit origin/upstream roles and clean feature branches.
- Runtime, upstream and RLR commits are pinned.
- Legacy entries have an owner and migration decision.
- Unexecuted GPU/Blender/RLR/E2E checks are recorded as `not_run`.
- The reference fork builds cleanly with audio enabled and the relevant
  original Habitat tests have exact recorded results. A real upstream failure
  may remain `fail`, but M0 cannot silently relabel it `not_run`.

## M1: Habitat Visual and Room Canary

Deliverables: one Habitat-native room, one Blender custom room and one
legacy-apartment real-surface export; RGB/depth/semantic multi-sensor capture;
coordinate/unit manifests and visual evidence.

Exit criteria: all three room types load reproducibly; custom openings and
connectivity are preserved; camera/listener/source transforms agree; visual
quality is sufficient for the task or a bounded optional-backend gap is recorded.

## M2: Articulated Dog Runtime

Deliverables: one `canary_qualified` canonical dog package, baked Walk/Idle poses, root
trajectory, semantic anchors, contacts and canonical pose hashes.

Exit criteria: exactly 75 poses execute without a free-running action clock;
all views share the same per-frame pose hash; deformation/contact QA passes;
visual mouth articulation is absent. This does not grant
`approved_for_dataset`; central dataset admission remains M6 work.

## M3: Acoustic Scene and Materials

Deliverables: Acoustic Scene Package schema/compiler, explicit adapter
ingestion, material coverage, exported debug mesh and extreme-material canary.

Exit criteria: every production triangle is assigned; no unintended fallback
is used; openings/geometry survive; absorption extremes create a repeatable
RIR/EDT/DRR difference; production uses no AABB room proxy.

## M4: Multi-Source RLR

Deliverables: modern RLR C API adapter, named sources/listeners, per-pair IRs,
independent stems, reset/temporal policy and performance report.

Exit criteria: at least two sources maintain actor/event/anchor identity;
source registration order does not create a systematic output change; each
pair result is independently readable.

## M5: Timeline and Counterfactual Episode

Deliverables: timeline builder/semantic validator, deterministic fixed-state
capture, exact frame/sample assembly and vocalizing-actor swap pair.

Exit criteria: 75 frames, 80,000 samples and 240,000 ticks read back exactly;
the counterfactual pair has identical visual hashes; only declared audio/source
variables change; no mouth motion is present.

## M6: Dataset MVP

Deliverables: stable CLI, asset/scene/episode registries, QA aggregation,
provenance manifests, structured rejection and deterministic rerun.

Exit criteria: two actor instances of one canonical Dog asset + custom room +
at least two named sources are admitted end to end; the same request/seed
reproduces compatible timeline/manifests; `not_run` cannot be promoted to
`pass`.

## M7: Benchmark and Paper Release

Deliverables: Dynamic Articulated Source Attribution task, splits, loaders,
baselines, ablations, metrics, release manifests and paper artifacts.

Exit criteria: visual/audio/audio-visual baselines and counterfactual/sync/anchor
ablations run on frozen splits; reused/extended/original claims and all required
citations are consistent across code, dataset card and paper.
