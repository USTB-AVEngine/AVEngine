# MP3D multi-view finalizer and dry exact48 checkpoint — 2026-08-25

## Scope and boundary

This checkpoint fixes the finalization code and proves pre-audio exact48
feasibility for the current 160-point MP3D design. The dry selection is not the
final held-out and must not be consumed by training. Current target-only and
audio batch execution remains separate runtime work.

Implementation commit:

`319d091 feat(qa): finalize MP3D multiview partitions`

The finalizer accepts only points with both same-renderer target-only truth and
a fresh RLR receipt bound to the same visual capture and AudioProgram. It then:

- preserves and binds each point's TA/T7/T5 rows;
- materializes selector-compatible point specs with region, actual placement,
  source assets, motion case, program and `twin_of`;
- evaluates every target-only off-screen candidate, retains all candidate
  records, and selects the maximum deterministic exact left/right subset;
- builds one audio-only visual-state refusal candidate per point at its T5 event
  frame and applies the existing 10–15 percent quota selector;
- reruns per-type answer-key balance; and
- publishes through the existing atomic no-replace directory mechanism.

It adds no hash, frozen baseline or dataset-admission gate.

## Complete 40-view design

Fresh design root:

`/data/jzy/tmp/mp3d_region_40view_qa_design_1540e5c_r1`

It was generated from the retained native160 batch without rewriting that
batch. The result contains:

- 10 region instances;
- 40 camera views;
- 160 points, four motion cases per view;
- 160 unique randomized AudioProgram IDs;
- 480 questions: 160 TA, 160 T7 and 160 T5;
- 40/40 view-level answer-balance reports passing.

All 40 view documents correctly remain
`pending_randomized_audio_render`; this is design evidence, not an accepted
batch.

## Dry exact48 feasibility

Dry-only selector inputs:

`/data/jzy/tmp/mp3d_heldout48_dry_inputs_1540e5c_r1`

Dry-only selector result:

`/data/jzy/tmp/mp3d_heldout48_dry_selection_1540e5c_r1`

Every provisional spec explicitly records target-only and randomized audio as
pending and carries `dry_selection_only=true`. The existing exact48 MILP
nevertheless proves that the planned metadata/answer layout can support a
whole-point partition:

- population: 160 points / 480 questions;
- held-out: exactly 48 points / 144 questions;
- train-eligible: 112 points / 336 questions;
- point intersection: empty;
- question intersection: zero;
- question partition: complete;
- split twin groups: none;
- TA held-out A:B = 24:24 and train A:B = 56:56;
- T7 held-out A:B = 24:24 and train A:B = 56:56;
- four motion cases: exactly 12 held-out points each;
- all 40 actual placement IDs represented by one or two held-out points;
- region allocation: five points for regions 000–007 and four points for
  regions 008–009.

This is not final because the source population has not yet passed target-only,
RLR, off-screen/refusal finalization. The output's `research_only` label and
the external dry-only marker do not override that missing runtime authority.

## Exposed source-diversity limitation

The dry selector also makes one important deficiency explicit: all 160 points
belong to one `beagle × beagle` asset-pair category. Therefore this population
cannot satisfy a held-out stratification claim over multiple actor types.

The current Burmese cat artifact was audited read-only. Its latest manifests
remain `pipeline_shakedown_candidate` or
`technical_spike_only_pending_deformation_and_visual_qa`; there is no current
Habitat/SPEAR stage registration, emitter endpoint or runtime receipt. It is
not counted or inserted as a fallback.

## Long-running native jobs

Target-only160 fresh root:

`/data/jzy/tmp/mp3d_region_target_only160_1540e5c_r1`

The sequential runner uses explicit GPU0, checks that device before every
point, writes one log/progress row per point, stops on the first failure, and
never replaces a prior output. A separate independent verifier waits for batch
completion and will compare every shared normal-pass array and frame record
against the retained native160 source, recompile all target-only truth, and
summarize off-screen candidates.

Audio160 planned root:

`/data/jzy/tmp/mp3d_region_audio160_ef26330_r1`

Its low-priority waiter creates no output until target-only160 has a passing
receipt. It will then run exactly one fresh RLR process at a time, bind each
receipt to its target visual and unique AudioProgram, and stop at the first
failure. No persistent RIR cache or parallel lane is used.

## Tests

Focused finalizer, refusal, off-screen, heldout-selector and answer-balance
tests report:

```text
28 passed
```

The full source-branch regression at `319d091` reports:

```text
3368 passed, 118 skipped, 52 subtests passed in 304.36s
```

Log: `/data/jzy/tmp/data_diversification_319d091_full_unit_r1.log`.
The code is merged into the diversification integration branch as `ef26330`.
Real finalization, final exact48 selection, source-type expansion and clean
integration regression after runtime completion remain pending.
