# MP3D real-audio refusal mini-pilot checkpoint — 2026-08-25

## Scope and boundary

This checkpoint runs the existing 10–15 percent refusal policy on current
Habitat-native visual evidence and one fresh RLR audio render. It does not
claim an accepted multi-view production batch, trainer consumption, exact-48
selection, or dataset admission.

Implementation commit:

`a897ea9 feat(qa): run MP3D audio-only refusal quota`

All outputs remain `research_only=true`, `episode_counted=false`, and formal
dataset count zero. The builder appends refusal rows and verifies that the
entire base-question prefix is unchanged.

## Current native inputs

Visual input:

`/data/jzy/tmp/mp3d_target_only_wip_region005_p011_gpu0_r1`

This is the region-005 both-moving capture from the same-renderer target-only
checkpoint. It carries exact normal/target actor, camera, and sensor readback
plus per-frame pixel visibility truth.

A new randomized AudioProgram was authored for this viewpoint under:

`/data/jzy/tmp/mp3d_region005_view15758_qa_design_d5e1036_r1`

The both-moving point was then rendered through a fresh 25-keyframe RLR
sequence into:

`/data/jzy/tmp/mp3d_region005_p011_refusal_audio_d5e1036_r1`

Its receipt binds the exact target-only visual capture and M1 listener. It
records `prior_listener_rir_reused=false`, 80,000 samples at 16 kHz, six
events, and trajectory identity
`6a1f36bd9cc285d9d127cfa0e9d78fdcf8c21a5eafac134e6f36fc5fe5ea63b1`.

Independent waveform verification:

`/data/jzy/tmp/mp3d_region005_p011_refusal_audio_d5e1036_verification_r1.json`

The verifier reloaded the WAVs, checked the receipt/visual/timeline facts,
confirmed all declared outputs exist, and measured:

- binaural mixture peak: `0.0527534`;
- maximum mixture-versus-two-stem sum error: `3.3689e-9`.

The RLR process completed and wrote the full receipt and WAV set, but its SSH
wrapper did not receive EOF from an inherited output descriptor after the
remote Python process exited. The local wait was interrupted only after the
server showed no matching process. Therefore this checkpoint relies on the
independently reloaded receipt and waveforms, not on that wrapper's final exit
status.

## Refusal construction

The two unchanged base documents are:

- 12 region-000 TA/T7/T5 questions:
  `/data/jzy/tmp/mp3d_region000_view01246_qa_final_412348e_r1/questions.json`;
- two region-005 same-renderer off-screen questions:
  `/data/jzy/tmp/mp3d_region005_p011_offscreen_minipilot_d5e1036_r1/questions.json`.

This produces 14 ordinary questions. The current pixel truth yields two
refusal candidates, one per actor. The existing deterministic quota selector
selects both because `2 / (14 + 2) = 0.125`, inside the required 10–15 percent
range.

Each refusal asks for the actor's camera-image visibility state while explicitly
providing only the real audio and withholding video. The gold answer is
`cannot determine from audio alone`; the hidden visual fact is retained only
for audit:

- actor0/source1, frame 9: `visible_clear`, 6 visible / 6 target pixels;
- actor1/source2, frame 63: `visible_occluded`, 3 visible / 10 target pixels.

Both rows carry:

- `provided_modalities=[audio]`;
- `withheld_modalities=[video]`;
- `modality_mask={audio: true, video: false}`;
- `refusal_gold=true`;
- a structural answerability certificate naming same-renderer camera pixels as
  the required withheld fact.

This is a structural modality claim, not a claim that a learned model failed.
The requested fact is defined by the withheld video/pixel-visibility authority;
the real audio input exists but is not treated as containing camera raster
state.

## Final artifact and balance

Final committed-code artifact:

`/data/jzy/tmp/mp3d_refusal_minipilot_a897ea9_r1`

It is byte-identical to the pre-commit tool run. The final document reports:

- 14 unchanged base questions;
- two selected refusal rows;
- 16 total questions;
- refusal fraction `0.125`;
- combined answer-balance status `pass`.

Binary rows remain balanced by type: TA and T7 retain 2:2 displayed keys and
off-screen retains 1:1. T5 and refusal rows remain explicitly non-binary and
are not coerced into A/B counts.

## Classifier boundary

An attempted extra WAV SHA-256 verification inside the refusal builder was
rejected because no concrete failure justified adding another hash gate. It
was not bypassed. The builder uses the existing audio receipt identity and
requires every declared output file; the separate runtime verifier records the
waveform and existing-receipt evidence.

## Tests

Focused refusal, off-screen, extended-question, generator, modality-mask, and
answer-balance tests report:

```text
61 passed
```

The full source-branch unit regression at `a897ea9` reports:

```text
3366 passed, 118 skipped, 52 subtests passed in 346.66s
```

Log: `/data/jzy/tmp/data_diversification_a897ea9_full_unit_r1.log`.
Ruff check/format and `git diff --check` pass. Clean-integration merge and the
fresh accepted multi-view/exact-48 data closure remain pending.
