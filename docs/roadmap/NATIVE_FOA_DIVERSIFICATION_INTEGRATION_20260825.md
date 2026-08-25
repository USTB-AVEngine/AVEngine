# Native FOA × data-diversification integration checkpoint — 2026-08-25

## Scope and branch state

The native FOA dual-output branch now contains the current data-diversification
integration tree at merge commit:

`ab48cbc Merge data diversification into native FOA dual`

This merge is intentionally retained on the independent
`codex/native-foa-dual` worktree while the existing binaural-only audio160 job
continues from its frozen `efd15c4` integration checkout. The running batch is
not allowed to change audio layout halfway through. Nothing was merged to
`main`.

## Conflict resolution

Only three files required manual resolution. The result preserves both sides'
current authority:

- `current_mp3d_dynamic_audio.py` keeps strict AudioProgram event/timeline
  validation, explicit visual-axis-to-audio-endpoint identity, custom M1
  listener IDs, native endpoint registration evidence, and materialized event
  counts from data diversification;
- the same core independently renders binaural and native FOA RIR sequences,
  passes the exact listener ID to both contexts, validates registration for
  each layout, and requires source IDs, keyframe ticks/samples and trajectory
  identity to match before convolution;
- binaural remains the legacy `audio` record while `audio_layouts`,
  `rir_layouts`, dual identity and native FOA WAV sidecars retain
  `[W,Y,Z,X]`, ACN/N3D, `avengine_world`, no HRTF, no resampling, no amplitude
  normalization and no limiter;
- tests retain both the current region visual/listener/registration cases and
  the dual-layout/sidecar cases; and
- both historical roadmap checkpoint sections are retained.

No RIR cache, new hash, frozen baseline, admission gate or fallback layout was
added.

## Existing runtime evidence

The retained dual-layout runtime-b result remains:

`/data/jzy/tmp/foa_dual_runtime_b_ff582ea_r1`

It proves one real MP3D trajectory with 25 keyframes and 80,000 samples at
16 kHz, producing both binaural and native four-channel FOA under one source /
trajectory identity. It predates the current region source-axis/listener
changes, so a current-code runtime rerun remains required before claiming the
new 160-point population is dual-output.

The Spatial-Omni trainer integration already contains the synchronized native
FOA + video GPU smoke. Its retained output is:

`/data/jzy/tmp/so_native_foa_video_c13a493_gpu0_r1`

That real smoke used batch size 2, video and time-aligned media, native
`[W,Y,Z,X]` FOA, real SO-Encoder `[1,50,768]`, k4 projector `[1,12,3584]`, and
the synced processor's 12 spatial placeholders. It completed two
forward/backward optimizer steps plus validation/generation. This proves the
single-GPU trainer path; it does not prove DDP or long-run convergence.

## Verification

Merged core and adjacent current QA tests:

```text
79 passed
```

Top-level FOA audio/contract/spatial/evidence tests:

```text
42 passed
```

Full AVEngine unit regression:

```text
3391 passed, 118 skipped, 52 subtests passed in 351.72s
```

Log:

`/data/jzy/tmp/foa_diversification_merge_precommit_full_unit_r1.log`

Ruff and `git diff --check` pass.

## Remaining runtime action

After the frozen binaural audio160 finishes, run a fresh current region point
through `ab48cbc`, verify dual registration and both mixtures, then decide how
to produce the current multi-view native-FOA population without mutating the
already retained binaural batch. Clean integration merge remains pending that
current-code runtime evidence.
