# Spatial-Omni native FOA dataset checkpoint (2026-08-24)

## Scope

This slice replaces the out-of-repository QA dataset experiment that derived
"foaish" channels from a binaural mixture. It does not reuse that pseudo-spatial
conversion. The checked-in builder accepts only the native RLR FOA output added
by `ff582ea`:

```text
<audio-root>/<point_id>/audio/foa/mixture.wav
<audio-root>/<point_id>/audio/foa/mixture.wav.json
```

The builder is `tools/qa/build_spatial_omni_native_foa_dataset.py`. It is
import-safe and publishes a fresh output directory through the existing atomic
no-replace publisher.

## Audio boundary

Every source WAVE is read with
`read_float32_wav(..., verify_sidecar=True)`, which verifies the existing audio
and sidecar content hashes. No new hash, frozen schema, baseline, or independent
gate is added. The dataset layer additionally requires:

- canonical float32 WAVE, 4 channels, 16 kHz, exactly 80,000 samples;
- native order-1 layout `[W,Y,Z,X]`, ACN indices `[0,1,2,3]`, N3D,
  `avengine_world`, format ID `rlr_foa_acn_n3d_world_v1`;
- `audio_role=dynamic_wet_mixture`;
- no HRTF, resampling, amplitude normalization, or limiter.

The original WAVE and sidecar are copied without renaming `mixture.wav`, so the
existing sidecar remains valid after publication. There is no search path for a
binaural mixture, no mid/side remix, and no foaish fallback.

## QA rows and split

- Split assignment is deterministic by the transitive `twin_of` group; a main
  point and every twin stay in the same train/valid split.
- Existing MCQ rows retain their options and canonical answer.
- `numeric_banded` rows retain `answer_numeric`, `unit`, and positive ascending
  `scoring_bands`.
- An explicit refusal row receives audio only, omits `video_path`, and carries
  `input_modalities=[audio]` plus `modality_mask={audio:true, video:false}`. The
  builder recognizes the real generator contract only when all four declarations
  agree: `type_id=QV2-TR-REFUSAL`, `refusal_gold=true`,
  `provided_modalities=[audio]`, and `withheld_modalities=[video]`; inconsistent
  or legacy-name-only markers fail instead of silently selecting a modality.
- Every non-refusal row, including numeric and off-screen questions, requires
  the point review clip and carries both audio and video modalities.

`DATASET_PROVENANCE.json` identifies native RLR FOA, records the no-transform
copy semantics, and remains `research_only=true`, `episode_counted=false`,
`formal_dataset_count=0`, and `qualification_claim=false`.

This is not yet a trainer-ready claim. The current ordinary Spatial-Omni video
collator does not consume `modality_mask` and expects a video for every row when
video is enabled. Provenance therefore records trainer compatibility as blocked;
refusal rows must not enter training until sparse-video/modality-mask behavior is
explicitly wired and tested on the trainer side.

## Evidence boundary

Server CPU verification used the target checkout explicitly through
`PYTHONPATH=src`:

```text
python -m pytest -q tests/unit/test_build_spatial_omni_native_foa_dataset.py \
  tests/unit/test_current_mp3d_dynamic_audio.py \
  tests/unit/test_m5_1_acoustics.py tests/unit/test_m5_cli.py
43 passed
```

Ruff and `git diff --check` passed. The strict builder validator also read the
retained runtime-b `audio/foa/mixture.wav` and its existing sidecar successfully
in place; that command was read-only and did not rerun RLR.

CPU unit tests cover incorrect channel/rate/sample layouts, wrong spatial format
and foaish-only input, forbidden processing metadata, twin split isolation,
numeric fields, refusal video exclusion and modality mask, mandatory clips, and
atomic no-clobber publication.

The retained runtime-b point at
`/data/jzy/tmp/foa_dual_runtime_b_ff582ea_r1` may be read back as one real source
validation, but it is MP3D research evidence only. This slice does not launch
RLR/GPU work, generate an Apartment dataset, train Spatial-Omni, or change the
formal dataset denominator from zero.
