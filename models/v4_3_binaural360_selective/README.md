# v4_3 binaural 360 selective model

This directory is the strict boundary for the experimental model based on the
Spatial `v4_3_new_IPD_Enhancer` architecture. The required architecture source
is copied into `avengine_v43/model.py`; the runner does not import model code
from the sibling Spatial checkout.

All v4-specific model adaptation, label conversion, training, evaluation and
tests must stay below this directory. Production AVEngine packages under
`src/avengine/` and general tools under `tools/` must not import this
experiment.

This is an owner-private, test-only experiment. It is not part of the public
AVEngine engine, is not open source and will not be released as part of that
engine. It remains permanently on `feature/v43-binaural360-model` and must
never be merged into the AVEngine Native/public branch.

## Environment and external artifacts

The existing `locate` environment is authoritative for this private
experiment's HDF5 build, tests, training and inference. It is not an AVEngine
runtime environment. Public engine plan/RIR/mixture work uses
`avengine-habitat-runtime` separately.

Set explicit paths before using the commands below:

```bash
export AVENGINE_NATIVE_ROOT=/data/jzy/code/AVEngine-habitat-native
export V43_PYTHON=/data/jzy/miniconda3/envs/locate/bin/python
export CLAP_CHECKPOINT=/home/Models/laion_clap/630k-audioset-best.pt
nvidia-smi
export V43_DEVICE=cuda:N  # replace N with an index confirmed free above
```

## Current contract

- five-second, 16 kHz, two-channel HRTF binaural input;
- two simultaneously active sources;
- the same mixture queried with two different text targets;
- 75 temporal outputs and 360 circular one-degree azimuth bins;
- localization/separation parameters initialized from scratch;
- frozen pretrained LAION CLAP used only as the text encoder;
- no old v4 localization checkpoint;
- DoA and target-cardinality losses;
- deterministic CUDA execution and exact mid-epoch resume.

The current first pass does not require wet stems because it does not optimize
a waveform separation loss. It trains and evaluates circular localization,
but a successful execution is not by itself a convergence or accuracy claim.

## Layout

- `avengine_v43/labels.py`: native 360 labels and deterministic dataset
  selection helpers.
- `avengine_v43/hdf5_data.py`: validated fixed-size training and variable-size
  test-only HDF5 query banks.
- `avengine_v43/evaluation.py`: circular-error, target-region and same-room /
  cross-room comparison helpers.
- `avengine_v43/model.py`: local v4_3 Progressive Refinement architecture.
- `build_hdf5_cache.py`: resumable conversion from the delivered WAV closure
  to one HDF5 file.
- `build_evaluation_hdf5.py`: variable-size test-only HDF5 construction for a
  second room.
- `run_training_smoke.py`: model construction, five-second/360 adaptation,
  text-only query path and real GPU update.
- `train.py`: fixed 800/100/100 training, validation, best-checkpoint test and
  exact mid-epoch resume.
- `visualize_inference.py`: deterministic test-sample inference and a
  1920x480 RGB+Topdown+GT/predicted-DoA review video with the original
  binaural track preserved.
- `evaluate_dataset.py`: aggregate held-out evaluation split by motion case,
  generic source motion, ordered source pair, text class and target azimuth
  quadrant, with explicit tail failures and synchronized-video links.
- `compare_evaluations.py`: fail-closed same-checkpoint comparison of a
  same-room baseline and a cross-room zero-shot report.
- `tests/`: model-experiment tests.

Generated results and checkpoints used here stay under
`${AVENGINE_NATIVE_ROOT}/tmp/` or another explicit dataset root; this private
worktree has no implicit relative `tmp` target. Generated artifacts must not
be committed.

## Build the training cache

The original v4 loaders use HDF5. This experiment keeps that useful idea but
stores the complete 1,000-sample closure in one file instead of opening one
HDF5 file per sample. The conversion reads each delivered WAV once. Formal
training then preloads the 612 MB cache into host RAM and does not repeatedly
open WAV files across epochs.

Run from this directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  build_hdf5_cache.py \
  --dataset-index "${AVENGINE_NATIVE_ROOT}/tmp/m7/apartment_generated_assets_training_index_unique1000_visual_20260723_01/dataset_index.json" \
  --trajectory-bank "${AVENGINE_NATIVE_ROOT}/tmp/m7/apartment_generated_assets_rir_plan_unique1000_20260723_01/trajectory_bank.json" \
  --rir-plan "${AVENGINE_NATIVE_ROOT}/tmp/m7/apartment_generated_assets_rir_plan_unique1000_20260723_01/rir_job_plan.json" \
  --output "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5"
```

The builder writes a sibling `.incomplete` file, records completed sample
count, flushes periodically and atomically renames it after full readback.

## Train

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  train.py \
  --dataset-h5 "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5" \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --output-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_training" \
  --device "${V43_DEVICE}" \
  --epochs 20 \
  --batch-size 8 \
  --validation-batch-size 16
```

Resume from the exact next batch:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  train.py \
  --dataset-h5 "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5" \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --output-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_training" \
  --resume-from "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_training/latest.pt" \
  --device "${V43_DEVICE}" \
  --epochs 20 \
  --batch-size 8 \
  --validation-batch-size 16
```

`latest.pt` and `best.pt` omit the frozen CLAP parameters; the external CLAP
checkpoint remains an explicit input. `--max-global-steps` exists only for
development interruption/resume verification.

## Executed evidence

- The 1,000-sample HDF5 cache passed readback with 1,600 train, 200 validation
  and 200 test text queries. Its first construction took 347.76 seconds while
  the shared disk was under contention.
- Host-RAM preload plus the first batch took 0.70 seconds in the final path.
  A 100-batch random-read probe dropped from 0.406 seconds per batch on the
  shared file to 0.0055 seconds per batch from RAM.
- A stopped-at-step-1 run resumed to step 2 and matched an uninterrupted
  two-step run exactly across all 351 saved model tensors.
- One complete epoch ran 200 train batches in 72.56 seconds, then evaluated
  all 200 validation and 200 test queries. The execution result is
  `${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_formal_h5_epoch1_20260723_01/training_summary.json`.

The one-epoch test mean circular error was 69.77 degrees and the two text
queries produced almost identical predictions. It therefore proves that the
pipeline runs, checkpoints and evaluates, but it does **not** yet prove useful
selective localization quality.

## Visualize one real test episode

The visualization runner refuses non-test samples, verifies that the HDF5
query, dataset-index episode and source video agree, runs both text queries
against the same binaural mixture and writes both numeric tracks and a muxed
review video:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  visualize_inference.py \
  --dataset-h5 "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5" \
  --dataset-index "${AVENGINE_NATIVE_ROOT}/tmp/m7/apartment_generated_assets_training_index_unique1000_visual_20260723_01/dataset_index.json" \
  --checkpoint "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_training_100ep_b56_20260724_01/best.pt" \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --sample-id border_collie_human__recombined_both_moving_0124__v00 \
  --source-video "${AVENGINE_NATIVE_ROOT}/tmp/m7/apartment_asset_bound_ue_unique1000_full_20260723_01/border_collie_human__recombined_both_moving_0124/ue_topdown_binaural.mp4" \
  --output-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_inference_visualization_20260724_01" \
  --device "${V43_DEVICE}"
```

In each compass and trajectory plot, solid lines with filled markers are
ground truth; dashed lines with open markers are predictions. Source/query
identity also has a fixed blue/orange encoding. The panel uses the native
AVEngine convention: front 0 degrees, right +90, rear 180 and left 270/-90.

Executed test-split evidence is under
`${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_inference_visualization_20260724_01/`.
The reviewed episode is the both-moving Border Collie plus human sample
`border_collie_human__recombined_both_moving_0124__v00`. With the epoch-90
best checkpoint, the dog query has 5.09-degree mean / 3.65-degree median
circular error, while the human query has 17.10-degree mean / 1.89-degree
median error. The latter mismatch is concentrated in a small number of large
errors, including a visible final-frame front/rear failure. The two text
queries differ on all 75 frames and target-cardinality accuracy is 100% for
this example. These are per-example diagnostic results, not aggregate model
qualification.

Use `ue_topdown_binaural_with_doa_prediction_v2.mp4` for human review. It is
1920x480, 75 frames at 15 fps, exactly five seconds, and preserves the
original AAC two-channel audio packet stream. `inference.json` contains the
complete GT/prediction/error tracks; `render_manifest_v2.json` records final
video readback and visual encoding.

## Audit the complete held-out Apartment split

The aggregate evaluator reopens the same fixed HDF5/index/checkpoint triple,
runs all 200 test queries, and writes `evaluation.json`,
`per_query_metrics.csv` and a compact `REVIEW_INDEX.html`. The report makes
the long error tail visible by motion case, generic source motion, text class,
ordered source pair and front/right/rear/left target region. RGB and Topdown
remain synchronized review media and are not silently presented as model
inputs.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  evaluate_dataset.py \
  --dataset-h5 "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5" \
  --dataset-index "${AVENGINE_NATIVE_ROOT}/tmp/m7/apartment_generated_assets_training_index_unique1000_visual_20260723_01/dataset_index.json" \
  --checkpoint "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_training_100ep_b56_20260724_01/best.pt" \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --training-room-id apartment_0000 \
  --evaluation-room-id apartment_0000 \
  --output-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_test_slice_evaluation_20260726_03" \
  --device "${V43_DEVICE}" --batch-size 64
```

## Execute the balanced-360 cross-room diagnostic

The Kujiale cache is test-only. Build it from the public engine mixture and
the same listener-pose-bound plan, then evaluate it without fine-tuning. The
formal public-engine dependency chain for this run is:

- research cleanup package:
  `${AVENGINE_NATIVE_ROOT}/tmp/m3/kujiale_0020_usd_semantic_rlr_cleanup_20260726_03/manifest.json`;
- balanced plan:
  `${AVENGINE_NATIVE_ROOT}/tmp/m7/kujiale_0020_zero_shot_balanced360_plan_100_20260726_01/`;
- RIR cache:
  `${AVENGINE_NATIVE_ROOT}/tmp/m7/kujiale_0020_real_usd_rir_cache_balanced360_2587_20260726_02/`;
- binaural mixture:
  `${AVENGINE_NATIVE_ROOT}/tmp/m7/kujiale_0020_zero_shot_balanced360_binaural_100_20260726_02/`.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  build_evaluation_hdf5.py \
  --audio-batch-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/kujiale_0020_zero_shot_balanced360_binaural_100_20260726_02" \
  --trajectory-bank "${AVENGINE_NATIVE_ROOT}/tmp/m7/kujiale_0020_zero_shot_balanced360_plan_100_20260726_01/trajectory_bank.json" \
  --rir-plan "${AVENGINE_NATIVE_ROOT}/tmp/m7/kujiale_0020_zero_shot_balanced360_plan_100_20260726_01/rir_job_plan.json" \
  --room-id kujiale_0020_livingroom_491 \
  --output-h5 "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_kujiale_0020_zero_shot_balanced360_100_20260726_02.h5" \
  --output-index "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_kujiale_0020_zero_shot_balanced360_100_index_20260726_02.json"

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  evaluate_dataset.py \
  --dataset-h5 "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_kujiale_0020_zero_shot_balanced360_100_20260726_02.h5" \
  --dataset-index "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_kujiale_0020_zero_shot_balanced360_100_index_20260726_02.json" \
  --checkpoint "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_binaural360_training_100ep_b56_20260724_01/best.pt" \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --training-room-id apartment_0000 \
  --evaluation-room-id kujiale_0020_livingroom_491 \
  --output-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_kujiale_0020_zero_shot_balanced360_evaluation_20260726_02" \
  --device "${V43_DEVICE}" --batch-size 64

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${V43_PYTHON}" \
  compare_evaluations.py \
  --baseline-evaluation "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_test_slice_evaluation_20260726_03/evaluation.json" \
  --candidate-evaluation "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_kujiale_0020_zero_shot_balanced360_evaluation_20260726_02/evaluation.json" \
  --output-root "${AVENGINE_NATIVE_ROOT}/tmp/m7/v43_apartment_vs_kujiale_balanced360_comparison_20260726_02"
```

## Executed balanced-360 evidence

- The `locate` environment passed all 68 private model tests, compiled all 18
  Python files and passed `--help` checks for the three evaluation CLIs. The
  Kujiale evaluation HDF5 contains 100 test samples and 200 text queries with
  exactly the three requested captions.
- Apartment `_03` is `same_room_held_out`. Across 15,000 evaluated frames, its
  mean/median/P90 MAE are `10.09961011963139`,
  `0.9337444305419922` and `16.706146240234375` degrees. Its fractions over
  45 and 90 degrees are `0.06886666666666667` and
  `0.04273333333333333`; its uniform target-region macro MAE is
  `9.05570884846393` degrees.
- Kujiale `_02` is `cross_room_zero_shot`. Across 15,000 evaluated frames, its
  mean/median/P90 MAE are `60.28606991654138`, `44.0` and
  `146.60128784179688` degrees. Its fractions over 45 and 90 degrees are
  `0.4938666666666667` and `0.29733333333333334`; its uniform target-region
  macro MAE is `62.06663361693215` degrees.
- The comparison uses the same Apartment-trained `best.pt` and the same CLAP
  checkpoint. Kujiale minus Apartment changes mean/median/P90 MAE by
  `+50.186459796909986`, `+43.06625556945801` and
  `+129.8951416015625` degrees, changes the over-45 and over-90 fractions by
  `+0.425` and `+0.2546`, and changes region-macro MAE by
  `+53.01092476846822` degrees.

The Kujiale acoustic package still uses `research_placeholder` materials and
topology-failing geometry. This single fixed Apartment-to-Kujiale research
diagnostic shows substantial cross-room degradation. It is not evidence of
broad generalization, physical-acoustics calibration, real-room truth, dataset
qualification or real-world generalization.
