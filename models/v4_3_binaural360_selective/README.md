# v4_3 binaural 360 selective model

This directory is the strict boundary for the experimental model based on the
Spatial `v4_3_new_IPD_Enhancer` architecture. The required architecture source
is copied into `avengine_v43/model.py`; the runner does not import model code
from the sibling Spatial checkout.

All v4-specific model adaptation, label conversion, training, evaluation and
tests must stay below this directory. Production AVEngine packages under
`src/avengine/` and general tools under `tools/` must not import this
experiment.

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
- `avengine_v43/hdf5_data.py`: validated single-file HDF5 query bank.
- `avengine_v43/model.py`: local v4_3 Progressive Refinement architecture.
- `build_hdf5_cache.py`: resumable conversion from the delivered WAV closure
  to one HDF5 file.
- `run_training_smoke.py`: model construction, five-second/360 adaptation,
  text-only query path and real GPU update.
- `train.py`: fixed 800/100/100 training, validation, best-checkpoint test and
  exact mid-epoch resume.
- `tests/`: model-experiment tests.

Generated results and checkpoints stay under the repository's externalized
`tmp/` target or another configured dataset root; they must not be committed.

## Build the training cache

The original v4 loaders use HDF5. This experiment keeps that useful idea but
stores the complete 1,000-sample closure in one file instead of opening one
HDF5 file per sample. The conversion reads each delivered WAV once. Formal
training then preloads the 612 MB cache into host RAM and does not repeatedly
open WAV files across epochs.

Run from this directory:

```bash
PYTHONPATH=. python \
  build_hdf5_cache.py \
  --dataset-index ../../tmp/m7/apartment_generated_assets_training_index_unique1000_visual_20260723_01/dataset_index.json \
  --trajectory-bank ../../tmp/m7/apartment_generated_assets_rir_plan_unique1000_20260723_01/trajectory_bank.json \
  --rir-plan ../../tmp/m7/apartment_generated_assets_rir_plan_unique1000_20260723_01/rir_job_plan.json \
  --output ../../tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5
```

The builder writes a sibling `.incomplete` file, records completed sample
count, flushes periodically and atomically renames it after full readback.

## Train

```bash
PYTHONPATH=. python \
  train.py \
  --dataset-h5 ../../tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5 \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --output-root ../../tmp/m7/v43_binaural360_training \
  --device cuda:3 \
  --epochs 20 \
  --batch-size 8 \
  --validation-batch-size 16
```

Resume from the exact next batch:

```bash
PYTHONPATH=. python \
  train.py \
  --dataset-h5 ../../tmp/m7/v43_apartment_1000_training_cache_20260723_01.h5 \
  --clap-checkpoint "${CLAP_CHECKPOINT}" \
  --output-root ../../tmp/m7/v43_binaural360_training \
  --resume-from ../../tmp/m7/v43_binaural360_training/latest.pt \
  --device cuda:3 \
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
  `tmp/m7/v43_formal_h5_epoch1_20260723_01/training_summary.json`.

The one-epoch test mean circular error was 69.77 degrees and the two text
queries produced almost identical predictions. It therefore proves that the
pipeline runs, checkpoints and evaluates, but it does **not** yet prove useful
selective localization quality.
