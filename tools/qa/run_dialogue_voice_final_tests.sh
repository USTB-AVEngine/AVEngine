#!/usr/bin/env bash
set -euo pipefail
out=/data/jzy/tmp/claude_hardbank_20260923/v7_dialogue_voices_20261012
wt=/data/jzy/code/AVEngine-v7-dialogue-voices-20261012
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONPATH="$wt/tools/qa:$wt:$wt/src" TMPDIR="$out/stage2/cache" XDG_CACHE_HOME="$out/cache/xdg" HF_HOME="$out/cache/huggingface" TORCH_HOME="$out/cache/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_CONF=background_thread:false _RJEM_MALLOC_CONF=background_thread:false TOKENIZERS_PARALLELISM=false
cd "$wt"
export TEST_COMMIT="$(git rev-parse HEAD)"
exec nice -n 10 /usr/bin/time -v -o "$out/stage2/evidence/all_grouped_test_phases_time.txt" /data/jzy/envs/room-selection-merged-20261003/bin/python -u "$wt/tools/qa/finish_dialogue_voice_cpu_tests.py" --phase both > "$out/stage2/logs/all_grouped_test_phases.log" 2>&1
