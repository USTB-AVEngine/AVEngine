#!/usr/bin/env bash
set -euo pipefail
out=/data/jzy/tmp/claude_hardbank_20260923/v7_dialogue_voices_20261012
wt=/data/jzy/code/AVEngine-v7-dialogue-voices-20261012
old=/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round5_20261004
model_base=/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round4_20261004
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_CONF=background_thread:false _RJEM_MALLOC_CONF=background_thread:false TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$wt/tools/qa:$old/cache/source_normalization_deps_py310:$model_base/cache/cosy_packages:$model_base/cache/CosyVoice_code:$model_base/cache/CosyVoice_code/third_party/Matcha-TTS:$wt/src"
export TMPDIR="$out/stage2/cache" XDG_CACHE_HOME="$out/cache/xdg" HF_HOME="$out/cache/huggingface" TORCH_HOME="$out/cache/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR="$out/stage2/cache/inductor" TRITON_CACHE_DIR="$out/stage2/cache/triton"
exec nice -n 10 /data/jzy/envs/avledger-20260919-py310/bin/python -u "$wt/tools/qa/expand_dialogue_voices_pipe_cpu.py" > "$out/stage2/logs/batch_pipe.log" 2>&1
