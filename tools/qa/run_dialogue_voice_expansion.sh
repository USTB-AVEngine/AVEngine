#!/usr/bin/env bash
# Task-specific CPU runner; every output/cache stays in the authorized artifact root.
set -euo pipefail
out=/data/jzy/tmp/claude_hardbank_20260923/v7_dialogue_voices_20261012
wt=$(cd "$(dirname "$0")/../.." && pwd)
old=/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round5_20261004
model_base=/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round4_20261004
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR="$out/stage2/cache/inductor" TRITON_CACHE_DIR="$out/stage2/cache/triton"
export RAYON_NUM_THREADS=1 HF_ENABLE_PARALLEL_LOADING=false
export TOKENIZERS_PARALLELISM=false MALLOC_CONF=background_thread:false _RJEM_MALLOC_CONF=background_thread:false
export PYTHONPATH="$wt/tools/qa:$old/cache/source_normalization_deps_py310:$model_base/cache/cosy_packages:$model_base/cache/CosyVoice_code:$model_base/cache/CosyVoice_code/third_party/Matcha-TTS:$wt/src"
export TMPDIR="$out/stage2/cache" XDG_CACHE_HOME="$out/cache/xdg" HF_HOME="$out/cache/huggingface" TORCH_HOME="$out/cache/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:?choose calibrate, references, plan, pilot, batch or replacement}" in
 calibrate) program=calibrate_dialogue_voice_language.py; label=calibration ;;
 references) program=select_dialogue_voice_references.py; label=references ;;
 plan) program=plan_dialogue_voice_expansion.py; label=plan ;;
 pilot|batch|replacement) program=expand_dialogue_voices_cpu.py; label=$1 ;;
 *) exit 2 ;;
esac
shift
if [[ "$program" == expand_dialogue_voices_cpu.py ]]; then
 exec nice -n 10 /usr/bin/time -v -o "$out/stage2/evidence/${label}_time.txt" /data/jzy/envs/avledger-20260919-py310/bin/python -u "$wt/tools/qa/$program" --phase "$label" "$@" > "$out/stage2/logs/$label.log" 2>&1
else
 exec nice -n 10 /usr/bin/time -v -o "$out/stage2/evidence/${label}_time.txt" /data/jzy/envs/avledger-20260919-py310/bin/python -u "$wt/tools/qa/$program" "$@" > "$out/stage2/logs/$label.log" 2>&1
fi
