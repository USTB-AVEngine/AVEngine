#!/usr/bin/env bash
# Run the retained dialogue language audit on CPU using the round5 dependency layout.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run_dialogue_voice_audit.sh MANIFEST OUTPUT' >&2; exit 2; }
manifest=$1
out=$2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
old=/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round5_20261004
round4=/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round4_20261004
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$out/cache/tmp" XDG_CACHE_HOME="$out/cache/xdg" HF_HOME="$out/cache/huggingface" TORCH_HOME="$out/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$old/cache/source_normalization_deps_py310:$round4/cache/cosy_packages:$round4/cache/CosyVoice_code:$round4/cache/CosyVoice_code/third_party/Matcha-TTS:$repo/src"
cd "$repo"
exec nice -n 10 /usr/bin/time -v -o "$out/evidence/language_time.txt" /data/jzy/envs/avledger-20260919-py310/bin/python -u tools/qa/audit_dialogue_voice_language.py --manifest "$manifest" --output "$out" --model /data/models/whisper-small --workers 4 --threads 2
