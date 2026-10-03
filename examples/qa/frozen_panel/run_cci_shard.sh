#!/usr/bin/env bash
# One explicitly assigned GPU, one output root; call from the AVEngine worktree.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo 'usage: bash run_cci_shard.sh PYTHON CONFIG OUTPUT SHARD_INDEX SHARD_COUNT ASSIGNED_GPU' >&2
  exit 2
fi
panel_python=$1
panel_config=$2
panel_output=$3
panel_shard=$4
panel_shards=$5
panel_gpu=$6
panel_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$panel_repo/src:$panel_repo"
exec "$panel_python" "$panel_repo/tools/qa/run_frozen_panel_shard.py" \
  --config "$panel_config" --output "$panel_output" \
  --shard-index "$panel_shard" --shard-count "$panel_shards" --gpu "$panel_gpu" --resume
