#!/usr/bin/env bash
set -euo pipefail
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=${CCI_DATAGEN_BASE:-/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex}
TORCH_PYTHON=${DATAGEN_TORCH_PYTHON:-/mnt/afs/xinyuanqian/jzy/hardbank/envs/avledger-py310/bin/python}
GEN_PYTHON=${DATAGEN_PYTHON:-$BASE/env/bin/python}
mkdir -p "$BASE/tmp"
export TMPDIR="$BASE/tmp"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# The launcher is Python 3.10; generation packages are Python 3.12.
# Generation Python sees its own overlay via its private .pth instead.
export PYTHONPATH="$BASE/code/src:$BASE/code"
export DATAGEN_LAUNCH_ID=${DATAGEN_LAUNCH_ID:-${LADDER_JOB_ID:-datagen}_$(date -u +%Y%m%dT%H%M%S%N)_${BASHPID}_${1:?probe or pilot}}
NNODES=${SENSECORE_PYTORCH_NNODES:-1}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-0}
# The existing exclusive scheduler starts one command after torchrun exits.
# A multi-node allocation needs an owner-selected per-node launch handoff.
if [[ "$NNODES" != 1 ]]; then
  echo 'exclusive datagen launcher requires one node; multi-node per-node handoff is missing' >&2
  exit 6
fi
VISIBLE=$($TORCH_PYTHON -B "$SOURCE/resources.py" --output "$BASE/evidence/launch-resource-${NODE_RANK}-${DATAGEN_LAUNCH_ID}.json" >/dev/null; $TORCH_PYTHON -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpus"]["visible_count"])' "$BASE/evidence/launch-resource-${NODE_RANK}-${DATAGEN_LAUNCH_ID}.json")
RANKS_PER_GPU=${DATAGEN_RANKS_PER_GPU:-${LADDER_WORKERS_PER_GPU:-2}}
[[ "$RANKS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || exit 2
NPROC=${DATAGEN_NPROC_PER_NODE:-$(( VISIBLE > 0 ? VISIBLE * RANKS_PER_GPU : 1 ))}
ROOT=${DATAGEN_RUN_ROOT:-$BASE/output/${1}}
mkdir -p "$ROOT"
# P19 plans all use our own roots; the scheduler, queue and existing runs stay untouched.
if [[ "$NODE_RANK" == 0 ]]; then
  exec 9>"$ROOT/.launch.lock"
  flock -n 9 || { echo 'this generation root is already running' >&2; exit 3; }
fi
COMMON=(--nproc_per_node "$NPROC" --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${DATAGEN_MASTER_PORT:-29573}")
ACTION=$1;shift
case "$ACTION" in
 probe)
   exec "$TORCH_PYTHON" -B -m torch.distributed.run "${COMMON[@]}" "$SOURCE/probe_entry.py" --output-root "$ROOT" "$@"
   ;;
 pilot)
   source "$BASE/activate_generation.sh"
   export PYTHONPATH="$BASE/code/src:$BASE/code"
   MANIFEST=${DATAGEN_MANIFEST:-$BASE/configs/pilot64.json}
   [[ -s "$BASE/evidence/NATIVE_READY.json" ]] || { echo 'native environment preflight has not passed; see evidence/native_import.json' >&2; exit 4; }
   exec "$TORCH_PYTHON" -B -m torch.distributed.run "${COMMON[@]}" "$SOURCE/generate.py" --manifest "$MANIFEST" --run-root "$ROOT" --python "$GEN_PYTHON" "$@"
   ;;
 *) echo 'usage: run_p19.sh probe|pilot [generate.py arguments]' >&2;exit 2;;
esac
