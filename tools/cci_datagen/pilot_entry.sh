#!/usr/bin/env bash
set -euo pipefail
BASE=${CCI_DATAGEN_BASE:-/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex}
SOURCE=$BASE/code/tools/cci_datagen
source "$BASE/activate_generation.sh"
# No formal test data, no fallback to old v6 validation rooms. The sampler's
# accepted training requests must exist before this plan can run.
REQUESTS=${DATAGEN_REQUEST_LIST:-$BASE/configs/pilot64_request_paths.json}
[[ -f "$REQUESTS" ]] || { echo 'missing accepted 64-world train request list from v7 sampler handoff' >&2; exit 5; }
[[ -s "$BASE/evidence/NATIVE_READY.json" ]] || { echo 'blocked native environment: see evidence/native_import.json' >&2; exit 4; }
"$DATAGEN_PYTHON" -B "$SOURCE/make_pilot.py" --requests "$REQUESTS" --recommended-houses "$BASE/mirror/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/report_final/recommended_houses.json" --code-root "$BASE/code" --output "$BASE/configs/pilot64.json"
"$DATAGEN_PYTHON" -B "$SOURCE/native_preflight.py" --output "$BASE/evidence/pilot_native_cpu.json"
SCENE=$BASE/mirror/data/datasets/habitat_data/scene_datasets/hm3d/train/00059-kJxT5qssH4H/kJxT5qssH4H.glb
[[ -f "$SCENE" ]] || { echo 'first training-house scan is missing' >&2; exit 6; }
GPU_ID=$("$DATAGEN_PYTHON" -B -c 'import sys;sys.path.insert(0,sys.argv[1]);from resources import gpu_inventory;g=gpu_inventory()["visible"];assert g,"no authorized visible GPU";print(g[0]["index"])' "$SOURCE")
"$DATAGEN_PYTHON" -B "$SOURCE/native_preflight.py" --gpu-device-id "$GPU_ID" --render --scene "$SCENE" --output "$BASE/evidence/pilot_egl.json"
exec bash "$SOURCE/run_p19.sh" pilot --mode dedicated --prune-owned-intermediates "$@"
