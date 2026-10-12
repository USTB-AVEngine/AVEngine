#!/usr/bin/env bash
set -euo pipefail
source /data/jzy/tmp/claude_hardbank_20260923/v7_dialogue_voices_20261012/stage2/scripts/env_cpu.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$wt/tools/qa:$PYTHONPATH"
nice -n 10 /data/jzy/envs/avledger-20260919-py310/bin/python "$wt/tools/qa/finalize_dialogue_voice_report.py" > "$out/stage2/logs/final_report.log" 2>&1
exec nice -n 10 /data/jzy/envs/avledger-20260919-py310/bin/python "$wt/tools/qa/finalize_dialogue_voice_receipt.py" > "$out/stage2/logs/final_delivery.log" 2>&1
