#!/bin/zsh
# Re-admit every static batch with the upright correction measured from the
# watertight proxies the first pass already produced.
set -e
cd /data/jzy/code/SPEAR-lead-b
PY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
AUTHOR=/tmp/author_admission.py

run() {
  local ws=$1 prior=$2; shift 2
  echo "=== $ws"
  rm -rf $ws/admission_authorities_up $ws/admission_up
  $PY $AUTHOR \
    --decision-batch $ws/static_decision/static_object_decision_batch_manifest.json \
    --decision-root $ws/static_decision \
    --review-root $ws/static_review \
    --output-dir $ws/admission_authorities_up \
    --measure-upright-from $ws/$prior \
    "$@"
  $PY tools/run_controlled_static_object_admission.py \
    --decision-batch $ws/static_decision/static_object_decision_batch_manifest.json \
    --plan $ws/admission_authorities_up/admission_plan.json \
    --output-root $ws/admission_up
}

run tmp/audio_playback_floor_v2 admission --yaw 28d5cf8bc2a9=90

run tmp/audio_playback_statics_v2 admission_v2 \
  --yaw 2026d11447e9=90 --yaw 352efe1972a4=0 --yaw 7dbef6e77e15=0 \
  --yaw a37a4cf6d5c7=0 --yaw 2183947ffddb=53.3 --yaw 8569086c6422=136.6 \
  --yaw cafcf63bf0e8=137.8

run tmp/audio_playback_level_camera_v3 admission \
  --yaw 282ee90d2dd9=0 --yaw b744cb8818e2=0 --yaw 3c57b10560b9=113.2 \
  --yaw c810ce1566c0=136.5

run tmp/audio_playback_statics_v1 admission_v5 \
  --yaw 477fecc3d676=90 --yaw 61ea3095f68c=132 --yaw b11e127682a0=136.7 \
  --yaw fdbb7157722f=132.4

echo "ALL_READMITTED"
