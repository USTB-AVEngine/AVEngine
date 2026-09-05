#!/usr/bin/env bash
# Take one generated animal from a raw reconstruction to a reviewable rigged
# asset, trying the configured preparation ladder that works rather than one
# fixed recipe.
#
# The default policy is visual_review: the first structurally complete rung is
# rendered at ordinary viewing distance and handed to a human reviewer.
# strict_metrics retains the historical metric gates when explicitly selected.
#
# Usage:
#   tools/assets/run_generated_animal_chain.sh \
#     --raw /path/raw.glb --workdir /path/out --front-yaw-deg -179.706 \
#     --donor-rig /path/donor_walk_idle.glb [--runtime-python /path/python] \
#     [--policy-config examples/assets/generated_animal_review_policy_v1.json] \
#     [--strategy visual_review|strict_metrics] [--asset-id ID] [--review-id ID]
#
# --spear-root and --rigger-root are retained as deprecated compatibility
# options; helpers and model roots are resolved by AVEngine.

set -uo pipefail

RAW="" WORKDIR="" YAW="" DONOR="" SPEAR_ROOT="" RIGGER_ROOT=""
RUNTIME_PYTHON=python3
GPU=0 BLENDER="${BLENDER:-blender}"
POLICY_CONFIG="" STRATEGY="" ASSET_ID="" REVIEW_ID=""
LADDER_OVERRIDE="" PICK="" RETRY_BAND="" RIG_RETRIES="" RELIEF_SMOOTH=""

while [ $# -gt 0 ]; do
  case "$1" in
    --raw) RAW="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --front-yaw-deg) YAW="$2"; shift 2 ;;
    --donor-rig) DONOR="$2"; shift 2 ;;
    --spear-root) SPEAR_ROOT="$2"; shift 2 ;;
    --rigger-root) RIGGER_ROOT="$2"; shift 2 ;;
    --runtime-python) RUNTIME_PYTHON="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --policy-config|--config) POLICY_CONFIG="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --asset-id) ASSET_ID="$2"; shift 2 ;;
    --review-id) REVIEW_ID="$2"; shift 2 ;;
    --port-base) echo "warning: --port-base is deprecated and ignored; SkinTokens uses a private Unix socket" >&2; shift 2 ;;
    --ladder) LADDER_OVERRIDE="$2"; shift 2 ;;
    --pick) PICK="$2"; shift 2 ;;
    --retry-band) RETRY_BAND="$2"; shift 2 ;;
    --rig-retries) RIG_RETRIES="$2"; shift 2 ;;
    --relief-smooth) RELIEF_SMOOTH="$2"; shift 2 ;;
    --blender) BLENDER="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

for required in RAW WORKDIR YAW DONOR; do
  if [ -z "${!required}" ]; then
    echo "missing --${required,,} (see the usage comment)" >&2
    exit 64
  fi
done

if ! RAW="$(realpath -e -- "$RAW")"; then
  echo "raw mesh does not exist: $RAW" >&2
  exit 64
fi
if ! DONOR="$(realpath -e -- "$DONOR")"; then
  echo "donor rig does not exist: $DONOR" >&2
  exit 64
fi
WORKDIR="$(realpath -m -- "$WORKDIR")"

if [ -n "$SPEAR_ROOT" ]; then
  echo "warning: --spear-root is deprecated and ignored; generated-animal helpers are sourced from this AVEngine checkout" >&2
fi
if [ -n "$RIGGER_ROOT" ]; then
  echo "warning: --rigger-root is deprecated and ignored; SkinTokens source is sourced from this AVEngine checkout" >&2
fi

# fresh/no-clobber: an existing workdir is evidence from another run.
if [ -e "$WORKDIR" ]; then
  echo "workdir already exists, refusing to overwrite: $WORKDIR" >&2
  exit 65
fi
mkdir -p "$WORKDIR"

AVENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_MODULE="$AVENGINE_ROOT/tools/assets/animal_review_policy.py"
if [ -z "$POLICY_CONFIG" ]; then
  POLICY_CONFIG="$AVENGINE_ROOT/examples/assets/generated_animal_review_policy_v1.json"
fi
if ! POLICY_CONFIG="$(realpath -e -- "$POLICY_CONFIG")"; then
  echo "animal review policy does not exist: $POLICY_CONFIG" >&2
  exit 64
fi
cp -- "$POLICY_CONFIG" "$WORKDIR/review_policy.json" || {
  echo "could not snapshot animal review policy" >&2
  exit 65
}
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

POLICY_ARGS=(--config "$POLICY_CONFIG")
[ -n "$STRATEGY" ] && POLICY_ARGS+=(--strategy "$STRATEGY")
POLICY_JSON="$( "$RUNTIME_PYTHON" "$POLICY_MODULE" "${POLICY_ARGS[@]}" --emit-json )" || {
  echo "could not load animal review policy: $POLICY_CONFIG" >&2
  exit 64
}
policy_value() {
  "$RUNTIME_PYTHON" -c 'import json,sys
value=json.loads(sys.argv[1])
for key in sys.argv[2].split("."):
    value=value[key]
if value is None:
    print("")
elif isinstance(value,bool):
    print("1" if value else "0")
else:
    print(value)' "$POLICY_JSON" "$1"
}
POLICY_ID="$(policy_value policy_id)"
STRATEGY="$(policy_value strategy)"
METRIC_GATES="$(policy_value gate_metrics)"
[ -n "$PICK" ] || PICK="$(policy_value runner.pick)"
if [ "$PICK" != "first" ] && [ "$PICK" != "best" ]; then
  echo "--pick must be first or best" >&2
  exit 64
fi
[ -n "$RETRY_BAND" ] || RETRY_BAND="$(policy_value runner.retry_band)"
[ -n "$RIG_RETRIES" ] || RIG_RETRIES="$(policy_value runner.rig_retries)"
[ -n "$RELIEF_SMOOTH" ] || RELIEF_SMOOTH="$(policy_value runner.relief_smooth)"
WALK_ACTION="$(policy_value render.walking_action)"
WALK_FRAMES="$(policy_value render.walking_frames)"
WALK_ZOOM="$(policy_value render.walking_zoom)"
WALK_DIR="$(policy_value render.walking_dir)"
TURN_ACTION="$(policy_value render.turntable_action)"
TURN_FRAMES="$(policy_value render.turntable_frames)"
TURN_POSE_RATIO="$(policy_value render.turntable_pose_ratio)"
TURN_DIR="$(policy_value render.turntable_dir)"
VIEW_DISTANCE="$(policy_value render.view_distance)"
DEFORM_ACTION="$(policy_value measurement.deformation_action)"
SAMPLE_COUNT="$(policy_value measurement.sample_count)"
SHARD_EDGE_GROWTH="$(policy_value measurement.shard_edge_growth_threshold)"
UNDERSIDE_FRACTION="$(policy_value measurement.underside_height_fraction)"
MAX_ABS_POSITION="$(policy_value closure.maximum_abs_position)"
MAX_ABS_SCALE="$(policy_value closure.maximum_abs_scale)"
MIN_HEAD_SURVIVAL="$(policy_value metrics.retopology.min_head_survival)"
FACE_TOLERANCE="$(policy_value metrics.retopology.face_tolerance)"
MAX_SHARD_SHARE="$(policy_value metrics.rigged.max_shard_share)"
MAX_SHARE_OVER_10X="$(policy_value metrics.rigged.max_share_over_10x)"
TARGET_FRONT_AXIS="$(policy_value retarget.target_front_axis)"
MOTION_BASIS_YAW_DEG="$(policy_value retarget.motion_basis_yaw_deg)"
SIDE_CHAIN_MODE="$(policy_value retarget.side_chain_mode)"

# Emit configured rungs as tab-separated values.  An explicit --ladder remains
# a diagnostic compatibility override; the checked-in default is always JSON.
RUNGS=()
if [ -n "$LADDER_OVERRIDE" ]; then
  IFS=',' read -r -a override_rungs <<< "$LADDER_OVERRIDE"
  for rung in "${override_rungs[@]}"; do
    IFS=':' read -r mode budget divisor extra <<< "$rung"
    if { [ "$mode" != "plain" ] && [ "$mode" != "remesh" ]; } \
       || ! [[ "$budget" =~ ^[1-9][0-9]*$ ]] \
       || ! [[ "${divisor:-0}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
       || [ -n "${extra:-}" ]; then
      echo "invalid --ladder rung: $rung" >&2
      exit 64
    fi
    RUNGS+=( "${mode}_${budget}_${divisor:-0}:${mode}:${budget}:${divisor:-0}:${RELIEF_SMOOTH}" )
  done
else
  ladder_lines="$( "$RUNTIME_PYTHON" "$POLICY_MODULE" "${POLICY_ARGS[@]}" --emit-ladder )" || {
    echo "could not read configured animal ladder" >&2
    exit 64
  }
  while IFS=$'\t' read -r rung_id mode budget divisor smoothing; do
    [ -n "$rung_id" ] || continue
    RUNGS+=( "$rung_id:$mode:$budget:$divisor:$smoothing" )
  done <<< "$ladder_lines"
fi
[ "${#RUNGS[@]}" -gt 0 ] || { echo "configured animal ladder is empty" >&2; exit 64; }

prepare() {  # rung_dir mode budget divisor smoothing
  local dir="$1" mode="$2" budget="$3" divisor="$4" smoothing="$5"
  local extra=()
  [ "$mode" = "plain" ] && extra=(--skip-remesh --relief-smooth-iterations 0)
  [ "$mode" = "remesh" ] && extra=(--voxel-divisor "$divisor"
                                   --relief-smooth-iterations "$smoothing")
  "$BLENDER" --background --python-exit-code 1 --python \
    "$AVENGINE_ROOT/tools/assets/retopologize_for_rigging.py" -- \
    --input "$RAW" --output "$dir/prepared.glb" --report "$dir/prepared.json" \
    --target-faces "$budget" --front-yaw-deg "$YAW" "${extra[@]}" \
    > "$dir/prepare.log" 2>&1 || return 1
  [ -s "$dir/prepared.glb" ] && [ -s "$dir/prepared.json" ]
}

rig_and_animate() {  # rung_dir
  local dir="$1"
  "$RUNTIME_PYTHON" "$AVENGINE_ROOT/tools/assets/run_skintokens_rig.py" \
    --input "$dir/prepared.glb" --output "$dir/rig.glb" \
    --blender "$BLENDER" --use-skeleton --use-transfer --use-postprocess \
    > "$dir/rig.log" 2>&1 || return 1
  [ -s "$dir/rig.glb" ] || return 1

  "$BLENDER" --background --python-exit-code 1 --python \
    "$AVENGINE_ROOT/tools/assets/blender_normalize_generated_animal_heading.py" -- \
    --input "$dir/rig.glb" --output "$dir/normalized.glb" --manifest "$dir/heading.json" \
    --reviewed-source-front-yaw-deg "$YAW" --target-front-axis "$TARGET_FRONT_AXIS" \
    --review-evidence "$dir/prepared.json" > "$dir/heading.log" 2>&1 || return 1
  grep -q "NORMALIZATION_OK" "$dir/heading.log" || return 1
  [ -s "$dir/normalized.glb" ] && [ -s "$dir/heading.json" ] || return 1

  "$BLENDER" --background --python-exit-code 1 --python "$AVENGINE_ROOT/tools/assets/probe_heading_axis.py" -- \
    "$dir/normalized.glb" "$dir/heading_probe.png" > "$dir/probe.log" 2>&1 || return 1
  [ -s "$dir/heading_probe.png" ] || return 1

  "$BLENDER" --background --python-exit-code 1 --python \
    "$AVENGINE_ROOT/tools/assets/blender_level_generated_animal_support_plane.py" -- \
    --input "$dir/normalized.glb" --output "$dir/leveled.glb" \
    --manifest "$dir/level.json" --front-axis "$TARGET_FRONT_AXIS" \
    --review-evidence "$dir/heading.json" > "$dir/level.log" 2>&1 || return 1
  grep -q "LEVELING_OK" "$dir/level.log" || return 1
  [ -s "$dir/leveled.glb" ] && [ -s "$dir/level.json" ] || return 1

  "$BLENDER" --background --python-exit-code 1 --python \
    "$AVENGINE_ROOT/tools/assets/blender_retarget_quaternius_to_generated_quadruped.py" -- \
    --target-glb "$dir/leveled.glb" --source-rig-glb "$DONOR" \
    --output-glb "$dir/animated.glb" --manifest "$dir/retarget.json" \
    --target-front-axis "$TARGET_FRONT_AXIS" --technical-spike-only \
    --motion-basis-yaw-deg "$MOTION_BASIS_YAW_DEG" --side-chain-mode "$SIDE_CHAIN_MODE" > "$dir/retarget.log" 2>&1 || return 1
  grep -q "RETARGET_OK" "$dir/retarget.log" || return 1
  [ -s "$dir/animated.glb" ] && [ -s "$dir/retarget.json" ] || return 1

  "$BLENDER" --background --factory-startup --python-exit-code 1 --python \
    "$AVENGINE_ROOT/tools/assets/validate_animated_animal_closure.py" -- \
    "$dir/animated.glb" "$dir/level.json" "$dir/retarget.json" "$dir/closure.json" \
    --policy-config "$POLICY_CONFIG" --strategy "$STRATEGY" \
    > "$dir/closure.log" 2>&1 || return 1
  [ -s "$dir/closure.json" ] || return 1

  "$BLENDER" --background --python-exit-code 1 --python "$AVENGINE_ROOT/tools/assets/measure_walk_deformation.py" -- \
    "$dir/animated.glb" "$dir/walk_deformation.json" "$DEFORM_ACTION" \
    "$SAMPLE_COUNT" "$SHARD_EDGE_GROWTH" "$UNDERSIDE_FRACTION" \
    "$MAX_ABS_POSITION" "$MAX_ABS_SCALE" > "$dir/deform.log" 2>&1 || return 1
  grep -E "^WALK_DEFORMATION_OK" "$dir/deform.log" || return 1
  [ -s "$dir/walk_deformation.json" ]
}

ACCEPTED="" BEST_SHARDS=""
for rung in "${RUNGS[@]}"; do
  IFS=':' read -r rung_id mode budget divisor smoothing <<< "$rung"
  dir="$WORKDIR/rung_${rung_id}"
  mkdir -p "$dir"
  step "rung $rung_id ($mode $budget faces divisor ${divisor:-n/a})"

  if ! prepare "$dir" "$mode" "$budget" "$divisor" "$smoothing"; then
    echo "preparation failed, see $dir/prepare.log" >&2
    continue
  fi
  gate_args=( "$dir/prepared.json" --policy-config "$POLICY_CONFIG" --strategy "$STRATEGY" )
  if ! "$RUNTIME_PYTHON" "$AVENGINE_ROOT/tools/assets/gate_retopology.py" "${gate_args[@]}" \
       > "$dir/retopo_gate.log" 2>&1; then
    echo "retopology gate failed, see $dir/retopo_gate.log" >&2
    continue
  fi
  if ! rig_and_animate "$dir"; then
    echo "rig, closure, or animation failed, see the logs under $dir" >&2
    continue
  fi
  gate_args=( "$dir/walk_deformation.json" --policy-config "$POLICY_CONFIG" --strategy "$STRATEGY" )
  if ! "$RUNTIME_PYTHON" "$AVENGINE_ROOT/tools/assets/gate_rigged_asset.py" "${gate_args[@]}" \
       > "$dir/rigged_gate.log" 2>&1; then
    echo "rigged-asset gate failed, see $dir/rigged_gate.log" >&2
    continue
  fi

  shards="$( "$RUNTIME_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['worst_share_area_shards'])" \
           "$dir/walk_deformation.json" )"
  # Preserve the historical stochastic-rigger retry only for strict metrics.
  if [ "$METRIC_GATES" = "1" ] && [ "$RIG_RETRIES" -gt 0 ] && \
     [ "$( "$RUNTIME_PYTHON" -c "print(1 if float('$shards') > float('$RETRY_BAND') * float('$MAX_SHARD_SHARE') else 0)" )" = "1" ]; then
    for attempt in $(seq 1 "$RIG_RETRIES"); do
      retry="$dir/retry$attempt"
      mkdir -p "$retry"
      cp "$dir/prepared.glb" "$retry/prepared.glb"
      cp "$dir/prepared.json" "$retry/prepared.json"
      rig_and_animate "$retry" > "$retry/retry.log" 2>&1 || continue
      retry_shards="$( "$RUNTIME_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['worst_share_area_shards'])" \
        "$retry/walk_deformation.json" )"
      if [ "$( "$RUNTIME_PYTHON" -c "print(1 if float('$retry_shards') < float('$shards') else 0)" )" = "1" ]; then
        shards="$retry_shards"
        for name in animated.glb walk_deformation.json closure.json heading.json level.json retarget.json heading_probe.png; do
          [ -f "$retry/$name" ] && cp "$retry/$name" "$dir/$name"
        done
      fi
    done
    gate_args=( "$dir/walk_deformation.json" --policy-config "$POLICY_CONFIG" --strategy "$STRATEGY" )
    "$RUNTIME_PYTHON" "$AVENGINE_ROOT/tools/assets/gate_rigged_asset.py" "${gate_args[@]}" \
      > "$dir/rigged_gate_after_retry.log" 2>&1 || continue
  fi

  candidate_wins=0
  if [ -z "$ACCEPTED" ] || [ "$STRATEGY" = "visual_review" ] || [ "$PICK" = "first" ]; then
    candidate_wins=1
  elif [ "$( "$RUNTIME_PYTHON" -c "print(1 if float('$shards') < float('$BEST_SHARDS') else 0)" )" = "1" ]; then
    candidate_wins=1
  fi
  if [ "$candidate_wins" = "1" ]; then
    ACCEPTED="$rung_id"
    BEST_SHARDS="$shards"
    for name in prepared.glb prepared.json rig.glb normalized.glb leveled.glb animated.glb \
                walk_deformation.json closure.json heading.json level.json retarget.json heading_probe.png; do
      [ -f "$dir/$name" ] && cp "$dir/$name" "$WORKDIR/$name"
    done
  fi
  # Visual review always hands off the first structurally complete rung.
  [ "$STRATEGY" = "visual_review" ] && break
  [ "$PICK" = "first" ] && [ "$candidate_wins" = "1" ] && break
done

if [ -z "$ACCEPTED" ]; then
  echo "no structurally complete rung in the configured ladder passed the hard checks" >&2
  exit 66
fi

rendered_count() {
  find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}
check_render_dir() {
  local dir="$1" expected="$2" count
  [ -d "$dir" ] || return 1
  count="$(rendered_count "$dir")"
  [ "$count" -eq "$expected" ] || return 1
  while IFS= read -r file; do
    [ -r "$file" ] || return 1
  done < <(find "$dir" -maxdepth 1 -type f -name '*.png')
}

step "structurally complete rung $ACCEPTED, rendering ordinary-distance reviews"
if ! "$BLENDER" --background --python-exit-code 1 --python "$AVENGINE_ROOT/tools/assets/render_walk_review.py" -- \
     "$WORKDIR/animated.glb" "$WORKDIR/$WALK_DIR" "$WALK_ACTION" "$WALK_FRAMES" "$WALK_ZOOM" \
     > "$WORKDIR/walk.log" 2>&1; then
  echo "walking review render failed, see $WORKDIR/walk.log" >&2
  exit 67
fi
grep -E "^CLEANWALK_OK" "$WORKDIR/walk.log" >/dev/null || {
  echo "walking review render did not report completion" >&2
  exit 67
}
check_render_dir "$WORKDIR/$WALK_DIR" "$WALK_FRAMES" || {
  echo "walking review render is incomplete or unreadable" >&2
  exit 67
}

if ! "$BLENDER" --background --python-exit-code 1 --python "$AVENGINE_ROOT/tools/assets/render_turntable_review.py" -- \
     "$WORKDIR/animated.glb" "$WORKDIR/$TURN_DIR" "$TURN_ACTION" "$TURN_FRAMES" "$TURN_POSE_RATIO" \
     > "$WORKDIR/turntable.log" 2>&1; then
  echo "turntable review render failed, see $WORKDIR/turntable.log" >&2
  exit 67
fi
grep -E "^TURNTABLE_OK" "$WORKDIR/turntable.log" >/dev/null || {
  echo "turntable review render did not report completion" >&2
  exit 67
}
check_render_dir "$WORKDIR/$TURN_DIR" "$TURN_FRAMES" || {
  echo "turntable review render is incomplete or unreadable" >&2
  exit 67
}

STATUS="accepted_for_dataset_asset"
if [ "$STRATEGY" = "visual_review" ]; then
  STATUS="needs_visual_review"
fi
"$RUNTIME_PYTHON" - \
  "$WORKDIR/ladder.json" "$STATUS" "$POLICY_ID" "$STRATEGY" \
  "$ACCEPTED" "$BEST_SHARDS" "$PICK" "$VIEW_DISTANCE" "${RUNGS[@]}" <<'PY' || {
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "kind": "generated_animal_ladder_result",
    "status": sys.argv[2],
    "policy_id": sys.argv[3],
    "strategy": sys.argv[4],
    "accepted_rung": sys.argv[5],
    "worst_share_area_shards": float(sys.argv[6]),
    "pick": sys.argv[7],
    "view_distance": sys.argv[8],
    "ladder": sys.argv[9:],
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
PY
  echo "could not write ladder result" >&2
  exit 67
}

if [ "$STRATEGY" = "visual_review" ]; then
  if [ -z "$REVIEW_ID" ]; then REVIEW_ID="$(basename "$WORKDIR")"; fi
  manifest_args=(review-manifest --output "$WORKDIR/visual_review.json" --review-id "$REVIEW_ID" \
                 --review-path "$WORKDIR" --policy-id "$POLICY_ID" --strategy "$STRATEGY" \
                 --accepted-rung "$ACCEPTED" --walking-render "$WORKDIR/$WALK_DIR" \
                 --turntable-render "$WORKDIR/$TURN_DIR")
  [ -n "$ASSET_ID" ] && manifest_args+=(--asset-id "$ASSET_ID")
  "$RUNTIME_PYTHON" "$POLICY_MODULE" "${manifest_args[@]}" > "$WORKDIR/visual_review.log" 2>&1 || {
    echo "could not write visual review manifest" >&2
    exit 67
  }
fi
echo "=== CHAIN_DONE $WORKDIR rung=$ACCEPTED status=$STATUS ==="
