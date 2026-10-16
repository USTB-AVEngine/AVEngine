#!/usr/bin/env bash
# Take one generated animal from a raw reconstruction to a reviewable rigged
# asset, trying the cheapest preparation that works rather than one fixed
# recipe.
#
# Neither the recipe nor its settings generalised from the animal they were
# chosen on. A plain weld-and-collapse to 25k is enough for some breeds and
# produces a better-looking asset than a voxel retopology does - measured, one
# cat reads 0.011 percent of area stretched past ten times during the walk at
# 25k against 0.123 at 80k retopologised. For others the same plain route
# starves the head to 0.54 of its share or cannot reach the target at all,
# because collapse decimation cannot touch a non-manifold edge and a
# reconstruction carrying thousands of them spends the whole budget elsewhere.
#
# The inversion is total: measured on the share of posed area torn into shards at
# the worst frame of the walk, one cat reads 0.24 percent on a plain collapse to
# 25k and 3.90 at 80k, while a dog reads 1.27 at 25k and 0.47 at 80k. The same
# setting is the best and the worst option depending on the breed, which is why
# no fixed recipe can work and why the gate rather than the setting is the thing
# that travels.
#
# So this walks a ladder. Each rung is prepared, gated before rigging, rigged,
# animated, and gated again on how the surface deforms. With --pick first (the
# default) the first rung that passes both wins and the rest are not attempted;
# with --pick best every rung runs and the one with the least tearing wins, which
# costs the whole ladder and is worth it for an asset going into production.
# Evidence from the rungs that failed is kept.
#
# See docs/assets/MESH_DENSITY_AND_TEARING_20260825.md for the measurements.
#
# Usage:
#   tools/assets/run_generated_animal_chain.sh \
#     --raw /path/raw.glb --workdir /path/out --front-yaw-deg -179.706 \
#     --donor-rig /path/donor_walk_idle.glb --spear-root /path/SPEAR \
#     --rigger-root /path/SkinTokens [--gpu 0] [--port-base 59875] \
#     [--ladder plain:25000:0,plain:80000:0,remesh:80000:800,remesh:80000:700,remesh:120000:800]
#     [--pick first|best] [--retry-band 0.8] [--rig-retries 1]

set -uo pipefail

RAW="" WORKDIR="" YAW="" DONOR="" SPEAR_ROOT="" RIGGER_ROOT=""
GPU=0 PORT_BASE=59875 RELIEF_SMOOTH=-1 RETRY_BAND=0.8 RIG_RETRIES=1
LADDER="plain:25000:0,plain:80000:0,remesh:80000:800,remesh:80000:700,remesh:120000:800"
PICK=first
BLENDER="${BLENDER:-blender}"

while [ $# -gt 0 ]; do
  case "$1" in
    --raw) RAW="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --front-yaw-deg) YAW="$2"; shift 2 ;;
    --donor-rig) DONOR="$2"; shift 2 ;;
    --spear-root) SPEAR_ROOT="$2"; shift 2 ;;
    --rigger-root) RIGGER_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --ladder) LADDER="$2"; shift 2 ;;
    --pick) PICK="$2"; shift 2 ;;
    --retry-band) RETRY_BAND="$2"; shift 2 ;;
    --rig-retries) RIG_RETRIES="$2"; shift 2 ;;
    --relief-smooth) RELIEF_SMOOTH="$2"; shift 2 ;;
    --blender) BLENDER="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

for required in RAW WORKDIR YAW DONOR SPEAR_ROOT RIGGER_ROOT; do
  if [ -z "${!required}" ]; then
    echo "missing --${required,,} (see the usage comment)" >&2
    exit 64
  fi
done

# fresh/no-clobber: an existing workdir is evidence from another run.
if [ -e "$WORKDIR" ]; then
  echo "workdir already exists, refusing to overwrite: $WORKDIR" >&2
  exit 65
fi
mkdir -p "$WORKDIR"

AVENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BPY_PORT=$((PORT_BASE + 1))
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

prepare() {  # rung_dir mode budget divisor
  local dir="$1" mode="$2" budget="$3" divisor="$4"
  local extra=()
  [ "$mode" = "plain" ] && extra=(--skip-remesh --relief-smooth-iterations 0)
  [ "$mode" = "remesh" ] && extra=(--voxel-divisor "$divisor"
                                   --relief-smooth-iterations "$RELIEF_SMOOTH")
  "$BLENDER" --background --python \
    "$AVENGINE_ROOT/tools/assets/retopologize_for_rigging.py" -- \
    --input "$RAW" --output "$dir/prepared.glb" --report "$dir/prepared.json" \
    --target-faces "$budget" --front-yaw-deg "$YAW" "${extra[@]}" \
    > "$dir/prepare.log" 2>&1
}

rig_and_animate() {  # rung_dir
  local dir="$1"
  mkdir -p "$dir/rig_patch"
  # Two patches the vendored rigger needs, applied per run rather than in place.
  # Its wait is 30 seconds against a mesh that takes minutes, and its helper asks
  # bottle for 0.0.0.0 on a dual-stack host, where tornado binds :: first and
  # then fails on 0.0.0.0 inside the server thread - it prints "Listening" and
  # serves nothing.
  sed "s/def wait_for_bpy_server(timeout=30)/def wait_for_bpy_server(timeout=1200)/" \
    "$RIGGER_ROOT/demo.py" > "$dir/rig_patch/demo_longwait.py"
  cp "$AVENGINE_ROOT/tools/assets/rigger_loopback_bpy_server.py" \
    "$dir/rig_patch/bpy_server.py"
  for shared in experiments assets configs models; do
    ln -sfn "$RIGGER_ROOT/$shared" "$dir/rig_patch/$shared"
  done
  cp -r "$RIGGER_ROOT/src" "$dir/rig_patch/src"
  sed -i "s/^PORT = .*/PORT = $PORT_BASE/; s/^BPY_PORT = .*/BPY_PORT = $BPY_PORT/" \
    "$dir/rig_patch/src/server/spec.py"
  fuser -k "${PORT_BASE}/tcp" "${BPY_PORT}/tcp" 2>/dev/null || true
  sleep 2
  ( cd "$dir/rig_patch" && "$RIGGER_ROOT/.venv/bin/python" demo_longwait.py \
      --input "$dir/prepared.glb" --output "$dir/rig" \
      --model_ckpt experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt \
      --use_skeleton --use_transfer --use_postprocess > "$dir/rig.log" 2>&1 ) || return 1
  local rigged="$dir/rig.glb"
  [ -f "$rigged" ] || rigged="$dir/rig"

  # Log first, then look for the success marker in the log. Piping Blender into
  # `grep -q` makes grep close the pipe on its first match, which sends Blender
  # SIGPIPE and turns a completed step into a failed one under pipefail.
  ( cd "$SPEAR_ROOT" && \
    "$BLENDER" --background --python tools/blender_normalize_generated_animal_heading.py -- \
      --input "$rigged" --output "$dir/normalized.glb" --manifest "$dir/heading.json" \
      --reviewed-source-front-yaw-deg "$YAW" --target-front-axis positive-x \
      --review-evidence "$dir/prepared.json" ) > "$dir/heading.log" 2>&1
  grep -q "NORMALIZATION_OK" "$dir/heading.log" || return 1
  # The forward estimator has been wrong with high confidence; look at the probe.
  "$BLENDER" --background --python "$AVENGINE_ROOT/tools/assets/probe_heading_axis.py" -- \
    "$dir/normalized.glb" "$dir/heading_probe.png" > "$dir/probe.log" 2>&1
  ( cd "$SPEAR_ROOT" && \
    "$BLENDER" --background --python tools/blender_level_generated_animal_support_plane.py -- \
      --input "$dir/normalized.glb" --output "$dir/leveled.glb" \
      --manifest "$dir/level.json" --front-axis positive-x \
      --review-evidence "$dir/heading.json" ) > "$dir/level.log" 2>&1
  grep -q "LEVELING_OK" "$dir/level.log" || return 1
  ( cd "$SPEAR_ROOT" && \
    "$BLENDER" --background --python tools/blender_retarget_quaternius_to_generated_quadruped.py -- \
      --target-glb "$dir/leveled.glb" --source-rig-glb "$DONOR" \
      --output-glb "$dir/animated.glb" --manifest "$dir/retarget.json" \
      --target-front-axis positive-x --technical-spike-only \
      --motion-basis-yaw-deg 0 --side-chain-mode matched ) > "$dir/retarget.log" 2>&1
  grep -q "RETARGET_OK" "$dir/retarget.log" || return 1
  # Sweep the cycle rather than sampling one pose: a single frame at 35 percent
  # through the action understated the worst frame by ten to thirteen times.
  "$BLENDER" --background --python "$AVENGINE_ROOT/tools/assets/measure_walk_deformation.py" -- \
    "$dir/animated.glb" "$dir/walk_deformation.json" Walking 16 > "$dir/deform.log" 2>&1
  grep -E "^WALK_DEFORMATION_OK" "$dir/deform.log" || return 1
}

ACCEPTED="" BEST_SHARDS=""
IFS=',' read -r -a RUNGS <<< "$LADDER"
for rung in "${RUNGS[@]}"; do
  IFS=':' read -r mode budget divisor <<< "$rung"
  dir="$WORKDIR/rung_${mode}_${budget}_${divisor}"
  mkdir -p "$dir"
  step "rung ${mode} ${budget} faces divisor ${divisor:-n/a}"

  if ! prepare "$dir" "$mode" "$budget" "$divisor"; then
    echo "preparation failed, see $dir/prepare.log" >&2
    continue
  fi
  if ! python3 "$AVENGINE_ROOT/tools/assets/gate_retopology.py" "$dir/prepared.json"; then
    continue
  fi
  if ! rig_and_animate "$dir"; then
    echo "rig or animation failed, see the logs under $dir" >&2
    continue
  fi
  if ! python3 "$AVENGINE_ROOT/tools/assets/gate_rigged_asset.py" \
       "$dir/walk_deformation.json"; then
    continue
  fi

  shards=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['worst_share_area_shards'])" \
           "$dir/walk_deformation.json")
  # The rigger is stochastic: the same rung of the same ladder has measured 0.0187
  # to 0.0263 across six attempts. A reading that only just cleared the threshold
  # cleared it on the draw, so re-rig that one and keep the better result. Rungs
  # with real margin never enter this branch, which is why retries are not paid
  # for on every asset.
  marginal=$(python3 -c "print(1 if float('$shards') > $RETRY_BAND * 0.025 else 0)")
  if [ "$marginal" = "1" ] && [ "$RIG_RETRIES" -gt 0 ]; then
    echo "--- $shards is inside the noise band, re-rigging up to $RIG_RETRIES time(s)"
    for attempt in $(seq 1 "$RIG_RETRIES"); do
      retry="$dir/retry$attempt"
      mkdir -p "$retry"
      cp "$dir/prepared.glb" "$retry/prepared.glb"
      cp "$dir/prepared.json" "$retry/prepared.json"
      rig_and_animate "$retry" > /dev/null 2>&1 || continue
      again=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['worst_share_area_shards'])" \
              "$retry/walk_deformation.json")
      echo "--- retry $attempt: $again"
      if [ "$(python3 -c "print(1 if float('$again') < float('$shards') else 0)")" = "1" ]; then
        shards="$again"
        for name in animated.glb walk_deformation.json heading.json level.json \
                    retarget.json heading_probe.png; do
          [ -f "$retry/$name" ] && cp "$retry/$name" "$dir/$name"
        done
      fi
    done
    if ! python3 "$AVENGINE_ROOT/tools/assets/gate_rigged_asset.py" \
         "$dir/walk_deformation.json" > /dev/null; then
      continue
    fi
  fi
  if [ -z "$ACCEPTED" ] || \
     [ "$(python3 -c "print(1 if float('$shards') < float('$BEST_SHARDS') else 0)")" = "1" ]; then
    ACCEPTED="$rung"
    BEST_SHARDS="$shards"
    for name in prepared.glb prepared.json animated.glb walk_deformation.json \
                heading.json level.json retarget.json heading_probe.png; do
      [ -f "$dir/$name" ] && cp "$dir/$name" "$WORKDIR/$name"
    done
  fi
  [ "$PICK" = "first" ] && break
done

if [ -z "$ACCEPTED" ]; then
  echo "no rung in [$LADDER] passed both gates; the rejections above name the" >&2
  echo "reading that failed. A starved head calls for a larger budget or the" >&2
  echo "remesh route; a surface that keeps folding calls for fewer faces." >&2
  exit 66
fi

step "accepted rung $ACCEPTED (shards $BEST_SHARDS), rendering reviews"
"$BLENDER" --background --python "$AVENGINE_ROOT/tools/assets/render_walk_review.py" -- \
  "$WORKDIR/animated.glb" "$WORKDIR/walk" Walking 16 1.15 > "$WORKDIR/walk.log" 2>&1
grep -E "CLEANWALK_OK" "$WORKDIR/walk.log"
"$BLENDER" --background --python "$AVENGINE_ROOT/tools/assets/render_turntable_review.py" -- \
  "$WORKDIR/animated.glb" "$WORKDIR/turntable" Walking 36 0.0 > "$WORKDIR/turntable.log" 2>&1
grep -E "TURNTABLE_OK" "$WORKDIR/turntable.log"
printf '{"accepted_rung": "%s", "worst_share_area_shards": %s, "pick": "%s", "ladder": "%s"}\n' \
  "$ACCEPTED" "$BEST_SHARDS" "$PICK" "$LADDER" > "$WORKDIR/ladder.json"
echo "=== CHAIN_DONE $WORKDIR rung=$ACCEPTED ==="
