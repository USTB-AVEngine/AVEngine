#!/usr/bin/env bash
# Take one generated animal from a raw reconstruction to a reviewable rigged
# asset: prepare the mesh, rig it, orient it, transfer the donor walk, and
# render the reviews.
#
# The preparation step is a voxel retopology rather than a plain decimation.
# A collapse decimator cannot touch a non-manifold edge, and a reconstruction
# carrying thousands of them spends the whole reduction budget on whichever end
# of the animal happens to be clean - measured, the head kept half the share the
# tail kept, which is what a destroyed face looks like as a number. Remeshing
# first removes that failure for every breed. See
# docs/assets/MESH_DENSITY_AND_TEARING_20260825.md for the evidence.
#
# The gate between preparation and rigging is there because rigging is the
# expensive step: a mesh that lost its head is cheaper to reject than to rig.
#
# Usage:
#   tools/assets/run_generated_animal_chain.sh \
#     --raw /path/raw.glb --workdir /path/out --front-yaw-deg -179.706 \
#     --donor-rig /path/donor_walk_idle.glb --spear-root /path/SPEAR \
#     --rigger-root /path/SkinTokens [--gpu 0] [--port-base 59875] \
#     [--target-faces 80000] [--voxel-divisor 800] [--relief-smooth -1]

set -euo pipefail

RAW="" WORKDIR="" YAW="" DONOR="" SPEAR_ROOT="" RIGGER_ROOT=""
GPU=0 PORT_BASE=59875 TARGET_FACES=80000 VOXEL_DIVISOR=800 RELIEF_SMOOTH=-1
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
    --target-faces) TARGET_FACES="$2"; shift 2 ;;
    --voxel-divisor) VOXEL_DIVISOR="$2"; shift 2 ;;
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

step "1/6 retopologize to $TARGET_FACES faces at diagonal/$VOXEL_DIVISOR"
"$BLENDER" --background --python "$AVENGINE_ROOT/tools/assets/retopologize_for_rigging.py" -- \
  --input "$RAW" --output "$WORKDIR/prepared.glb" --report "$WORKDIR/prepared.json" \
  --target-faces "$TARGET_FACES" --voxel-divisor "$VOXEL_DIVISOR" \
  --relief-smooth-iterations "$RELIEF_SMOOTH" 2>&1 | grep -E "^RETOPOLOGY_OK"

step "2/6 gate the preparation before spending a rigging slot"
python3 "$AVENGINE_ROOT/tools/assets/gate_retopology.py" "$WORKDIR/prepared.json"

step "3/6 rig"
mkdir -p "$WORKDIR/rig_patch"
# The rigger binds its tornado helper on every interface, so two runs on one
# host collide on the port rather than queueing. Each run gets its own copy of
# the source tree with its own ports.
sed "s/def wait_for_bpy_server(timeout=30)/def wait_for_bpy_server(timeout=1200)/" \
  "$RIGGER_ROOT/demo.py" > "$WORKDIR/rig_patch/demo_longwait.py"
for shared in experiments assets configs models; do
  ln -sfn "$RIGGER_ROOT/$shared" "$WORKDIR/rig_patch/$shared"
done
cp -r "$RIGGER_ROOT/src" "$WORKDIR/rig_patch/src"
sed -i "s/^PORT = .*/PORT = $PORT_BASE/; s/^BPY_PORT = .*/BPY_PORT = $BPY_PORT/" \
  "$WORKDIR/rig_patch/src/server/spec.py"
fuser -k "${PORT_BASE}/tcp" "${BPY_PORT}/tcp" 2>/dev/null || true
sleep 2
( cd "$WORKDIR/rig_patch" && "$RIGGER_ROOT/.venv/bin/python" demo_longwait.py \
    --input "$WORKDIR/prepared.glb" --output "$WORKDIR/rig" \
    --model_ckpt experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt \
    --use_skeleton --use_transfer --use_postprocess > "$WORKDIR/rig.log" 2>&1 )
RIGGED="$WORKDIR/rig.glb"
[ -f "$RIGGED" ] || RIGGED="$WORKDIR/rig"

step "4/6 orient and level"
cd "$SPEAR_ROOT"
"$BLENDER" --background --python tools/blender_normalize_generated_animal_heading.py -- \
  --input "$RIGGED" --output "$WORKDIR/normalized.glb" --manifest "$WORKDIR/heading.json" \
  --reviewed-source-front-yaw-deg "$YAW" --target-front-axis positive-x \
  --review-evidence "$WORKDIR/prepared.json" 2>&1 | grep -E "NORMALIZATION_OK"
# The forward estimator has been wrong with high confidence; look at the probe.
"$BLENDER" --background --python "$AVENGINE_ROOT/tools/assets/probe_heading_axis.py" -- \
  "$WORKDIR/normalized.glb" "$WORKDIR/heading_probe.png" 2>&1 | grep -E "PROBE_OK"
"$BLENDER" --background --python tools/blender_level_generated_animal_support_plane.py -- \
  --input "$WORKDIR/normalized.glb" --output "$WORKDIR/leveled.glb" \
  --manifest "$WORKDIR/level.json" --front-axis positive-x \
  --review-evidence "$WORKDIR/heading.json" 2>&1 | grep -E "LEVELING_OK"

step "5/6 transfer the donor walk"
"$BLENDER" --background --python tools/blender_retarget_quaternius_to_generated_quadruped.py -- \
  --target-glb "$WORKDIR/leveled.glb" --source-rig-glb "$DONOR" \
  --output-glb "$WORKDIR/animated.glb" --manifest "$WORKDIR/retarget.json" \
  --target-front-axis positive-x --technical-spike-only \
  --motion-basis-yaw-deg 0 --side-chain-mode matched 2>&1 | grep -E "RETARGET_OK"

step "6/6 measure and review"
cd "$AVENGINE_ROOT"
"$BLENDER" --background --python tools/assets/measure_deformation_stretch.py -- \
  "$WORKDIR/animated.glb" "$WORKDIR/stretch.json" Walking 0.35 2>&1 | grep -E "STRETCH"
"$BLENDER" --background --python tools/assets/render_walk_review.py -- \
  "$WORKDIR/animated.glb" "$WORKDIR/walk" Walking 16 1.15 2>&1 | grep -E "CLEANWALK_OK"
"$BLENDER" --background --python tools/assets/render_turntable_review.py -- \
  "$WORKDIR/animated.glb" "$WORKDIR/turntable" Walking 36 0.0 2>&1 | grep -E "TURNTABLE_OK"

echo "=== CHAIN_DONE $WORKDIR ==="
