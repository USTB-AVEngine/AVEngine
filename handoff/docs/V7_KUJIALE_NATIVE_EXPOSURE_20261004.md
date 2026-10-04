# Kujiale native exposure handoff, 2026-10-04

Source commit `86e7cbcae5688f5ad0f907ab2b486620c1792dff` on own `codex/v7-kujiale-round9-20261004`; not pushed.
Native rendering and all 44-room re-admission are pending. Shared `/data`
is below the owner's 80 GB floor; no UE/GPU/CCI jobs were launched.

The old native bake cleared light transforms after baking mesh points.
The new bake preserves authored light world transforms, and the editor
asserts native lamp position/direction readbacks. Use fresh isolated map and
destination names, not an audit-only reapplication of the reflection.
Raw InteriorAgent and shared Content remain untouched.

Unified profile: `kujiale-native-authored-lights-histogram-v1`,
EV100 -4..16, histogram -10..20 at 10..90 percentiles, bias 0, WB 4000 K,
RGB-only EyeAdaptation show-flag setter and 300 native warmup frames.
This is a uniform initial candidate, not a calibrated brightness result.
The native formal backend requires measured authored-light and new room QA.

48g outputs: `/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round9`. `pending_native_v3/jobs.json` has 21 houses / 44 rooms.
0031/R1 and 0066/R1 retain fewer than three original frozen viewpoints;
do not fabricate placements or relax admission. Delivery page is
`delivery_v1/BEFORE_AFTER_pending.html`: 44 rows / 173 embedded old images,
zero new images. Do not send it to the owner as a completed acceptance page.

CPU fast suite: 371 pass / 36 skip; original frozen QA/material files are
byte-identical (50 files). The standalone lamp helper runs in UE's embedded
Python 3.11.8 without NumPy/SciPy. Native Unreal component APIs and new
brightness/contrast/VRAM/throughput remain unverified.

Canonical v2: after the prepared d64cc247 port, apply
`evidence/UE_R9_CODE_AFTER_V2_PORT.patch`, SHA256 `91cb716e86336150becdafedef2f95eeb155cf3bdcad5c6628adddc50cdcad86`,
then regenerate TOOL_INDEX in the recipient's own checkout. Source apply
check passes; upstream QA/gold and v2 semantics are preserved. Peer checkout
was not modified, and the v2 GPU pipeline is not qualified.

Before generation, bind the new map, native_lighting_audit and
native_room_visual_qa while retaining original material bindings, native
floor surface and pose settling. Re-run the old room gates and formal scene
canary. Older 29-room passes, 3.9 GB peak and 8.41-hour v7-S estimate do not
qualify this new visual setting. Restore shared disk floor/headroom first.
