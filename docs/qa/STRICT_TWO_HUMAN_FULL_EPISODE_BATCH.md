# Strict two-human full-Episode batch procedure

This workflow promotes the passed single-frame strict gates to complete
75-frame, 5-second research Episodes. It remains fail-closed: planned rows are
not formal data, and changing a mirror, voice, transcript, or answer order does
not create an independent Episode.

## Current boundary

`apartment_0000` is the only room with the complete M/F/C cooked runtime,
camera/RGB/metric-depth/two-target-only/live-readback path, occupied-floor
evidence, and exact acoustic package. The generated 100-row single-room bank is
therefore interim. Only its first 20 rows are the initial mechanism pilot. The
requested final 100 requires at least three genuinely runnable room IDs.

The current cooked package also contains real `debug_0000` and `debug_0001`
maps. They are not counted as ready rooms until native smoke loading, bounded
floor/placement evidence, exact acoustic geometry/material registration, RIR,
and one strict full75 canary pass. They are not residential-room claims.

## CPU planning

Run:

```bash
python3 tools/qa/build_strict_two_human_full_episode_batch.py \
  --output tmp/lead_a_strict_two_human_full_episode_batch_v1/cpu_plan_v1
```

The builder reads the retained 1000-Episode Apartment source bank and checks
selected roots against native runtime readbacks. Required output invariants:

- 100 unique 0.75 m camera clusters;
- 100 unique retained source Episodes;
- 50 left and 50 right targets;
- 20 rows for each of five motion/camera mechanisms;
- 200 unique exact-RIR job IDs;
- ten batches of ten;
- zero formal Episodes and no qualification claim.

Run the extra-map audit separately:

```bash
python3 tools/qa/audit_strict_two_human_room_expansion.py \
  --output tmp/lead_a_strict_two_human_full_episode_batch_v1/room_expansion_audit_v1
```

Prepare the two cooked debug-map probes with the repository environment:

```bash
.venv/bin/python tools/qa/build_strict_two_human_debug_room_preflight.py \
  --output tmp/lead_a_strict_two_human_full_episode_batch_v1/debug_room_cpu_preflight_v1
```

This emits a complete 75-frame M/F suite and a sparse `[0, 15, 74]` visual
probe request for each map. The generated stereo silence is mux transport only;
it is never acoustic evidence. Proposed floor placements remain provisional,
and each pending acoustic plan remains non-executable until exact native
surface geometry and a reviewed material mapping exist.

## Four full75 canaries

The CPU plan publishes `canary_plan.json` for strict sparse rows 1–4. Run them
one at a time. The launcher refuses to proceed unless physical GPU1 has zero
compute processes, passes `--graphics-adapter 1`, omits all sparse
`--frame-index` arguments, and records the capture process exit code.

```bash
python3 tools/qa/run_strict_two_human_full75_canary.py \
  --canary-plan tmp/lead_a_strict_two_human_full_episode_batch_v1/cpu_plan_v1/canary_plan.json \
  --canary-index 1 \
  --receipt tmp/lead_a_strict_two_human_full_episode_batch_v1/full75_canaries/strict_01_existing_canary.launch.json
```

Then finalize the same canary:

```bash
python3 tools/qa/finalize_strict_two_human_full75_canary.py \
  --canary-plan tmp/lead_a_strict_two_human_full_episode_batch_v1/cpu_plan_v1/canary_plan.json \
  --canary-index 1 \
  --capture-root tmp/lead_a_strict_two_human_full_episode_batch_v1/full75_canaries/strict_01_existing_canary \
  --launch-receipt tmp/lead_a_strict_two_human_full_episode_batch_v1/full75_canaries/strict_01_existing_canary.launch.json \
  --output tmp/lead_a_strict_two_human_full_episode_batch_v1/full75_canary_final/strict_01_existing_canary
```

The finalizer requires all 75 normal RGB frames, metric depth, two 75-frame
target-only passes, normal and target-only runtime alignment, live Blueprint /
mesh / skeleton / Idle / stable-tag / emitter readbacks, target visibility of
at least 0.8 while speaking, distractor visibility of at least 0.5, exact two-
source RIR evidence, a 5-second 16 kHz stereo mixture, and two 75-frame videos.

Do not launch the first 20 mechanism Episodes until all four canaries pass. Do
not call rows 21–100 final until three real room closures exist.
