# Strict two-human native-rate dynamic candidate v1

This CPU-only revision replaces the temporal policy of the three legacy moving
canaries without changing their controlled sound-event programs. It does not
modify A, launch Unreal, authorize GPU work, or qualify an Episode for release.

## Candidate timing

| mechanism / slot | retained native frames | output active frames | active speed (m/s) | phase cadence | speech window / overlap |
|---|---:|---:|---:|---:|---:|
| target_moves / source1 | 0–26 | 6–32 | 0.850134 | 0.9375 Hz | 7–31 / 25 frames |
| distractor_moves / source2 | 2–17 | 21–36 | 0.850134 | 0.9375 Hz | 7–50 / 16 frames |
| both_move / source1 | 39–49 | 14–24 | 0.828940 | 0.9375 Hz | 7–31 / 11 frames |
| both_move / source2 | 62–73 | 14–25 | 0.731777 | 0.9375 Hz | 7–31 / 12 frames |

Every active output interval has exactly the retained native sample/interval
count at 15 Hz, `time_scale=1`, and no global stretch. Before the interval the
actor binds Idle and holds the first native root; after it the actor binds Idle
and holds the final root. The two action-boundary root steps are exactly zero,
so the action switch does not also teleport the actor. Live animation blending
and foot contact remain runtime questions, not CPU claims.

The 75 frames cover the complete 5-second media interval: frame timestamps are
0 through 74/15 seconds, and the final frame coverage ends at 75/15 = 5 seconds.

## Sound and projection boundary

The complete materialized `audio_program` and source activation objects are
copied without mutation. Sound content, start sample, speech frame window,
target source1 activation, and silent source2 activation therefore stay fixed.
The root timing changed, so old exact RIRs are explicitly not reusable; a fresh
exact RIR plan must bind the unchanged sound event to the candidate emitter
trajectory after accepted live ground snap.

Analytic static-camera pinhole checks keep the target/distractor sides as L/R,
L/R, and R/L for target_moves, distractor_moves, and both_move. All actor-center
depths are 2.07–3.76 m, minimum dead-zone margin is at least 0.064, and minimum
horizontal actor separation is at least 1.20 m. This is only a static analytic
projection check: it is not pixel visibility, occlusion, segmentation,
collision, or metric-depth evidence.

## Fail-closed decision

Every generated receipt has status `RELEASE_BLOCKED`, `formal_episode_count=0`,
`release_qualified=false`, and `gpu_launch_authorized=false`. Its deterministic
blockers are:

1. fresh full75 normal/target-only pixels not verified;
2. live floor identity, ground gaps, and foot/toe traces not verified;
3. canonical contact phase and planted-foot slip not verified;
4. live Walking asset and Idle/Walking skeletal transition continuity not
   verified; and
5. a fresh exact RIR for the revised emitter timing not built.

The three old globally stretched materializations remain
`reject_nonrelease_motion_realism_gate` pipeline evidence. The new candidate
documents do not mutate or upgrade them.

## CPU-only commands

```bash
python tools/qa/build_strict_two_human_native_rate_dynamic_candidates.py \
  --source-root /path/to/legacy_dynamic_materializations \
  --output-dir /new/empty/output

python tools/qa/validate_strict_two_human_native_rate_dynamic_candidates.py \
  --preflight /new/empty/output/native_strict_two_human_target_moves_native_rate_candidate_v1.json \
  --receipt /new/empty/output/native_strict_two_human_target_moves_native_rate_candidate_v1_receipt.json \
  --source-directory /path/to/legacy_dynamic_materializations/target_moves \
  --replay

python -m unittest -v \
  tests/unit/test_strict_two_human_native_rate_dynamic_candidates.py
```

The builder refuses to overwrite any output. Repeat validation for
`distractor_moves` and `both_move` with their matching source directories.
