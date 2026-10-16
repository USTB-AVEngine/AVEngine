# Strict two-human motion-realism gate staging

This directory is an independent CPU-only handoff. It does not modify the A
repository and does not launch Unreal, SPEAR, RIR work, or any GPU process.

## Outcome

The current dynamic canaries are useful pipeline/visibility/acoustic evidence,
but they are not motion-realism release evidence. Their retained native anchor
windows were globally resampled across all 75 frames (4.933333 seconds):

| canary / slot | native frames | path (m) | old speed (m/s) | native-rate speed (m/s) | stretch | old/native cadence |
|---|---:|---:|---:|---:|---:|---:|
| target_moves / source1 | 0..26 | 1.473566 | 0.298696 | 0.850134 | 2.846x | 0.329392 / 0.9375 Hz |
| distractor_moves / source2 | 2..17 | 0.850134 | 0.172325 | 0.850134 | 4.933x | 0.190034 / 0.9375 Hz |
| both_move / source1 | 39..49 | 0.552627 | 0.112019 | 0.828940 | 7.400x | 0.126689 / 0.9375 Hz |
| both_move / source2 | 62..73 | 0.536636 | 0.108778 | 0.731777 | 6.727x | 0.139358 / 0.9375 Hz |

The first deterministic blocker for every legacy moving slot is
`missing_motion_realism_profile`. The receipts also record the measured global
stretch, speed, cadence, and missing live foot-plant evidence as additional
blockers. Existing visibility/acoustic conclusions are not recomputed or
invalidated; the release classification is narrowed to
`nonrelease_pipeline_evidence_only`.

The native-rate candidate revision now produces three full75 CPU preflights.
Their moving intervals are target source1 f6–f32, distractor source2 f21–f36,
and both-move source1/source2 f14–f24/f14–f25. Active speeds are
0.731777–0.850134 m/s at 0.9375 Hz; frames outside each interval are Idle and
hold the nearest boundary root. All six candidate/receipt artifacts remain
`RELEASE_BLOCKED` pending fresh pixels, live ground/foot evidence, live Walking
readback, and a fresh exact RIR. See
`docs/qa/strict_two_human_native_rate_dynamic_candidates.md`.

## Release contract

A moving slot passes only when all of the following are present and consistent:

1. One explicit output active interval with exactly the same interval count and
   15 Hz rate as its retained native source frame range.
2. `time_scale=1`, `global_time_stretch_applied=false`, Walking only inside the
   active interval, and Idle with a held boundary root outside it.
3. Active root speed within 5% of the speed derived from the retained native
   path length and exact native interval duration.
4. Live Walking play length and play rate 1, 48,000 timeline ticks/second,
   51,200 ticks/cycle, phase/tick agreement, and cadence within 2% of both the
   live clip cadence and retained native phase cadence.
5. A reviewed canonical Walking contact-phase authority plus live foot/toe
   floor traces for every active frame. Phase error is limited to half of one
   15 Hz frame; planted-foot slip is capped at 0.02 m/frame. Clearance and
   penetration remain governed by the separate accepted strict ground-contact
   release profile.

These are gate thresholds, not population-level claims about human walking.
Speed and cadence are candidate/asset-specific and derive from retained native
motion plus the live Walking asset. The 5%/2% tolerances are hard maximum error
budgets. Foot contact cannot be inferred from actor bounds or pixels alone.

## Files and exact copy map

| staging source | intended A repository destination |
|---|---|
| `README.md` | `docs/qa/strict_two_human_motion_realism_audit_handoff.md` |
| `tools/qa/build_strict_two_human_motion_realism_receipt.py` | `tools/qa/build_strict_two_human_motion_realism_receipt.py` |
| `tools/qa/validate_strict_two_human_motion_realism_receipt.py` | `tools/qa/validate_strict_two_human_motion_realism_receipt.py` |
| `tests/unit/test_strict_two_human_motion_realism_receipt.py` | `tests/unit/test_strict_two_human_motion_realism_receipt.py` |
| `docs/qa/strict_two_human_motion_realism_gate.md` | `docs/qa/strict_two_human_motion_realism_gate.md` |
| `examples/qa/native_strict_two_human_target_moves_v2_motion_realism_reject_v1.json` | same path |
| `examples/qa/native_strict_two_human_distractor_moves_v2_motion_realism_reject_v1.json` | same path |
| `examples/qa/native_strict_two_human_both_move_v1_motion_realism_reject_v1.json` | same path |
| `tools/qa/build_strict_two_human_native_rate_dynamic_candidates.py` | same path |
| `tools/qa/validate_strict_two_human_native_rate_dynamic_candidates.py` | same path |
| `tests/unit/test_strict_two_human_native_rate_dynamic_candidates.py` | same path |
| `docs/qa/strict_two_human_native_rate_dynamic_candidates.md` | same path |
| `examples/qa/native_strict_two_human_target_moves_native_rate_candidate_v1.json` | same path |
| `examples/qa/native_strict_two_human_target_moves_native_rate_candidate_v1_receipt.json` | same path |
| `examples/qa/native_strict_two_human_distractor_moves_native_rate_candidate_v1.json` | same path |
| `examples/qa/native_strict_two_human_distractor_moves_native_rate_candidate_v1_receipt.json` | same path |
| `examples/qa/native_strict_two_human_both_move_native_rate_candidate_v1.json` | same path |
| `examples/qa/native_strict_two_human_both_move_native_rate_candidate_v1_receipt.json` | same path |

`.source_inputs/` is audit scratch input and must not be copied.

## Runtime compatibility

The handoff requires Python 3.10 or newer because it uses the standard-library
`itertools.pairwise` API. The target A runtime is Python 3.11. Local CPU tests
were executed under Python 3.13; Python 3.9 is intentionally unsupported.

## CPU-only commands

```bash
python -m py_compile \
  tools/qa/build_strict_two_human_motion_realism_receipt.py \
  tools/qa/validate_strict_two_human_motion_realism_receipt.py

pytest -q tests/unit/test_strict_two_human_motion_realism_receipt.py

python tools/qa/build_strict_two_human_motion_realism_receipt.py \
  --materialization-root /path/to/materialized_canary \
  --output /new/path/motion_realism_receipt.json

python tools/qa/validate_strict_two_human_motion_realism_receipt.py \
  --receipt /new/path/motion_realism_receipt.json \
  --materialization-root /path/to/materialized_canary \
  --expect-status reject_nonrelease_motion_realism_gate

python tools/qa/build_strict_two_human_native_rate_dynamic_candidates.py \
  --source-root /path/to/legacy_dynamic_materializations \
  --output-dir /new/empty/output

python -m unittest -v \
  tests/unit/test_strict_two_human_motion_realism_receipt.py \
  tests/unit/test_strict_two_human_native_rate_dynamic_candidates.py
```

Use `--require-release-pass` only in the eventual release pipeline; it exits
nonzero for these legacy receipts.
