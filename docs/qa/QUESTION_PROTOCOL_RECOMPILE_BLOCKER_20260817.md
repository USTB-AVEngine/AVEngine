# QuestionSpec official compile blocker (2026-08-17)

Status: `pass` (2026-08-21). Resolved via allowed repair 2: the four
binding manifests were reissued against the current registry after
proving `git diff c8cf55f..HEAD` on
`examples/runtime/source_asset_runtime_profiles.json` is append-only
(149 insertions, one ordinary top-level revision bump, every
pre-existing asset row byte-identical), so each locked episode resolves
the same source1/source2 rows. Reissued manifests live beside their
originals (`native_binding_pixel_v3`, `binding_v1_registry_reissue`,
`stationary_binding_gpu1_v1_registry_reissue`,
`right_entry_binding_gpu1_v1_registry_reissue`), each with a
`REISSUE_NOTE.json`; the originals are untouched. The catalog points at
the reissued copies. Delivery:
`tmp/lead_a_question_protocol_paper_ready_v3` — compile and
`validate --require-paper-ready` both return
`candidate_case_count 2230`, `episode_count 6`, all three statuses
`pass`, and the five canary overlays are the 6e43273 RGB-underlay
renderer output. No validator, hash, or Facts file was edited.

The original blocker text is preserved below for history.

## Already done (do not redo)

Branch `cc-qa-overlay-rgb`, commit `6e43273`
`fix(qa): overlay canary masks on native RGB`.

That change only blends existing green/red canary masks onto the native
RGB frame so a reviewer can see the occluder. Owner visual review of the
five canaries passed via a probe. The probe is **not** an official
delivery. Do not reopen the overlay renderer unless a new compile proves
mask/RGB shape mismatch or frame-index error.

## Current failure

```bash
PYTHONPATH=src python tools/qa/compile_question_protocol_coverage.py compile \
  --output tmp/lead_a_question_protocol_paper_ready_v3
```

fails in `_load_native_episode` **before** overlay rendering:

```text
dynamic_corgi_british_0036.asset_registry size mismatch: declared 45316, actual 52336
```

Declared lock (written 2026-08-08, still in four binding manifests):

| field | value |
|---|---|
| path | `examples/runtime/source_asset_runtime_profiles.json` in this repo |
| size | `45316` |
| sha256 | `d13cb629d4387899980c198caac5a998dd2bf6e9cecee52821890673af9561cf` |

Those bytes are exactly commit `c8cf55f`
(`feat(runtime): register accepted generated animals`, 2026-07-29).

HEAD of the same path is `52336` /
`cbb8543ce1f6e01823cb5b8579de0089a70c69aaa45ea276f365d9be5348bdb3`.
Later in-place edits after the lock:

- `897dee7` 2026-08-11 preflight strict two-human canary
- `6c0b31f` 2026-08-11 register strict construction adult runtime
- `0f4785b` 2026-08-13 close dynamic two-human CPU planning chain
- `3f46658` 2026-08-16 bind Beagle anatomical basis (`revision` 20260811_v7 → 20260816_v8)

## Which catalog episodes fail

From `examples/qa/native_question_episode_catalog_v1.json`:

| episode_key | binding manifest | registry path | lock |
|---|---|---|---|
| `dynamic_corgi_british_0036` | `tmp/lead_a_native_dynamic_episode_v1/native_binding_pixel_v2/manifest.json` | this repo `examples/runtime/source_asset_runtime_profiles.json` | **fail** |
| `dynamic_full_occlusion_0323` | `tmp/lead_a_native_full_occlusion_reappearance_0323_v1/binding_v1/manifest.json` | same | **fail** |
| `paper_balance_stationary_first` | `tmp/lead_a_native_paper_balance_v1/stationary_binding_gpu1_v1/manifest.json` | same | **fail** |
| `paper_balance_right_entry` | `tmp/lead_a_native_paper_balance_v1/right_entry_binding_gpu1_v1/manifest.json` | same | **fail** |
| `furniture_partial_0089` | `tmp/lead_a_native_scenarios_v1/occlusion_0089_final_binding/manifest.json` | `AVEngine-habitat-native-acoustic-fix` copy, 14138 / `e58b6ede…` | pass |
| `entry_left_0323` | `tmp/lead_a_native_entry_0323_v1/binding_side_v3/manifest.json` | same acoustic-fix copy | pass |

The first failing episode stops the compiler. Fix the shared lead-a
registry lock, then re-run; do not skip remaining episodes.

## Forbidden

- Do not edit declared `size_bytes` / `sha256` in a binding manifest just
  to make the gate pass.
- Do not weaken `_verify_record` / `_load_native_episode`.
- Do not treat `tmp/cc_overlay_probe.py` or `tmp/cc_overlay_probe_out/`
  as `paper_ready` evidence.
- Do not regenerate Facts, masks, or questions to paper over the lock.
- Do not mutate `source_asset_runtime_profiles.json` in place again for
  an unrelated SPEAR/Beagle field if old manifests still point at it.

`AGENTS.md` already forbids editing a hash to pass a gate. Record an
exact repair with identity, not a patched checksum.

## Allowed repairs (pick one and prove it)

1. **Freeze the locked bytes.** Keep the four manifests pointing at
   `d13cb629…`. Put those exact `c8cf55f` bytes on an immutable evidence
   path and point the manifests there. New Beagle / two-human registry
   rows belong in a new revision file or new path.
2. **Re-issue the four binding manifests** to the current registry only
   after proving those episodes still resolve the same `source1` /
   `source2` asset IDs and that current-code QuestionSpec answers are
   unchanged. Then compile a new no-clobber directory.
3. **Split the registry.** Stop using one mutable JSON file as both the
   live SPEAR runtime table and hash-locked protocol input.

## Acceptance

```bash
PYTHONPATH=src python tools/qa/compile_question_protocol_coverage.py compile \
  --output tmp/lead_a_question_protocol_paper_ready_v3

PYTHONPATH=src python tools/qa/compile_question_protocol_coverage.py validate \
  --input tmp/lead_a_question_protocol_paper_ready_v3 --require-paper-ready
```

Required:

- `minimum_protocol_status`, `visual_canary_status`, and
  `paper_balance_status` remain `pass`
- `candidate_case_count` remains 2230
- episode count unchanged
- five canary overlays are the RGB-underlay renderer from `6e43273`,
  not the old dark-background sheets
- no validator, hash, or Facts file was edited only to clear this gate

When this passes, update this file to `pass` and point at the new
delivery directory. Do not mark it pass from the overlay probe.
