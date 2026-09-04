# QA v3 engine completion — 2026-09-04

Status: implementation in progress. User authorized this goal and the eight-step completion plan on 2026-09-04. Server worktree: `/data/jzy/tmp/wt-qa-v3-engine-completion`, branch `codex/qa-v3-engine-completion`. All implementation is in this AVEngine repository; old checkouts and user changes are retained.

## Completed source integration

Actual remote main was `e39c2b7`; remote QA was `307db37`. QA had 179 commits beyond their shared ancestor and main had 5. Local `main=d637609` was 98 commits behind remote main and was not used as the integration base.

Merge commit `9326bcd` contains both lines. The only conflict was the generated tool index, regenerated from the combined tree. Three inherited trailing blank lines were removed. Full regression: **3815 passed, 118 skipped, 52 subtests passed in 249.14 s**. Skips require retained legacy evidence or explicit native fixtures. This result establishes source integration, not media or dataset certification.

The pre-integration inventory had 49 local branches, 33 remote branches and 42 worktrees. Audio and calibration branches are QA ancestors; taxonomy is patch-equivalent to QA's `a16ebf0`. The two apparently unique static-sound commits contain the same functional files as main; patch differences come only from the generated tool index. They must not be cherry-picked again.

The original `/data/jzy/code/AVEngine-lead-a` remains on the user's dirty `cc-static-sound-sources` worktree. Its three tracked edits add HM3D train/val dataset-config selection; other untracked review material is also retained. No old branches or artifacts have been deleted.

## Active implementation

- Root: scene resource paths, scheduler input-copy failure, request/configuration and remaining QA consumers.
- Scoring worker: strict angle policy and explicit published azimuth conventions.
- Studio worker: current-source launcher/configuration and the existing HM3D split-specific configuration change.
- Asset worker: AVEngine-owned controlled-human loader and generic Unreal Editor import, selectively adapted from the old SPEAR work. The old SPEAR repository is read-only input.

Workers own disjoint files; root reviews and commits explicit paths. Do not stage every modified file indiscriminately.

## Newly reproduced scheduler failure

Before the repair, a real card8 request through `run_qa_v3_room_profile_scheduler.py` failed because the scheduler records its input scene under the run directory while `resolve_production_scene_config` refused every path outside `examples/qa/scenes`. The copied scene was otherwise valid.

Evidence: `tmp/scheduler_before_scene_path_fix/scene_profile_matrix.json` records `pipeline_error` and the rejected path. This is a directory-policy error, not an invalid floor or room. The repair allows explicit recorded scene inputs while preserving completeness, room and measured-floor checks. Resource paths are resolved relative to the declaring configuration, so recording or launching from another cwd does not switch inputs. Both example camera requests now point to this AVEngine checkout by a relative path instead of `wt-qa-v3-pilot`.

Focused regression for scene inputs, scheduler, extended profiles, floor, clearance and walkability: **110 passed**. The first post-repair run exposed a second path issue: the canonical actor-selection reader correctly rejected an unresolved repository `tmp` symlink. QA entrypoints now normalize output roots before producing those files, without weakening that reader. A real card8 request launched from `/data/jzy/tmp` then completed with **1 geometry candidate, 0 pipeline errors** at `tmp/scheduler_after_scene_and_output_path_fix/`; its recorded camera input resolves to the current AVEngine checkout. No visual/audio certification is claimed for this design-only run.

## Source boundary findings

Current Apartment capture already uses AVEngine's client and an installed `avengine_spear_ext` built from AVEngine source. The packaged UE stage records `native/spear/unreal` as its source. These are installation/build products, not external checkout calls. No need to rebuild unchanged native code merely because unrelated Python commits changed.

Known remaining work: ordinary Studio/Python source selection; controlled-human asset tools still outside AVEngine; current asset/event consumers; F2 profiles/solvers/audibility; final request/answer-form and score wiring. Historical scripts remain historical, not production entrypoints. Generic Unreal asset importing has not been shown to require the omitted SpCoreEditor/SpServicesEditor modules; include only capabilities actually needed.

## Remaining goal

Finish the above consumers, run one coherent two-room end-to-end batch, prepare the real human calibration pack, merge verified engine changes to remote main and verify the default entry from main. Then add representative HM3D/MP3D rooms, complete approved remaining card paths, run a small cross-room pilot and configurable production/evaluation. Human and model results must be reported separately from code and media completion.

No new hash locks, frozen contracts or certification gates are planned. Use ordinary configuration, identity, types and tests; preserve existing authentication, data-safety and formal-evidence protections. Run with fresh outputs and inspect devices/processes before native jobs.
