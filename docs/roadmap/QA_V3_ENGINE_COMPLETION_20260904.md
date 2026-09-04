# QA v3 engine completion — 2026-09-04

Status: implementation in progress. User authorized this goal and the eight-step completion plan on 2026-09-04. Server worktree: `/data/jzy/tmp/wt-qa-v3-engine-completion`, branch `codex/qa-v3-engine-completion`. All implementation is in this AVEngine repository; old checkouts and user changes are retained.

## Current checkpoint — 2026-09-05

This section supersedes older “pending” statements in the chronological notes below.

- Implementation is current through b433205; it is 245 commits ahead of and 0 behind origin/main=e39c2b7 before this roadmap commit. Main has not yet been pushed. The previous reconciliation classified every non-main remote history; a final refresh is required because one absorbed remote branch was deleted upstream.
- The unified mixed pipeline completed from the current AVEngine worktree at qa_v3_pipeline_mixed_full_20260905_v3: card7 animal uses 75 frames/80,000 audio samples and card13 four-human speech uses 150 frames/160,000 samples. UE readback, RLR audio, media clocks and four released questions passed.
- F2 off-screen identity completed as a pixel-joined research candidate in both Apartment and Kujiale. Main/GateB geometry is invariant, GateA changes the audio-bound identity, and exact native masks prove the declared early/late visibility windows.
- F2 direction exposed a real modality leak: query_requires_visibility=false meant “visibility unrestricted”, so a Kujiale audio-only target remained visible. Commit e1a81a5 adds the declared any/visible/out_of_view policy and checks both main and GateA emitter geometry over the complete configured query window. It does not branch on room, profile ID or a fixed frame.
- Fresh design f2_current_native_inputs_v6 requested eight cells per F2 profile in each room. Each room produced 22/24 candidates; the two explicit shortfalls are the full-circle answer sectors wholly inside the camera cone. Every retained candidate passed main and GateA window geometry.
- Fresh native replacement f2_direction_native_20260905_v2 reran only the failed Kujiale full-circle slice. UE captured 75 frames; main and GateA each rendered 80,000 samples; execution labels are independently recorded while AudioProgram materialization remains variant A. Native target-only masks show both referenced source slots at zero target pixels in every frame 30–37. closure_report_v2.json is research_candidate_pixel_joined.
- Commit 92c5562 separates the external execution label from the internal AudioProgram variant. Old receipts remain readable but are reported as unverified; new main/GateA receipts must identify their execution variant. Commit 49772ce adds the reusable, no-clobber F2 direction pixel joiner used on the replacement slice.
- Independent review found three verifier weaknesses after the first replacement run: an empty/one-sided audio batch could appear complete, runtime evidence could be joined to a same-named fact from another run, and the 5-degree design margin was incorrectly treated as the physical camera edge. Commits 038ccb4, ba2e98b, 5010726, 83cc0ee, 1590473 and b433205 now use the physical HFOV, require explicit emitter paths and complete audio pairs, bind visual/audio/pixel artifacts to the exact facts, and reject old incomplete verification summaries. The strengthened verifier and joiner pass on the retained v2 native outputs.
- MP3D has real native visual, dynamic RLR and mux evidence. HM3D 00800 has a direct current-source E2E receipt with visual, FOA, binaural and machine audition, but remains research_only: acoustic geometry, ray leakage, physical material truth and full placement qualification are not complete.
- Current-source animal post-processing has a CPU/Blender technical closure without fixed rig counts. The retained Pixal3D candidate remains too fragmented for runtime registration. Controlled humans, static speakers and speech metadata are inside AVEngine; the representative runtime coverage matrix still needs final expansion.
- Model discovery is complete. v43 cannot run because its trained checkpoint/cache is absent and it assumes 5-second/75-frame binaural input. Spatial-Omni weights exist, but current AVEngine mixed/F2 outputs need real FOA and an AVEngine-owned media adapter; a true V-only path also needs explicit null-audio handling. No external model checkout is a production AVEngine entrypoint.
- The live Studio source boundary is now repaired. The old PID 198415 from /data/jzy/code/AVEngine-lead-a was stopped only after proving the queue had no active tasks. PID 892496 now serves the same 379 retained tasks from this worktree, and /api/health reports this repository root. The HM3D fleet crontab also uses this worktree and the declared Habitat Python; a current-source dry-run submitted nothing and preserved all task states. The cutover receipt is studio_source_cutover_20260905_v1.
- Maintained HM3D download and FOA-axis tools no longer change directory to or default a scene inside the SoundSpaces Git checkout. External installed runtimes, packaged UE stages, SDKs, datasets and model weights remain declared dependencies/data rather than project-code entrypoints. Historical f15 reproduction scripts retain explicit old-source inputs and are not reachable from Studio templates.

Current ordered remainder:

1. From a clean final integration revision, rerun the full regression and a minimal representative pre-main batch: Apartment card7 at 75 frames plus card13 at 150 frames, and at least one standard Kujiale profile through UE, RLR, verification and question assembly.
2. Refresh branch reconciliation at that exact integration HEAD, then merge and push the verified integration to main.
3. Rerun the representative mixed 75/150-frame E2E from the exact main revision and retain complete provenance.
4. Expand current-main coverage for static playback, cat, four-human card13/card14, mixed human/animal, card12’s four declared sound classes and the remaining standard profiles.
5. Close or explicitly retain the HM3D/Pixal3D research boundaries, then implement AVEngine-owned A-only/V-only/AV adapters with real FOA and aligned clocks.
6. Run real model pilots and human calibration; keep code, media, model and human evidence as separate claims.


## Completed source integration

Actual remote main was `e39c2b7`; remote QA was `307db37`. QA had 179 commits beyond their shared ancestor and main had 5. Local `main=d637609` was 98 commits behind remote main and was not used as the integration base.

Merge commit `9326bcd` contains both lines. The only conflict was the generated tool index, regenerated from the combined tree. Three inherited trailing blank lines were removed. Full regression: **3815 passed, 118 skipped, 52 subtests passed in 249.14 s**. Skips require retained legacy evidence or explicit native fixtures. This result establishes source integration, not media or dataset certification.

The pre-integration inventory had 49 local branches, 33 remote branches and 42 worktrees. Audio and calibration branches are QA ancestors; taxonomy is patch-equivalent to QA's `a16ebf0`. The two apparently unique static-sound commits contain the same functional files as main; patch differences come only from the generated tool index. They must not be cherry-picked again.

The original `/data/jzy/code/AVEngine-lead-a` remains on the user's dirty `cc-static-sound-sources` worktree. Its three tracked edits add HM3D train/val dataset-config selection; other untracked review material is also retained. No old branches or artifacts have been deleted.

## Current implementation and verification

Committed after the merge:

- `3c43d6a`: scene/resource path resolution and recorded scheduler inputs.
- `50c1ea3`: Studio selects its configured AVEngine source before imports; HM3D train/val dataset selection incorporates the three original tracked edits while preserving the original worktree. Focused Studio/source-launch checks passed; no running Studio service was restarted.
- `095eb27`: catalog-driven controlled-human source loader, four-color catalog and generic UE importer in AVEngine. Adapted source provenance and SPEAR MIT attribution are recorded; asset GLBs remain declared external data. Python contains no four-color or asset-ID whitelist.
- `b5f3c3c`: explicit speaker, utterance, transcript and split metadata now survives raw, prepared, event pool, runtime PoolClip and sound registry. A fresh 100-utterance train-only chain was completed and checked.
- `38ff9ec`: rigid runtime bindings, controlled-human/static UE asset tools, explicit content mounts, configurable native capture clock and measured emitter children. This commit has unit and Editor import/reload evidence; actual packaged capture remains pending.
- `566d5d5`: the first real UE import exposed a literal backslash-n at the end of its manifest. Fixed JSON serialization and added a round-trip/no-replace regression.

QA consumers committed as `3d155a6` remain under independent review: request budgets/answer forms, scene-derived angle bands, strict versus legacy angle credit, whole query-window separation, and published interval scoring. Latest complete top-level QA regression: **679 passed in 57.83 s**, log `tmp/qa_consolidated_20260904_v1.log`. This is separate from the earlier complete merge regression and does not cover later static runtime work.

Actual design-only runs:

- `tmp/request_plan_300_two_rooms`: each room plans 150 candidates with two answer forms, totaling 300 questions per room.
- `tmp/strict_domain_card1_budget12`: requested three forward and three backward candidates; produced three forward and one backward candidate, with zero pipeline errors and two search shortfalls. Published facts carry strict full-credit policy and a 15-degree credit radius. No media or certification is claimed.

Controlled-human runtime work:

- Fresh data: `/data/avengine_external/assets/controlled_humans_20260904_v1`, four retained normalized GLBs with compact, truthful source descriptions.
- Fresh UE stage: `/data/avengine_external/ue-package-stages/qa_v3_controlled_humans_20260904_v1`; assembled from AVEngine native source and declared package content. Reused Editor binaries were built from the same unchanged native source; no external checkout supplies project code.
- `tmp/ue_controlled_yellow_import_v1`: preserved first import and malformed JSON evidence.
- `tmp/ue_controlled_yellow_import_v2`: rerun through repaired AVEngine importer exited 0 and produced readable JSON, a saved skeletal mesh, Blueprint, 80 bones, three materials, seven textures and Idle/Walking assets. Independent fresh-process reload exited 0 and emitted `CONTROLLED_HUMAN_IMPORT_VERIFY_OK` at `tmp/ue_controlled_yellow_reload_v2`; `verification_result.json` points to that actual log. Package/capture and runtime-registry integration remain outstanding.

Parallel ownership: root handles UE execution, actor selection, shared runtime spawn/audio readback and QA integration; one worker owns static registry/schema and current Apartment rigid renderer; another owns speech metadata through event preparation/registration; a third independently reviews the pending QA changes. Do not stage all dirty files together.

## Newly reproduced scheduler failure

Before the repair, a real card8 request through `run_qa_v3_room_profile_scheduler.py` failed because the scheduler records its input scene under the run directory while `resolve_production_scene_config` refused every path outside `examples/qa/scenes`. The copied scene was otherwise valid.

Evidence: `tmp/scheduler_before_scene_path_fix/scene_profile_matrix.json` records `pipeline_error` and the rejected path. This is a directory-policy error, not an invalid floor or room. The repair allows explicit recorded scene inputs while preserving completeness, room and measured-floor checks. Resource paths are resolved relative to the declaring configuration, so recording or launching from another cwd does not switch inputs. Both example camera requests now point to this AVEngine checkout by a relative path instead of `wt-qa-v3-pilot`.

Focused regression for scene inputs, scheduler, extended profiles, floor, clearance and walkability: **110 passed**. The first post-repair run exposed a second path issue: the canonical actor-selection reader correctly rejected an unresolved repository `tmp` symlink. QA entrypoints now normalize output roots before producing those files, without weakening that reader. A real card8 request launched from `/data/jzy/tmp` then completed with **1 geometry candidate, 0 pipeline errors** at `tmp/scheduler_after_scene_and_output_path_fix/`; its recorded camera input resolves to the current AVEngine checkout. No visual/audio certification is claimed for this design-only run.

## Source boundary findings

Current Apartment capture already uses AVEngine's client and an installed `avengine_spear_ext` built from AVEngine source. The packaged UE stage records `native/spear/unreal` as its source. These are installation/build products, not external checkout calls. No need to rebuild unchanged native code merely because unrelated Python commits changed.

Known remaining work: final default-environment source cutover; current asset/event consumers and complete speech scheduling; static mesh runtime integration; F2 profiles/solvers/audibility; review and final commit of request/answer-form and score consumers. Optional active-looking generated-asset launch scripts still reference old source checkouts and require separate closure before the goal is complete. Historical scripts remain historical, not production entrypoints. Generic Unreal asset importing has not been shown to require the omitted SpCoreEditor/SpServicesEditor modules; include only capabilities actually needed.

## Data discrepancies checked against retained assets

The 600 VCTK utterances are **400 train and 200 eval**, not all eval. The raw clip sidecars are the authority; prepared/event pool records currently lose speaker/transcript/split metadata. Four existing, distinct train speakers have complete utterances totaling about 7.62 seconds, so three 0.3-second gaps fit within ten seconds without clipping speech.

Twelve static playback objects exist as data, but the top-level asset index calls them formal while their individual asset records still call them research. No blanket formal-admission claim is made. They are not yet in the Apartment runtime registry or UE stage. Existing code assumes skeletal actors and replaces emitter height instead of transforming a complete local offset; this is being repaired through explicit rigid bindings and actual child-component world readback.

## Latest actual request and asset evidence

`tmp/request_open_only_actual_v1` requested two Open questions through the actual scheduler from another cwd, using this goal's content stage. Both geometry candidates were generated without shortfall or pipeline errors; the batch's `questions.jsonl` contains exactly two `form=open` rows. Internal fact records retain both gold views for counterfactual audits. `tmp/request_open_only_pilot_v1` then used the actual assembler with its inherited quota and reported two questions, plus two separately counted GateA views. No synthetic media were created to make this pass.

Both bookshelf speakers imported at `tmp/ue_static_speakers_import_v2` and reloaded in a separate Editor process at `tmp/ue_static_speakers_reload_v2`. Native mesh bounds match source orientation and 33 cm height. These are research runtime records; no change was made to the inconsistent formal labels in the shared source index. The current runtime registry has the existing articulated records, the yellow human, and these two rigid objects. Rigid identity uses object type/category, and multiple named emitter anchors are allowed with one explicitly selected default.

`tmp/ue_qa_assets_dependency_export_v1` contains a real Editor export (512 packages, 1372 dependency edges). `tmp/qa_assets_closure_v1/closure_report.json` maps the selected registry assets, Apartment map and camera to 496 physical content packages. The new maintained tools operate on explicit content mounts rather than requiring the old SPEAR checkout directory layout.

Packaging attempts are retained under the fresh stage's `packaging_runs/`. The first failed before cook due to missing Editor target receipt; the matching receipt was added from the same unchanged native build. The second UAT process exited zero but did not include the new assets because `CookDir` was supplied as package names instead of physical directories. It is an invalid result for this goal. A third fresh packaging run uses physical content directories and must pass actual pak-content inspection before any capture uses it.

The complete-utterance scheduler now consumes actual clip durations and preserves voice identities. A finite matching algorithm replaces greedy selection where speaker/utterance conflicts require reassignment. A 100-entry train pool produced four complete, distinct utterances within ten seconds (8.290125 seconds including gaps); this is scheduling evidence, not rendered audio/video. The four voices must be independently bound to colors/slots per episode, not fixed by selection order.

A further source-position issue was confirmed: Apartment floor is 27.11 cm, while existing dog emitter offsets include 40-45 cm forward displacement and 58-65 cm local height. The old fallback discarded the forward offset and treated local height as absolute world height. New articulated captures now create the same measured emitter children as rigid captures; audio will consume their actual world poses. Legacy captures remain readable. Explicit canonical-height research policy must remain explicit and may not be silently ignored.

## Probe resource incident and recovery

Two temporary VCTK selection probes from this task accumulated all four-speaker/utterance combinations (PID 4143088, about 195 GiB RSS) and a very large sentence-mask dynamic-programming state table (PID 4143923, about 8.3 GiB). These were exploratory scripts, not an AVEngine FLAC-decoding failure. Both were identified through their parent commands and this worktree's PYTHONPATH, their source was saved under `tmp/vctk_probe_memory_diagnosis`, and they were terminated by exact PID. Both exited; MemAvailable recovered from below 1 GiB to 203 GiB. Other users' training and inference jobs were preserved.

The interrupted static-speaker import at `tmp/ue_static_speakers_import_v1` is not a success. Its log, launch parameters and intervention records are retained. Do not start another Editor against this stage until its exact former process has exited. Further speech work uses the already selected four complete train utterances and does not enumerate all possible combinations. Future native preflight checks include available RAM and tracked task-owned probe processes as well as GPU occupancy.

## Remaining goal

The authoritative ordered remainder is listed in the current checkpoint above. Main integration, representative current-main reruns, AVEngine-owned A/V/AV experiments and real human calibration are still required before the goal is complete.

No new hash locks, frozen contracts or certification gates are planned. Use ordinary configuration, identity, types and tests; preserve existing authentication, data-safety and formal-evidence protections. Run with fresh outputs and inspect devices/processes before native jobs.
