# Generate and audit QA-01 through QA-25

Run from the AVEngine repository in its installed native audio environment:

```bash
python tools/qa/run_answer_prior_pipeline.py --config /path/to/run.json --output /path/to/fresh_run
# Continue the same inputs after interruption; completed jobs are not rendered again.
python tools/qa/run_answer_prior_pipeline.py --config /path/to/run.json --output /path/to/fresh_run --resume
```

`examples/qa/answer_prior_pipeline.json` is a configurable starting point. Replace its input paths; generated media and external assets stay outside Git. No agent or model service is required by this pipeline. Sound assets must already exist and carry their normal registry/provenance metadata. TTS recordings can be ordinary pool assets; automatic semantic question authoring is a separate extension.

## Inputs and stages

Provide exactly one of `bank_config` (the existing retained-bank generator config) or `bank_in` (a completed bank with full generator forms). The existing config supplies `sources_manifest`, `acceptance_policy`, `target_questions`, `items_per_type`, `seed` and optional per-scene/type limits. Optional `binding_exports` are assembled native binding exports, merged in the listed order with their full answer forms.

The optional `audio_balance` block uses retained, reviewed two-actor episodes containing `request.json`, `plan/episode_plan.json`, `capture/`, and their delivery/input references. Its source list may span supported room backends. It changes audio only, keeps the visual world identity, and checks the source's existing runtime routing and visibility constraints. The present replay entry supports exactly two registered visible actor slots, not arbitrary actor counts. Other scene generators remain available through the ordinary bank configuration.

The pool contains `sounds` with unique `sound_asset_id`, `sound_class`, absolute `path` to mono PCM, `sample_rate_hz`, `sample_count` and normal compatibility/provenance fields. Its clock must match the retained episode. Registered species, asset and sound compatibility still apply. Human nonverbal classes require the explicit per-run mapping; device sounds are not assigned to humans as a fallback. Complete recordings must fit with a three-second wet-tail reserve. Source crop provenance remains separate from local playback sample offsets.

By default the code allocates 50% single-event scenes, 25% disjoint two-event scenes, 12.5% overlapping two-event scenes and 12.5% three-event scenes. This targets both QA-01 answers, QA-05 overlap/disjoint, and QA-23 counts 1/2/3. `profile_weights` can override this mix (`single`, `sequential`, `overlap`, `three_events`). Compatible sound classes and appearances with less coverage are preferred. After native rendering, QA-01/05 branch requests and QA-23 counts are checked against realized facts; labels are never edited to fit the plan. QA-18, QA-21 and QA-22 are also generated and counted.

Use `variants` to bound audio work. `tools/qa/run_audio_answer_balance.py --config audio.json --output fresh_audio --max-new-jobs 1` supports a bounded native canary; its receipt says `partial_budget` until the plan is finished. An interrupted render's completed audio report can be reused on `--resume` only when its saved event plan matches the fresh attempt. A failed attempt remains intact; diagnose it before retrying. The per-job attempt limit defaults to two, and failures stop dependent work rather than looping indefinitely. Run one controller per output directory.

A prepared `audio_summary` can replace the `audio_balance` block when assembling a bank from an already completed audio run. Partial audio summaries are rejected by the full pipeline. This is useful for separating expensive native work from CPU-only bank export.

## Output and interpretation

`complete.json` points to the final bank. Public questions/media, private full answer forms, source provenance, per-question strata and world-level splits are generated together. `report.json` recounts the final merged bank rather than carrying the ordinary-only count. `private/answer_priors.json` reports all 25 types, answer entropy and dominance, train-fitted constant/prompt baselines, MCQ sizes/chance/position distributions, wording order and split sample support.

Existing `private/splits.jsonl` assignments are preserved. When generating a new bank that extends an old dataset, provide `split_reference` pointing to that bank. All audio/appearance variants inherit the retained world's split; novel worlds are assigned deterministically from `seed` with an 80/10/10 draw. Small runs may have empty splits, which remain explicit audit deficits. Variants never count as new independent worlds.

The audit is report-only: 60% answer dominance, four extensional MCQ choices, 24 validation questions and ten percentage points of option-position deviation. Intrinsic binary/three-band domains retain explicit chance baselines; other small domains keep usable Open questions. QA-21 uses observed sound classes rather than padding options with a global taxonomy. Under-supported types are excluded from the supported macro-average, while full averages and other prior flags remain available. No type is silently deleted.

New closed forms declare `closed_set_policy=token_negation_v2`; new set/bank reports declare `form_denominator=offered_forms`. Retained undeclared forms/sets still use the legacy scorer. Use each declared form when building model views; do not reconstruct a binding answer by falling back to option zero. The new angular policy is `threshold_graded`, while historical `continuous` remains available.

This closure fixes reusable generation/scoring and makes remaining priors visible. It does not certify that the old retained population is balanced. Visibility/motion distributions (including QA-08/09/24), late QA-19 windows and concentrated QA-25 angles can require different retained scenes or new visual production. No new visual rendering or empirical model/human admission is implied by an audio pilot or a passing unit test.
