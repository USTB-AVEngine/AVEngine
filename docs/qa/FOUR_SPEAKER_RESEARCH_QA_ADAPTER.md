# Four-speaker research QA adapter

Reader-facing question organization is maintained in [the unified QA catalog](QA_UNIFIED_CATALOG_20260905.md). Card numbers below are historical mappings, not separate implemented types.

This helper consumes existing four-speaker SPEAR/audio research inputs and
reuses the current QuestionSpec/Facts evaluator. It does not add a QuestionSpec
type, alter the protocol catalog, schedule audio, or infer positions from a
plan.

It verifies:

- all four voice-binding WAVs are complete mono int16 16 kHz PCM and non-silent;
- frame readbacks cover the four emitter actors and have a self-consistent clock
  and finite positions/rotations;
- audio_program and research_report event IDs, endpoints, transcripts, dry
  windows and clip boundaries agree;
- each report output_stem is read as actual stereo int16 16 kHz PCM. RLR wet tails
  may extend beyond a dry speech window, but must remain inside the stem;
- optional pixel visibility is consulted for the selected question targets.
  The default legacy mode retains its original all-speaker condition; explicit
  `--all-speaking-targets` checks each speaker at that speaker's utterance frame
  and defers only that target's question pair when evidence is missing.

The mapping reuses existing types:

- card 13, "what did the blue-shirt speaker say" ->
  appearance_to_spoken_content;
- card 14, "what color did the speaker of this sentence wear" ->
  sound_to_appearance;
- speaking order -> who_spoke_first.

The adapter writes existing avengine_qa_fact_table_v1-shaped inputs for
inspection. It omits DOA and moving facts when the producer did not compute
them. If native audio or pixel truth is absent, affected samples are
not_run/deferred. An out-of-view non-target actor is not a room-level failure.
No formal conditional baseline, human admission, or new frozen contract is
claimed.

With `--all-speaking-targets`, four unique speaking targets produce at most nine question instances: one order question and two existing question forms per speaker. Default behavior remains the original three instances. Native masks support visibility only; shirt colors are author-controlled metadata, not pixel color-recognition evidence.

Example:

    PYTHONPATH=/data/jzy/tmp/wt-multi-home-qa-examples/src     /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python     tools/qa/adapt_four_speaker_research_qa.py       --frame-readbacks /path/frame_readbacks.json       --voice-binding /path/four_vctk_context_voice_binding_v1.json       --audio-program /path/audio_program.json       --research-report /path/research_report.json       --pixel-visibility-truth /path/pixel_visibility_truth.json       --output /data/datasets/avengine_workspaces/.../four_speaker_qa_v1
