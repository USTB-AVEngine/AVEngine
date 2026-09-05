# Four-speaker research QA adapter

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
- optional pixel visibility is consulted only for the concrete all-speaker
  condition of the card-13/card-14 examples.

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

Example:

    PYTHONPATH=/data/jzy/tmp/wt-multi-home-qa-examples/src     /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python     tools/qa/adapt_four_speaker_research_qa.py       --frame-readbacks /path/frame_readbacks.json       --voice-binding /path/four_vctk_context_voice_binding_v1.json       --audio-program /path/audio_program.json       --research-report /path/research_report.json       --pixel-visibility-truth /path/pixel_visibility_truth.json       --output /data/datasets/avengine_workspaces/.../four_speaker_qa_v1
