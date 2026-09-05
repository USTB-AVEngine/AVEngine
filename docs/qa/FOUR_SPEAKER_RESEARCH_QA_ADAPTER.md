# Four-speaker research QA adapter

tools/qa/adapt_four_speaker_research_qa.py consumes the existing four-speaker
inputs and reuses the current QuestionSpec/Facts evaluator.

It does not add a QuestionSpec type, alter the protocol catalog, schedule audio,
or infer positions from a plan. It checks:

- all four voice-binding WAVs are mono int16 16 kHz, non-empty, and preserve the
  complete registered transcripts;
- 150-frame SPEAR emitter/listener readbacks cover exactly the four bound actors;
- when supplied, the existing sequential audio_program.json and
  research_report.json contain four complete, non-overlapping events with
  non-zero rendered PCM evidence;
- optional pixel visibility is consulted only for the concrete four-speaker
  visibility condition of the card-13/card-14 examples.

The mapping is:

- card 13+§uçâç\the blue-shirt speaker said wha{§uçâç] -> existing
  appearance_to_spoken_content;
- card 14"éÝyø§yÜwho said this sentence wears wha{§uçâç] -> existing
  sound_to_appearance;
- speaking order -> existing who_spoke_first.

The adapter emits research_only examples and writes the existing
avengine_qa_fact_table_v1-shaped inputs for inspection. If native audio or
pixel truth is absent, the affected samples are not_run/deferred. An
out-of-view non-target actor is not a room-level failure. No formal conditional
baseline, human admission, or new frozen contract is claimed.

Example:

    PYTHONPATH=/data/jzy/tmp/wt-multi-home-qa-examples/src     /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python     tools/qa/adapt_four_speaker_research_qa.py       --frame-readbacks /path/frame_readbacks.json       --voice-binding /path/four_vctk_context_voice_binding_v1.json       --audio-program /path/audio_program.json       --research-report /path/research_report.json       --pixel-visibility-truth /path/pixel_visibility_truth.json       --output /data/datasets/avengine_workspaces/.../four_speaker_qa_v1
