# Speech-content review

Use the AVEngine tool with an environment containing the installed `openai-whisper`
package, Torch and FFmpeg. Keep model weights in a declared external data root;
the tool requires a local weight file and does not fetch models.

```json
{
  "model_path": "/path/to/whisper-model.pt",
  "device": "cuda:0",
  "decoding": {
    "language": "en",
    "task": "transcribe",
    "temperature": 0.0,
    "condition_on_previous_text": false,
    "fp16": true
  },
  "items": [
    {"id": "clip_full", "audio_path": "/path/to/mixture.wav"},
    {"id": "clip_window", "audio_path": "/path/to/mixture.wav",
     "window_seconds": [0.5, 2.5]}
  ]
}
```

Paths may be relative to the request file. Check the selected device before
launching; use a new output directory:

```sh
python tools/qa/transcribe_audio_review.py --request request.json --output fresh_asr
```

The SDK converts audio to mono/16 kHz for ASR without replacing the original
binaural file. Event windows chosen from engine truth are assisted diagnostics;
they are not blind question answering. Segment timestamps in the result are
relative to the recorded window origin. Full-clip and window results must be
reported separately. ASR errors do not by themselves prove that speech is
unintelligible, and correct text does not prove which visible person said it.

For word error rate, create scorer items with `answer_type: "transcript_wer"`,
`truth`, `model_answer`, and `question_id`, then run:

```sh
python tools/qa/score_open_answers.py --items items.json \
  --params examples/qa/transcript_scoring_v1.json --out fresh_scores.json
```

Normalization is explicit in the parameter file. The scorer reports word edit
count, WER, exact match and `max(0, 1-WER)`; it does not silently apply a paraphrase
judge. The supplied policy uses Unicode normalization, case folding, punctuation
separation and whitespace words. Select a suitable policy for the language and
report it with the result. This diagnostic is not a spatial or modality-necessity
certificate and does not replace human calibration.

## Native capture warmup

SceneCapture defaults live in
`src/avengine/backends/spear_ue/capture_defaults.json` and are included with the
installed Python package. Both current native RGB capture and the timeline pixel
probe accept `--capture-warmup-config FILE.json`; that file may override any of the
four warmup settings. The actual values and discarded-frame count are recorded in
the output. The minimum settling period prevents an early low-change plateau from
ending warmup before streamed textures arrive. It is a configurable render setting,
not a claim that every scene is visually qualified.
