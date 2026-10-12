# Stable Audio Open training-source top-up

Thin appliance and pet classes may append owner-authorized Stable Audio Open 1.0
sources to a new admission input without changing an earlier input. Generated
sources are identified as generated_audio with
label_provenance=generated_stable_audio_open_1.0, Stability AI Community License
and the explicit owner note "owner 2026-10-12 接受 IV(b) 风险". This is a recorded
owner decision about output restrictions, not a legal clearance.

The existing recorded admission builder handles the explicit generated-source
exception through generated_source_authorization. Real FSD50K source policy
still requires the existing collection and license evidence. Generated sources
do not claim CC0/CC-BY or official ZIP membership. Their generation, blind
Whisper and activity receipts bind the actual source waveform. The unchanged
existing-class gate requires BEATs target score at least 0.1917129977 and margin
at least 0.05 over incompatible AudioSet heads. Existing subtype metadata rules,
activity measurement and normalization remain in force. Generated sources are
excluded from real-source threshold calibration.

allowed_splits=["train"] is required for generated audio. Both v7 samplers filter
inputs against the scene split before selection; bus assembly rejects a
generated source when the scene is not explicitly train. Development/test scenes
keep their real recording inputs.

## Tools

- tools/assets/generate_sao_candidates.py writes offline candidate plans,
  prompt snapshots, seed/batch-condition records and 44.1 kHz stereo plus
  channel-averaged 16 kHz mono WAVs. An existing recorded candidate is verified
  and skipped. Each initial class uses three times its shortage; a short class
  may use one further shortage of fresh retry prompts. The first round uses
  duration-aligned shorter tensors; the retry uses the official full
  model_config sample_size, then truncates to the requested duration. Both use
  100 steps and cfg 7.
- tools/qa/admit_sao_topup.py provides score, dedup, build and publish actions.
  BEATs scores use the thin top-up model procedure. Fingerprints compare all
  existing v9 inputs and generated candidates in both alignment directions.
  Gain, resampling and cropping controls accompany the comparisons. Its scope
  does not establish a fresh STARSS/DCASE/TAU exclusion scan.
- tools/qa/whisper_sao_screen.py performs automatic multilingual free
  transcription with Whisper-small and upstream default no-speech handling.
  It retains original greedy output, all fallback decoding attempts and
  probabilities. Any remaining Unicode letter or digit blocks admission;
  timestamp omissions do not hide a non-skipped lexical decoding. The
  no-speech skip requires probability above 0.6 and average log probability
  below -1, as in upstream Whisper. No lexical whitelist or reference text is
  used.
- tools/qa/report_sao_topup.py renders a single Chinese HTML file with embedded
  waveform/spectrogram examples and BEATs distributions from final artifacts.

Supply external output, model, encoder, checkpoint, ontology and input paths as
CLI arguments; do not store private server paths in this repository.
Run generation and GPU scoring serially on the owner-selected GPU with the
requested resource bounds. The artifact supervisor and task scripts belong in
the external output directory. Generation, scoring and Whisper stages resume
from complete JSONL records. Preserve completed prompt snapshots and stage
artifacts; use a fresh stage name when changing the candidate pool.

## Validation and limits

The final publisher checks unique IDs, every existing and new input's mono
16 kHz header, all 27 classes' distinct origins, generated-source metadata and
train/development/test filtering before writing the new v10 files. CCI paths
derive from the paired v9 prefix, and the pull list includes newly used normalized
waveforms plus the CCI input manifest. Preparing those files does not run a CCI
job or copy files to CCI.

Regression tests cover explicit owner authorization, unchanged real-recording
policy, threshold boundaries, multilingual lexical rejection, bound activity
receipts and split exclusion. Machine readback is separate from those tests.
Whisper may misdetect speech and spectral fingerprints may miss subjective
similarity or masked sounds. No human listening is claimed.
