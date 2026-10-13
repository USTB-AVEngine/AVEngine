# Stable Audio Open training-source top-up

Owner-authorized Stable Audio Open 1.0 sources retain source_type=generated_audio,
label_provenance=generated_stable_audio_open_1.0, Stability AI Community License
and the note "owner 2026-10-12 接受 IV(b) 风险". This records the owner decision;
it does not establish legal clearance. Existing v9/v10 inputs, original audio,
models and completed artifacts remain read-only. Use a new stage and output name
when changing the screening method.

The recorded admission builder uses its explicit generated_source_authorization
exception. Real-source collection/license policy remains unchanged. Generated
sources do not claim CC0/CC-BY or official ZIP membership. The target classifier
boundary remains BEATs AS2M cpt2 confidence >=0.1917129977 and incompatible-class
margin >=0.05, with the original subtype metadata rules, automatic activity and
normalization. Effective activity and the longest activity group must each be
>=1 second. Generated audio is excluded from real-source threshold calibration.

Every generated record requires allowed_splits=["train"] and split=train.
Both samplers filter by scene split; bus assembly also refuses generated inputs
outside an explicitly training scene. These requirements continue to apply.

## Human speech policy, owner revision 2026-10-13

Whisper transcription is recording-only. The metadata field is
whisper_transcription_recording_only, with role=recording_only and
participates_in_admission=false. Historical decisions/probabilities are retained;
letters, digits and hallucinated captions do not block admission.

The replacement score is the maximum BEATs posterior over Speech and Singing,
their ontology descendants, Whispering and Hubbub/speech babble. This includes
male/female/child speech, Conversation, Narration, and singing subclasses.
Calibration includes every real v9 recording in the requested 13 target classes
and at least 100 VCTK speech_playback records. The tool uses all such VCTK rows.
The proposed threshold is the maximum real-positive score plus 0.01. It is
usable only when positive retention is 100% and negative rejection is >=95%.
When calibration fails, threshold is null and generated records remain blocked.
Third-round generation is not started. The v11 files retain v9 plus zero qualified
generated records and explicitly mark the top-up as blocked; original records
and paired CCI paths remain unchanged. A pull list contains only the new CCI JSON.
A poor operating point must not be silently substituted for the requested one.
The emitted human_speech_screen binds the measured audio and calibration receipt.
Old generated records lacking the revised receipt need readmission under this
policy; their historical input files are not edited.

## Fingerprint repair

Arbitrary crops move the STFT frame grid and can lose exact spectral hashes;
sparse ticks can contain fewer than 12 unique landmark hashes. Use weak spectral
retrieval (at least 4 shared hashes) and compare sparse (<12 hash) queries or
references against every counterpart. Confirm an actual duplicate by identical
normalized PCM or a full shorter waveform's normalized correlation >=0.995 over
>=1 second. At 8 kHz, verify both original 16 kHz sample phases so odd-sample
crop starts do not become an unhandled half-sample shift. Correlation is gain
invariant and estimates local-window means/energies.

Gain, resampling and cropping controls validate both retrieval and alignment.
Required detection is >=99%. A failed control never becomes a per-audio rejection;
only an actual reference match blocks a candidate. The resulting receipt states
its approximate retrieval scope and measured recall. It does not establish
subjective/perceptual similarity, absence of masked sounds or a fresh scan of
STARSS/DCASE/TAU. Preserve earlier fingerprint receipts as historical evidence.

## Tools and execution

- tools/assets/generate_sao_candidates.py: offline official SAO generation,
  100 steps/cfg 7, seed and full batch conditioning, 44.1 kHz stereo followed
  by channel averaging and resampling to 16 kHz mono. Completed rows are skipped.
  Initial round uses three times the original shortage; the historical second
  round uses one further shortage, fresh prompts and the official tensor size.
- tools/qa/calibrate_sao_human_speech.py: full real-positive and VCTK calibration,
  identical thin top-up float32 BEATs 5-second chunk pooling.
- tools/qa/admit_sao_topup.py: existing target scoring/building, recording-only
  Whisper conversion, calibrated human metadata and unchanged recorded builder.
  Its historical v10 publisher remains exclusive-create and is not invoked for
  v11. Its dedup action routes to the repaired implementation.
- tools/qa/readmit_sao_topup.py: controls, repaired v9/generated deduplication,
  and readmission; pipeline runs those stages in order. Source identities and
  external input checksums are recorded with the execution.
- tools/qa/whisper_sao_screen.py: retained historical multilingual decoding and
  probability receipts; historical pass/blocked values are not current gates.
- tools/qa/report_sao_topup.py: historical v10 report, preserved without overwrite.
- tools/qa/report_sao_readmission.py: new standalone Chinese report with human
  calibration distributions, fingerprint controls, candidate results and limits.

Pass all external paths through CLI arguments. Run GPU stages serially on one
owner-authorized card after the full XML check; jobs use nice -n 10 and bounded
CPU parallelism. Artifact supervision, deadlines and progress files belong in
the external output root. Do not edit shared environments, push commits, submit
ACP/CCI jobs or mirror files to CCI as part of preparation.

Regression checks cover unchanged classifier boundaries, explicit rights,
train-only handling, recording-only multilingual transcripts, validated human
receipts, failed calibration, arbitrary crop phase, sparse fingerprints and
nonmatching waveforms with identical spectra. Full fast-unit results and real
waveform/model readbacks are separate evidence. No human listening is claimed.
