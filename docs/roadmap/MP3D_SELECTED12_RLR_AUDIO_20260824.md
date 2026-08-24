# MP3D selected-view RLR audio canary (2026-08-24)

## Scope

This canary joins the first fresh selected-twelve dynamic visual capture to the
current visual-axis/audio-endpoint identity adapter and the retained whole-scene
MP3D acoustic package. It renders one listener/view only. It is not a ten-region
audio bank, four-point mini-pilot, qualification run or formal Episode.

The server integration code used for the audio invocation was `b7f2fb9`. The
visual input was the current-schema capture:

`/data/jzy/tmp/mp3d_selected12_dynamic_60b1ede_r1/mp3d_camera_placement_11523`

The exact M1 request is:

`/data/jzy/tmp/mp3d_probe_subset_37317f4_r1/requests/mp3d_camera_placement_11523.json`

## Fresh native RLR result

The fresh/no-clobber output is:

`/data/jzy/tmp/mp3d_selected12_audio_b7f2fb9_11523_r1`

Its native receipt reports `status=pass`, `research_only=true`,
`episode_counted=false`, formal dataset count zero and no qualification claim.
The output contains two 80,000-sample dry buses, two 80,000x2 binaural stems and
one 80,000x2 binaural mixture at 16 kHz.

The adapter verifies one explicit identity chain for each source:

- visual `source0/actor0/beagle Xtra Mouth` to `beagle_0_muzzle`;
- visual `source1/actor1/beagle Xtra Mouth` to `beagle_1_muzzle`.

The M1 listener ID `listener0` was passed to the native RLR renderer rather than
only copied into the receipt. The returned sequence metadata contains 25 native
registration receipts and 50 existing per-keyframe/source IR identities. Every
receipt reports the same listener and ordered endpoints as natively realized;
`prior_listener_rir_reused=false`.

## Independent waveform and identity verification

The independent verifier is:

`/data/jzy/tmp/mp3d_selected12_audio_b7f2fb9_11523_r1_verification.log`

It decodes all five WAVs, checks sample rate, shapes, finite values and existing
receipt output identities, then validates the exact six AudioProgram intervals.
Every scheduled event has nonzero dry and matching binaural-stem activity inside
its half-open sample window. The first nonzero stem offsets after the scheduled
starts are 82, 115, 18, 104, 90 and 101 samples. These are waveform activity
observations, not perceptual or source-onset labels; `perceptual_onset_claim`
remains false.

The renderer used one persistent RLR context for 25 stride-3 keyframes. The
receipt preserves the renderer-returned trajectory identity, all native
endpoint registration readbacks and every existing per-keyframe/source IR
identity. It does not compute or introduce a new hash policy.

## Transport boundary

The remote native Python child completed, wrote the full receipt and WAV set,
and no remote renderer process remained. The outer SSH/tee channel did not close
after remote completion and emitted no final CLI line. After confirming the
remote child was gone and the artifact was complete, the local wrapper was
interrupted; its transport exit is therefore 255 and is explicitly recorded by
the independent verifier. The native receipt status is pass, but this run does
not claim an end-to-end transport-wrapper exit of zero.

The empty/partial wrapper log is retained at:

`/data/jzy/tmp/mp3d_selected12_audio_b7f2fb9_11523_r1.log`

## Remaining boundary

This single-view research canary does not provide region-specific actor paths,
four views per region, formal room qualification, decoded perceptual onset,
T5/off-screen questions, held-out data or a mini-pilot. Each additional listener
view still requires its own fresh RIR identity and output.
