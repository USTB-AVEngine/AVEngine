# MP3D visual-to-audio identity evidence (2026-08-24)

## Scope and code identity

This server-only slice is implemented at
`d2e9448f3cabeb35582422306a96b514aae2106e` on
`codex/mp3d-audio-identity`. It changes only the current MP3D dynamic-audio
adapter and its focused unit tests. It does not change camera/source legality,
audio rendering, RLR propagation, spatial-exchange handling or dataset
admission.

The retained current-MP3D visual capture names its two array axes as
`source0/actor0/beagle Xtra Mouth` and
`source1/actor1/beagle Xtra Mouth`. The AudioProgram instead owns
`beagle_0_muzzle` and `beagle_1_muzzle`. The adapter now carries one explicit
route binding between those namespaces. It never derives that bridge from
array order, sorted IDs or numeric suffixes.

## Identity and timing evidence

Before any RLR call, the adapter now requires and cross-checks:

- the native-Habitat visual receipt's ordered `source_ids`, `actor_ids` and
  `emitter_link_names`;
- the receipt-declared `[75,2,3]` source-position NPY against every position in
  `frame_records.json`, exactly and on the same axes;
- the visual receipt's M1 path, room and source order against the listener/view
  request supplied to the audio render;
- one complete ordered unique bijection from visual source axes to AudioProgram
  candidates;
- the registered endpoint entity instance and emitter anchor for every bound
  endpoint, including rejection of two endpoint IDs that resolve to one
  physical entity anchor;
- every endpoint's exact event IDs, sound-asset IDs and resolved dry-audio
  asset IDs.

The research receipt writes those facts once per entity: `axis_index`, visual
`source_id`, `actor_id`, visual `emitter_link_name`, audio `endpoint_id`, the
registered endpoint binding, event IDs, sound/dry-asset IDs and the authority
used for each join. A visual-axis swap, an endpoint swap, a duplicate endpoint
or a missing endpoint fails as identity corruption. Two actors crossing or
exchanging spatial positions remains valid because identity stays attached to
the named axis.

The receipt also writes all exact AudioProgram event start/end samples and
ticks plus 75 adjacent frame records with their exact sample interval and
active event/endpoint IDs. The last boundaries remain 80,000 samples and
240,000 ticks. These are scheduled intervals only. No independent decoded-WAV
activity detector runs in this slice, so waveform activity stays `pending`,
the truth value is `null`, and `audible_onset_claim=false`.

Every invocation still creates a fresh/no-clobber output and calls the RIR
sequence renderer for the supplied listener trajectory. The receipt binds the
M1 request ID, room, camera rig, view, listener and visual capture, and records
that no prior listener RIR artifact was reused. A capture that declares an old
M1 listener/view path is rejected before RLR.

## Verification

The post-commit focused and related CPU log is:

`/data/jzy/tmp/mp3d_audio_identity_d2e9448_unit_r1.log`

It records `43 passed` across the current MP3D identity slice, Apartment caller
compatibility, M6 AudioProgram validation and M6 dry-audio assembly. Ruff,
Ruff formatting and `git diff --check` pass.

The read-only retained-input audit is:

`/data/jzy/tmp/mp3d_audio_identity_d2e9448_retained_capture_audit_r1.log`

It checks all six retained native Habitat captures. Each receipt/JSON/NPY trio
resolves the same two explicit endpoints for all 75 frames, each capture binds
its own M1 request ID, the registered dry asset resolves with its existing
digest, and all six event plus 75-frame timing records close at the exact final
sample/tick. The audit reports `status=pass` while retaining waveform truth as
`pending`.

## Hash and evidence boundary

No new hash algorithm, frozen contract, baseline or permanent gate is added.
The receipt applies the already-used `_input_record` mechanism to the visual
receipt and source-position NPY because they are external generated artifacts
that directly determine which physical emitter feeds each audio endpoint. A
post-render replacement could otherwise preserve every semantic filename while
silently changing the source axes; Git, version numbers, primary keys,
transactions, uniqueness, types and fixture tests do not own or protect files
under `/data/jzy/tmp`. The hashes bind only this execution receipt and do not
filter candidates or freeze a reusable dataset baseline.

This slice did not start GPU, Habitat, RLR or a new audio render. It therefore
does not certify an RIR, decoded waveform onset, audible event, viewpoint bank,
mini-pilot or Episode. Existing and future outputs remain `research_only=true`,
`episode_counted=false` and formal dataset count zero until their own native
runtime evidence exists.
