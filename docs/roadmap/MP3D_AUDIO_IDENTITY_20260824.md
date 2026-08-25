# MP3D visual-to-audio identity evidence (2026-08-24)

## Scope and code identity

The initial server-only slice was implemented at
`d2e9448f3cabeb35582422306a96b514aae2106e`; the current independently reviewed
implementation is `bf6c45850be301e1c9389ab45dcc246e8cb4bdb6` on
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
- the visual receipt's embedded capture-time M1 identity against the supplied
  request content: request and room IDs, the complete camera rig/calibration
  and `world_from_rig`, complete listener/`rig_from_listener`, source-axis order
  and QA views; the historical input path is recorded but is not content
  authority;
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
that no prior listener RIR artifact was reused. A different path carrying the
same complete M1 content is allowed; rewriting the same path after capture is
rejected because it differs from the embedded capture-time identity.

## Verification

The post-commit focused and related CPU log is:

`/data/jzy/tmp/mp3d_audio_identity_d2e9448_unit_r1.log`

It records `43 passed` across the current MP3D identity slice, Apartment caller
compatibility, M6 AudioProgram validation and M6 dry-audio assembly. Ruff,
Ruff formatting and `git diff --check` pass.

The historical first-pass retained-input audit was:

`/data/jzy/tmp/mp3d_audio_identity_d2e9448_retained_capture_audit_r1.log`

That audit checked the older receipt/path-era boundary only. It did not and
could not prove a capture-time M1 value snapshot, the new per-axis binding
records or per-frame pose-alignment identity. It is superseded by the follow-up
below and must not be used to admit those six legacy captures.

## Independent-review correction

The branch first merged integration `60b1ede` as `63fa373`, bringing in the new
visual receipt schema. The review repair is
`bf6c45850be301e1c9389ab45dcc246e8cb4bdb6`.

The current audio adapter now additionally requires:

- exact `status=research_only`, `research_only=true` and
  `episode_counted=false` on the visual receipt;
- `actors.axis_bindings` aligned per axis with the explicit visual/audio bridge,
  while treating runtime link IDs as actor-local rather than globally unique;
- `pts_ticks == frame_index * 3200` and the same source/actor order in every
  frame's `pose_alignment`;
- a real, non-complex numeric `[75,2,3]` NPY equal to the JSON positions;
- all required capture-time M1 fields equal by value. Optional future identity
  fields are retained and do not fail the comparison;
- the supplied M1 `listener_id` passed explicitly to the RIR renderer;
- the renderer-returned `sequence.metadata.listener_id`, `source_ids` and
  native `endpoint_receipts`, plus its existing `trajectory_sha256` and
  `ir_sha256_by_keyframe_source`, copied into the receipt after checking every
  keyframe and source is covered;
- `audio_program.event_count` taken from the materialized variant rather than
  the pre-variant program.

The fresh post-commit CPU/adjacent log is:

`/data/jzy/tmp/mp3d_audio_identity_bf6c458_unit_r1.log`

It records `101 passed`, including current MP3D audio identity, Apartment
caller compatibility, M6 scheduling/assembly and the new visual/readback
schema. Ruff check/format, byte compilation and `git diff --check` pass. Tests
prove same-path M1 rewriting fails, same-content relocation passes, a custom
non-`listener0` ID reaches the RIR call and receipt, materialized event count is
used, and missing keyframe/source RIR hash metadata fails.

The six retained `509838e` captures predate `m1_capture_identity`,
`actors.axis_bindings` and per-frame `pose_alignment`. The current read-only
audit therefore verifies expected rejection rather than upgrading them:

`/data/jzy/tmp/mp3d_audio_identity_bf6c458_legacy_capture_audit_r1.log`

All six fail closed on missing `actors.axis_bindings`. A new native visual
capture with the current schema is required before a current MP3D audio/RIR run.

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

The review repair also propagates the RIR renderer's already-existing
`trajectory_sha256` and per-keyframe/source IR hashes. It neither computes a
new digest nor introduces a new policy; it verifies that the existing native
metadata covers the listener view, every acoustic keyframe and every named
endpoint before preserving it in the audio receipt.

This slice did not start GPU, Habitat, RLR or a new audio render. It therefore
does not certify an RIR, decoded waveform onset, audible event, viewpoint bank,
mini-pilot or Episode. Existing and future outputs remain `research_only=true`,
`episode_counted=false` and formal dataset count zero until their own native
runtime evidence exists.
