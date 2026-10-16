# M5 Status — Exact Timeline and Counterfactual Episode

Status: `pass` for the bounded two-Beagle research canary described here.
`qualification_claim` remains `false`; M5 does not admit a room, asset,
episode, or dataset sample. Dataset admission remains M6 work.

## Frozen result

The retained local bundle is
`tmp/m5/formal_20260718_02/evidence.json`. It was captured from a clean
AVEngine worktree at commit
`58d0f5224d33bd7955f4b1f7201137865539f32e` and is bound to request SHA-256
`76c5106d9f85735a0f559f5689af57f8fa42298107ac7f432699754cf004cfdc`.
The canonical evidence-content hash is
`c75674338ac972ac846bb74621ef7a3bde6d51c3ae219ee4ab840e82767cd428`;
the serialized evidence file SHA-256 is
`fd3efefd8ceb0c97bf1afbebb327ad023a9354a92dd3dd0929aacfef9c2a1ec7`.

All nine declared checks passed. A separate verifier then read the retained
bytes and passed all 12 independent verification groups, including artifact
closure, exact counterfactual comparison, observation-array readback,
per-frame/per-source RIR hashes, dry/RIR/stem/mixture reconstruction, spatial
metric boundaries, confined media paths, and video packet/readback checks.

## Contract closed by M5

- Timeline v2 remains unchanged: five seconds, 48 kHz integer ticks, 15 fps,
  75 frames, 16 kHz audio, 80,000 presentation samples, and 240,000 ticks.
- Exactly one formal view, `view0`, owns co-located RGB/depth/semantic
  observations. The right-side Topdown panel is review-only QA media.
- Both named articulated actors remain visible under distinct semantic IDs.
  Their dynamic muzzle anchors drive the same trajectory for FOA and binaural
  propagation.
- Both sources are active in the same six request-owned sample windows. The
  dry clip interval, fade, gain, event windows, actor offsets, identities,
  semantic IDs, and emitter links are explicit request fields.
- Episode A and B have byte-identical visual packet payloads. The only
  semantic intervention is the declared swap of dry-audio assets between the
  two named source routes; derived request/timeline/manifest hashes may change
  as a consequence.
- Every source retains an independent four-channel FOA stem and two-channel
  binaural stem. Exact authoritative WAV mixtures are retained separately;
  the MP4 carries a two-channel AAC listening copy.
- Raw FOA remains ACN/N3D `[W, Y, Z, X]` in `avengine_world`. Binaural remains
  `[left, right]` using the hash-bound MIT KEMAR HRTF.
- Per-source ILD, frequency-dependent IPD, and ITD diagnostics are reported.
  Two GCC boundary estimates are rejected rather than counted as successful
  evidence. Mixture metrics remain diagnostic-only and cannot establish
  source-specific acceptance.
- Both formal and Topdown review videos read back as 75-frame, 15 fps,
  five-second H.264 streams with two-channel 16 kHz AAC. AAC decoding may
  expose codec padding; the presentation duration and authoritative WAV are
  exactly 80,000 samples.

## Claim boundary

This result proves a deterministic M5 software/episode canary using two
instances of the bounded Beagle research asset in the controlled Blender room.
It is not a claim that the dry calls, room materials, HRTF, or generated
episode are admitted for release. The retained bundle closes the bytes needed
to audit its result but is not a redistributable, complete upstream-runtime
package.

M5.1 is a separate research comparison extension. It adds one human and one
dog, detailed source/event/flag records, real-room reviews, and the migrated
18-second legacy Apartment route without changing immutable Timeline v2 or
retroactively broadening this M5 pass.
