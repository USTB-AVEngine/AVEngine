# M4 Status: Named Multi-Source RLR and Spatial WAV Outputs

Overall status: bounded M4 software/source-pose canary `pass`. The retained
native run passed all 10/10 declared formal checks and all 14/14 independently
recomputed verifier checks. This is not an asset/dataset admission result;
source presence, unit tests or an unverified WAV alone would not constitute an
M4 pass.

## Gate purpose

M4 replaces list-position and deprecated single-source assumptions with a
modern named RLR boundary. At least two stable source IDs and exactly one
formal listener are realized in one output-layout context; every
listener/source pair produces an owned IR, and AVEngine retains independent
source stems before canonical mixture assembly.

RLR remains the reused geometric acoustic propagation algorithm. The Habitat
fork owns the modern native context, endpoint identity/readback and owned IR
adapter. AVEngine owns request/identity contracts, deterministic audio
assembly, spatial-format contracts, evidence retention and independent
verification.

## Implemented scope

- Portable source/listener IDs with bytewise canonical source realization.
- At least two simultaneously registered sources and exactly one listener.
- Listener pose bound to the formal M1 camera rig.
- Native source/listener registration receipts for IDs, indices, transforms,
  radius, orientation, layout, channel count, HRTF path and realized state.
- Stable-ID source/listener setters, temporal-coherence execution and complete
  context reset/reload behavior in the fork.
- Owned all-pair IR output addressed by `(listener_id, source_id)`.
- Exact mapped full-indirect RIR equality under canonical versus reversed caller
  registration order.
- Deterministic mono dry signals, full linear convolution, independent
  per-source FOA/binaural stems and canonical full-tail mixtures.
- Direct-arrival, FOA axis/world-frame, binaural left/right, lifecycle and
  one/multi-source performance evidence.
- Atomic evidence publication followed by confined independent verification and
  tamper rejection.

## Frozen audio contracts

### Dataset-authority FOA WAV

| Field | Contract |
| --- | --- |
| Layout ID | `rlr_foa_acn_n3d_world_v1` |
| Channels | `[W, Y, Z, X]` |
| ACN indices | `[0, 1, 2, 3]` |
| Normalization | `N3D` |
| Coordinate frame | right-handed `avengine_world` |
| Axes | +X right, +Y up, +Z back, -Z forward |
| Array/WAV encoding | channel-major API; little-endian IEEE float32 interleaved WAV |
| Processing | no implicit resampling, normalization, limiter or crop |

Each source retains its own four-channel FOA stem. The canary mixture is the
canonical stable-source-ID sum of those stems; sources do not occupy separate
final mixture channels.

### Direct-listening binaural WAV

| Field | Contract |
| --- | --- |
| Rendering method | RLR native binaural listener |
| Channels | `[left, right]` |
| HRTF | explicit MIT KEMAR normal-pinna SOFA asset |
| Render rate | 16 kHz |
| HRTF input rate | 44.1 kHz |
| Rate adaptation | inside the exact runtime-locked RLR binary only |
| AVEngine resampling | forbidden/not performed |
| Processing | no implicit normalization, limiter or crop |

The HRTF asset and its license evidence are copied into the formal evidence
tree and authenticated against the dedicated M4 runtime lock. Horizontal
direct-only probes must show the correct left/right preference before the
binaural output can pass.

## Formal record

The authoritative values below were copied from the completed retained canary,
its independent verification and the final regression runs.

| Record | Formal value |
| --- | --- |
| Gate status | bounded `pass`; 10/10 declared formal checks |
| Formal run date | `2026-07-17` |
| Retained evidence root | `tmp/m4/formal_20260717_01` |
| AVEngine implementation commit | `2980ecd5fbe2b680eef2942e04b59fdb1f8842ba` |
| Habitat fork implementation commit | `5641931245a76439cc1493d87d79dc518c6f453a` |
| M4 runtime-lock SHA-256 | `18ef3b4f121ce9a904fccd1dbbacf404227e84cc65b4f1a306d7ba05b95caab7` |
| Habitat native binding SHA-256 | `395cb0e6789d2664d5d8b6abac1aa48217851cac008875718cb55cf461ed5cc7` |
| RLR native library SHA-256 | `31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4` |
| Explicit HRTF SHA-256 | `2768ac841213a7ae11d1ea7fd0f25a69b39216102dc5dd913ea6ba0f0dc57e28` |
| HRTF license-evidence SHA-256 | `f8fc1df6758230ba10c507b595ae6953a78024dbbe6ca494ee83fcd3c76ae441` |
| Native evidence path | `tmp/m4/formal_20260717_01/m4_canary_evidence.json` |
| Native evidence file SHA-256 | `99293d4f2dd6527a7b75392ce7fab157215d6d16d5ac98f39b997eba6e84140e` |
| Native evidence content SHA-256 | `925b4d445dc1ffcaad898319717957f408afb1c5ad32da3d7bdd47e7f8465025` |
| Independent verifier | `pass`; 14/14 independently recomputed checks |
| Focused AVEngine M4 tests | `85 passed in 1.08s` |
| Focused Habitat fork tests | C++: 13 cases/213 checks `pass`; Python: 21 `pass` |
| Complete AVEngine regression suite | Python 3.12 runtime: `1001 passed in 103.43s`; repository venv: `1001 passed in 93.30s` |

## Formal check summary

| Boundary | Required result | Formal result |
| --- | --- | --- |
| Request/reference/hash closure | exact, confined and independently replayed | `pass` |
| Source/listener cardinality | at least two sources; exactly one listener | `pass` |
| Listener/camera co-location | exact formal M1 transform | `pass` |
| Native endpoint receipts | every requested field and canonical index agree | `pass` |
| Pair closure | one independently readable IR per declared source | `pass` |
| Caller-order invariance | mapped full-indirect arrays byte-exact | `pass` |
| FOA format | six-cardinal ACN/N3D/world contract passes | `pass` |
| FOA listener rotation | raw world-frame array unchanged | `pass` |
| Explicit HRTF | asset/license/runtime pins agree | `pass` |
| Binaural direction | +X favors right; -X favors left by threshold | `pass` |
| Stem reconstruction | every stored stem equals dry convolved with pair IR | `pass` |
| Mixture reconstruction | retained WAV equals canonical stem sum | `pass` |
| Direct arrival | pair geometry agrees within request threshold | `pass` |
| Temporal source update | stable ID retained and moved response changes | `pass` |
| Reset boundary | reset/reload first temporal frame is byte-exact | `pass` |
| Performance | fresh one-source/multi-source measurements retained | `pass` |
| Tamper resistance | malformed/rebound artifacts fail closed | `pass` |

## Formal measurements

| Measurement | Formal value | Gate |
| --- | ---: | --- |
| Source 0 direct-arrival absolute error | `0.19400875709052912` samples | `pass` under 2-sample maximum |
| Source 1 direct-arrival absolute error | `0.8605445162613137` samples | `pass` under 2-sample maximum |
| Binaural +X left-minus-right ILD | `-8.988038352508855 dB` | `pass`; right favored beyond 6 dB |
| Binaural -X left-minus-right ILD | `8.988036343159923 dB` | `pass`; left favored beyond 6 dB |
| FOA listener-rotation maximum difference | `0.0` | exact world alignment `pass` |
| Two-source median wall time | `0.449173025 s` | measured, no hard speed gate |
| Two-source pair throughput | `4.452627136280056 pairs/s` | finite positive `pass` |
| Two-source/one-source median wall-time ratio | `1.4245201890133403` | measurement only |

## Retained artifacts

The self-contained formal bundle retains:

- request, M1/M3/identity references, selected Acoustic Scene Package, runtime
  lock, HRTF and license inputs;
- both source-order FOA IR sets and the explicit-HRTF binaural pair IRs;
- per-source dry WAV, FOA stem WAV and binaural stem WAV with authenticated
  sidecars;
- four-channel FOA and two-channel binaural full-tail canary mixtures;
- FOA cardinal/world-rotation and binaural horizontal probe arrays;
- fresh, updated and reset temporal lifecycle arrays;
- native configuration/upload/endpoint receipts and performance measurements;
- one canonical evidence JSON binding every retained artifact.

The retained formal tree is `tmp/m4/formal_20260717_01`; its evidence file and
hashes are frozen in the formal-record table above. See
[M4_EXECUTION.md](M4_EXECUTION.md) for the exact replay sequence.

## Bounded claim and M2 anchor gap

The checked-in source identity manifest retains actor, event, semantic anchor
and dry-audio routing, but its current positions are the formal M1 static source
poses. Each source's M2 dynamic-anchor evidence is explicitly `not_run` and
`qualification_claim` is false at that boundary.

Therefore, a passing M4 record means only that the fixed named software path,
static source poses, RLR pair routing, spatial formats, stems, lifecycle and
evidence verifier worked together. It does not prove that an animal's
event-time mouth/body anchor was resolved, and it does not grant animal asset,
physical material, room, episode or dataset admission. Those claims require
their own later evidence and decisions.

## M5 handoff and non-claims

M4 WAVs retain the complete convolution tail and are intentionally independent
of a final episode timeline. M4 does not claim:

- exact five-second/80,000-sample episode assembly;
- 75-frame/240,000-tick audiovisual synchronization;
- a visual-invariant vocalizing-actor counterfactual pair;
- final tail/crop policy;
- FOA support inside ordinary MP4 players;
- two-channel binaural video mux or codec/readback correctness.

M5 owns those timeline, counterfactual and video responsibilities. The
four-channel FOA authority should remain an independent WAV even after M5; the
review video should carry the separately verified two-channel binaural track.
