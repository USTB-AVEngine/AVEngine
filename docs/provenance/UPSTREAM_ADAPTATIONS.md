# Upstream Adaptations

Status: migration source map; selected third-party source has not yet been
integrated into the canonical repository.

The canonical product source repository is
[`USTB-AVEngine/AVEngine`](https://github.com/USTB-AVEngine/AVEngine).
This document records where required upstream behavior comes from without
turning another Git repository into a runtime dependency.

## Treatment vocabulary

- **adapted**: selected upstream source is retained with bounded AVEngine
  changes and its original license notice.
- **reimplemented**: AVEngine-owned code reproduces a required interface or
  behavior after studying the named source; it is not represented as copied
  upstream source.
- **external runtime/data**: AVEngine configures or calls an installation or
  dataset whose bytes are not part of this repository.

These labels describe provenance only. They do not change which backend owns
production output or grant redistribution rights for an external asset.

## Current source map

| Foundation | Upstream source | Intended bounded scope | Current migration state |
| --- | --- | --- | --- |
| Habitat-Sim | [facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim) | AVEngine-required runtime, bindings, articulated-pose opt-in and acoustic-context integration | Still executed from the manifest-pinned transition fork; no integrated-source claim yet |
| RLR Audio Propagation | [facebookresearch/rlr-audio-propagation](https://github.com/facebookresearch/rlr-audio-propagation) | AVEngine/Habitat adapter source, headers and small interface/build configuration required for FOA and binaural propagation | The pinned distribution provides headers/configuration and a precompiled shared library, not propagation-engine source; the library remains external unless source is independently obtained and reviewed |
| SPEAR | [spear-sim/spear](https://github.com/spear-sim/spear) | Selected Python client, UE plugin/control source, project configuration and build helpers required by Apartment and Kujiale | Still executed from the maintained transition checkout; no integrated-source claim yet |
| Unreal Engine | Epic-distributed installation | Editor/runtime used by the selected SPEAR integration | External runtime only; engine code, binaries, content and examples are not imported |
| MP3D, native Apartment and InteriorAgent/Kujiale | Their separately authorized dataset/project sources | Scene and room inputs for their declared production routes | External data only; no dataset or native room package is imported |

The exact revisions currently used by the transition runtime are recorded in
`manifest.yaml` and `THIRD_PARTY_NOTICES.md`. This file does not duplicate
those values or create another lock.

## AVEngine-owned adapter code already present

`src/avengine/optional_backends/` and the corresponding tools contain
AVEngine-owned planners, route validators and evidence builders. At the current
transition point they call or prepare external Habitat/SPEAR/UE execution; their
presence does not mean the upstream runtime/client/plugin source has already
been integrated.

When selected source lands, extend this document with the checked-in target
path, upstream source path and one treatment label. Use AVEngine naming and
style for AVEngine-owned interfaces rather than preserving an unrelated
upstream repository layout merely for appearance.

## Production role mapping

| Room family | Visual role | Material/acoustic and task authority |
| --- | --- | --- |
| MP3D | Habitat-Sim production visual; UE is `comparison_visual` only | RLR with SoundSpaces material authority on the same Habitat scene/state; AVEngine owns the Episode |
| `apartment_0000` | Native UE/SPEAR production visual | AVEngine owns Timeline, navigation semantics, source state, audio, Topdown, labels and admission |
| InteriorAgent/Kujiale | UE/SPEAR USD/MDL production visual over external data | AVEngine owns Timeline, navigation semantics, source state, audio, Topdown, labels and admission |
| Skokloster | Excluded | No execution or dataset counting without explicit owner reauthorization |

## License and exclusion boundary

Habitat-Sim and SPEAR-owned code use MIT terms; RLR Audio Propagation uses CC
BY-NC 4.0. SPEAR's Boost, rpclib and yaml-cpp dependencies retain their own
BSL-1.0 or MIT terms. The root AVEngine license does not replace those terms.
Applicable license text
and modification notice must accompany any selected imported source. See
`THIRD_PARTY_NOTICES.md` for the current rights summary.

The following are intentionally not migration candidates:

- Unreal Engine installations, Epic content and packaged engine binaries;
- MP3D, InteriorAgent/Kujiale, native Apartment and other room assets;
- model weights, source media, HRTF and other data assets;
- environments, caches, object files, compiled libraries and build trees; and
- generated image, audio, video and evidence output.

Historical release manifests continue to name the repositories that actually
produced them. Do not rewrite historical provenance to look like the future
single-source layout.
