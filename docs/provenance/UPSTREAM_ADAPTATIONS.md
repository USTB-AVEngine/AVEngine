# Upstream Adaptations

Status: Habitat H1 source-only staging has landed; runtime/build cutover and
the remaining third-party source migration are still pending.

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
| Habitat-Sim | [facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim) | AVEngine-required C++ runtime closure, bindings, articulated-pose opt-in and acoustic-context adapter source | Selected source is staged at `native/habitat/` from upstream `57ee4941dc4765240f0f91f70b2c97a919bf9038` through transition fork `e9c81c10834f7e89f33f4e0602c75535a84e054b`; current runtime remains on the manifest-pinned fork pending cutover |
| RLR Audio Propagation | [facebookresearch/rlr-audio-propagation](https://github.com/facebookresearch/rlr-audio-propagation) | AVEngine/Habitat adapter source plus a legal user-installed header/library SDK required for FOA and binaural propagation | The pinned distribution provides headers/configuration and a precompiled shared library, not propagation-engine source; the engine remains an external CC-BY-NC 4.0 SDK and is not integrated as source |
| SPEAR | [spear-sim/spear](https://github.com/spear-sim/spear) | Selected Python client, UE plugin/control source, project configuration and build helpers required by Apartment and Kujiale | S1 reimplements only one launch-settings helper at `src/avengine/backends/spear_ue/launch.py`; the SPEAR Python runtime, UE plugin/control source, project configuration and build helpers remain in the maintained transition checkout |
| Unreal Engine | Epic-distributed installation | Editor/runtime used by the selected SPEAR integration | External runtime only; engine code, binaries, content and examples are not imported |
| MP3D, native Apartment and InteriorAgent/Kujiale | Their separately authorized dataset/project sources | Scene and room inputs for their declared production routes | External data only; no dataset or native room package is imported |

The manifest and notices retain the transition runtime revisions. This H1
record identifies the checked-in source origin and does not claim a separate
runtime lock or a completed cutover.

## Habitat-Sim H1 staged source

The H1 import is **adapted** MIT-licensed source, not a repository-history
merge or a runtime dependency declaration. The compact module map is:

| AVEngine target | Original path at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | H1 treatment |
| --- | --- | --- |
| `native/habitat/esp/{assets,core,geo,gfx,gfx_batch,io,metadata,nav,physics,scene,sensor,sim}/**` | matching `src/esp/**` | adapted selected Habitat C++ closure |
| `native/habitat/esp/audio/**` | `src/esp/audio/**` | adapted AVEngine RLR context source; RLR engine source is not present |
| `native/habitat/esp/bindings/**` | `src/esp/bindings/**` | adapted bindings including the audio binding entry point; no H1 build registration |
| `native/habitat/esp/physics/bullet/BulletPhysicsManager.cpp` | matching source path | adapted opt-in `avengine_native_gltf_skin_frame` fork change |
| `native/habitat/shaders/gfx/**` | `src/shaders/gfx/**` | adapted shader source only |
| `native/habitat/config/default.physics_config.json` | `data/default.physics_config.json` | adapted small configuration |

`native/habitat/README.md` records the same source map, MIT text location,
and exclusions. H1 excludes CMake/build/package/Python wiring, vendored
dependencies, tests/docs/examples, binaries, PBR assets, and all
`RedwoodNoiseModel.{cpp,h,cu,cuh}` files. The RLR engine, headers, material
configuration, and library remain a legal user-installed external CC-BY-NC
SDK; H1 includes neither a solver source nor a bundled binary.

## SPEAR S1 launch-settings adaptation

| AVEngine target | Original path at `spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7` | Treatment |
| --- | --- | --- |
| `src/avengine/backends/spear_ue/launch.py::parallel_instance_settings` | `examples/render_in_apartment.py::parallel_instance_settings` | **reimplemented** only the port/adapter validation and collision-free per-worker setting dictionary used by the Apartment runner |

The S1 helper retains the upstream MIT attribution and text at
`LICENSES/SPEAR-MIT.txt`. It intentionally does not import the upstream
`examples/` directory. The runner still imports the external `spear` Python
runtime and still receives an external SPEAR/UE project through `--spear-root`;
this selected helper does not claim client, plugin, project-control or runtime
cutover.

## AVEngine-owned adapter code already present

`src/avengine/optional_backends/` and the corresponding tools contain
AVEngine-owned planners, route validators and evidence builders. At the current
transition point they call or prepare external Habitat/SPEAR/UE execution; their
presence does not by itself complete the native build/runtime cutover.

The H1 map above supplies the target path, original source path, treatment,
and license boundary for the staged Habitat modules. Apply the same mapping
discipline when other selected third-party source lands.

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
