# Upstream Adaptations

Status: Habitat H1/S3 staging and the H4a isolated `gfx_batch` build slice
have landed; runtime/build cutover and the remaining third-party source
migration are still pending.

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
| Habitat-Sim | [facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim) | AVEngine-required C++ runtime closure, bindings, articulated-pose opt-in, acoustic-context adapter, Python package, shader and generated-header input source | Selected source is staged at `native/habitat/` from upstream `57ee4941dc4765240f0f91f70b2c97a919bf9038` through transition fork `e9c81c10834f7e89f33f4e0602c75535a84e054b`; H4a builds only standalone `gfx_batch`, and the current runtime remains on the manifest-pinned fork pending cutover |
| RLR Audio Propagation | [facebookresearch/rlr-audio-propagation](https://github.com/facebookresearch/rlr-audio-propagation) | AVEngine/Habitat adapter source plus a legal user-installed header/library SDK required for FOA and binaural propagation | The pinned distribution provides headers/configuration and a precompiled shared library, not propagation-engine source; the engine remains an external CC-BY-NC 4.0 SDK and is not integrated as source |
| SPEAR | [spear-sim/spear](https://github.com/spear-sim/spear) | Selected Python client, UE plugin/control source, project configuration and build helpers required by Apartment and Kujiale | S1 reimplements one launch-settings helper and S2 stages only the selected python_ext source under native/spear/python_ext; the SPEAR Python runtime, extension build, UE plugin/control source, project configuration and build helpers remain in the maintained transition checkout |
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

## Habitat-Sim S3 staged Python, shader, and configuration-template source

S3 is **adapted** MIT-licensed source from the same stated transition revision.
It is source-only staging, not a package/build/runtime cutover. The exact map
is:

| AVEngine target | Original path at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | S3 treatment |
| --- | --- | --- |
| `native/habitat/python/habitat_sim/**/*.py` | `src_python/habitat_sim/**/*.py` | adapted complete 56-file tracked Python source closure held outside the active `src/` import root; no package metadata, compiled extension, or runtime-selection change |
| `native/habitat/shaders/gfx_batch/{Shaders.conf,**/*.{frag,vert,geom,glsl}}` | matching `src/shaders/gfx_batch/**` paths | adapted one shader resource declaration plus 11 tracked source shaders |
| `native/habitat/esp/{core,gfx,physics,sensor}/configure.h.cmake` | matching `src/esp/*/configure.h.cmake` paths | adapted four pure configuration-template inputs for later generated headers; no CMake build script or wiring |

The non-importable staging root preserves current transition behavior: existing
`PYTHONPATH=src` runs continue to load the installed external `habitat_sim`
package until later AVEngine-owned build/install work supplies the matching
compiled binding. The one non-Python tracked item below upstream
`src_python/habitat_sim/` is `sensors/noise_models/data/redwood-depth-dist-model.npy`;
it is data and is intentionally excluded. S3 leaves its associated Python
source unchanged and does not claim that the omitted data or the compiled
`habitat_sim_bindings` extension is available. It imports no PBR default
configuration/images, RLR header/configuration/library, dependency source,
binary, cache, build tree, test, example, or dataset.

## Habitat H4a isolated `gfx_batch` build wiring

| AVEngine target | Reference examined | Treatment |
| --- | --- | --- |
| `native/habitat/CMakeLists.txt`, `native/habitat/cmake/GfxBatchSources.cmake` | `src/esp/gfx_batch/CMakeLists.txt` at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | **reimplemented** AVEngine-owned CMake wiring for the staged `gfx_batch` source; no upstream CMake file is copied |

H4a explicitly lists only the four `gfx_batch` translation units, generates
the four staged configuration headers, and embeds the staged shader resource.
It finds installed Corrade/Magnum packages only; it adds no dependency source,
`FetchContent`, submodule, checkout path, package installation, core/PBR/RLR/
audio/Python/binding configuration, runtime selection, or cutover. CUDA is
default-off solely because the selected source excludes its helper headers; a
later CUDA integration requires explicit source selection and native validation.

## Habitat external PBR IBL adapter

| AVEngine target | Original path at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | Treatment |
| --- | --- | --- |
| `native/habitat/esp/assets/ResourceManager.{cpp,h}` | `src/esp/assets/ResourceManager.{cpp,h}` | **adapted** Habitat PBR IBL loading: replace the compiled `PbrIBlImageResources` group with user-provided external files resolved through `AVENGINE_HABITAT_PBR_ASSET_ROOT` |

This narrow source change preserves the upstream `PbrShaderAttributes`
fallback names and lighting/map flags. Only an enabled renderer IBL request
needs the external root; it resolves logical BRDF-LUT and environment-map names
below `bluts/` and `env_maps/`, while an explicit absolute path remains
user-managed. Missing roots, files, or decodable 2D images fail that enabled
render request rather than silently dropping IBL. No PBR configuration, image,
HDR, BRDF table, resource group, dependency source, build wiring, or runtime
cutover is included.

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

## SPEAR S2 python_ext source staging

| AVEngine target | Original path at spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 | Treatment |
| --- | --- | --- |
| native/spear/python_ext/cpp/{assert.h,client.h,func_signature_registry.h,msgpack_adaptors.h,msgpack_utils.h,spear_ext.cpp,std.h,types.h} | matching python_ext/cpp paths | **adapted** selected C++ extension source, with upstream MIT headers retained |
| native/spear/python_ext/python/spear_ext/__init__.py | python_ext/python/spear_ext/__init__.py | **adapted** upstream package wrapper |

S2 is source-only staging. It excludes upstream CMakeLists.txt, pyproject.toml,
checkout-relative rpclib paths, dependency source, build trees, and compiled
extension artifacts. It does not change Python imports, package installation,
extension build wiring, or the external SPEAR/UE runtime path.

## AVEngine-owned adapter code already present

`src/avengine/optional_backends/` and the corresponding tools contain
AVEngine-owned planners, route validators and evidence builders. At the current
transition point they call or prepare external Habitat/SPEAR/UE execution; their
presence does not by itself complete the native build/runtime cutover.

The H1/S3 maps above supply the target path, original source path, treatment,
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
