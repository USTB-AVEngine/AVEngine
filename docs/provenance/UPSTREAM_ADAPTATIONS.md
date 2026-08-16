# Upstream Adaptations

Status: Habitat H1/S3 staging and H4a/H4c/H4d/H5a build slices, plus the
source-only SPEAR UE closure, have landed; runtime/build cutover and remaining
third-party source migration are still pending.

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
| Habitat-Sim | [facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim) | AVEngine-required C++ runtime closure, bindings, articulated-pose opt-in, acoustic-context adapter, Python package, shader and generated-header input source | Selected source is staged at `native/habitat/` from upstream `57ee4941dc4765240f0f91f70b2c97a919bf9038` through transition fork `e9c81c10834f7e89f33f4e0602c75535a84e054b`; H4a builds standalone `gfx_batch`, H4c a non-binding core static slice, H4d its static importer consumer interface, and H5a an optional staged Python extension, while the current runtime remains on the manifest-pinned fork pending cutover |
| RLR Audio Propagation | [facebookresearch/rlr-audio-propagation](https://github.com/facebookresearch/rlr-audio-propagation) | AVEngine/Habitat adapter source plus a legal user-installed header/library SDK required for FOA and binaural propagation | The pinned distribution provides headers/configuration and a precompiled shared library, not propagation-engine source; the engine remains an external CC-BY-NC 4.0 SDK and is not integrated as source |
| SPEAR | [spear-sim/spear](https://github.com/spear-sim/spear) | Selected host/game client and optional native module source, plus the source-only UE plugin/control and project-configuration closure required by Apartment and Kujiale | S1 reimplements one launch-settings helper, S2 stages selected extension source, S3a stages the namespaced host/game client, and S3b adds an AVEngine-owned optional native build through external rpclib/Python/nanobind dependencies. The selected UE source/configuration/build-rule closure now lives at `native/spear/unreal/` with an explicit-SDK rule and narrow Editor-game bridge. It excludes UE, Content/assets, generated output and binaries; current runners still require their external runtime/project assembly pending a separate runtime cutover. |
| Eastforward SPEAR fork helper slice | [Eastforward/spear](https://github.com/Eastforward/spear) behavior origins `0a9ba3ded8ffa07a3bc3684279845da22dc123e0`, `c8ba04076a32060e35020deb8f706c4b13951cae`, `ff6e44736f68c72ce4140152e2dadb4b58dc0b28`, and `a5168b8c357afa494f6200dedb03b93c3a59be57`; local MIT transition snapshot `251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7` (SPEAR-lead-b) carries the selected bytes | Three rig-query and two lighting RPC helpers written in the maintained transition fork | S3d adapts only those helper closures into `avengine.backends.spear_ue`; the local carrier is not described as a public fork ref, and neither helper path is attributed to `spear-sim/spear` |
| Unreal Engine | Epic-distributed installation | Editor/runtime used by the selected SPEAR integration | External runtime only; engine code, binaries, content and examples are not imported |
| MP3D, native Apartment and InteriorAgent/Kujiale | Their separately authorized dataset/project sources | Scene and room inputs for their declared production routes | External data only; no dataset or native room package is imported |

The manifest and notices retain the transition runtime revisions. This H1
record identifies the checked-in source origin and does not claim a separate
runtime lock or a completed cutover.

## SPEAR UE source-only closure

`native/spear/unreal/` is adapted source, not a repository-history merge or a
runtime/package claim. `native/spear/unreal/PROVENANCE.md` records the public
`spear-sim/spear@2d4575587a9be39a40ba89c4259836a85ccd3f3f` baseline, the
transition carrier from which the bytes were selected, every retained module,
and its bounded AVEngine changes. It keeps only the game-side runtime modules,
the limited Editor-game launch bridge required by the existing `UnrealEditor
-game` route, UE project/configuration inputs, and a Build.cs rule that accepts
only explicit user-installed Boost/rpclib/yaml-cpp prefixes.

The closure contains no UE installation, UE Content, room data, generated
files, compiled libraries or packaged runtime. It does not change a runner or
retire an external SPEAR/UE runtime assembly. Most selected code retains the
SPEAR/Intel MIT notice; `Assert.{h,cpp}` additionally retain the PPK_ASSERT
WTFPLv2 notice and `SuppressCompilerWarnings.h` the Microsoft MIT notice. See
the leaf provenance record, `THIRD_PARTY_NOTICES.md`,
`LICENSES/PPK_ASSERT-WTFPLv2.txt`, and `LICENSES/Microsoft-MIT.txt`.

## Habitat-Sim H1 staged source

The H1 import is **adapted** MIT-licensed source, not a repository-history
merge or a runtime dependency declaration. The compact module map is:

| AVEngine target | Original path at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | H1 treatment |
| --- | --- | --- |
| `native/habitat/esp/{assets,core,geo,gfx,gfx_batch,io,metadata,nav,physics,scene,sensor,sim}/**` | matching `src/esp/**` | adapted selected Habitat C++ closure |
| `native/habitat/esp/audio/**` | `src/esp/audio/**` | adapted AVEngine RLR context source; RLR engine source is not present |
| `native/habitat/esp/bindings/**` | `src/esp/bindings/**` | adapted bindings including the audio binding entry point; only H5a's explicit opt-in registers its selected binding translation units |
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

## Habitat H4c non-binding core static wiring

| AVEngine target | Reference examined | Treatment |
| --- | --- | --- |
| `native/habitat/CMakeLists.txt`, `native/habitat/cmake/CoreSources.cmake` | `src/esp/CMakeLists.txt` at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | **reimplemented** AVEngine-owned CMake wiring and an explicit 112-C++ source selection for the staged non-binding core; no upstream CMake file is copied |
| `native/habitat/esp/metadata/URDFParser.cpp` | matching `src/esp/metadata/URDFParser.cpp` at the same revision | **adapted** one include-path normalization from the transition vendored layout to the installed tinyxml2 public header |

H4c makes `AVEngine::HabitatCore` a static target over the selected `core`,
`geo`, `gfx`, `assets`, `metadata`, `io`, `scene`, non-Bullet `physics`,
`nav`, `sensor`, and `sim` closure. It depends on the H4a
`AVEngine::HabitatGfxBatch` target and embeds the staged `GfxShaderResources`
declaration only.

It resolves Corrade/Magnum, RapidJSON, tinyxml2, and RecastNavigation as
external CMake packages; it adds no dependency source, `FetchContent`,
submodule, checkout path, package installation, or vendored dependency bytes.
A fresh H10 CPU-only validation against temporary external H5/H6/H9 prefixes
completed all 122 Ninja steps for `avengine_habitat_core`. That result is static
compilation evidence only, not a selected build/runtime path or a cutover.

H4c intentionally excludes `BackgroundRenderer`, the RLR/audio adapter source,
Bullet source, bindings, CUDA noise source, and all PBR image resources. It
does not recreate `PbrIBlImageResources` or add PBR assets/default
configuration; the separate external PBR IBL adapter below remains the
renderer asset-resolution behavior.

## Habitat external RLR SDK adapter wiring

The optional AVEngine-owned `AVEngine::HabitatRlrAudio` static target selects
only the staged `esp/audio/RLRAcousticContext.cpp` adapter. It resolves the
legal user-installed RLR distribution strictly through
`AVENGINE_RLR_SDK_ROOT`, whose documented official package layout contains
`headers/RLRAudioPropagation.h` and
`libs/linux/x64/libRLRAudioPropagation.so`. The CMake imported target exposes
those external include and library paths without a checkout search,
`FetchContent`, submodule, copied binary, install rule, or RPATH. The adapter
source remains adapted MIT code; the RLR engine, header, shared library,
material configuration, and solver data remain external CC-BY-NC 4.0 SDK
inputs. The adapter option alone leaves `ESP_BUILD_WITH_AUDIO` disabled.

A separate default-off `AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR` option
regenerates the staged sensor configure header and links
`AVEngine::HabitatCore` publicly to the same imported SDK. It preserves the
selected legacy `AudioSensor` source through the SDK deprecated C++ wrapper;
its public C++ types require the transitive include/link interface. It adds no
bindings, package installation, runtime resolver, copied library, scene
propagation claim, or runtime cutover.

## Habitat H4d static importer consumer wiring

H4d is reimplemented AVEngine-owned CMake usage-interface wiring. It resolves
only the installed Magnum AnySceneImporter, AnyImageImporter, and
AnyImageConverter targets plus the installed MagnumPlugins GltfImporter,
PrimitiveImporter, StanfordImporter, StbImageImporter, and StbImageConverter
targets. For a static installation their exported INTERFACE_SOURCES register
each plugin in a final executable or binding that links AVEngine::HabitatCore.
H4d writes no manual plugin-import macro and adds no dependency source, archive,
checkout path, runtime source path, plugin data, PBR asset, or cutover claim.
A fresh H14 CPU-only 132-step validation against temporary H11/H6/H9 prefixes
linked only AVEngine::HabitatCore, compiled all eight registrations in the final
consumer, and decoded the MP3D GLB, semantic PLY, PBR PNG, and PBR HDR inputs.
That is static build and decoder evidence only, not a runtime or build cutover.

## Habitat H5a optional M1 Python binding wiring

| AVEngine target | Reference examined | Treatment |
| --- | --- | --- |
| `native/habitat/CMakeLists.txt`, `native/habitat/cmake/PythonBindingSources.cmake` | transition `src/esp/bindings/CMakeLists.txt` and the selected `src/esp/bindings/*.cpp` files at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` | **reimplemented** AVEngine-owned Python 3.12 module wiring, optional explicit-prefix facade/binding/config installation, and an explicit selection of 17 already adapted binding translation units; no upstream CMake file is copied |

H5a makes the default-off `AVEngine::HabitatPythonBindings` module target over
exactly `Bindings`, `AudioPropagationBindings`, `AttributesBindings`,
`AttributesManagersBindings`, `ConfigBindings`, `CoreBindings`, `GeoBindings`,
`GfxBindings`, `MetadataMediatorBindings`, `GfxReplayBindings`,
`PhysicsBindings`, `PhysicsObjectBindings`, `PhysicsWrapperManagerBindings`,
`SceneBindings`, `SensorBindings`, `ShortestPathBindings`, and `SimBindings`.
It requires external Python 3.12 development, pybind11, and MagnumBindings;
the H19 temporary validation used official `pybind11@a2e59f0e7065404b44dfe92a28aca47ba1378dc4`
and `magnum-bindings@45811bb52e749677d5bc43d62b384ec546ed93bc` archives, neither
of which is vendored or recorded as a runtime path.

The examined installed MagnumBindings layout contains
`<include>/Magnum/PythonBindings.h`, while its installed config-component
lookup cannot discover a `Python` component at that nested location. H5a calls
`find_package(MagnumBindings CONFIG REQUIRED)` without a component, then owns
`AVEngine::MagnumBindingsPython` solely to find that public header under the
configured include root and link `Magnum::Magnum`. It neither modifies the
external prefix nor presents an AVEngine target as `MagnumBindings::Python`.

The module links `AVEngine::HabitatCore`; H4d's already exported static plugin
INTERFACE_SOURCES therefore compile the eight importer/converter registration
units in the final extension without a handwritten plugin macro. It requires
an explicit non-source `AVENGINE_HABITAT_PYTHON_OUTPUT_DIR`, resolves every
existing path component (including symlinks) before containment, and permits
only a relative path whose first component is exact `..`, emits only
`habitat_sim/_ext/habitat_sim_bindings` in staging mode. The default-off H5b
`AVENGINE_HABITAT_INSTALL_RUNTIME` mode instead requires an explicit
`AVENGINE_HABITAT_RUNTIME_PREFIX` outside both source and build trees.
Configure rejects existing or symlinked canonical `<prefix>/habitat_sim` and
`<prefix>/config` roots (with `_ext` covered by `habitat_sim`) and an existing
or symlinked complete build-intermediate package target, so a pre-seeded deep
symlink cannot become an install path. The extension remains a build-tree
intermediate and installs only the selected facade, binding, and
`config/default.physics_config.json` into that prefix. It
sets the selected native default physics path to the installed config's absolute
path so the `MetadataMediator` no longer resolves it through a caller CWD. It
does not install an RPATH, dependencies, RLR, PBR assets, or data, and does not
change AVEngine's current resolver. Fresh H24 validation against external
H19/H11/H6/H9 prefixes and explicit H9 Recast completed 148/148 Ninja steps,
ran an isolated import of `quaternion`, `corrade`, `magnum.scenegraph`, and
`habitat_sim`, and constructed visual configuration, `NavMeshSettings`, and
`ShortestPath` objects. RLR adapter and legacy audio were OFF;
`audio_enabled` and `built_with_bullet` were both false. H23 rejected an
output symlink to `native/habitat/`, and H24 rejected the source child
`native/habitat/..output`, both before any source-tree output was created. A
fresh ordinary CMake configure kept both H5 modes OFF by default. A separate
fresh H5b configure retained the H24 external package layout and EGL setting,
selected a fresh PIC RecastNavigation 1.6.0 prefix with the same
`DT_VIRTUAL_QUERYFILTER` setting, built the complete extension, and installed
the 56 selected facade `.py` files, its binding, and the default physics config.
An isolated `python -S` import from an unrelated CWD loaded facade and binding
only from that prefix and proved that `SimulatorConfiguration` and
`default_sim_settings` both select its absolute physics-config path. This is
not a full Simulator run, runtime/build cutover, or a claim that external
assets or dependencies were migrated.

| AVEngine target | Original path at `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e054b` | Treatment |
| --- | --- | --- |
| `native/habitat/python/habitat_sim/utils/settings.py` | `src_python/habitat_sim/utils/settings.py` | **adapted** default physics-config selection to read the selected native `SimulatorConfiguration` value rather than a CWD-relative literal, so an H5b prefix uses its installed config |

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
`examples/` directory. S1 itself did not move any client import; S3c later
retargets the direct host/game runner imports to AVEngine's namespaced client.
Neither change claims plugin, project-control, UE Editor Python, or runtime
cutover.

## SPEAR S2 python_ext source staging

| AVEngine target | Original path at spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 | Treatment |
| --- | --- | --- |
| native/spear/python_ext/cpp/{assert.h,client.h,func_signature_registry.h,msgpack_adaptors.h,msgpack_utils.h,spear_ext.cpp,std.h,types.h} | matching python_ext/cpp paths | **adapted** selected C++ extension source, with upstream MIT headers retained |
| native/spear/python_ext/python/spear_ext/__init__.py (removed by S3b) | python_ext/python/spear_ext/__init__.py | **adapted** upstream wrapper staged only by S2; S3b deliberately removes it rather than install a global alias |

S2 is source-only staging. It excludes upstream CMakeLists.txt, pyproject.toml,
checkout-relative rpclib paths, dependency source, build trees, and compiled
extension artifacts. It does not change Python imports, package installation,
extension build wiring, or the external SPEAR/UE runtime path.

## SPEAR S3b optional native extension build

| AVEngine target | Reference examined | Treatment |
| --- | --- | --- |
| native/spear/python_ext/CMakeLists.txt | upstream python_ext build assumptions at spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 | **reimplemented** AVEngine-owned CMake build for only the selected C++ extension source |
| native/spear/python_ext/cpp/spear_ext.cpp | python_ext/cpp/spear_ext.cpp | **adapted** module initializer renamed from spear_ext to avengine_spear_ext |
| native/spear/python_ext/cpp/msgpack_utils.h | python_ext/cpp/msgpack_utils.h | **adapted** GCC-standard C++20 portability adjustment: move the span explicit specialization to namespace scope and normalize the primary template closing brace, retaining the conversion body byte-for-byte |

S3b requires an explicit external rpclib SDK root with an installed CMake
export, an explicit nanobind CMake directory, Python 3.11, and Threads. Both
rpclib and nanobind lookups are constrained to those supplied roots; it neither
uses an ambient package nor records a checkout path. It installs only the
top-level AVEngine-local module avengine_spear_ext to an explicit external
staging prefix, and installs neither a spear package nor a spear_ext
compatibility alias.

The focused native evidence is a temporary standard-C++20 MessagePack
BIN-to-span byte/size smoke plus a fresh Make build, install, dynamic-link
inspection, and Python -S import of the staged module and AVEngine client.
Those checks make no RPC or UE call. The non-owning span's full retention
behavior remains limited to the selected client conversion path and is not a
claim of UE protocol or runtime compatibility.

S3b adds no upstream build metadata, dependency source, prebuilt extension,
wheel, RPATH, runtime resolver, runner, UE/plugin/project/content path, or
runtime cutover claim.

## SPEAR S3a namespaced host/game Python client closure

| AVEngine target | Original path at spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 | Treatment |
| --- | --- | --- |
| src/avengine/backends/spear_ue/client selected instance, unreal_object, five utils and fourteen services | selected matching python/spear modules | **adapted** 21-file host/game client closure with imports retargeted to the AVEngine namespace |
| src/avengine/backends/spear_ue/client/config/default_config*.yaml | matching python/spear/config files | **adapted** three default configuration inputs packaged with that client |

The selected upstream implementation files also receive seven mechanical trailing-whitespace removals only; no code path or behavior is changed by that formatting normalization.

S3a excludes the upstream root package initializer, editor helpers,
pipeline/tool/math utilities, initialize_world_service, examples, tests, build
metadata, C++ build wiring, rpclib and every UE/project/content path. AVEngine
supplies a new package initializer: it registers neither a global spear nor a
spear_ext alias, and uses the future AVEngine-native module name
avengine_spear_ext. Importing configuration and service definitions does not
require that optional extension; constructing client.Instance produces a clear
error until the later extension build/install layer supplies it. This is not a
native-extension build, a UE/editor compatibility claim, a runner cutover, or
a replacement for the maintained transition runtime.

## SPEAR S3c host/game runner import retargeting

S3c changes the direct host/game configuration call sites in the Apartment,
MP3D, ReplicaCAD, Kujiale, Skokloster diagnostic, and packaged probe runners
to use `avengine.backends.spear_ue.client`. The launchers that need
per-worker settings use AVEngine's `parallel_instance_settings`. Those
paths no longer inject a
SPEAR checkout `python/` or `examples/` directory merely to obtain the host
client or launch settings. The standalone Kujiale canary removes its stale
`--spear-root` CLI option because its explicit Unreal editor and project are the
actual external runtime inputs.

This is import wiring only. The optional `avengine_spear_ext` extension must
still be built and installed before an Instance can start, and no UE Editor
Python compatibility is claimed. S3d subsequently adapts the five exact
runner-facing helpers from the Eastforward fork into AVEngine; S3c/S3d still
do not claim a checkout-free
SPEAR runtime because UE, its project, and room assets remain external.

## SPEAR S3d retained runner-helper closure

S3d adapts the small helper closure used directly by the retained Apartment,
MP3D, and packaged-GLB runners.  Its source is the maintained
reachable Eastforward fork commits, not `spear-sim/spear`: the rig helper
was introduced in `0a9ba3ded8ffa07a3bc3684279845da22dc123e0`, gained the runner-facing component
selection in `c8ba04076a32060e35020deb8f706c4b13951cae`, and carries behavior through `ff6e44736f68c72ce4140152e2dadb4b58dc0b28`;
the lighting helper was introduced in `a5168b8c357afa494f6200dedb03b93c3a59be57`. The selected
bytes are carried by local MIT transition snapshot `251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7`
(SPEAR-lead-b), which is not represented as a public fork ref. The
Eastforward fork carries the retained SPEAR MIT notice at
`LICENSES/SPEAR-MIT.txt`.

| AVEngine target | Original path at the Eastforward fork snapshot | Treatment |
| --- | --- | --- |
| `src/avengine/backends/spear_ue/rig_direction.py` | `tools/spike_rlr/rig_direction_check.py` | **adapted** bounded support closure for exactly `select_skeletal_mesh_component`, `sample_body_bone_position_in_frame`, and `sample_body_basis_in_frame`; calibration, CLI, file-writing, and other rig helpers are excluded |
| `src/avengine/backends/spear_ue/lighting.py` | `examples/render_in_gpurir_room.py` | **adapted** exactly `spawn_directional_light` and `spawn_sky`; room construction, point/reflection lights, asset selection, rendering, CLI, and all other example behavior are excluded |
| Apartment, MP3D, and packaged-GLB runner call sites | their former `sys.path` imports of the two external helper files | **rewired** to AVEngine-local modules; `--spear-root` remains only where it identifies the external UE runtime/project/assets, and the ReplicaCAD route reaches the same packaged-GLB adapter without a direct change |

S3d changes no client/native-extension wiring, UE Editor behavior, project,
asset, or runtime boundary.  It removes only the helper-source directory
injection; a real UE session still requires its explicitly supplied external
runtime inputs.

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
