# Habitat-Sim H1/S3 staging and H4a/H4c/H4d/H5a/H5b build slices

This directory holds the bounded, source-only Habitat-Sim staging needed for
AVEngine's MP3D native runtime. H1 stages the selected C++ runtime closure;
S3 adds the selected Python package under `python/habitat_sim/`, `gfx_batch`
shader sources, and four generated-header input templates. H4a adds an
AVEngine-owned standalone CMake build of `gfx_batch`; H4c adds a non-binding
core static target; H4d forwards selected static importer registrations; and
H5a optionally builds the selected Python binding closure into an explicit
external staging root. H5b can additionally install that selected facade,
binding, and small default physics configuration into an explicit external
prefix. Existing executions still use the manifest-pinned transition fork.
Neither mode changes the selected runtime path or claims a completed cutover.

## Origin and treatment

| Field | Recorded value |
| --- | --- |
| Upstream | `facebookresearch/habitat-sim@57ee4941dc4765240f0f91f70b2c97a919bf9038` |
| Transition source used for staging | `Eastforward/habitat-sim-AVEngine@e9c81c10834f7e89f33f4e0602c75535a84e054b` |
| Relationship | **adapted** selected upstream C++ source plus the transition fork's AVEngine-specific adapter changes |
| License | MIT; full text is retained at [`LICENSES/Habitat-Sim-MIT.txt`](../../LICENSES/Habitat-Sim-MIT.txt) |

H1 copies tracked files from the stated transition revision. The retained
source headers and the root MIT text preserve the Meta copyright and license
terms for every direct adapted file or module.

## Selected module map

| AVEngine target | Original path at the staging revision | Relationship and H1 scope |
| --- | --- | --- |
| `esp/{assets,core,geo,gfx,gfx_batch,io,metadata,nav,physics,scene,sensor,sim}/**` | matching `src/esp/**` paths | **adapted** Habitat runtime closure; no H1 behavior change |
| `esp/audio/**` | `src/esp/audio/**` | **adapted** AVEngine RLR context adapter plus `RLRAcousticContextTestAccess.h`, a retained MIT internal parser test seam included by runtime source; no RLR engine source is included |
| `esp/bindings/**` | `src/esp/bindings/**` | **adapted** native bindings, including the transition fork's audio binding entry point; only H5a's explicit opt-in registers its selected binding translation units |
| `esp/physics/bullet/BulletPhysicsManager.cpp` | matching `src/esp/physics/bullet/BulletPhysicsManager.cpp` | **adapted** upstream Bullet manager with the fork's opt-in `avengine_native_gltf_skin_frame` behavior |
| `shaders/gfx/**` | `src/shaders/gfx/**` | **adapted** shader source only |
| `config/default.physics_config.json` | `data/default.physics_config.json` | **adapted** small default physics configuration |

The staged `esp/audio` adapter calls the RLR API through a separately
installed SDK. Its source is MIT, but the RLR engine, headers, configuration,
and library remain an external user-provided CC-BY-NC 4.0 dependency.
`audio/RLRAcousticContextTestAccess.h` is retained because
`RLRAcousticContext.cpp` includes it for an internal parser seam; it is not
an imported test directory or test suite.

## S3 selected Python, shader, and generated-header source

S3 is a second source-only staging step from the same transition revision. It
copies the complete tracked Python source closure (56 `.py` files), the one
`gfx_batch/Shaders.conf` resource declaration and its 11 tracked GLSL/GLSL-like
shader sources, plus the four `configure.h.cmake` templates that future native
build wiring uses to generate headers.

| AVEngine target | Original path at the staging revision | S3 scope |
| --- | --- | --- |
| `python/habitat_sim/**/*.py` | `src_python/habitat_sim/**/*.py` | **adapted** complete tracked Python source closure, held under a non-importable staging root until later native installation; no copied package metadata, extension binary, or runtime-path change |
| `shaders/gfx_batch/{Shaders.conf,**/*.{frag,vert,geom,glsl}}` | matching `src/shaders/gfx_batch/**` paths | **adapted** shader resource declaration and source-only `gfx_batch` closure |
| `esp/{core,gfx,physics,sensor}/configure.h.cmake` | matching `src/esp/*/configure.h.cmake` paths | **adapted** four pure configuration-template inputs for later generated headers; no CMake build script is staged |

`python/habitat_sim/` deliberately remains below `native/habitat/`, rather
than the repository `src/` import root: existing transition runs with
`PYTHONPATH=src` must continue to resolve the installed external
`habitat_sim` package until an AVEngine-owned build installs its matching
compiled binding. The only non-Python tracked item in the upstream Python
package is `sensors/noise_models/data/redwood-depth-dist-model.npy`; it is data
and is intentionally not copied. `redwood_depth_noise_model.py` remains
unmodified source, but S3 does not configure or claim runtime availability of
that omitted noise-model data. Likewise, `python/habitat_sim/_ext/__init__.py`
is source only: no compiled `habitat_sim_bindings` extension is included.

## H4a isolated `gfx_batch` build

`CMakeLists.txt` and `cmake/GfxBatchSources.cmake` are AVEngine-owned,
reimplemented build wiring for only the selected `esp/gfx_batch` sources.
They generate the four staged `configure.h` templates in the build tree and
embed the staged `shaders/gfx_batch` resource through Corrade. The target
resolves installed Corrade/Magnum CMake packages and builds only
`avengine_habitat_gfx_batch`; it adds no dependency source, checkout path,
package installation, or runtime resolution.

This slice does not configure Habitat core, PBR resources/assets, RLR/audio,
Python/bindings, or a runtime package. CUDA is default-off here because the
selected source excludes the CUDA helper headers; that is not a final CPU-only
decision for a later native integration.

## H4c non-binding core static build

`CMakeLists.txt` and `cmake/CoreSources.cmake` define
`AVEngine::HabitatCore` as a non-binding static target with an explicit
112-translation-unit C++ closure. It depends on
`AVEngine::HabitatGfxBatch` and embeds the staged `GfxShaderResources` from
`shaders/gfx/Shaders.conf`. The selected closure covers `core`, `geo`, `gfx`,
`assets`, `metadata`, `io`, `scene`, non-Bullet `physics`, `nav`, `sensor`,
and `sim`.

The target resolves Corrade/Magnum, RapidJSON, tinyxml2, and RecastNavigation
as externally installed CMake packages. A fresh H10 CPU-only validation
against temporary external H5/H6/H9 dependency prefixes completed all 122
Ninja steps for `avengine_habitat_core`. Those temporary dependencies and
outputs are validation inputs only: no dependency source or library is
vendored, and no private temporary path is committed as build configuration.
`URDFParser.cpp` adapts the transition vendored include spelling to the
installed tinyxml2 public header `<tinyxml2.h>`.

H4c contains no `BackgroundRenderer`, RLR/audio adapter in its core archive,
Bullet source, bindings, CUDA noise source, PBR image resource, PBR asset, or
default PBR configuration. In particular, it adds `GfxShaderResources` only
and does not recreate `PbrIBlImageResources`. The existing external PBR
asset-root behavior below remains unchanged.

This is static compilation evidence, not a runtime/package/build cutover.
H4d exposes exactly eight user-installed static importer/converter targets to
the final consumer of AVEngine::HabitatCore. Their exported CMake
INTERFACE_SOURCES compile registration units in that consumer; AVEngine supplies
no handwritten import macro, importer source checkout, plugin asset, or runtime
source path. When MAGNUM_TARGET_EGL selects the installed WindowlessEglApplication,
the gfx_batch public interface propagates the same source-platform definition
to its core consumer. A fresh H14 CPU-only 132-step validation against
temporary H11/H6/H9 external prefixes linked only AVEngine::HabitatCore, compiled
all eight registration units in that final consumer, and decoded the MP3D GLB,
semantic PLY, PBR PNG, and PBR HDR inputs. Scene assets, bindings, the RLR SDK,
and end-to-end native execution remain separate work; the manifest-pinned
transition runtime is still selected.

## H5a staged binding and H5b opt-in installed prefix

`AVENGINE_HABITAT_BUILD_PYTHON_BINDINGS` is OFF by default. When enabled, the
AVEngine-owned `CMakeLists.txt` and `cmake/PythonBindingSources.cmake` build
`AVEngine::HabitatPythonBindings` as the Python 3.12 module
`habitat_sim_bindings`. The explicit list has 17 adapted staging translation
units: `Bindings`, `AudioPropagationBindings`, `AttributesBindings`,
`AttributesManagersBindings`, `ConfigBindings`, `CoreBindings`, `GeoBindings`,
`GfxBindings`, `MetadataMediatorBindings`, `GfxReplayBindings`,
`PhysicsBindings`, `PhysicsObjectBindings`, `PhysicsWrapperManagerBindings`,
`SceneBindings`, `SensorBindings`, `ShortestPathBindings`, and `SimBindings`.
The final extension links `AVEngine::HabitatCore`, so H4d's eight static
Magnum/MagnumPlugins registration sources are compiled automatically through
the existing usage interface; it writes no `CORRADE_PLUGIN_IMPORT` macro.

This build requires externally installed Python 3.12 development, pybind11,
and MagnumBindings. The examined installed MagnumBindings package exposes the
Python binding headers under `Magnum/PythonBindings.h`, but its config-component
lookup cannot locate that path as a `Python` component. H5a therefore makes no
provider modification and does not invent `MagnumBindings::Python`: its own
`AVEngine::MagnumBindingsPython` interface target finds that public header below
the configured MagnumBindings include root and links `Magnum::Magnum`.

`AVENGINE_HABITAT_PYTHON_OUTPUT_DIR` is mandatory and must resolve outside
`native/habitat/`: before directory creation, CMake resolves every existing
path component (including symlinks) and permits only a relative path whose
first component is the exact parent component `..`; a source child named
`..output` is rejected. In normal H5a staging mode it emits only the extension
under `<output>/habitat_sim/_ext/`; a caller supplies the already selected
`python/habitat_sim` facade and external Corrade/Magnum Python runtime.

`AVENGINE_HABITAT_INSTALL_RUNTIME=ON` is a separate default-off H5b mode. It
requires `AVENGINE_HABITAT_BUILD_PYTHON_BINDINGS=ON` and an explicit
`AVENGINE_HABITAT_RUNTIME_PREFIX` outside both `native/habitat/` and the CMake
build tree. Configure rejects existing or symlinked canonical
`<prefix>/habitat_sim` and `<prefix>/config` target roots (`_ext` is covered by
`habitat_sim`), plus an existing or symlinked complete build-intermediate
package target; install therefore cannot follow a pre-seeded deep symlink. The
extension remains a build-tree intermediate; `cmake --install` then installs
only:

```text
<prefix>/habitat_sim/**/*.py
<prefix>/habitat_sim/_ext/habitat_sim_bindings.<platform suffix>
<prefix>/config/default.physics_config.json
```

Do not pass a different `--prefix` to `cmake --install`: the selected native
default physics path is compiled as
`<prefix>/config/default.physics_config.json` so it does not depend on the
caller's current working directory. The selected Python
`utils/settings.py` reads that native default through `SimulatorConfiguration`
instead of retaining its historical CWD-relative `data/default.physics_config.json`
literal. This does not install Corrade, Magnum, pybind11, Python, RLR, PBR
assets, datasets, an RPATH, or an additional source checkout; those remain
external caller-provided dependencies or data.

Installed-prefix M1 entry points require
`AVENGINE_HABITAT_MAGNUM_PYTHON_SITE` before importing Habitat. It names an
external site-packages directory compatible with the selected interpreter; the
resolved site must contain `corrade/__init__.py`, `magnum/__init__.py`, and
top-level `_corrade` and `_magnum` extensions carrying one of that
interpreter's extension suffixes. The M1 adapter activates the installed prefix
first and this site second, then rejects preloaded or imported Corrade/Magnum
package or extension paths outside the site. This runtime variable names no
specific host path, is not CMake/install metadata, and does not vendor the
external site.

Fresh H24 validation used H5a staging with H19/H11/H6/H9 external prefixes,
completed 148/148 Ninja steps, imported `quaternion`, `corrade`,
`magnum.scenegraph`, and `habitat_sim`, and constructed a visual configuration,
`NavMeshSettings`, and `ShortestPath`. It kept the RLR adapter and legacy audio
sensor OFF, observed `audio_enabled=False` and `built_with_bullet=False`, and
found no checkout/RLR or H6 Recast/Detour path in generated build metadata.
H23 rejected an output symlink to `native/habitat/`, and H24 rejected the
source child `native/habitat/..output`, both before source-tree output. A fresh
ordinary CMake configure with no H5 options left both binding and install modes
OFF. A separate fresh H5b configure retained the H24 external dependency layout
and EGL setting, selected a fresh PIC RecastNavigation 1.6.0 prefix with the
same virtual-query-filter setting, built the full extension, and installed 56
facade `.py` files, the extension, and the default physics config. An isolated
`python -S` import from an unrelated CWD loaded the facade and binding from that
installed prefix using the external H19 Corrade/Magnum Python packages, kept
`built_with_bullet=False`, and verified that both `SimulatorConfiguration` and
`default_sim_settings` resolve the installed config's absolute path. Neither
mode is a full Simulator run or cutover.

## Optional external RLR SDK adapter

`AVENGINE_HABITAT_BUILD_RLR_ADAPTER` is OFF by default. When enabled, it adds
the PIC static `AVEngine::HabitatRlrAudio` target for only
`esp/audio/RLRAcousticContext.cpp`. Its sole RLR dependency is the imported
`AVEngine::RlrSdk` target; `AVENGINE_RLR_SDK_ROOT` must name the user-installed
official `RLRAudioPropagationPkg` directory containing
`headers/RLRAudioPropagation.h` and
`libs/linux/x64/libRLRAudioPropagation.so`.

The CMake module never searches a Habitat/RLR checkout or arbitrary system
paths, and it neither copies nor installs the RLR library or adds an RLR
RPATH. AVEngine M3/M4 callers provide all runtime inputs explicitly:

```bash
export AVENGINE_HABITAT_RUNTIME_PREFIX=/external/installed-habitat
export AVENGINE_HABITAT_MAGNUM_PYTHON_SITE=/external/magnum-python-site
export AVENGINE_RLR_SDK_ROOT=/external/RLRAudioPropagationPkg
```

The M3/M4 loader resolves each path, rejects a Git-checkout root or a symlink
escape, preloads exactly
`$AVENGINE_RLR_SDK_ROOT/libs/linux/x64/libRLRAudioPropagation.so` before the
installed Python binding imports, then verifies that Linux process mappings use
only that declared library. It never probes for a binding-neighbor RLR library
or falls back to `AVENGINE_HABITAT_RUNTIME_ROOT`. A missing SDK, an incompatible
installed prefix, or `RLR_ADAPTER_ENABLED=False` is a native-runtime
`blocked` condition for the M3/M4 CLI; it is not a legacy AudioSensor fallback.

With `AVENGINE_HABITAT_BUILD_RLR_ADAPTER` alone, this layer keeps
`ESP_BUILD_WITH_AUDIO` OFF and does not alter legacy `AudioSensor`. When the
separate opt-in Python binding target is also selected, only the modern
`RLRAcousticContext` API is exposed; `audio_enabled` remains false. The
separate default-off `AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR` option
regenerates that macro for the whole core and links the core publicly to the
same `AVEngine::RlrSdk` target. It preserves the existing legacy
`AudioSensor` calls through the SDK deprecated C++ wrapper; this is a
compile/link and setup boundary, not Python bindings, package installation,
runtime resolver, scene propagation, or runtime cutover. RLR material JSON and
other SDK data remain caller-provided external inputs.

Python callers must consult `habitat_sim.RLR_ADAPTER_ENABLED` before using
the modern API. When it is false, the legacy-compatible RLR attributes are
`None` and omitted from the facade and `habitat_sim.audio` star exports; this
does not treat a disabled adapter as an available audio backend.

## H1 exclusions

H1 itself intentionally excluded upstream Python code; S3's separate
source-only Python selection is documented above. The combined staging still
intentionally excludes:

- upstream root and per-module CMake build scripts, packaging, and build wiring;
- vendored/submodule dependencies, `.git` metadata, build trees, binaries, and caches;
- documentation, examples, test directories, datasets, generated output, and scene assets;
- all `RedwoodNoiseModel.{cpp,h,cu,cuh}` sources;
- PBR demonstration/IBL assets, BRDF tables, textures, fonts, and HDR files; and
- RLR package headers, material configuration, shared libraries, and solver source.

This directory contains PBR renderer and shader source only; no separately
licensed PBR asset is included.

## External PBR IBL assets

The staged renderer does not embed a Corrade PBR image resource group. When a
Habitat renderer uses `PbrShaderAttributes` with `enable_ibl=true`, set
`AVENGINE_HABITAT_PBR_ASSET_ROOT` to a user-provided asset directory. Relative
BRDF-LUT and environment-map names resolve under `bluts/` and `env_maps/`
respectively; an explicitly absolute name remains a user-managed asset path.
The existing `PbrShaderAttributes` fallback names (including
`brdflut_ldr_512x512.png` and `lythwood_room_1k.hdr`) and its lighting/map
flags are unchanged. A missing root, image, or decodable 2D image is a clear
renderer failure when IBL is enabled; rendererless and `enable_ibl=false`
paths do not require this variable. No PBR image, HDR, BRDF table, or default
PBR configuration is added to this repository, and this source-only change
does not claim a native build or runtime cutover.

## Next integration layer

H4a/H4c/H4d/H5a/H5b provide source-owned static build slices, an importer
consumer interface, an optional staged Python extension, and an opt-in
installed-prefix package. A later, separately reviewed change must select that
runtime prefix in AVEngine, complete compatibility integration, remove the
external Habitat source-path dependency, and run the required fresh native
equivalence checks.
