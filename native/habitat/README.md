# Habitat-Sim H1 source staging

This directory holds the bounded, source-only Habitat-Sim staging needed for
AVEngine's MP3D native runtime. Existing builds and executions still use the
manifest-pinned transition fork. H1 does not add build wiring, change a
runtime path, or claim a completed native-runtime cutover.

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
| `esp/bindings/**` | `src/esp/bindings/**` | **adapted** native bindings, including the transition fork's audio binding entry point; build registration is deliberately deferred |
| `esp/physics/bullet/BulletPhysicsManager.cpp` | matching `src/esp/physics/bullet/BulletPhysicsManager.cpp` | **adapted** upstream Bullet manager with the fork's opt-in `avengine_native_gltf_skin_frame` behavior |
| `shaders/gfx/**` | `src/shaders/gfx/**` | **adapted** shader source only |
| `config/default.physics_config.json` | `data/default.physics_config.json` | **adapted** small default physics configuration |

The staged `esp/audio` adapter calls the RLR API through a separately
installed SDK. Its source is MIT, but the RLR engine, headers, configuration,
and library remain an external user-provided CC-BY-NC 4.0 dependency.
`audio/RLRAcousticContextTestAccess.h` is retained because
`RLRAcousticContext.cpp` includes it for an internal parser seam; it is not
an imported test directory or test suite.

## Explicitly deferred from H1

H1 intentionally excludes:

- upstream root and per-module CMake files/templates, packaging, and Python code;
- vendored/submodule dependencies, `.git` metadata, build trees, binaries, and caches;
- documentation, examples, test directories, datasets, generated output, and scene assets;
- all `RedwoodNoiseModel.{cpp,h,cu,cuh}` sources;
- PBR demonstration/IBL assets, BRDF tables, textures, fonts, and HDR files; and
- RLR package headers, material configuration, shared libraries, and solver source.

This directory contains PBR renderer and shader source only; no separately
licensed PBR asset is included.

## Next integration layer

A later, separately reviewed change must supply AVEngine-owned build wiring
and compatibility integration, then remove the external Habitat source-path
dependency and run the required fresh native equivalence checks.
