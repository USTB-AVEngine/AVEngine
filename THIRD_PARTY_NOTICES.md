# Third-Party Notices and Release Holds

This file records known third-party foundations, room-specific backend paths,
and asset-policy constraints for the AVEngine research project. It is not a
complete software bill of materials and not legal advice.

The original AVEngine repository is currently private and all-rights-reserved;
see `LICENSE`. Do not describe the complete project, its data, or all of its
dependencies as MIT-licensed.

## Primary runtime foundations and source transition

AVEngine is designed as an extension of, not a replacement for, the following
projects. During the current migration their selected runtime code is still
executed from the pinned Habitat fork. The target architecture selectively
integrates required distributable source into the canonical AVEngine repository
without changing the upstream terms.

| Component | Pinned revision | Terms | Required treatment |
| --- | --- | --- | --- |
| Habitat-Sim selected H1/S3 source and H4a/H4c/H4d/H5a build wiring | upstream `57ee4941dc4765240f0f91f70b2c97a919bf9038`; transition fork `e9c81c10834f7e89f33f4e0602c75535a84e054b` | MIT | H1/S3 stage selected source at `native/habitat/`; H4a reimplements a standalone `gfx_batch` slice, H4c a non-binding `AVEngine::HabitatCore` static slice, H4d its consumer-facing static importer interface, and H5a an optional Python extension selection. Preserve Meta copyright, MIT text, provenance, and citations; build/runtime cutover remains pending |
| RLR Audio Propagation distribution | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` | CC BY-NC 4.0 | the engine, headers, configuration, and precompiled library remain a legal user-installed external SDK; retain attribution and terms, and do not claim propagation-engine source integration |
| pybind11 temporary H19 validation dependency | `a2e59f0e7065404b44dfe92a28aca47ba1378dc4` | BSD-3-Clause | externally installed Python-binding support used by H5a validation only; do not vendor its source, binary, package, or license claim into the Habitat source slice without a separately reviewed distribution decision |
| MagnumBindings H19 external dependency | `45811bb52e749677d5bc43d62b384ec546ed93bc` | MIT | user-provided Corrade/Magnum Python-binding headers/runtime for H5a validation and H5b installed-prefix M1; do not vendor its source, binary, package, or installed prefix into AVEngine Git |
| SPEAR-selected S1/S2/S3a/S3b source and UE source-only closure | public `spear-sim/spear@2d4575587a9be39a40ba89c4259836a85ccd3f3f`; selected bytes carried by `251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7` | SPEAR/Intel MIT, with the limited embedded PPK_ASSERT WTFPLv2 and Microsoft MIT notices below | S1 reimplements the Apartment launch helper; S2/S3a stage selected client and extension source; S3b builds only the selected extension under the AVEngine-local avengine_spear_ext name. `native/spear/unreal/` now stages the selected UE plugin/project/configuration/build-rule source closure. It imports no UE, Content/assets, generated output or binaries; runtime/project cutover remains pending. Retain `LICENSES/SPEAR-MIT.txt`, `LICENSES/PPK_ASSERT-WTFPLv2.txt`, and `LICENSES/Microsoft-MIT.txt`. |
| PPK_ASSERT fragments in the SPEAR UE closure | source URL preserved in `native/spear/unreal/plugins/SpCore/Source/SpCore/Assert.{h,cpp}`: `https://github.com/gpakosz/PPK_ASSERT`; no separate source revision is asserted | WTFPLv2 | limited to the two retained Assert files; preserve Gregory Pakosz attribution and `LICENSES/PPK_ASSERT-WTFPLv2.txt` |
| Microsoft compiler-warning helper in the SPEAR UE closure | Microsoft copyright/ MIT notice preserved in `native/spear/unreal/plugins/SpCore/Source/SpCore/SuppressCompilerWarnings.h`; no separate source revision is asserted | MIT | limited to that retained header; preserve the Microsoft attribution and `LICENSES/Microsoft-MIT.txt` |
| Eastforward SPEAR S3d helper slice | reachable Eastforward fork commits `0a9ba3ded8ffa07a3bc3684279845da22dc123e0`, `c8ba04076a32060e35020deb8f706c4b13951cae`, `ff6e44736f68c72ce4140152e2dadb4b58dc0b28`, and `a5168b8c357afa494f6200dedb03b93c3a59be57`; selected bytes carried by local MIT transition snapshot `251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7` (SPEAR-lead-b) | MIT | source for exactly three rig-query and two lighting helpers; retain `LICENSES/SPEAR-MIT.txt`. The local carrier is not a public fork ref, and this helper slice is not attributed to official `spear-sim/spear`; UE/project/assets remain external |
| rpclib external S3b build SDK | 2.3.0 | MIT | user-installed external C++ SDK used to build the optional S3b extension; its source, headers, archive, CMake export, and any compiled result are not imported into AVEngine Git |
| nanobind external S3b build dependency | 2.7.0 | BSD-3-Clause | user-installed external Python binding support used to build the optional S3b extension; no nanobind source, package, shared library, or wheel is imported into AVEngine Git |

The root AVEngine all-rights-reserved notice does not relicense imported
third-party code. H1/S3/H4a/H4c/H4d/H5a retain the Habitat MIT text at
`LICENSES/Habitat-Sim-MIT.txt`; `native/habitat/README.md` and
`docs/provenance/UPSTREAM_ADAPTATIONS.md` record its per-module source map.
H4a/H4c build source-owned static slices; H4d only exposes installed static
importer targets to their final consumers; H5a optionally builds the selected
binding closure against external Python dependencies. Existing runtime and
selected build paths still use the manifest-pinned transition fork until a
later cutover.

H4c is an explicit 112-C++ non-binding `AVEngine::HabitatCore` slice layered
on H4a `gfx_batch` and `GfxShaderResources`. Its H10 fresh 122/122 static
compilation used temporary external H5/H6/H9 CMake dependencies; none of those
dependency sources or libraries are vendored, and that validation does not
select a runtime or build cutover. H4d adds only the eight installed
Magnum/MagnumPlugins importer/converter targets to the core's final-consumer
link interface, letting a static package supply its registration sources without
bundling plugin code or a source checkout. Its fresh H14 CPU-only 132-step
validation linked only the core target, then compiled the registrations and
decoded external GLB/PLY/PNG/HDR data; this remains static evidence rather than
a runtime or build cutover.

H5a adds a default-off Python 3.12 extension target over 17 selected adapted
binding translation units. It requires an explicit non-source output root and
writes only `habitat_sim/_ext/habitat_sim_bindings`; it does not install or copy
the staged Python facade, alter the runtime resolver, add a RLR/Bullet
capability, copy a shared library, or add RPATH. H23 rejected a symlink-to-source
output and H24 rejected the source child `..output`, both before they could
write there; H24 then compiled 148/148 Ninja steps against external H19/H11/H6/H9
prefixes and imported the
staged facade with external `quaternion`, Corrade and Magnum runtime packages;
that is isolated build/import evidence, not a runtime cutover or redistribution
of pybind11/MagnumBindings packages.

H1/S3 import no RLR header, material configuration, shared library, or solver
source. The AVEngine/Habitat adapter source is staged under `native/habitat`,
while the RLR engine remains an external user-provided CC-BY-NC 4.0 SDK. S3
adds Python/shader/template source only. H4c/H4d add no RLR/audio adapter,
Bullet, bindings, BackgroundRenderer, CUDA noise, PBR image resource, PBR
asset/default configuration, compiled extension, importer source, or runtime path.

H5a adds only opt-in binding build wiring and no RLR SDK/header/library,
Bullet source, package copy, install rule, source checkout path, or runtime
resolver/cutover. The H24 module readback had no RPATH/RUNPATH and no RLR
library dependency.

The default-off `AVEngine::HabitatRlrAudio` static adapter target resolves the
same external RLR SDK only through `AVENGINE_RLR_SDK_ROOT`. That root is the
user-installed official `RLRAudioPropagationPkg` layout, not a Git checkout;
AVEngine neither copies nor installs `libRLRAudioPropagation.so` and adds no
RLR RPATH. Users remain responsible for the platform loader configuration
(for Linux, the SDK `libs/linux/x64` directory on `LD_LIBRARY_PATH`).

The independent default-off `AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR`
option uses that same external SDK for the selected legacy `AudioSensor`
C++ wrapper. It changes no license boundary and adds no RLR source, copied
library, Python binding, runtime resolver, package installation, or runtime
cutover. The wrapper is deprecated by its own SDK header, so this opt-in is
limited to the existing legacy call path and its ABI-compatible user SDK.

The staged Habitat PBR adapter removes the compiled PBR image resource group:
enabled renderer IBL resolves only user-provided images from
`AVENGINE_HABITAT_PBR_ASSET_ROOT` or an explicit absolute user path.
AVEngine now includes the upstream MIT 718-byte
`brown_photostudio.pbr_config.json` from Habitat-Sim
`4d92aed0ba8db4d63bb945d53a67cad3ef8f7584`; this is configuration, not
image content. The BRDF LUT remains external under its MIT terms and Brown
Photostudio HDR remains external under Poly Haven CC0. Preserve the upstream
`data/pbr/license.txt` with any local external asset package.

SPEAR S1 contains only the reimplemented launch-settings behavior named above.
SPEAR S2 adds selected extension source, S3a adds the selected namespaced
host/game Python client closure, and S3b adds AVEngine-owned CMake wiring for
the optional top-level avengine_spear_ext module. S3d separately adapts the
Eastforward-fork helper slice recorded in the table above; it does not change
the client, native extension, UE Editor, project, asset, or runtime boundary.
S3b accepts only an explicit external rpclib SDK CMake export and external
Python/nanobind support; it installs no spear package or spear_ext alias and
copies no SDK or prebuilt extension into Git.

Separately, `native/spear/unreal/` contains the selected source-only UE
plugin/project/configuration/build-rule closure described in the table above.
Together, the selected SPEAR source slices contain no upstream examples,
general SPEAR CMake or pyproject wiring, dependency source, precompiled
extension, binary, asset, or build product. The optional extension build/import
and the UE source slice do not change a runner, perform RPC/UE execution, or
complete a runtime cutover.

SPEAR-owned source also calls Boost (BSL-1.0), rpclib (MIT) and yaml-cpp (MIT).
S3b uses rpclib and nanobind only as external build dependencies; its fresh
module has no RPATH/RUNPATH or dynamic librpc dependency. Keep the applicable
rpclib, bundled MessagePack/Asio, nanobind, and other transitive notices with
any redistributed built module; do not migrate their source, archive, prefix,
or precompiled library into Git. The SPEAR MIT grant does not cover Unreal
Engine, Epic content, room assets, or dependencies under a different license.

The current Habitat transition fork contains additional vendored/submodule dependencies with
their own notices, including Corrade, Magnum and plugins, RapidJSON, pybind11,
Assimp, GLFW, Recast Navigation, Bullet3, OpenEXR, zstd, tinyobjloader, and
m.css. Consult the fork's `THIRD_PARTY_NOTICES.md` and each dependency's own
license before redistribution. Its Python dependency set is not yet captured
by a complete resolved lock/SBOM.

Habitat demonstration resources can have separate terms: examples include a
MIT BRDF lookup table and ProggyClean font, CC0 Poly Haven HDR content, and CC
BY 4.0 King's Hall and Van Gogh Room assets. Code licensing does not override
asset licensing.

## Other foundations and external content

| Component or content | Known terms/status | AVEngine policy |
| --- | --- | --- |
| SPEAR-owned and adapted code | SPEAR/Intel MIT; the retained PPK_ASSERT and Microsoft fragments keep their own WTFPLv2/MIT notices; accompanying asset claims require asset-level verification | selectively integrate required client/plugin/control source; Apartment and Kujiale use it for production visual execution, while MP3D UE remains comparison-only |
| Unreal Engine, Epic content, Marketplace content | Epic EULA/proprietary terms; not covered by SPEAR MIT | external installation only; infer no right to redistribute engine binaries, editor content, examples, or assets |
| Native SPEAR `apartment_0000` content | asset-level redistribution evidence incomplete | production visual runtime input; keep scene content outside Git and hold redistribution |
| InteriorAgent/Kujiale content | non-commercial research/education terms and no-redistribution boundary; asset-level evidence remains external | production visual runtime input through the USD/MDL adapter; keep the downloaded dataset outside Git and hold redistribution |
| gpuRIR | AGPL-3.0 | optional comparison backend; never call the combined stack “all MIT” |
| ReplicaCAD | CC BY-NC 4.0 | non-commercial, attribution-required use only |
| Rocketbox tooling/content snapshot | local route records MIT | verify each distributed asset and retain source evidence |
| FLUX.2-klein-4B snapshot | local Apache-2.0 evidence; model revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` | preserve model revision/license/provenance |
| Pixal3D | MIT plus notices/dependencies; model revision `0b31f9160aa400719af409098bff7936a932f726` | preserve NOTICE and dependency terms |
| TRELLIS.2 | MIT with nested third-party exceptions; model revision `af44b45f2e35a493886929c6d786e563ec68364d` | audit renderer/export dependencies before distribution |
| SkinTokens | code `273b691d35989d71cd17ff2895fdc735097b92d1`, checkpoint revision `79736cad0fd84de384d5eede659b4ebd24effe33`; training-data rights are mixed | do not infer commercial dataset clearance from code license |
| AudioSet ontology | CC BY-SA 4.0 | ontology terms do not license the referenced YouTube audio |
| Stable Audio Open local weights | license snapshot missing | `redistribution_unknown`; hold weights and outputs |
| Hunyuan3D-2.1 | code `82920d643c0dc2f7bfd7255f45f62d386edfe60c`; Tencent custom community license with territory/use restrictions | hold code, weights, and derived release content pending review |
| RigAnything legacy route | local evidence records research/non-commercial terms | exclude from a commercial route |
| Quaternius animal assets | one farm-animal archive contains CC0 evidence; another pack lacks embedded evidence | admit item by item only after license snapshot and hash verification |

## Explicit release holds

The following must not enter a public dataset or binary bundle until their
rights and provenance are resolved:

- Hunyuan code, weights, and outputs used by the legacy route;
- AudioSet-derived WAVs or muxed videos without per-audio authorization;
- Stable Audio Open weights and outputs lacking a preserved license/model card;
- Kujiale/SPEAR apartment content without asset-level evidence;
- Assimp `test/models-nonbsd`, unaudited Bullet extras/data, and other dependency
  test assets outside the selected build artifact;
- unresolved dog textures and any untracked legacy asset groups;
- commercial use of RLR, ReplicaCAD, RigAnything, or checkpoint-derived assets
  whose terms or training sources prohibit or do not establish that use.

Derived output is not automatically free of source restrictions. Every future
AssetPackage, RoomPackage, AudioAsset, and released episode must carry stable
source IDs, hashes, license evidence, lineage, and a machine-readable admission
decision.

The model revisions above are the locally audited legacy pins recorded in
`AGENTS.md` and `manifest.yaml`; they are not a declaration that those models or
their outputs are admitted. Any pin change requires a fresh terms audit.
