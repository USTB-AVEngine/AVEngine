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
| Habitat-Sim | `57ee4941dc4765240f0f91f70b2c97a919bf9038` | MIT | preserve Meta copyright, license, history, and citations |
| RLR Audio Propagation distribution | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` | CC BY-NC 4.0 | this pin provides headers, configuration and a precompiled shared library rather than propagation-engine source; keep the library external, retain attribution and terms, and do not claim source integration |
| SPEAR-owned runtime source | `684789cc71eaaf39e362d9d7a7b4b0b7f0af8568` (runtime source unchanged from `3f741db4414f6c68bd26865f197752935a01af6e`) | MIT | preserve SPEAR copyright and license; integrate only selected client/plugin/control source and record path provenance |

The root AVEngine all-rights-reserved notice does not relicense imported
third-party code. When selected source lands, its applicable license text and
path mapping must land with it. Until then,
`docs/provenance/UPSTREAM_ADAPTATIONS.md` reports the migration state rather
than claiming that the code is already integrated.

SPEAR-owned source also calls Boost (BSL-1.0), rpclib (MIT) and yaml-cpp (MIT).
Keep their notices and build them or resolve them as external dependencies;
do not migrate precompiled third-party libraries into Git. The SPEAR MIT grant
does not cover Unreal Engine, Epic content, room assets or dependencies under
a different license.

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
| SPEAR-owned code | MIT; accompanying asset claims require asset-level verification | selectively integrate required client/plugin/control source; Apartment and Kujiale use it for production visual execution, while MP3D UE remains comparison-only |
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
