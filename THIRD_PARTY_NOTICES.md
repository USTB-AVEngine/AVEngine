# Third-Party Notices and Release Holds

This file records known third-party foundations, optional legacy paths, and
asset-policy constraints for the AVEngine research project. It is an M0 audit,
not a complete software bill of materials and not legal advice.

The original AVEngine repository is currently private and all-rights-reserved;
see `LICENSE`. Do not describe the complete project, its data, or all of its
dependencies as MIT-licensed.

## Primary runtime foundations

AVEngine is designed as an extension of, not a replacement for, the following
projects. Their code is kept in the separate runtime fork.

| Component | Pinned revision | Terms | Required treatment |
| --- | --- | --- | --- |
| Habitat-Sim | `57ee4941dc4765240f0f91f70b2c97a919bf9038` | MIT | preserve Meta copyright, license, history, and citations |
| RLR Audio Propagation | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` | CC BY-NC 4.0 | non-commercial use only; attribution, license link, and modification notice required |

The Habitat fork contains additional vendored/submodule dependencies with
their own notices, including Corrade, Magnum and plugins, RapidJSON, pybind11,
Assimp, GLFW, Recast Navigation, Bullet3, OpenEXR, zstd, tinyobjloader, and
m.css. Consult the fork's `THIRD_PARTY_NOTICES.md` and each dependency's own
license before redistribution. Its Python dependency set is not yet captured
by a complete resolved lock/SBOM.

Habitat demonstration resources can have separate terms: examples include a
MIT BRDF lookup table and ProggyClean font, CC0 Poly Haven HDR content, and CC
BY 4.0 King's Hall and Van Gogh Room assets. Code licensing does not override
asset licensing.

## Legacy and optional routes

| Component or content | Known terms/status | AVEngine policy |
| --- | --- | --- |
| SPEAR code | MIT; accompanying asset claims require asset-level verification | optional frozen UE comparison backend |
| Unreal Engine, Epic content, Marketplace content | Epic EULA/proprietary terms; not covered by SPEAR MIT | optional legacy backend only; infer no right to redistribute engine binaries, editor content, examples, or assets |
| Kujiale/SPEAR apartment | asset-level evidence incomplete | hold redistribution and dataset admission |
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
